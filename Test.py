import argparse as ap
from pathlib import Path
import gc
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from torch_geometric import degree
from torch_geometric.data import Data as PyG_data
from max_nbr import max_nbr
import json
from TrainUtils import generate_negative_edges
import numpy as np
from sklearn.metrics import (
	roc_auc_score, average_precision_score,
	confusion_matrix, log_loss, brier_score_loss,
	mean_squared_error, mean_absolute_error
)
from sklearn.calibration import calibration_curve

class NodeOnlyWrapper(nn.Module):
	def __init__(self, model):
		super().__init__()
		self.model = model
	def forward(self, x, supervision_edges):
		x = self.model.layer_norm_input(x)
		return self.model.NOD(x, supervision_edges)

class GNNWrapper(nn.Module):
	def __init__(self, model):
		super().__init__()
		self.model = model
	def forward(self, x, supervision_edges, message_edges, message_edgewt=None):
		x = self.model.layer_norm_input(x)
		x = self.model.GNN(x, message_edges, message_edgewt)
		return self.model.NOD(x, supervision_edges)

class BatchLoader(IterableDataset):
	def __init__(
			self,
			positive_edges,
			negative_edges,
			positive_edgewts,
			batch_size,
			num_batches = None,
			coverage_fraction = 0.2,
			device = torch.device('cuda'),
			threads = 1,
			nbr_wt_intensity=1,
			extra_samples = 30
			):
		super().__init__()
		assert positive_edges.size() == negative_edges.size(), "This batch loader needs equal number of postive and negative edges"
		self.device = device
		self.coverage_fraction = coverage_fraction
		self.batch_size = batch_size
		self.nbr_weight_intensity = nbr_wt_intensity
		self.threads = threads
		self.max_neighbors = 30
		self.extra_samples = extra_samples 

		self.all_edges = torch.stack((positive_edges,negative_edges)).to(self.device)
		self.num_edges = self.all_edges.size(1)
		self.edge_idx = torch.arange(self.num_edges, device = self.device)
		self.positive_edgewts = positive_edgewts.to(self.device)
		self.unsampled = torch.ones((2,self.num_edges), dtype=torch.bool, device=device)
		self.num_unsampled = self.unsampled.sum(dim=1)

		self.uniform_weights = torch.ones((2,self.num_edges), device = device)
		self.sample_size_unsampled = int(coverage_fraction * batch_size)
		self.sample_size_total = self.batch_size - self.sample_size_unsampled
		self.batch_mask = torch.zeros((2,self.num_edges), dtype=torch.bool, device=device)
		
		self.num_messages = self.num_edges - self.batch_size
		self.bidirectional_message_edges = torch.zeros((2,self.num_messages*2), dtype = torch.long, device = self.device)
		self.message_edgewts = torch.zeros(self.num_messages*2, dtype = torch.long, device = self.device)
		self.pred_labels = torch.zeros((2*batch_size), device = self.device)
		self.pred_edgewts = torch.zeros((2*batch_size), device = self.device)

		self.final_message_mask = torch.zeros(self.num_messages*2, dtype = torch.bool, device = self.device)
		
		self.sampling_fn = self.sample_with_unsampled_tracking

	def sample_from_total(self):
		self.batch_mask.fill_(False)
		indices_total = torch.multinomial(self.uniform_weights, self.batch_size, replacement = False)
		self.batch_mask.scatter_(1, indices_total, True)

	def sample_with_unsampled_tracking(self):
		self.batch_mask.fill_(False)
		indices_total = torch.multinomial(self.uniform_weights, self.sample_size_total, replacement = False)
		self.unsampled.scatter_(1, indices_total, False)
		self.batch_mask.scatter_(1, indices_total, True)

		self.num_unsampled = self.unsampled.sum(dim=1)
		for i in range(1):
			unsampled_indices = self.edge_idx[self.unsampled[i]]
			if self.num_unsampled[i] > self.sample_size_unsampled:
				sampled_from_unsampled = torch.multinomial(self.uniform_weights[0,:self.num_unsampled[i]], self.sample_size_unsampled, replacement=False)
				self.batch_mask[i,unsampled_indices[sampled_from_unsampled]] = True

			else:
				self.batch_mask[i,unsampled_indices] = True
				num_resample_from_total = self.sample_size_unsampled - self.num_unsampled[i]
				if num_resample_from_total > 0:
					available = ~self.batch_mask[i]
					resampled_from_total = torch.multinomial(self.uniform_weights[0,available], num_resample_from_total, replacement=False)
					global_available_idx = self.edge_idx[available]
					resampled_from_total = global_available_idx[resampled_from_total]
					self.batch_mask[i,resampled_from_total] = True
					
		if self.num_unsampled.sum() == 0:
			self.sampling_fn = self.sample_from_total

	def __iter__(self):
		return self
	
	def __next__(self):
		self.sampling_fn()
		pred_edges = self.all_edges[self.batch_mask]
		message_mask = ~self.batch_mask[0]
		message_edges = self.all_edges[0,message_mask]

		self.bidirectional_message_edges.zero_()
		self.bidirectional_message_edges[:, :self.num_messages] = message_edges
		self.bidirectional_message_edges[:, self.num_messages:] = message_edges.flip(0)

		self.pred_edgewts.zero_()
		# self.message_edgewts.zero_()

		# Subset edge attributes if available  
		self.pred_edgewts[:self.batch_size] = self.positive_edgewts[self.batch_mask[0]]
		message_edgewts = self.all_edgewts[0,message_mask]
		
		self.message_edgewts[:self.num_messages] = message_edgewts
		self.message_edgewts[self.num_messages:] = message_edgewts
		
		# Apply neighborhood restriction
		
		self.final_message_mask.fill_(False)

		src, dst = self.bidirectional_message_edges
		
		# Compute degrees for message nodes  
		degrees = degree(src, num_nodes = src.max()+1)  
		deg_src = degrees[src]  
		deg_dst = degrees[dst]
		
		# Determine neighbor weights based on source and destination degrees
		weights = deg_src / deg_dst 
		# Augment neighbor weights with edge weights if available
		weights.mul_(self.message_edgewts)

		# Intensify or dampen neighbor weights
		weights.pow_(self.nbr_weight_intensity)

		# Identify violators (nodes with degree greater than max_neighbors)
		violators_mask = deg_dst > self.max_neighbors  
		# violator_dst_nodes = torch.unique(dst[violators_mask]) 
		# nonviolator_indices = violators_mask.nonzero(as_tuple=False).view(-1)
		# Mark non-violators as True in final_message_mask
		self.final_message_mask[violators_mask] = True  

		self.final_message_mask |= max_nbr(dst, weights, violators_mask, self.max_neighbors, nthreads=self.threads)

		final_message_edges = self.bidirectional_message_edges[:,self.final_message_mask]

		self.pred_labels.zero_()
		self.pred_labels[:self.batch_size] = 1.0

		batch = PyG_data(
			pred_edges = pred_edges,
			pred_labels = (self.batch_mask & self.data.edge_labels).float(),
			pred_edgewts = self.pred_edgewts,
			message_edges = final_message_edges,
			message_edgewts = self.message_edgewts[self.final_message_mask],
		)

		if self.num_unsampled.sum() == 0:
			if self.extra_samples == 0:
				raise StopIteration
			self.extra_samples -= 1

		return batch

		


	
def evaluate_predictions(edge_prob, edge_label, edgewt_pred, edgewt_actual, threshold=0.5, n_bins=10):
	"""
	Unified evaluation for probabilistic edge prediction.

	Args:
		y_true (array): Ground truth labels (binary {0,1} or continuous values).
		y_prob (array): Predicted probabilities or continuous outputs.
		task (str): "binary" (classification) or "continuous" (regression).
		threshold (float): Threshold for confusion matrix in binary task.
		n_bins (int): Number of bins for calibration curve.

	Returns:
		results (dict): Dictionary of evaluation metrics.
	"""
	edge_label = edge_label.cpu().numpy()
	edge_prob = edge_prob.cpu().numpy()
	edgewt_actual = edge_label.cpu().numpy()
	edgewt_pred = edge_prob.cpu().numpy()

	results = {}


	# Confusion matrix at fixed threshold
	y_pred = (edge_prob >= threshold).astype(int)
	results["ConfusionMatrix"] = confusion_matrix(edge_label, y_pred)

	# Probabilistic metrics
	results["LogLoss"] = log_loss(edge_label, edge_prob, eps=1e-15)
	results["BrierScore"] = brier_score_loss(edge_label, edge_prob)

	# Ranking metrics
	results["ROC-AUC"] = roc_auc_score(edge_label, edge_prob)
	results["PR-AUC"] = average_precision_score(edge_label, edge_prob)

	# Calibration curve (fraction of positives vs predicted prob)
	prob_true, prob_pred = calibration_curve(edge_label, edge_prob, n_bins=n_bins, strategy='uniform')
	results["CalibrationCurve"] = (prob_true, prob_pred)

	# Continuous regression-style metrics for edge weigh predictions
	results["MSE"] = mean_squared_error(edgewt_actual, edgewt_pred)
	results["MAE"] = mean_absolute_error(edgewt_actual, edgewt_pred)

	# R^2 variance explained
	ss_res = np.sum((edgewt_actual - edgewt_pred) ** 2)
	ss_tot = np.sum((edgewt_actual - np.mean(edgewt_pred)) ** 2)
	results["R2"] = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

	return results


if __name__ == "__main__":

	parser = ap.ArgumentParser(description="GraphSAGE model for edge detection")

	parser.add_argument("--input", "-i",
		type=str,
		help="Path to the test positive graph (.pt).",
		metavar="<path/file>"
	)
	parser.add_argument("--processed", "-p",
		type=str,
		help="Path to the processed input data (.pt)"
	)
	parser.add_argument("--model","-m",
		type=str,
		help="Path to the model file (.pt)",
		required=True
	)
	parser.add_argument("--batch_size","-b",
		type=int,
		help="Batch size for model inference",
		default=262144
	)
	parser.add_argument("--threads", "-t",
		type=int,
		help="Number of CPU threads to use",
		default=1
	)

	args = parser.parse_args()

	gpu_yes = torch.cuda.is_available()

	if gpu_yes:
		work_device = torch.device('cuda')
	else:
		work_device = torch.device('cpu')

	positive_data = torch.load(args.input,weights_only = False).to(work_device)
	negative_edges = generate_negative_edges(positive_data,negative_positive_ratio=1)
	node_embeddings = positive_data.x

	dataset = BatchLoader(
		positive_edges = positive_data.edge_index,
		negative_edges = negative_edges,
		positive_edgewts = positive_data.edge_attr,
		batch_size= 20000,
		threads=4,
		nbr_wt_intensity=nbr_wt_intensity,
		extra_samples=50,
		device=work_device
	)

	del positive_data
	gc.collect()
	torch.cuda.empty_cache()

	loader = DataLoader(dataset, batch_size=None)