import argparse as ap
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from torch_geometric import degree
from torch_geometric.data import Data as PyG_data
from max_nbr import max_nbr
import json
import TrainUtils as utils
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
	def __init__(self, data: PyG_data, batch_size, num_batches = None, coverage_fraction = 0.2, device = torch.device('cuda'), threads = 1, extra_samples = 30):
		"""
		Args:
			data (torch_geometric.data.Data): Graph dataset containing:
			 - positive and negative edge indices (edges)
			 - node embeddings (node_embeddings)
			 - labels that say whether the edge is positive or negative (edge_labels: bool)
			 - edge weights (edge_wts)
			batch_size (int): Number of datapoints per batch.
			coverage_fraction (float): Fraction of unsampled datapoints to sample in each batch.
		Returns:
			a batch of prediction and message edges (all edges - prediction edges),
			their attributes,
			and node embeddings
		"""
		super().__init__()
		self.device = device
		self.data = data.to(device)
		self.coverage_fraction = coverage_fraction
		self.batch_size = batch_size
		self.num_edges = data.edges.size(0)
		self.unsampled = torch.ones(self.num_edges, dtype=torch.bool, device=device)
		self.uniform_weights = torch.ones(self.num_edges, device = device)
		self.sample_size_unsampled = int(coverage_fraction*batch_size)
		self.sample_size_total = self.batch_size - self.sample_size_unsampled
		self.batch_mask = torch.zeros(self.num_edges, dtype=torch.bool, device=device)
		self.edge_idx = torch.arange(self.num_edges)
		self.sampling_fn = self.sample_with_unsampled_tracking
		self.num_messages = self.data.edge_labels.sum() - self.batch_size
		self.bidirectional_message_edges = torch.zeros((2,self.num_messages*2), dtype = torch.long, device = self.device)
		self.message_edgewts = torch.zeros(self.num_messages*2, dtype = torch.long, device = self.device)
		self.max_neighbors = 30
		self.final_message_mask = torch.zeros(self.num_message_edges*2, dtype = torch.bool, device = self.device)
		self.threads = threads
		self.num_unsampled = self.unsampled.sum()

		self.extra_samples = extra_samples 

	def sample_from_total(self):
		self.batch_mask.fill_(False)
		indices_total = torch.multinomial(self.uniform_weights, self.batch_size)
		self.batch_mask[indices_total] = True

	def sample_with_unsampled_tracking(self):
		self.batch_mask.fill_(False)
		indices_total = torch.multinomial(self.uniform_weights, self.sample_size_total)
		self.unsampled[indices_total] = False
		self.batch_mask[indices_total] = True

		unsampled_indices = self.edge_idx[self.unsampled]
		self.num_unsampled = self.unsampled.sum()
		if self.num_unsampled > self.sample_size_unsampled:
			sampled_from_unsampled = torch.multinomial(self.uniform_probs[self.unsampled_edges], self.sample_size_unsampled, replacement=False)
			self.batch_mask[unsampled_indices[sampled_from_unsampled]] = True

		else:
			self.batch_mask[unsampled_indices] = True
			num_resample_from_total = self.sample_size_unsampled - self.num_unsampled
			if num_resample_from_total > 0:
				resampled_from_total = torch.multinomial(self.uniform_probs, num_resample_from_total, replacement=False)
				self.batch_mask[resampled_from_total] = True

			self.sampling_fn = self.sample_from_total

	def __iter__(self):
		return self
	
	def __next__(self):
		self.sampling_fn()
		pred_edges = self.data.edges[self.batch_mask]
		message_mask = ~self.batch_mask & self.data.edge_labels
		message_edges = self.data.edges[message_mask]
		num_messages = message_mask.sum()
		num_bidir_messages = 2 * num_messages

		self.bidirectional_message_edges.zero_()
		self.bidirectional_message_edges[:, :num_messages] = message_edges
		self.bidirectional_message_edges[:, num_messages:num_bidir_messages] = message_edges.flip(0)

		self.pred_edgewts.zero_()
		self.message_edgewts.zero_()

		# Subset edge attributes if available  
		pred_edgewts = self.data.edgewts[self.batch_mask]
		message_edgewts = self.data.edgewts[message_mask]
		
		self.message_edgewts[:num_messages] = message_edgewts
		self.message_edgewts[num_messages:num_bidir_messages] = message_edgewts
		
		# Apply neighborhood restriction
		
		self.final_message_mask.fill_(False)

		src, dst = self.bidirectional_message_edges[:num_bidir_messages]
		
		# Compute degrees for message nodes  
		degrees = degree(src, num_nodes = src.max()+1)  
		deg_src = degrees[src]  
		deg_dst = degrees[dst]
		
		# Determine neighbor weights based on source and destination degrees
		weights = deg_src / deg_dst 
		# Augment neighbor weights with edge weights if available
		if self.edge_attr is not None:
			weights.mul_(self.message_edgewts)

		# Intensify or dampen neighbor weights
		weights.pow_(self.nbr_weight_intensity)

		# Identify violators (nodes with degree greater than max_neighbors)
		violators_mask = deg_dst > self.max_neighbors  
		# violator_dst_nodes = torch.unique(dst[violators_mask]) 
		# nonviolator_indices = violators_mask.nonzero(as_tuple=False).view(-1)
		# Mark non-violators as True in final_message_mask
		self.final_message_mask[:num_bidir_messages][violators_mask] = True  

		restricted_violators = max_nbr(dst, weights, violators_mask, self.max_neighbors, nthreads=self.threads)

		final_message_edges[:num_bidir_messages][restricted_violators] = True

		final_message_edges = self.bidirectional_message_edges[:,self.final_message_mask]

		batch = PyG_data(
			pred_edges = pred_edges,
			pred_labels = (self.batch_mask & self.data.edge_labels).float(),
			pred_edgewts = pred_edgewts,
			message_edges = final_message_edges,
			message_edgewts = self.message_edgewts[self.final_message_mask],
		)

		if self.num_unsampled == 0:
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