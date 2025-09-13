import argparse as ap
from pathlib import Path
import gc
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torch_geometric.utils import degree
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
	def forward(self, x, supervision_edges, message_edges, message_edgewts=None):
		x = self.model.layer_norm_input(x)
		x = self.model.GNN(x, message_edges, message_edgewts)
		return self.model.NOD(x, supervision_edges)
	
class SimpleDataset(Dataset):
	def __init__(self, data):
		super().__init__()
		self.data = data
	def __len__(self):
		return self.data.shape[1]
	def __getitem__(self, idx):
		return idx, self.data[:, idx]

class BatchLoader(IterableDataset):
	def __init__(
			self,
			positive_edges,
			negative_edges,
			positive_edgewts,
			batch_size,
			coverage_fraction = 0.2,
			device = torch.device('cuda'),
			threads = 1,
			nbr_wt_intensity=1.0,
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
		self.num_edges = self.all_edges.size(2)
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
		self.message_edgewts = torch.zeros(self.num_messages*2, device = self.device)
		self.eval_edges = torch.zeros((2,2*batch_size), dtype = torch.long, device = self.device)
		self.eval_labels = torch.zeros((2*batch_size), device = self.device)
		self.eval_labels[:self.batch_size] = 1.0
		self.eval_edge_indices = torch.zeros_like(self.eval_labels, dtype = torch.long, device = self.device)
		self.eval_edgewts = torch.zeros((2*batch_size), device = self.device)

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
		for i in range(2):
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
		self.eval_edges[:,:self.batch_size] = self.all_edges[0,:,self.batch_mask[0]]
		self.eval_edges[:,self.batch_size:] = self.all_edges[1,:,self.batch_mask[1]]
		message_mask = ~self.batch_mask[0]
		message_edges = self.all_edges[0,:,message_mask]

		# self.bidirectional_message_edges.zero_()
		self.bidirectional_message_edges[:, :self.num_messages] = message_edges
		self.bidirectional_message_edges[:, self.num_messages:] = message_edges.flip(0)

		# self.eval_edgewts.zero_()
		# self.message_edgewts.zero_()

		# Subset edge attributes if available  
		self.eval_edgewts[:self.batch_size] = self.positive_edgewts[self.batch_mask[0]]
		message_edgewts = self.positive_edgewts[message_mask]
		
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
		self.final_message_mask[~violators_mask] = True  

		self.final_message_mask |= max_nbr(dst, weights, violators_mask, self.max_neighbors, nthreads=self.threads)

		final_message_edges = self.bidirectional_message_edges[:,self.final_message_mask]

		self.eval_edge_indices[:self.batch_size] = self.edge_idx[self.batch_mask[0]]
		self.eval_edge_indices[self.batch_size:] = self.edge_idx[self.batch_mask[1]] + self.num_edges

		batch = PyG_data(
			eval_edges = self.eval_edges,
			eval_labels = (self.eval_labels),
			eval_edgewts = self.eval_edgewts,
			eval_edge_indices = self.eval_edge_indices,
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
	parser.add_argument("--model","-m",
		type=str,
		help="Path to the model file (.pt)",
		required=True
	)
	parser.add_argument("--threads", "-t",
		type=int,
		help="Number of CPU threads to use",
		default=1
	)
	parser.add_argument("--outdir", "-o",
		type=str,
		help="Output directory",
		default="."
	)
	parser.add_argument("--hyperparams", "-p",
		type=str,
		help="Path to the hyperparameter file (.json)",
		default=None
	)

	args = parser.parse_args()

	gpu_yes = torch.cuda.is_available()

	if gpu_yes:
		work_device = torch.device('cuda')
	else:
		work_device = torch.device('cpu')

	with open(args.hyperparams, "r") as f:
		params = json.load(f)
	

	positive_data = torch.load(args.input,weights_only = False)
	negative_edges = generate_negative_edges(positive_data,negative_positive_ratio=1, device=work_device)
	node_embeddings = positive_data.x.to(work_device)

	# model_state_dict = torch.load(args.model, map_location=work_device, weights_only=False)

	# hidden_dims = generate_hidden_dims(params['depth'], params['last_layer_size']) + [params['last_layer_size']]

	# model = DualLayerModel(
	# 	in_channels = node_embeddings.size(1),
	# 	hidden_channels=hidden_dims,
	# 	dropout=params["dropout"],
	# 	gnn_dropout=0.5,
	# ).to(work_device)

	# model.load_state_dict(model_state_dict)
	# torch.save(model, f"{args.outdir}/PiPPINN_trained_full_model.pt")

	model = torch.load(args.model, map_location=work_device, weights_only=False)

	NOD_wrapper = NodeOnlyWrapper(model).to(work_device)
	GNN_wrapper = GNNWrapper(model).to(work_device)


	dataset = BatchLoader(
		positive_edges = positive_data.edge_index.to(work_device),
		negative_edges = negative_edges,
		positive_edgewts = positive_data.edge_attr.to(work_device),
		batch_size = 20000,
		threads= args.threads,
		nbr_wt_intensity = params["nbr_weight_intensity"],
		extra_samples=50,
		device=work_device
	)

	edge_labels = torch.zeros(2*dataset.all_edges.size(2), device='cpu')
	edge_labels[:positive_data.edge_index.size(1)] = 1.0

	edge_weights = torch.zeros(2*dataset.all_edges.size(2)).to('cpu')
	edge_weights[:positive_data.edge_index.size(1)] = positive_data.edge_attr

	del positive_data
	gc.collect()
	torch.cuda.empty_cache()

	GNN_loader = DataLoader(dataset, batch_size=None)
	NOD_loader = DataLoader(SimpleDataset(dataset.all_edges.reshape(2, -1).to('cpu')), batch_size=80000, shuffle=False)

	edge_prob_NOD = torch.zeros(2*dataset.all_edges.size(2), device='cpu')
	pred_edgewts_NOD = torch.zeros_like(edge_prob_NOD, device = 'cpu')

	edge_prob_GNN = torch.zeros_like(edge_prob_NOD, device = 'cpu')
	pred_edgewts_GNN = torch.zeros_like(edge_prob_NOD, device = 'cpu')

	for idx, batch in NOD_loader:
		batch = batch.to(work_device)
		with torch.no_grad():
			edge_predictor, edgewts = NOD_wrapper(node_embeddings, batch.T)
		edge_prob_NOD[idx] = torch.sigmoid(edge_predictor).squeeze(-1).to('cpu')
		pred_edgewts_NOD[idx] = edgewts.squeeze(-1).to('cpu')

	for batch in GNN_loader:
		batch = batch.to(work_device)
		with torch.no_grad():
			edge_predictor, predicted_edgewts = GNN_wrapper(
				x = node_embeddings,
				supervision_edges = batch.eval_edges,
				message_edges = batch.message_edges,
				message_edgewts = batch.message_edgewts
				)
			
		edge_predictions = torch.sigmoid(edge_predictor).to('cpu')
		predicted_edgewts = predicted_edgewts.to('cpu')

		# Average predictions over multiple evaluations
		batch.eval_edge_indices = batch.eval_edge_indices.to('cpu')
		edge_prob_GNN[batch.eval_edge_indices] = (edge_prob_GNN[batch.eval_edge_indices] + edge_predictions.squeeze(-1))/2
		pred_edgewts_GNN[batch.eval_edge_indices] = (pred_edgewts_GNN[batch.eval_edge_indices] + predicted_edgewts.squeeze(-1))/2

	results_NOD = evaluate_predictions(
		edge_prob = edge_prob_NOD,
		edge_label = edge_labels,
		edgewt_pred = pred_edgewts_NOD,
		edgewt_actual = edge_weights,
		threshold=0.5,
		n_bins=10
	)

	results_GNN = evaluate_predictions(
		edge_prob = edge_prob_GNN,
		edge_label = edge_labels,
		edgewt_pred = pred_edgewts_GNN,
		edgewt_actual = edge_weights,
		threshold=0.5,
		n_bins=10
	)

	# Plot calibration curves
	import matplotlib.pyplot as plt
	plt.figure(figsize=(8, 6))
	prob_true_NOD, prob_pred_NOD = results_NOD["CalibrationCurve"]
	prob_true_GNN, prob_pred_GNN = results_GNN["CalibrationCurve"]
	plt.plot(prob_pred_NOD, prob_true_NOD, marker='x', label='NOD Model')
	plt.plot(prob_pred_GNN, prob_true_GNN, marker='o', label='GNN Model')
	plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
	plt.xlabel('Mean Predicted Probability')
	plt.ylabel('Fraction of Positives')
	plt.title('Calibration Curves')
	plt.legend()
	plt.grid()
	plt.savefig(f"{args.outdir}calibration_curves.svg")
	plt.close()

	# Save results to CSV
	import csv
	with open(f"{args.outdir}/evaluation_results.csv", mode='w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["Metric", "NOD Model", "GNN Model"])
		for key in results_NOD.keys():
			if key == "CalibrationCurve":
				continue
			if key == "ConfusionMatrix":
				metrics = ["TN", "FP", "FN", "TP"]
				writer.writerow(metrics + list(results_NOD[key].flatten()) + list(results_GNN[key].flatten()))
			else:
				writer.writerow([key, results_NOD[key], results_GNN[key]])