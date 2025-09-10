import argparse as ap
from pathlib import Path
import sys
import torch
import torch.nn as nn
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