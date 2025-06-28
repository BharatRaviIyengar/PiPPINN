import argparse as ap
from pathlib import Path
import sys
import torch
import json
import TrainUtils as utils
from OptimizeHyperparams import run_training

if __name__ == "__main__":

	parser = ap.ArgumentParser(description="GraphSAGE model for edge detection")

	parser.add_argument("--input_data", "-i",
		type=str, required=True,
		help="Path to the data collection to retrain (.pt).",
		metavar="<path/file>"
	)
	parser.add_argument("--parameters", "-p",
		type=str, required=True,
		help="Path to the best hyperparameters file (json)",
		metavar="<path/file>"
	)
	parser.add_argument("--output", "-o",
		type=str, default=None,
		help="Path to the trained GNN (.pt)",
		metavar="<path/file>"
	)
	parser.add_argument("--threads", "-t",
		type=int,
		help="Number of CPU threads to use",
		default=1
	)
	parser.add_argument("--epochs", "-e",
		type=int,
		help="Number of training epochs",
		default=50
	)

	args = parser.parse_args()

	gpu_yes = torch.cuda.is_available()

	if len(sys.argv) == 1:
		print("Error: essential arguments not provided.")
		parser.print_help() # Print the help message
		sys.exit(1)

	with open(args.parameters, "r") as f:
		best_params = json.load(f)

	if args.threads > 1:
		torch.set_num_threads(args.threads)
	if gpu_yes:
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	input_data = torch.load(args.input_data, weights_only=False)

	# Retrain the model with the best hyperparameters
	print("Retraining the model with the best hyperparameters...")

	# Initialize the model with the best hyperparameters

	results, best_model = run_training(best_params, input_data, device=device)
	
	torch.save(best_model.state_dict(), "args.output")

	print(f"Retraining complete. Best model saved as {args.output}.")