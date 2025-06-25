import argparse as ap
from pathlib import Path
import sys
from time import time
import torch
import json
import TrainUtils as utils


if __name__ == "__main__":

	parser = ap.ArgumentParser(description="GraphSAGE model for edge detection")

	parser.add_argument("--input_data", "-i",
		type=str, required=True,
		help="Path to the data collection to retrain (.pt). Output from OptimizeHyperparams.py",
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
	dataset = input_data["data_for_training"]
	data_for_training = []
	for data in dataset:
		data_for_training.append(utils.generate_batch(data))


	# Retrain the model with the best hyperparameters
	print("Retraining the model with the best hyperparameters...")

	# Initialize the model with the best hyperparameters
	best_model = utils.GraphSAGE(
		in_channels=data_for_training[0]["val_graph"].x.size(1),
		hidden_channels=best_params["hidden_channels"],
		dropout=best_params["dropout"]
	).to(device)

	best_optimizer = torch.optim.Adam(
		best_model.parameters(),
		lr=best_params["learning_rate"],
		weight_decay=best_params["weight_decay"]
	)

	# Retrain the model
	best_val_loss = float('inf')
	for epoch in range(args.epochs):
		total_train_loss = 0.0
		total_val_loss = 0.0

		for data in data_for_training:
			# Training
			for batch in data["train_batch_loader"]:
				total_train_loss += utils.process_data(batch, model=best_model, optimizer=best_optimizer, device=device, is_training=True)

			# Validation
			total_val_loss += utils.process_data(data["val_graph"], model=best_model, optimizer=best_optimizer, device=device, is_training=False)

		print(f"Epoch {epoch + 1}/{args.epochs}, Training Loss: {total_train_loss:.4f}, Validation Loss: {total_val_loss:.4f}")

		if total_val_loss < best_val_loss:
			best_val_loss = total_val_loss
			epochs_without_improvement = 0
			early_stopping_epoch = epoch + 1
			epochs_without_improvement += 1
		else:
			if epochs_without_improvement >= best_params["patience"]:
				print(f"Early stopping triggered after {epoch + 1} epochs.")
				torch.save(best_model.state_dict(), args.output)  # Save the globally best model
				break
				

	print(f"Retraining complete. Best model saved as {args.output}.")