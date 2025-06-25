import argparse as ap
from pathlib import Path
import sys
from time import time
import torch
from torch_geometric.utils import degree
import optuna, json
import TrainUtils as utils
from glob import glob

if __name__ == "__main__":

	parser = ap.ArgumentParser(description="GraphSAGE model for edge detection")

	parser.add_argument("--input", "-i",
		type=str, required=True,
		help="Path to the input graph data files (.pt)",
		metavar="file-pattern, comma-separated for multiple files"
	)
	parser.add_argument("--threads", "-t",
		type=int,
		help="Number of CPU threads to use",
		default=1
	)
	parser.add_argument("--val_fraction",
		type=float,
		default= 0.2,
		help= "Fraction of data for validation (rest for training)"
	)
	parser.add_argument("--epochs", "-e",
		type=int,
		help="Number of training epochs",
		default=100
	)
	parser.add_argument("--batch-size", "-b",
		type=int,
		help="Minibatch size for training",
		default=0
	)
	parser.add_argument("--output", "-o",
		type=str,
		help="Output File for best hyperparameters (json)",
		default=None
	)
	parser.add_argument("--save_input_data",
		type=str,
		help="Save split data and negative edges to file (.pt)",
		default=None
	)

	args = parser.parse_args()

	gpu_yes = torch.cuda.is_available()

	if len(sys.argv) == 1:
		print("Error: essential arguments not provided.")
		parser.print_help() # Print the help message
		sys.exit(1)

	if args.batch_size == 0:
		args.batch_size = 32768
	
	# Torch settings
	torch.set_num_threads(args.threads)
	torch.set_num_interop_threads(args.threads)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	input_graphs_filenames = []
	# Split input patterns and load graphs
	for pattern in args.input.split(","):
		input_graphs_filenames.extend(glob(pattern.strip()))

	dataset, node_feature_dimension = utils.load_data(
		input_graphs_filenames=input_graphs_filenames, 
		val_fraction=args.val_fraction, 
		batch_size=args.batch_size, 
		save_graphs_to=args.save_input_data,
		device=device
		)
	hidden_channel_values = (node_feature_dimension * torch.tensor([0.1, 0.2, 0.4, 0.8, 1, 1.5, 2])).int().tolist()

	def objective(trial):
		# Suggest hyperparameters
		learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
		weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
		dropout = trial.suggest_uniform("dropout", 0.05, 0.5)
		hidden_channels = trial.suggest_categorical("hidden_channels", hidden_channel_values)
		patience = trial.suggest_int("patience", 5, 20)
		centrality_fraction = trial.suggest_uniform("centrality_fraction", 0.2, 0.8)

		data_for_training = [utils.generate_batch(data, centrality_fraction,device=device) for data in dataset]

		# Initialize model and optimizer
		model = utils.GraphSAGE(
			in_channels=data_for_training[0]["val_graph"].x.size(1),
			hidden_channels=hidden_channels,
			dropout = dropout
		).to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

		# Training loop
		best_val_loss = float('inf')
		best_train_loss = float('inf')
		epochs_without_improvement = 0
		early_stopping_epoch = args.epochs

		for epoch in range(args.epochs):
			total_train_loss = 0.0  # Reset total training loss for the epoch
			total_val_loss = 0.0  # Reset total validation loss for the epoch

			
			for data in data_for_training:
				# Training
				for batch in data["train_batch_loader"]:
					total_train_loss += utils.process_data(batch, model=model, optimizer=optimizer, device=device, is_training=True)

				# Validation
				total_val_loss += utils.process_data(data["val_graph"], model=model, optimizer=optimizer, device=device, is_training=False)

			# Log losses
			print(f"Epoch {epoch + 1}/{args.epochs}, Training Loss: {total_train_loss:.4f}, Validation Loss: {total_val_loss:.4f}")

			# Early stopping logic
			if total_val_loss < best_val_loss:
				best_val_loss = total_val_loss
				best_train_loss = total_train_loss
				epochs_without_improvement = 0
				early_stopping_epoch = epoch + 1
			else:
				epochs_without_improvement += 1
				if epochs_without_improvement >= patience:
					print(f"Early stopping triggered after {epoch + 1} epochs.")
					break

		# Return the best validation loss as the objective value
		trial.set_user_attr("early_stopping_epoch", early_stopping_epoch)
		trial.set_user_attr("best_val_loss", best_val_loss)  # Store the best validation loss
		trial.set_user_attr("best_train_loss", best_train_loss)  # Store the best training loss
		
		return best_val_loss

	# Begin training
	start_time = time()

	study = optuna.create_study(direction="minimize")
	study.optimize(objective, n_trials=100)

	best_trial = study.best_trial
	best_params_with_epoch = best_trial.params  # Start with the best hyperparameters
	best_params_with_epoch["early_stopping_epoch"] = best_trial.user_attrs["early_stopping_epoch"]

	# Save the best hyperparameters
	
	if args.output is None:
		args.output = "best_hyperparameters.json"

	with open(args.output, "w") as f:
		json.dump(study.best_params, f)

	elapsed_time = time() - start_time