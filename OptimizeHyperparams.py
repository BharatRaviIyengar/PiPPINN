import argparse as ap
from pathlib import Path
import sys
from time import time
import torch
import optuna, json
import TrainUtils as utils
from glob import glob


def suggest_params(trial, search_space):
	params = {}
	for name, spec in search_space.items():
		suggest_type = spec["type"]
		if suggest_type == "float":
			params[name] = trial.suggest_float(
				name, spec["low"], spec["high"], log=spec.get("log", False)
			)
		elif suggest_type == "int":
			params[name] = trial.suggest_int(
				name, spec["low"], spec["high"], log=spec.get("log", False)
			)
		elif suggest_type == "categorical":
			params[name] = trial.suggest_categorical(name, spec["choices"])
	return params


def generate_param_sets(num_trials, outfile=None):
	study = optuna.create_study(direction="minimize")
	search_space = json.load(open("search_space.json"))
	trial_parameters = []
	for _ in range(num_trials):
		trial = study.ask()
		params = suggest_params(trial, search_space)
		trial_parameters.append(params)

	if outfile is not None:
		# Save all suggested parameters
		with open(outfile, "w") as f:
			json.dump(trial_parameters, f, indent=2)
	
	return trial_parameters

def load_param_set(infile,trial_ids=[0]):
	with open(infile, "r") as f:
		trial_parameters = json.load(f)
	return [trial_parameters[i] for i in trial_ids]

def run_training(params):
	centrality_fraction = params['centrality_fraction']
	hidden_channels = params['hidden_channels']
	dropout = params['dropout']
	learning_rate = params['learning_rate']
	weight_decay = params['weight_decay']
	patience = params['patience']

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
	
	result = {
		"params": params,
		"best_train_loss": best_train_loss,
		"best_val_loss": best_val_loss,
		"early_stopping_epoch": early_stopping_epoch
		}

	return result

	

if __name__ == "__main__":

	parser = ap.ArgumentParser(description="GraphSAGE model for edge detection")

	parser.add_argument("--input", "-i",
		type=str,
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
		help="Maximum number of training epochs",
		default=200
	)
	parser.add_argument("--batch-size", "-b",
		type=int,
		help="Minibatch size for training",
		default=0
	)
	parser.add_argument("--save_input_data",
		type=str,
		help="Save split data and negative edges to file (.pt)",
		default=None
	)
	parser.add_argument("--run-mode","-m",
		choices=["G","T","I","A"],
		help="Run mode: [G] generate trials and save hyperparameter values to json. [T] Train + validate on a chosen trial. [I] perform inference on a set of trial results. [A] perform all actions in the same execution",
		default="A"
	)
	parser.add_argument("--num_trials","-n",
		type=int,
		help="Number of trials to generate",
		default=100
	)
	parser.add_argument("--param_ranges",
		type=str,
		help="JSON file for hyperparameter ranges.",
		default=None
	)
	parser.add_argument("--trial_ids",
		type=str,
		help="Train the model with specified trial parameter combinations (comma separated list)",
		default=None
	)
	parser.add_argument("--trials_params",
		type=str,
		help="Save/load generated parameter sets to/from <filename>.",
		default=None
	)
	parser.add_argument("--trial_result","-r",
		type=str,
		help="Save/load output of trial(s)",
		default=None
	)
	parser.add_argument("--params_best", "-p",
		type=str,
		help="Output File for best hyperparameters (json)",
		default=None
	)

	SEED = 48149
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.cuda.manual_seed_all(SEED)

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

	if args.runmode == "G":
		if args.param_ranges is not None:
			if args.trials_params is None:
				args.trials_params = f"{Path(args.param_ranges).with_suffix('')}_{args.num_trials}_trials.json"
			generate_param_sets(args.num_trials, args.trials_params)
		else:
			print("Error: input file not provided")
		sys.exit(0)
	elif args.runmode =="T":
		if args.trials_params is None or args.trial_ids is None:
			print("Error input information not provided")
			sys.exit(0)
		trial_ids = [int(x.strip()) for x in args.trial_ids.split(",")]
		paramset = load_param_set(args.trials_params,trial_ids)
		results = []
		for trial_id, params in enumerate(paramset):
			result = run_training(params)
			result["trial_id"] = trial_id
			results.append(result)
		if args.trial_result is None:
			args.trial_result = f"{Path(args.trials_params).with_suffix('')}_{args.trial_ids}_results.json"
		with open(args.trial_result, "w") as f:
			json.dump(results, f, indent=2)
	elif args.runmode=="I":
		# Example results (replace with actual results from manual execution)
		trial_results = [
			{"params": trial_parameters[0], "val_loss": 0.123, "early_stopping_epoch": 50},
			{"params": trial_parameters[1], "val_loss": 0.145, "early_stopping_epoch": 60},
			# Add more results here
		]

		# Create a new Optuna study
		study = optuna.create_study(direction="minimize")

		# Add trials to the study
		for result in trial_results:
			trial = optuna.trial.create_trial(
				params=result["params"],
				distributions={
					"learning_rate": optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True),
					"weight_decay": optuna.distributions.FloatDistribution(1e-6, 1e-3, log=True),
					"dropout": optuna.distributions.FloatDistribution(0.05, 0.5),
					"hidden_channels": optuna.distributions.CategoricalDistribution(hidden_channel_values),
					"patience": optuna.distributions.IntDistribution(5, 20),
					"centrality_fraction": optuna.distributions.FloatDistribution(0.2, 0.8),
				},
				value=result["val_loss"],
				user_attrs={"early_stopping_epoch": result["early_stopping_epoch"]}
			)
			study.add_trial(trial)

		# Print the best trial
		print("Best trial:", study.best_trial)




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