import argparse as ap
from pathlib import Path
import sys
from time import time
import torch
import optuna, json
import TrainUtils as utils
from glob import glob

def parse_num_list(s):
	result = set()
	for part in s.split(','):
		if '-' in part:
			start, end = map(int, part.split('-'))
			result.update(range(start, end + 1))
		else:
			result.add(int(part))
	return sorted(result)


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

def optuna_dist_from_suggested_params(search_space):
	distributions = {}
	for name, spec in search_space.items():
		suggest_type = spec["type"]
		if suggest_type == "float":
			distributions[name] = optuna.distributions.FloatDistribution(
				spec["low"], spec["high"], log=spec.get("log", False)
			)
		elif suggest_type == "int":
			distributions[name] = optuna.distributions.IntDistribution(
				spec["low"], spec["high"], log=spec.get("log", False)
			)
		elif suggest_type == "categorical":
			distributions[name] = optuna.distributions.CategoricalDistribution( spec["choices"])
	return distributions

def generate_param_sets(num_trials, search_space, outfile=None):
	study = optuna.create_study(direction="minimize")
	search_space = json.load(open(search_space))
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

def run_training(params, num_batches, batch_size, dataset, device):
	centrality_fraction = params['centrality_fraction']
	hidden_channels = params['hidden_channels']
	dropout = params['dropout']
	learning_rate = params['learning_rate']
	weight_decay = params['weight_decay']
	patience = 10

	data_for_training = [utils.generate_batch(data, num_batches, batch_size, centrality_fraction,device=device) for data in dataset]

	all_sampled_edges = [torch.zeros(data["Train"].edge_index.size(1), dtype=torch.bool, device="cpu") for data in dataset]

	frac_sampled = torch.zeros(len(dataset))

	# Initialize model and optimizer
	model = utils.GraphSAGE(
		in_channels=data_for_training[0]["input_channels"],
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

		
		for idx, data in enumerate(data_for_training):
			
			# Training
			batch_count = 0
			model.train()
			for sampled_edges, batch in data["train_batch_loader"]:
				total_train_loss += utils.process_data(batch, model=model, optimizer=optimizer, device=device, is_training=True)
				all_sampled_edges[idx][sampled_edges] = True
				
				batch_count += 1
				if batch_count % 10 == 0:
					unsampled = ~all_sampled_edges[idx]
						# Promote sampling of unsampled edges #
					data["train_data_sampler"].edge_probs[unsampled] *= 1.1
					data["train_data_sampler"].uniform_probs[unsampled] *= 1.1
					data["train_data_sampler"].edge_probs[~unsampled] = data["train_data_sampler"].edge_centrality_scores[~unsampled]
					data["train_data_sampler"].uniform_probs[~unsampled].fill_(1.0)

			# Validation
			model.eval()
			with torch.inference_mode():
				for batch in data["val_batch_loader"]:
					total_val_loss += utils.process_data(batch, model=model, optimizer=optimizer, device=device, is_training=False)

		# Log losses
		print(f"Epoch {epoch + 1}/{args.epochs}, Training Loss: {total_train_loss:.4f}, Validation Loss: {total_val_loss:.4f}")
		
		for i,x in enumerate(all_sampled_edges):
			frac_sampled[i] = x.sum()/x.size(0)

		# Early stopping logic
		if total_val_loss < best_val_loss:
			best_val_loss = total_val_loss
			best_train_loss = total_train_loss
			epochs_without_improvement = 0
			early_stopping_epoch = epoch + 1
		else:
			epochs_without_improvement += 1
			if epochs_without_improvement >= patience and torch.all(frac_sampled > 0.95):
				print(f"Early stopping triggered after {epoch + 1} epochs.")
				break
	
	result = {
		"params": params,
		"best_train_loss": best_train_loss,
		"best_val_loss": best_val_loss,
		"early_stopping_epoch": early_stopping_epoch
		}

	return result, model

	

if __name__ == "__main__":

	parser = ap.ArgumentParser(description="GraphSAGE model for edge detection")

	parser.add_argument("--input", "-i",
		type=str,
		help="Path to the input graph data files (.pt)",
		metavar="file-pattern, comma-separated for multiple files",
		default=None
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
	parser.add_argument("--batch_size", "-b",
		type=int,
		help="Minibatch size for training",
		default=5000
	)
	parser.add_argument("--num_batches",
		type=int,
		help="Number of minibatches for training",
		default=300
	)
	parser.add_argument("--training_data",
		type=str,
		help="Save split data and negative edges to file (.pt)",
		default=None
	)
	parser.add_argument("--runmode","-m",
		choices=["G","T","I","A"],
		help="Run mode: [G] generate trials and save hyperparameter values to json. [T] Train + validate on a chosen trial. [I] perform inference on a set of trial results. [A] perform all actions in the same execution",
		default="A"
	)
	parser.add_argument("--num_trials","-n",
		type=int,
		help="Number of trials to generate",
		default=60
	)
	parser.add_argument("--param_ranges","-p",
		type=str,
		help="JSON file for hyperparameter ranges.",
		default=None
	)
	parser.add_argument("--trial_ids",
		type=parse_num_list,
		help="e.g. 0,2,5-7",
		default=[]
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
	parser.add_argument("--params_best",
		type=str,
		help="Output File for best hyperparameters (json)",
		default=None
	)

	SEED = 48149
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.cuda.manual_seed_all(SEED)
	sampler = optuna.samplers.TPESampler(seed=SEED)

	args = parser.parse_args()

	gpu_yes = torch.cuda.is_available()

	if len(sys.argv) == 1:
		print("Error: essential arguments not provided.")
		parser.print_help() # Print the help message
		sys.exit(1)

	if not gpu_yes:
		print("GPU not available: Quitting")
		sys.exit(0)

	print("Parsed arguments\n===================")
	for arg, value in vars(args).items():
		print(f"{arg}: {value}")

	# Torch settings
	torch.set_num_threads(args.threads)
	torch.set_num_interop_threads(args.threads)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	if args.runmode == "G":
		if args.param_ranges is not None and args.input is not None:
			if args.trials_params is None:
				args.trials_params = f"{Path(args.param_ranges).with_suffix('')}_{args.num_trials}_trials.json"
			with open(args.param_ranges,"r") as f:
				search_space = json.load(f)
			generate_param_sets(args.num_trials, args.param_ranges, args.trials_params)
		else:
			print("Error: input file(s) not provided")

		input_graphs_filenames = []
		# Split input patterns and load graphs
		for pattern in args.input.split(","):
			input_graphs_filenames.extend(glob(pattern.strip()))

		_ = utils.load_data(
			input_graphs_filenames=input_graphs_filenames, 
			val_fraction=args.val_fraction, 
			save_graphs_to=args.training_data,
			device=device
		)
		sys.exit(0)

	elif args.runmode =="T":
		if args.trials_params is None or not args.trial_ids:
			print("Error input information not provided")
			sys.exit(0)
		paramset = load_param_set(args.trials_params, args.trial_ids)
		results = []
		dataset = torch.load(args.training_data, weights_only = False)

		for trial_id, params in enumerate(paramset):
			result, _ = run_training(params, args.num_batches, args.batch_size, dataset, device)
			result["trial_id"] = args.trial_ids[trial_id]
			results.append(result)
		if args.trial_result is None:
			args.trial_result = f"{Path(args.trials_params).with_suffix('')}_{args.trial_ids}_results.json"
		with open(args.trial_result, "w") as f:
			json.dump(results, f, indent=2)

	elif args.runmode=="I":
		with open(args.trial_result, "r") as f:
			trial_results = json.load(f)

		# Create a new Optuna study
		study = optuna.create_study(direction="minimize")

		# Add trials to the study
		for result in trial_results:
			trial = optuna.trial.create_trial(
				params=result["params"],
				distributions= optuna_dist_from_suggested_params(result["params"]),
				value=result["best_val_loss"],
				user_attrs={
					"early_stopping_epoch": result["early_stopping_epoch"],
					"best_train_loss" : result["best_train_loss"]
					}
			)
			study.add_trial(trial)

		# Print the best trial
		print("Best trial:", study.best_trial)
		with open(args.params_best, "w") as f:
			json.dump(study.best_trial, f, indent=2)

	elif args.runmode=="A":
		input_graphs_filenames = []
		# Split input patterns and load graphs
		for pattern in args.input.split(","):
			input_graphs_filenames.extend(glob(pattern.strip()))
		
		dataset, node_feature_dimension = utils.load_data(
		input_graphs_filenames=input_graphs_filenames, 
		val_fraction=args.val_fraction, 
		save_graphs_to=args.training_data,
		device=device
		)
		if args.param_ranges is None:
			print("Error: param_ranges file not provided.")
			sys.exit(1)
		with open(args.param_ranges,"r") as f:
			search_space = json.load(f)

		def objective(trial):
			params = suggest_params(trial, search_space)
			result, _ = run_training(params, args.batch_size, dataset, device)
			trial.set_user_attr("early_stopping_epoch", result["early_stopping_epoch"])
			trial.set_user_attr("best_train_loss", result["best_train_loss"])

			return result["best_val_loss"]
		
		study = optuna.create_study(direction="minimize")
		study.optimize(objective, n_trials=args.num_trials)
		best_trial = study.best_trial
		with open(args.params_best, "w") as f:
			json.dump(study.best_trial, f, indent=2)