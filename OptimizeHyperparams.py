import argparse as ap
from pathlib import Path
import sys
from time import time
import torch
import TrainUtils as utils
import optuna, json
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.pruners import HyperbandPruner


from glob import glob
import gc

def generate_hidden_dims(input_dim, depth, last_layer_size):
	n_layers = depth - 1  # intermediate layers count (excluding first)
	
	if n_layers == 0:
		return []  # no intermediate layers, just input and last layer

	decay_factor = (last_layer_size / input_dim) ** (1 / n_layers)
	hidden_dims = [int(input_dim * decay_factor ** i) for i in range(n_layers)]

	return hidden_dims



def run_training(params:dict, num_batches:int, batch_size:int, dataset:list, device:torch.device, max_epochs = 200, threads:int=1):
	""" Run training for a single trial with the given parameters."""

	centrality_fraction = params['centrality_fraction']
	dropout = params['dropout']
	weight_decay = params['weight_decay']
	patience = 20
	scheduler_factor = params['scheduler_factor']
	nbr_wt_intensity = params['nbr_weight_intensity']
	hidden_channels = params['hidden_channels']
	network_skip_factor = params['network_skip_factor']
	
	data_for_training = [utils.generate_batch(data, num_batches, batch_size, centrality_fraction, nbr_wt_intensity=nbr_wt_intensity, device=device, threads=threads) for data in dataset]

	del dataset
	gc.collect()
	torch.cuda.empty_cache()

	# Initialize model and optimizer
	model = utils.DualLayerModel(
			in_channels=data_for_training[0]["input_channels"],
			hidden_channels=hidden_channels,
			dropout=dropout,
			network_skip=0.5
		).to(device)
	
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer= optimizer,
		mode='min',
		factor=scheduler_factor,
		patience=10,
		cooldown=2,
		min_lr=1e-6
	)

	network_skip_scheduler = utils.DecayScheduler(model, 'network_skip', initial_value=0.6, factor=network_skip_factor, cooldown=2, min_value=0.05)

	# Training loop
	best_val_loss = float('inf')
	best_train_loss = float('inf')
	epochs_without_improvement = 0
	early_stopping_epoch = max_epochs

	for epoch in range(max_epochs):
		total_train_loss = 0.0  # Reset total training loss for the epoch
		total_val_loss = 0.0  # Reset total validation loss for the epoch
		val_batch_count = 0
		train_batch_count = 0
		
		for data in data_for_training:
			# Training
			model.train()
			for batch in data["train_batch_loader"]:
				train_batch_count += 1
				total_train_loss += utils.process_data(batch, model=model, optimizer=optimizer, device=device, is_training=True)

			# Validation
			model.eval()
			with torch.no_grad():
				for batch in data["val_batch_loader"]:
					val_batch_count += 1
					total_val_loss += utils.process_data(batch, model=model, optimizer=optimizer, device=device, is_training=False)

		# Average losses
		average_train_loss = total_train_loss / train_batch_count
		average_val_loss = total_val_loss / val_batch_count
		scheduler.step(average_val_loss)
		network_skip_scheduler.step()

		# Early stopping logic
		if average_val_loss < best_val_loss:
			best_val_loss = average_val_loss
			best_train_loss = average_train_loss
			epochs_without_improvement = 0
			early_stopping_epoch = epoch + 1
		else:
			epochs_without_improvement += 1
		
		yield {
		"epoch": epoch + 1,
		"average_train_loss": average_train_loss,
		"average_val_loss": average_val_loss,
		"best_val_loss": best_val_loss,
		"best_train_loss": best_train_loss,
		"early_stopping_epoch": early_stopping_epoch,
		"learning_rate": optimizer.param_groups[0]['lr'],
		}
		
		if epochs_without_improvement >= patience:
			print(f"Early stopping triggered after {epoch + 1} epochs.")
			break

if __name__ == "__main__":

	parser = ap.ArgumentParser(description="GraphSAGE model for edge detection")

	parser.add_argument("--threads", "-t",
		type=int,
		help="Number of CPU threads to use",
		default=1
	)
	parser.add_argument("--batch_size", "-b",
		type=int,
		help="Minibatch size for training",
		default=5000
	)
	parser.add_argument("--num_batches",
		type=int,
		help="Number of minibatches for training",
		default=None
	)
	parser.add_argument("--training_data",
		type=str,
		help="Save split data and negative edges to file (.pt)",
		default=None
	)
	parser.add_argument("--num_trials","-n",
		type=int,
		help="Number of trials to generate",
		default=40
	)
	parser.add_argument("--params_best",
		type=str,
		help="Output File for best hyperparameters (json)",
		default=None
	)
	parser.add_argument("--journal_file",
		type=str,
		help="Path to the Optuna journal file for storing study results",
		default=None
	)

	SEED = 48149
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.cuda.manual_seed_all(SEED)

	args = parser.parse_args()
	torch.num_threads = args.threads
	torch.num_interop_threads = args.threads

	gpu_yes = torch.cuda.is_available()

	if len(sys.argv) == 1:
		print("Error: essential arguments not provided.")
		parser.print_help() # Print the help message
		sys.exit(1)

	if not gpu_yes:
		print("GPU not available: Quitting")
		sys.exit(0)

	if args.journal_file is not None:
			journal_file = args.journal_file
	else:
			journal_file = f"{Path(__file__).parent.resolve()}/OptunaJournal.log"

	storage = JournalStorage(JournalFileBackend(journal_file))
	pruner = HyperbandPruner(min_resource=15, reduction_factor=3)
	sampler = TPESampler(seed=SEED, multivariate=True)

	print("Parsed arguments\n===================")
	for arg, value in vars(args).items():
		print(f"{arg}: {value}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	dataset = torch.load(args.training_data, weights_only = False)
	input_channels = dataset[0]["Val"].x.size(1)

	def early_stop_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
		if len(study.trials) > 10:
			recent = [t.value for t in study.trials[-10:] if t.value is not None]
			if max(recent) - min(recent) < 1e-4:
				raise optuna.exceptions.OptunaError("Stopping: Converged")

	def objective(trial):
		best_val_loss = float('inf')
		best_train_loss = float('inf')
		early_stopping_epoch = 0
		depth = trial.suggest_int("depth", 3, 5)
		last_layer_size = trial.suggest_categorical("last_layer_size", [768, 1024, 1536, 2048])
		hidden_channels = generate_hidden_dims(input_channels, depth, last_layer_size) + [last_layer_size]
		params = {
			"centrality_fraction": trial.suggest_float("centrality_fraction", 0.2, 0.69, log=True),
			"dropout": trial.suggest_float("dropout", 0.1, 0.35),
			"weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
			"scheduler_factor": trial.suggest_float("scheduler_factor", 0.1, 0.5),
			"nbr_weight_intensity": trial.suggest_float("nbr_weight_intensity", 0.4, 2.5, log=True),
			"network_skip_factor": trial.suggest_float("network_skip_factor", 0.1, 0.9, log=True),
			"hidden_channels" : hidden_channels
		}
		for result in run_training(params, args.num_batches, args.batch_size, dataset, device, threads=args.threads):
			epoch = result["epoch"]
			average_val_loss = result["average_val_loss"]
			best_val_loss = result["best_val_loss"]
			best_train_loss = result["best_train_loss"]
			early_stopping_epoch = result["early_stopping_epoch"]
			for key, value in result.items():
				print(f"Epoch {epoch}: {key} = {value}")
			trial.report(average_val_loss, step=epoch)
			if trial.should_prune():
				raise optuna.TrialPruned()
			
		trial.set_user_attr("early_stopping_epoch", early_stopping_epoch)
		trial.set_user_attr("best_train_loss", best_train_loss)
		return best_val_loss
	
	study = optuna.create_study(
		study_name="PiPPINN_HPO",
		direction="minimize",
		sampler=sampler,
		storage=storage,
		load_if_exists=True,
		pruner=pruner
	)

	study.optimize(objective, n_trials=args.num_trials, callbacks=[early_stop_callback])
	best_trial = study.best_trial
	if study.best_trial.number == 0:
		with open(args.params_best, "w") as f:
			json.dump({
				"params": best_trial.params,
				"value": best_trial.value,
				"user_attrs": best_trial.user_attrs
			}, f, indent=2)