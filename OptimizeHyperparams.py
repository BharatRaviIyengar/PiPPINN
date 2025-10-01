import argparse as ap
from pathlib import Path
import sys
import torch
import TrainUtils as utils
import optuna, json
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.pruners import HyperbandPruner
from collections import deque

import gc

def generate_hidden_dims(input_dim, depth, last_layer_size):
	n_layers = depth - 1  # intermediate layers count (excluding first)
	
	if n_layers == 0:
		return []  # no intermediate layers, just input and last layer

	decay_factor = (last_layer_size / input_dim) ** (1 / n_layers)
	hidden_dims = [int(input_dim * decay_factor ** i) for i in range(n_layers)]

	return hidden_dims

def choose_best_trials(study:optuna.Study, topk:int=30):
	completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
	def get_stability(trial:optuna.trial.FrozenTrial):
		losses = list(trial.intermediate_values.values())
		best_index = losses.index(trial.value)
		pre_stability = ((torch.tensor(losses[best_index-5:best_index]) - trial.value)**2).mean().item() if best_index >=5 else float('-inf')
		post_stability = ((torch.tensor(losses[best_index+1:best_index+6]) - trial.value)**2).mean().item()
		return 0.25*pre_stability/trial.value + post_stability/trial.value
	def get_epoch_penalty(trial:optuna.trial.FrozenTrial):
		min_netskip = 0.05
		losses = list(trial.intermediate_values.values())
		best_index = losses.index(trial.value)
		best_epoch = best_index + 1
		n_decays = best_epoch // 3 # cooldown of 2 epochs, so decay happens every 3 epochs
		network_skip_at_best_loss = (0.6 * (trial.params['network_skip_factor'] ** n_decays))
		if network_skip_at_best_loss > min_netskip or best_epoch < 10:
			epoch_penalty = float('inf')
		else:
			epoch_penalty = best_epoch/200
		return epoch_penalty
	def composite_score(trial: optuna.trial.FrozenTrial):
		val_loss = trial.value
		stability = get_stability(trial)
		return val_loss *stability* get_epoch_penalty(trial)	
	best_trials = sorted(completed_trials, key=composite_score)[:topk]
	return best_trials

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

	utils.ME_loss.num_positive_edges = data_for_training[0]["train_sampler"].num_supervision_edges
	utils.ME_loss.margin = params['margin']
	utils.ME_loss.margin_loss_coef = params['margin_loss_coef']
	utils.ME_loss.entropy_coef = params['entropy_coef']


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

	network_skip_scheduler = utils.DecayScheduler(model, 'network_skip', initial_value=0.6, factor=network_skip_factor, cooldown=2, min_value=0.001)

	total_val_samples = sum([(data["val_sampler"].num_supervision_edges + data["val_sampler"].num_negative_edges)*data["val_sampler"].num_batches for data in data_for_training])

	preds_buf = torch.zeros(total_val_samples, dtype=torch.float32)
	labels_buf = torch.zeros(total_val_samples, dtype=torch.int8)

	# Training loop
	best_val_loss = float('inf')
	best_train_loss = float('inf')
	auc_at_best_loss = 0.0
	epochs_without_improvement = 0
	best_loss_epoch = max_epochs
	pre_best_losses = deque(maxlen=6)
	post_best_losses = deque(maxlen=5)
	network_skip_at_best_loss = 0.6
	best_auc = 0.0
	best_auc_epoch = max_epochs

	for epoch in range(max_epochs):
		total_train_loss = 0.0  # Reset total training loss for the epoch
		total_val_loss = 0.0  # Reset total validation loss for the epoch
		val_batch_count = 0
		train_batch_count = 0
		fill_idx = 0

		for data in data_for_training:
			# Training
			model.train()
			for batch in data["train_batch_loader"]:
				train_batch_count += 1
				train_loss = utils.process_data(batch, model=model, optimizer=optimizer, device=device, is_training=True, return_output=False)
				total_train_loss += train_loss # type: ignore

			# Validation
			model.eval()
			with torch.no_grad():
				for batch in data["val_batch_loader"]:
					val_batch_count += 1
					val_loss, edge_prob, edge_labels = utils.process_data(batch, model=model, optimizer=optimizer, device=device, is_training=False, return_output=True) # type: ignore
					total_val_loss += val_loss
					n = edge_prob.size(0)
					preds_buf[fill_idx:fill_idx+n] = edge_prob.cpu()
					labels_buf[fill_idx:fill_idx+n] = edge_labels.cpu()
					fill_idx += n

		# Average losses
		average_train_loss = total_train_loss / train_batch_count
		average_val_loss = total_val_loss / val_batch_count
		scheduler.step(average_val_loss)
		network_skip_scheduler.step()
		auc = utils.auc_score(labels_buf, preds_buf)
		if auc > best_auc:
			best_auc = auc
			best_auc_epoch = epoch + 1
		# Early stopping logic
		if average_val_loss < best_val_loss:
			best_val_loss = average_val_loss
			best_train_loss = average_train_loss
			epochs_without_improvement = 0
			pre_best_losses.append(average_val_loss)
			best_loss_epoch = epoch + 1
			auc_at_best_loss = auc
			network_skip_at_best_loss = model.network_skip
		else:
			epochs_without_improvement += 1
			post_best_losses.append(average_val_loss)

		pre_stability = ((torch.tensor(pre_best_losses)[:-1] - best_val_loss)**2).mean().item()/best_val_loss
		post_stability = ((torch.tensor(post_best_losses) - best_val_loss)**2).mean().item()/best_val_loss

		pre_best_losses.clear()
		post_best_losses.clear()
		
		yield {
		"epoch": epoch + 1,
		"average_train_loss": average_train_loss,
		"average_val_loss": average_val_loss,
		"best_val_loss": best_val_loss,
		"best_train_loss": best_train_loss,
		"best_loss_epoch": best_loss_epoch,
		"learning_rate": optimizer.param_groups[0]['lr'],
		"auc_at_best_loss": auc_at_best_loss,
		"network_skip_at_best_loss": network_skip_at_best_loss,
		"pre_stability": pre_stability,
		"post_stability": post_stability,
		"best_auc": best_auc,
		"best_auc_epoch": best_auc_epoch
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


	# if args.journal_file is not None:
	# 	journal_file = args.journal_file
	# else:
	# 	journal_file = f"{Path(__file__).parent.resolve()}/OptunaJournal.log"

	storage = JournalStorage(JournalFileBackend(args.journal_file))
	study = optuna.load_study(study_name="PiPPINN_HPO",storage=storage)
	best_trials = choose_best_trials(study)

	best_trials = [x for x in best_trials if x.params["depth"] > 3]

	new_study_file = f"{Path(args.journal_file).with_suffix('')}_from_BestTrials.log"

	storage = JournalStorage(JournalFileBackend(new_study_file))

	pruner = HyperbandPruner(min_resource=15, reduction_factor=3)
	sampler = TPESampler(seed=SEED, multivariate=True)

	print("Parsed arguments\n===================")
	for arg, value in vars(args).items():
		print(f"{arg}: {value}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	dataset = torch.load(args.training_data, weights_only = False)
	input_channels = dataset[0]["Val"].x.size(1)

	# def early_stop_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
	# 	if len(study.trials) > 10:
	# 		recent = [t.value for t in study.trials[-10:] if t.value is not None]
	# 		if max(recent) - min(recent) < 1e-4:
	# 			raise optuna.exceptions.OptunaError("Stopping: Converged")

	def objective(trial):
		best_val_loss = float('inf')
		best_train_loss = float('inf')
		best_loss_epoch = 0
		auc_at_best_loss = 0.0
		best_auc = 0.0
		best_auc_epoch = 0
		network_skip_at_best_loss = 0.6
		pre_stability = float('inf')
		post_stability = float('inf')
		composite_score = float('inf')
		depth = trial.suggest_categorical("depth", [4, 5])
		last_layer_size = trial.suggest_categorical("last_layer_size", [768, 1024, 1536, 2048])
		hidden_channels = generate_hidden_dims(input_channels, depth, last_layer_size) + [last_layer_size]
		params = {
			"centrality_fraction": trial.suggest_float("centrality_fraction", 0.2, 0.69, log=True),
			"dropout": trial.suggest_float("dropout", 0.1, 0.35),
			"weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
			"scheduler_factor": trial.suggest_float("scheduler_factor", 0.1, 0.5),
			"nbr_weight_intensity": trial.suggest_float("nbr_weight_intensity", 0.4, 2.5, log=True),
			"network_skip_factor": trial.suggest_float("network_skip_factor", 0.1, 0.9, log=True),
			"margin": trial.suggest_float("margin", 0.1, 1.0, log=True),
			"margin_loss_coef": trial.suggest_float("margin_loss_coef", 0.01, 0.1, log=True),
			"entropy_coef": trial.suggest_float("entropy_coef", 0.001, 0.01, log=True),
			"hidden_channels" : hidden_channels
		}
		for result in run_training(params, args.num_batches, args.batch_size, dataset, device, threads=args.threads):
			epoch = result["epoch"]
			average_val_loss = result["average_val_loss"]
			best_val_loss = result["best_val_loss"]
			best_train_loss = result["best_train_loss"]
			best_loss_epoch = result["best_loss_epoch"]
			auc_at_best_loss = result["auc_at_best_loss"]
			network_skip_at_best_loss = result["network_skip_at_best_loss"]
			pre_stability = result["pre_stability"]
			post_stability = result["post_stability"]
			composite_score = best_val_loss * (0.25*pre_stability + post_stability) * (best_loss_epoch/200 if best_loss_epoch >=10 and network_skip_at_best_loss <=0.05 else float('inf'))
			trial.report(average_val_loss, step=epoch)
			if trial.should_prune():
				raise optuna.TrialPruned()
			
		trial.set_user_attr("best_loss_epoch", best_loss_epoch)
		trial.set_user_attr("best_train_loss", best_train_loss)
		trial.set_user_attr("auc_at_best_loss", auc_at_best_loss)
		trial.set_user_attr("network_skip_at_best_loss", network_skip_at_best_loss)
		trial.set_user_attr("pre_stability", pre_stability)
		trial.set_user_attr("post_stability", post_stability)
		trial.set_user_attr("best_auc", best_auc)
		trial.set_user_attr("best_auc_epoch", best_auc_epoch)
		trial.set_user_attr("composite_score", composite_score)

		return best_val_loss
	
	study = optuna.create_study(
		study_name="PiPPINN_HPO",
		direction="minimize",
		sampler=sampler,
		storage=storage,
		load_if_exists=True,
		pruner=pruner
	)

	for trial in best_trials:
		params = trial.params
		study.enqueue_trial(params)

	study.optimize(objective, n_trials=args.num_trials)
	best_trial = study.best_trial
	if study.best_trial.number == 0:
		with open(args.params_best, "w") as f:
			json.dump({
				"params": best_trial.params,
				"value": best_trial.value,
				"user_attrs": best_trial.user_attrs
			}, f, indent=2)