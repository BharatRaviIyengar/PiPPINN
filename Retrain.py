import argparse as ap
from pathlib import Path
import sys
import torch
import json
import TrainUtils as utils
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import gc
from collections import deque

def generate_hidden_dims(depth, last_layer_size):
	n_layers = depth - 1  # intermediate layers count (excluding first)
	start_size = 2048

	if n_layers == 0:
		return []  # no intermediate layers, just input and last layer

	decay_factor = (last_layer_size / start_size) ** (1 / n_layers)
	sizes = []
	for i in range(1, n_layers):
		size = int(start_size * (decay_factor ** i))
		sizes.append(size)
	return sizes

def choose_best_trials(study:optuna.Study, topk:int=5):
	completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]

	def get_trial_scores(trial:optuna.trial.FrozenTrial):
		best_loss = trial.value
		best_loss_epoch = trial.user_attrs["best_loss_epoch"] - 1
		network_skip_at_best_loss = trial.user_attrs["network_skip_at_best_loss"]
		auc_at_best_loss = trial.user_attrs["auc_at_best_loss"]
		best_auc = trial.user_attrs["best_auc"]
		best_auc_epoch = trial.user_attrs["best_auc_epoch"] - 1 
		min_desired_netskip = 0.007
		min_possible_netskip = 0.001
		max_epochs = 200

		losses = list(trial.intermediate_values.values())

		if best_loss_epoch <= 5 or network_skip_at_best_loss < min_desired_netskip:
			return float('inf')

		# Stability calculation
		pre_stability = ((torch.tensor(losses[best_loss_epoch-5:best_loss_epoch])/best_loss - 1)**2).mean().item()
		post_stability = ((torch.tensor(losses[best_loss_epoch+1:best_loss_epoch+6])/best_loss - 1)**2).mean().item()

		stability = 0.25*pre_stability + 0.75*post_stability

		# AUC-related scores
		auc_gap = abs(best_auc - auc_at_best_loss)
		auc_penalty_loss = ((1 - auc_at_best_loss) * 10) ** 2
		auc_penalty_best = ((1 - best_auc) * 10) ** 2

		# Epoch & timing normalization ---
		epoch_penalty = min(best_loss_epoch / max_epochs, 1.0)
		if network_skip_at_best_loss > min_desired_netskip:
				epoch_penalty *= network_skip_at_best_loss/min_desired_netskip  
		
		auc_gap_correction = 1 + auc_gap / (best_auc + 1e-6)
		auc_timing_penalty = min(abs(best_auc_epoch - best_loss_epoch) / max_epochs, 1.0)
		auc_timing_penalty *= auc_gap_correction  # smaller if AUCs are close

		composite_score = (
				1.0 * best_loss +
				1.0 * auc_penalty_loss +
				0.3 * stability +
				0.2 * epoch_penalty +
				0.3 * auc_timing_penalty + 
				0.5 * auc_penalty_best
			)
		return composite_score

	best_trials = sorted(completed_trials, key=get_trial_scores)[:topk]
	return best_trials

def run_training(params:dict, num_batches, batch_size:int, dataset:list, model_outfile, device:torch.device, max_epochs = 200, threads:int=1, runs=1):
	""" Run training for a single trial with the given parameters."""

	centrality_fraction = params['centrality_fraction']
	dropout = params['dropout']
	weight_decay = params['weight_decay']
	patience = 20
	scheduler_factor = params['scheduler_factor']
	nbr_wt_intensity = params['nbr_weight_intensity']
	network_skip_factor = params['network_skip_factor']
	min_epochs = 10

	hidden_channels = generate_hidden_dims(params['depth'], params['last_layer_size']) + [params['last_layer_size']]

	
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

	min_network_skip = 0.001
	network_skip_scheduler = utils.DecayScheduler(model, 'network_skip', initial_value=0.6, factor=network_skip_factor, cooldown=2, min_value=min_network_skip)

	total_val_samples = sum([(data["val_sampler"].num_supervision_edges + data["val_sampler"].num_negative_edges)*data["val_sampler"].num_batches for data in data_for_training])

	preds_buf = torch.zeros(total_val_samples, dtype=torch.float32)
	labels_buf = torch.zeros(total_val_samples, dtype=torch.int8)

	# Training loop
	pre_best_losses = deque(maxlen=6)
	post_best_losses = deque(maxlen=5)

	training_stats = []

	for run in range(runs):
		best_val_loss = float('inf')
		best_train_loss = float('inf')
		auc_at_best_loss = float('-inf')
		epochs_without_improvement = 0
		best_loss_epoch = max_epochs
		network_skip_at_best_loss = 0.6
		best_auc = float('-inf')
		best_auc_epoch = max_epochs
		pre_stability = float('inf')
		post_stability = float('inf')

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
				best_loss_epoch = epoch + 1
				auc_at_best_loss = auc
				network_skip_at_best_loss = model.network_skip
				pre_best_losses.extend(post_best_losses)
				pre_best_losses.append(average_val_loss)
				post_best_losses.clear()
			else:
				epochs_without_improvement += 1
				post_best_losses.append(average_val_loss)
			

			if epochs_without_improvement >= patience:
				print(f"Early stopping triggered after {epoch + 1} epochs.")
				break

		pre_stability = ((torch.tensor(pre_best_losses)[:-1] - best_val_loss)**2).mean().item()
		post_stability = ((torch.tensor(post_best_losses) - best_val_loss)**2).mean().item()

		training_stats.append({
			"best_val_loss": best_val_loss,
			"best_train_loss": best_train_loss,
			"best_loss_epoch": best_loss_epoch,
			"pre_stability": pre_stability,
			"post_stability": post_stability,
			"auc_at_best_loss": auc_at_best_loss,
			"network_skip_at_best_loss": network_skip_at_best_loss,
			"best_auc": best_auc,
			"best_auc_epoch": best_auc_epoch,
		})

	return training_stats

if __name__ == "__main__":

	parser = ap.ArgumentParser(description="GraphSAGE model for edge detection")

	parser.add_argument("--input_data", "-i",
		type=str, required=True,
		help="Path to the data collection to retrain (.pt).",
		metavar="<path/file>"
	)
	parser.add_argument("--trials",
		type=str, required=True,
		help="Path to the trials file",
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

	args = parser.parse_args()

	gpu_yes = torch.cuda.is_available()

	if len(sys.argv) == 1:
		print("Error: essential arguments not provided.")
		parser.print_help() # Print the help message
		sys.exit(1)

	storage = JournalStorage(JournalFileBackend(args.trials))

	study = optuna.load_study(
		study_name="PiPPINN_HPO",
		storage=storage
	)

	if args.threads > 1:
		torch.set_num_threads(args.threads)
	if gpu_yes:
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	input_data = torch.load(args.input_data, weights_only=False)

	best_params = study.best_trial.params

	best_params_file = Path(args.output).parent.joinpath("Best_Params.json")

	with open(best_params_file,"w") as f:
		json.dump(best_params, f, indent=2)

	# Retrain the model with the best hyperparameters
	print("Retraining the model with the best hyperparameters...")

	# Initialize the model with the best hyperparameters

	out = run_training(params=best_params, dataset=input_data, batch_size=40000, num_batches=None, device=device, model_outfile= args.output, threads=args.threads)

	model_stats = f"{Path(args.output).with_suffix('')}_outstats.json"

	with open(model_stats,"w") as f:
		json.dump(out, f, indent=2)
	
	print(f"Retraining complete. Best model saved as {args.output} and its loss stats in {model_stats}")