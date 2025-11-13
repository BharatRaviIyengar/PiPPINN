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
import pickle
from copy import deepcopy

def generate_hidden_dims(input_dim, depth, last_layer_size):
	n_layers = depth - 1  # intermediate layers count (excluding first)
	
	if n_layers == 0:
		return []  # no intermediate layers, just input and last layer

	decay_factor = (last_layer_size / input_dim) ** (1 / n_layers)
	hidden_dims = [int(input_dim * decay_factor ** i) for i in range(n_layers)]

	return hidden_dims

def get_trial_stability(trial:optuna.trial.FrozenTrial, radius:int=5):
	best_score = trial.value
	best_score_epoch = trial.user_attrs["best_score_epoch"] - 1
	max_epochs = 200
	scores = list(trial.intermediate_values.values())
	startindex = max(0, best_score_epoch - radius)
	endindex = min(max_epochs, best_score_epoch + radius + 1)
	stability = (torch.tensor(scores[startindex:endindex]) - best_score)**2
	return stability.mean().item()
	
	
def choose_best_trials(study:optuna.Study, λ = 10, topk:int=5):
	scored_trials = [(trial, trial.value + λ*get_trial_stability(trial)) for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]

	best_trials = [(t, scores) for t, scores in sorted(scored_trials, key=lambda x: x[1])[:topk]]
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

	input_channels = dataset[0]["Val"].x.size(1)

	hidden_channels = generate_hidden_dims(input_channels, params['depth'], params['last_layer_size']) + [params['last_layer_size']]

	data_for_training = [utils.generate_batch(data, num_batches, batch_size, centrality_fraction, nbr_wt_intensity=nbr_wt_intensity, device=device, threads=threads) for data in dataset]

	utils.ME_loss.num_positive_edges = data_for_training[0]["train_sampler"].num_supervision_edges
	utils.ME_loss.margin = params['margin']
	utils.ME_loss.margin_loss_coef = params['margin_loss_coef']
	utils.ME_loss.entropy_coef = params['entropy_coef']
	min_network_skip = 0.001

	total_val_samples = sum([(data["val_sampler"].num_supervision_edges + data["val_sampler"].num_negative_edges)*data["val_sampler"].num_batches for data in data_for_training])

	preds_buf = torch.zeros(total_val_samples, dtype=torch.float32)
	labels_buf = torch.zeros(total_val_samples, dtype=torch.int8)

	pre_best_losses = deque(maxlen=6)
	post_best_losses = deque(maxlen=5)
	min_desired_netskip = 0.007
	best_model_score = float('inf')
	training_stats = []


	del dataset
	gc.collect()
	torch.cuda.empty_cache()

	for run in range(runs):
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
		network_skip_scheduler = utils.DecayScheduler(model, 'network_skip', initial_value=0.6, factor=network_skip_factor, cooldown=2, min_value=min_network_skip)
		
		best_val_score = float('inf')
		best_val_loss = float('inf')
		best_auc_penalty = float('inf')
		best_train_loss = float('inf')
		auc_at_best_loss = float('-inf')
		epochs_without_improvement = 0
		best_loss_epoch = max_epochs
		network_skip_at_best_loss = 0.6
		best_auc = float('-inf')
		best_auc_epoch = max_epochs
		pre_stability = float('inf')
		post_stability = float('inf')
		best_model = None

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
			post_best_losses.append(average_val_loss)
			
			auc = utils.auc_score(preds_buf, labels_buf)
			auc_penalty = ((1 - auc) * 10) ** 2

			combined_score = average_val_loss + auc_penalty

			print(f"Run {run+1}, Epoch {epoch+1:03d}: Train Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}, AUC: {auc:.4f}, Combined Score: {combined_score:.4f}, Network Skip: {model.network_skip:.4f}")
						
			if auc > best_auc:
				best_auc = auc
				best_auc_epoch = epoch + 1
				best_auc_penalty = auc_penalty


			# Early stopping logic
			if combined_score < best_val_score:
				best_val_loss = average_val_loss
				best_val_score = combined_score
				best_train_loss = average_train_loss
				epochs_without_improvement = 0
				best_loss_epoch = epoch + 1
				auc_at_best_loss = auc
				network_skip_at_best_loss = model.network_skip
				pre_best_losses.extend(post_best_losses)
				pre_best_losses.append(average_val_loss)
				post_best_losses.clear()

				## Save best model based on composite score
				if best_loss_epoch > 5 and network_skip_at_best_loss <= min_desired_netskip:
					with torch.no_grad():
						best_model = deepcopy(model).to('cpu')

			else:
				epochs_without_improvement += 1
				
			scheduler.step(average_val_loss)
			network_skip_scheduler.step()

			if epochs_without_improvement >= patience:
				print(f"Early stopping triggered after {epoch + 1} epochs.")
				break

		# AUC related penalties
		auc_gap = abs(best_auc - auc_at_best_loss)
		auc_gap_correction = 1 + auc_gap / (best_auc + 1e-6)
		auc_timing_penalty = min(abs(best_auc_epoch - best_loss_epoch) / max_epochs, 1.0)
		auc_timing_penalty *= auc_gap_correction

		# Epoch penalty
		epoch_penalty = min(best_loss_epoch / max_epochs, 1.0)
		if network_skip_at_best_loss > min_desired_netskip:
			epoch_penalty *= network_skip_at_best_loss/min_desired_netskip

		# Final stability calculation
		pre_stability = ((torch.tensor(pre_best_losses)[:-1] - best_val_loss)**2).mean().item() if len(pre_best_losses) >= 5 else float('inf')
		post_stability = ((torch.tensor(post_best_losses) - best_val_loss)**2).mean().item()

		stability = 0.25*pre_stability + 0.75*post_stability

		overall_composite_score = best_val_score + 0.3*stability + 0.2*epoch_penalty + 0.3*auc_timing_penalty + 0.5*best_auc_penalty

		if best_model is not None and overall_composite_score < best_model_score:
			best_model_score = overall_composite_score
			torch.save(best_model, model_outfile)

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
			"best_val_score": best_val_score,
		})

	return training_stats

if __name__ == "__main__":

	parser = ap.ArgumentParser(description="Retrain the PiPPINN model using the best hyperparameters from a previous Optuna HPO study.")

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
	parser.add_argument("--runs", "-r",
		type=int, default=10,
		help="Number of training runs to perform per trial"
	)
	parser.add_argument("--num_trial", "-n",
		type=int, default=0,
		choices=range(0,5),
		help="Trial number to use for retraining (0 for best trial)"
	)
	parser.add_argument("--outdir", "-o",
		type=str, default=None,
		help="output directory to save the trained model and stats (defaults to the trials file directory)",
		metavar="<path/file>"
	)
	parser.add_argument("--prefix",
		type=str, default="PiPPINN_retrained",
		help="Prefix for the output model and stats files",
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

	if args.threads > 1:
		torch.set_num_threads(args.threads)
	if gpu_yes:
		device = torch.device("cuda")
	else:
		print("Cannot train without a GPU. Exiting.")
		sys.exit(1)

	input_data = torch.load(args.input_data, weights_only=False)

	if args.outdir is None:
		args.outdir = str(Path(args.trials).parent)

	best_trials_file = Path(args.outdir).joinpath("BestTrials.pkl")
	model_outfile = Path(args.outdir).joinpath(f"{args.prefix}_trial{args.num_trial}.pt")
	model_stats_file = Path(args.outdir).joinpath(f"{args.prefix}_trial{args.num_trial}_outstats.json")
	try:
		best_trials = pickle.load(open(best_trials_file,"rb"))
	except FileNotFoundError:
		storage = JournalStorage(JournalFileBackend(args.trials))
		study = optuna.load_study(study_name="PiPPINN_HPO",storage=storage)
		best_trials = choose_best_trials(study, topk=5)
		pickle.dump(best_trials, open(best_trials_file,"wb"))

	best_params = best_trials[args.num_trial][0].params

	model_stats = run_training(params=best_params, dataset=input_data, batch_size=40000, num_batches=None, device=device, model_outfile= model_outfile, threads=args.threads, runs=args.runs)

	with open(model_stats_file,"w") as f:
		json.dump(model_stats, f, indent=2)
	