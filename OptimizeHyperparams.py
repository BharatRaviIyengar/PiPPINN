import argparse as ap
from pathlib import Path
import sys
import torch
import TrainUtils as TU
from GVAE_model import GVAE_Model as GVAE, process_data_GVAE as process_data, TrainingParameters
import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.pruners import HyperbandPruner
import math

import gc


def run_training(params:dict, num_batches:int, batch_size:int, dataset:list, max_epochs = 200):
	""" Run training for a single trial with the given parameters."""

	learning_rate = params['learning_rate']
	dropout = params['dropout']
	weight_decay = params['weight_decay']
	patience = 15
	scheduler_factor = params['scheduler_factor']
	latent_dimension = params['latent_dimension']
	num_encoder_layers = params['num_encoder_layers']
	num_decoder_layers = params['num_decoder_layers']
	kld_coefficient = params['kld_coefficient']
	mse_coefficient = params['mse_coefficient']

	training_parameters = TrainingParameters(
		mse_coefficient = mse_coefficient,
		kld_coefficient = 0.0
	)

	median_centralities = [data["Train"].node_degree.median().item() for data in dataset]
	global_reference_centrality = sum(median_centralities)/len(median_centralities)

	batch_loader_params = {
		"supervision_fraction": 0.3,
		"uniform_message_fraction": params["uniform_message_fraction"],
		"fraction_from_unsupervised": 0.3,
		"max_neighbors": 45,
		"neighborhood_intensity": 1.0,
		"global_reference_centrality": global_reference_centrality,
		"global_reference_centrality_weight": params["global_reference_centrality_weight"],
		"false_negative_threshold": 0.4,
		"negative_label_hardness": 1.0,
		"track_coverage_multiple": True
	}

	data_for_training = [
		TU.generate_batch(
			data=data,
			batch_size=batch_size,
			num_batches=num_batches,
			batch_parameters=batch_loader_params
		)
		  for data in dataset]

	del dataset
	gc.collect()
	torch.cuda.empty_cache()

	# Initialize model and optimizer
	model = GVAE(
		input_dimension = data_for_training[0]["input_channels"],
		num_encoder_layers = num_encoder_layers,
		latent_dimension = latent_dimension,
		num_decoder_layers = num_decoder_layers,
		dropout = dropout,
	).cuda()

	optimizer = torch.optim.Adam(
		model.parameters(),
		lr=learning_rate,
		weight_decay=weight_decay
	)

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer= optimizer,
		mode='min',
		factor=scheduler_factor,
		patience=10,
		cooldown=2,
		min_lr=1e-6
	)

	kl_warmup = TU.ParameterScheduler(
		class_object = training_parameters,
		attr_name = "kld_coefficient",
		initial_value = 0.0,
		final_value = kld_coefficient,
		linear_change = True,
		factor = 10.0 
		)

	total_val_samples = sum(
		[
			data["num_val_samples"]	for data in data_for_training
	 	]
	 )

	preds_buf = torch.zeros(total_val_samples, dtype=torch.float32, device="cpu")
	labels_buf = torch.zeros(total_val_samples, dtype=torch.float32, device="cpu")

	model_device = next(model.parameters()).device

	train_losses_sum = torch.zeros(4, device=model_device)
	val_losses_sum = torch.zeros(4, device=model_device)

	best_composite_score = float('inf')
	val_loss_at_best_score = float('inf')
	train_loss_at_best_score = float('inf')
	auc_at_best_score = float('-inf')
	epochs_without_improvement = 0
	best_score_epoch = max_epochs
	best_auc = float('-inf')
	best_auc_epoch = max_epochs
	validation_epoch = 0
	
	for epoch in range(max_epochs):
		# Epoch loop BEGIN
		train_losses_sum.zero_()
		train_batch_count = 0

		kl_is_warmed_up = math.isclose(
			training_parameters.kld_coefficient,
			kld_coefficient,
		)

		model.train()
		# Training loop BEGIN (cover all datasets)
		for data in data_for_training:
			for batch in data["train_batch_loader"]:
				train_batch_count += 1
				train_output = process_data(
					batch,
					model=model,
					optimizer=optimizer,
					training_parameters=training_parameters,
					return_output=False
					)
				train_losses_sum += train_output["loss_values"]
		# Training loop END

		if not kl_is_warmed_up:
			kl_warmup.step()
			continue

		# Validation — only after KL has warmed up #
		val_losses_sum.zero_()
		val_batch_count = 0
		fill_idx = 0
		model.eval()
		# Validation loop BEGIN
		with torch.no_grad():
			for data in data_for_training:
				for batch in data["val_batch_loader"]:
					val_batch_count += 1
					val_output = process_data(
						batch,
						model=model,
						optimizer=optimizer,
						training_parameters=training_parameters,
						return_output=True
						)
					
					val_losses_sum += val_output["loss_values"]

					edge_logits = val_output["edge_prediction_logits"]
					edge_labels = val_output["edge_labels"]

					n = edge_logits.size(0)

					preds_buf[fill_idx:fill_idx + n] = edge_logits
					labels_buf[fill_idx:fill_idx + n] = (edge_labels > 0.5).float()

					fill_idx += n
		# Validation loop END

		validation_epoch+=1

		# Average losses
		average_train_losses = (train_losses_sum / train_batch_count).cpu().tolist()
		average_val_losses = (val_losses_sum / val_batch_count).cpu().tolist()

		assert fill_idx == total_val_samples

		# Early stopping logic
		auc = TU.auc_score(preds_buf, labels_buf)
		auc_penalty = 1 / (1 + math.exp(-20*(0.9 - auc)))
		composite_score = average_val_losses[0]*(1 + auc_penalty)
		if auc > best_auc:
			best_auc = auc
			best_auc_epoch = epoch + 1
		if best_composite_score > composite_score:
			best_composite_score = composite_score
			val_loss_at_best_score = average_val_losses[0]
			train_loss_at_best_score = average_train_losses[0]
			epochs_without_improvement = 0
			best_score_epoch = epoch + 1
			auc_at_best_score = auc
		else:
			epochs_without_improvement += 1
			
		yield {
		"epoch": epoch + 1,
		"validation_epoch": validation_epoch,
		"average_train_loss": average_train_losses,
		"average_val_loss": average_val_losses,
		"val_loss_at_best_score": val_loss_at_best_score,
		"train_loss_at_best_score": train_loss_at_best_score,
		"best_score_epoch": best_score_epoch,
		"learning_rate": optimizer.param_groups[0]['lr'],
		"auc_at_best_score": auc_at_best_score,
		"best_auc": best_auc,
		"best_auc_epoch": best_auc_epoch,
		"composite_score": composite_score,
		"best_composite_score": best_composite_score
		}

		# Step the scheduler
		scheduler.step(average_val_losses[0])
		
		if epochs_without_improvement >= patience:
			print(f"Early stopping triggered after {epoch + 1} epochs.")
			break

if __name__ == "__main__":

	parser = ap.ArgumentParser(description="Optimize hyperparameters for PiPPINN using Optuna")

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
		help="Load split data and negative edges from file (.pt)",
		required=True
	)
	parser.add_argument("--num_trials","-n",
		type=int,
		help="Number of trials to generate",
		default=50
	)
	parser.add_argument("--journal_file",
		type=str,
		help="Path to the Optuna journal file for storing study results",
		required=True
	)

	SEED = 48149
	torch.manual_seed(SEED)

	args = parser.parse_args()
	torch.set_num_threads(args.threads)
	torch.set_num_interop_threads(args.threads)

	if len(sys.argv) == 1:
		print("Error: essential arguments not provided.")
		parser.print_help() # Print the help message
		sys.exit(1)

	if not torch.cuda.is_available():
		raise RuntimeError("PiPPINN training requires an available CUDA GPU.")

	torch.cuda.manual_seed(SEED)
	torch.cuda.manual_seed_all(SEED)


	storage = JournalStorage(JournalFileBackend(args.journal_file))

	pruner = HyperbandPruner(min_resource=15, reduction_factor=3)
	sampler = TPESampler(seed=SEED, multivariate=True)

	print("Parsed arguments\n===================")
	for arg, value in vars(args).items():
		print(f"{arg}: {value}")

	print("Using device: cuda")

	dataset = torch.load(args.training_data, map_location="cpu", weights_only=False)
	input_channels = dataset[0]["Val"].x.size(1)

	def objective(trial):
		val_loss_at_best_score = float('inf')
		train_loss_at_best_score = float('inf')
		composite_score = float('inf')
		best_composite_score = float('inf')
		best_score_epoch = 0
		auc_at_best_score = float('-inf')
		best_auc = float('-inf')
		best_auc_epoch = 0
		composite_score = float('inf')
		params = {
			"learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
			"scheduler_factor": trial.suggest_float("scheduler_factor", 0.1, 0.5),
			"dropout": trial.suggest_float("dropout", 0.1, 0.35),
			"weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
			"uniform_message_fraction": trial.suggest_float("uniform_message_fraction", 0.2, 0.8),
			"global_reference_centrality_weight": trial.suggest_float("global_reference_centrality_weight", 0.0, 1.0),
			"negative_label_hardness": trial.suggest_float("negative_label_hardness", 0.1, 3.0, log=True),
			"latent_dimension": trial.suggest_categorical("latent_dimension", [64, 128, 256, 512]),
			"num_encoder_layers": trial.suggest_int("num_encoder_layers", 2, 4),
			"num_decoder_layers": trial.suggest_int("num_decoder_layers", 1, 3),
			"kld_coefficient": trial.suggest_float("kld_coefficient", 0.01, 1.0, log=True),
			"mse_coefficient": trial.suggest_float("mse_coefficient", 0.01, 1.0, log=True)
		}
		try:
			for result in run_training(params, args.num_batches, args.batch_size, dataset):
				validation_epoch = result["validation_epoch"]
				composite_score = result["composite_score"]
				val_loss_at_best_score = result["val_loss_at_best_score"]
				train_loss_at_best_score = result["train_loss_at_best_score"]
				best_score_epoch = result["best_score_epoch"]
				auc_at_best_score = result["auc_at_best_score"]
				best_auc = result["best_auc"]
				best_auc_epoch = result["best_auc_epoch"]
				best_composite_score = result["best_composite_score"]
				trial.report(composite_score, step=validation_epoch)
				if trial.should_prune():
					raise optuna.TrialPruned()
				
			trial.set_user_attr("best_score_epoch", best_score_epoch)
			trial.set_user_attr("train_loss_at_best_score", train_loss_at_best_score)
			trial.set_user_attr("val_loss_at_best_score", val_loss_at_best_score)
			trial.set_user_attr("auc_at_best_score", auc_at_best_score)
			trial.set_user_attr("best_auc", best_auc)
			trial.set_user_attr("best_auc_epoch", best_auc_epoch)
		except RuntimeError as e:
			if "out of memory" in str(e):
				print("CUDA OOM encountered, pruning trial")
				torch.cuda.empty_cache()
				raise optuna.TrialPruned()
			else:
				raise

		return best_composite_score
	
	study = optuna.create_study(
		study_name="PiPPINN_HPO",
		direction="minimize",
		sampler=sampler,
		storage=storage,
		load_if_exists=True,
		pruner=pruner
	)

	# if not study.user_attrs.get("enqueued", False):
	# 	for trial in best_trials:
	# 		params = trial.params
	# 		study.enqueue_trial(params)
	# 	study.set_user_attr("enqueued", True)

	study.optimize(objective, n_trials=args.num_trials)
