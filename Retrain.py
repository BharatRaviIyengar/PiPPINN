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

def run_training(params:dict, num_batches, batch_size:int, dataset:list, model_outfile, device:torch.device, max_epochs = 200, threads:int=1, dual_head:bool=False):
	""" Run training for a single trial with the given parameters."""

	centrality_fraction = params['centrality_fraction']
	hidden_channels = 1024
	dropout = params['dropout']
	weight_decay = params['weight_decay']
	patience = 20
	scheduler_factor = params['scheduler_factor']
	nbr_wt_intensity = params['nbr_weight_intensity']
	GNN_head_weight = params['GNN_head_weight']
	NOD_head_weight = 1- GNN_head_weight
	head_weights = [GNN_head_weight, NOD_head_weight]

	data_for_training = [utils.generate_batch(data, num_batches, batch_size, centrality_fraction, nbr_wt_intensity=nbr_wt_intensity, device=device, threads=threads) for data in dataset]

	del dataset
	gc.collect()
	torch.cuda.empty_cache()

	# Initialize model and optimizer
	model = utils.DualLayerModel(
			in_channels=data_for_training[0]["input_channels"],
			hidden_channels=hidden_channels,
			dropout=dropout,
			gnn_dropout=0.5
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

	GNN_dropout_scheduler = utils.DecayScheduler(model, 'gnn_dropout', initial_value=0.5, factor=GNN_dropout_factor, cooldown=2, min_value=0.05)

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
		GNN_dropout_scheduler.step()

		# Early stopping logic
		if average_val_loss < best_val_loss:
			best_val_loss = average_val_loss
			best_train_loss = average_train_loss
			epochs_without_improvement = 0
			best_epoch = epoch
			torch.save(model.state_dict(), model_outfile)
		else:
			epochs_without_improvement += 1
		

		if epochs_without_improvement >= patience:
			print(f"Early stopping triggered after {epoch + 1} epochs.")
			break
	
	return {"best_val_loss" : best_val_loss, "best_train_loss" : best_train_loss, "best_epoch" : best_epoch}

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