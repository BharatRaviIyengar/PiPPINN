"""CUDA smoke test for the current PiPPINN GVAE training path.

This deliberately avoids Optuna so numerical and memory failures are reported at
the batch where they occur. It uses the same sampler settings as
OptimizeHyperparams.py and, by default, also runs one production-sized
validation batch.

Each batch reports neighbor entries retained and rejected by the per-node cap.
Counts include both directions of each message edge.

Example:
    python DebugGVAE.py \
        --training-data Data_For_Training.pt \
        --batch-size 40000 \
        --steps 10
"""

import argparse
import contextlib
import math
import sys

import torch
import torch.nn.functional as F

import TrainUtils as TU
from GVAE_model import GVAE_Model, KL_loss, TrainingParameters


LOSS_NAMES = ("total", "bce", "mse", "kld")


class NonFiniteError(RuntimeError):
	"""Raised as soon as a non-finite tensor is encountered."""


def print_settings(title: str, settings: dict) -> None:
	print(title, flush=True)
	for name, value in sorted(settings.items()):
		print(f"  {name}: {value}", flush=True)


def tensor_summary(tensor: torch.Tensor) -> str:
	detached = tensor.detach()
	finite = torch.isfinite(detached)
	num_nonfinite = detached.numel() - finite.sum().item()
	message = (
		f"shape={tuple(detached.shape)}, dtype={detached.dtype}, "
		f"device={detached.device}, nonfinite={num_nonfinite}"
	)
	if finite.any():
		finite_values = detached[finite].float()
		message += (
			f", finite_min={finite_values.min().item():.6g}, "
			f"finite_max={finite_values.max().item():.6g}"
		)
	return message


def require_finite(name: str, tensor: torch.Tensor) -> None:
	if tensor.is_floating_point() and not torch.isfinite(tensor).all():
		raise NonFiniteError(f"{name} is non-finite: {tensor_summary(tensor)}")


def check_batch(batch) -> None:
	for name in (
		"node_features",
		"supervision_labels",
		"positive_supervision_weights",
		"neighborhood_weights",
	):
		require_finite(f"batch.{name}", getattr(batch, name))


def check_model_parameters(model: torch.nn.Module, stage: str) -> None:
	for name, parameter in model.named_parameters():
		require_finite(f"parameter {name!r} {stage}", parameter)


def gradient_summary(model: torch.nn.Module) -> tuple[float, str, float]:
	norm_squared = 0.0
	max_abs = -1.0
	max_abs_name = ""
	found_gradient = False
	missing_gradients = []

	for name, parameter in model.named_parameters():
		if parameter.grad is None:
			if parameter.requires_grad:
				missing_gradients.append(name)
			continue
		found_gradient = True
		gradient = parameter.grad.detach()
		require_finite(f"gradient {name!r}", gradient)

		parameter_norm = gradient.float().norm(2).item()
		norm_squared += parameter_norm * parameter_norm
		parameter_max = gradient.abs().max().item()
		if parameter_max > max_abs:
			max_abs = parameter_max
			max_abs_name = name

	if not found_gradient:
		raise RuntimeError("Backward completed without producing any parameter gradients")
	if missing_gradients:
		raise RuntimeError(
			"Backward left trainable parameters without gradients: "
			+ ", ".join(missing_gradients)
		)
	global_norm = math.sqrt(norm_squared)
	if not math.isfinite(global_norm):
		raise NonFiniteError(
			"The global gradient norm overflowed even though individual gradient "
			"elements were finite"
		)

	return global_norm, max_abs_name, max_abs


def memory_summary(device: torch.device) -> str:
	gib = 1024 ** 3
	free, total = torch.cuda.mem_get_info(device)
	return (
		f"allocated={torch.cuda.memory_allocated(device) / gib:.2f} GiB, "
		f"reserved={torch.cuda.memory_reserved(device) / gib:.2f} GiB, "
		f"peak_allocated={torch.cuda.max_memory_allocated(device) / gib:.2f} GiB, "
		f"peak_reserved={torch.cuda.max_memory_reserved(device) / gib:.2f} GiB, "
		f"device_free={free / gib:.2f} GiB, device_total={total / gib:.2f} GiB"
	)


def neighborhood_summary(batch) -> str:
	"""Count directed neighbor entries discarded by the sampler's per-node cap."""
	neighbor_counts = (batch.neighborhood_matrix >= 0).sum(dim=1)
	neighbor_cap = batch.neighborhood_matrix.size(1)
	candidates = batch.num_neighbor_candidates
	kept = neighbor_counts.sum().item()
	rejected = candidates - kept
	rejection_fraction = rejected / candidates if candidates else 0.0
	nodes_at_cap = (neighbor_counts == neighbor_cap).sum().item()
	nodes_without_neighbors = (neighbor_counts == 0).sum().item()
	return (
		f"neighbor_candidates={candidates}, neighbors_kept={kept}, "
		f"neighbors_rejected={rejected} ({rejection_fraction:.1%}), "
		f"neighbor_cap={neighbor_cap}, nodes_at_neighbor_cap={nodes_at_cap}, "
		f"nodes_without_neighbors={nodes_without_neighbors}"
	)


def compute_losses(
	batch,
	model: GVAE_Model,
	training_parameters: TrainingParameters,
):
	model_output = model(
		batch.node_features,
		batch.supervision_edges,
		batch.neighborhood_matrix,
		batch.neighborhood_weights,
	)
	edge_logits, edge_strengths, node_mu, node_logvar = model_output[:4]

	for name, tensor in (
		("edge_logits", edge_logits),
		("edge_strengths", edge_strengths),
		("node_mu", node_mu),
		("node_logvar", node_logvar),
	):
		require_finite(name, tensor)

	bce = F.binary_cross_entropy_with_logits(
		edge_logits,
		batch.supervision_labels,
	)
	mse = F.mse_loss(
		edge_strengths[batch.num_negative_edges:],
		batch.positive_supervision_weights,
	)
	kld = KL_loss(node_mu, node_logvar)
	total = (
		bce
		+ training_parameters.mse_coefficient * mse
		+ training_parameters.kld_coefficient * kld
	)

	losses = torch.stack((total, bce, mse, kld))
	for name, value in zip(LOSS_NAMES, losses):
		require_finite(f"{name} loss", value)

	return losses


def run_training_step(
	step: int,
	batch,
	model: GVAE_Model,
	optimizer: torch.optim.Optimizer,
	training_parameters: TrainingParameters,
	detect_anomaly: bool,
	device: torch.device,
) -> None:
	check_batch(batch)
	print(f"train step {step} neighborhoods: {neighborhood_summary(batch)}", flush=True)
	check_model_parameters(model, "before forward")
	optimizer.zero_grad(set_to_none=True)

	anomaly_context = (
		torch.autograd.detect_anomaly(check_nan=True)
		if detect_anomaly
		else contextlib.nullcontext()
	)
	with anomaly_context:
		losses = compute_losses(batch, model, training_parameters)
		losses[0].backward()

	gradient_norm, largest_name, largest_gradient = gradient_summary(model)
	optimizer.step()
	check_model_parameters(model, "after optimizer step")

	loss_values = ", ".join(
		f"{name}={value:.6g}"
		for name, value in zip(LOSS_NAMES, losses.detach().cpu().tolist())
	)
	print(
		f"train step {step}: nodes={batch.node_features.size(0)}, "
		f"supervision_edges={batch.supervision_edges.size(1)}, "
		f"positive_supervision_edges={batch.positive_supervision_weights.size(0)}, "
		f"negative_supervision_edges={batch.num_negative_edges}; {loss_values}; "
		f"grad_norm={gradient_norm:.6g}; "
		f"max_abs_grad={largest_gradient:.6g} ({largest_name}); "
		f"{memory_summary(device)}",
		flush=True,
	)


def run_validation_step(
	batch,
	model: GVAE_Model,
	training_parameters: TrainingParameters,
	device: torch.device,
) -> None:
	check_batch(batch)
	print(f"validation neighborhoods: {neighborhood_summary(batch)}", flush=True)
	check_model_parameters(model, "before validation")
	model.eval()
	with torch.no_grad():
		losses = compute_losses(batch, model, training_parameters)

	loss_values = ", ".join(
		f"{name}={value:.6g}"
		for name, value in zip(LOSS_NAMES, losses.detach().cpu().tolist())
	)
	print(
		f"validation step: nodes={batch.node_features.size(0)}, "
		f"supervision_edges={batch.supervision_edges.size(1)}, "
		f"positive_supervision_edges={batch.positive_supervision_weights.size(0)}, "
		f"negative_supervision_edges={batch.num_negative_edges}; {loss_values}; "
		f"{memory_summary(device)}",
		flush=True,
	)


def parse_args():
	parser = argparse.ArgumentParser(
		description="Check the PiPPINN GVAE for non-finite gradients and CUDA OOMs",
	)
	parser.add_argument("--training-data", required=True)
	parser.add_argument("--dataset-index", type=int, default=0)
	parser.add_argument("--batch-size", type=int, default=40000)
	parser.add_argument("--steps", type=int, default=10)
	parser.add_argument("--threads", type=int, default=1)
	parser.add_argument("--seed", type=int, default=48149)

	parser.add_argument("--learning-rate", type=float, default=1e-3)
	parser.add_argument("--weight-decay", type=float, default=1e-3)
	parser.add_argument("--dropout", type=float, default=0.35)
	parser.add_argument("--latent-dimension", type=int, default=512)
	parser.add_argument("--num-encoder-layers", type=int, default=4)
	parser.add_argument("--num-decoder-layers", type=int, default=3)
	parser.add_argument("--edge-chunk-size", type=int, default=12000)
	parser.add_argument("--kld-coefficient", type=float, default=1.0)
	parser.add_argument("--mse-coefficient", type=float, default=1.0)
	parser.add_argument("--uniform-message-fraction", type=float, default=0.2)
	parser.add_argument("--global-reference-centrality-weight", type=float, default=0.5)

	parser.add_argument(
	"--detect-anomaly",
	action="store_true",
	help="Enable PyTorch autograd anomaly detection",
	)
	parser.add_argument(
		"--validation",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Run one production-sized validation batch (default: enabled)",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	if args.steps <= 0:
		raise ValueError("--steps must be positive")
	if args.batch_size <= 0:
		raise ValueError("--batch-size must be positive")
	if args.edge_chunk_size <= 0:
		raise ValueError("--edge-chunk-size must be positive")
	if not torch.cuda.is_available():
		raise RuntimeError("This debugger requires a CUDA GPU")

	torch.set_num_threads(args.threads)
	torch.set_num_interop_threads(args.threads)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	device = torch.device("cuda")
	print_settings("Command-line configuration", vars(args))

	print(f"Loading {args.training_data}", flush=True)
	dataset = torch.load(args.training_data, map_location="cpu", weights_only=False)
	if not 0 <= args.dataset_index < len(dataset):
		raise IndexError(
			f"--dataset-index {args.dataset_index} is outside [0, {len(dataset) - 1}]"
		)

	global_reference_centrality = sum(
		data["Train"].node_degree.median().item() for data in dataset
	) / len(dataset)
	selected_data = dataset[args.dataset_index]
	data_name = selected_data.get("Data_name", f"dataset[{args.dataset_index}]")

	batch_parameters = {
		"supervision_fraction": 0.3,
		"uniform_message_fraction": args.uniform_message_fraction,
		"fraction_from_unsupervised": 0.3,
		"max_neighbors": 45,
		"neighborhood_intensity": 1.0,
		"global_reference_centrality": global_reference_centrality,
		"global_reference_centrality_weight": args.global_reference_centrality_weight,
		"false_negative_threshold": 0.4,
		"negative_label_hardness": 1.0,
		"track_coverage_multiple": True,
	}
	print_settings("Effective batch-sampler parameters", batch_parameters)
	loaders = TU.generate_batch(
		data=selected_data,
		num_batches=args.steps,
		batch_size=args.batch_size,
		batch_parameters=batch_parameters,
	)

	model = GVAE_Model(
		input_dimension=loaders["input_channels"],
		num_encoder_layers=args.num_encoder_layers,
		latent_dimension=args.latent_dimension,
		num_decoder_layers=args.num_decoder_layers,
		dropout=args.dropout,
	).to(device)
	model.decoder.edge_chunk_size = args.edge_chunk_size
	optimizer = torch.optim.Adam(
		model.parameters(),
		lr=args.learning_rate,
		weight_decay=args.weight_decay,
	)
	training_parameters = TrainingParameters(
		mse_coefficient=args.mse_coefficient,
		kld_coefficient=args.kld_coefficient,
	)

	parameter_count = sum(parameter.numel() for parameter in model.parameters())
	print_settings(
		"Effective model parameters",
		{
			"input_dimension": loaders["input_channels"],
			"latent_dimension": args.latent_dimension,
			"num_encoder_layers": args.num_encoder_layers,
			"num_decoder_layers": args.num_decoder_layers,
			"dropout": args.dropout,
			"edge_chunk_size": model.decoder.edge_chunk_size,
			"similarity_block_size": model.decoder.similarity_block_size,
			"parameter_count": parameter_count,
		},
	)
	print(
		f"Testing {data_name!r} with {parameter_count:,} parameters on "
		f"{torch.cuda.get_device_name(device)}",
		flush=True,
	)
	print(f"Initial CUDA memory: {memory_summary(device)}", flush=True)
	torch.cuda.reset_peak_memory_stats(device)

	try:
		model.train()
		for step, batch in enumerate(loaders["train_batch_loader"], start=1):
			run_training_step(
				step,
				batch,
				model,
				optimizer,
				training_parameters,
				args.detect_anomaly,
				device,
			)

		if args.validation:
			validation_batch = next(iter(loaders["val_batch_loader"]))
			run_validation_step(
				validation_batch,
				model,
				training_parameters,
				device,
			)
	except NonFiniteError as error:
		print(f"NON-FINITE FAILURE: {error}", file=sys.stderr, flush=True)
		print(f"CUDA memory at failure: {memory_summary(device)}", file=sys.stderr)
		return 2
	except torch.OutOfMemoryError as error:
		print(f"CUDA OOM: {error}", file=sys.stderr, flush=True)
		print(f"CUDA memory at OOM: {memory_summary(device)}", file=sys.stderr)
		return 3
	except RuntimeError as error:
		if "out of memory" not in str(error).lower():
			raise
		print(f"CUDA OOM: {error}", file=sys.stderr, flush=True)
		print(f"CUDA memory at OOM: {memory_summary(device)}", file=sys.stderr)
		return 3

	print("PASS: all checked tensors and gradients were finite; no CUDA OOM occurred.")
	return 0


if __name__ == "__main__":
	try:
		exit_code = main()
	except torch.OutOfMemoryError as error:
		print(f"CUDA OOM before batch execution: {error}", file=sys.stderr, flush=True)
		if torch.cuda.is_available():
			print(
				f"CUDA memory at OOM: {memory_summary(torch.device('cuda'))}",
				file=sys.stderr,
			)
		exit_code = 3
	except RuntimeError as error:
		if "out of memory" not in str(error).lower():
			raise
		print(f"CUDA OOM before batch execution: {error}", file=sys.stderr, flush=True)
		if torch.cuda.is_available():
			print(
				f"CUDA memory at OOM: {memory_summary(torch.device('cuda'))}",
				file=sys.stderr,
			)
		exit_code = 3
	raise SystemExit(exit_code)
