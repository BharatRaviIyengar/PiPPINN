import torch
from esm import pretrained, Alphabet
import argparse as ap
from pathlib import Path
from warnings import warn
from mamba_ssm import Mamba
from EncodeProteins import load_model_without_regression, get_batch_idx, prepare_dataloader
from Typing import Tuple
import numpy as np
extra_toks_per_seq = 1

# Don't even try training without a GPU :D
gpu = torch.cuda.is_available()

if not gpu:
    raise RuntimeError("This script requires a GPU to run.")

def chunk_sequences(sequence, chunk_size, stride=1):
    """Yield chunks of sequences of a specified size."""
    nchunks = (len(sequence) - chunk_size + stride)
    if stride > chunk_size:
        warn("Stride greater than chunk size. You will lose some positions in the sequence.")
    return [sequence[i:i+chunk_size] for i in range(0, nchunks, stride)]

def bidirectional_chunk_seq(seq: str, chunk_size: int, stride: int):

	L = len(seq)
	center = L // 2
	x = (L - chunk_size + stride) % stride  # uncovered length in one-directional

	# Shift center so that we align both directions evenly
	# This helps balance the ends and eliminate uncovered tail
	start_center = center - x // 2

	# Generate chunk starts to the left
	left_starts = list(range(start_center - stride, -1, -stride))[::-1]

	# Generate chunk starts to the right
	right_starts = list(range(start_center, L - chunk_size + 1, stride))

	all_starts = left_starts + right_starts
	chunks = [seq[i:i+chunk_size] for i in all_starts]

	# Centered positional encoding
	center_index = len(left_starts)  # index in the list where position is 0
	positions = np.arange(len(chunks)) - center_index
	positions = positions / max(1, len(positions) // 2)  # normalized

	return chunks, positions


chunk_size = 1022 # Limit for ESM models

def get_esm_embeddings(sequences: Tuple[str,str]):
    """Get ESM embeddings for a list of sequences."""
    data_loader = prepare_dataloader(sequences)
    all_reps = []
    all_labels = []
    for _, (labels, strs, toks) in enumerate(data_loader):
      bsize = len(strs)
      if gpu:
        toks = toks.to(device="cuda", non_blocking=True)
      out = model(toks, repr_layers=[layers])
      reps = [out["representations"][layers][i][1:len(strs[i])+1].to(device="cpu") for i in range(bsize)]
      all_reps.extend(reps)
      all_labels.extend(labels)
    return all_labels, all_reps

# TODO implement long seq esm inference with output as a list of 3D tensors

def get_long_esm_embeddings(sequences:Tuple[str,str], chunk_size:int=chunk_size, stride=1):
    """Get ESM embeddings for long sequences by chunking."""
    all_reps = []
    all_labels = []
    for seqid, sequence in sequences:
        chunks = chunk_sequences(sequence, chunk_size, stride)
        _, chunk_reps = get_esm_embeddings([chunk])
        for i,chunk in chunks:
            
            reps.append(chunk_reps[0])  # Assuming single sequence per chunk
        all_reps.append(torch.stack(reps))
        all_labels.append(labels[0])  # Assuming single label per sequence
    return all_labels, all_reps