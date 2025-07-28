import torch
from esm import pretrained, Alphabet
import argparse as ap
from pathlib import Path
from warnings import warn
from mamba_ssm import Mamba
from EncodeProteins import load_model_without_regression, get_batch_idx, prepare_dataloader
from Typing import Tuple, List
import numpy as np
extra_toks_per_seq = 1

# Don't even try training without a GPU :D
gpu = torch.cuda.is_available()

chunk_size = 1022 # Limit for ESM models
if not gpu:
    raise RuntimeError("This script requires a GPU to run.")

def chunk_sequences_optimal(sequence, chunk_size, seqid = "", stride=1):
    """Yield chunks of sequences of a specified size."""
    if stride > chunk_size:
        warn("Stride greater than chunk size. You will lose some positions in the sequence.")
    L = len(sequence)
    range_stop = (L - chunk_size + stride)
    if range_stop % stride == 0:
        starts = np.arange(0, range_stop, stride)
    else:
        n_chunks = int(np.ceil((L - chunk_size) / stride)) + 1
        stride = int(np.floor(L - chunk_size) / (n_chunks - 1))
        starts = np.arange(n_chunks) * stride
    seqlist = [(seqid+str(i), sequence[i:i+chunk_size])  for i in starts]
    position_embedding = torch.tensor(starts / L, dtype=torch.float32)
    return seqlist, position_embedding


def get_esm_embeddings(sequences: List[Tuple[str,str]]):
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