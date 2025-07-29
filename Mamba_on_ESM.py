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
chunk_size = 1022 # Limit for ESM models
max_tokens_per_batch = 65536
# Don't even try training without a GPU :D
gpu = torch.cuda.is_available()


if not gpu:
    raise RuntimeError("This script requires a GPU to run.")

def chunk_sequences_optimal(sequence, chunk_size, seqid = "", stride=1):
    """Yield chunks of sequences of a specified size."""
    if stride > chunk_size:
        warn("Stride greater than chunk size. You will lose some positions in the sequence.")
    L = len(sequence)
    range_stop = (L - chunk_size + stride)
    if range_stop % stride == 0:
        starts = torch.arange(0, range_stop, stride)
    else:
        n_chunks =(torch.ceil((L - chunk_size) // stride)) + 1
        stride = torch.floor(L - chunk_size) // (n_chunks - 1)
        starts = torch.arange(n_chunks) * stride
    seqlist = [(f"{seqid}_{i}", sequence[start : start+chunk_size])  for i,start in enumerate(starts)]
    offsets = torch.arange(chunk_size).unsqueeze(1)
    position_embeddings = (starts + offsets) / L
    return seqlist, position_embeddings.T


def get_esm_embeddings(model, layers, sequences: List[Tuple[str,str]]):
    """Get ESM embeddings for a list of sequences."""
    data_loader = prepare_dataloader(sequences)
    all_reps = []
    all_labels = []
    for labels, strs, toks in data_loader:
      bsize = len(strs)
      if gpu:
        toks = toks.to(device="cuda", non_blocking=True)
      out = model(toks, repr_layers=[layers])
      reps = [out["representations"][layers][i][1:len(strs[i])+1].to(device="cpu") for i in range(bsize)]
      all_reps.extend(reps)
      all_labels.extend(labels)
    return all_labels, all_reps

def get_long_esm_embeddings(model, layers, sequences:Tuple[str,str], chunk_size:int=chunk_size, stride=1):
    """Get ESM embeddings for long sequences by chunking."""
    all_reps = []
    all_labels = []
    for seqid, sequence in sequences:
        chunks, positional_embeddings = chunk_sequences_optimal(sequence, chunk_size, seqid = seqid, stride = stride)
        reps = torch.empty(len(chunks), chunk_size, model.embed_dim + 1)
        reps[:,:,-1] = positional_embeddings
        data_loader = prepare_dataloader(chunks)
        offset = 0
        for _, _, toks in data_loader:
            batch_size = len(toks)
            if gpu:
                toks = toks.to(device="cuda", non_blocking=True)
            out = model(toks, repr_layers=[layers])
            reps[offset:offset+batch_size,:,:model.embed_dim] = out["representations"][layers].to("cpu")
            offset += batch_size
        all_reps.append(reps)
        all_labels.append(seqid)
    return all_labels, all_reps