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

def get_long_esm_embeddings(model, layers, sequences:Tuple[str,str], chunk_size:int=chunk_size, stride=1, device="cuda"):
    """Get ESM embeddings for long sequences by chunking."""
    all_reps = []
    all_labels = []
    for seqid, sequence in sequences:
        chunks, positional_embeddings = chunk_sequences_optimal(sequence, chunk_size, seqid=seqid, stride=stride)
        T = len(chunks) * chunk_size
        reps = torch.empty(T, model.embed_dim + 1, device = device)
        # Positional embeddings
        reps[:, -1] = positional_embeddings.flatten()
        data_loader = prepare_dataloader(chunks)
        offset = 0
        for _, _, toks in data_loader:
            toks = toks.to(device = device, non_blocking=True)
            out = model(toks, repr_layers=[layers])
            embedding = out["representations"][layers].reshape(-1, model.embed_dim)
            reps[offset:offset + embedding.size(0), :model.embed_dim] = embedding
            offset += embedding.size(0)
        all_reps.append(reps.to(device="cpu"))
        all_labels.append(seqid)
    return all_labels, all_reps

def bucketize_sequences(sequences: List[str], seqids: List[str], bucket_width = 50, embedding_size = 2561, max_batch_memory = None) -> List[List[int]]:
    """Bucketize sequences into batches based on their lengths."""
    seq_lengths = torch.tensor([len(seq)  for seq in sequences], dtype=torch.int32)
    boundaries = (bucket_width * seq_lengths // bucket_width).unique()
    buckets = torch.bucketize(seq_lengths, boundaries)
    batches = []
    for i in boundaries.unique():
        batch_indices = (buckets == i).nonzero(as_tuple=False).squeeze(1)
        batch_memory = seq_lengths[batch_indices].sum() * 4 * embedding_size/1024**2  # in MB
        num_sub_batches = (batch_memory // max_batch_memory) + 1 if max_batch_memory else 1
        sub_batches = torch.chunk(batch_indices, num_sub_batches)
        batch_sequences = [[(seqids[idx], sequences[idx]) for idx in sub_batch] for sub_batch in sub_batches]
        batches.extend(batch_sequences)
    return batches