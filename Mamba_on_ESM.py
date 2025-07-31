import torch
from esm import pretrained, Alphabet
import argparse as ap
from pathlib import Path
from warnings import warn
from mamba_ssm import Mamba
from EncodeProteins import load_model_without_regression, get_batch_idx, prepare_dataloader
from Typing import Tuple, List
import numpy as np
from collections import namedtuple

extra_toks_per_seq = 1
max_chunk_size = 1022 # Limit for ESM models
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

def get_long_esm_embeddings(model, layers, sequences:Tuple[str,str], chunk_size:int=max_chunk_size, stride=1, device="cuda"):
    """Get ESM embeddings for long sequences by chunking.
    Args:
        model: ESM model to use for embeddings.
        layers: Layer from which to extract embeddings.
        sequences (List[Tuple[str,str]]): List of tuples containing sequence identifiers and sequences.
        chunk_size (int): Size of the chunks to split the sequences into. Default is set to max_chunk_size (1022).
        stride (int): Stride for chunking the sequences. Default is 1.
        device (str): Device to run the model on. Default is "cuda".
    """
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

def optimial_stride(seq_length, chunk_size, max_stride = max_chunk_size //3, factor = 3):
    """Calculate the optimal stride for chunking sequences based on their length and chunk size.
    Args:
        seq_length (torch.tensor): Length(s) of the sequence(s) to be chunked.
        chunk_size (int): Size of the chunks to split the sequence(s) into.
        max_stride (int): Maximum allowed stride.
    Returns:
        torch.tensor : Optimal stride for chunking.
    """
    assert torch.all(seq_length > chunk_size), "Sequence length must be greater than chunk size"
    stride = seq_length // (chunk_size * factor)
    return torch.clamp(stride, min=1, max=max_stride)

def bucketize_sequences(sequences: List[str], seqids: List[str], bucket_width = 50, embedding_size = 2561, short_seq_size_limit = max_chunk_size, chunk_size = max_chunk_size, factor = 3, max_batch_memory = None) -> List[List[int]]:
    """Bucketize sequences into batches of short and long sequences, based on their lengths and size limit of short sequences.
    Args:
        sequences (List[str]): List of protein sequences.
        seqids (List[str]): List of sequence identifiers corresponding to the sequences.
        bucket_width (int): Width of the buckets for grouping sequences by length.
        embedding_size (int): Size of the embeddings used in the model. Default is 2561 for ESM2-3B model.
        short_seq_size_limit (int): Maximum length for a sequence to be considered short. Default is set to max size processable by ESM2 (1022).
        max_batch_memory (float): Maximum memory allowed for a batch in MB. If None, no limit is applied.
    """
    seq_lengths = torch.tensor([len(seq)  for seq in sequences], dtype=torch.int32)
    short_seqs = seq_lengths <= short_seq_size_limit
    strides = torch.ones(len(sequences), dtype=torch.int32)
    strides[~short_seqs] = optimial_stride(seq_lengths[~short_seqs], chunk_size, factor=factor)
    def process_subset(subset_mask):
        indices = torch.nonzero(subset_mask).squeeze(1)
        lengths = seq_lengths[indices]
        boundaries = (bucket_width * lengths // bucket_width).unique(sorted=False)
        buckets = torch.bucketize(lengths, boundaries)
        result_batches = []

        for boundary in boundaries:
            batch_idxs = indices[buckets == (boundary == boundaries).nonzero(as_tuple=False).item()]
            batch_memory = seq_lengths[batch_idxs].sum() * 4 * embedding_size / 1024**2  # MB
            num_sub_batches = int(batch_memory // max_batch_memory) + 1 if max_batch_memory else 1
            sub_batches = torch.chunk(batch_idxs, num_sub_batches)
            batch_seqs = [[(seqids[i], sequences[i]) for i in sb] for sb in sub_batches]
            result_batches.extend(batch_seqs)

        return result_batches
    batches = namedtuple("batches", ["short", "long"])
    return batches(short=process_subset(short_seqs), long = process_subset(~short_seqs))