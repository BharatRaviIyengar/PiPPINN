import torch
from esm import Alphabet, pretrained
import sys, os
import argparse as ap
from pathlib import Path

esmdict = {
		"15B": {"name": "esm2_t48_15B_UR50D"},
		"3B": {"name" : "esm2_t36_3B_UR50D"},
		"650M":{"name" : "esm2_t33_650M_UR50D"}
}

extra_toks_per_seq = 1

gpu = torch.cuda.is_available()

def chunk_list(lst, k):
	"""Split a list into k chunks."""
	return [lst[i:i + k] for i in range(0, len(lst), k)]

def load_model_without_regression(model_location):
	"""Load from local path. Ignore regression weights"""
	model_location = Path(model_location)
	model_data = torch.load(str(model_location), map_location="cpu", weights_only = False)
	model_name = model_location.stem
	regression_data = None
	return pretrained.load_model_and_alphabet_core(model_name, model_data, regression_data)

def get_batch_idx(seqlist):
	"""Create batch indices based on token count."""
	batch_indices = []
	current_batch = []
	current_batch_token_count = 0

	for idx, (_, seq) in enumerate(seqlist):
		seq_len = len(seq) + extra_toks_per_seq

		if current_batch_token_count + seq_len > args.toks_per_batch:
			batch_indices.append(current_batch)
			current_batch = []
			current_batch_token_count = 0

		current_batch.append(idx)
		current_batch_token_count += seq_len

	if current_batch:
		batch_indices.append(current_batch)

	return batch_indices
		
def prepare_dataloader(seqlist):
	"""Prepare PyTorch DataLoader."""
	batches = get_batch_idx(seqlist)
	return torch.utils.data.DataLoader(
		seqlist, collate_fn= batch_converter, batch_sampler=batches
	)

def getreps(seqlist):
	"""Get representations for sequences that fit into ESM2 transformer model (<=1022)"""
	data_loader = prepare_dataloader(seqlist)
	all_reps = []
	all_labels = []
	for _, (labels, strs, toks) in enumerate(data_loader):
		bsize = len(strs)
		if gpu:
			toks = toks.to(device="cuda", non_blocking=True)
		out = model(toks, repr_layers=[layers])
		reps = [out["representations"][layers][i][1:len(strs[i])+1].to(device="cpu") for i in range(bsize)]
		seqreps = [r.mean(0) for r in reps]
		all_reps.extend(seqreps)
		all_labels.extend(labels)
	return all_labels, all_reps

def getreps_for_longseq(longseqlist):
	"""Get representations for long sequences."""
	all_seqids = []
	all_reps = []
	for seqid, seqchunklist in longseqlist:
		_, chunkreps = getreps(seqchunklist)
		fullrep = torch.stack(chunkreps).mean(0)
		all_reps.append(fullrep)
		all_seqids.append(seqid)
	return all_seqids, all_reps

def run_inference(shortseqs,longseqs, len_emb_short, len_emb_long,outfile):
	"""Run inference and save results."""
	short_seqids, short_seqreps = getreps(shortseqs)
	long_seqids, long_seqreps = getreps_for_longseq(longseqs)
	all_seqids = short_seqids + long_seqids
	all_seqreps = torch.cat([short_seqreps, long_seqreps], dim=0) 
	len_embs = torch.cat([len_emb_short, len_emb_long], dim=0)
	all_seqreps = torch.cat(all_seqreps, len_embs, dim=1)
	torch.save({"seqids": all_seqids, "representations": all_seqreps},outfile)

def concatenate_pt_files(file_list, output_file):
	"""Concatenate multiple .pt files into a single .pt file."""
	all_seqids = []
	all_representations = []

	try:
		for pt_file in file_list:
			data = torch.load(pt_file)
			all_seqids.extend(data["seqids"])
			all_representations.append(data["representations"])

		# Concatenate all representations
		concatenated_representations = torch.cat(all_representations, dim=0)

		# Save the concatenated result
		torch.save({"seqids": all_seqids, "representations": concatenated_representations}, output_file)
		print(f"Concatenated .pt files saved to {output_file}")

		# Remove temporary files after successful concatenation
		for pt_file in file_list:
			os.remove(pt_file)

	except Exception as e:
		print(f"Error during concatenation: {e}")
		print("Temporary files were not deleted to avoid data loss.")
		raise

if __name__ == "__main__":
	
	parser = ap.ArgumentParser(
			description="In Silico evolution of proteins towards structure",
			formatter_class=ap.ArgumentDefaultsHelpFormatter
		)
	parser.add_argument(
		"--input", "-i", 
		type=str, 
		help="Input sequence file (tab separated: first column = ID, second column = sequence)", 
		metavar="<path/filename>",
		required=True
	)
	parser.add_argument(
		"--window", "-w", 
		type=int, 
		help="Overlap window for long sequences",
		choices=range(1,1022), 
		metavar= "[1-1022]",
		default = 1
	)
	parser.add_argument(
		"--modelname", "-m", 
		type=str, 
		help="ESM-2 model to use for encoding", 
		choices=["15B", "3B", "650M"], 
		default="3B"
	)
	parser.add_argument(
		"--modelpath",
		type=str,
		metavar="<Path>",
		default=None
	)
	parser.add_argument(
		"--toks_per_batch", 
		type=int, 
		help="Number of tokens per batch of sequences to be processed by PyTorch-ESM", 
		default=4096, 
		metavar="<Int>"
	)
	parser.add_argument(
		"--outfile", "-o", 
		type=str,
		help="Output file", 
		default = None, 
		metavar="<path/filename>"
	)
	parser.add_argument(
		"--output_batches", 
		type=int,
		help="Create output file in batches of N", 
		default = 1, 
		metavar="<Int>"
	)

	args = parser.parse_args()
	if len(sys.argv) == 1:
		print("Error: essential arguments not provided.")
		parser.print_help() # Print the help message
		sys.exit(1)
		
	print("Parsed arguments\n===================")
	for arg, value in vars(args).items():
		print(f"{arg}: {value}")

	if args.outfile is None:
		args.outfile = f"{Path(args.infile).with_suffix('')}_embeddings_{args.model}.pt"

	chunk_size = 1022

	shortseqs = []
	longseqs = []
	len_shortseqs = []
	len_longseqs = []

	if args.modelpath is not None:
		model, alphabet =  load_model_without_regression(args.modelpath)
	else:
		model_info = esmdict[args.model]
		model, alphabet = pretrained.load_model_and_alphabet(model_info["name"])

	layers = len(model.layers)
	batch_converter = alphabet.get_batch_converter()

	if gpu:
		model = model.to('cuda')
	model.eval()
	
	with open(args.input,"r") as f:
		for line in f:
			cols = line.split("\t")
			sequence = cols[1].rstrip()
			seqid = cols[0]
			lenseq = len(sequence)
			if lenseq>chunk_size:
				nchunks = lenseq - chunk_size + 1
				subseqlist = [(f"{seqid}_{i}", sequence[j:j+chunk_size]) for i,j in enumerate(range(0, nchunks , args.window))]

				last_start = nchunks + (nchunks % args.window)
				if lenseq - last_start > 0:
					subseqlist.append((f"{seqid}_{len(subseqlist)}", sequence[-chunk_size:]))
				longseqs.append((seqid,subseqlist))
				len_longseqs.append(lenseq)
			else:
				shortseqs.append((seqid,sequence))
				len_shortseqs.append(lenseq)

	print("Sequences loaded\n")

	len_allseqs = len_shortseqs + len_longseqs

	nseqs_by_type = [len(x) for x in (len_shortseqs,len_longseqs)]

	prot_len_log = torch.log(torch.tensor(len_allseqs))
	len_min = torch.min(prot_len_log)

	prot_len_log_scaled = (2*(prot_len_log - len_min) / (torch.max(prot_len_log) - len_min) - 1).unsqueeze(1)

	len_embedding_short = prot_len_log_scaled[:nseqs_by_type[0]]
	len_embedding_long = prot_len_log_scaled[nseqs_by_type[0]:]

	with torch.inference_mode():
		if args.output_batches > 1:
			shortseqs_chunk, longseqs_chunk, len_emb_short_chunk, len_emb_long_chunk = [
				chunk_list(lst, args.output_batches) for lst in (shortseqs, longseqs, len_embedding_short, len_embedding_long)
			]

			tempfiles = [f"tmp_{k}{args.outfile}" for k in range(args.output_batches)]

			for k in range(args.output_batches):
				run_inference(shortseqs_chunk[k], longseqs_chunk[k], len_emb_short_chunk[k], len_emb_long_chunk[k], tempfiles[k])
				print(f"Processed batch {k} and to temporary file {tempfiles[k]}")
			concatenate_pt_files(tempfiles, args.outfile)
		else:
			run_inference(shortseqs, longseqs, len_shortseqs, len_longseqs, args.outfile)
			
	print("Embedded all sequences sucessfully")

	
		