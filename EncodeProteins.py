import torch
from esm import Alphabet, pretrained
import sys
import argparse as ap
from pathlib import Path

esmdict = {
		"15B": {"name": "esm2_t48_15B_UR50D"},
		"3B": {"name" : "esm2_t36_3B_UR50D"},
		"650M":{"name" : "esm2_t33_650M_UR50D"}
}

extra_toks_per_seq = 1

shortseqs = []
longseqs = []

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

	def load_model_without_regression(model_location):
		"""Load from local path. The regression weights need to be co-located"""
		model_location = Path(model_location)
		model_data = torch.load(str(model_location), map_location="cpu", weights_only = False)
		model_name = model_location.stem
		regression_data = None
		return pretrained.load_model_and_alphabet_core(model_name, model_data, regression_data)

	gpu = torch.cuda.is_available()
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

	def get_batch_idx(seqlist):
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
		batches = get_batch_idx(seqlist)
		return torch.utils.data.DataLoader(
			seqlist, collate_fn= batch_converter, batch_sampler=batches
		)

	def getreps(seqlist):
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
		all_seqids = []
		all_reps = []
		for seqid, seqchunklist in longseqlist:
			_, chunkreps = getreps(seqchunklist)
			fullrep = torch.stack(chunkreps).mean(0)
			all_reps.append(fullrep)
			all_seqids.append(seqid)
		return all_seqids, all_reps

	all_lengths = []
	with open(args.input,"r") as f:
		for i in f.readlines():
			y = i.split("\t")
			sequence = y[1].rstrip()
			seqid = y[0]
			lenseq = len(sequence)
			all_lengths.append(lenseq)
			if lenseq>chunk_size:
				nchunks = lenseq - chunk_size + 1
				subseqlist = [(f"{seqid}_{i}", sequence[j:j+chunk_size]) for i,j in enumerate(range(0, nchunks , args.window))]

				last_start = nchunks + (nchunks % args.window)
				if lenseq - last_start > 0:
					subseqlist.append((f"{seqid}_{len(subseqlist)}", sequence[-chunk_size:]))
				longseqs.append((seqid,subseqlist))

	with torch.no_grad():
		short_seqids, short_seqreps = getreps(shortseqs)
		long_seqids, long_seqreps = getreps_for_longseq(longseqs)
	all_seqids = short_seqids + long_seqids
	all_seqreps = torch.cat([short_seqreps, long_seqreps], dim=0) 

	prot_len_log = torch.log(torch.tensor(all_lengths))
	len_min = torch.min(prot_len_log)

	prot_len_log_scaled = (2*(prot_len_log - len_min) / (torch.max(prot_len_log) - len_min) - 1).unsqueeze(1)

	all_seqreps = torch.cat([all_seqreps, prot_len_log_scaled], dim=1)

	torch.save({"seqids": all_seqids, "representations": all_seqreps}, args.outfile)
