import argparse as ap
from pathlib import Path
import sys
import torch
import json
import TrainUtils as utils

if __name__ == "__main__":

	parser = ap.ArgumentParser(description="GraphSAGE model for edge detection")

	parser.add_argument("--input", "-i",
		type=str,
		help="Path to the test positive graph (.pt).",
		metavar="<path/file>"
	)
	parser.add_argument("--processed", "-p",
		type=str,
		help="Path to the processed input data (.pt)"
	)
	parser.add_argument("--model","-m",
		type=str,
		help="Path to the model file (.pt)"
	)
	parser.add_argument("--batch_size","-b",
		type=int,
		help="Batch size for model inference",
		default=262144
	)
	parser.add_argument("--head","-h",
		type=str,
		choices=["gnn","nod","both"],
		help="Perform inference using the prediction head(s): gnn = Graph Neural Network, nod = Node Only Decoder, both = both heads",
		default="both"
	)
	parser.add_argument("--threads", "-t",
		type=int,
		help="Number of CPU threads to use",
		default=1
	)

	args = parser.parse_args()

	gpu_yes = torch.cuda.is_available()