import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree
import argparse as ap
from pathlib import Path
import json

if __name__ == "__main__":

	parser = ap.ArgumentParser(
        description="In Silico evolution of proteins towards structure",
        formatter_class=ap.ArgumentDefaultsHelpFormatter
	)

	parser.add_argument(
		"--node_embeddings", "-n", 
		type=str, 
		help="Node embeddings file (pytorch file)", 
		metavar="<path/filename>",
		required=True
	)

	parser.add_argument(
		"--edges", "-e",
		type=str,
		help="Edges file (tab separated file)",
		metavar="<path/filename>",
		required=True
	)

	parser.add_argument(
		"--output", "-o",
		type=str,
		help="Output file (pytorch file)",
		metavar="<path/filename>",
		default=None
	)

	parser.add_argument(
		"--expand_embedding_with_degree",
		help="add scaled degree to node embeddings",
		action="store_true"
	)

	args = parser.parse_args()

	if args.output is None:
		args.output = f"{Path(args.edges).with_suffix('')}_PyG.pt"
		print(f"Output file not specified. Using {args.output} as output file.")

	print("Parsed arguments\n===================")
	for arg, value in vars(args).items():
		print(f"{arg}: {value}")

	# Load node embeddings
	
	node_data = torch.load(args.node_embeddings, weights_only=False)
	node_names = node_data['seqids']
	node_embeddings = node_data['representations']
	if not isinstance(node_embeddings, torch.Tensor):
		raise ValueError("Node embeddings should be a PyTorch tensor.") 
	if node_embeddings.dim() != 2:
		raise ValueError("Node embeddings should be a 2D tensor.")
	node_embed_dimension = node_embeddings.size(1)
	node_name_to_index = {name: i for i, name in enumerate(node_names)} 

	# Load edges
	source_nodes = []
	target_nodes = []
	edge_weights = []

with open(args.edges, 'r') as f:
	for line in f:
		line = line.strip()
		if not line or line.startswith('#'):
			continue
		parts = line.split('\t')
		if len(parts) < 2:
			raise ValueError(f"Invalid edge format: {line}")
		if len(parts) > 2:
			try:
				edge_weight = float(parts[2])/1000
			except ValueError:
				raise ValueError(f"Invalid edge weight: {parts[2]}")
		else:
			edge_weight = 1.0
		if parts[0] not in node_name_to_index or parts[1] not in node_name_to_index:
			raise ValueError(f"Node names {parts[0]} or {parts[2]} not found in node embeddings.")
		source_nodes.append(parts[0])
		target_nodes.append(parts[1])
		edge_weights.append(edge_weight)

	# Convert to PyTorch tensors
	source_indices = [node_name_to_index[name] for name in source_nodes]
	target_indices = [node_name_to_index[name] for name in target_nodes]
	edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
	edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
	node_degree = degree(torch.cat([edge_index[0],edge_index[1]], dim=0), num_nodes = node_embeddings.size(0))

	if args.expand_embedding_with_degree:
		log_degree = torch.log10(node_degree + 1e-6).unsqueeze(1)
		node_embeddings = torch.cat([node_embeddings, log_degree], dim=1)
	
	# Create a PyTorch Geometric Data object
	data = Data(
		x=node_embeddings,
		edge_index=edge_index,
		edge_attr=edge_attr,
		node_degree=node_degree
	)

	# Save the data object
	torch.save(data, args.output)
	print(f"Graph data saved to {args.output}")

	# Save node labels
	node_list_file = args.output.replace('_PyG.pt','_node_list.json')
	with open(node_list_file, 'w') as f:
		json.dump(node_name_to_index, f)

	print(f"Node labels saved to {node_list_file}")