import argparse as ap
from pathlib import Path
from collections import defaultdict
import sys
from time import time, strftime, gmtime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import subgraph, degree
from torch_scatter import scatter_max


class Pool_SAGEConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		
		# Linear fully connected pooling
		self.pool = nn.Linear(in_channels, in_channels)

		# Final linear layer after aggregation
		self.final_lin = nn.Linear(in_channels * 2, out_channels)
	
	def forward(self, x, edge_index, edge_weight):
		src, dst = edge_index
		
		# Pool and activate neighbor messages
		pooled = self.pool(x[src] * edge_weight.unsqueeze(-1))
		pooled = F.relu(pooled)
		
		# Aggregate neighbor messages via max
		aggregate, _ = scatter_max(pooled, dst, dim=0, dim_size=x.size(0))
		
		# Concatenate self-representation and aggregated neighbors
		h = torch.cat([x, aggregate], dim=-1)
		
		# Final transformation
		out = self.final_lin(h)
		return F.relu(out)

class GraphSAGE(nn.Module):
	def __init__(self, in_channels, hidden_channels):
		super().__init__()
		self.conv1 = Pool_SAGEConv(in_channels, hidden_channels)
		self.conv2 = Pool_SAGEConv(hidden_channels, hidden_channels)

		# Edge prediction head
		self.edge_pred = nn.Linear(hidden_channels * 2, 1)
		self.edge_weight_pred = nn.Linear(hidden_channels * 2, 1)
	
	def forward(self, x, message_edges, supervision_edges, message_edgewt = None):
		x = self.conv1(x, message_edges, message_edgewt)
		x = F.relu(x)
		x = self.conv2(x, message_edges, message_edgewt)

		edge_embeddings = torch.cat([x[supervision_edges[0]], x[supervision_edges[1]]], dim=-1)

		edge_weights = self.edge_weight_pred(edge_embeddings)
		edge_predictor = self.edge_pred(edge_embeddings)
		
		return edge_weights, edge_predictor

def node_to_edge_map(edge_index):
	node_to_edges = defaultdict(list)
	src, dst = edge_index
	for i in range(edge_index.size(1)):
		u, v = src[i].item(), dst[i].item()
		node_to_edges[u].append(i)
		node_to_edges[v].append(i) # undirected graph
	return node_to_edges

def split_data(graph, train_to_val_ratio = 7, node_centrality=None, max_attempts = 30, pure_val_edge_fraction = 0.3):
	# Not generating test set #
	# Define test set separatly #
	val_fraction = 1/(1 + train_to_val_ratio)
	train_fraction = 1 - val_fraction
	all_nodes = torch.arange(graph.num_nodes)
	val_mask = torch.zeros_like(all_nodes, dtype=torch.bool)
	

	src, dst = graph.edge_index
	num_nodes = graph.x.size(0)
	num_edges = graph.edge_index.size(1) 

	# Compute centrality if not provided
	if node_centrality is None:
		node_centrality = degree(torch.cat([src, dst], dim=0), num_nodes=num_nodes)
	
	average_centrality = node_centrality.mean(0)
	n_edges_val = int(val_fraction * num_edges)
	edge_val_upperlimit = int(n_edges_val * 1.05)

	approx_nodes_val = n_edges_val/average_centrality

	for _ in range(max_attempts):
		node_idx_val = torch.randperm(num_nodes)[:approx_nodes_val]
		mask_val_pure = torch.isin(src, node_idx_val) & torch.isin(dst, node_idx_val)
		mask_val_any = (torch.isin(src, node_idx_val) | torch.isin(dst, node_idx_val))
		mask_val_xor = mask_val_any & ~mask_val_pure
		semi_val_edges = graph.edge_index[:, mask_val_xor]
		pure_val_edges = graph.edge_index[:, mask_val_pure]
		total_val_edges = graph.edge_index[:, mask_val_any]
		num_any_val = mask_val_any.sum().item()
		num_pure_val = mask_val_pure.sum().item()
		desired_pure_val = int(num_any_val*pure_val_edge_fraction)
		desired_pure_val_upperbound = int(desired_pure_val * 1.05)
		
		if desired_pure_val <= num_pure_val <= desired_pure_val_upperbound and n_edges_val <= num_any_val <= edge_val_upperlimit:
			break
	else:
		print("Warning: could not match edge count for validation set closely.")
	# val_mask[val_nodes] = True
	# train_nodes = all_nodes[~val_mask]

	# Calculate split indices
	node_idx_train = 

	
	
	log_node_centrality = torch.log10(node_centrality)
	bins = torch.tensor(range(0,6),dtype=torch.float32)
	# Or a more natural binning #
	# min = log_node_centrality.min(0)
	# max = log_node_centrality.max(0)
	# nbins = 10
	# interval = (max-min)/nbins
	# bins = torch.arange(range(min,max,interval))
	log_nc_bin_index = torch.bucketize(log_node_centrality, bins, right=True)


	# Compute edge centrality based on node centrality
	edge_centrality = node_centrality[src] + node_centrality[dst]
	
	# Sort edges by centrality
	sorted_index = torch.argsort(edge_centrality)
	sorted_edges = graph.edge_index[:, sorted_index]
	sorted_edgewt = graph.edge_attr[sorted_index]

	# Split edges into train, validation, and test sets
	train_edges = sorted_edges[:, :indices_train]
	val_edges = sorted_edges[:, indices_train:indices_val]
	test_edges = sorted_edges[:, indices_val:]

	# Split edge weights into train, validation, and test sets
	train_edgewt = sorted_edgewt[:indices_train]
	val_edgewt = sorted_edgewt[indices_train:indices_val]
	test_edgewt = sorted_edgewt[indices_val:]

	# Get the unique nodes involved in each set of edges
	train_nodes = torch.unique(torch.flatten(train_edges))
	val_nodes = torch.unique(torch.flatten(val_edges))
	test_nodes = torch.unique(torch.flatten(test_edges))

	# Extract node embeddings for the respective sets
	train_node_embed = graph.x[train_nodes]
	val_node_embed = graph.x[val_nodes]
	test_node_embed = graph.x[test_nodes]

	# Create separate Data objects for train, validation, and test sets
	train_graph = Data(
		x=train_node_embed,
		edge_index=train_edges,
		edge_attr=train_edgewt
	)

	val_graph = Data(
		x=val_node_embed,
		edge_index=val_edges,
		edge_attr=val_edgewt
	)

	test_graph = Data(
		x=test_node_embed,
		edge_index=test_edges,
		edge_attr=test_edgewt
	)

	return train_graph, val_graph, test_graph


def key_edges(node1, node2, total_nodes):
	src_, dst_ = torch.min(node1, node2), torch.max(node1, node2)
	return src_ * total_nodes + dst_

# Generate negative edges 
def generate_negative_edges(positive_graph, negative_positive_ratio = 2, device=None):
	if device is None:
		device = positive_graph.device
	num_nodes = positive_graph.x.size(0)
	num_negative_edges = positive_graph.edge_index.size(1) * negative_positive_ratio
	# Compute positive edge keys
	edge_keys = key_edges(positive_graph.edge_index[0], positive_graph.edge_index[1], num_nodes)

	# Big oversample (to make sure we have enough negatives)
	batch_size = int(num_negative_edges * 1.5)  # Oversample by 50%
	
	# Sample a big batch at once
	src = torch.randint(0, num_nodes, (batch_size,), device=device)
	dst = torch.randint(0, num_nodes, (batch_size,), device=device)
	
	# Compute keys for sampled edges
	sample_keys = key_edges(src, dst, num_nodes)
	
	# Filter: remove edges that already exist
	mask = ~torch.isin(sample_keys, edge_keys)
	
	# Keep only the valid negative edges
	valid_src = src[mask]
	valid_dst = dst[mask]
	
	# Make sure we have enough negatives
	if valid_src.size(0) < num_negative_edges:
		# If not enough, sample again recursively (rare)
		more_negatives = generate_negative_edges(
			num_negative_edges - valid_src.size(0), num_nodes, edge_index, device)
		
		# Combine old and new negatives
		valid_edges = torch.cat([
			torch.stack([valid_src, valid_dst], dim=0),
			more_negatives
		], dim=1)
	else:
		# Otherwise just slice
		valid_edges = torch.stack([valid_src[:num_negative_edges], valid_dst[:num_negative_edges]], dim=0)

	return valid_edges

def ratio_type(string):
	vals = list(map(float, string.split(',')))
	if len(vals) != 2:
		raise ap.ArgumentTypeError("Expected two numbers separated by a comma (e.g., 0.8,0.2)")
	if abs(sum(vals) - 1.0) > 1e-6:
		raise ap.ArgumentTypeError(f"Train/val ratios must sum to 1, got {sum(vals)}")
	return vals

def mask_edges_random(probability_mask, edge_list, centrality=None):
	if centrality is None or centrality.max() == centrality.min():
		normalized_scores = torch.ones(edge_list.size(1))
	else:
		normalized_scores = (centrality - centrality.min()) / (centrality.max() - centrality.min())
	bernoulli_mask = torch.bernoulli(normalized_scores * probability_mask).bool()
	masked_edges = edge_list[:, bernoulli_mask]
	return bernoulli_mask, masked_edges

def mask_batch_edges(batch, orig_edge_list):
	mask = torch.all(torch.isin(orig_edge_list, batch.n_id), dim=0)
	return mask #, batch_edge

if __name__ == "__main__":

	parser = ap.ArgumentParser(description="GraphSAGE model for edge detection")

	parser.add_argument("--input", "-i",
		type=str, required=True,
		help="Path to the input graph data file (.pt)",
		metavar="<path/file>"
	)
	parser.add_argument("--output", "-o",
		type=str, default=None,
		help="Path to the trained GNN (.pt)",
		metavar="<path/file>"
	)
	parser.add_argument("--threads", "-t",
		type=int,
		help="Number of CPU threads to use",
		default=1
	)
	parser.add_argument("--split","--sr",
		type=ratio_type,
		default=[0.85, 0.1],
		help= "Train/validation data split ratio as two numbers separated by comma, e.g., 0.8, 0.2"
	)
	parser.add_argument("--epochs", "-e",
		type=int,
		help="Number of training epochs",
		default=50
	)
	parser.add_argument("--batch-size", "-b",
		type=int,
		help="Minibatch size for training",
		default=0
	)
	parser.add_argument("--learning-rate", "--lr",
		type=float,
		help="Learning rate for the optimizer",
		default=0.001
	)
	parser.add_argument("--weight-decay", "--wd",
		type=float,
		help="Weight decay for the optimizer",
		default=1e-4
	)
	parser.add_argument("--dropout", "-d",
		type=float,
		help="Dropout rate for the model",
		default=0.5
	)
	parser.add_argument("--hidden-channels", "--hc",
		type=float, help="Fraction of number of input channels to use as the number of hidden channels",
		default=0.05
	)

	args = parser.parse_args()

	if len(sys.argv) == 1:
		print("Error: essential arguments not provided.")
		parser.print_help() # Print the help message
		sys.exit(1)

	# Load the graph data
	ingraph = torch.load(args.input)

	# Torch settings
	torch.set_num_threads(args.threads)
	torch.set_num_interop_threads(args.threads)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	# Create the model
	num_hidden_channels = int(ingraph.num_features * args.hidden_channels)
	model = GraphSAGE(
		in_channels = ingraph.num_features,
		hidden_channels = num_hidden_channels,
		out_channels = 2
	).to(device)

	# Define the loss functions

	bce_loss = nn.BCEWithLogitsLoss()
	mse_loss = nn.MSELoss()

	optimizer = torch.optim.Adam(
		model.parameters(),
		lr=args.learning_rate,
		weight_decay=args.weight_decay
	)

	# Split data into training and validation set
	train_graph, val_graph, test_graph = split_data(ingraph)

	negative_edgeset = generate_negative_edges(ingraph,2)

	# Create data batches #
	sampler = EdgeCentralitySampler(train_graph.edge_index, centrality, batch_size=8192)  # 8192 positive edges
	batch_loader = torch.utils.data.DataLoader(sampler, batch_size=None)

	# Begin training
	start_time = time()
	model.train()
	total_loss = 0.0

	for _ in args.epochs:
		for minibatch in batch_loader:
			# Mask edges randomly for positive edges
			positive_message_edges, _ = mask_edges_random(0.7, minibatch.edge_index, centrality=None)

			# Generate negative edges (ensure they don't overlap with existing edges)
			negative_edges = generate_negative_edges(minibatch, device=None)

			# Mask edges randomly for negative edges
			negative_message_edges, _ = mask_edges_random(0.7, minibatch.edge_index, centrality=None)

			# Concatenate positive and negative edges
			minibatch.edge_index = torch.cat([minibatch.edge_index, negative_edges], dim=-1)  # Now 16,384 edges
			minibatch.edge_attr = torch.cat([minibatch.edge_attr, torch.zeros(negative_edges.size(0))], dim=-1)  # Zero edge weights for negative edges

			# Concatenate positive and negative message edges
			all_message_edges = torch.cat([positive_message_edges, negative_message_edges], dim=-1)

			# Sample neighbors for the minibatch
			nbr_sample_batches = NeighborLoader(
				minibatch,
				num_neighbors=[30, 20],
				batch_size=64,
			)
			
			for batch in nbr_sample_batches:
				# Mask the message edges for the batch
				mask_message_edges = mask_batch_edges(batch, all_message_edges)
				mask_negative_edges = mask_batch_edges(batch, negative_edges)
				
				# Forward pass through the model (separate positive and negative edges)
				edge_probability, edge_weight_pred = model(
					batch.x, 
					batch.edge_index[mask_message_edges], 
					batch.edge_index[~mask_message_edges]
				)
				
				# Create labels for the edges
				labels = torch.ones(edge_probability.size(0))  # Positive edges have label 1
				labels[mask_negative_edges] = 0  # Negative edges have label 0
				
				# Compute BCE and MSE losses
				loss = bce_loss(edge_probability, labels) + mse_loss(edge_weight_pred, batch.edge_attr[~mask_message_edges])
				
				# Accumulate the total loss
				total_loss += loss.item()

			# Zero the gradients, perform backpropagation, and optimize
			optimizer.zero_grad()
			total_loss.backward()
			optimizer.step()

	
	elapsed_time = time() - start_time

	# Save the model
	if args.output is None:
		args.output = f"{Path(args.input).with_suffix('')}_GNN_trained.pt"
		print(f"Output file not specified. Using {args.output} as output file.")
	torch.save(model.state_dict(), args.output)

	# Save model metadata
	metadata_file = args.output.replace('.pt', '_metadata.txt')
	with open(metadata_file, 'w') as f:
		f.write(f"Input file: {args.input}\n")
		f.write(f"Output file: {args.output}\n")
		f.write(f"Epochs: {args.epochs}\n")
		f.write(f"Batch size: {args.batch_size}\n")
		f.write(f"Learning rate: {args.learning_rate}\n")
		f.write(f"Weight decay: {args.weight_decay}\n")
		f.write(f"Dropout rate: {args.dropout}\n")
		f.write(f"Input channels: {ingraph.num_features}\n")
		f.write(f"Hidden channels: {num_hidden_channels}\n")
		f.write(f"Training time:{strftime("%Hh%Mm%Ss", gmtime(elapsed_time))}")
