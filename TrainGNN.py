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
from glob import glob

gpu_yes = torch.cuda.is_available()

class Pool_SAGEConv(nn.Module):
	"""
	Implements a pooling-based GraphSAGE convolution layer for aggregating neighbor information.

	Args:
		in_channels (int): Number of input features per node.
		out_channels (int): Number of output features per node.

	Methods:
		forward(x, edge_index, edge_weight):
			Performs the forward pass of the convolution layer.
			Args:
				x (torch.Tensor): Node features.
				edge_index (torch.Tensor): Edge indices.
				edge_weight (torch.Tensor): Edge weights.
			Returns:
				torch.Tensor: Output node features after aggregation.
	"""

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
	"""
	Defines the GraphSAGE model with edge prediction and edge weight prediction capabilities.

	Args:
		in_channels (int): Number of input features per node.
		hidden_channels (int): Number of hidden features per node.

	Methods:
		forward(x, message_edges, supervision_edges, message_edgewt):
			Performs the forward pass of the GraphSAGE model.
			Args:
				x (torch.Tensor): Node features.
				message_edges (torch.Tensor): Edges used for message passing.
				supervision_edges (torch.Tensor): Edges used for supervision.
				message_edgewt (torch.Tensor): Edge weights for message passing.
			Returns:
				torch.Tensor: Predicted edge weights and edge existence probabilities.
	"""

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
	
class EdgeSampler(torch.utils.data.IterableDataset):
	"""
	Samples minibatches of edges based on centrality or uniform probability

	Args:
		edges (torch.Tensor): Edge index tensor (2 x num_edges).
		node_embeddings (torch.Tensor, optional): Node feature embeddings. Defaults to None.
		edge_attr (torch.Tensor, optional): Edge attributes. Defaults to None.
		batch_size (int, optional): Number of edges to sample per minibatch. Defaults to 1000.
		num_batches (int, optional): Number of batches to create. Defaults to None.
		centrality (torch.Tensor, optional): Centrality scores for nodes. Defaults to None.

	Methods:
		__iter__():
			Generates minibatches of edges and their corresponding subgraphs.
			Returns:
				torch_geometric.data.Data or torch.Tensor: Minibatch data object or edge index tensor.
	"""
	def __init__(self, edges, node_embeddings=None, edge_attr=None, batch_size = 1000, num_batches = None, centrality = None, centrality_fraction = None):
		super().__init__()
		self.edges = edges  # full_data should be a PyG Data object
		self.num_batches = num_batches
		self.edge_attr = edge_attr
		self.node_embeddings = node_embeddings
		self.batch_size = batch_size
		self.centrality_fraction = centrality_fraction

		# Compute edge probabilities for sampling
		if centrality is None:
			self.edge_probs = torch.ones(edges.size(0))
		else:
			src, dst = self.edges
			self.edge_probs = centrality[src] + centrality[dst]

		# Define sampling method
		if self.centrality_fraction is None or self.centrality_fraction == 1 or centrality is None:
			self.sampling_fn = self.sample_edges_basic
		else:
			self.sampling_fn = self.sample_edges_strata

		# Determine the output batch creation method
		self.create_output_batch = self.return_edges_only if self.node_embeddings is None else self.return_PyG_Data

	def sample_edges_basic(self):
			return torch.multinomial(self.edge_probs, self.batch_size, replacement=False)

	def sample_edges_strata(self):
		num_edges = self.edges.size(1)
		mask = torch.ones(num_edges, dtype=torch.bool)

		centrality_batch_size = int(self.batch_size * self.centrality_fraction) # type: ignore
		uniform_batch_size = self.batch_size - centrality_batch_size

		# Centrality-based sampling
		centrality_sampled = torch.multinomial(self.edge_probs, centrality_batch_size, replacement=False)
		mask[centrality_sampled] = False

		# Uniform sampling from the rest
		uniform_pool = torch.where(mask)[0]
		uniform_probs = torch.ones(uniform_pool.size(0))
		uniform_sampled = uniform_pool[torch.multinomial(uniform_probs, uniform_batch_size, replacement=False)]

		return torch.cat([centrality_sampled, uniform_sampled])
	
	def return_edges_only(self):
		"""Sample edges based on edge probabilities."""
		sampled_edge_idx = self.sampling_fn()
		return self.edges[:, sampled_edge_idx]
	
	def return_PyG_Data(self):
			"""Create a PyG Data object for the sampled edges."""
			batch_edges = self.sample_edges_only()
			nodes_in_batch = batch_edges.flatten().unique()

			# Subgraph will relabel nodes automatically
			sub_edge_index, edge_mask = subgraph(
				nodes_in_batch,
				self.full_data.edge_index,
				relabel_nodes=True,
				return_edge_mask=True
			)
			# Subset node features
			sub_x = self.node_embeddings[nodes_in_batch,:] # type: ignore
			
			# Create a PyG Data object
			batch = Data(
				x=sub_x,
				edge_index=sub_edge_index,
				n_id = nodes_in_batch
			)

			# Subset edge attributes if available
			if self.edge_attr is not None:
				batch.edge_attr=self.edge_attr[edge_mask]

			return batch
	
	def __iter__(self):
		n = 0
		while self.num_batches is None or n < self.num_batches:
			n += 1
			batch = self.create_output_batch()
			yield batch


def node_to_edge_map(edge_index):
	node_to_edges = defaultdict(list)
	src, dst = edge_index
	for i in range(edge_index.size(1)):
		u, v = src[i].item(), dst[i].item()
		node_to_edges[u].append(i)
		node_to_edges[v].append(i) # undirected graph
	return node_to_edges

def subgraph_with_relabel(original_graph, edge_mask):
	num_nodes = original_graph.x.size(0)
	selected_edges = original_graph.edge_index[:, edge_mask] 
	selected_nodes = torch.unique(selected_edges)
	mapping = -torch.ones(num_nodes, dtype=torch.long)  # default to -1
	mapping[selected_nodes] = torch.arange(selected_nodes.size(0))

	# 2. Remap edge indices using vectorized lookup
	remapped_edge_index = mapping[selected_edges]
	outgraph = Data(
		x = original_graph.x[selected_nodes,:],
		edge_index = remapped_edge_index,
		edge_attr = original_graph.edge_attr[edge_mask],
		old_node_ids = selected_nodes
	)
	return outgraph

def bisect_data(graph, second_edge_fraction=0.3, node_centrality=None, max_attempts=50, second_edge_fraction_pure=0.09):
	"""
	Split a graph into two subgraphs based on edge centrality and node sampling.

	Args:
		graph (torch_geometric.data.Data): Input graph.
		second_edge_fraction (float): Fraction of edges to include in the second graph.
		node_centrality (torch.Tensor): Node centrality scores (optional).
		max_attempts (int): Maximum attempts to match edge counts.
		second_edge_fraction_pure (float): Fraction of pure edges in the second graph.

	Returns:
		first_graph (torch_geometric.data.Data): First subgraph.
		second_graph (torch_geometric.data.Data): Second subgraph.
	"""
	src, dst = graph.edge_index
	num_nodes = graph.x.size(0)
	num_edges = graph.edge_index.size(1)

	# Compute centrality if not provided
	if node_centrality is None:
		node_centrality = degree(torch.cat([src, dst], dim=0), num_nodes=num_nodes)

	average_centrality = node_centrality.mean()
	n_edges_second = int(second_edge_fraction * num_edges)
	max_edges_second = int(n_edges_second * 1.05)
	n_desired_pure_second_edges = int(num_edges * second_edge_fraction_pure)
	max_desired_pure_second_edges = int(n_desired_pure_second_edges * 1.05)
	approx_nodes_second = int(n_edges_second / average_centrality)

	# Preallocate masks
	node_mask = torch.empty(num_nodes, dtype=torch.bool, device=src.device)
	mask_src = torch.empty_like(src, dtype=torch.bool)
	mask_dst = torch.empty_like(dst, dtype=torch.bool)
	mask_second_pure = torch.empty_like(src, dtype=torch.bool)
	mask_second = torch.empty_like(src, dtype=torch.bool)

	num_pure_second_edges = 0
	num_any_second_edges = 0

	for _ in range(max_attempts):
		# Reset node_mask
		node_mask.fill_(False)

		# Sample nodes and set mask
		node_idx_second = torch.randperm(num_nodes, device=src.device)[:approx_nodes_second]
		node_mask[node_idx_second] = True

		# Compute edge masks
		torch.index_select(node_mask, 0, src, out=mask_src)
		torch.index_select(node_mask, 0, dst, out=mask_dst)

		# mask_second_pure = mask_src & mask_dst
		torch.logical_and(mask_src, mask_dst, out=mask_second_pure)
		# mask_second = mask_src | mask_dst
		torch.logical_or(mask_src, mask_dst, out=mask_second)

		# Count edges
		num_any_second_edges = mask_second.sum().item()
		num_pure_second_edges = mask_second_pure.sum().item()

		# Check constraints
		if (n_desired_pure_second_edges <= num_pure_second_edges <= max_desired_pure_second_edges and
			n_edges_second <= num_any_second_edges <= max_edges_second):
			break
	else:
		print(f"Warning: Could not match edge count for second set closely.\n"
			f"Desired: Any={n_edges_second}, Pure={n_desired_pure_second_edges}; "
			f"Actual: Any={num_any_second_edges}, Pure={num_pure_second_edges}")

	mask_first = ~mask_second

	first_graph = subgraph_with_relabel(graph, mask_first)
	second_graph = subgraph_with_relabel(graph, mask_second)

	return first_graph, second_graph


def key_edges(node1, node2, total_nodes):
	src_, dst_ = torch.min(node1, node2), torch.max(node1, node2)
	return src_ * total_nodes + dst_

# Generate negative edges 
def generate_negative_edges(positive_graph, negative_positive_ratio=2, device=None):
	"""
	Generate negative edges for a graph.

	Args:
		positive_graph (torch_geometric.data.Data): Input graph with positive edges.
		negative_positive_ratio (int): Ratio of negative edges to positive edges.
		device (torch.device): Device to perform computations on.

	Returns:
		torch.Tensor: Negative edges (2 x num_negative_edges).
	"""
	if device is None:
		device = positive_graph.device

	num_nodes = positive_graph.x.size(0)
	num_positive_edges = positive_graph.edge_index.size(1)
	num_negative_edges = num_positive_edges * negative_positive_ratio

	# Compute positive edge keys
	edge_keys = key_edges(positive_graph.edge_index[0], positive_graph.edge_index[1], num_nodes)

	# Initialize negative edges
	valid_src = torch.empty(0, device=device, dtype=torch.long)
	valid_dst = torch.empty(0, device=device, dtype=torch.long)

	# Loop until enough negative edges are generated
	while valid_src.size(0) < num_negative_edges:
		# Oversample to ensure enough valid negatives
		batch_size = int((num_negative_edges - valid_src.size(0)) * 1.5)  # Oversample by 50%
		src = torch.randint(0, num_nodes, (batch_size,), device=device)
		dst = torch.randint(0, num_nodes, (batch_size,), device=device)

		# Compute keys for sampled edges
		sample_keys = key_edges(src, dst, num_nodes)

		# Filter: remove edges that already exist
		mask = ~torch.isin(sample_keys, edge_keys)

		# Append valid negative edges
		valid_src = torch.cat([valid_src, src[mask]])
		valid_dst = torch.cat([valid_dst, dst[mask]])

	# Slice to the required number of negative edges
	valid_src = valid_src[:num_negative_edges]
	valid_dst = valid_dst[:num_negative_edges]

	# Return negative edges
	return torch.stack([valid_src, valid_dst], dim=0)

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

bce_loss = nn.BCEWithLogitsLoss()
mse_loss = nn.MSELoss()

def train_and_evaluate(
		train_graph,
		val_graph,
		optimizer,
		learning_rate,
		hidden_channels,
		dropout,
		message_edge_fraction,
		max_epochs=50,
		patience=5,
		device='cuda' if gpu_yes else 'cpu'
):
	model = GraphSAGE(
		in_channels = train_graph.x.size(1),
		hidden_channels = num_hidden_channels,
		out_channels = 2
	).to(device)

	

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
	parser.add_argument("--train_fraction",
		type=float,
		default= 0.8,
		help= "Train fraction of data to train (rest for validation)"
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
		default=0.1
	)

	args = parser.parse_args()

	if len(sys.argv) == 1:
		print("Error: essential arguments not provided.")
		parser.print_help() # Print the help message
		sys.exit(1)


	# Torch settings
	torch.set_num_threads(args.threads)
	torch.set_num_interop_threads(args.threads)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	# Load input graphs
	input_graphs_filenames = []
	for pattern in args.input.split(","):
		input_graphs_filenames.extend(glob(pattern.strip()))
	
	ingraphs = []
	for files in input_graphs_filenames:
		try:
			ingraphs.append(torch.load(files, weights_only=False))
		except Exception as e:
			print(f"Error loading file {files}: {e}")
			continue

	data_for_training = []
	for graph in ingraphs:
		try:
			graph.node_degree
		except NameError:
			graph.node_degree = degree(torch.cat([graph.edge_index[0], graph.edge_index[1]], dim=0), num_nodes=graph.x.size(0))

		# Split graph into training and validation sets
		train, val = bisect_data(graph, second_edge_fraction=1 - args.train_fraction)

		# Generate negative edges
		negative_edges = generate_negative_edges(graph, negative_positive_ratio=2) # should be a Data type same as positive edges TODO #

		data_sizes = [x.size(0) for x in (train.edge_index,negative_edges)]

		# Create minibatch sampler for positive edges
		positive_sampler = EdgeSampler(train, batch_size=8192, centrality=train.node_degree)  # Positive edges based on centrality
		positive_batch_loader = torch.utils.data.DataLoader(positive_sampler, batch_size=None)

		# Create minibatch sampler for negative edges
		negative_sampler = EdgeSampler(train, batch_size=8192, centrality=None)  # Negative edges sampled randomly
		negative_batch_loader = torch.utils.data.DataLoader(negative_sampler, batch_size=None)

		# Store data in a structured format
		data_for_training.append({
			"train_graph": train,
			"val_graph": val,
			"negative_edges": negative_edges,
			"positive_batch_loader": positive_batch_loader,
			"negative_batch_loader": negative_batch_loader
		})
	

	# Begin training
	start_time = time()
	total_loss = 0.0

	# Define hyperparameter	setup

	for _ in args.epochs:
		for data in data_for_training:
			positive_batch_loader = data["positive_batch_loader"]
			negative_batch_loader = data["positive_batch_loader"]
			for positive_batch, negative_batch in zip(positive_batch_loader, negative_batch_loader):
				# Mask edges randomly for positive edges
				positive_message_edges, _ = mask_edges_random(0.7, positive_batch.edge_index, centrality=None)

				# Mask edges randomly for negative edges
				negative_message_edges, _ = mask_edges_random(0.7, negative_batch.edge_index, centrality=None)

				# Concatenate positive and negative edges TODO fix this
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
			
			# Validation #
			model.eval()
			with torch.inference_mode():
				val_message_edges, _ = mask_edges_random(0.7, val_graph.edge_index, centrality=None)
				val_negative_edges = generate_negative_edges(val_graph, device=None)
				val_message_edges = torch.cat([val_message_edges, val_negative_edges], dim=-1)

				# Sample neighbors for the validation set
				nbr_sample_batches_val = NeighborLoader(
					val_graph,
					num_neighbors=[30, 20],
					batch_size=64,
				)

				val_loss = 0.0
				for batch in nbr_sample_batches_val:
					mask_val_message_edges = mask_batch_edges(batch, val_message_edges)
					
					# Forward pass through the model
					edge_probability, edge_weight_pred = model(
						batch.x, 
						batch.edge_index[mask_val_message_edges], 
						batch.edge_index[~mask_val_message_edges]
					)
					
					# Create labels for the edges
					labels = torch.ones(edge_probability.size(0))
					labels[mask_val_message_edges] = 0
					# Compute BCE and MSE losses
					val_loss += bce_loss(edge_probability, labels) + mse_loss(edge_weight_pred, batch.edge_attr[~mask_val_message_edges])
				val_loss /= len(nbr_sample_batches_val)
				print(f"Epoch {_+1}/{args.epochs}, Training Loss: {total_loss:.4f}, Validation Loss: {val_loss:.4f}")
			total_loss = 0.0  # Reset total loss for the next epoch
			model.train()


	
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
