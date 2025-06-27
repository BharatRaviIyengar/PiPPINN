import torch
import torch.nn as nn
from torch.nn.functional import relu as ReLU
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import degree
from torch_geometric.utils.map import map_index
from torch_scatter import scatter_max
from warnings import warn
from pathlib import Path

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

	def __init__(self, in_channels, out_channels, message_passing=True):
		super().__init__()
		
		# Linear fully connected pooling
		self.pool = nn.Linear(in_channels, in_channels)

		# Final linear layer after aggregation
		self.final_lin = nn.Linear(in_channels * 2, out_channels)

		# Learnable parameter that controls how much the edge weight influences message aggregation
		self.edge_weight_message_coefficient = nn.Parameter(torch.tensor(0.5))
		
		self.forward = self.forward_with_message_pooling if message_passing else self.forward_without_message_pooling
			
	def forward_with_message_pooling(self, x, edge_index, edge_weight):
		src, dst = edge_index

		all_src = torch.cat([src, dst], dim=0)
		all_dst = torch.cat([dst, src], dim=0)
		mask = all_src != all_dst


		# Collect all neighbors
		all_neighbors = torch.cat([x[src], x[dst]], dim=0) 
		all_indices = torch.cat([dst, src], dim=0)
		all_edgewts = torch.cat([edge_weight, edge_weight], dim=0)

		# Remove self-embedding
		mask = all_indices != own_indices
		filtered_neighbors = all_neighbors[mask]
		filtered_indices = all_indices[mask]
		filtered_edgewts = all_edgewts[mask]

		edge_features = filtered_neighbors * (1 + self.edge_weight_message_coefficient * filtered_edgewts.unsqueeze(-1))

		# Pool and activate neighbor messages
		pooled = self.pool(edge_features)
		pooled = ReLU(pooled)
		
		# Aggregate neighbor messages via max
		aggregate, _ = scatter_max(pooled, filtered_indices, dim=0, dim_size=x.size(0))
		
		# Concatenate self-representation and aggregated neighbors
		h = torch.cat([x, aggregate], dim=-1)
		
		# Final transformation
		return ReLU(self.final_lin(h))
		
		
	def forward_without_message_pooling(self, x, edge_index, edge_weight):
		h = torch.cat([x, torch.zeros_like(x)], dim=-1)
		# Final transformation
		return ReLU(self.final_lin(h))


class GraphSAGE(nn.Module):
	"""
	Defines the GraphSAGE model with multiple forward logics for different use cases.
	"""

	def __init__(self, in_channels, hidden_channels, dropout=0.0, mode="message_passing"):
		"""
		Args:
			in_channels (int): Number of input features per node.
			hidden_channels (int): Number of hidden features per node.
			dropout (float): Dropout rate.
			mode (str): Mode of operation. Options are:
						- "message_passing": Use message edges for training/inference.
						- "no_message_passing": Skip message passing, use node embeddings only.
		"""
		super().__init__()
		
		self.dropout = nn.Dropout(p=dropout)
		self.in_channels = in_channels
		self.hidden_channels = hidden_channels

		# Edge prediction head
		self.edge_pred = nn.Linear(self.hidden_channels * 2, 1)
		self.edge_weight_pred = nn.Linear(self.hidden_channels * 2, 1)

		# Initialize the convolution layers based on the mode
		self.set_mode(mode)

	def set_mode(self, mode):
		"""
		Dynamically set the forward logic based on the mode.

		Args:
			mode (str): The mode to set. Options are:
						- "message_passing": Use message passing.
						- "no_message_passing": Skip message passing.
		"""
		if mode == "message_passing":
			self.conv1 = Pool_SAGEConv(self.in_channels, self.hidden_channels)
			self.conv2 = Pool_SAGEConv(self.hidden_channels, self.hidden_channels)
		elif mode == "no_message_passing":
			self.conv1 = Pool_SAGEConv(self.in_channels, self.hidden_channels, message_passing=False)
			self.conv2 = Pool_SAGEConv(self.hidden_channels, self.hidden_channels, message_passing=False)
		else:
			raise ValueError(f"Invalid mode: {mode}")
		
		self.mode = mode

	def forward(self, x, prediction_edges, message_edges, message_edgewt=None):
		"""
		Forward logic with message passing.
		"""
		if message_edgewt is None:
			message_edgewt = torch.ones(message_edges.size(1), device=message_edges.device)

		# Perform message passing
		x = self.conv1(x, message_edges, message_edgewt)
		x = self.dropout(x)

		x = self.conv2(x, message_edges, message_edgewt)
		x = self.dropout(x)

		# Predict edges
		edge_embeddings = torch.cat([x[prediction_edges[0]], x[prediction_edges[1]]], dim=-1)
		edge_weights = ReLU(self.edge_weight_pred(edge_embeddings))
		edge_predictor = self.edge_pred(edge_embeddings)
		return edge_weights, edge_predictor
	
class EdgeSampler(torch.utils.data.IterableDataset):
	"""
	Samples minibatches of edges based on centrality or uniform probability.

	Args:
		positive_edges (torch.Tensor): Edge index tensor (2 x num_edges).
		node_embeddings (torch.Tensor, optional): Node feature embeddings. Defaults to None.
		edge_attr (torch.Tensor, optional): Edge attributes. Defaults to None.
		batch_size (int, optional): Number of edges to sample per minibatch. Defaults to 1000.
		num_batches (int, optional): Number of batches to create. Defaults to None.
		centrality (torch.Tensor, optional): Centrality scores for nodes. Defaults to None.
		centrality_fraction (float, optional): Fraction of edges to sample based on centrality. Defaults to None.
		negative_edges (torch.Tensor, optional): Negative edge index tensor. Defaults to None.
		negative_batch_size (int, optional): Number of negative edges to sample per minibatch. Defaults to None.

	Methods:
		__iter__():
			Generates minibatches of edges and their corresponding subgraphs.
			Returns:
				torch_geometric.data.Data
	"""
	def __init__(self, positive_edges, node_embeddings=None, edge_attr=None, batch_size=1000, num_batches=None, centrality=None, centrality_fraction=None, negative_edges=None, negative_batch_size=None):
		super().__init__()
		self.device = positive_edges.device  # Assign device from positive_edges
		self.positive_edges = positive_edges.to(self.device)  # Ensure positive_edges is on the correct device
		self.num_batches = num_batches
		self.edge_attr = edge_attr.to(self.device) if edge_attr is not None else None
		self.node_embeddings = node_embeddings.to(self.device) if node_embeddings is not None else None
		self.batch_size = batch_size
		self.centrality_fraction = centrality_fraction
		self.sampled_edges = set()
		self.total_positive_edges = positive_edges.size(1)
		self.total_positive_nodes = positive_edges.max().item() + 1

		# Ensure node indices and edge_attributes are compatible with edge list
		if node_embeddings is not None:
			if self.total_positive_nodes > node_embeddings.size(0):
				self.node_embeddings = None
				warn("Node embeddings incompatible with edge list. Proceeding without them.")

		if edge_attr is not None and edge_attr.size(0) != self.total_positive_edges:
			self.edge_attr = None
			warn("Edge attribute size mismatch. Proceeding without them.")

		# Compute edge probabilities for sampling
		if centrality is not None and self.total_positive_nodes < centrality.size(0):
			src, dst = self.positive_edges
			self.edge_probs = (centrality[src] + centrality[dst]).to(self.device)
		else:
			self.edge_probs = torch.ones(positive_edges.size(1), device=self.device)

		if negative_edges is not None:
			self.negative_edges = negative_edges.to(self.device)
			self.num_negative_edges = negative_edges.size(1)
			if negative_batch_size is None:
				self.negative_batch_size = self.batch_size * 2
			else:
				self.negative_batch_size = negative_batch_size

			self.create_output_batch = self.create_output_batch_positive_and_negative
			self.max_nodes = max(self.negative_edges.max().item() + 1, self.total_positive_nodes)
		else:
			self.create_output_batch = self.create_output_batch_only_positive
			self.max_nodes = self.total_positive_nodes  # Assuming edges are 0-indexed

		# Create node mask on the correct device
		self.node_mask = torch.zeros(self.max_nodes, dtype=torch.bool, device=self.device)

		# Define sampling method
		if self.centrality_fraction is None or self.centrality_fraction == 1 or centrality is None:
			self.sampling_fn = self.sample_edges_basic
		else:
			self.sampling_fn = self.sample_edges_strata

	def sample_edges_basic(self):
		return torch.multinomial(self.edge_probs, self.batch_size, replacement=False)

	def sample_edges_strata(self):
		mask = torch.ones(self.total_positive_edges, dtype=torch.bool, device=self.device)

		centrality_batch_size = int(self.batch_size * self.centrality_fraction)  # type: ignore
		uniform_batch_size = self.batch_size - centrality_batch_size

		# Centrality-based sampling
		centrality_sampled = torch.multinomial(self.edge_probs, centrality_batch_size, replacement=False)
		mask[centrality_sampled] = False

		# Uniform sampling from the rest
		uniform_pool = torch.where(mask)[0]
		uniform_probs = torch.ones(uniform_pool.size(0), device=self.device)
		uniform_sampled = uniform_pool[torch.multinomial(uniform_probs, uniform_batch_size, replacement=False)]

		return torch.cat([centrality_sampled, uniform_sampled])

	def create_output_batch_only_positive(self):
		"""Create a PyG Data object for the sampled edges."""
		# Sample positive edges
		sampled_edge_idx = self.sampling_fn()
		batch_edges = self.positive_edges[:, sampled_edge_idx]

		# Track sampled positive edges
		self.sampled_edges.update(sampled_edge_idx.tolist())

		# Create node mask and relabel nodes
		node_mask = torch.zeros(self.max_nodes, dtype=torch.bool, device=self.device)
		node_mask[batch_edges.flatten()] = True
		nodes_in_batch = torch.where(node_mask)[0]

		remapped_edge_index, _ = map_index(batch_edges.view(-1), nodes_in_batch, max_index=batch_edges.max().item(), inclusive=True)
		remapped_edge_index = remapped_edge_index.view(2, -1)

		# Create a PyG Data object
		batch = Data(
			edge_index=remapped_edge_index,
		)
		if self.node_embeddings is not None:
			batch.x = self.node_embeddings[node_mask, :]
			batch.n_id = nodes_in_batch

		# Subset edge attributes if available
		if self.edge_attr is not None:
			batch.edge_attr = self.edge_attr[sampled_edge_idx]

		return batch

	def create_output_batch_positive_and_negative(self):
		"""Create a PyG Data object for the sampled edges."""
		# Sample positive edges
		sampled_edge_idx = self.sampling_fn()
		positive_batch = self.positive_edges[:, sampled_edge_idx]
		
		negative_batch_idx = torch.multinomial(
			torch.ones(self.num_negative_edges, device=self.device),
			self.negative_batch_size,
			replacement=False
			).to(self.device)
		# Sample negative edges
		negative_batch = self.negative_edges[:, negative_batch_idx]

		# Combine positive and negative edges
		batch_edges = torch.cat([positive_batch, negative_batch], dim=1)

		# Create edge type mask
		positive_edge_mask = torch.zeros(self.total_positive_edges, dtype=torch.bool, device=self.device)
		positive_edge_mask[:positive_batch.size(1)] = True  # Mark positive edges as True

		# Track sampled positive edges
		self.sampled_edges.update(sampled_edge_idx.tolist())

		# Create node mask and relabel nodes
		node_mask = torch.zeros(self.max_nodes, dtype=torch.bool, device=self.device)
		node_mask[batch_edges.flatten()] = True
		nodes_in_batch = torch.where(node_mask)[0]

		remapped_edge_index, _ = map_index(batch_edges.view(-1), nodes_in_batch, max_index = batch_edges.max().item(), inclusive=True)
		remapped_edge_index = remapped_edge_index.view(2, -1)

		# Create a PyG Data object
		batch = Data(
			edge_index=remapped_edge_index,
			positive_edges=positive_edge_mask  # Mark edges as positive or negative
		)
		if self.node_embeddings is not None:
			batch.x = self.node_embeddings[node_mask, :]
			batch.n_id = nodes_in_batch

		# Subset edge attributes if available
		if self.edge_attr is not None:
			positive_edge_attr = self.edge_attr[sampled_edge_idx]
			negative_edge_attr = torch.zeros(negative_batch.size(1), device=self.device)
			batch.edge_attr = torch.cat([positive_edge_attr, negative_edge_attr], dim=0)

		return batch

	def __iter__(self):
		n = 0
		while (self.num_batches is None or n < self.num_batches) and len(self.sampled_edges) < self.total_positive_edges:
			n += 1
			batch = self.create_output_batch()
			yield batch

def subgraph_with_relabel(original_graph, edge_mask):
	device = original_graph.edge_index.device
	num_nodes = original_graph.x.size(0)

	# Select edges and nodes
	selected_edges = original_graph.edge_index[:, edge_mask]
	node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
	node_mask[selected_edges.flatten()] = True
	selected_nodes = torch.where(node_mask)[0]

	# Relabel edges
	remapped_edge_index, _ = map_index(selected_edges.view(-1), selected_nodes, max_index=num_nodes, inclusive=True)
	remapped_edge_index = remapped_edge_index.view(2, -1)

	# Create the output graph
	outgraph = Data(
		x=original_graph.x[selected_nodes, :],
		edge_index=remapped_edge_index,
		edge_attr=original_graph.edge_attr[edge_mask],
		n_id=selected_nodes,
		e_id=torch.where(edge_mask)[0]  # Edge indices in the new graph
	)
	try:
		outgraph.node_degree = original_graph.node_degree[selected_nodes]
	except NameError:
		warn("Node degrees not present in original graph.")
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
	device = graph.edge_index.device
	num_nodes = graph.x.size(0)
	num_edges = graph.edge_index.size(1)
	src, dst = graph.edge_index

	# Compute centrality if not provided
	if node_centrality is None:
		node_centrality = degree(torch.cat([graph.edge_index[0], graph.edge_index[1]]), num_nodes=num_nodes).to(device)

	# Precompute constants
	average_centrality = node_centrality.mean()
	n_edges_second = int(second_edge_fraction * num_edges)
	max_edges_second = int(n_edges_second * 1.05)
	n_desired_pure_second_edges = int(num_edges * second_edge_fraction_pure)
	max_desired_pure_second_edges = int(n_desired_pure_second_edges * 1.05)
	approx_nodes_second = int(n_edges_second / average_centrality)

	# Preallocate masks
	node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
	mask_src = torch.empty_like(src, dtype=torch.bool, device=device)
	mask_dst = torch.empty_like(dst, dtype=torch.bool, device=device)
	mask_second_pure = torch.empty_like(src, dtype=torch.bool, device=device)
	mask_second = torch.empty_like(src, dtype=torch.bool, device=device)

	num_pure_second_edges = 0
	num_any_second_edges = 0

	for _ in range(max_attempts):
		# Reset node_mask
		node_mask.fill_(False)

		# Sample nodes and set mask
		node_idx_second = torch.randperm(num_nodes, device=device)[:approx_nodes_second]
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
def generate_negative_edges(positive_graph, negative_positive_ratio=2, device=None, max_batch_size=100000):
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
		device = positive_graph.edge_index.device 

	num_nodes = positive_graph.x.size(0)
	num_positive_edges = positive_graph.edge_index.size(1)
	num_negative_edges = num_positive_edges * negative_positive_ratio

	# Compute positive edge keys
	edge_keys = key_edges(positive_graph.edge_index[0], positive_graph.edge_index[1], num_nodes).to(device)

	# Initialize negative edges
	valid_src = torch.empty(0, device=device, dtype=torch.long)
	valid_dst = torch.empty(0, device=device, dtype=torch.long)

	# Loop until enough negative edges are generated
	while valid_src.size(0) < num_negative_edges:
		# Oversample to ensure enough valid negatives
		batch_size = min(int((num_negative_edges - valid_src.size(0)) * 1.5),max_batch_size)

		src = torch.randint(0, num_nodes, (batch_size,), device=device)
		dst = torch.randint(0, num_nodes, (batch_size,), device=device)

		# Compute keys for sampled edges
		sample_keys = key_edges(src, dst, num_nodes)

		# Filter: remove edges that already exist
		mask = ~torch.isin(sample_keys, edge_keys) & (src != dst)

		# Append valid negative edges
		valid_src = torch.cat([valid_src, src[mask]])
		valid_dst = torch.cat([valid_dst, dst[mask]])

	# Slice to the required number of negative edges
	valid_src = valid_src[:num_negative_edges]
	valid_dst = valid_dst[:num_negative_edges]

	# Return negative edges
	return torch.stack([valid_src, valid_dst], dim=0)

def normalize_values(values, min_val=None, max_val=None):
	"""
	Normalize values to the range [0, 1].

	Args:
		values (torch.Tensor): Input tensor of values.
		min_val (float, optional): Minimum value for normalization. If None, uses min of values.
		max_val (float, optional): Maximum value for normalization. If None, uses max of values.

	Returns:
		torch.Tensor: Normalized values.
	"""
	if min_val is None:
		min_val = values.min()
	if max_val is None:
		max_val = values.max()
	return (values - min_val) / (max_val - min_val)


def mask_edges_random(probability_of_masking, edge_list, normalized_centrality=None, device=None): 
	"""
	Randomly mask edges based on a probability mask and optional centrality scores.

	Args:
		probability_of_masking (float): Probability of masking edges.
		edge_list (torch.Tensor): Edge index tensor of shape (2, num_edges).
		normalized_centrality (torch.Tensor, optional): Normalized centrality scores for edges (scaled in the [0,1] range). Defaults to None.
		device (torch.device, optional): Device to perform computations on. Defaults to None.

	Returns:
		torch.Tensor: Boolean mask indicating which edges are selected.
		torch.Tensor: Masked edge list.
	"""
	# Ensure device compatibility
	if device is None:
		device = edge_list.device
	edge_list = edge_list.to(device)

	# Normalize centrality scores if provided
	if normalized_centrality is not None:
		final_probabilities = normalized_centrality.to(device) * probability_of_masking
	else:
		final_probabilities = torch.ones(edge_list.size(1), device=device) * probability_of_masking

	# Generate Bernoulli mask
	bernoulli_mask = torch.bernoulli(final_probabilities).bool()

	# Mask edges
	masked_edges = edge_list[:, bernoulli_mask]

	return bernoulli_mask, masked_edges

bce_loss = nn.BCEWithLogitsLoss()
mse_loss = nn.MSELoss()


def process_data(data, model, optimizer, device, is_training=False):
	"""
	Process a single batch of data for training or validation.

	Args:
		data (torch_geometric.data.Data): Data object containing positive edges, negative edges, edge features, and node features.
		model (nn.Module): The GraphSAGE model.
		device (torch.device): Device to perform computations on.
		is_training (bool): Whether the function is being called during training.

	Returns:
		float: Total loss for the batch.
	"""
	# Move data to the correct device
	data = data.to(device)

	# Mask edges randomly for both positive and negative edges
	all_message_edges, _ = mask_edges_random(0.7, data.edge_index, device=device)

	# Sample neighbors for the minibatch
	nbr_sample_batches = NeighborLoader(
		data,
		num_neighbors=[30, 20],
		batch_size=64,
	)

	total_loss = 0.0

	# Set model mode and optimizer behavior
	if is_training:
		model.train()
		optimizer.zero_grad()  # Zero gradients before backward calls
		conditional_backward = lambda loss: loss.backward()  # Define backpropagation
	else:
		model.eval()
		conditional_backward = lambda loss: None  # No-op for validation

	# Iterate over neighbor sample batches
	for batch in nbr_sample_batches:
		# Move batch to the correct device
		batch = batch.to(device)

		# Mask the message edges for the batch
		mask_message_edges = all_message_edges[batch.e_id]
		mask_positive_edges = batch.positive_edges[batch.e_id]

		# Forward pass through the model
		edge_probability, edge_weight_pred = model(
			batch.x,
			message_edges=batch.edge_index[:,mask_message_edges],
			prediction_edges=batch.edge_index[:,~mask_message_edges],
			message_edgewt = batch.edge_attr[mask_message_edges] if batch.edge_attr is not None else None
		)

		# Create labels for the edges
		labels = torch.zeros(edge_probability.size(0), device=device)
		labels[mask_positive_edges] = 1

		# Compute BCE and MSE losses
		loss = bce_loss(edge_probability, labels) + mse_loss(edge_weight_pred, batch.edge_attr[~mask_message_edges])

		# Accumulate the total loss (for logging)
		total_loss += loss.item()

		# Conditionally backpropagate this batch loss
		conditional_backward(loss)

	# Update optimizer once after accumulating gradients from all batches
	if is_training:
		optimizer.step()

	return total_loss


def load_data(input_graphs_filenames, val_fraction, batch_size, save_graphs_to=None, device=None):
	negative_batch_size = 2 * batch_size
	data_to_save = dict() if save_graphs_to is not None else None
	ingraphs = []
	for files in input_graphs_filenames:
		try:
			ingraphs.append(torch.load(files, weights_only=False))
		except Exception as e:
			print(f"Error loading file {files}: {e}")
			continue
	data_to_save = []
	for fileidx, graph in enumerate(ingraphs):
		# Compute node degrees if not already present
		try:
			graph.node_degree
		except AttributeError:
			graph.node_degree = degree(torch.cat([graph.edge_index[0], graph.edge_index[1]], dim=0), num_nodes=graph.x.size(0))

		# Split graph into training and validation sets
		train, val = bisect_data(graph, second_edge_fraction=val_fraction)

		# Generate negative edges
		negative_edges_for_training = generate_negative_edges(train, negative_positive_ratio=2, device=device)
		negative_edges_for_validation = generate_negative_edges(val, negative_positive_ratio=2, device=device)

		# Save graphs if required
		if save_graphs_to is not None:
			data_to_save.append({
				"Data_name" : Path(input_graphs_filenames[fileidx]).stem,
				"Train": train,
				"Train_Neg": negative_edges_for_training,
				"Val": val,
				"Val_Neg": negative_edges_for_validation,
				"batch_size": batch_size,
				"negative_batch_size": negative_batch_size
			})

		# Save processed graphs to file
		if save_graphs_to is not None:
			torch.save(data_to_save, save_graphs_to)
			print(f"Graphs saved to {save_graphs_to}")

		node_feature_dimension = train.x.size(1)

	return data_to_save, node_feature_dimension
	

def generate_batch(data, centrality_fraction=0.6, device = None):
	"""Generate a batch of data for training and validation."""

	# Create minibatch sampler for training set
	data_sampler = EdgeSampler(
		positive_edges=data["Train"].edge_index,
		node_embeddings=data["Train"].x,
		edge_attr=data["Train"].edge_attr,
		batch_size=data["batch_size"],
		num_batches=None,
		centrality=data["Train"].node_degree,
		centrality_fraction=centrality_fraction,
		negative_edges=data["Train_Neg"],
		negative_batch_size=data["negative_batch_size"]
	)
	minibatch_loader = torch.utils.data.DataLoader(data_sampler, batch_size=None)

	# Prepare validation graph
	val = data["Val"].clone().to(device)
	val.edge_index = torch.cat([val.edge_index, data["Val_Neg"]], dim=1)
	if val.edge_attr is not None:
		val.edge_attr = torch.cat(
			[val.edge_attr, torch.zeros(data["Val_Neg"].size(1), dtype=torch.float32, device=val.edge_index.device)],
			dim=0
		)

	data_for_training = {
		"train_batch_loader": minibatch_loader,
		"val_graph": val
	}

	return data_for_training