import torch
import torch.nn as nn
from torch.nn.functional import relu as ReLU, binary_cross_entropy_with_logits as bce_loss, mse_loss
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

		edge_features = x[src] * (1 + self.edge_weight_message_coefficient * edge_weight.unsqueeze(-1))

		# Pool and activate neighbor messages
		pooled = ReLU(self.pool(edge_features))
		
		# Aggregate neighbor messages via max
		aggregate, _ = scatter_max(pooled, dst, dim=0, dim_size=x.size(0))
		
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
		self.edge_pred = nn.Linear(self.hidden_channels, 1)
		self.edge_weight_pred = nn.Linear(self.hidden_channels, 1)

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
		edge_embeddings = x[prediction_edges[0]] + x[prediction_edges[1]]
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
	def __init__(self, positive_edges, node_embeddings=None, edge_attr=None, batch_size=1000, num_batches=None, centrality=None, centrality_fraction=None, negative_edges=None, negative_batch_size=None, message_fraction = 0.7, max_neighbors = 30, alpha_max = 10.0, sharpness_nbr_max = 1):
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
		self.centrality = normalize_values(centrality)
		self.message_fraction = message_fraction
		self.max_neighbors = max_neighbors
		self.alpha_max = alpha_max
		self.sharpness_nbr_max = sharpness_nbr_max

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
			if negative_batch_size is None:
				self.negative_batch_size = self.batch_size * 2
			else:
				self.negative_batch_size = negative_batch_size
		else:
			self.negative_edges = generate_negative_edges(self.positive_edges,device=self.device)
			self.negative_batch_size = self.batch_size * 2
		
		self.num_negative_edges = self.negative_edges.size(1)

		self.max_nodes = max(self.negative_edges.max().item() + 1, self.total_positive_nodes)

		# Create node mask on the correct device
		self.node_mask = torch.zeros(self.max_nodes, dtype=torch.bool, device=self.device)

		# Define sampling method for positive edges
		if self.batch_size >= self.total_positive_edges:
			self.sampling_fn = lambda: torch.arange(self.total_positive_edges)
		elif self.centrality_fraction is None or self.centrality_fraction == 1 or centrality is None:
			self.sampling_fn = self.sample_edges_basic
		else:
			self.sampling_fn = self.sample_edges_strata

		# Define sampling method for negative edges
		if self.negative_batch_size >= self.num_negative_edges:
			self.sample_negative_edges = lambda : torch.arange(self.num_negative_edges)
		else:
			self.sample_negative_edges = lambda: torch.multinomial(
			torch.ones(self.num_negative_edges, device=self.device),
			self.negative_batch_size,
			replacement=False
			)

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

	def create_output_batch(self):
		"""Create a PyG Data object for the sampled edges."""
		# Sample positive edges
		sampled_edge_idx = self.sampling_fn()
		# Track sampled positive edges
		self.sampled_edges.update(sampled_edge_idx.tolist())

		positive_batch = self.positive_edges[:, sampled_edge_idx]
		positive_rev = positive_batch.flip(0)
		positive_bidirectional = torch.cat([positive_batch, positive_rev], dim=1)

		src, dst = positive_bidirectional

		positive_batch_size = src.size(0)
		dst_indegree = torch.bincount(dst,minlength=positive_batch_size)[dst]

		alpha_v  = self.alpha_max * torch.sigmoid(self.sharpness_nbr_max*(dst_indegree - self.max_neighbors))
		
		ratio = ((self.centrality[dst] + 1e-6) / (self.centrality[src] + 1e-6)).clamp(min=1e-6)
		log_weight = alpha_v * torch.log(ratio)
		message_edge_weight = torch.sigmoid(alpha_v * log_weight)

		message_indices = torch.multinomial(message_edge_weight, int(self.message_fraction*positive_batch_size), replacement = False)

		negative_batch_idx = self.sample_negative_edges().to(self.device)

		supervision_edge_mask = torch.ones(positive_batch_size + self.negative_batch_size, dtype = torch.bool)

		supervision_edge_mask[message_indices] = False

		negative_edge_mask = torch.zeros_like(supervision_edge_mask, dtype = torch.bool)

		negative_edge_mask[-self.negative_batch_size:] = True

	
		batch_edges = torch.cat([
			positive_bidirectional[:,message_indices],
			positive_bidirectional[:,supervision_edge_mask[:positive_batch_size]],
			self.negative_edges[:, negative_batch_idx]
			],
			dim=1)
		
		print(f"Batch_edges shape = {batch_edges.shape}")

		# Create node mask and relabel nodes
		node_mask = torch.zeros(self.max_nodes, dtype=torch.bool, device=self.device)
		node_mask[batch_edges.flatten()] = True
		nodes_in_batch = torch.where(node_mask)[0]

		remapped_edge_index, _ = map_index(batch_edges.view(-1), nodes_in_batch, max_index = batch_edges.max().item(), inclusive=True)
		remapped_edge_index = remapped_edge_index.view(2, -1)

		supervision_edges = remapped_edge_index[:,supervision_edge_mask]
		message_edges = remapped_edge_index[:,~supervision_edge_mask]
		supervision_labels = torch.ones(supervision_edges.size(1), dtype = torch.float)
		supervision_labels[-self.negative_batch_size:] = 0.0
		srcs, dsts = supervision_edges
		supervision_importance = (self.centrality[srcs] + self.centrality[dsts])/2
		supervision_importance[:-self.negative_batch_size] = 1.0 - supervision_importance[:-self.negative_batch_size]

		# Create a PyG Data object
		batch = Data(
			message_edges = message_edges,
			supervision_edges = supervision_edges,
			supervision_labels = supervision_labels,
			supervision_importance = supervision_importance
		)
		if self.node_embeddings is not None:
			batch.node_features = self.node_embeddings[node_mask, :]
			#batch.n_id = nodes_in_batch

		# Subset edge attributes if available
		if self.edge_attr is not None:
			sampled_positive_edgewts = self.edge_attr[sampled_edge_idx]
			all_edge_attr = torch.cat([
				sampled_positive_edgewts, 
				sampled_positive_edgewts.flip(0), 
				torch.zeros(self.negative_batch_size)]
				)
			batch.message_edgewts = all_edge_attr[~supervision_edge_mask]
			batch.supervision_edgewts = all_edge_attr[supervision_edge_mask]
		return batch
	
	def __iter__(self):
		n = 0
		self.sampled_edges = set()
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

	# Set model mode and optimizer behavior
	if is_training:
		optimizer.zero_grad()  # Zero gradients before backward calls
		conditional_backward = lambda loss: loss.backward()  # Define backpropagation
	else:
		conditional_backward = lambda loss: None  # No-op for validation

	edge_probability, edge_weight_pred = model(
		data.node_features,
		message_edges=data.message_edges,
		prediction_edges=data.supervision_edges,
		message_edgewt = data.message_edgewts
	)
	
	# Compute BCE and MSE losses

	loss = bce_loss(edge_probability.squeeze(-1), data.supervision_labels, weight=data.supervision_importance) + mse_loss(edge_weight_pred.squeeze(-1), data.supervision_edgewts)

	conditional_backward(loss)

	if is_training:
		optimizer.step()

	return loss.item()


def load_data(input_graphs_filenames, val_fraction, save_graphs_to=None, device=None):
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
			})

		# Save processed graphs to file
		if save_graphs_to is not None:
			torch.save(data_to_save, save_graphs_to)
			print(f"Graphs saved to {save_graphs_to}")

	return data_to_save
	

def generate_batch(data, batch_size, centrality_fraction=0.6, device = None):
	"""Generate a batch of data for training and validation."""
	negative_batch_size = 2 * batch_size
	# Create minibatch sampler for training set
	data_sampler = EdgeSampler(
		positive_edges=data["Train"].edge_index,
		node_embeddings=data["Train"].x,
		edge_attr=data["Train"].edge_attr,
		batch_size=batch_size,
		num_batches=None,
		centrality=data["Train"].node_degree,
		centrality_fraction=centrality_fraction,
		negative_edges=data["Train_Neg"],
		negative_batch_size=negative_batch_size
	)
	minibatch_loader = torch.utils.data.DataLoader(data_sampler, batch_size=None)

	data_sampler_val = EdgeSampler(
		positive_edges=data["Val"].edge_index,
		node_embeddings=data["Val"].x,
		edge_attr = data["Val"].edge_attr,
		batch_size=data["Val"].edge_index.size(1),
		num_batches=None,
		centrality=data["Val"].node_degree,
		centrality_fraction=centrality_fraction,
		negative_edges=data["Val_Neg"],
		negative_batch_size=2*data["Val"].edge_index.size(1)
	)
	# Prepare validation graph
	val_data = data_sampler_val.create_output_batch()

	data_for_training = {
		"train_batch_loader": minibatch_loader,
		"val_graph": val_data
	}

	return data_for_training