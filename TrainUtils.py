from pyparsing import Dict
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.utils.map import map_index
from warnings import warn
from pathlib import Path
from BatchUtils import BatchGenerator

def generate_hidden_dims(input_dim, num_layers, output_dim):
		
	if num_layers == 0:
		return []  # no intermediate layers, just input and last layer

	decay_factor = (output_dim / input_dim) ** (1 / num_layers)
	hidden_dims = [int(input_dim * decay_factor ** i) for i in range(num_layers)]

	return hidden_dims

def build_MLP(dims, activation=nn.ReLU, dropout=0.0, use_layernorm=True, normalize_input=False, activate_final=False):
	layers = []
	if normalize_input:
		layers.append(nn.LayerNorm(dims[0]))
	for i in range(len(dims) - 1):
		layers.append(nn.Linear(dims[i], dims[i + 1]))
		if i < len(dims) - 2:
			if use_layernorm:
				layers.append(nn.LayerNorm(dims[i + 1]))
			layers.append(activation())
			if dropout > 0:
				layers.append(nn.Dropout(dropout))
	if activate_final:
		layers.append(activation())
	return nn.Sequential(*layers)

class BatchStream:
	def __init__(self, cpp_generator, num_batches):
		self.cpp_generator = cpp_generator
		self.num_batches = num_batches
		(
			self.num_supervision_edges,
			self.num_negative_edges
		) = cpp_generator.edge_counts()

		if not torch.cuda.is_available():
			raise RuntimeError("EdgeSampler requires an available CUDA GPU.")

	def __len__(self):
		return self.num_batches

	def __iter__(self):
		for _ in range(self.num_batches):
			with torch.no_grad():
				(
					node_features,
					supervision_edges,
					supervision_labels,
					positive_supervision_weights,
					neighborhood_matrix,
					neighborhood_weights,
				) = self.cpp_generator.next_batch()

			batch = Data(
				node_features=node_features,
				supervision_edges=supervision_edges,
				supervision_labels=supervision_labels,
				positive_supervision_weights=positive_supervision_weights,
				neighborhood_matrix=neighborhood_matrix,
				neighborhood_weights=neighborhood_weights,
				num_negative_edges=self.num_negative_edges
			)

			yield batch.to('cuda')


def subgraph_with_relabel(original_graph: Data, edge_mask: torch.Tensor) -> Data:
	"""
	Extracts a subgraph from the original graph using the provided edge mask,
	relabels node indices to be contiguous, and returns a new Data object.

	Args:
		original_graph (torch_geometric.data.Data): The input graph.
		edge_mask (torch.Tensor): Boolean mask indicating which edges to include.

	Returns:
		torch_geometric.data.Data: The relabeled subgraph.
	"""
	device = original_graph.edge_index.device
	num_nodes = original_graph.x.size(0)

	# Select edges and nodes
	selected_edges = original_graph.edge_index[:, edge_mask]
	node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
	node_mask[selected_edges.flatten()] = True
	selected_nodes = node_mask.nonzero(as_tuple=False).view(-1)

	# Relabel edges
	remapped_edge_index, _ = map_index(selected_edges.view(-1), selected_nodes, max_index=selected_nodes.max()+1, inclusive=True)
	remapped_edge_index = remapped_edge_index.view(2, -1)

	# Create the output graph
	outgraph = Data(
		x=original_graph.x[selected_nodes, :],
		edge_index=remapped_edge_index,
		edge_attr=original_graph.edge_attr[edge_mask],
		n_id=selected_nodes,
		e_id=edge_mask.nonzero(as_tuple=False).view(-1)  # Edge indices in the new graph
	)
	try:
		outgraph.node_degree = original_graph.node_degree[selected_nodes]
	except NameError:
		warn("Node degrees not present in original graph.")
	return outgraph

def bisect_data(graph: Data, second_edge_fraction=0.3, node_centrality=None, max_attempts=50, second_edge_fraction_pure=0.09):
	"""
	Splits a graph into two subgraphs based on edge centrality and node sampling.

	The second subgraph contains a specified fraction of edges, with a subset
	being "pure" (both endpoints in the sampled node set). The function attempts
	to match the desired edge counts within a tolerance.

	Args:
		graph (torch_geometric.data.Data): Input graph.
		second_edge_fraction (float): Fraction of edges for the second subgraph.
		node_centrality (torch.Tensor, optional): Node centrality scores.
		max_attempts (int): Maximum attempts to match edge counts.
		second_edge_fraction_pure (float): Fraction of pure edges in the second subgraph.

	Returns:
		tuple: (first_graph, second_graph)
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
def generate_negative_edges(positive_graph: Data, negative_positive_ratio=2, device=None, max_batch_size=100000) -> torch.Tensor:
	"""
	Generates negative edges for a graph by sampling node pairs that do not exist
	as positive edges and are not self-loops.

	Args:
		positive_graph (torch_geometric.data.Data): Input graph with positive edges.
		negative_positive_ratio (int): Ratio of negative to positive edges.
		device (torch.device, optional): Device for computation.
		max_batch_size (int): Maximum batch size for sampling.

	Returns:
		torch.Tensor: Negative edges (shape [2, num_negative_edges]).
	"""
	if device is None:
		device = positive_graph.edge_index.device
	else:
		positive_graph.to(device)

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


def auc_score(preds: torch.Tensor, labels: torch.Tensor):
	"""
	Compute ROC-AUC score for binary classification.
	
	Args:
		preds: torch.Tensor of predicted probabilities or logits, shape [N]
		labels: torch.Tensor (float) of ground-truth labels (0 or 1), shape [N]
	
	Returns:
		auc: float scalar
	"""
	
	
	# Sort by predictions descending
	sorted_preds, idx = torch.sort(preds, descending=True)
	sorted_labels = labels[idx]

	tpr = torch.zeros(labels.size(0)+1, dtype=torch.float)
	fpr = torch.zeros(labels.size(0)+1, dtype=torch.float)

	total_pos = labels.sum()
	total_neg = labels.size(0) - total_pos
	
	# Cumulative sums of positives and negatives
	tpr[1:] = torch.cumsum(sorted_labels, dim=0)
	fpr[1:] = torch.cumsum(1 - sorted_labels, dim=0)
	
	tpr /= total_pos  # TPR
	fpr /= total_neg  # FPR
	
	# Compute AUC using trapezoid rule
	auc = torch.trapz(tpr, fpr).item()
	return auc

def load_data(input_graphs_filenames, val_fraction, save_graphs_to=None, device=None):
	"""
	Loads graph data from disk, splits each graph into training and validation sets,
	generates negative edges for both, and optionally saves the processed graphs.

	Args:
		input_graphs_filenames (list of str): List of file paths to input graph data (.pt files).
		val_fraction (float): Fraction of edges to use for validation split.
		save_graphs_to (str, optional): Path to save processed graphs. If None, graphs are not saved.
		device (torch.device, optional): Device to move tensors to. If None, uses the graph's device.

	Returns:
		list: List of dictionaries, each containing:
			- "Data_name": Name of the graph (from filename stem).
			- "Train": Training graph (torch_geometric.data.Data).
			- "Train_Neg": Negative edges for training (torch.Tensor).
			- "Val": Validation graph (torch_geometric.data.Data).
			- "Val_Neg": Negative edges for validation (torch.Tensor).
	"""
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
	

def generate_batch(data, num_batches, batch_size, batch_parameters):
	"""
	Generates training and validation batch loaders and samplers.

	Args:
		data (dict): Dictionary containing 'Train', 'Train_Neg', 'Val', 'Val_Neg' Data objects.
		num_batches (int): Number of batches for training.
		batch_size (int): Batch size for training.
		batch_paramters [train and val] (dict): Dictionary containing parameters for the batch loader.
			- supervision_fraction: float (0.0 — 1.0)
			- uniform_message_fraction: float (0.05 — 1.0)
			- fraction_from_unsupervised: float (0.0 — 1.0)
			- max_neighbors: int
			- neighborhood_intensity: float (> 0.0)
			- reference_centrality: float (> 2.0)
			- false_negative_threshold: float (0.0 — 0.5)
			- negative_label_hardness: float (> 0.0)
			- track_coverage_multiple: bool

	Returns:
		dict: Contains train/val samplers, loaders, and input channel size.
	"""
	# Create minibatch sampler for training set
	if num_batches is None:
		num_batches = data["Train"].edge_index.size(1)//int(batch_size*0.8) # type: ignore 


	train_data_sampler = BatchGenerator(
		positive_edges = data["Train"].edge_index,
		positive_edge_weights = data["Train"].edge_attr,
		node_features = data["Train"].x,
		node_centrality = data["Train"].node_degree,
		negative_edges=data["Train_Neg"],
		batch_size=batch_size,
		negative_batch_size = int(2*batch_size * batch_parameters["supervision_fraction"]),
		**batch_parameters
	)

	train_loader = BatchStream(train_data_sampler, num_batches)

	# Create minibatch sampler for validation set
	val_batch_size = data["Val"].edge_index.size(1) // 10
	num_val_batches = data["Val"].edge_index.size(1) // val_batch_size + 1

	val_parameters = {
    **batch_parameters,
    "fraction_from_unsupervised": 0.0,
	}

	val_data_sampler = BatchGenerator(
			positive_edges = data["Val"].edge_index,
			positive_edge_weights = data["Val"].edge_attr,
			node_features = data["Val"].x,
			node_centrality = data["Val"].node_degree,
			negative_edges=data["Val_Neg"],
			batch_size=val_batch_size,
			negative_batch_size = int(2*val_batch_size * val_parameters["supervision_fraction"]),
			**val_parameters
	)

	val_loader = BatchStream(val_data_sampler, num_val_batches)

	data_for_training = {
		"train_batch_loader": train_loader,
		"val_batch_loader": val_loader,
		"input_channels": data["Val"].x.size(1),
		"num_train_samples": num_batches * train_loader.num_supervision_edges,
		"num_val_samples": num_val_batches * val_loader.num_supervision_edges
	}

	return data_for_training

class ParameterScheduler:
		def __init__(self, class_object, attr_name, initial_value, final_value, linear_change = True, factor=1.1, warmup_epochs=0):
				assert factor > 1.0
				assert initial_value != final_value
				assert warmup_epochs >= 0
				self.increasing = final_value > initial_value
				if linear_change:
					self.factor = (final_value - initial_value)/factor
					self.change_fn = lambda value: value + self.factor
				else:
					self.factor = factor if self.increasing else 1.0 / factor
					if initial_value == 0 and self.increasing:
						raise ValueError(
								"Multiplicative scheduling cannot increase from zero"
						)
					self.change_fn = lambda value: value * self.factor
				
				self.class_object = class_object
				self.attr_name = attr_name
				self.initial_value = initial_value
				self.final_value = final_value
				self.warmup_epochs = warmup_epochs
				self.epoch = 0
				self.current_value = initial_value

				setattr(class_object, attr_name, initial_value)

		def step(self):
				self.epoch += 1

				if self.epoch <= self.warmup_epochs:
						return self.current_value

				candidate = self.change_fn(self.current_value)

				if self.increasing:
						new_value = min(candidate, self.final_value)
				else:
						new_value = max(candidate, self.final_value)

				if new_value != self.current_value:
						self.current_value = new_value
						setattr(self.class_object, self.attr_name, new_value)

				return self.current_value
