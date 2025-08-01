import torch
import torch.nn as nn
from torch.nn.functional import relu as ReLU, binary_cross_entropy_with_logits as bce_loss, mse_loss
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.utils.map import map_index
from torch_scatter import scatter_max
from warnings import warn
from pathlib import Path
from max_nbr import max_nbr
from collections import namedtuple

class NodeOnlyDecoder(nn.Module):
	"""
	Implements a simple decoder that predicts edges based on node features.

	Args:
		in_channels (int): Number of input features per node.
		hidden_channels (list[int], optional): List of hidden layer sizes.
		dropout (float, optional): Dropout rate.
	Methods:
		forward(x, edge_index):
			Performs the forward pass of the decoder.
			Args:
				x (torch.Tensor): Node features.
				edge_index (torch.Tensor): Edge indices.
			Returns:
				torch.Tensor: Output edge features.
	"""

	def __init__(self, in_channels, hidden_channels=[2048, 1024, 512], dropout=0.0):
		""" Initializes the NodeOnlyDecoder.
		Args:
			in_channels (int): Number of input features per node.
			hidden_channels (list): List of hidden layer sizes.
		"""
		super().__init__()
		self.in_channels = in_channels*2
		if hidden_channels is None:
			hidden_channels = [2048, 1024, 512]
		self.hidden_channels = hidden_channels
		self.dropout = dropout
		self.dims = [self.in_channels] + self.hidden_channels + [1]
		self.edge_probability = self.generate_conv_layer()
		self.edge_weight_pred = self.generate_conv_layer()

	def generate_conv_layer(self):
		layers = []
		for i in range(len(self.dims) - 2):
			layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
			layers.append(nn.BatchNorm1d(self.dims[i + 1]))
			layers.append(nn.ReLU())
			if self.dropout > 0:
				layers.append(nn.Dropout(p=self.dropout))
		layers.append(nn.Linear(self.dims[-2], self.dims[-1]))
		return nn.Sequential(*layers)

	def forward(self, x, edge_index):
		add = x[edge_index[0]] + x[edge_index[1]]
		prod = x[edge_index[0]] * x[edge_index[1]]
		edge_features = torch.cat([add, prod], dim=-1)
		edge_probabilities = self.edge_probability(edge_features)
		edge_weights = ReLU(self.edge_weight_pred(edge_features))
		return edge_weights, edge_probabilities


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
		self.BN_pool = nn.BatchNorm1d(in_channels)

		# Final linear layer after aggregation
		self.final_lin = nn.Linear(in_channels * 2, out_channels)
		self.BN_final = nn.BatchNorm1d(out_channels)

		# Learnable parameter that controls how much the edge weight influences message aggregation
		self.edge_weight_message_coefficient = nn.Parameter(torch.tensor(0.5))

		
	def forward(self, x, edge_index, edge_weight):
		src, dst = edge_index

		edge_features = x[src] * (1 + self.edge_weight_message_coefficient * edge_weight.unsqueeze(-1))

		# Pool and activate neighbor messages
		pooled = self.pool(edge_features)
		pooled = self.BN_pool(pooled)
		pooled = ReLU(pooled)
		
		# Aggregate neighbor messages via max
		aggregate, _ = scatter_max(pooled, dst, dim=0, dim_size=x.size(0))
		
		# Concatenate self-representation and aggregated neighbors
		h = torch.cat([x, aggregate], dim=-1)
		h = self.final_lin(h)
		h = self.BN_final(h)
		# Final transformation
		return ReLU(h)
		
		
	# def forward_without_message_pooling(self, x, edge_index, edge_weight):
	# 	h = torch.cat([x, torch.zeros_like(x)], dim=-1)
	# 	# Final transformation
	# 	return ReLU(self.final_lin(h))


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
		self.edge_pred = nn.Linear(2*self.hidden_channels, 1)
		self.edge_weight_pred = nn.Linear(2*self.hidden_channels, 1)

		# Initialize the convolution layers based on the mode
		self.set_mode(mode)

	def forward(self, x, supervision_edges, message_edges, message_edgewt=None):
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
		add = x[supervision_edges[0]] + x[supervision_edges[1]]
		prod = x[supervision_edges[0]] * x[supervision_edges[1]]
		edge_embeddings = torch.cat([add, prod], dim=-1)
		edge_weights = ReLU(self.edge_weight_pred(edge_embeddings))
		edge_predictor = self.edge_pred(edge_embeddings)
		return edge_weights, edge_predictor
	
class DualHeadModel(nn.Module):
	"""
	Model with two heads: a GNN-based head and a node-only decoder head.

	Args:
		in_channels (int): Number of input features per node.
		hidden_channels (int): Number of hidden features per node.
		dropout (float, optional): Dropout rate.
	"""
	def __init__(self, in_channels, hidden_channels, dropout=0.0, pred_head = None):
		super().__init__()
		self.GNN = GraphSAGE(in_channels=in_channels, hidden_channels=hidden_channels, dropout=dropout)
		self.NOD = NodeOnlyDecoder(in_channels=in_channels, dropout=dropout)

		if pred_head == "gnn":
			self.forward = self.forward_gnn
		elif pred_head == "nod":
			self.forward = self.forward_nod
		else:
			self.DualOutput = namedtuple("DualOutput", ["gnn", "nod"])
			self.forward = self.forward_both

	def forward_gnn(self, x, supervision_edges, message_edges, message_edgewt=None):
		return self.GNN(x, supervision_edges, message_edges, message_edgewt)

	def forward_nod(self, x, supervision_edges, message_edges=None, message_edgewt=None):
		return self.NOD(x, supervision_edges)
	
	def forward_both(self, x, supervision_edges, message_edges, message_edgewt=None):
		outputs = self.DualOutput(
			gnn = self.GNN(x, supervision_edges, message_edges, message_edgewt),
			nod = self.NOD(x, supervision_edges)
		)
		return outputs

	

	
class EdgeSampler(torch.utils.data.IterableDataset):
	"""
	Samples minibatches of edges based on centrality or uniform probability.

	Args:  
		positive_graph (torch_geometric.data.Data): Input graph with positive edges.  
		batch_size (int, optional): Number of edges to sample per minibatch. Defaults to 1000.  
		num_batches (int, optional): Number of batches to create. Defaults to 100.  
		centrality (torch.Tensor, optional): Centrality scores for nodes. Defaults to None.  
		centrality_fraction (float, optional): Fraction of edges to sample based on centrality. Defaults to 0.5.  
		negative_edges (torch.Tensor, optional): Negative edge index tensor. Defaults to None.  
		negative_batch_size (int, optional): Number of negative edges to sample per minibatch. Defaults to None.  

	Methods:  
		__iter__():  
			Generates minibatches of edges and their corresponding subgraphs.  
			Returns:  
				torch_geometric.data.Data  
	"""  
	def __init__(self, 
			  positive_graph : Data,
			  batch_size=1000,
			  num_batches=100,
			  centrality=None,
			  centrality_fraction=0.5,
			  negative_edges=None,
			  negative_batch_size=None,
			  supervision_fraction = 0.3,
			  max_neighbors = 30,
			  frac_sample_from_unsampled=0.1,
			  nbr_weight_intensity=1.0,
			  device=None,
			  threads=1):  
		super().__init__()
		self.device = device if device is not None else positive_graph.edge_index.device
		self.positive_edges = positive_graph.edge_index.to(self.device)
		self.num_batches = num_batches  
		self.edge_attr = positive_graph.edge_attr.to(self.device) if "edge_attr" in positive_graph else None
		self.node_embeddings = positive_graph.x.to(self.device) if "x" in positive_graph else None
		self.batch_size = batch_size 
		self.centrality_fraction = centrality_fraction  
		self.total_positive_edges = self.positive_edges.size(1)  
		self.total_positive_nodes = self.positive_edges.max().item() + 1  
		self.supervision_fraction = supervision_fraction  
		self.max_neighbors = max_neighbors  
		self.frac_sample_from_unsampled = frac_sample_from_unsampled  
		self.nbr_weight_intensity = nbr_weight_intensity
		self.num_supervision_edges = int(self.batch_size * self.supervision_fraction)
		self.num_message_edges = self.batch_size - self.num_supervision_edges
		self.threads = threads

		# Ensure node indices and edge_attributes are compatible with edge list  
		if self.node_embeddings is not None and self.total_positive_nodes > self.node_embeddings.size(0):  
			self.node_embeddings = None  
			warn("Node embeddings incompatible with edge list. Proceeding without them.")  

		if self.edge_attr is not None and self.edge_attr.size(0) != self.total_positive_edges:  
			self.edge_attr = None  
			warn("Edge attribute size mismatch. Proceeding without them.")  

		# Compute edge probabilities for sampling  
		if centrality is None or self.total_positive_nodes > centrality.size(0):
			warn("Centrality not provided or incompatible with edge list. Recalculating using torch_geometric.utils.degree.") 
			centrality = degree(self.positive_edges.flatten(), num_nodes=self.total_positive_nodes).to(self.device)

		self.centrality = centrality
		self.edge_centrality_scores = self.get_edge_centrality(self.positive_edges)
			
		if negative_edges is not None:  
			self.negative_edges = negative_edges.to(self.device) 
		else:  
			self.negative_edges = generate_negative_edges(self.positive_graph,device=self.device).to(self.device)
		
		if negative_batch_size is None:  
				self.negative_batch_size = int(self.batch_size*supervision_fraction)*2 # Twice the number of supervision edges  
		else:  
			self.negative_batch_size = negative_batch_size  
		
		self.num_negative_edges = self.negative_edges.size(1) 

		# Define sampling method for negative edges  
		if self.negative_batch_size >= self.num_negative_edges:  
			self.sample_negative_edges = lambda : torch.arange(self.num_negative_edges)  
		else:
			self.negative_sampling_weights = torch.ones(self.num_negative_edges, device=self.device)

			self.sample_negative_edges = lambda: torch.multinomial(  
			self.negative_sampling_weights,  
			self.negative_batch_size,  
			replacement=False
			)

		self.max_nodes = max(self.negative_edges.max().item() + 1, self.total_positive_nodes) 
		
		# Preallocate tensors
		
		
		# Node mask to track nodes in the batch
		self.node_mask = torch.zeros(self.max_nodes, dtype=torch.bool, device=self.device)  
		# Track unsampled edges
		self.unsampled_edges = torch.ones(self.total_positive_edges, dtype=torch.bool, device = self.device)  
		# Tensor of ones for uniform random sampling
		self.uniform_probs = torch.ones(self.total_positive_edges, device = self.device)  
		# Masking edges sampled based on centrality
		self.strata_mask_hubs = torch.ones(self.total_positive_edges, dtype=torch.bool, device = self.device)
		# Indices of positive edges		
		self.positive_edge_idx = torch.arange(self.total_positive_edges, device = self.device) 
		# Mask for supervision edges
		self.supervision_edge_mask = torch.zeros(self.batch_size, dtype = torch.bool, device = self.device)
		# Number of supervision edges positive and message edges
		self.num_all_sup_edges = self.num_supervision_edges + self.negative_batch_size
		# Batch edges tensor to hold sampled edges — positive and negative
		self.batch_edges = torch.zeros((2,self.num_message_edges*2 + self.num_all_sup_edges), dtype= torch.long, device = self.device)
		# Supervision labels for edges — positive edges are labeled 1, negative edges are labeled 0
		self.supervision_labels = torch.zeros(self.num_all_sup_edges, device = self.device, dtype=torch.float)
		self.supervision_labels[:self.num_supervision_edges] = 1.0
		# Supervision edge weights for training
		self.supervision_edgewts = torch.zeros_like(self.supervision_labels, dtype= torch.float, device = self.device)
		# Supervision edge importance for loss calculation
		self.supervision_importance = torch.zeros_like(self.supervision_labels, dtype= torch.float, device = self.device)
		# Message edge weights for training
		self.message_edgewts = torch.zeros(2*self.num_message_edges, dtype=torch.float, device = self.device)
		# Indices of sampled positive edges
		self.positive_batch_indices = torch.zeros(self.batch_size, dtype=torch.long, device = self.device)
		# Bidirectional message edges for message passing
		self.bidirectional_message_edges = torch.zeros((2,self.num_message_edges*2), dtype = torch.long, device = self.device)
		# Final message mask to filter valid message edges
		self.final_message_mask = torch.zeros(self.num_message_edges*2, dtype = torch.bool, device = self.device)
		self.positive_wt_scaling = (1/self.edge_centrality_scores.log2()).mean()
		self.negative_wt_scaling = 2*self.edge_centrality_scores.mean()
		
		# Define sampling method for positive edges  
		if self.batch_size >= self.total_positive_edges:  
			self.sampling_fn = lambda: torch.arange(self.total_positive_edges)  
		elif self.centrality_fraction == 1.0:
			self.sampling_fn = self.sample_edges_basic  
		else:  
			self.sampling_fn = self.sample_edges_strata_with_unsampled_tracking  
		

	def get_edge_centrality(self,edge_list):  
		return self.centrality[edge_list[0]] + self.centrality[edge_list[1]]  

	def sample_edges_basic(self):  
		return torch.multinomial(self.uniform_probs, self.batch_size, replacement=False) 
	
	def sample_edges_strata(self, sample_size):
		"""
		Samples a batch of positive edges using stratified sampling based on centrality and uniform probability.

		The batch is composed of:
		- A fraction of edges sampled according to node centrality scores.
		- The remaining edges sampled uniformly from the rest.

		Centrality-based edges are selected first, then uniform edges are sampled from the remaining unsampled edges.
		The function updates internal masks to track sampled hubs and fills a preallocated buffer with the selected edge indices.

		Args:
			sample_size (int): Number of edges to sample in the batch.

		Returns:
			torch.Tensor: Indices of sampled positive edges for the batch (shape: [sample_size]).
		"""
		centrality_batch_size = int(sample_size * self.centrality_fraction)
		uniform_batch_size = sample_size - centrality_batch_size  

		# Centrality-based sampling  
		centrality_sampled_edges = torch.multinomial(self.edge_centrality_scores, centrality_batch_size, replacement=False)  
		self.strata_mask_hubs[centrality_sampled_edges] = False  

		# Uniform sampling from the rest  
		uniform_sampled_indices = torch.multinomial(self.uniform_probs[self.strata_mask_hubs], uniform_batch_size, replacement=False)  
		uniform_sampled_edges = self.positive_edge_idx[self.strata_mask_hubs][uniform_sampled_indices]  
		self.strata_mask_hubs.fill_(True) 

		self.positive_batch_indices[:centrality_batch_size] = centrality_sampled_edges
		self.positive_batch_indices[centrality_batch_size:sample_size] = uniform_sampled_edges
		return self.positive_batch_indices  

	def sample_edges_strata_total(self):  
		sampled_edges = self.sample_edges_strata(self.batch_size)  
		return sampled_edges  
		
	def sample_edges_strata_with_unsampled_tracking(self):  
		"""
		Stratified sampling of edges with tracking of unsampled edges.

		This function samples a batch of edges by combining:
		- A fraction from all edges (using centrality and uniform sampling).
		- A fraction from edges that have not been sampled yet.

		Once all unsampled edges are used, it resets the sampling function to sample from all edges.

		Returns:
			torch.Tensor: Indices of sampled positive edges for the batch.
		"""
		# Sample from total  
		num_sample_from_total = int(self.batch_size * (1 - self.frac_sample_from_unsampled))  
		sampled_from_total = self.sample_edges_strata(num_sample_from_total)  
		self.unsampled_edges[sampled_from_total] = False  

		# Sampling from unsampled   
		num_unsampled = self.unsampled_edges.sum()
		num_sample_from_unsampled = self.batch_size - num_sample_from_total

		# Precompute unsampled edge subset and corresponding uniform probs
		unsampled_edge_indices = self.positive_edge_idx[self.unsampled_edges]

		if num_unsampled > num_sample_from_unsampled:

			sampled_from_unsampled = torch.multinomial(self.uniform_probs[self.unsampled_edges], num_sample_from_unsampled, replacement=False)

			# Fill the remaining slots in the preallocated buffer
			self.positive_batch_indices[num_sample_from_total:self.batch_size] = unsampled_edge_indices[sampled_from_unsampled]

			# Track that these unsampled edges have now been used
			original_indices = unsampled_edge_indices[sampled_from_unsampled]
			self.unsampled_edges[original_indices] = False

		else:
			# Fill all unsampled edges
			self.positive_batch_indices[num_sample_from_total:num_sample_from_total + num_unsampled] = unsampled_edge_indices

			# Mark all unsampled edges as used
			self.unsampled_edges.fill_(False) 

			# Reset sampling function to sample from total edges
			self.sampling_fn = self.sample_edges_strata_total

			# Resample remaining from total
			num_resample_from_total = self.batch_size - (num_sample_from_total + num_unsampled)
			if num_resample_from_total > 0:
				resampled_from_total = torch.multinomial(self.uniform_probs, num_resample_from_total, replacement=False)

				# Fill the remaining slots in the preallocated buffer
				self.positive_batch_indices[-num_resample_from_total:] = self.positive_edge_idx[resampled_from_total]

		return self.positive_batch_indices


	def create_output_batch(self):  
		"""
		Creates a PyTorch Geometric Data object for the current minibatch.

		Steps:
		- Samples positive and negative edges for the batch.
		- Assigns supervision and message edges.
		- Relabels nodes and edges for the batch.
		- Restricts the number of message edges per node to `max_neighbors`.
		- Computes supervision labels and importance weights.
		- Assigns edge weights if available.

		Returns:
			batch (torch_geometric.data.Data): Data object containing batch edges (message and supervision), labels and weights, and node features.
		"""
		# Sample positive edges  
		self.sampling_fn()  
		positive_batch = self.positive_edges.index_select(1,self.positive_batch_indices).to(self.device)

		# Define supervision edges 
		self.supervision_edge_mask.fill_(False)

		# Sample supervision edges  
		supervision_positive_idx = torch.multinomial(self.uniform_probs[:self.batch_size], self.num_supervision_edges, replacement=False)  
		self.supervision_edge_mask[supervision_positive_idx] = True  

		# Sample negative edges 
		negative_batch_edges = self.negative_edges.index_select(1,self.sample_negative_edges()).to(self.device)
	
		self.batch_edges.zero_()
		
		# Add message edges (non-supervision positive edges)
		self.batch_edges[:, :self.num_message_edges] = positive_batch[:, ~self.supervision_edge_mask]

		# Add positive supervision edges
		self.batch_edges[:, self.num_message_edges:self.batch_size] = positive_batch[:, supervision_positive_idx]

		del positive_batch

		# Add negative edges
		self.batch_edges[:, -self.negative_batch_size:] = negative_batch_edges

		self.supervision_importance.zero_()
		# Learning weights for negative edges (high score negatives between hubs)  
		self.supervision_importance[self.num_supervision_edges:] = (self.centrality[negative_batch_edges[0]] + self.centrality[negative_batch_edges[1]])/self.negative_wt_scaling

		# Learning weights for positive edges (high score positives between peripheral nodes)  
		self.supervision_importance[:self.num_supervision_edges] = (1/torch.log2(self.edge_centrality_scores[self.positive_batch_indices[self.supervision_edge_mask]]))/self.positive_wt_scaling
		
		
		# Relabel batch edges #  
		
		self.node_mask.fill_(False)  
		self.node_mask[self.batch_edges.flatten()] = True  
		nodes_in_batch = self.node_mask.nonzero(as_tuple=False).view(-1)

		remapped_edge_index, _ = map_index(self.batch_edges.view(-1), nodes_in_batch, max_index = nodes_in_batch.max()+1, inclusive=True)  
		remapped_edge_index = remapped_edge_index.view(2, -1)  
		
		tentative_message_edges = remapped_edge_index[:,:self.num_message_edges]

		# Make messages directional  
		self.bidirectional_message_edges.zero_()
		self.bidirectional_message_edges[:, :self.num_message_edges] = tentative_message_edges
		self.bidirectional_message_edges[:, self.num_message_edges:] = tentative_message_edges.flip(0)

		self.supervision_edgewts.zero_()
		self.message_edgewts.zero_()
		# Subset edge attributes if available  
		if self.edge_attr is not None:  
			sampled_positive_edgewts = self.edge_attr.index_select(0,self.positive_batch_indices)  
			message_edgewts = sampled_positive_edgewts[~self.supervision_edge_mask]

			# Assign positive supervision edge weights (negative edges have zero weight)
			self.supervision_edgewts[:self.num_supervision_edges] = sampled_positive_edgewts[self.supervision_edge_mask]
			
			self.message_edgewts[:self.num_message_edges] = message_edgewts
			self.message_edgewts[self.num_message_edges:] = message_edgewts
		
		# Apply neighborhood restriction
		
		self.final_message_mask.fill_(False)

		src, dst = self.bidirectional_message_edges 
		
		# Compute degrees for message nodes  
		degrees = degree(src, num_nodes = src.max()+1)  
		deg_src = degrees[src]  
		deg_dst = degrees[dst]
		
		# Determine neighbor weights based on source and destination degrees
		weights = deg_src / deg_dst 
		# Augment neighbor weights with edge weights if available
		if self.edge_attr is not None:
			weights.mul_(self.message_edgewts)

		# Intensify or dampen neighbor weights
		weights.pow_(self.nbr_weight_intensity)

		# Identify violators (nodes with degree greater than max_neighbors)
		violators_mask = deg_dst > self.max_neighbors  
		# violator_dst_nodes = torch.unique(dst[violators_mask]) 

		# Mark non-violators as True in final_message_mask
		self.final_message_mask[~violators_mask] = True  

		self.final_message_mask |= max_nbr(dst, weights, violators_mask, self.max_neighbors, nthreads=self.threads)

		# for d in violator_dst_nodes:  
		# 	edge_indices = (dst == d).nonzero(as_tuple=False).view(-1)  
		# 	w = weights[edge_indices]  
		# 	sampled = torch.multinomial(w, self.max_neighbors, replacement=False)  
		# 	self.final_message_mask[sampled] = True   

		final_message_edges = self.bidirectional_message_edges[:,self.final_message_mask]  

		# Create a PyG Data object  
		batch = Data(  
			message_edges = final_message_edges,  
			supervision_edges = remapped_edge_index[:, -self.num_all_sup_edges:], 
			supervision_labels = self.supervision_labels,  
			supervision_importance = self.supervision_importance  
		).to(self.device)
		if self.node_embeddings is not None:  
			batch.node_features = self.node_embeddings[self.node_mask, :].to(self.device)

			
		# Assign edge weights to the batch (if available, otherwise zero)
		batch.message_edgewts = self.message_edgewts[self.final_message_mask]  
		batch.supervision_edgewts = self.supervision_edgewts

		return batch  
	
	def __iter__(self):  
		n = 0  
		while (self.num_batches is None or n < self.num_batches):  
			n += 1  
			yield self.create_output_batch()

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

def normalize_values(values: torch.tensor, min_val=None, max_val=None):
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

def calculate_loss(model_output, data, head_weights = [0.5, 0.5]):
	"""
	Calculates the loss for the model output.

	Args:
		model_output (tuple): Output from the model containing edge probabilities and edge weights. DualHeadModel outputs two tuples of (edge_weights, edge_predictor) where the first tuple is for the GNN and the second is for the node-only decoder.
		data (torch_geometric.data.Data): Data object containing supervision labels and importance.
		head_weights (torch.Tensor): Weights for the heads in the model.
	Returns:
		torch.Tensor: Computed loss value.
	"""
	assert len(model_output) == len(head_weights), "Mismatch between model output and head weights length."
	total_loss = torch.tensor(0.0, device=data.supervision_labels.device, requires_grad=True)

	for i, w in enumerate(head_weights):
		edge_probability, edge_weight_pred = model_output[i]
		loss = bce_loss(edge_probability.squeeze(-1), data.supervision_labels, weight=data.supervision_importance) + mse_loss(edge_weight_pred.squeeze(-1), data.supervision_edgewts)
		total_loss = total_loss + w * loss
	return total_loss

def process_data(data:Data, model:nn.Module, optimizer:torch.optim.Optimizer, device:torch.device, head_weights = [0.5, 0.5], is_training=False):
	"""
    Processes a single batch for training or validation.

    Moves data to the correct device, performs a forward pass, computes loss,
    and (if training) performs backpropagation and optimizer step.

    Args:
        data (torch_geometric.data.Data): Batch data object.
        model (nn.Module): The GraphSAGE model.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device for computation.
        is_training (bool): If True, performs training steps.

    Returns:
        float: Loss value for the batch.
    """
	# Move data to the correct device
	data = data.to(device)

	# Set model mode and optimizer behavior
	if is_training:
		optimizer.zero_grad()  # Zero gradients before backward calls
		conditional_backward = lambda loss: loss.backward()  # Define backpropagation
	else:
		conditional_backward = lambda loss: None  # No-op for validation

	model_output = model(
		data.node_features,
		message_edges=data.message_edges,
		supervision_edges=data.supervision_edges,
		message_edgewt = data.message_edgewts
	)
	
	# Compute BCE and MSE losses

	loss = calculate_loss(model_output, data, head_weights)
	conditional_backward(loss)

	if is_training:
		optimizer.step()

	return loss.item()


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
	

def generate_batch(data, num_batches, batch_size, centrality_fraction=0.6, nbr_wt_intensity=1.0, device = None, threads=1):
	"""
    Generates training and validation batch loaders and samplers.

    Args:
        data (dict): Dictionary containing 'Train', 'Train_Neg', 'Val', 'Val_Neg' Data objects.
        num_batches (int): Number of batches for training.
        batch_size (int): Batch size for training.
        centrality_fraction (float): Fraction of edges to sample by centrality.
        device (torch.device, optional): Device for computation.

    Returns:
        dict: Contains train/val samplers, loaders, and input channel size.
    """
	# Create minibatch sampler for training set
	if num_batches is None:
		num_batches = data["Train"].edge_index.size(1)//int(batch_size*0.8) # type: ignore 

	train_data_sampler = EdgeSampler(
		positive_graph=data["Train"],
		batch_size=batch_size,
		num_batches=num_batches,
		centrality=data["Train"].node_degree,
		centrality_fraction=centrality_fraction,
		negative_edges=data["Train_Neg"],
		nbr_weight_intensity=nbr_wt_intensity,
		threads=threads
	)
	train_loader = torch.utils.data.DataLoader(train_data_sampler, batch_size=None)

	# Create minibatch sampler for validation set
	val_batch_size = data["Val"].edge_index.size(1) // 10
	num_val_batches = data["Val"].edge_index.size(1) // val_batch_size + 1

	val_data_sampler = EdgeSampler(
		positive_graph=data["Val"],
		batch_size=val_batch_size,
		num_batches=num_val_batches,
		centrality=data["Val"].node_degree,
		centrality_fraction=centrality_fraction,
		negative_edges=data["Val_Neg"],
		nbr_weight_intensity=nbr_wt_intensity,
		threads=threads
		)

	val_loader = torch.utils.data.DataLoader(val_data_sampler, batch_size=None)

	data_for_training = {
		"train_sampler": train_data_sampler,
		"val_sampler": val_data_sampler,
		"train_batch_loader": train_loader,
		"val_batch_loader": val_loader,
		"input_channels": data["Val"].x.size(1)
	}

	return data_for_training