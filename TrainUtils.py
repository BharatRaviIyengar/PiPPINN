import torch
import torch.nn as nn
from torch.nn.functional import relu as ReLU, binary_cross_entropy_with_logits as bce_logit_loss, binary_cross_entropy as bce_loss, mse_loss, softplus, cosine_similarity as cosim
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.utils.map import map_index
from torch_scatter import scatter_add, scatter_mean
from torch_scatter.composite import scatter_softmax
from warnings import warn
from pathlib import Path
from max_nbr import max_nbr


class NeighborhoodEncoder(nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.0):
		super().__init__()
		self.pool = nn.Linear(in_channels, in_channels)
		self.LN_pool = nn.LayerNorm(in_channels)
		self.in_channels = in_channels
		self.hidden_channels = hidden_channels
		self.dropout = dropout
		self.dims = [self.in_channels] + self.hidden_channels

		self.edge_weight_message_coefficient = nn.Parameter(torch.tensor(0.5))
		self.final_transform = self.build_mlp()

		self.out = nn.Linear(self.dims[-1], out_channels)
	
	def build_mlp(self):
		layers = []
		layers.append(nn.LayerNorm(self.dims[0]))
		for i in range(len(self.dims) - 1):
			layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
			layers.append(nn.LayerNorm(self.dims[i + 1]))
			layers.append(nn.ReLU())
			if self.dropout > 0 and i < len(self.dims) - 1:
				layers.append(nn.Dropout(p=self.dropout))		
		return nn.Sequential(*layers)

	def forward(self, x, edge_index, edge_weight):
		src, dst = edge_index
		EWMC = softplus(self.edge_weight_message_coefficient)
		neighborhood_features = x[src] * (1 + EWMC * edge_weight.unsqueeze(-1))
		pooled = self.pool(neighborhood_features)
		pooled = self.LN_pool(pooled)
		pooled = ReLU(pooled)
		
		# Aggregate neighbor messages via softmax attention
		attn = scatter_softmax(pooled, dst, dim=0)
		aggregate = scatter_mean(pooled * attn, dst, dim=0, dim_size=x.size(0))
		neighborhood = self.final_transform(aggregate)
		return self.out(neighborhood)
	
class Decoder(nn.Module):
	def __init__(self, in_channels, hidden_channels, dropout=0.0):
		super().__init__()
		self.in_channels = in_channels
		self.hidden_channels = hidden_channels
		self.dropout = dropout
		self.dims = [2*self.in_channels] + self.hidden_channels
		self.edge_embedder = self.build_mlp()
		self.edge_wt_head = nn.Linear(self.dims[-1]+1, 1)
		self.edge_prob_head = nn.Linear(self.dims[-1],1)

		# self.coef_matrix = nn.Parameter(torch.randn(in_channels, coef_matrix_cols))
		self.w_node = nn.Parameter(torch.tensor(0.5))
		self.w_nbr = nn.Parameter(torch.tensor(0.5))

	def build_mlp(self):
		layers = []
		layers.append(nn.LayerNorm(self.dims[0]))
		for i in range(len(self.dims) - 1):
			layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
			layers.append(nn.LayerNorm(self.dims[i + 1]))
			layers.append(nn.ReLU())
			if self.dropout > 0 and i < len(self.dims) - 1:
				layers.append(nn.Dropout(p=self.dropout))
		return nn.Sequential(*layers)
	
	def congruence_score(self, node_embeddings, message_edges, supervision_edges, softmax_intensity=10.0):
		msg_src, msg_dst = message_edges
		sup_src, sup_dst = supervision_edges
		device = node_embeddings.device
		N = node_embeddings.size(0)

		# Step 1: Build directed supervision pairs
		sup_query = torch.cat([sup_src, sup_dst])   # X
		sup_partner = torch.cat([sup_dst, sup_src]) # Y
		M = sup_query.size(0)

		# Step 2: Gather message neighbors of each Y
		mask = torch.isin(msg_dst, sup_partner)

		msg_neighbors = msg_src[mask]      # Z => neighbors of Y
		msg_partner = msg_dst[mask]        # Y

		# Step 3: Map partner nodes Y -> directed supervision index
		# Each sup_partner appears in sup_query/sup_partner pairs
		partner_index = torch.full((N,), -1, device=device)
		partner_index[sup_partner] = torch.arange(M, device=device)

		msg_partner_idx = partner_index[msg_partner]

		# Step 4: Count how many message neighbors per supervision query
		neighbor_count = scatter_add(
			torch.ones_like(msg_partner_idx),
			msg_partner_idx,
			dim=0,
			dim_size=M
		)

		# Step 5: Expand sup_query nodes (global indices) to align with neighbors
		expanded_queries = sup_query.repeat_interleave(neighbor_count)

		# Step 6: Compute similarity: supervised node vs message neighbors
		cosine_sim = cosim(
			node_embeddings[expanded_queries],	# X
			node_embeddings[msg_neighbors],	# Z
			dim=-1
		)

		# Step 7: Softmax per supervision query node
		weights = scatter_softmax(
			softmax_intensity * cosine_sim,
			expanded_queries,
			dim=0
		)

		# Step 8: Aggregate to one congruence score per node
		congruence_node = scatter_mean(
			weights,
			expanded_queries,
			dim=0,
			dim_size=N
		)

		# Step 9: Map back to supervision edges
		congruence_per_edge = congruence_node[sup_src] + congruence_node[sup_dst]

		return congruence_per_edge

	def forward(self, node_embeddings, nbrs_embeddings, edge_index):
		u, v  = edge_index
		additive = node_embeddings[u] + node_embeddings[v]
		multiplicative = node_embeddings[u] * node_embeddings[v]
		# combined = additive * self.coef_matrix @ multiplicative
		# # alternative:
		combined = torch.cat([additive, multiplicative], dim=-1)
		edge_features = self.edge_embedder(combined)
		nbrs_similarity = torch.sum(nbrs_embeddings[u] * nbrs_embeddings[v], dim=-1, keepdim=True)
		node_contribution = self.edge_prob_head(edge_features) * self.w_node
		nbr_contribution =  nbrs_similarity * self.w_nbr
		edge_probabilities =  node_contribution + nbr_contribution
		fraction_node_contribution = node_contribution / (node_contribution + nbr_contribution + 1e-8)
		edge_weights = ReLU(self.edge_wt_head(torch.cat([edge_features, nbrs_similarity], dim=-1)))
		return edge_probabilities, edge_weights, fraction_node_contribution

def hinge_loss(margin: float, positive_logits, negative_logits):
	loss_margin = torch.relu(margin - positive_logits).mean() + torch.relu(margin + negative_logits).mean()
	return loss_margin

def total_entropy(logits):
	log_p = -softplus(-logits)
	log_1mp = -softplus(logits)
	p = torch.exp(log_p)
	entropy = -(p * log_p + (1 - p) * log_1mp).mean()
	return entropy

class Margin_and_Entropy():
	def __init__(self, num_positive_edges = 12000, margin = 0.5, margin_loss_coef = 0.05, entropy_coef = 0.005):
		self.num_positive_edges = num_positive_edges
		self.margin = margin
		self.margin_loss_coef = margin_loss_coef
		self.entropy_coef = entropy_coef
	def __call__(self, logits):
		positive_logits = logits[:self.num_positive_edges]
		negative_logits = logits[self.num_positive_edges:]
		loss_margin = hinge_loss(self.margin, positive_logits, negative_logits)
		entropy = total_entropy(logits)
		return self.margin_loss_coef * loss_margin - self.entropy_coef * entropy


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
			  threads=1,
				soft_negative_labels=False,
				max_neg_edge_centrality=None,
				negative_label_hardness=1,
				uniform_supervision_weights = False
				):  
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
		self.num_sample_from_total = int(self.batch_size * (1 - self.frac_sample_from_unsampled))
		self.num_sample_from_unsampled = self.batch_size - self.num_sample_from_total
		self.nbr_weight_intensity = nbr_weight_intensity
		self.num_supervision_edges = int(self.batch_size * self.supervision_fraction)
		self.num_message_edges = self.batch_size - self.num_supervision_edges
		self.threads = threads
		self.uniform_supervision_weights = uniform_supervision_weights
		self.soft_negative_labels = soft_negative_labels

		assert (1 - self.centrality_fraction) >= self.supervision_fraction, "Supervision edges should always be sampled uniformly. The number of uniformly sampled edges in the batch should be at least as much as the number of supervision edges"


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
			self.negative_edges = generate_negative_edges(positive_graph,device=self.device).to(self.device)
		
		if negative_batch_size is None:  
				self.negative_batch_size = self.num_supervision_edges*2 # Twice the number of supervision edges  
		else:  
			self.negative_batch_size = negative_batch_size  
		
		self.num_negative_edges = self.negative_edges.size(1) 

		if self.soft_negative_labels:
			self.max_neg_edge_centrality = max_neg_edge_centrality if max_neg_edge_centrality is not None else self.get_edge_centrality(self.negative_edges).max().item()
			self.negative_label_hardness = negative_label_hardness

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
		# Track uniformly sampled edges
		self.is_uniform_edge = torch.zeros(self.batch_size, dtype=torch.bool)
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

		if uniform_supervision_weights:
			self.supervision_importance[:self.num_supervision_edges] = 1/self.num_supervision_edges
			self.supervision_importance[self.num_supervision_edges:] = 1/self.negative_batch_size
		

	def get_edge_centrality(self,edge_list):  
		return self.centrality[edge_list[0]] + self.centrality[edge_list[1]]  
	
	def generate_soft_negative_labels(self, edge_centrality, min_centrality=2, max_centrality = None, hardness=1):
		max_centrality = edge_centrality.max() if max_centrality is None else max_centrality
		negative_probs = 0.5 * ((max_centrality - edge_centrality) / (max_centrality - min_centrality + 1e-8)) ** hardness
		return negative_probs.clamp(min=0.0, max=0.5)
	
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
		self.strata_mask_hubs.fill_(True)
		self.is_uniform_edge.fill_(False)
		centrality_batch_size = int(sample_size * self.centrality_fraction)
		uniform_batch_size = sample_size - centrality_batch_size

		# Uniform sampling
		uniform_sampled_edges = torch.multinomial(self.uniform_probs, uniform_batch_size, replacement = False)
		self.strata_mask_hubs[uniform_sampled_edges] = False

		# Centrality-based sampling  
		centrality_sampled_indices = torch.multinomial(self.edge_centrality_scores[self.strata_mask_hubs], centrality_batch_size, replacement=False)
		centrality_sampled_edges = self.positive_edge_idx[self.strata_mask_hubs][centrality_sampled_indices]

		self.positive_batch_indices[:uniform_batch_size] = uniform_sampled_edges
		self.is_uniform_edge[:uniform_batch_size] = True
		self.positive_batch_indices[uniform_batch_size:sample_size] = centrality_sampled_edges
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
		sampled_from_total = self.sample_edges_strata(self.num_sample_from_total)  
		self.unsampled_edges[sampled_from_total] = False  

		# Sampling from unsampled   
		num_unsampled = self.unsampled_edges.sum()

		# Precompute unsampled edge subset and corresponding uniform probs
		unsampled_edge_indices = self.positive_edge_idx[self.unsampled_edges]

		if num_unsampled > self.num_sample_from_unsampled:

			sampled_from_unsampled = torch.multinomial(self.uniform_probs[self.unsampled_edges], self.num_sample_from_unsampled, replacement=False)

			# Fill the remaining slots in the preallocated buffer
			self.positive_batch_indices[self.num_sample_from_total:self.batch_size] = unsampled_edge_indices[sampled_from_unsampled]

			# Track that these unsampled edges have now been used
			original_indices = unsampled_edge_indices[sampled_from_unsampled]
			self.unsampled_edges[original_indices] = False

			# Mark these edges as being uniformly sampled
			self.is_uniform_edge[self.num_sample_from_total:self.batch_size] = True

		else:
			# Fill all unsampled edges
			self.positive_batch_indices[self.num_sample_from_total:self.num_sample_from_total + num_unsampled] = unsampled_edge_indices

			# Mark these edges as being uniformly sampled
			self.is_uniform_edge[self.num_sample_from_total:self.num_sample_from_total + num_unsampled] = True

			# Mark all unsampled edges as used
			self.unsampled_edges.fill_(False) 

			# Reset sampling function to sample from total edges
			self.sampling_fn = self.sample_edges_strata_total

			# Resample remaining from total
			num_resample_from_total = self.batch_size - (self.num_sample_from_total + num_unsampled)
			if num_resample_from_total > 0:
				self.uniform_probs[self.positive_batch_indices[:-num_resample_from_total]] = 0
				resampled_from_total = torch.multinomial(self.uniform_probs, num_resample_from_total, replacement=False)
				
				# Fill the remaining slots in the preallocated buffer
				self.positive_batch_indices[-num_resample_from_total:] = self.positive_edge_idx[resampled_from_total]

				# Mark these edges as being uniformly sampled
				self.is_uniform_edge[-num_resample_from_total:] = True
				self.uniform_probs[self.positive_batch_indices[:-num_resample_from_total]] = 1

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
		uniform_edges = self.is_uniform_edge.nonzero(as_tuple=False).view(-1)
		if self.unsampled_edges.any():
			uniform_edges = uniform_edges[torch.randperm(uniform_edges.size(0), device=uniform_edges.device)]
		supervision_positive_idx = uniform_edges[:self.num_supervision_edges]
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

		if self.soft_negative_labels:
			negative_edge_centrality = self.get_edge_centrality(negative_batch_edges)
			self.supervision_labels[self.num_supervision_edges:] = self.generate_soft_negative_labels(negative_edge_centrality, max_centrality=self.max_neg_edge_centrality, hardness=self.negative_label_hardness)

		if not self.soft_negative_labels and not self.uniform_supervision_weights:
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

ME_loss = Margin_and_Entropy()

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

def process_data(data:Data, model:nn.Module, optimizer:torch.optim.Optimizer, device:torch.device, is_training=False, return_output=False):
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

	edge_probability, edge_weights = model(
		data.node_features,
		message_edges=data.message_edges,
		supervision_edges=data.supervision_edges,
		message_edgewt = data.message_edgewts
	)
	edge_probability = edge_probability.squeeze(-1)

	# Compute losses

	bce_edge_classification_loss = bce_logit_loss(edge_probability,data.supervision_labels,weight=data.supervision_importance)
	mse_edge_weight_loss = mse_loss(edge_weights.squeeze(-1), data.supervision_edgewts)
	margin_and_entropy_loss = ME_loss(edge_probability)

	loss = bce_edge_classification_loss + mse_edge_weight_loss + margin_and_entropy_loss
	
	# loss = calculate_loss(model_output, data, head_weights)
	conditional_backward(loss)

	if is_training:
		optimizer.step()

	if return_output:
		return loss.item(), edge_probability.detach().cpu(), data.supervision_labels.detach().cpu()
	else:
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