import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from TrainUtils import build_MLP, generate_hidden_dims

bce_logits_loss = F.binary_cross_entropy_with_logits
cosim = F.cosine_similarity

# PositiveLinear ensures weight >= 0
class PositiveLinear(nn.Module):
	def __init__(self, in_features, out_features, bias=True, epsilon=1e-6):
		super().__init__()
		self.epsilon = epsilon
		self.raw_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
		if bias:
			self.bias = nn.Parameter(torch.zeros(out_features))
		else:
			self.register_parameter('bias', None)

	def forward(self, x):
		# softplus ensures strictly positive weights
		weight = F.softplus(self.raw_weight) + self.epsilon
		return F.linear(x, weight, self.bias)

# MonotoneMap using PositiveLinear layers
class MonotoneMap(nn.Module):
	def __init__(self, dims = [1,8,8,1], activation=nn.Softplus, epsilon=1e-6):
		"""
		dims: list of layer sizes, e.g., [1, 8, 8, 1] for 1D input/output
		activation: monotone increasing activation
		"""
		super().__init__()
		assert dims[0] == 1 and dims[-1] == 1, "Input and output must be 1D"
		self.epsilon = epsilon
		self.activation = activation()
		layers = []
		for i in range(len(dims) - 2):
			layers.append(PositiveLinear(dims[i], dims[i+1], epsilon=epsilon))
		# final layer (still PositiveLinear to ensure monotonicity)
		layers.append(PositiveLinear(dims[-2], dims[-1], epsilon=epsilon))
		self.layers = nn.ModuleList(layers)

	def forward(self, x):
		h = x
		for layer in self.layers[:-1]:
			h = layer(h)
			h = self.activation(h)
		# final layer without activation
		h = self.layers[-1](h)
		return h

class NodeEncoder(nn.Module):
	# Simple MLP encoder that produces Gaussian parameters for each node's latent representation
	def __init__(self, input_dimension, num_layers, output_dimension, dropout=0.0):
		super().__init__()
		self.channels = generate_hidden_dims(input_dimension, num_layers, output_dimension) + [output_dimension]
		self.dropout = dropout
		self.transform = build_MLP(self.channels, dropout=dropout, use_layernorm=True)
		self.gaussian_mu_head = nn.Linear(output_dimension, output_dimension)
		self.gaussian_logvar_head = nn.Linear(output_dimension, output_dimension)

	def forward(self, x):
		x = self.transform(x)
		mu = self.gaussian_mu_head(x)
		logvar = self.gaussian_logvar_head(x)
		std = torch.exp(0.5 * logvar)
		return mu, std
	
class Decoder(nn.Module):
	def __init__(self, in_channels, num_decoder_layers, dropout=0.0):
		super().__init__()
		self.in_channels = in_channels
		self.hidden_channels = [in_channels] * num_decoder_layers
		self.dropout = dropout
		self.dims = [2*self.in_channels] + self.hidden_channels
		self.edge_embedder = build_MLP(dims=self.dims, dropout=self.dropout, use_layernorm=True, normalize_input=False)
		self.edge_wt_head = nn.Linear(self.dims[-1], 1)
		self.edge_prob_head = nn.Linear(self.dims[-1],1)

		# Initialize learnable monotonic non-linear functions that can translate similarity scores to edge probabilities and edge strengths.
		# This ensures that the higher scores always mean higher edge probabilities and strengths, while allowing the model to learn the optimal nonlinear mapping from similarity to edge properties.

		self.monomap_EdgeStrength_Congruence = MonotoneMap()
		self.monomap_EdgeExistence_Congruence = MonotoneMap()
		self.monomap_EdgeExistence_NbrSimilarity = MonotoneMap()

		self.transitivity_sharpness = 1.0
		self.congruence_sharpness = 1.0

	def Transitivity_and_Congruence(self, node_latent, supervision_edges, neighborhood_matrix, neighborhood_strength_matrix):
		'''
		Compute the likelihood of edge existence and strength based on transitivity and congruence.

		Transitivity is determined by neighborhood similarity i.e. if u and v have similar neighbors, they are more likely to be connected.

		Congruence is determined by the similarity of u and v to each other's neighbors i.e. if u is similar to v's neighbors and v is similar to u's neighbors, they are more likely to be connected.

		Inputs:
		- node_latent: Tensor of shape [num_nodes, latent_dim], latent representations of nodes
		- supervision_edges: Tensor of shape [2, num_edges], pairs of nodes for which we want to predict edge existence and strength
		- neighborhood_matrix: Tensor of shape [num_nodes, max_neighbors], indices of neighbors for each node. If a node has fewer than max_neighbors, the remaining entries are filled with -1.
		- neighborhood_strength_matrix: Tensor of shape [num_nodes, max_neighbors], strengths of edges to neighbors for each node. If a node has fewer than max_neighbors, the remaining entries are filled with 0.
		'''

		u, v  = supervision_edges
		nbrs_u = neighborhood_matrix[u] # neighbors of u. Dimension = [num_edges, max_neighbors]
		nbrs_v = neighborhood_matrix[v] # neighbors of v. Dimension = [num_edges, max_neighbors]


		nbrs_u_mask = (nbrs_u != -1) # mask for valid neighbors of u
		nbrs_v_mask = (nbrs_v != -1) # mask for valid neighbors of v

		transitivity_impossible = ~(nbrs_u_mask.any(dim=-1) & nbrs_v_mask.any(dim=-1)) # only consider edges where both u and v have at least one neighbor

		# Replace invalid neighbor indices with 0 to avoid indexing errors 
		nbrs_u_safe = nbrs_u.clamp_min(0)
		nbrs_v_safe = nbrs_v.clamp_min(0)

		latents_Nu = F.normalize(node_latent[nbrs_u_safe], p=2, dim=-1) # latent features of neighbors of u. Normalized along the feature (last) dimension for cosine similarity computation.
		latents_Nv = F.normalize(node_latent[nbrs_v_safe], p=2, dim=-1) # latent features of neighbors of v. Normalized along the feature (last) dimension for cosine similarity computation.

		latents_u	= F.normalize(node_latent[u], p=2, dim=-1).unsqueeze(1) # latent features of u
		latents_v	= F.normalize(node_latent[v], p=2, dim=-1).unsqueeze(1) # latent features of v

		# Calculate pairwise cosine similarity between neighbors of u and neighbors of v (Transitivity)
		
		nsim = torch.bmm(latents_Nu, latents_Nv.transpose(1, 2)) # shape: (num_edges, max_neighbors, max_neighbors)

		# Mask out invalid pairs of neighbors (i.e., where either neighbor is invalid)

		pair_mask = nbrs_u_mask.unsqueeze(2) & nbrs_v_mask.unsqueeze(1) # shape: (num_edges, max_neighbors, max_neighbors)

		nsim.masked_fill_(~pair_mask, float('-inf')) # mask out invalid pairs
		
		# Compute the neighborhood similarity scores for u and v based on their neighbors' similarities
		# LogSumExp is used to aggregate the similarity scores such that higher similarity scores dominate the aggregation, while still allowing for contributions from lower scores.
		# This is biologically reasonable as it allows for the possibility that even if most neighbors are dissimilar, a few highly similar neighbors can still indicate a strong relationship. At the same time, more similar neighbors will increase the score, reflecting the idea that having more similar neighbors strengthens the evidence for a connection.

		Nu_to_Nv = self.transitivity_sharpness * torch.logsumexp(nsim / self.transitivity_sharpness, dim=2)  # [num_edges, max_nbrs]
		Nv_to_Nu = self.transitivity_sharpness * torch.logsumexp(nsim / self.transitivity_sharpness, dim=1)  # [num_edges, max_nbrs]

		num_Nu = nbrs_u_mask.sum(dim=-1).clamp_min(1)
		num_Nv = nbrs_v_mask.sum(dim=-1).clamp_min(1)

		Nu_to_Nv.masked_fill_(~nbrs_u_mask, 0.0)
		Nv_to_Nu.masked_fill_(~nbrs_v_mask, 0.0)

		score_u = Nu_to_Nv.sum(dim=1) / num_Nu
		score_v = Nv_to_Nu.sum(dim=1) / num_Nv

		# Determine the neighborhood score based on the number of neighbors for u and v. The score is taken from the node with fewer neighbors to avoid biasing towards nodes with more neighbors, which could artificially deflate similarity scores.
		neighborhood_score = torch.where(num_Nu <= num_Nv, score_u, score_v)

		# Perform pairwise cosine similarity between u and neighbors of v, and v and neighbors of u (Congruence)

		sim_u_to_Nv = torch.bmm(latents_u, latents_Nv.transpose(1, 2)).squeeze(1) # shape: (num_edges, max_neighbors)
		sim_v_to_Nu = torch.bmm(latents_v, latents_Nu.transpose(1, 2)).squeeze(1) # shape: (num_edges, max_neighbors)

		combined_mask = torch.cat([nbrs_v_mask, nbrs_u_mask], dim=1) # shape: (num_edges, 2*max_neighbors)

		assert combined_mask.any(dim=1).all(), \
    "Congruence calculation requires at least one valid neighbor per supervision edge."

		combined_sim = torch.cat([sim_u_to_Nv, sim_v_to_Nu], dim=1) # shape: (num_edges, 2*max_neighbors)

		combined_edge_strengths = torch.cat([neighborhood_strength_matrix[u], neighborhood_strength_matrix[v]], dim=1) # shape: (num_edges, 2*max_neighbors)
		
		combined_sim.masked_fill_(~combined_mask, float('-inf')) # mask out invalid pairs

		# Compute attention weights using softmax over the combined similarity scores. This allows the model to focus on the most relevant neighbor pairs when aggregating information for edge existence and strength predictions. A softmax is better choice than logsumexp here because we just want to find one good evidence of a congruent neighbor pair, rather than aggregating all the evidence. The softmax will assign higher weights to the most similar pairs, while still allowing for contributions from less similar pairs.

		attention = torch.softmax(combined_sim / self.congruence_sharpness, dim=1)

		combined_sim.masked_fill_(~combined_mask, 0.0) # set invalid pairs to 0 for aggregation

		combined_edge_strengths.masked_fill_(~combined_mask, 0.0) # set invalid pairs to 0 for aggregation

		# Scalar evidence for edge existence
		congruence_score = (attention * combined_sim).sum(dim=1)

		# Strength evidence, preserving the same local congruence structure
		congruence_strength = (attention * combined_edge_strengths).sum(dim=1)

		# Translate the neighborhood similarity and congruence scores into edge existence probabilities and edge strengths using the learnable monotonic mappings. This ensures that higher similarity scores always correspond to higher probabilities and strengths, while allowing the model to learn the optimal nonlinear mapping from similarity to edge properties.
		ExistenceByTransitivity = self.monomap_EdgeExistence_NbrSimilarity(neighborhood_score.unsqueeze(-1)).squeeze(-1)

		ExistenceByTransitivity[transitivity_impossible] = 0.0 # if transitivity is not possible (i.e., one of the nodes has no neighbors), we explicitly set the existence probability to 0.0, as we have no evidence to support the existence of an edge based on transitivity.

		ExistenceByCongruence = self.monomap_EdgeExistence_Congruence(congruence_score.unsqueeze(-1)).squeeze(-1)

		StrengthByCongruence = self.monomap_EdgeStrength_Congruence(congruence_strength.unsqueeze(-1)).squeeze(-1)

		return ExistenceByTransitivity, ExistenceByCongruence, StrengthByCongruence

	def forward(self, nodes_latent, supervision_edges, neighborhood_matrix, neighborhood_strength_matrix):

		u, v  = supervision_edges
		additive = nodes_latent[u] + nodes_latent[v]
		multiplicative = nodes_latent[u] * nodes_latent[v]
		combined = torch.cat([additive, multiplicative], dim=-1)
		edge_features = self.edge_embedder(combined)

		ExistenceByTransitivity, ExistenceByCongruence, StrengthByCongruence = self.Transitivity_and_Congruence(nodes_latent, supervision_edges, neighborhood_matrix, neighborhood_strength_matrix)

		ExistenceViaDecoder = self.edge_prob_head(edge_features).squeeze(-1)

		edge_prob_logits =  ExistenceByCongruence + ExistenceByTransitivity  + ExistenceViaDecoder

		StrengthViaDecoder = F.relu(self.edge_wt_head(edge_features).squeeze(-1))

		edge_strengths = StrengthViaDecoder + StrengthByCongruence

		individual_contributions = {
			"ExistenceByCongruence": ExistenceByCongruence.detach().cpu(),
			"ExistenceByTransitivity": ExistenceByTransitivity.detach().cpu(),
			"ExistenceViaDecoder": ExistenceViaDecoder.detach().cpu(),
			"StrengthByCongruence": StrengthByCongruence.detach().cpu(),
			"StrengthViaDecoder": StrengthViaDecoder.detach().cpu()
		}
		return edge_prob_logits, edge_strengths, individual_contributions
	
def reparameterize(mu, std):
	eps = torch.randn_like(std)
	return mu + eps * std


class GVAE_Model(nn.Module):
	def __init__(self, input_dimension, num_encoder_layers, latent_dimension,num_decoder_layers, dropout=0.0):
		super().__init__()
		self.node_encoder = NodeEncoder(input_dimension, num_encoder_layers, latent_dimension, dropout)
		self.decoder = Decoder(latent_dimension, num_decoder_layers, dropout)

	def forward(self, x, supervision_edges, neighborhood_matrix, neighborhood_strength_matrix):
		# Encode nodes
		node_mu, node_std = self.node_encoder(x)
		nodes_latent = reparameterize(node_mu, node_std)

		# Decode edges
		edge_prob_logits, edge_strengths, edge_contributions = self.decoder(
			nodes_latent,
			supervision_edges,
			neighborhood_matrix,
			neighborhood_strength_matrix
		)
		return edge_prob_logits, edge_strengths, edge_contributions, node_mu, node_std
	
def KL_loss(mu, std):
	# num_nodes = mu.size(0)
	kld = -0.5 * torch.sum(1 + torch.log(std.pow(2) + 1e-8) - mu.pow(2) - std.pow(2))
	return kld / mu.numel()  # Normalize by the number of elements in mu
	

def process_data_GVAE(data:Data, model:nn.Module, optimizer:torch.optim.Optimizer, device:torch.device, mse_coefficient=1.0, kld_coefficient=1.0, is_training=False, return_output=False):
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

	edge_prob_logits, edge_strengths, edge_contributions, node_mu, node_std = model(
    data.node_features,
    data.supervision_edges,
    data.neighborhood_matrix,
    data.neighborhood_weights,
)

	# Compute losses

	bce_edge_classification_loss = bce_logits_loss(edge_prob_logits, data.supervision_labels)
	
	mse_edge_strength_loss = F.mse_loss(
    edge_strengths[:data.num_positive_supervision_edges],
    data.supervision_edgewts,
)
	

	loss = (
		bce_edge_classification_loss +
		mse_coefficient * mse_edge_strength_loss +
		kld_coefficient * KL_loss(node_mu, node_std)
	)

	# loss = calculate_loss(model_output, data, head_weights)
	conditional_backward(loss)

	if is_training:
		optimizer.step()

	if return_output:
		return loss.item(), edge_prob_logits.detach().cpu(), data.supervision_labels.detach().cpu()
	else:
		return loss.item()