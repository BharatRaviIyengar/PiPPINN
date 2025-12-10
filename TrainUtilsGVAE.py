import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_scatter import scatter_max, scatter_mean, scatter_add, scatter_softmax
from warnings import warn
from pathlib import Path
from max_nbr import max_nbr
from TrainUtils import EdgeSampler

bce_logits_loss = F.binary_cross_entropy_with_logits
cosim = F.cosine_similarity

class SoftplusSimplex(nn.Module):
	def __init__(self, n, init=0.0):
		super().__init__()
		self.raw = nn.Parameter(torch.full((n,), init))

	def forward(self):
		u = F.softplus(self.raw)
		return u / u.sum()


def build_MLP(dims, activation=nn.ReLU, dropout=0.0, use_layernorm=False, normalize_input=False):
	layers = []
	if normalize_input:
		layers.append(nn.LayerNorm(dims[0]))
	for i in range(len(dims) - 1):
		layers.append(nn.Linear(dims[i], dims[i + 1]))
		if use_layernorm:
			layers.append(nn.LayerNorm(dims[i + 1]))
		if i < len(dims) - 1:
			layers.append(activation())
			if dropout > 0:
				layers.append(nn.Dropout(dropout))
	return nn.Sequential(*layers)

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
	def __init__(self, dims = [1,32,32,1], activation=nn.Softplus, epsilon=1e-6):
		"""
		dims: list of layer sizes, e.g., [1, 32, 32, 1] for 1D input/output
		activation: monotone increasing activation
		dropout: dropout rate
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


class NeighborhoodEncoder(nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.0):
		super().__init__()
		self.pool = nn.Linear(in_channels, in_channels)
		self.LN_pool = nn.LayerNorm(in_channels)
		self.in_channels = in_channels
		self.hidden_channels = hidden_channels
		self.dropout = dropout
		self.dims = [self.in_channels] + self.hidden_channels

		self.edge_strength_message_coefficient = nn.Parameter(torch.tensor(0.5))
		self.final_transform = build_MLP(dims=self.dims, dropout=self.dropout, use_layernorm=True, normalize_input=False)

		self.gaussian_mu_head = nn.Linear(self.dims[-1], out_channels)
		self.gaussian_logvar_head = nn.Linear(self.dims[-1], out_channels)

	def forward(self, x, edge_index, edge_strength):
		src, dst = edge_index
		EWMC = F.softplus(self.edge_strength_message_coefficient)
		neighborhood_features = x[src] * (1 + EWMC * edge_strength.unsqueeze(-1))
		pooled = self.pool(neighborhood_features)
		pooled = self.LN_pool(pooled)
		pooled = F.relu(pooled)
		
		# Aggregate neighbor messages via max
		aggregate, _ = scatter_max(pooled, dst, dim=0, dim_size=x.size(0))
		neighborhood = self.final_transform(aggregate)
		mu = self.gaussian_mu_head(neighborhood)
		logvar = self.gaussian_logvar_head(neighborhood)
		std = torch.exp(0.5 * logvar)
		return mu, std
	
class NodeEncoder(nn.Module):
	def __init__(self, in_channels, out_channels, dropout=0.0):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.dropout = dropout
		self.gaussian_mu_head = nn.Linear(self.in_channels, self.out_channels)
		self.gaussian_logvar_head = nn.Linear(self.in_channels, self.out_channels)
	def forward(self, x):
		mu = self.gaussian_mu_head(x)
		logvar = self.gaussian_logvar_head(x)
		std = torch.exp(0.5 * logvar)
		return mu, std


class Decoder(nn.Module):
	def __init__(self, in_channels, hidden_channels, dropout=0.0):
		super().__init__()
		self.in_channels = in_channels
		self.hidden_channels = hidden_channels
		self.dropout = dropout
		self.dims = [2*self.in_channels] + self.hidden_channels
		self.edge_embedder = build_MLP(dims=self.dims, dropout=self.dropout, use_layernorm=True, normalize_input=False)
		self.edge_wt_head = nn.Linear(self.dims[-1]+1, 1)
		self.edge_prob_head = nn.Linear(self.dims[-1],1)

		# self.coef_matrix = nn.Parameter(torch.randn(in_channels, coef_matrix_cols))
		self.sim_scale = nn.Parameter(torch.tensor(1.0))
		self.sim_shift = nn.Parameter(torch.tensor(0.0))

		self.edge_prob_coefficients = SoftplusSimplex(n=3, init=1.0/3.0)

		self.monotonic_map = MonotoneMap()
	
	def congruence_score(self, nodes_latent, supervision_edges, message_edges, message_edgestr = None, softmax_intensity=10.0):
		msg_src, msg_dst = message_edges
		sup_src, sup_dst = supervision_edges
		device = nodes_latent.device
		N = nodes_latent.size(0)

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
			nodes_latent[expanded_queries],	# X
			nodes_latent[msg_neighbors],	# Z
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
			weights*cosine_sim,
			expanded_queries,
			dim=0,
			dim_size=N
		)

		# Step 9: Map back to supervision edges
		ExistenceByCongruence = congruence_node[sup_src] + congruence_node[sup_dst]

		if message_edgestr is not None:
			weighted_edges = self.monotonic_map(cosine_sim) * message_edgestr[mask]
			edge_strength_node = scatter_mean(
				weighted_edges,
				expanded_queries,
				dim=0,
				dim_size=N
			)
			StrengthByCongruence = edge_strength_node[sup_src] + edge_strength_node[sup_dst]
			return ExistenceByCongruence, StrengthByCongruence
		else:
			return ExistenceByCongruence

	def forward(self, nodes_latent, nbrs_latent, supervision_edges, message_edges=None, message_edgestr=None):
		u, v  = supervision_edges
		additive = nodes_latent[u] + nodes_latent[v]
		multiplicative = nodes_latent[u] * nodes_latent[v]
		# combined = additive * self.coef_matrix @ multiplicative
		# # alternative:
		combined = torch.cat([additive, multiplicative], dim=-1)
		edge_features = self.edge_embedder(combined)
		nbrs_similarity = torch.sum(nbrs_latent[u] * nbrs_latent[v], dim=-1, keepdim=True)
		ExistenceByCongruence, StrengthByCongruence = self.congruence_score(nodes_latent, supervision_edges, message_edges, message_edgestr)
		P_cong = self.sim_scale * (ExistenceByCongruence + self.sim_shift)
		P_nbr = self.sim_scale * (nbrs_similarity + self.sim_shift)
		P_dec = self.edge_prob_head(edge_features)
		edge_prob_logits =  self.edge_prob_coefficients * torch.cat([P_dec, P_nbr, P_cong], dim=-1)
		fractional_contribution = self.edge_prob_coefficients
		edge_strengths = F.relu(self.edge_wt_head(torch.cat([edge_features, nbrs_similarity], dim=-1)))
		return edge_prob_logits, edge_strengths, fraction_node_contribution
	
def reparameterize(mu, std):
	eps = torch.randn_like(std)
	return mu + eps * std


class GVAE_Model(nn.Module):
	def __init__(self, node_in_channels, node_latent_channels,
				 nbr_in_channels, nbr_hidden_channels, nbr_latent_channels,
				 decoder_hidden_channels,
				 dropout=0.0):
		super().__init__()
		self.node_encoder = NodeEncoder(node_in_channels, node_latent_channels, dropout)
		self.neighborhood_encoder = NeighborhoodEncoder(nbr_in_channels, nbr_hidden_channels, nbr_latent_channels, dropout)
		self.decoder = Decoder(node_latent_channels, decoder_hidden_channels, dropout)

	def forward(self, x, supervision_edges, message_edges, message_edgestr):
		# Encode nodes
		node_mu, node_std = self.node_encoder(x)
		nodes_latent = reparameterize(node_mu, node_std)

		# Encode neighborhoods
		nbr_mu, nbr_std = self.neighborhood_encoder(x, message_edges, message_edgestr)
		nbrs_latent = reparameterize(nbr_mu, nbr_std)

		# Decode edges
		edge_probabilities, edge_strengths, edge_explained_by_nodes = self.decoder(
			nodes_latent,
			nbrs_latent,
			supervision_edges
		)
		return edge_probabilities, edge_strengths, edge_explained_by_nodes, node_mu, node_std, nbr_mu, nbr_std
	
def KL_loss(mu, std):
	num_nodes = mu.size(0)
	kld = -0.5 * torch.sum(1 + torch.log(std.pow(2) + 1e-8) - mu.pow(2) - std.pow(2))
	return kld / num_nodes
	

def process_data(data:Data, model:nn.Module, optimizer:torch.optim.Optimizer, device:torch.device, masked_supervision_fraction = 0.0, is_training=False, return_output=False):
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

	mask_supervision = torch.rand(data.supervision_labels.size(), device=device) < masked_supervision_fraction

	# Forward pass

	# Encode
	node_mu, node_std = model.node_encoder(data.node_features)
	nodes_latent = reparameterize(node_mu, node_std)

	nbr_mu, nbr_std = model.neighborhood_encoder(
		data.node_features,
		data.message_edges,
		data.message_edgestrs
	)
	nbr_latent = reparameterize(nbr_mu, nbr_std)

	# Decode train supervision edges

	train_logits, train_edge_strengths, _ = model.decoder(
	nodes_latent,
	nbr_latent,
	data.supervision_train_edges
	)

	train_logits = train_logits.squeeze(-1)
	train_edge_strengths = train_edge_strengths.squeeze(-1)

	# Decode masked supervision edges

	masked_logits, masked_edge_strengths, _ = model.decoder(
		nodes_latent,
		nbr_latent,
		data.supervision_masked_edges
	)

	masked_logits = masked_logits.squeeze(-1)
	masked_edge_strengths = masked_edge_strengths.squeeze(-1)

	# Compute losses

	bce_edge_classification_loss = bce_logits_loss(edge_probability,data.supervision_labels,weight=data.supervision_importance)
	mse_edge_strength_loss = F.mse_loss(edge_strengths.squeeze(-1), data.supervision_edgestrs)
	margin_and_entropy_loss = ME_loss(edge_probability)
	KL_loss_node = KL_loss(node_mu, node_std)
	KL_loss_nbr = KL_loss(nbr_mu, nbr_std)

	loss = bce_edge_classification_loss + mse_edge_strength_loss + margin_and_entropy_loss + KL_loss_node + KL_loss_nbr
	
	# loss = calculate_loss(model_output, data, head_weights)
	conditional_backward(loss)

	if is_training:
		optimizer.step()

	if return_output:
		return loss.item(), edge_probability.detach().cpu(), data.supervision_labels.detach().cpu()
	else:
		return loss.item()