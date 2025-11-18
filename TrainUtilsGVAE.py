import torch
import torch.nn as nn
from torch.nn.functional import relu as ReLU, binary_cross_entropy_with_logits as bce_logit_loss, binary_cross_entropy as bce_loss, mse_loss, softplus, normalize as Normalize
from torch_geometric.data import Data
from torch_scatter import scatter_max
from warnings import warn
from pathlib import Path
from max_nbr import max_nbr
from TrainUtils import EdgeSampler

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

		self.gaussian_mu_head = nn.Linear(self.dims[-1], out_channels)
		self.gaussian_logvar_head = nn.Linear(self.dims[-1], out_channels)
	
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

	def forward(self, nodes_latent, nbrs_latent, edge_index):
		u, v  = edge_index
		additive = nodes_latent[u] + nodes_latent[v]
		multiplicative = nodes_latent[u] * nodes_latent[v]
		# combined = additive * self.coef_matrix @ multiplicative
		# # alternative:
		combined = torch.cat([additive, multiplicative], dim=-1)
		edge_features = self.edge_embedder(combined)
		nbrs_similarity = torch.sum(nbrs_latent[u] * nbrs_latent[v], dim=-1, keepdim=True)
		node_contribution = self.edge_prob_head(edge_features) * self.w_node
		nbr_contribution =  nbrs_similarity * self.w_nbr
		edge_probabilities =  node_contribution + nbr_contribution
		fraction_node_contribution = node_contribution / (node_contribution + nbr_contribution + 1e-8)
		edge_weights = ReLU(self.edge_wt_head(torch.cat([edge_features, nbrs_similarity], dim=-1)))
		return edge_probabilities, edge_weights, fraction_node_contribution
	
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

	def forward(self, x, supervision_edges, message_edges, message_edgewt):
		# Encode nodes
		node_mu, node_std = self.node_encoder(x)
		nodes_latent = reparameterize(node_mu, node_std)

		# Encode neighborhoods
		nbr_mu, nbr_std = self.neighborhood_encoder(x, message_edges, message_edgewt)
		nbrs_latent = reparameterize(nbr_mu, nbr_std)

		# Decode edges
		edge_probabilities, edge_weights, edge_explained_by_nodes = self.decoder(
			nodes_latent,
			nbrs_latent,
			supervision_edges
		)
		return edge_probabilities, edge_weights, edge_explained_by_nodes, node_mu, node_std, nbr_mu, nbr_std
	
def KL_loss(mu, std):
	num_nodes = mu.size(0)
	kld = -0.5 * torch.sum(1 + torch.log(std.pow(2) + 1e-8) - mu.pow(2) - std.pow(2))
	return kld / num_nodes
	

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

	edge_probability, edge_weights, edge_explained_by_nodes, node_mu, node_std, nbr_mu, nbr_std = model(
		data.node_features,
		supervision_edges=data.supervision_edges,
		message_edges=data.message_edges,
		message_edgewt = data.message_edgewts
	)
	edge_probability = edge_probability.squeeze(-1)

	# Compute losses

	bce_edge_classification_loss = bce_logit_loss(edge_probability,data.supervision_labels,weight=data.supervision_importance)
	mse_edge_weight_loss = mse_loss(edge_weights.squeeze(-1), data.supervision_edgewts)
	margin_and_entropy_loss = ME_loss(edge_probability)
	KL_loss_node = KL_loss(node_mu, node_std)
	KL_loss_nbr = KL_loss(nbr_mu, nbr_std)

	loss = bce_edge_classification_loss + mse_edge_weight_loss + margin_and_entropy_loss + KL_loss_node + KL_loss_nbr
	
	# loss = calculate_loss(model_output, data, head_weights)
	conditional_backward(loss)

	if is_training:
		optimizer.step()

	if return_output:
		return loss.item(), edge_probability.detach().cpu(), data.supervision_labels.detach().cpu()
	else:
		return loss.item()