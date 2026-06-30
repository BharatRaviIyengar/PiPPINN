#include <ATen/ATen.h>
#include <torch/torch.h>
#include <omp.h>
#include <torch/extension.h>
#include <vector>
#include <tuple>

// This function samples a maximum of `max_neighbors` neighbors for each unique destination node in `dst`, based on the provided `weights`. It returns a boolean mask indicating which edges are selected.

at::Tensor max_nbr(
	torch::Tensor dst, // Destination nodes for each edge
	torch::Tensor weights, // Weights for each edge, used for sampling
	torch::Tensor violators_mask, // Boolean mask indicating which edges have destination nodes with more than `max_neighbors` edges
	int64_t max_neighbors = 60, // Maximum number of neighbors to sample for each destination node
	int64_t nthreads = 1 // Number of threads to use for parallel processing during maximum neighbor sampling
) {

	TORCH_CHECK(dst.dim() == 1 && weights.dim() == 1, "dst and weights must be 1D");
	TORCH_CHECK(dst.size(0) == weights.size(0), "dst and weights must be same size");
	TORCH_CHECK(dst.dtype() == torch::kInt64, "dst must be int64");
	TORCH_CHECK(weights.dtype() == torch::kFloat32 || weights.dtype() == torch::kFloat64, "weights must be float32/float64");


	torch::Device device = dst.device();
	weights = weights.to(device);

	auto options = torch::TensorOptions().dtype(torch::kBool).device(device);
	torch::Tensor sampled_mask = torch::zeros({dst.size(0)}, options);

	auto unique_dst = std::get<0>(at::_unique(dst.masked_select(violators_mask), /*sorted=*/false)).cpu(); // Get unique destination nodes that violate the max_neighbors constraint
	const int64_t num_dst = unique_dst.size(0);

	omp_set_num_threads(nthreads);
	#pragma omp parallel for
	for (int64_t i = 0; i < num_dst; ++i) {
		int64_t d = unique_dst[i].item<int64_t>();
		auto indices = torch::nonzero(dst == d).view(-1); // Get indices of edges that have destination node `d`
		if (indices.numel() == 0) continue;

		auto w = weights.index_select(0, indices); // Get weights for edges with destination node `d`

		// int64_t k = std::min(max_neighbors, indices.size(0)); WE DONT NEED THIS LINE, WE ALREADY HAVE A VIOLATORS MASK, SO WE KNOW THAT THE NUMBER OF EDGES IS GREATER THAN MAX_NEIGHBORS

		auto sampled = torch::multinomial(w, max_neighbors, /*replacement=*/false); // Sample `max_neighbors` neighbors based on weights
		auto global_sampled = indices.index_select(0, sampled); // Get the global indices of the sampled neighbors
		sampled_mask.index_fill_(0, global_sampled, true); // Update the sampled_mask to indicate which edges are selected
	}

	return sampled_mask.to(device); // Return the sampled mask on the same device as the input tensors
}

at::Tensor generate_neighborhood(
	int64_t num_nodes_in_batch,
	torch::Tensor message_edges,
    c10::optional<torch::Tensor> edge_weights_opt = c10::nullopt,
	int64_t max_neighbors = 60,
	int64_t nthreads = 1
){	
	auto message_edges_cpu = message_edges.cpu();

	int64_t maxindex = message_edges_cpu.max().item<int64_t>();
	TORCH_CHECK(maxindex < num_nodes_in_batch, "message_edges contains indices that exceed the number of nodes in the batch");

	torch::Tensor neighborhood = torch::full({num_nodes_in_batch, max_neighbors}, -1, torch::dtype(torch::kInt64).device(message_edges_cpu.device())); // Initialize neighborhood tensor with -1, indicating no neighbor

    if (edge_weights_opt.has_value()) {
        auto edge_weights = edge_weights_opt.value();
        TORCH_CHECK(
            edge_weights.dim() == 1 &&
            edge_weights.size(0) == message_edges_cpu.size(1),
            "edge_weights must be 1D with length num_edges"
        );
        torch::Tensor edge_weight_matrix = torch::full({num_nodes_in_batch, max_neighbors}, -1.0, torch::dtype(torch::kFloat32).device(message_edges_cpu.device())); // Initialize edge weight matrix with -1.0
    }

	torch::Tensor current_index = torch::zeros({num_nodes_in_batch}, torch::dtype(torch::kInt64).device(message_edges_cpu.device())); // Initialize a tensor to keep track of the current neighbor index for each node in the batch

	for (int64_t i = 0; i < message_edges_cpu.size(1); ++i) {
		int64_t src = message_edges_cpu[0][i].item<int64_t>();
		int64_t j = current_index[src].item<int64_t>();
		if (j < max_neighbors){
			neighborhood[src][j] = message_edges_cpu[1][i];
            if (edge_weights_opt.has_value()) {
                edge_weight_matrix[src][j] = edge_weights[i];
            }
			current_index[src] += 1;
		}
		
	}

    if (edge_weights_opt.has_value()) {
        return std::make_tuple(neighborhood.to(message_edges.device()), edge_weight_matrix.to(message_edges.device()));
    }
    else {
        return std::make_tuple(neighborhood.to(message_edges.device()), torch::Tensor());
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
restrict_neighborhood(
    torch::Tensor bidirectional_message_edges,
    torch::Tensor node_degrees,
    torch::Tensor final_message_mask,
    c10::optional<torch::Tensor> edge_weights_opt = c10::nullopt,
    double intensity = 1.0,
    int64_t max_neighbors = 60,
    int64_t nthreads = 1
) {
    TORCH_CHECK(
        bidirectional_message_edges.dim() == 2 &&
        bidirectional_message_edges.size(0) == 2,
        "bidirectional_message_edges must have shape [2, num_edges]"
    );

    TORCH_CHECK(
        bidirectional_message_edges.dtype() == torch::kInt64,
        "bidirectional_message_edges must be int64"
    );

    TORCH_CHECK(
        node_degrees.dim() == 1,
        "node_degrees must be 1D"
    );

    TORCH_CHECK(
        final_message_mask.dim() == 1 &&
        final_message_mask.size(0) == bidirectional_message_edges.size(1),
        "final_message_mask must be 1D with length num_edges"
    );

    TORCH_CHECK(
        final_message_mask.dtype() == torch::kBool,
        "final_message_mask must be bool"
    );

    auto src = bidirectional_message_edges[0];
    auto dst = bidirectional_message_edges[1];

    auto deg_src = node_degrees.index_select(0, src).to(torch::kFloat32);;
    auto deg_dst = node_degrees.index_select(0, dst).to(torch::kFloat32);;

    int64_t num_nodes_in_batch = node_degrees.size(0);

    auto weights = deg_src / deg_dst.clamp_min(1.0);

    if (edge_weights_opt.has_value()) {
        auto edge_weights = edge_weights_opt.value();

        TORCH_CHECK(
            edge_weights.dim() == 1 &&
            edge_weights.size(0) == src.size(0),
            "edge_weights must be 1D with length num_edges"
        );

        weights = weights * edge_weights;
    }

    weights = weights.pow(intensity);

    auto violators_mask = deg_dst > max_neighbors;

    final_message_mask.fill_(false);

    // Keep all non-violating edges.
    final_message_mask.masked_fill_(~violators_mask, true);

    // Sample among violating destination nodes.
    auto sampled_mask = max_nbr(
        dst,
        weights,
        violators_mask,
        max_neighbors,
        nthreads
    );

    final_message_mask.logical_or_(sampled_mask);

    // Boolean column filtering: [2, E] -> [2, E_restricted]
    auto final_message_edges =
        bidirectional_message_edges.index(
            {torch::indexing::Slice(), final_message_mask}
        );

    auto neighborhood_data = generate_neighborhood(
        num_nodes_in_batch,
        final_message_edges,
        max_neighbors,
        nthreads
    ).to(node_degrees.device());

    neighborhood = std::get<0>(neighborhood_data);
    neighborhood_weights = std::get<1>(neighborhood_data);

    return std::make_tuple(
        final_message_edges,
        neighborhood,
        neighborhood_weights
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("restrict_neighborhood", &restrict_neighborhood, "Restrict Neighborhood",
		py::arg("bidirectional_message_edges"),
		py::arg("node_degrees"),
		py::arg("final_message_mask"),
		py::arg("edge_weights") = py::none(),
		py::arg("intensity") = 1.0,
		py::arg("max_neighbors") = 60,
		py::arg("nthreads") = 1
	);
}