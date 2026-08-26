#include <omp.h>
#include <torch/extension.h>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>

std::tuple<torch::Tensor, torch::Tensor>
restrict_neighborhood(
	torch::Tensor bidirectional_message_edges,
	torch::Tensor node_degrees,
	torch::Tensor edge_strength,
	float intensity = 1.0,
	int64_t max_neighbors = 60,
	int64_t nthreads = 1
) {
	TORCH_CHECK(
		bidirectional_message_edges.dim() == 2 &&
		bidirectional_message_edges.size(0) == 2 &&
		bidirectional_message_edges.dtype() == torch::kInt64,
		"bidirectional_message_edges must be a 2D integer tensor with shape [2, num_edges]"
	);

	TORCH_CHECK(
		node_degrees.dim() == 1,
		"node_degrees must be 1D"
	);

	TORCH_CHECK(
		edge_strength.dim() == 1 &&
		edge_strength.size(0) == bidirectional_message_edges.size(1),
		"edge_strength must be 1D with length num_edges"
	);

	TORCH_CHECK(
		max_neighbors > 0 && nthreads > 0 && std::isfinite(intensity),
		"max_neighbors and nthreads must be positive, intensity must be finite"
	);

	auto int64_options = torch::TensorOptions()
		.device(torch::kCPU)
		.dtype(torch::kInt64);

	auto float_options = torch::TensorOptions()
			.device(torch::kCPU)
			.dtype(torch::kFloat32);

	const auto output_device = bidirectional_message_edges.device();
	const int64_t num_nodes = node_degrees.size(0);
	const int64_t num_edges = bidirectional_message_edges.size(1);

	auto edges = bidirectional_message_edges.cpu().contiguous();
	auto edge_str = edge_strength.to(float_options).contiguous();
	auto degrees = node_degrees.to(int64_options).contiguous();

	

	// Offsets for grouping edges by destination node, used for efficient neighbor sampling
	auto offsets = torch::zeros({degrees.size(0) + 1}, int64_options);
	offsets.slice(0,1).copy_(degrees.cumsum(0));

	// auto src = edges[0];
	// auto dst = edges[1];
	
	// Get raw pointers to the data of the tensors for efficient access in the following computations
	
	const auto* edges_ptr = edges.data_ptr<int64_t>();
	const auto* edge_str_ptr = edge_str.data_ptr<float>();
	const auto* degrees_ptr = degrees.data_ptr<int64_t>();
	const auto* offsets_ptr = offsets.data_ptr<int64_t>();

	const auto* src_ptr = edges_ptr;
	const auto* dst_ptr = edges_ptr + num_edges;

	TORCH_CHECK(
    offsets_ptr[num_nodes] == num_edges,
    "node_degrees must sum to the number of message edges"
	);
	
	auto grouped_edges = torch::empty({num_edges}, int64_options);
	auto cursor = offsets.slice(0, 0, -1).clone(); // Initialize cursor to track the current position for each destination node

	auto* grouped_ptr = grouped_edges.data_ptr<int64_t>();
	auto* cursor_ptr = cursor.data_ptr<int64_t>();

	// Fill grouped edges //

	for (int64_t e=0; e < num_edges; ++e) {
		TORCH_CHECK(
			src_ptr[e] >= 0 && src_ptr[e] < num_nodes,
			"Source node index out of bounds"
		);
		TORCH_CHECK(
			dst_ptr[e] >= 0 && dst_ptr[e] < num_nodes,
			"Destination node index out of bounds"
		);
		const int64_t current_node = dst_ptr[e]; // Get the destination node for the current edge
		const int64_t current_index = cursor_ptr[current_node]; // Get the current index for this destination node in the grouped edges
		grouped_ptr[current_index] = e; // Assign the current edge index to the grouped edges at the current index for this destination node
		cursor_ptr[current_node] += 1; // Move the cursor for this destination node to the next position
	}

	// Generate random numbers for sampling edges based on weights //

	auto uniform_random = torch::rand({num_edges}, float_options); // Generate uniform random numbers for each edge

	auto keys = torch::empty({num_edges}, float_options); // Initialize a tensor to hold the keys for sampling

	const auto* uniform_ptr = uniform_random.data_ptr<float>();
	auto* key_ptr = keys.data_ptr<float>();

	constexpr float min_uniform =	std::numeric_limits<float>::min();

	// Generate keys for sampling based on the uniform random numbers and edge weights. The keys are computed as -log(u) / w, where u is a uniform random number and w is the weight of the edge. This transformation allows for sampling edges based on their weights. This is a common technique in weighted random sampling where the final weight describes the Poisson rate (waiting time) of choosing an edge.

	for (int64_t e=0; e < num_edges; ++e) {
		const int64_t src = src_ptr[e];
		const int64_t dst = dst_ptr[e];

		// The weight of an edge is the centrality of the source node relative to the destination node. This calculation ensures that central destination nodes prefer edges from central source nodes. At the same time, peripheral destination nodes can accept edges from both central and peripheral source nodes. Biologically, this means we collect information about important proteins (e.g. master regulators) preferentially from other important proteins, while also allowing peripheral proteins to receive information from both important and peripheral proteins.

		float weight = static_cast<float>(degrees_ptr[src])/static_cast<float>(std::max<int64_t>(1, degrees_ptr[dst]));

		// Further weigh the edges based on their strengths.
		weight *= edge_str_ptr[e];

		// Control the weight intensity.
		weight = std::pow(weight, intensity);

		TORCH_CHECK(std::isfinite(weight) && weight > 0.0, "Edge weight must be positive and finite");

		float u = std::max(uniform_ptr[e], min_uniform);
		key_ptr[e] = -std::log(u) / weight;

	}

	// Initialize output variables

	auto neighborhood_matrix = torch::full({num_nodes, max_neighbors}, -1, int64_options); // Initialize a neighborhood tensor with -1, indicating no neighbor

	auto neighbor_strength_matrix = torch::full({num_nodes, max_neighbors}, -1.0, float_options); // Initialize a neighborhood weights tensor with -1.0

	auto* neighborhood_ptr = neighborhood_matrix.data_ptr<int64_t>();
	auto* neighbor_strength_ptr = neighbor_strength_matrix.data_ptr<float>();

	#pragma omp parallel for num_threads(nthreads)
	for (int64_t node=0; node < num_nodes; ++node) {

		const int64_t start = offsets_ptr[node];
		const int64_t end = offsets_ptr[node+1];
		const int64_t num_neighbors = end - start;

		auto* group_start = grouped_ptr + start;
		auto* group_end = grouped_ptr + end;

		// This partially sorts the edges such that the first `max_neighbors` edges have the smallest keys, which correspond to the highest weights. This allows us to select the top `max_neighbors` edges for each destination node.
		if (num_neighbors > max_neighbors) {
			std::nth_element(
				group_start,
				group_start + max_neighbors,
				group_end,
				[key_ptr](int64_t a, int64_t b) {
				return key_ptr[a] < key_ptr[b];
			});
		} 

		const int64_t limit = std::min(num_neighbors, max_neighbors);
		const int64_t row_offset = node * max_neighbors;
		for (int64_t col= 0; col < limit; ++col) {
			const int64_t edge_index = grouped_ptr[start + col];
			neighborhood_ptr[row_offset + col] = src_ptr[edge_index]; // Store the source node of the selected edge in the neighborhood matrix
			neighbor_strength_ptr[row_offset + col] = edge_str_ptr[edge_index]; // Store the strength of the selected edge in the neighbor strength matrix
		}
	}
	return std::make_tuple(neighborhood_matrix.to(output_device), neighbor_strength_matrix.to(output_device));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("restrict_neighborhood", &restrict_neighborhood, "Restrict Neighborhood",
		py::arg("bidirectional_message_edges"),
		py::arg("node_degrees"),
		py::arg("edge_strength"),
		py::arg("intensity") = 1.0,
		py::arg("max_neighbors") = 60,
		py::arg("nthreads") = 1
	);
}