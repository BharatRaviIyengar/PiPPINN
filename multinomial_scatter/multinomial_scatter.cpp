#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <vector>

#ifndef MAX_K
#define MAX_K 30  // fallback default
#endif

constexpr int MAX_K_CONST = MAX_K;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> group_edges_by_dst(
	torch::Tensor edges_dst,    // [E], int64, destination node for each edge
	torch::Tensor edges_ids,    // [E], int64, original edge ids
	torch::Tensor edges_probs,  // [E], float, edge probabilities
	int64_t num_nodes
) {
	TORCH_CHECK(edges_dst.dim() == 1 && edges_ids.dim() == 1 && edges_probs.dim() == 1, "All inputs must be 1D");
	TORCH_CHECK(edges_dst.size(0) == edges_ids.size(0) && edges_dst.size(0) == edges_probs.size(0), "All inputs must have same length");
	TORCH_CHECK(edges_dst.dtype() == torch::kInt64 && edges_ids.dtype() == torch::kInt64 && edges_probs.dtype() == torch::kFloat32);

	int64_t num_edges = edges_dst.size(0);

	auto options_int64 = torch::TensorOptions().dtype(torch::kInt64).device(edges_dst.device());
	auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(edges_probs.device());

	torch::Tensor dst_offsets = torch::zeros({num_nodes + 1}, options_int64);  
	torch::Tensor grouped_edge_ids = torch::empty_like(edges_ids);
	torch::Tensor grouped_edge_probs = torch::empty_like(edges_probs);

	auto dst_offsets_acc = dst_offsets.accessor<int64_t,1>();
	auto grouped_edge_ids_acc = grouped_edge_ids.accessor<int64_t,1>();
	auto grouped_edge_probs_acc = grouped_edge_probs.accessor<float,1>();
	auto edges_dst_acc = edges_dst.accessor<int64_t,1>();
	auto edges_ids_acc = edges_ids.accessor<int64_t,1>();
	auto edges_probs_acc = edges_probs.accessor<float,1>();

	// Step 1: Count edges per node
	for (int64_t i = 0; i < num_edges; ++i) {
		int64_t dst = edges_dst_acc[i];
		TORCH_CHECK(dst >= 0 && dst < num_nodes, "dst_nodes out of range");
		dst_offsets_acc[dst + 1]++;
	}

	// Step 2: Prefix sum offsets
	for (int64_t i = 1; i <= num_nodes; ++i) {
		dst_offsets_acc[i] += dst_offsets_acc[i - 1];
	}

	// Step 3: Track current position per dst node
	std::vector<int64_t> current_pos(num_nodes, 0);

	// Step 4: Scatter edges and probs
	for (int64_t i = 0; i < num_edges; ++i) {
		int64_t dst = edges_dst_acc[i];
		int64_t pos = dst_offsets_acc[dst] + current_pos[dst];
		grouped_edge_ids_acc[pos] = edges_ids_acc[i];
		grouped_edge_probs_acc[pos] = edges_probs_acc[i];
		current_pos[dst]++;
	}

	return std::make_tuple(dst_offsets, grouped_edge_ids, grouped_edge_probs);
}


// Forward declaration of CUDA launcher
void grouped_multinomial_gpu(
	torch::Tensor sorted_probs,
	torch::Tensor dst_nodes,
	torch::Tensor sorted_edge_idx,
	torch::Tensor dst_offsets,
	int64_t num_nodes,
	int64_t max_neighbors,
	torch::Tensor out_samples,
	uint64_t seed);

// =======================
// CPU implementation
// =======================
torch::Tensor grouped_multinomial_cpu(
	torch::Tensor sorted_probs,
	torch::Tensor sorted_edge_idx,
	torch::Tensor dst_offsets,
	int64_t num_nodes,
	int64_t k,
	uint64_t seed
) {
	auto options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
	auto out = torch::full({num_nodes, k}, -1, options);

	std::mt19937 rng(seed);

	for (int64_t node = 0; node < num_nodes; ++node) {
		int64_t start = dst_offsets[node].item<int64_t>();
		int64_t end = dst_offsets[node + 1].item<int64_t>();
		int64_t num_edges = end - start;

		if (num_edges == 0) continue;

		std::vector<float> probs(num_edges);
		for (int64_t j = 0; j < num_edges; ++j) {
			probs[j] = sorted_probs[start + j].item<float>();
		}

		float total_prob = std::accumulate(probs.begin(), probs.end(), 0.0f);
		if (total_prob == 0.0f) total_prob = 1.0f;
		for (auto &p : probs) p /= total_prob;

		std::vector<int64_t> indices(num_edges);
		std::iota(indices.begin(), indices.end(), 0);

		for (int64_t sample_i = 0; sample_i < k; ++sample_i) {
			if (indices.empty()) break;

			std::discrete_distribution<int> dist(probs.begin(), probs.end());
			int chosen_idx = dist(rng);

			out[node][sample_i] = sorted_edge_idx[start + indices[chosen_idx]].item<int64_t>();

			indices.erase(indices.begin() + chosen_idx);
			probs.erase(probs.begin() + chosen_idx);

			float sum_prob = std::accumulate(probs.begin(), probs.end(), 0.0f);
			if (sum_prob == 0.0f) break;

			for (auto &p : probs) p /= sum_prob;
		}
	}
	return out;
}



// =======================
// Dispatch wrapper
// =======================
torch::Tensor grouped_multinomial(
	torch::Tensor edge_probs,
	torch::Tensor dst_nodes,
	torch::Tensor edge_ids,
	torch::Tensor dst_offsets,
	int64_t num_nodes,
	uint64_t seed
) {
	auto [dst_offsets, grouped_edge_ids, grouped_edge_probs] = group_edges_by_dst(dst_nodes, edge_ids, edge_probs, num_nodes);
	if (edge_probs.is_cuda()) {
		auto out = torch::full({num_nodes, k}, -1, edge_ids.options());
		grouped_multinomial_gpu(grouped_edge_probs, dst_nodes, grouped_edge_ids, dst_offsets, num_nodes, out, seed);
		return out;
	} else {
		return grouped_multinomial_cpu(grouped_edge_probs, dst_nodes, grouped_edge_ids, dst_offsets, num_nodes, out, seed);
	}
}

// =======================
// Pybind11 binding
// =======================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("grouped_multinomial", &grouped_multinomial, "Grouped Multinomial Sampling (CPU + CUDA)");
}
