#include <ATen/ATen.h>
#include <torch/torch.h>
#include <omp.h>
#include <torch/extension.h>
#include <vector>
#include <unordered_map>

at::Tensor max_nbr(
	torch::Tensor dst,
	torch::Tensor weights,
	c10::optional<torch::Tensor> mask_opt = c10::nullopt,
	int64_t max_neighbors = 30,
	int64_t nthreads = 1
) {

	TORCH_CHECK(dst.dim() == 1 && weights.dim() == 1, "dst and weights must be 1D");
	TORCH_CHECK(dst.size(0) == weights.size(0), "dst and weights must be same size");
	TORCH_CHECK(dst.dtype() == torch::kInt64, "dst must be int64");
	TORCH_CHECK(weights.dtype() == torch::kFloat32 || weights.dtype() == torch::kFloat64, "weights must be float32/float64");


	torch::Device device = dst.device();
	weights = weights.to(device);

	auto options = torch::TensorOptions().dtype(torch::kBool).device(device);
	torch::Tensor sampled_mask = torch::zeros({dst.size(0)}, options);

	torch::Tensor mask;
	if (mask_opt.has_value()) {
		mask = mask_opt.value();
		TORCH_CHECK(mask.dim() == 1 && mask.size(0) == dst.size(0), "mask must be 1D and match dst");
		TORCH_CHECK(mask.dtype() == torch::kBool, "mask must be a boolean tensor");
		mask = mask.to(device);
	} else {
		mask = torch::ones(dst.size(0), torch::dtype(torch::kBool).device(device));
	}

	auto unique_dst = std::get<0>(at::_unique(dst.masked_select(mask), /*sorted=*/false)).cpu();
	const int64_t num_dst = unique_dst.size(0);

	omp_set_num_threads(nthreads);
	#pragma omp parallel for
	for (int64_t i = 0; i < num_dst; ++i) {
		int64_t d = unique_dst[i].item<int64_t>();
		auto indices = torch::nonzero(dst == d).view(-1);
		if (indices.numel() == 0) continue;

		auto w = weights.index_select(0, indices);
		int64_t k = std::min(max_neighbors, indices.size(0));
		auto sampled = torch::multinomial(w, k, /*replacement=*/false);
		auto global_sampled = indices.index_select(0, sampled);

		sampled_mask.index_fill_(0, global_sampled, true);
	}

	return sampled_mask;
}

std::tuple<at::Tensor, at::Tensor> remap_edges_cuda(at::Tensor edges);

std::tuple<torch::Tensor, torch::Tensor> remap_edges_cpu(torch::Tensor edges){
	auto flat_edges = edges.view(-1);
	auto flat_accessor = flat_edges.accessor<int64_t, 1>();

	std::unordered_map<int64_t, int64_t> node_to_new;
	std::vector<int64_t> new_to_node;

	int64_t next_id = 0;

	// Allocate output
	auto remapped = torch::empty_like(flat_edges);
	auto remap_accessor = remapped.accessor<int64_t, 1>();

	for (int64_t i = 0; i < flat_edges.size(0); ++i) {
		int64_t orig_id = flat_accessor[i];

		// Assign new ID if unseen
		auto it = node_to_new.find(orig_id);
		if (it == node_to_new.end()) {
			node_to_new[orig_id] = next_id;
			new_to_node.push_back(orig_id);
			remap_accessor[i] = next_id;
			++next_id;
		} else {
			remap_accessor[i] = it->second;
		}
	}
	auto options = torch::TensorOptions().dtype(torch::kInt64);
	torch::Tensor mapping = torch::from_blob(new_to_node.data(), {static_cast<int64_t>(new_to_node.size())}, options).clone();
	return {remapped.view({2, -1}), mapping};
}




std::tuple<torch::Tensor, torch::Tensor> remap_edges(torch::Tensor edges) {
	if (edges.is_cuda()) {
		#if defined(WITH_CUDA)
			return remap_edges_cuda(edges);
		#else
			TORCH_CHECK(false, "CUDA not available. Recompile with nvcc.");
		#endif
	}
	return remap_edges_cpu(edges);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("max_nbr", &max_nbr,
		"Maximum neighborhood restriction",
		py::arg("dst"),
		py::arg("weights"),
		py::arg("mask") = py::none(),
		py::arg("max_neighbors") = 30,
		py::arg("nthreads") = 1
	);
	m.def("remap_edges", &remap_edges, "Remap edge indices (CPU/CUDA)");
}

