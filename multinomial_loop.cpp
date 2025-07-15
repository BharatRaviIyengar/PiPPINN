#include <ATen/ATen.h>
#include <torch/torch.h>
#include <omp.h>
#include <torch/extension.h>
#include <vector>

at::Tensor batch_multinomial_sampling(
	torch::Tensor dst,
	torch::Tensor weights,
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
	
	auto unique_dst = std::get<0>(at::_unique(dst, /*sorted=*/false));
	auto num_dst = unique_dst.size(0);
	

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("batch_multinomial_sampling", &batch_multinomial_sampling, "Batch multinomial sampling (CPU)");
}