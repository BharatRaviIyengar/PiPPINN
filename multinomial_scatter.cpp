#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Declaration of CUDA kernel (defined in grouped_multinomial_kernel.cu)
__global__ void grouped_multinomial_kernel(
	const float* __restrict__ edge_probs,
	const int64_t* __restrict__ dst_nodes,
	const int64_t* __restrict__ edge_ids,
	const int64_t* __restrict__ dst_offsets,
	const int k,
	const int num_nodes,
	int64_t* __restrict__ out_samples,
	uint64_t seed);

// Launcher function (calls the kernel)
void grouped_multinomial_launcher(
	torch::Tensor edge_probs,
	torch::Tensor dst_nodes,
	torch::Tensor edge_ids,
	torch::Tensor dst_offsets,
	int k,
	int num_nodes,
	torch::Tensor out_samples,
	uint64_t seed) {

	const int threads = 256;
	const int blocks = (num_nodes + threads - 1) / threads;
	const int max_edges_per_node = 128; // Size of shared memory per block (adjust as needed)

	grouped_multinomial_kernel<<<blocks, threads, max_edges_per_node * sizeof(float)>>>(
		edge_probs.data_ptr<float>(),
		dst_nodes.data_ptr<int64_t>(),
		edge_ids.data_ptr<int64_t>(),
		dst_offsets.data_ptr<int64_t>(),
		k,
		num_nodes,
		out_samples.data_ptr<int64_t>(),
		seed
	);
}

// Python binding function
torch::Tensor grouped_multinomial(
	torch::Tensor edge_probs,
	torch::Tensor dst_nodes,
	torch::Tensor edge_ids,
	torch::Tensor dst_offsets,
	int64_t k,
	int64_t num_nodes,
	uint64_t seed) {

	auto out = torch::full({num_nodes, k}, -1, edge_ids.options()); // Initialize output
	grouped_multinomial_launcher(edge_probs, dst_nodes, edge_ids, dst_offsets, k, num_nodes, out, seed);
	return out;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("grouped_multinomial", &grouped_multinomial, "Grouped Multinomial Sampling (CUDA)");
}
