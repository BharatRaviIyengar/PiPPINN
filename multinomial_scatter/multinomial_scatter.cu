#include <curand_kernel.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef MAX_K
#define MAX_K 30  // Default, can be overridden at compile time
#endif


__global__ void grouped_multinomial_kernel(
	const float* __restrict__ edge_probs,
	const int64_t* __restrict__ edge_ids,
	const int64_t* __restrict__ dst_offsets,
	const int num_nodes,
	int64_t* __restrict__ out_samples,
	uint64_t seed)
{
	int node_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (node_id >= num_nodes)
		return;

	int start = dst_offsets[node_id];
	int end = dst_offsets[node_id + 1];
	int num_neighbors = end - start;

	int64_t base_idx = node_id * MAX_K;

	// Not enough neighbors to sample without replacement — discard
	if (num_neighbors < MAX_K) {
		#pragma unroll MAX_K
		for (int i = 0; i < MAX_K; ++i) {
			out_samples[base_idx + i] = -1;
		}
		return;
	}

	extern __shared__ float cum_probs[];

	float total_prob = 0.0f;
	for (int i = 0; i < num_neighbors; ++i) {
		total_prob += edge_probs[start + i];
	}
	if (total_prob == 0.0f)
		total_prob = 1.0f;

	float running = 0.0f;
	for (int i = 0; i < num_neighbors; ++i) {
		running += edge_probs[start + i] / total_prob;
		cum_probs[i] = running;
	}

	curandState rng;
	curand_init(seed + node_id, 0, 0, &rng);

	#pragma unroll MAX_K
	for (int sample_i = 0; sample_i < MAX_K; sample_i++) {
		float r = curand_uniform(&rng);

		// Binary search CDF to find index
		int left = 0, right = num_neighbors - 1;
		while (left < right) {
			int mid = (left + right) / 2;
			if (r <= cum_probs[mid]) right = mid;
			else left = mid + 1;
		}
		int chosen = left;

		out_samples[base_idx + sample_i] = edge_ids[start + chosen];

		// Remove chosen probability and rebuild CDF
		float removed_prob = cum_probs[chosen] - (chosen == 0 ? 0 : cum_probs[chosen - 1]);
		for (int i = chosen; i < num_neighbors - 1; i++) {
			cum_probs[i] = cum_probs[i+1];
		}
		num_neighbors--;

		// Renormalize remaining CDF
		float total_prob_new = 1.0f - removed_prob;
		if (total_prob_new <= 0.0f) {
			// no probs left, fill remaining samples with -1
			#pragma unroll MAX_K
			for (int j = sample_i + 1; j < MAX_K; j++) {
				out_samples[base_idx + j] = -1;
			}
			break;
		}
		for (int i = 0; i < num_neighbors; i++) {
			float prob_i = cum_probs[i] - (i == 0 ? 0 : cum_probs[i-1]);
			prob_i /= total_prob_new;
			cum_probs[i] = (i == 0 ? 0 : cum_probs[i-1]) + prob_i;
		}
	}
}

void grouped_multinomial_gpu(
	torch::Tensor edge_probs,
	torch::Tensor edge_ids,
	torch::Tensor dst_offsets,
	int num_nodes,
	int max_neighbors,
	torch::Tensor out_samples,
	uint64_t seed)
{
	const int threads = 256;
	const int blocks = (num_nodes + threads - 1) / threads;

	size_t shared_mem_bytes = max_neighbors * sizeof(float);

	grouped_multinomial_kernel<<<blocks, threads, shared_mem_bytes>>>(
		edge_probs.data_ptr<float>(),
		edge_ids.data_ptr<int64_t>(),
		dst_offsets.data_ptr<int64_t>(),
		num_nodes,
		out_samples.data_ptr<int64_t>(),
		seed
	);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
		throw std::runtime_error("grouped_multinomial_kernel launch failed");
	}
}
