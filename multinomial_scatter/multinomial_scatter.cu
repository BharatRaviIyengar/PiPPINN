#include <curand_kernel.h>

#define MAX_K 64

__global__ void grouped_multinomial_kernel(
	const float* __restrict__ edge_probs,
	const int64_t* __restrict__ dst_nodes,
	const int64_t* __restrict__ edge_ids,
	const int64_t* __restrict__ dst_offsets,
	const int k,
	const int num_nodes,
	int64_t* __restrict__ out_samples,
	uint64_t seed)
{
	int node_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (node_id >= num_nodes)
		return;

	int start = dst_offsets[node_id];
	int end = dst_offsets[node_id + 1];
	int len = end - start;

	if (len == 0) {
		for (int i = 0; i < k; ++i)
			out_samples[node_id * k + i] = -1;
		return;
	}

	// Normalize probs and compute cumulative
	extern __shared__ float cum_probs[];
	float total_prob = 0.0f;

	for (int i = 0; i < len; ++i) {
		total_prob += edge_probs[start + i];
	}
	if (total_prob == 0.0f)
		total_prob = 1.0f;

	float running = 0.0f;
	for (int i = 0; i < len; ++i) {
		running += edge_probs[start + i] / total_prob;
		cum_probs[i] = running;
	}

	// Random number generator
	curandState rng;
	curand_init(seed + node_id, 0, 0, &rng);

	for (int s = 0; s < k; ++s) {
		float r = curand_uniform(&rng);
		for (int i = 0; i < len; ++i) {
			if (r < cum_probs[i]) {
				out_samples[node_id * k + s] = edge_ids[start + i];
				break;
			}
		}
	}
}
