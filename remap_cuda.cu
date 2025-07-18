#include <torch/extension.h>
#include <unordered_map>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ inline uint32_t hash(int32_t key, int32_t TABLE_SIZE) {
	uint64_t h = ((uint64_t)key * 0x9e3779b97f4a7c15ULL) >> 5;
	return static_cast<uint32_t>(h & (TABLE_SIZE - 1));
}

__global__ void remap_kernel(
    const int64_t* in_data,
    int64_t* out_data,
    int32_t* table_keys,
    int32_t* table_vals,
    int32_t* counter,
    int64_t N,
    int32_t TABLE_SIZE
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int32_t node_id = static_cast<int32_t>(in_data[idx]);
    uint32_t h = hash(node_id, TABLE_SIZE);

    while (true) {
        int32_t old = atomicCAS(&table_keys[h], -1, node_id);
        if (old == -1 || old == node_id) {
            if (old == -1) {
                table_vals[h] = atomicAdd(counter, 1);
            }
            out_data[idx] = static_cast<int64_t>(table_vals[h]);
            break;
        }
        h = (h + 1) & (TABLE_SIZE - 1);
    }
}





std::tuple<torch::Tensor, torch::Tensor> remap_edges_cuda(torch::Tensor edges) {
	auto flat_edges = edges.view(-1);  // [2E]
	int64_t N = flat_edges.size(0);

	auto remapped = torch::empty_like(flat_edges);

	const int32_t TABLE_SIZE = 1 << 30;

	// Allocate hash table (device)
	auto table_keys = torch::full({TABLE_SIZE}, -1, torch::dtype(torch::kInt32).device(edges.device()));
    auto table_vals = torch::zeros({TABLE_SIZE}, torch::dtype(torch::kInt32).device(edges.device()));
    auto counter = torch::zeros({1}, torch::dtype(torch::kInt32).device(edges.device()));

	const int threads = 256;
	const int blocks = (N + threads - 1) / threads;

	remap_kernel<<<blocks, threads>>>(
		flat_edges.data_ptr<int64_t>(),
		remapped.data_ptr<int64_t>(),
		table_keys.data_ptr<int32_t>(),
		table_vals.data_ptr<int32_t>(),
		counter.data_ptr<int32_t>(),
		N,
		TABLE_SIZE
	);

	cudaDeviceSynchronize();  // optional: ensure completion

    auto mapping = torch::empty(counter.item<int32_t>(), torch::dtype(torch::kInt64).device(edges.device()));

    auto keys_cpu = table_keys.cpu();
    auto vals_cpu = table_vals.cpu();

    std::vector<int64_t> id_to_node(counter.item<int32_t>(), -1);  // int64_t for original node IDs
    for (int32_t i = 0; i < TABLE_SIZE; ++i) {
        if (vals_cpu[i].item<int32_t>() >= 0 && keys_cpu[i].item<int32_t>() >= 0) {
            id_to_node[vals_cpu[i].item<int32_t>()] = static_cast<int64_t>(keys_cpu[i].item<int32_t>());
        }
    }


	mapping = torch::from_blob(id_to_node.data(), {static_cast<int64_t>(id_to_node.size())}, torch::dtype(torch::kInt64)).clone();

	return {remapped.view({2, -1}), mapping};

}