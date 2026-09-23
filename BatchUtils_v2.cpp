#include <torch/extension.h>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <vector>
#include <ATen/Parallel.h>

const auto int64_options = torch::TensorOptions()
		.device(torch::kCPU)
		.dtype(torch::kInt64);

const auto float_options = torch::TensorOptions()
		.device(torch::kCPU)
		.dtype(torch::kFloat32);

const auto bool_options = torch::TensorOptions()
		.device(torch::kCPU)
		.dtype(torch::kBool);

class BatchGenerator {
	public:
		BatchGenerator(
			torch::Tensor positive_edges,
			torch::Tensor positive_edge_weights,
			torch::Tensor node_features,
			torch::Tensor node_centrality,
			torch::Tensor negative_edges,
			int64_t batch_size,
			int64_t negative_batch_size,
			double supervision_fraction,
			double centrality_fraction,
			double fraction_from_unsupervised,
			int64_t max_neighbors,
			double neighborhood_intensity,
			double reference_centrality,
			double false_negative_threshold,
			double negative_label_hardness,
			bool track_coverage_once = true
		) :
			// A class member ends in "_". These tensors remain alive for as long as
			// the BatchGenerator object remains alive.
			positive_edges_(positive_edges.detach().cpu().contiguous()),
			positive_edge_weights_(positive_edge_weights.detach().cpu().contiguous()),
			node_features_(node_features.detach().cpu().contiguous()),
			node_centrality_(node_centrality.detach().cpu().contiguous()),
			negative_edges_(negative_edges.detach().cpu().contiguous()),
			batch_size_(batch_size),
			negative_batch_size_(negative_batch_size),
			supervision_fraction_(supervision_fraction),
			centrality_fraction_(centrality_fraction),
			fraction_from_unsupervised_(fraction_from_unsupervised),
			max_neighbors_(max_neighbors),
			neighborhood_intensity_(neighborhood_intensity),
			reference_centrality_(reference_centrality),
			false_negative_threshold_(false_negative_threshold),
			negative_label_hardness_(negative_label_hardness),
			min_centrality_(2.0),
			track_coverage_once_(track_coverage_once)
		{
			validate_inputs();

			num_positive_edges_ = positive_edges_.size(1);
			num_negative_edges_ = negative_edges_.size(1);
			num_nodes_ = node_features_.size(0);

			num_positive_supervision_edges_ =
				static_cast<int64_t>(batch_size_ * supervision_fraction_);
			num_message_edges_ = batch_size_ - num_positive_supervision_edges_;
			num_all_supervision_edges_ =
				num_positive_supervision_edges_ + negative_batch_size_;

			TORCH_CHECK(
				num_positive_supervision_edges_ > 0,
				"supervision_fraction produces zero positive supervision edges"
			);
			TORCH_CHECK(
				(1.0 - centrality_fraction_) >= supervision_fraction_,
				"The uniform fraction must be at least the supervision fraction"
			);
			TORCH_CHECK(
				fraction_from_unsupervised >= 0.0 &&
				fraction_from_unsupervised <= 1.0,
				"fraction_from_unsupervised must be in [0, 1]"
			);

			// Quantities that never change between batches.
			positive_edge_indices_ = torch::arange(num_positive_edges_, int64_options);

			supervision_order_ = torch::randperm(num_positive_edges_, int64_options);

			supervision_cursor_ = 0;

			mandatory_supervision_block_size = static_cast<int64_t>(num_positive_supervision_edges_ * fraction_from_unsupervised_);

			// Fixed-size scratch buffers. Private methods will fill these later.
			sampled_positive_indices_ = torch::empty({batch_size_}, int64_options);

			batch_edges_ = torch::empty(
				{2, batch_size_ + negative_batch_size_}, int64_options
			);

			supervision_labels_ = torch::ones({num_all_supervision_edges_}, float_options);

			positive_edge_strengths_ = torch::empty({batch_size_}, float_options);

			bidirectional_message_edges_ = torch::empty(
				{2, 2 * num_message_edges_}, int64_options
			);
			bidirectional_message_weights_ = torch::empty(
				{2 * num_message_edges_}, float_options
			);

			uniform_negative_wts_ = torch::ones({num_negative_edges_}, float_options);

			num_batch_edges_ = batch_size_ + negative_batch_size_;

			max_nodes_per_batch_ = std::min<int64_t>(
					num_nodes_,
					2 * num_batch_edges_
			);

			nodes_in_batch_.reserve(max_nodes_per_batch_);
			new_labels_.assign(num_nodes_, -1);
			num_nodes_in_batch_ = 0;

			positive_edge_centrality_ = get_edge_centrality(positive_edges_); 

		}

		// Public methods are the operations Python is allowed to request.

		int64_t batch_size() const
		{
			return batch_size_;
		}

		int64_t num_message_edges() const
		{
			return num_message_edges_;
		}

		int64_t num_positive_supervision_edges() const
		{
			return num_positive_supervision_edges_;
		}

		void next_batch();

	private:
		// Private methods and data can only be used by BatchGenerator itself.
		void validate_inputs() const
		{
			TORCH_CHECK(
				positive_edges_.dtype() == torch::kInt64 &&
				positive_edges_.dim() == 2 && positive_edges_.size(0) == 2,
				"positive_edges must be an int64 tensor with shape [2, num_edges]"
			);
			TORCH_CHECK(
				negative_edges_.dtype() == torch::kInt64 &&
				negative_edges_.dim() == 2 && negative_edges_.size(0) == 2,
				"negative_edges must be an int64 tensor with shape [2, num_edges]"
			);
			TORCH_CHECK(
				positive_edge_weights_.dtype() == torch::kFloat32 &&
				positive_edge_weights_.dim() == 1 &&
				positive_edge_weights_.size(0) == positive_edges_.size(1),
				"positive_edge_weights must be float32 with length num_positive_edges"
			);
			TORCH_CHECK(
				node_features_.dtype() == torch::kFloat32 && node_features_.dim() == 2,
				"node_features must be a 2D float32 tensor"
			);
			TORCH_CHECK(
				node_centrality_.dtype() == torch::kFloat32 &&
				node_centrality_.dim() == 1 &&
				node_centrality_.size(0) == node_features_.size(0),
				"node_centrality must be float32 with one value per node"
			);
			TORCH_CHECK(
				positive_edges_.size(1) > 0 && negative_edges_.size(1) > 0,
				"positive_edges and negative_edges must not be empty"
			);
			TORCH_CHECK(
				batch_size_ > 0 && batch_size_ < positive_edges_.size(1),
				"batch_size must be positive and smaller than num_positive_edges"
			);
			TORCH_CHECK(
				negative_batch_size_ > 0 && negative_batch_size_ < negative_edges_.size(1),
				"negative_batch_size must be positive and smaller than num_negative_edges"
			);
			TORCH_CHECK(
				supervision_fraction_ > 0.0 && supervision_fraction_ < 1.0,
				"supervision_fraction must be between 0 and 1"
			);
			TORCH_CHECK(
				centrality_fraction_ >= 0.0 && centrality_fraction_ < 1.0,
				"centrality_fraction must be in [0, 1)"
			);
			TORCH_CHECK(
				max_neighbors_ > 0 &&
				std::isfinite(neighborhood_intensity_),
				"max_neighbors and neighborhood_intensity must be positive; neighborhood_intensity must be finite"
			);

			const int64_t total_nodes = node_features_.size(0);
			TORCH_CHECK(
				positive_edges_.min().item<int64_t>() >= 0 &&
				positive_edges_.max().item<int64_t>() < total_nodes,
				"positive_edges contains a node index outside node_features"
			);
			TORCH_CHECK(
				negative_edges_.min().item<int64_t>() >= 0 &&
				negative_edges_.max().item<int64_t>() < total_nodes,
				"negative_edges contains a node index outside node_features"
			);
		}

		// Immutable graph data.
		torch::Tensor positive_edges_;
		torch::Tensor positive_edge_weights_;
		torch::Tensor node_features_;
		torch::Tensor node_centrality_;
		torch::Tensor negative_edges_;
		torch::Tensor positive_edge_indices_;
		torch::Tensor uniform_negative_wts_;
		torch::Tensor positive_edge_centrality_;

		// Scratch buffers.
		torch::Tensor sampled_positive_indices_;
		torch::Tensor batch_edges_;
		torch::Tensor bidirectional_message_edges_;
		torch::Tensor bidirectional_message_weights_;
		torch::Tensor batch_negative_indices_;
		torch::Tensor supervision_labels_;
		torch::Tensor positive_edge_strengths_;
	
		// Configuration and derived sizes.
		int64_t batch_size_;
		int64_t negative_batch_size_;
		double supervision_fraction_;
		double centrality_fraction_;
		int64_t max_neighbors_;
		double neighborhood_intensity_;
		int64_t num_positive_edges_;
		int64_t num_negative_edges_;
		int64_t num_nodes_;
		int64_t num_positive_supervision_edges_;
		int64_t num_message_edges_;
		int64_t num_all_supervision_edges_;
		int64_t mandatory_supervision_block_size;

		int64_t supervision_cursor_;
		torch::Tensor supervision_order_;

		bool track_coverage_once_;

		double reference_centrality_;
		double min_centrality_;
		double false_negative_threshold_;
		double negative_label_hardness_;

		std::vector<int32_t> new_labels_;
		std::vector<int64_t> nodes_in_batch_;
		int64_t num_batch_edges_;
		int64_t max_nodes_per_batch_;
		int64_t num_nodes_in_batch_;


		void shuffle_order()
		{
			if (track_coverage_once_) {
				return; // Do not reshuffle if we only want to track coverage once
			} else {
				supervision_order_ = torch::randperm(num_positive_edges_, int64_options);
				supervision_cursor_ = 0; // Reset the cursor to the beginning
			}
		}

		torch::Tensor get_edge_centrality(const torch::Tensor& edges) const{
			return (node_centrality_.index_select(0, edges.select(0, 0)) +
				node_centrality_.index_select(0, edges.select(0, 1))).contiguous();
		}

		void sample_negative_edges(){
			batch_negative_indices_ = torch::multinomial(
				uniform_negative_wts_,
				negative_batch_size_,
				false
			);
		};

		void get_negative_edge_data(){
			auto batch_negative_edges = batch_edges_.slice(
				1,
				0,
				negative_batch_size_
				);

			batch_negative_edges.copy_(
					negative_edges_.index_select(1, batch_negative_indices_)
			);

			auto edge_centrality = get_edge_centrality(batch_negative_edges);

			auto normalized_uncertainty = (
					(reference_centrality_ - edge_centrality) /
					(reference_centrality_ - min_centrality_ + 1e-8)
			).clamp(0.0, 1.0);

			auto soft_labels = false_negative_threshold_ * normalized_uncertainty.pow(negative_label_hardness_);

			// Copy labels into the appropriate output section.
			supervision_labels_.slice(
				0,
				0,
				negative_batch_size_
			).copy_(soft_labels);
		}

		void sample_positive_edges();
		std::tuple<torch::Tensor, torch::Tensor> generate_neighborhood();
		torch::Tensor relabel_edges();
};


// The sampling function is still an unfinished draft. Keep it available in
// the source file, but do not compile it until it becomes a private class
// method. This lets us compile and test the class skeleton independently.
#if 0
void BatchGenerator::sample_batch(){
		

		const int64_t random_width = batch_size_ + 1;
		const auto uniform_random = torch::rand({2, random_width}, float_options);
		
		const float* centrality_random_ptr = uniform_random[0].data_ptr<float>();
		const float* shuffle_random_ptr = centrality_random_ptr + random_width;
		const float* centrality_ptr = positive_edge_centrality_.data_ptr<float>();

		torch::Tensor sampled_edge_indices = torch::empty({2, batch_size}, int64_options);

		if (*coverage_incomplete){
			sampled_edge_indices.slice(1, 0, num_SFU).copy_(positive_edges.slice(1, *start_SFU, *start_SFU + num_SFU));
		} else{
			num_SFU = 0;
		}

		auto keys = torch::empty({num_edges}, float_options);
		for(int64_t e=0; e < num_edges; ++e){
			if(e >= start_SFU && e < start_SFU + num_SFU && *coverage_incomplete){
			keys[e] = std::numeric_limits<float>::infinity(); // Assign a very high key to ensure these edges are always excluded
			continue;
			}
			if(rand_uniform_ptr[e] < uniform_bernoulli_probability){
				keys[e] = -rand_weights_ptr[e];
			} else {
				keys[e] = -std::log(std::max(rand_weights_ptr[e], std::numeric_limits<float>::epsilon())) / centrality_ptr[e];
			}
		}

		auto remaining_batch_indices = std::nth_element(
			std::begin(keys),
			std::begin(keys) + (batch_size - num_SFU),
			std::end(keys)
		);

		start_SFU += num_SFU;
		if(start_SFU + num_SFU >= num_edges){
			*coverage_incomplete = false;
		}
		

		return sampled_edge_indices;

}
#endif

torch::Tensor BatchGenerator::relabel_edges(){

	nodes_in_batch_.clear();

	const int64_t num_batch_edges = batch_edges_.size(1);

	auto* edges_ptr = batch_edges_.data_ptr<int64_t>();

	auto* src_ptr = edges_ptr;
	auto* dst_ptr = edges_ptr + num_batch_edges;

	int32_t next_label = 0;

	for (int64_t e = 0; e < num_batch_edges; ++e) {

		const int64_t src = src_ptr[e];
		const int64_t dst = dst_ptr[e];

		TORCH_CHECK(
		src >= 0 && src < num_nodes_ &&
		dst >= 0 && dst < num_nodes_,
		"Node index outside valid range"
		);

		if(new_labels_[src] == -1){
			new_labels_[src] = next_label++;
			nodes_in_batch_.push_back(src);
			num_nodes_in_batch_++;
		}
		if(new_labels_[dst] == -1){
			new_labels_[dst] = next_label++;
			nodes_in_batch_.push_back(dst);
		}

		src_ptr[e] = new_labels_[src_ptr[e]];
		dst_ptr[e] = new_labels_[dst_ptr[e]];
	}

	auto nodes_tensor = torch::from_blob(nodes_in_batch_.data(), {static_cast<int64_t>(nodes_in_batch_.size())}, int64_options).clone();

	for (const int64_t node : nodes_in_batch_) {
    new_labels_[node] = -1;
	}
	
	num_nodes_in_batch_ = static_cast<int64_t>(nodes_in_batch_.size());

	return nodes_tensor;
}

std::tuple<torch::Tensor, torch::Tensor> BatchGenerator::generate_neighborhood() {

	// Generate Bidirectional Message Edges and Weights //

	auto message_edges = batch_edges_.slice(
    1,
    num_all_supervision_edges_,
    num_all_supervision_edges_ + num_message_edges_
	);

	auto message_weights = batch_edges_.slice(
		1,
		num_positive_supervision_edges_,
		batch_size_
	);

	bidirectional_message_edges_.slice(1, 0, num_message_edges_).copy_(message_edges);

	bidirectional_message_edges_.slice(1, num_message_edges_).copy_(message_edges.flip(0));

	bidirectional_message_weights_.slice(0, 0, num_message_edges_).copy_(message_weights);

	bidirectional_message_weights_.slice(0, num_message_edges_).copy_(message_weights);

	auto local_degrees = torch::bincount(
		bidirectional_message_edges_.select(0, 1),
		{},
		num_nodes_in_batch_
	);

	TORCH_CHECK(
		local_degrees.size(0) == num_nodes_in_batch_,
		"Local degrees tensor size exceeds number of nodes"
	);
	
	const int64_t num_edges = bidirectional_message_edges_.size(1); 

	
	// Offsets for grouping edges by destination node, used for efficient neighbor sampling
	auto offsets = torch::zeros({num_nodes_in_batch_ + 1}, int64_options);
	offsets.slice(0,1).copy_(local_degrees.cumsum(0));

	const auto* degrees_ptr = local_degrees.data_ptr<int64_t>();
	
	// Get raw pointers to the data of the tensors for efficient access in the following computations
	
	const auto* edges_ptr = bidirectional_message_edges_.data_ptr<int64_t>();
	const auto* edge_str_ptr = bidirectional_message_weights_.data_ptr<float>();
	const auto* offsets_ptr = offsets.data_ptr<int64_t>();

	const auto* src_ptr = edges_ptr;
	const auto* dst_ptr = edges_ptr + num_edges;

	TORCH_CHECK(
    offsets_ptr[num_nodes_in_batch_] == num_edges,
    "node_degrees must sum to the number of message edges"
	);
	
	auto grouped_edges = torch::empty({num_edges}, int64_options);
	auto cursor = offsets.slice(0, 0, -1).clone(); // Initialize cursor to track the current position for each destination node

	auto* grouped_ptr = grouped_edges.data_ptr<int64_t>();
	auto* cursor_ptr = cursor.data_ptr<int64_t>();

	// Fill grouped edges //

	for (int64_t e=0; e < num_edges; ++e) {
		TORCH_CHECK(
			src_ptr[e] >= 0 && src_ptr[e] < num_nodes_in_batch_,
			"Source node index out of bounds"
		);
		TORCH_CHECK(
			dst_ptr[e] >= 0 && dst_ptr[e] < num_nodes_in_batch_,
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

	at::parallel_for(0, num_edges, 4096, [&](int64_t begin, int64_t end){
		for (int64_t e=begin; e < end; ++e) {
			const int64_t src = src_ptr[e];
			const int64_t dst = dst_ptr[e];

			// The weight of an edge is the centrality of the source node relative to the destination node. This calculation ensures that central destination nodes prefer edges from central source nodes. At the same time, peripheral destination nodes can accept edges from both central and peripheral source nodes. Biologically, this means we collect information about important proteins (e.g. master regulators) preferentially from other important proteins, while also allowing peripheral proteins to receive information from both important and peripheral proteins.

			float weight = static_cast<float>(degrees_ptr[src])/static_cast<float>(std::max<int64_t>(1, degrees_ptr[dst]));

			// Further weigh the edges based on their strengths.
			weight *= edge_str_ptr[e];

			// Control the weight intensity.
			weight = std::pow(weight, intensity);

			TORCH_CHECK(std::isfinite(weight) && weight > 0.0, "Edge weight must be positive and finite");

			const float u = std::max(uniform_ptr[e], min_uniform);
			key_ptr[e] = -std::log(u) / weight;

		}
	});

	// Initialize output variables

	auto neighborhood_matrix = torch::full({num_nodes_in_batch_, max_neighbors}, -1, int64_options); // Initialize a neighborhood tensor with -1, indicating no neighbor

	auto neighbor_strength_matrix = torch::full({num_nodes_in_batch_, max_neighbors}, -1.0, float_options); // Initialize a neighborhood weights tensor with -1.0

	auto* neighborhood_ptr = neighborhood_matrix.data_ptr<int64_t>();
	auto* neighbor_strength_ptr = neighbor_strength_matrix.data_ptr<float>();

	at::parallel_for(0, num_nodes_in_batch_, 64,[&](int64_t begin, int64_t end){
		for (int64_t node = begin; node < end; ++node) {

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
	});

	return std::make_tuple(neighborhood_matrix.to(output_device), neighbor_strength_matrix.to(output_device));
}


void BatchGenerator::next_batch() {
	// This function will be implemented to generate the next batch of edges and their associated data.
	// It will use the private methods defined above to sample positive and negative edges, generate the neighborhood, and relabel edges as necessary.
	// The implementation will ensure that the batch is constructed according to the specified parameters and constraints.
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	py::class_<BatchGenerator>(m, "BatchGenerator")
		.def(
			py::init<
				torch::Tensor,
				torch::Tensor,
				torch::Tensor,
				torch::Tensor,
				torch::Tensor,
				int64_t,
				int64_t,
				double,
				double,
				double,
				int64_t,
				double,
				int64_t
			>(),
			py::arg("positive_edges"),
			py::arg("positive_edge_weights"),
			py::arg("node_features"),
			py::arg("node_centrality"),
			py::arg("negative_edges"),
			py::arg("batch_size"),
			py::arg("negative_batch_size"),
			py::arg("supervision_fraction") = 0.3,
			py::arg("centrality_fraction") = 0.5,
			py::arg("fraction_from_unsampled") = 0.1,
			py::arg("max_neighbors") = 60,
			py::arg("neighborhood_intensity") = 1.0,
			py::arg("nthreads") = 1
		)
		.def("reset_tracking", &BatchGenerator::reset_tracking)
		.def_property_readonly("batch_size", &BatchGenerator::batch_size)
		.def_property_readonly("num_message_edges", &BatchGenerator::num_message_edges)
		.def_property_readonly(
			"num_positive_supervision_edges",
			&BatchGenerator::num_positive_supervision_edges
		);

	m.def("restrict_neighborhood", &restrict_neighborhood, "Restrict Neighborhood",
		py::arg("bidirectional_message_edges"),
		py::arg("node_degrees"),
		py::arg("edge_strength"),
		py::arg("intensity") = 1.0,
		py::arg("max_neighbors") = 60,
		py::arg("nthreads") = 1
	);
		m.def(
		"relabel_edges_", &relabel_edges, "Relabel batch edges in place to have contiguous node indices starting from 0",
		py::arg("original_edges"),
		py::arg("max_nodes_in_graph")
	);
}
