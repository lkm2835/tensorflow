/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/graph_info.h"

#include <algorithm>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"

#include "tensorflow/lite/kmdebug.h"

namespace tflite {
namespace {

// Helper class that actually performs partitioning by node sub set.
// Outputs to a provided `NodeSubset` structure.
//
// Example usage:
// PartitionGraphIntoIndependentNodeSubsetsImpl partitioner(
//     info, nodes_to_part, node_subsets);
// partitioner.Partition();
//
// NOTE: Changing the partitioning logic would require a change to
// FP16GraphPartitionHelper.
// LINT.IfChange
class PartitionGraphIntoIndependentNodeSubsetsImpl {
 public:
  PartitionGraphIntoIndependentNodeSubsetsImpl(
      const GraphInfo* info, const TfLiteIntArray* nodes_to_partition,
      std::vector<NodeSubset>* node_subsets)
      : info_(info),
        node_subsets_(node_subsets),
        node_type_(info_->num_total_nodes(), NodeSubset::kTfNonPartition) {
    // Populate the node_type_ map.
    for (auto node_index : TfLiteIntArrayView(nodes_to_partition)) {
      node_type_[node_index] = NodeSubset::kTfPartition;
    }
  }

  // Actually partition the graph.
  void Partition() {
    // Initialize here to make Partition() re-entrant.
    #ifdef DEBUG
      SFLAG();
    #endif
    node_subsets_->clear();
    tensor_epochs_.clear();
    tensor_epochs_.resize(info_->num_tensors(), kEpochAlwaysReady);
    node_epochs_.clear();
    node_epochs_.resize(info_->num_execution_nodes(), kEpochNotReady);
    // Set computed tensors to be kEpochNotReady (initializer set everything to
    // AlwaysReady).
    for (int node_index = 0; node_index < info_->num_execution_nodes();
         node_index++) {
      const TfLiteNode& node = info_->node(node_index);
      for (int output_tensor_index : TfLiteIntArrayView(node.outputs)) {
        tensor_epochs_[output_tensor_index] = kEpochNotReady;
      }
    }

    // Do a graph traversal where each iteration in the loop is an epoch
    // that corresponds to a node sub set that only contains nodes that are of
    // the same node_type_.
    while (true) {
      BuildNodeSubset();
      if (node_subsets_->back().nodes.empty()) {
        node_subsets_->pop_back();
        break;
      }
       //     NodeSubset& test = node_subsets_[0][1];
      //std::cout << "TEST : " << test.input_tensors.size() << std::endl;
    }


    // Mark model outputs as node sub set outputs. All the rest have already
    // been identified.
    for (int output_index : info_->outputs()) {
      int output_epoch = tensor_epochs_[output_index];
      if (output_epoch == kEpochAlwaysReady) {
        // This happens when an input of subgraph is also an output of subgraph.
        continue;
      }
      NodeSubset& output_subset = (*node_subsets_)[output_epoch];
      output_subset.output_tensors.push_back(output_index);
    }
    // Make sure every node sub set's inputs and outputs are unique. Since the
    // list of inputs and outputs is generated in a way that produces
    // duplicates.

    for (NodeSubset& node_subset : *node_subsets_) {
      // Sort and uniquefy using standard library algorithms.
      auto uniquefy = [](std::vector<int>* items) {
        std::sort(items->begin(), items->end());
        auto last = std::unique(items->begin(), items->end());
        items->erase(last, items->end());
      };
      uniquefy(&node_subset.input_tensors);
      uniquefy(&node_subset.output_tensors);
    }
      //NodeSubset& test = node_subsets_[0][1];
     // std::cout << "TEST : " << test.input_tensors.size() << std::endl;
     // std::cout << "TEST : "  << test.output_tensors[0] << std::endl;

  }

 private:
  // Special integer values needed for tensor_epochs_ and node_epochs_.
  enum {
    // The node or tensor is not ready to be assigned an epoch. e.g. a node's
    // inputs have not all been assigned epochs.
    kEpochNotReady = -1,
    // Used for tensor_epochs_. This means that the tensor is always ready.
    // e.g. an input to the whole model or a constant that has no dependencies.
    kEpochAlwaysReady = -2
  };

  // Updates the node at `node_index` in the execution plan and returns true if
  // it is assigned to an epoch. False is returned if the node is already set to
  // an epoch, its inputs are not all assigned to epochs, or if it cannot be
  // assigned to the current epoch since the epoch's node_type doesn't match.
  bool UpdateNode(int node_index) {
    const TfLiteNode& node = info_->node(node_index);
    NodeSubset& current_subset = node_subsets_->back();
    int current_epoch = node_subsets_->size() - 1;
    // Check if node is already done.
    if (node_epochs_[node_index] != kEpochNotReady) {
      return false;
    }
    // See if all dependencies of this node are already assigned to a
    // node sub set.
    for (int input_tensor_index : TfLiteIntArrayView(node.inputs)) {
      if (input_tensor_index != kTfLiteOptionalTensor &&
          tensor_epochs_[input_tensor_index] == kEpochNotReady) {
        return false;
      }
    }

    int original_node_idx = info_->node_index(node_index);
    // When we are starting a new epoch, the first ready node defines
    // the type of that epoch.
    if (current_subset.type == NodeSubset::kTfUnexplored) {
      current_subset.type = node_type_[original_node_idx];
    }
    // The node gets assigned to this epoch if it is the same type as
    // the epoch's assigned type. Note, if this is the current ready
    // node encountered during this epoch, this condition will be
    // automatically true.
    if (current_subset.type == node_type_[original_node_idx]) {
      node_epochs_[node_index] = current_epoch;
      current_subset.nodes.push_back(original_node_idx);
      // All outputs of this node now are assigned to this epoch as
      // well.
      for (int output_tensor_index : TfLiteIntArrayView(node.outputs)) {
        tensor_epochs_[output_tensor_index] = current_epoch;
      }
      // Look at our inputs one more time to update that tensor's
      // epochs' outputs
      for (int input_tensor_index : TfLiteIntArrayView(node.inputs)) {
        if (input_tensor_index == kTfLiteOptionalTensor) {
          continue;
        }
        int input_epoch = tensor_epochs_[input_tensor_index];
        int node_epoch = current_epoch;

        if (input_epoch != node_epoch) {
          current_subset.input_tensors.push_back(input_tensor_index);
          // Set inputs to be outputs of the node sub set where they reside.
          // the if condition makes sure inputs to the whole computation
          // are not included (i.e. those initialized to -2 above).
          if (input_epoch >= 0) {
            NodeSubset& input_subset = (*node_subsets_)[input_epoch];
            input_subset.output_tensors.push_back(input_tensor_index);
          }
        }
      }
      return true;
    } else {
      return false;
    }
  }

  // Completely populates the current node_subset by doing graph traversal
  void BuildNodeSubset() {
    node_subsets_->emplace_back(NodeSubset());
    // loop until no more nodes can be updated.
    while (true) {
      bool did_something = false;
      for (int node_index = 0; node_index < info_->num_execution_nodes();
           node_index++) {
        if (UpdateNode(node_index)) {
          did_something = true;
        }
      }
      if (!did_something) return;
    }
  }

  // Temporary data needed for partitioning.
  const GraphInfo* info_;
  // List of node_subsets to populate
  std::vector<NodeSubset>* node_subsets_;
  // NOTE: This vector contains a place-holder for *all* nodes in the graph, not
  // just ones in the execution plan. This is because nodes_to_partition is
  // passed in as a list of original node indices & not execution plan indices.
  std::vector<NodeSubset::Type> node_type_;
  // Maps from tensor index to the epoch in which it is assigned. Also special
  // negative values of kEpochNotReady if not assigned, kEpochAlwaysReady if it
  // is an input to the whole model or a constant that has no dependencies.
  std::vector<int> tensor_epochs_;
  // Maps from tensor index to the epoch in which it is assigned. Also special
  // negative values of kEpochNotReady if not assigned.
  std::vector<int> node_epochs_;
};
// LINT.ThenChange(//tensorflow/lite/delegates/utils.h)

}  // namespace

TfLiteStatus PartitionGraphIntoIndependentNodeSubsets(
    const GraphInfo* info, const TfLiteIntArray* nodes_to_partition,
    std::vector<NodeSubset>* node_subsets) {
  PartitionGraphIntoIndependentNodeSubsetsImpl(info, nodes_to_partition,
                                               node_subsets)
      .Partition();
  return kTfLiteOk;
}

}  // namespace tflite
