/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/cl/inference_context.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <iostream>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/lite/delegates/gpu/cl/buffer.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/cl/model_hints.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"
#include "tensorflow/lite/delegates/gpu/cl/selectors/operation_selector.h"
#include "tensorflow/lite/delegates/gpu/cl/selectors/special_selector.h"
#include "tensorflow/lite/delegates/gpu/cl/storage_type_util.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/memory_management.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/add_bias.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/merge_padding_with.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

#include "tensorflow/lite/kmdebug.h"

namespace tflite {
namespace gpu {
namespace cl {

namespace {
bool IsReady(const absl::flat_hash_set<ValueId>& ready_tensors,
             const CLNode& node) {
  for (const ValueId in_id : node.inputs) {
    if (ready_tensors.find(in_id) == ready_tensors.end()) {
      return false;
    }
  }
  return true;
}

std::vector<std::pair<ValueId, TensorDescriptor>> GetCLNodeTensors(
    const CLNode& node) {
  std::vector<std::pair<ValueId, TensorDescriptor>> result;
  result.reserve(node.inputs.size() + node.outputs.size());
  const OperationDef op_def = node.operation->GetDefinition();
  for (int j = 0; j < node.inputs.size(); ++j) {
    result.push_back({node.inputs[j], op_def.src_tensors[j]});
  }
  for (int j = 0; j < node.outputs.size(); ++j) {
    result.push_back({node.outputs[j], op_def.dst_tensors[j]});
  }

  return result;
}

absl::Status MergeCLNodes(CLNode* src, CLNode* dst) {
  for (int j = 1; j < src->inputs.size(); ++j) {
    dst->inputs.push_back(src->inputs[j]);
  }
  dst->outputs[0] = src->outputs[0];
  dst->name += " linked : " + src->name;
  return dst->operation->AddOperation(src->operation.get());
}

void AddUsage(ValueId id, int task_index,
              std::map<ValueId, int2>* usage_records) {
  auto it = usage_records->find(id);
  if (it == usage_records->end()) {
    (*usage_records)[id].x = task_index;
    (*usage_records)[id].y = task_index;
  } else {
    (*usage_records)[id].y = task_index;
  }
}

// returns true if actual memory for this storage type will be allocated with
// clCreateBuffer.
bool IsBufferBased(const TensorStorageType& type) {
  return type == TensorStorageType::BUFFER ||
         type == TensorStorageType::IMAGE_BUFFER;
}

// Generic add is add that have several runtime inputs and they are not
// broadcasted, i.e. pointwise add for N tensors where N > 1.
bool IsGenericAdd(const Node& node, const std::vector<Value*>& inputs,
                  const std::vector<Value*>& outputs) {
  if (inputs.size() == 1) {
    return false;
  }
  const OperationType op_type = OperationTypeFromString(node.operation.type);
  if (op_type != OperationType::ADD) {
    return false;
  }

  const auto dst_shape = outputs[0]->tensor.shape;
  for (int i = 0; i < inputs.size(); ++i) {
    const auto src_shape = inputs[i]->tensor.shape;
    if (dst_shape.b != src_shape.b && src_shape.b == 1) {
      return false;
    }
    if (dst_shape.h != src_shape.h && src_shape.h == 1) {
      return false;
    }
    if (dst_shape.w != src_shape.w && src_shape.w == 1) {
      return false;
    }
    if (dst_shape.c != src_shape.c && src_shape.c == 1) {
      return false;
    }
  }
  return true;
}

}  // namespace

CLNode::CLNode(CLNode&& node)
    : operation(std::move(node.operation)),
      inputs(std::move(node.inputs)),
      outputs(std::move(node.outputs)),
      name(std::move(node.name)) {}

CLNode& CLNode::operator=(CLNode&& node) {
  if (this != &node) {
    operation = std::move(node.operation);
    inputs = std::move(node.inputs);
    outputs = std::move(node.outputs);
    name = std::move(node.name);
  }
  return *this;
}

absl::Status InferenceContext::InitFromGraph(
    const CreateInferenceInfo& create_info, const GraphFloat32& graph,
    Environment* env, std::vector<uint8_t>* serialized_model) {
  CreationContext creation_context;
  creation_context.device = env->GetDevicePtr();
  creation_context.context = &env->context();
  creation_context.queue = env->queue();
  creation_context.cache = env->program_cache();

  ReserveGraphTensors(create_info, creation_context.GetDeviceInfo(), graph);
  precision_ = create_info.precision;
  storage_type_ = create_info.storage_type;
  if (env->device().IsMali()) {
    need_flush_ = true;
    need_manual_release_ = true;

    flush_periodically_ = true;
    flush_period_ = 24;
  }
  if (env->device().IsPowerVR()) {
    need_flush_ = true;
  }
  CopyInAndOutIds(graph);
  RETURN_IF_ERROR(ConvertOperations(creation_context.GetDeviceInfo(), graph,
                                    create_info.hints));
  RETURN_IF_ERROR(Merge());
  RETURN_IF_ERROR(AllocateMemory(creation_context.context));
  BindMemoryToOperations();
  RETURN_IF_ERROR(Compile(creation_context));
  RETURN_IF_ERROR(UpdateParams());

  TuningParameters tuning_parameters;
  tuning_parameters.queue = env->profiling_queue();
  tuning_parameters.info = &env->device().info_;
  if (create_info.hints.Check(ModelHints::kFastTuning)) {
    tuning_parameters.tuning_type = TuningType::FAST;
  }
  if (tuning_parameters.info->IsMali()) {
    const MaliInfo& info = tuning_parameters.info->mali_info;
    if (info.IsMaliT6xx()) {
      // Mali T628 hangs forever in clFinish when used profiling queue
      // TuningType::FAST does not use profiling queue.
      tuning_parameters.tuning_type = TuningType::FAST;
    }
  }
  RETURN_IF_ERROR(Tune(tuning_parameters));

  if (serialized_model) {
    flatbuffers::FlatBufferBuilder builder;
    auto encoded_fb = Encode(*this, &builder);
    data::FinishInferenceContextBuffer(builder, encoded_fb);
    serialized_model->resize(builder.GetSize());
    std::memcpy(serialized_model->data(), builder.GetBufferPointer(),
                builder.GetSize());
  }
  for (auto& node : nodes_) {
    node.operation->args_.ReleaseCPURepresentation();
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::RestoreDeserialized(
    const std::vector<uint8_t>& serialized_model, Environment* env) {
  flatbuffers::Verifier verifier(serialized_model.data(),
                                 serialized_model.size());
  if (!data::VerifyInferenceContextBuffer(verifier)) {
    return absl::DataLossError("Deserialization failed.");
  }
  auto decoded_fb = data::GetInferenceContext(serialized_model.data());
  RETURN_IF_ERROR(Decode(&env->context(), decoded_fb, this));

  CreationContext creation_context;
  creation_context.device = env->GetDevicePtr();
  creation_context.context = &env->context();
  creation_context.queue = env->queue();
  creation_context.cache = env->program_cache();

  RETURN_IF_ERROR(AllocateMemory(creation_context.context));
  BindMemoryToOperations();
  for (auto& node : nodes_) {
    RETURN_IF_ERROR(node.operation->CompileDeserialized(creation_context));
  }
  RETURN_IF_ERROR(UpdateParams());
  for (auto& node : nodes_) {
    node.operation->args_.ReleaseCPURepresentation();
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::InitFromGraphWithTransforms(
    const CreateInferenceInfo& create_info, GraphFloat32* graph,
    Environment* env, std::vector<uint8_t>* serialized_model) {
  RETURN_IF_ERROR(RunGraphTransforms(graph));
  RETURN_IF_ERROR(InitFromGraph(create_info, *graph, env, serialized_model));
  return absl::OkStatus();
}

void InferenceContext::CopyInAndOutIds(const GraphFloat32& graph) {
  const auto inputs = graph.inputs();
  for (const auto& input : inputs) {
    input_ids_.push_back(input->id);
  }

  const auto variable_inputs = graph.variable_inputs();
  for (const auto& variable_input : variable_inputs) {
    variable_ids_and_refs_[variable_input->id] = variable_input->tensor.ref;
  }

  const auto outputs = graph.outputs();
  for (const auto& output : outputs) {
    output_ids_.push_back(output->id);
  }
}

void InferenceContext::ReserveGraphTensors(
    const CreateInferenceInfo& create_info, const DeviceInfo& device_info,
    const GraphFloat32& graph) {
  ValueId max_id;
  auto tensors = graph.values();
  auto data_type = DeduceDataTypeFromPrecision(create_info.precision);
  for (auto& t : tensors) {
    TensorStorageType storage_type = create_info.storage_type;
    const auto shape = graph.GetValue(t->id)->tensor.shape;
    Layout layout = shape.b == 1 ? Layout::HWC : Layout::BHWC;
    if (graph.IsGraphInput(t->id) || graph.IsGraphOutput(t->id)) {
      if (shape.c < 4 &&
          CanCreateTensorWithShape(
              device_info, shape,
              TensorDescriptor{data_type, TensorStorageType::SINGLE_TEXTURE_2D,
                               layout})) {
        storage_type = TensorStorageType::SINGLE_TEXTURE_2D;
      }
    }
    storage_type = SelectBestStorageType(device_info, shape, storage_type,
                                         data_type, layout);
    tensor_reserver_.Add(
        t->id, {shape, TensorDescriptor{data_type, storage_type, layout}});
    max_id = std::max(max_id, t->id);
  }
  tensor_reserver_.SetNext(max_id + 1);
}

absl::Status InferenceContext::ConvertOperations(const DeviceInfo& device_info,
                                                 const GraphFloat32& graph,
                                                 ModelHints hints) {
  std::map<ValueId, TensorDescriptor> tensor_descriptors;
  const auto values = graph.values();
  for (auto value : values) {
    tensor_descriptors[value->id] = tensor_reserver_.Get(value->id).descriptor;
  }
  std::set<NodeId> consumed_nodes;
  std::vector<Node*> graph_nodes = graph.nodes();
  std::map<ValueId, int>
      tensor_usages;  // keeps latest index of operation that updated tensor
  for (const auto& input_id : input_ids_) {
    tensor_usages[input_id] = -1;  // so as inputs "updated" before operation 0,
                                   // we will mark them with -1
  }
  for (int i = 0; i < graph_nodes.size(); ++i) {
    const Node& node = *graph_nodes[i];
    if (consumed_nodes.find(node.id) != consumed_nodes.end()) {
      continue;
    }
    std::string op_name = node.operation.type + " " + std::to_string(node.id);
    GPUOperationsSubgraph gpu_subgraph;
    if (hints.Check(ModelHints::kAllowSpecialKernels) &&
        GPUSubgraphFromGraph(device_info, precision_, graph, node.id,
                             tensor_descriptors, &consumed_nodes, &gpu_subgraph,
                             &op_name)
            .ok()) {
      // Mapping of subgraph (set of nodes) to GPU operations. Should happen
      // before straigtforward mapping.
    } else {
      // Straigtforward mapping of one graph node to GPU operations.
      auto inputs = graph.FindInputs(node.id);
      auto outputs = graph.FindOutputs(node.id);
      // Reordering of input ids and updating of temporary tensors_usage struct.
      // This stage is necessary because we are building OperationDef that rely
      // on order of input ids. But we also should have input id on first
      // position that potentially can be "linking" tensor and as result
      // eliminated(unused) We apply it only for ADD operation, because of ADD
      // associativity and ADD can be linked. In current approach "linking"
      // tensor can be only latest written tensor(during linear order of
      // execution) among input tensors.
      if (IsGenericAdd(node, inputs, outputs)) {
        int latest_written_tensor_index = 0;
        int last_usage = tensor_usages[inputs[0]->id];
        for (int j = 1; j < inputs.size(); ++j) {
          if (tensor_usages[inputs[j]->id] > last_usage) {
            last_usage = tensor_usages[inputs[j]->id];
            latest_written_tensor_index = j;
          }
        }
        std::swap(inputs[0], inputs[latest_written_tensor_index]);
      }
      consumed_nodes.insert(node.id);
      OperationDef op_def;
      op_def.precision = precision_;
      for (int j = 0; j < inputs.size(); ++j) {
        op_def.src_tensors.push_back(
            tensor_reserver_.Get(inputs[j]->id).descriptor);
      }
      for (int j = 0; j < outputs.size(); ++j) {
        op_def.dst_tensors.push_back(
            tensor_reserver_.Get(outputs[j]->id).descriptor);
      }
      RETURN_IF_ERROR(GPUOperationFromNode(device_info, op_def, hints, inputs,
                                           outputs, node, &gpu_subgraph));
    }
    absl::flat_hash_map<int, ValueId> mapping_to_global_ids;
    for (int j = 0; j < gpu_subgraph.new_tensors.size(); ++j) {
      const auto& t = gpu_subgraph.new_tensors[j];
      auto global_id = tensor_reserver_.Add({t.first, t.second});
      mapping_to_global_ids[j] = global_id;
    }
    for (auto& gpu_op : gpu_subgraph.operations) {
      CLNode cl_node;
      cl_node.operation = std::move(gpu_op.operation);
      cl_node.inputs.resize(gpu_op.input_ids.size());
      for (int j = 0; j < gpu_op.input_ids.size(); ++j) {
        int id = gpu_op.input_ids[j];
        if (id >= 0) {
          cl_node.inputs[j] = id;
        } else {
          cl_node.inputs[j] = mapping_to_global_ids[-(id + 1)];
        }
      }
      cl_node.outputs.resize(gpu_op.output_ids.size());
      for (int j = 0; j < gpu_op.output_ids.size(); ++j) {
        int id = gpu_op.output_ids[j];
        if (id >= 0) {
          cl_node.outputs[j] = id;
          tensor_usages[id] = i;
        } else {
          cl_node.outputs[j] = mapping_to_global_ids[-(id + 1)];
        }
      }
      cl_node.name = op_name;
      nodes_.push_back(std::move(cl_node));
    }
  }

  return absl::OkStatus();
}

absl::Status InferenceContext::Merge() {
  absl::flat_hash_set<ValueId> ready_tensors;
  for (const auto& input_id : input_ids_) {
    ready_tensors.insert(input_id);
  }
  for (int i = 0; i < nodes_.size(); ++i) {
    auto& node = nodes_[i];
    for (const auto& out_id : node.outputs) {
      ready_tensors.insert(out_id);
    }
    if (node.outputs.size() != 1) {
      continue;
    }
    std::vector<int> next_nodes;
    int link_index = 0;
    for (int j = i + 1; j < nodes_.size(); ++j) {
      for (int k = 0; k < nodes_[j].inputs.size(); ++k) {
        if (nodes_[j].inputs[k] == node.outputs[0]) {
          next_nodes.push_back(j);
          link_index = k;
        }
      }
    }
    if (next_nodes.size() != 1 || link_index != 0) {
      continue;
    }
    auto& linkable_node = nodes_[next_nodes[0]];
    if (!linkable_node.operation->IsLinkable() ||
        linkable_node.outputs.size() != 1 ||
        !IsReady(ready_tensors, linkable_node)) {
      continue;
    }
    const auto& original_dst_def =
        node.operation->GetDefinition().dst_tensors[0];
    const auto& link_dst_def =
        linkable_node.operation->GetDefinition().dst_tensors[0];
    if (original_dst_def != link_dst_def) {
      continue;
    }
    RETURN_IF_ERROR(MergeCLNodes(&linkable_node, &node));
    nodes_.erase(nodes_.begin() + next_nodes[0]);
    i -= 1;
  }
  return absl::OkStatus();
}

void InferenceContext::GetUsages(const std::function<bool(ValueId)>& functor,
                                 std::map<ValueId, int2>* usages) {
  for (ValueId in_id : input_ids_) {
    if (functor(in_id)) {
      AddUsage(in_id, 0, usages);
    }
  }
  for (int op_index = 0; op_index < nodes_.size(); ++op_index) {
    auto tensors = GetCLNodeTensors(nodes_[op_index]);
    for (auto& tensor : tensors) {
      if (functor(tensor.first)) {
        AddUsage(tensor.first, op_index, usages);
      }
    }
  }
  for (ValueId out_id : output_ids_) {
    if (functor(out_id)) {
      AddUsage(out_id, nodes_.size(), usages);
    }
  }
}

InferenceContext::TensorMemoryType InferenceContext::GetTensorMemoryType(
    ValueId id) {
  if (variable_ids_and_refs_.find(id) != variable_ids_and_refs_.end()) {
    return TensorMemoryType::VARIABLE;
  } else if (IsBufferBased(tensor_reserver_.Get(id).descriptor.storage_type)) {
    return TensorMemoryType::BUFFER;
  } else {
    return TensorMemoryType::STRONG_SHAPE;
  }
}

absl::Status InferenceContext::AllocateMemory(CLContext* context) {
  RETURN_IF_ERROR(AllocateMemoryForVariableTensors(context));
  RETURN_IF_ERROR(AllocateMemoryForBuffers(context));
  RETURN_IF_ERROR(AllocateMemoryForStrongShapes(context));
  return absl::OkStatus();
}

absl::Status InferenceContext::AllocateMemoryForVariableTensors(
    CLContext* context) {
  std::map<ValueId, int> ref_value_to_tensor_index;

  for (auto value_and_ref_value : variable_ids_and_refs_) {
    if (ref_value_to_tensor_index.find(value_and_ref_value.second) ==
        ref_value_to_tensor_index.end()) {
      const auto& t = tensor_reserver_.Get(value_and_ref_value.first);
      const auto& shape = t.shape;
      const auto& descriptor = t.descriptor;

      RETURN_IF_ERROR(
          CreateTensor(*context, shape, descriptor,
                       &variable_tensors_[value_and_ref_value.second]));
    }
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::AllocateMemoryForBuffers(CLContext* context) {
  std::map<ValueId, int2> buffer_usages;
  GetUsages(
      [this](ValueId id) {
        return GetTensorMemoryType(id) == TensorMemoryType::BUFFER;
      },
      &buffer_usages);

  std::vector<TensorUsageRecord<size_t>> buffer_usage_records;
  for (auto& usage : buffer_usages) {
    const auto& t = tensor_reserver_.Get(usage.first);
    const auto& shape = t.shape;
    const auto& descriptor = t.descriptor;
    const size_t element_size =
        descriptor.data_type == DataType::FLOAT32 ? 4 : 2;
    const size_t buffer_size =
        shape.b * shape.w * shape.h * AlignByN(shape.c, 4) * element_size;
    graph_ids_to_shared_buffer_tensors_[usage.first] =
        buffer_usage_records.size();
    buffer_usage_records.push_back({buffer_size,
                                    static_cast<TaskId>(usage.second.x),
                                    static_cast<TaskId>(usage.second.y)});
  }

  ObjectsAssignment<size_t> buffer_assignment;
  RETURN_IF_ERROR(AssignObjectsToTensors(
      buffer_usage_records, MemoryStrategy::GREEDY_BEST, &buffer_assignment));

  shared_buffers_.resize(buffer_assignment.object_sizes.size());
  for (int i = 0; i < buffer_assignment.object_sizes.size(); ++i) {
    RETURN_IF_ERROR(CreateReadWriteBuffer(buffer_assignment.object_sizes[i],
                                          context, &shared_buffers_[i]));
  }

  std::vector<bool> created_tensors(buffer_usage_records.size(), false);
  shared_buffer_tensors_.resize(buffer_usage_records.size());
  for (auto& node : nodes_) {
    auto tensors = GetCLNodeTensors(node);
    for (auto& t : tensors) {
      if (GetTensorMemoryType(t.first) != TensorMemoryType::BUFFER) continue;
      const int tensor_index = graph_ids_to_shared_buffer_tensors_[t.first];
      if (created_tensors[tensor_index]) continue;
      const auto& shape = tensor_reserver_.Get(t.first).shape;
      const int buffer_index = buffer_assignment.object_ids[tensor_index];
      RETURN_IF_ERROR(CreateSharedTensor(
          *context, shared_buffers_[buffer_index].GetMemoryPtr(), shape,
          t.second, &shared_buffer_tensors_[tensor_index]));
      created_tensors[tensor_index] = true;
    }
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::AllocateMemoryForStrongShapes(
    CLContext* context) {
  std::map<ValueId, int2> usages;
  GetUsages(
      [this](ValueId id) {
        return GetTensorMemoryType(id) == TensorMemoryType::STRONG_SHAPE;
      },
      &usages);

  std::vector<TensorUsageRecord<DummyTensor>> usage_records;
  std::map<ValueId, ValueId> remap_from_graph_ids;
  for (auto& usage : usages) {
    remap_from_graph_ids[usage.first] = usage_records.size();
    usage_records.push_back({tensor_reserver_.Get(usage.first),
                             static_cast<TaskId>(usage.second.x),
                             static_cast<TaskId>(usage.second.y)});
  }

  ObjectsAssignment<DummyTensor> assignment;
  RETURN_IF_ERROR(AssignObjectsToTensors(
      usage_records, MemoryStrategy::EQUALITY, &assignment));

  for (auto& node : nodes_) {
    auto tensors = GetCLNodeTensors(node);
    for (auto& t : tensors) {
      if (GetTensorMemoryType(t.first) != TensorMemoryType::STRONG_SHAPE) {
        continue;
      }
      const auto& shape = tensor_reserver_.Get(t.first).shape;
      const auto id = assignment.object_ids[remap_from_graph_ids[t.first]];
      graph_ids_to_strong_shape_tensors_[t.first] = id;
      const auto& it = strong_shape_tensors_.find(id);
      if (it == strong_shape_tensors_.end()) {
        RETURN_IF_ERROR(CreateTensor(*context, shape, t.second,
                                     &strong_shape_tensors_[id]));
      }
    }
  }
  return absl::OkStatus();
}

void InferenceContext::BindMemoryToOperations() {
  for (auto& node : nodes_) {
    for (int i = 0; i < node.inputs.size(); ++i) {
      node.operation->SetSrc(GetTensor(node.inputs[i]), i);
    }
    for (int i = 0; i < node.outputs.size(); ++i) {
      node.operation->SetDst(GetTensor(node.outputs[i]), i);
    }
  }
}

absl::Status InferenceContext::Compile(
    const CreationContext& creation_context) {
  for (auto& node : nodes_) {
    RETURN_IF_ERROR(node.operation->Compile(creation_context));
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::Tune(const TuningParameters& tuning_parameters) {
  for (auto& node : nodes_) {
    RETURN_IF_ERROR(node.operation->Tune(tuning_parameters));
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::UpdateParams() {
  for (auto& node : nodes_) {
    RETURN_IF_ERROR(node.operation->UpdateParams());
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::AddToQueue(CLCommandQueue* queue) {
  #ifdef DEBUG
    SFLAG();
  #endif
  if (need_manual_release_) {
    if (prev_enqueue_start_point_.is_valid()) {
      prev_enqueue_start_point_.Wait();
    }
    RETURN_IF_ERROR(queue->EnqueueEvent(&prev_enqueue_start_point_));
  }
  int counter = 0;
  for (auto& node : nodes_) {
    RETURN_IF_ERROR(node.operation->AddToQueue(queue));
    Tensor* t = node.operation->GetSrc()[0];
    std::cout << "addtoQ : " <<t->Channels() << std::endl;
    //std::cout << "addtoQ : " <<t->GetDescriptor().data.size() << std::endl;
    //for (int i = 0; i < 100; ++i)
      //std::cout << "addtoQ : " << *((float*)t->GetMemoryPtr()+i) << std::endl;
    counter++;
    if (flush_periodically_ && counter % flush_period_ == 0) {
      clFlush(queue->queue());
    }
  }
  if (need_flush_) {
    clFlush(queue->queue());
  }
  return absl::OkStatus();
}

absl::Status InferenceContext::Profile(ProfilingCommandQueue* queue,
                                       ProfilingInfo* result) {
  queue->ResetMeasurements();
  for (auto& node : nodes_) {
    queue->SetEventsLabel(node.name);
    RETURN_IF_ERROR(node.operation->AddToQueue(queue));
  }
  RETURN_IF_ERROR(queue->WaitForCompletion());
  *result = queue->GetProfilingInfo();
  return absl::OkStatus();
}

uint64_t InferenceContext::GetSizeOfMemoryAllocatedForIntermediateTensors()
    const {
  uint64_t total_memory = 0;
  for (const auto& t : strong_shape_tensors_) {
    total_memory += t.second.GetMemorySizeInBytes();
  }
  for (const auto& b : shared_buffers_) {
    total_memory += b.GetMemorySizeInBytes();
  }
  for (const auto& t : variable_tensors_) {
    total_memory += t.second.GetMemorySizeInBytes();
  }

  return total_memory;
}

Tensor* InferenceContext::GetTensor(ValueId id) {
  if (variable_ids_and_refs_.find(id) != variable_ids_and_refs_.end()) {
    return &variable_tensors_[variable_ids_and_refs_[id]];
  } else if (graph_ids_to_shared_buffer_tensors_.find(id) !=
             graph_ids_to_shared_buffer_tensors_.end()) {
    return &shared_buffer_tensors_[graph_ids_to_shared_buffer_tensors_[id]];
  } else {
    return &strong_shape_tensors_[graph_ids_to_strong_shape_tensors_[id]];
  }
}

absl::Status InferenceContext::SetInputTensor(ValueId id,
                                              const TensorFloat32& tensor,
                                              CLCommandQueue* queue) {
  return GetTensor(id)->WriteData(queue, tensor);
}

absl::Status InferenceContext::GetOutputTensor(ValueId id,
                                               CLCommandQueue* queue,
                                               TensorFloat32* result) {
  const auto& gpu_tensor = *GetTensor(id);
  const auto dst_shape = BHWC(gpu_tensor.Batch(), gpu_tensor.Height(),
                              gpu_tensor.Width(), gpu_tensor.Channels());
  result->id = id;
  result->shape = dst_shape;
  result->data.resize(dst_shape.DimensionsProduct());
  return gpu_tensor.ReadData(queue, result);
}

absl::Status RunGraphTransforms(GraphFloat32* graph) {
  auto merge_padding_transform = NewMergePaddingWithAdd();
  auto add_bias_transform = NewAddBias();
  ModelTransformer transformer(graph, /*reporter=*/nullptr);
  if (!transformer.Apply("add_bias", add_bias_transform.get())) {
    return absl::InternalError("Invalid add_bias transform");
  }
  if (!transformer.Apply("merge_padding", merge_padding_transform.get())) {
    return absl::InternalError("Invalid merge_padding transform");
  }
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
