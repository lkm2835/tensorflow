#include <iostream>
#include <cstring>

#include "tensorflow/lite/kmcontext.h"

#include "tensorflow/lite/schema/schema_generated.h"
using namespace std;

KmContext kmcontext;

void KmContext::channelPartitioning(vector<pair<int, float>>& layer) {
	vector<int> execution_plan;
	vector<float> ratios;
	for (const auto& l : layer) {
		execution_plan.push_back(l.first);
		ratios.push_back(l.second);
	}

	channelPartitioning(execution_plan, ratios);
}

void KmContext::channelPartitioning(string op_name, float ratio) {
	vector<int> execution_plan;
	vector<float> ratios;
	for (int execution_plan_index = 0;
			execution_plan_index < execution_plan_->size(); execution_plan_index++) {
		int node_index = (*execution_plan_)[execution_plan_index];
		
		TfLiteNode& node = (*nodes_and_registration_)[node_index].first;
		const TfLiteRegistration& registration = (*nodes_and_registration_)[node_index].second;
		
		if (strcmp(GetOpName(registration), op_name.c_str()) == 0) {
			execution_plan.push_back(execution_plan_index);
			ratios.push_back(ratio);
		}
	}
	
	channelPartitioning(execution_plan, ratios);
}

void KmContext::channelPartitioning(std::vector<int>& execution_plan, std::vector<float>& ratios) {
	for (int execution_plan_index = 0;
		 	execution_plan_index < execution_plan.size(); execution_plan_index++) {
		int node_index = execution_plan[execution_plan_index];
		if (!(node_index < nodes_and_registration_->size())) {
			cerr << "[" << node_index << "] layer is not exist." << endl;
			continue;
		}
		TfLiteNode& node = (*nodes_and_registration_)[node_index].first;
		const TfLiteRegistration& registration = (*nodes_and_registration_)[node_index].second;

		if (strcmp(GetOpName(registration), "TfLiteGpuDelegateV2") == 0) {
			// Unimplemented
			cerr << "[" << node_index << "] TfLiteGpuDelegateV2 ... Unimplemented" << endl;
			continue;
		}

		if (strcmp(GetOpName(registration), "CONV_2D") == 0) {
			for (int n = 1; n < node.inputs->size; ++n) {
				int tensor_index = node.inputs->data[n];
				TfLiteTensor& tensor = context_->tensors[tensor_index];
				void** data = &tensor.data.data;
				size_t bytes = tensor.bytes;
				int* dims = (int*)tensor.dims;
				if (n == 1) {
					int o = *(dims + 1);
					int w = *(dims + 2);
					int h = *(dims + 3);
					int i = *(dims + 4);
					int next_filter = w * h * i * ((int)bytes / (o * w * h * i));
					*data += (int)(next_filter * o * (1 - ratios[execution_plan_index])); 
				}
				if (n == 2) {
					int o = *(dims + 1);
					int next_bias = (int)bytes / o;
					*data += (int)(next_bias * o * (1 - ratios[execution_plan_index]));
				}

				*(dims + 1) *= ratios[execution_plan_index]; 
			}
			for (int n = 0; n < node.outputs->size; ++n) {
				int tensor_index = node.outputs->data[n];
				TfLiteTensor& tensor = context_->tensors[tensor_index];
				int* dims = (int*)tensor.dims;
				*(dims + 4) *= ratios[execution_plan_index];
			}
		}
		else {
			cerr << "[" << node_index << "] layer have to be CONV_2D or TfLiteGpuDelegateV2" << endl;
			continue;
		}
		
	}
}

void KmContext::printNodeIndex() {
  for (int execution_plan_index = 0;
		 execution_plan_index < execution_plan_->size(); execution_plan_index++) {
		int node_index = (*execution_plan_)[execution_plan_index];
		TfLiteNode& node = (*nodes_and_registration_)[node_index].first;
		const TfLiteRegistration& registration = (*nodes_and_registration_)[node_index].second;
		cout << endl << GetOpName(registration) << endl;

		cout << "input_index  : ";
		for (int i = 0; i < node.inputs->size; ++i) {
			cout << node.inputs->data[i] << " ";
		} cout << endl;
		
		cout << "output_index : ";
		for (int i = 0; i < node.outputs->size; ++i) {
			cout << node.outputs->data[i] << " ";
		} cout << endl;
	}
}

void KmContext::printNodeDims() {
  string input_shape_[3] = { "input_shape  : ",
								             "filter_shape : ",
								             "bias_shape   : " };
	
	string output_shape_[1] = {"output_shape : " };

  for (int execution_plan_index = 0;
		 execution_plan_index < kmcontext.execution_plan_->size(); execution_plan_index++) {
		int node_index = (*execution_plan_)[execution_plan_index];
		TfLiteNode& node = (*nodes_and_registration_)[node_index].first;
		const TfLiteRegistration& registration = (*nodes_and_registration_)[node_index].second;
		cout << endl << GetOpName(registration) << endl;

		for (int i = 0; i < node.inputs->size; ++i) {
			int tensor_index = node.inputs->data[i];
			int* dims = (int*)context_->tensors[tensor_index].dims;
			cout << input_shape_[i] << "[" << tensor_index << "] ->\t";
			for (int j = 1; j <= *dims; ++j) {
				cout << *(dims + j) << "  ";
			} cout << endl;
		}

		for (int i = 0; i < node.outputs->size; ++i) {
			int tensor_index = node.outputs->data[i];
			int* dims = (int*)context_->tensors[tensor_index].dims;
			cout << output_shape_[i] << "[" << tensor_index << "] ->\t";
			for (int j = 1; j <= *dims; ++j) {
				cout << *(dims + j) << "  ";
			} cout << endl;
		}
	}
}

void KmContext::setContext(TfLiteContext* context, std::vector<int>* execution_plan, 
                    std::vector<std::pair<TfLiteNode, TfLiteRegistration>>* nodes_and_registration) {
    context_ = context;
    execution_plan_ = execution_plan;
    nodes_and_registration_ = nodes_and_registration;
}


 const char* GetOpName(const TfLiteRegistration& op_reg) {
    return tflite::EnumNamesBuiltinOperator()[op_reg.builtin_code];
  }
