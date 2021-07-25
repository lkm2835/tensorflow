#include <iostream>

#include "tensorflow/lite/kmcontext.h"

#include "tensorflow/lite/schema/schema_generated.h"
using namespace std;

KmContext kmcontext;

void KmContext::setContext(TfLiteContext* context, std::vector<int>* execution_plan, 
                    std::vector<std::pair<TfLiteNode, TfLiteRegistration>>* nodes_and_registration) {
    context_ = context;
    execution_plan_ = execution_plan;
    nodes_and_registration_ = nodes_and_registration;
}


 const char* GetOpName(const TfLiteRegistration& op_reg) {
    return tflite::EnumNamesBuiltinOperator()[op_reg.builtin_code];
  }