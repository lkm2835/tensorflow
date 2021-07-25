#include <vector>
#include "tensorflow/lite/c/common.h"

#define KMCONTEXT() kmcontext.setContext(&context_, &execution_plan_, &nodes_and_registration_)

class KmContext {
  public:
    void setContext(TfLiteContext* context, std::vector<int>* execution_plan, 
                    std::vector<std::pair<TfLiteNode, TfLiteRegistration>>* nodes_and_registration);
    TfLiteContext* context_;
    std::vector<int>* execution_plan_;
    std::vector<std::pair<TfLiteNode, TfLiteRegistration>>* nodes_and_registration_;
};

const char* GetOpName(const TfLiteRegistration& op_reg);

extern KmContext kmcontext;