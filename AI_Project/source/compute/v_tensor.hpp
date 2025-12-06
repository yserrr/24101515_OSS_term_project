#ifndef MYPROJECT_MML_TENSOR_HPP
#define MYPROJECT_MML_TENSOR_HPP
#include <array>
#define V_MAX_OP_PARAMS      64
#define V_MAX_DIMS           4
#ifndef v_MAX_NAME
#   define v_MAX_NAME        64
#endif
#define v_MAX_SRC            10

enum v_tensor_flag
{
  TENSOR_FLAG_INPUT  = 1, // ...is an input for the GGML compute graph
  TENSOR_FLAG_OUTPUT = 2, // ...is an output for the GGML compute graph
  TENSOR_FLAG_PARAM  = 4, // ...contains trainable parameters
  TENSOR_FLAG_LOSS   = 8, // ...defines loss for numerical optimization (multiple loss tensors add up)
};

struct v_tensor
{
  v_data_type type;
  struct v_backend_buffer* buffer;
  int64_t ne[V_MAX_DIMS];
  size_t nb[V_MAX_DIMS];
  v_operation op;
  // op params - allocated as int32_t for alignment
  int32_t op_params[V_MAX_OP_PARAMS / sizeof(int32_t)];
  int32_t flags;
   v_tensor* src[v_MAX_SRC];
  // source tensor and offset for views
  struct v_tensor* view_src;
  size_t view_offs;
  void* data;
  char name[v_MAX_NAME];
  void* extra; // extra things e.g. for ggml-cuda.cu
  char padding[8];
};
using v_tensor_t = v_tensor*;
#endif //MYPROJECT_MML_TENSOR_HPP
