#ifndef MYPROJECT_V_TENSOR_HPP
#define MYPROJECT_V_TENSOR_HPP
#include <array>
#include "v_common.hpp"

// op params - allocated as int32_t for alignment
struct v_tensor {
  v_data_type type;
  v_backend_buffer* buffer;
  std::array<int64_t, V_MAX_DIMS> ne;
  std::array<int64_t, V_MAX_DIMS> nb;
  std::array<v_tensor*, V_MAX_SRC> src;
  std::array<int32_t, V_NUM_OP_PARAMS> op_params;
  std::array<char, V_MAX_NAME> name;
  v_operation op;
  v_tensor* view_src;
  int32_t flags;
  size_t view_offs;
  void* data;
  char padding[8];
};


#endif //MYPROJECT_V_TENSOR_HPP
