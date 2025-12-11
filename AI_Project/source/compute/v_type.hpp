#pragma once
#include <array>
#include "v.hpp"
#include "v_header.hpp"
#include "ggml-common.h"

struct v_type_traits {
  const char* type_name;
  int64_t blck_size;
  int64_t blck_size_interleave; // interleave elements in blocks
  size_t type_size;
  bool is_quantized;
  v_to_float_t to_float;
  v_from_float_t from_float_ref;
};

class v_type_trait {
public:
  std::array<v_type_traits, V_TYPE_COUNT> traits;
  v_type_trait();
};
