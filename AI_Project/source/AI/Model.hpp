//
// Created by dlwog on 25. 11. 8..
//

#ifndef MYPROJECT_MODEL_HPP
#define MYPROJECT_MODEL_HPP
#include "v.h"
#include "ggml-common.h"
#include "v_opt_common.hpp"

class Model
{
public:
  struct v_tensor* weight;
  struct v_tensor* input;
  v_backend_t backend = nullptr;
  v_backend_buffer_t buffer;
  struct v_ctx* ctx;
  void create();
};


#endif //MYPROJECT_MODEL_HPP
