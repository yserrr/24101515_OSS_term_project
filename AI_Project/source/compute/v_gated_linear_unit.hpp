#pragma once
#include "v_header.hpp"

V_API const char* v_glu_op_name(v_glu_op op);
V_API  v_unary_op v_get_unary_op(const v_tensor* tensor);
V_API  v_glu_op v_get_glu_op(const v_tensor* tensor);
