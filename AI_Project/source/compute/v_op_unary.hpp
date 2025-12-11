/// All code is adapted from ggml for personal educational purposes.(study, clone coding)
/// Core code under license is sourced from ggml (https://github.com/ggerganov/ggml)
#pragma once
#include "v_header.hpp"

v_tensor* v_unary(v_ctx* ctx, v_tensor* a, v_unary_op op);
v_tensor* v_unary_inplace(v_ctx* ctx, v_tensor* a, v_unary_op op);
v_tensor* v_unary_impl(v_ctx* ctx, v_tensor* a, v_unary_op op, bool inplace);

const char* v_unary_op_name(v_unary_op op);
v_tensor* v_abs(v_ctx* ctx, v_tensor* a);
v_tensor* v_sgn(v_ctx* ctx, v_tensor* a);
v_tensor* v_neg(v_ctx* ctx, v_tensor* a);
v_tensor* v_step(v_ctx* ctx, v_tensor* a);
v_tensor* v_tanh(v_ctx* ctx, v_tensor* a);
v_tensor* v_elu(v_ctx* ctx, v_tensor* a);
v_tensor* v_relu(v_ctx* ctx, v_tensor* a);
v_tensor* v_leaky_relu(v_ctx* ctx, v_tensor* a, float negative_slope, bool inplace);
v_tensor* v_sigmoid(v_ctx* ctx, v_tensor* a);
v_tensor* v_gelu(v_ctx* ctx, v_tensor* a);
v_tensor* v_cos_impl(v_ctx* ctx, v_tensor* a, bool inplace);
v_tensor* v_sin_impl(v_ctx* ctx, v_tensor* a, bool inplace);



v_tensor* v_sin_inplace(v_ctx* ctx, v_tensor* a);
v_tensor* v_abs_inplace(v_ctx* ctx, v_tensor* a);
v_tensor* v_sgn_inplace(v_ctx* ctx, v_tensor* a);
v_tensor* v_neg_inplace(v_ctx* ctx, v_tensor* a);
v_tensor* v_step_inplace(v_ctx* ctx, v_tensor* a);
v_tensor* v_tanh_inplace(v_ctx* ctx, v_tensor* a);
v_tensor* v_elu_inplace(v_ctx* ctx, v_tensor* a);
v_tensor* v_relu_inplace(v_ctx* ctx, v_tensor* a);
v_tensor* v_sigmoid_inplace(v_ctx* ctx, v_tensor* a);
v_tensor* v_gelu_inplace(v_ctx* ctx, v_tensor* a);
// GELU using erf (error function) when possible
// some backends may fallback to approximation based on Abramowitz and Stegun formula
v_tensor* v_gelu_erf(v_ctx* ctx, v_tensor* a);
v_tensor* v_gelu_erf_inplace(v_ctx* ctx, v_tensor* a);
v_tensor* v_gelu_quick(v_ctx* ctx, v_tensor* a);
v_tensor* v_gelu_quick_inplace(v_ctx* ctx, v_tensor* a);
v_tensor* v_silu(v_ctx* ctx, v_tensor* a);
v_tensor* v_silu_inplace(v_ctx* ctx, v_tensor* a);
v_tensor* v_silu_back(v_ctx* ctx, v_tensor* a, v_tensor* b); // a - x, b - dy
// hardswish(x) = x * relu6(x + 3) / 6
v_tensor* v_hardswish(v_ctx* ctx, v_tensor* a);
// hardsigmoid(x) = relu6(x + 3) / 6
v_tensor* v_hardsigmoid(v_ctx* ctx, v_tensor* a);
v_tensor* v_exp(v_ctx* ctx, v_tensor* a);
v_tensor* v_exp_inplace(v_ctx* ctx, v_tensor* a);
v_tensor* v_floor(v_ctx* ctx, v_tensor* a);
v_tensor* v_floor_inplace(v_ctx* ctx, v_tensor* a);
v_tensor* v_ceil(v_ctx* ctx, v_tensor* a);
v_tensor* v_ceil_inplace(v_ctx* ctx, v_tensor* a);
v_tensor* v_round(v_ctx* ctx, v_tensor* a);
v_tensor* v_round_inplace(v_ctx* ctx, v_tensor* a);

V_API v_tensor* v_sqr(v_ctx* ctx, v_tensor* a);
V_API v_tensor* v_sqr_inplace(v_ctx* ctx, v_tensor* a);
V_API v_tensor* v_sqrt(v_ctx* ctx, v_tensor* a);
V_API v_tensor* v_sqrt_inplace(v_ctx* ctx, v_tensor* a);
V_API v_tensor* v_log(v_ctx* ctx, v_tensor* a);
V_API v_tensor* v_log_inplace(v_ctx* ctx, v_tensor* a);
V_API v_tensor* v_sin(v_ctx* ctx, v_tensor* a);
V_API v_tensor* v_sin_inplace(v_ctx* ctx, v_tensor* a);
V_API v_tensor* v_cos(v_ctx* ctx, v_tensor* a);
V_API v_tensor* v_cos_inplace(v_ctx* ctx, v_tensor* a);
/**
* Truncates the fractional part of each element in the tensor (towards zero).
* For example: trunc(3.7) = 3.0, trunc(-2.9) = -2.0
* Similar to std::trunc in C/C++.
*/

v_tensor* v_trunc(v_ctx* ctx, v_tensor* a);
v_tensor* v_trunc_inplace(v_ctx* ctx, v_tensor* a);


// xIELU activation function
// x = x * (c_a(alpha_n) + c_b(alpha_p, beta) * sigmoid(beta * x)) + eps * (x > 0)
// where c_a = softplus and c_b(a, b) = softplus(a) + b are constraining functions
// that constrain the positive and negative source alpha values respectively
v_tensor* v_xielu(v_ctx* ctx, v_tensor* a, float alpha_n, float alpha_p, float beta, float eps);
