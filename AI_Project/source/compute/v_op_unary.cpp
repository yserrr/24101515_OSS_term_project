#include "v_header.hpp"
#include "v_op_unary.hpp"
#include "ggml-impl.hpp"

v_tensor* v_sin_impl(v_ctx* ctx, v_tensor* a, bool inplace) {
  v_tensor* result = inplace ? v_tensor_view(ctx, a) : v_dup_tensor(ctx, a);
  result->op       = V_OP_SIN;
  result->src[0]   = a;

  return result;
}


v_tensor* v_cos_impl(v_ctx* ctx,
                     v_tensor* a,
                     bool inplace) {
  v_tensor* result = inplace ? v_tensor_view(ctx, a) : v_dup_tensor(ctx, a);
  result->op       = V_OP_COS;
  result->src[0]   = a;

  return result;
}


v_tensor* v_leaky_relu(v_ctx* ctx,
                       v_tensor* a,
                       float negative_slope,
                       bool inplace) {
  v_tensor* result = inplace ? v_tensor_view(ctx, a) : v_dup_tensor(ctx, a);
  v_set_op_params(result, &negative_slope, sizeof(negative_slope));
  result->op     = V_OP_LEAKY_RELU;
  result->src[0] = a;
  return result;
}


v_tensor* v_sin(v_ctx* ctx,
                v_tensor* a) {
  return v_sin_impl(ctx, a, false);
};

v_tensor* v_sin_inplace(v_ctx* ctx,
                        v_tensor* a) {
  return v_sin_impl(ctx, a, true);
}

v_tensor* v_cos(v_ctx* ctx,
                v_tensor* a) {
  return v_cos_impl(ctx, a, false);
}

v_tensor* v_cos_inplace(v_ctx* ctx,
                      v_tensor* a) {
  return v_cos_impl(ctx, a, true);
}

v_tensor* v_log(v_ctx* ctx,
                v_tensor* a) {
  return v_unary(ctx, a, V_UNARY_OP_LOG);
}

///todo: change to unary
v_tensor* v_log_impl(v_ctx* ctx,
                     v_tensor* a,
                     bool inplace) {
  v_tensor* result = inplace ? v_tensor_view(ctx, a) : v_dup_tensor(ctx, a);
  result->op       = V_OP_LOG;
  result->src[0]   = a;
  return result;
}

v_tensor* v_log_inplace(v_ctx* ctx,
                        v_tensor* a) {
  return v_log_impl(ctx, a, true);
}


v_tensor* v_unary(v_ctx* ctx,
                  v_tensor* a,
                  v_unary_op op) {
  return v_unary_impl(ctx, a, op, false);
}

v_tensor* v_unary_inplace(v_ctx* ctx,
                          v_tensor* a,
                          v_unary_op op) {
  return v_unary_impl(ctx, a, op, true);
}

v_tensor* v_silu(v_ctx* ctx, v_tensor* a) {
  return v_unary(ctx, a, v_UNARY_OP_SILU);
}


v_tensor* v_silu_inplace(v_ctx* ctx, v_tensor* a) {
  return v_unary_inplace(ctx, a, v_UNARY_OP_SILU);
}

v_tensor* v_gelu_erf(v_ctx* ctx, v_tensor* a) {
  return v_unary(ctx, a, v_UNARY_OP_GELU_ERF);
}

v_tensor* v_gelu_erf_inplace(v_ctx* ctx, v_tensor* a) {
  return v_unary_inplace(ctx, a, v_UNARY_OP_GELU_ERF);
}

v_tensor* v_gelu_quick(v_ctx* ctx, v_tensor* a) {
  return v_unary(ctx, a, v_UNARY_OP_GELU_QUICK);
}

v_tensor* v_gelu_quick_inplace(v_ctx* ctx, v_tensor* a) {
  return v_unary_inplace(ctx, a, v_UNARY_OP_GELU_QUICK);
}

v_tensor* v_sigmoid(v_ctx* ctx, v_tensor* a) {
  return v_unary(ctx, a, v_UNARY_OP_SIGMOID);
}

v_tensor* v_sigmoid_inplace(v_ctx* ctx, v_tensor* a) {
  return v_unary_inplace(ctx, a, v_UNARY_OP_SIGMOID);
}


v_tensor* v_gelu(v_ctx* ctx, v_tensor* a) {
  return v_unary(ctx, a, v_UNARY_OP_GELU);
}

v_tensor* v_gelu_inplace(v_ctx* ctx, v_tensor* a) {
  return v_unary_inplace(ctx, a, v_UNARY_OP_GELU);
}

v_tensor* v_abs(v_ctx* ctx, v_tensor* a) {
  return v_unary(ctx, a, v_UNARY_OP_ABS);
}

v_tensor* v_abs_inplace(v_ctx* ctx, v_tensor* a) {
  return v_unary_inplace(ctx, a, v_UNARY_OP_ABS);
}


v_tensor* v_sgn(v_ctx* ctx, v_tensor* a) {
  return v_unary(ctx, a, v_UNARY_OP_SGN);
}

v_tensor* v_sgn_inplace(v_ctx* ctx, v_tensor* a) {
  return v_unary_inplace(ctx, a, v_UNARY_OP_SGN);
}

v_tensor* v_neg(v_ctx* ctx, v_tensor* a) {
  return v_unary(ctx, a, v_UNARY_OP_NEG);
}

v_tensor* v_neg_inplace(v_ctx* ctx, v_tensor* a) {
  return v_unary_inplace(ctx, a, v_UNARY_OP_NEG);
}

v_tensor* v_step(v_ctx* ctx, v_tensor* a) {
  return v_unary(ctx, a, v_UNARY_OP_RELU);
}

v_tensor* v_step_inplace(v_ctx* ctx, v_tensor* a) {
  return v_unary_inplace(ctx, a, v_UNARY_OP_STEP);
}

v_tensor* v_tanh(v_ctx* ctx, v_tensor* a) {
  return v_unary(ctx, a, v_UNARY_OP_TANH);
}

v_tensor* v_tanh_inplace(v_ctx* ctx, v_tensor* a) {
  return v_unary_inplace(ctx, a, v_UNARY_OP_TANH);
}

v_tensor* v_elu(v_ctx* ctx, v_tensor* a) {
  return v_unary(ctx, a, v_UNARY_OP_ELU);
}

v_tensor* v_elu_inplace(v_ctx* ctx, v_tensor* a) {
  return v_unary_inplace(ctx, a, v_UNARY_OP_ELU);
}


v_tensor* v_relu(v_ctx* ctx, v_tensor* a) {
  return v_unary(ctx, a, v_UNARY_OP_RELU);
}

v_tensor* v_relu_inplace(v_ctx* ctx, v_tensor* a) {
  return v_unary_inplace(ctx, a, v_UNARY_OP_RELU);
}

v_tensor* v_xielu(v_ctx* ctx,
                  v_tensor* a,
                  float alpha_n,
                  float alpha_p,
                  float beta,
                  float eps) {
  v_tensor* result = v_dup_tensor(ctx, a);

  v_set_op_params_i32(result, 0, (int32_t)v_UNARY_OP_XIELU);
  v_set_op_params_f32(result, 1, beta + v_softplus(alpha_n));
  v_set_op_params_f32(result, 2, v_softplus(alpha_p));
  v_set_op_params_f32(result, 3, beta);
  v_set_op_params_f32(result, 4, eps);

  result->op     = v_OP_UNARY;
  result->src[0] = a;

  return result;
}

v_tensor* v_hardswish(v_ctx* ctx, v_tensor* a) {
  return v_unary(ctx, a, v_UNARY_OP_HARDSWISH);
}

v_tensor* v_hardsigmoid(v_ctx* ctx, v_tensor* a) {
  return v_unary(ctx, a, v_UNARY_OP_HARDSIGMOID);
}

v_tensor* v_floor(v_ctx* ctx,
                  v_tensor* a) {
  return v_unary(ctx, a, v_UNARY_OP_FLOOR);
}

v_tensor* v_floor_inplace(v_ctx* ctx,
                          v_tensor* a) {
  return v_unary_inplace(ctx, a, v_UNARY_OP_FLOOR);
}

v_tensor* v_ceil(v_ctx* ctx,
                 v_tensor* a) {
  return v_unary(ctx, a, v_UNARY_OP_CEIL);
}

v_tensor* v_ceil_inplace(v_ctx* ctx, v_tensor* a) {
  return v_unary_inplace(ctx, a, v_UNARY_OP_CEIL);
}

v_tensor* v_round(v_ctx* ctx, v_tensor* a) {
  return v_unary(ctx, a, v_UNARY_OP_ROUND);
}

v_tensor* v_round_inplace(v_ctx* ctx, v_tensor* a) {
  return v_unary_inplace(ctx, a, v_UNARY_OP_ROUND);
}

v_tensor* v_trunc(v_ctx* ctx, v_tensor* a) {
  return v_unary(ctx, a, v_UNARY_OP_TRUNC);
}

v_tensor* v_trunc_inplace(v_ctx* ctx, v_tensor* a) {
  return v_unary_inplace(ctx, a, v_UNARY_OP_TRUNC);
}

v_tensor* v_glu(v_ctx* ctx, v_tensor* a, enum v_glu_op op, bool swapped) {
  return v_glu_impl(ctx, a, nullptr, op, swapped);
}

v_tensor* v_glu_split(v_ctx* ctx, v_tensor* a, v_tensor* b, v_glu_op op) {
  return v_glu_impl(ctx, a, b, op, false);
}

v_tensor* v_reglu(v_ctx* ctx, v_tensor* a) {
  return v_glu_impl(ctx, a, nullptr, v_GLU_OP_REGLU, false);
}

v_tensor* v_reglu_swapped(v_ctx* ctx, v_tensor* a) {
  return v_glu_impl(ctx, a, nullptr, v_GLU_OP_REGLU, true);
}

v_tensor* v_reglu_split(v_ctx* ctx, v_tensor* a, v_tensor* b) {
  return v_glu_impl(ctx, a, b, v_GLU_OP_REGLU, false);
}

v_tensor* v_geglu(v_ctx* ctx, v_tensor* a) {
  return v_glu_impl(ctx, a, nullptr, v_GLU_OP_GEGLU, false);
}

v_tensor* v_geglu_swapped(v_ctx* ctx, v_tensor* a) {
  return v_glu_impl(ctx, a, nullptr, v_GLU_OP_GEGLU, true);
}

v_tensor* v_geglu_split(v_ctx* ctx, v_tensor* a, v_tensor* b) {
  return v_glu_impl(ctx, a, b, v_GLU_OP_GEGLU, false);
}

v_tensor* v_swiglu(v_ctx* ctx, v_tensor* a) {
  return v_glu_impl(ctx, a, nullptr, v_GLU_OP_SWIGLU, false);
}

v_tensor* v_swiglu_swapped(v_ctx* ctx, v_tensor* a) {
  return v_glu_impl(ctx, a, nullptr, v_GLU_OP_SWIGLU, true);
}

v_tensor* v_swiglu_split(v_ctx* ctx, v_tensor* a, v_tensor* b) {
  return v_glu_impl(ctx, a, b, v_GLU_OP_SWIGLU, false);
}

v_tensor* v_geglu_erf(v_ctx* ctx, v_tensor* a) {
  return v_glu_impl(ctx, a, nullptr, v_GLU_OP_GEGLU_ERF, false);
}

v_tensor* v_geglu_erf_swapped(v_ctx* ctx, v_tensor* a) {
  return v_glu_impl(ctx, a, nullptr, v_GLU_OP_GEGLU_ERF, true);
}

v_tensor* v_geglu_erf_split(v_ctx* ctx, v_tensor* a, v_tensor* b) {
  return v_glu_impl(ctx, a, b, v_GLU_OP_GEGLU_ERF, false);
}

v_tensor* v_geglu_quick(v_ctx* ctx, v_tensor* a) {
  return v_glu_impl(ctx, a, nullptr, v_GLU_OP_GEGLU_QUICK, false);
}

v_tensor* v_geglu_quick_swapped(v_ctx* ctx, v_tensor* a) {
  return v_glu_impl(ctx, a, nullptr, v_GLU_OP_GEGLU_QUICK, true);
}

v_tensor* v_geglu_quick_split(v_ctx* ctx, v_tensor* a, v_tensor* b) {
  return v_glu_impl(ctx, a, b, v_GLU_OP_GEGLU_QUICK, false);
}

v_tensor* v_norm_inplace(v_ctx* ctx, v_tensor* a, float eps) {
  return v_norm_impl(ctx, a, eps, true);
}
