#include "v_gated_linear_unit.hpp"
#include "ggml-impl.hpp"

v_tensor* v_swiglu_oai(v_ctx* ctx,
                       v_tensor* a, v_tensor* b,
                       float alpha,
                       float limit) {
  v_tensor* result = v_glu_impl(ctx, a, b, v_GLU_OP_SWIGLU_OAI, false);
  v_set_op_params_f32(result, 2, alpha);
  v_set_op_params_f32(result, 3, limit);
  return result;
}


v_tensor* v_glu_impl(v_ctx* ctx,
                     v_tensor* a, v_tensor* b,
                     v_glu_op op,
                     bool swapped) {
  V_ASSERT((a)->is_contiguous_1());
  if (b) {
    V_ASSERT((b)->is_contiguous_1());
    V_ASSERT(v_are_same_shape(a, b));
    V_ASSERT(a->type == b->type);
  }
  int64_t ne[V_MAX_DIMS] = {a->ne[0] / 2};
  for (int i = 1; i < V_MAX_DIMS; i++) ne[i] = a->ne[i];
  v_tensor* result = v_new_tensor_impl(ctx, a->type, V_MAX_DIMS, b ? a->ne.data() : ne, nullptr, 0);
  v_set_op_params_i32(result, 0, op);
  v_set_op_params_i32(result, 1, swapped);
  result->op     = v_OP_GLU;
  result->src[0] = a;
  result->src[1] = b;
  return result;
}

v_glu_op v_get_glu_op(const v_tensor* tensor) {
  V_ASSERT(tensor->op == v_OP_GLU);
  return static_cast<v_glu_op>(v_get_op_params_i32(tensor, 0));
}


v_tensor* v_glu(v_ctx* ctx, v_tensor* a, v_glu_op op, bool swapped) {
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

v_tensor* v_gated_linear_attn(v_ctx* ctx,
                              v_tensor* k, v_tensor* v, v_tensor* q,
                              v_tensor* g, v_tensor* state,
                              float scale) {
  V_ASSERT(k->is_contiguous());
  V_ASSERT(v->is_contiguous());
  V_ASSERT(q->is_contiguous());
  V_ASSERT(g->is_contiguous());
  V_ASSERT(state->is_contiguous());

  const int64_t S        = k->ne[0];
  const int64_t H        = k->ne[1];
  const int64_t n_tokens = k->ne[2];
  const int64_t n_seqs   = state->ne[1];
  {
    V_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
    V_ASSERT(q->ne[0] == S && q->ne[1] == H && q->ne[2] == n_tokens);
    V_ASSERT(g->ne[0] == S && g->ne[1] == H && g->ne[2] == n_tokens);
    V_ASSERT(nelements(state) == S * S * H * n_seqs);
  }

  // concat output and new_state
  const int64_t ne[4] = {S * H, n_tokens + S * n_seqs, 1, 1};
  v_tensor* result    = v_new_tensor(ctx, v_TYPE_F32, 4, ne);
  v_set_op_params_f32(result, 0, scale);
  result->op     = v_OP_GATED_LINEAR_ATTN;
  result->src[0] = k;
  result->src[1] = v;
  result->src[2] = q;
  result->src[3] = g;
  result->src[4] = state;
  return result;
}
