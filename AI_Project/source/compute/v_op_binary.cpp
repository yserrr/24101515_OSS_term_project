#include "v.hpp"
#include "v_tensor.hpp"
#include "v_op_binary.hpp"

#include "v_util.hpp"

v_tensor* v_div_impl(v_ctx* ctx,
                     v_tensor* a, v_tensor* b,
                     bool inplace) {
  V_ASSERT(v_can_repeat(b, a));
  v_tensor* result = inplace ? v_tensor_view(ctx, a) : v_dup_tensor(ctx, a);
  result->op       = v_OP_DIV;
  result->src[0]   = a;
  result->src[1]   = b;
  return result;
}

v_tensor* v_repeat_4d(v_ctx* ctx,
                      v_tensor* a,
                      int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
  const bool can_repeat = a->is_empty() || (
    (ne0 % a->ne[0] == 0) && (ne1 % a->ne[1] == 0) &&
    (ne2 % a->ne[2] == 0) && (ne3 % a->ne[3] == 0));
  V_ASSERT(can_repeat);
  v_tensor* result = v_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
  result->op       = v_OP_REPEAT;
  result->src[0]   = a;
  return result;
}

// a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
// not used now.
v_tensor* v_out_prod(v_ctx* ctx,
                     v_tensor* a, v_tensor* b) {
  V_ASSERT(v_can_out_prod(a, b));
  V_ASSERT(!(a)->is_transposed());
  const int64_t ne[4] = {a->ne[0], b->ne[0], b->ne[2], b->ne[3]};
  v_tensor* result    = v_new_tensor(ctx, v_TYPE_F32, 4, ne);
  result->op          = V_OP_OUT_PROD;
  result->src[0]      = a;
  result->src[1]      = b;
  return result;
}


v_tensor* v_get_rows_back(v_ctx* ctx,
                          v_tensor* a, v_tensor* b, v_tensor* c) {
  V_ASSERT(a->is_matrix() && b->is_vector() && b->type == v_TYPE_I32);
  V_ASSERT(c->is_matrix() && (a->ne[0] == c->ne[0]));
  // TODO: implement non F32 return
  //struct v_tensor * result = v_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
  v_tensor* result = v_new_tensor_2d(ctx, v_TYPE_F32, c->ne[0], c->ne[1]);
  result->op       = V_OP_GET_ROWS_BACK;
  result->src[0]   = a;
  result->src[1]   = b;
  return result;
}


v_tensor* v_add1_impl(v_ctx* ctx,
                      v_tensor* a, v_tensor* b,
                      bool inplace) {
  V_ASSERT(b->is_scalar());
  V_ASSERT(v_is_padded_1d(a));
  v_tensor* result = inplace ? v_tensor_view(ctx, a) : v_dup_tensor(ctx, a);

  result->op     = v_OP_ADD1;
  result->src[0] = a;
  result->src[1] = b;
  return result;
}

v_tensor* v_add_id(v_ctx* ctx,
                   v_tensor* a,
                   v_tensor* b,
                   v_tensor* ids) {
  V_ASSERT(a->ne[0] == b->ne[0]);
  V_ASSERT(a->ne[1] == ids->ne[0]);
  V_ASSERT(a->ne[2] == ids->ne[1]);
  V_ASSERT(ids->type == v_TYPE_I32);
  v_tensor* result = v_dup_tensor(ctx, a);
  result->op       = v_OP_ADD_ID;
  result->src[0]   = a;
  result->src[1]   = b;
  result->src[2]   = ids;
  return result;
}

v_tensor* v_mul_impl(v_ctx* ctx,
                     v_tensor* a, v_tensor* b,
                     bool inplace) {
  V_ASSERT(v_can_repeat(b, a));
  v_tensor* result = inplace
                       ? v_tensor_view(ctx, a)
                       : v_dup_tensor(ctx, a);
  result->op     = v_OP_MUL;
  result->src[0] = a;
  result->src[1] = b;
  return result;
}

v_tensor* v_matmul(v_ctx* ctx,
                   v_tensor* a, v_tensor* b) {
  V_ASSERT(can_mul_mat(a, b));
  V_ASSERT(!(a)->is_transposed());
  const int64_t ne[4] = {a->ne[1], b->ne[1], b->ne[2], b->ne[3]};
  v_tensor* result    = v_new_tensor(ctx, v_TYPE_F32, 4, ne);
  result->op          = V_OP_MUL_MAT;
  result->src[0]      = a;
  result->src[1]      = b;
  return result;
}

v_tensor* v_sub_impl(v_ctx* ctx,
                     v_tensor* a,
                     v_tensor* b,
                     bool inplace) {
  V_ASSERT(v_can_repeat(b, a));

  v_tensor* result = inplace
                       ? v_tensor_view(ctx, a)
                       : v_dup_tensor(ctx, a);

  result->op     = v_OP_SUB;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

v_tensor* v_add_imple(v_ctx* ctx,
                      v_tensor* a, v_tensor* b,
                      bool inplace) {
  V_ASSERT(v_can_repeat(b, a));
  v_tensor* result = inplace
                       ? v_tensor_view(ctx, a)
                       : v_dup_tensor(ctx, a);
  result->op     = v_OP_ADD;
  result->src[0] = a;
  result->src[1] = b;
  return result;
}

v_tensor* v_acc_imple(v_ctx* ctx,
                      v_tensor* a, v_tensor* b,
                      size_t nb1, size_t nb2, size_t nb3, size_t offset,
                      bool inplace) {
  V_ASSERT(b->num_elements() <= a->num_elements());
  V_ASSERT(a->is_contiguous());
  V_ASSERT(a->type == v_TYPE_F32);
  V_ASSERT(b->type == v_TYPE_F32);
  v_tensor* result = inplace ? v_tensor_view(ctx, a) : v_dup_tensor(ctx, a);
  int32_t params[] = {
    static_cast<int32_t>(nb1),
    static_cast<int32_t>(nb2),
    static_cast<int32_t>(nb3),
    static_cast<int32_t>(offset),
    (inplace ? 1 : 0)
  };
  v_set_op_params(result, params, sizeof(params));

  result->op     = v_OP_ACC;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

v_tensor* v_acc(
  v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  size_t nb1,
  size_t nb2,
  size_t nb3,
  size_t offset) {
  return v_acc_imple(ctx, a, b, nb1, nb2, nb3, offset, false);
}

v_tensor* v_sub(
  v_ctx* ctx,
  v_tensor* a,
  v_tensor* b) {
  return v_sub_impl(ctx, a, b, false);
}

v_tensor* v_sub_inplace(
  v_ctx* ctx,
  v_tensor* a,
  v_tensor* b) {
  return v_sub_impl(ctx, a, b, true);
}

v_tensor* v_mul(v_ctx* ctx,
                v_tensor* a,
                v_tensor* b) {
  return v_mul_impl(ctx, a, b, false);
}

v_tensor* v_mul_inplace(v_ctx* ctx,
                        v_tensor* a,
                        v_tensor* b) {
  return v_mul_impl(ctx, a, b, true);
}

v_tensor* v_div(v_ctx* ctx,
                v_tensor* a,
                v_tensor* b) {
  return v_div_impl(ctx, a, b, false);
}

v_tensor* v_scale(
  v_ctx* ctx,
  v_tensor* a,
  float s) {
  return v_scale_impl(ctx, a, s, 0.05, false);
}

v_tensor* v_scale_inplace(
  v_ctx* ctx,
  v_tensor* a,
  float s) {
  return v_scale_impl(ctx, a, s, 0.0, true);
}

v_tensor* v_acc_inplace(v_ctx* ctx,
                        v_tensor* a,
                        v_tensor* b,
                        size_t nb1,
                        size_t nb2,
                        size_t nb3,
                        size_t offset) {
  return v_acc_imple(ctx, a, b, nb1, nb2, nb3, offset, true);
}

v_tensor* v_dup(v_ctx* ctx,
                v_tensor* a) {
  return v_dup_impl(ctx, a, false);
}


v_tensor* v_dup_inplace(
  v_ctx* ctx,
  v_tensor* a) {
  return v_dup_impl(ctx, a, true);
}


v_tensor* v_set(v_ctx* ctx,
                v_tensor* a,
                v_tensor* b,
                size_t nb1,
                size_t nb2,
                size_t nb3,
                size_t offset) {
  return v_set_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}
