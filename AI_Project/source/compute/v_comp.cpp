#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC
#include "v-backend.h"
#include "ggml-impl.h"
#include "v_hash.h"
#include "v_util.h"

v_tensor* v_add_imple(struct v_ctx* ctx,
                             v_tensor* a,
                             v_tensor* b,
                             bool inplace) {
  V_ASSERT(can_repeat(b, a));
  v_tensor* result = inplace
                              ? v_tensor_view(ctx, a)
                              : v_dup_tensor(ctx, a);
  result->op     = v_OP_ADD;
  result->src[0] = a;
  result->src[1] = b;
  return result;
}

v_tensor* v_set_zero(v_tensor* tensor) {
  if (is_empty(tensor)) {
    return tensor;
  }
  if (tensor->buffer) {
    v_backend_tensor_memset(tensor, 0, 0, num_bytes(tensor));
  }
  else {
    V_ASSERT(tensor->data);
    memset(tensor->data, 0, num_bytes(tensor));
  }
  return tensor;
}

v_tensor* v_unary_impl(struct v_ctx* ctx,
                              v_tensor* a,
                              enum v_unary_op op,
                              bool inplace) {
  V_ASSERT(v_is_contiguous_1(a));

  v_tensor* result = inplace
                              ? v_tensor_view(ctx, a)
                              : v_dup_tensor(ctx, a);

  v_set_op_params_i32(result, 0, (int32_t)op);

  result->op     = v_OP_UNARY;
  result->src[0] = a;

  return result;
}

v_tensor* v_get_rows(struct v_ctx* ctx,
                            v_tensor* a,
                            v_tensor* b) {
  V_ASSERT(a->ne[2] == b->ne[1]);
  V_ASSERT(a->ne[3] == b->ne[2]);
  V_ASSERT(b->ne[3] == 1);
  V_ASSERT(b->type == v_TYPE_I32);
  // TODO: implement non F32 return
  enum v_data_type type = v_TYPE_F32;
  if (a->type == v_TYPE_I32) {
    type = a->type;
  }
  v_tensor* result = v_new_tensor_4d(ctx, type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);
  result->op              = v_OP_GET_ROWS;
  result->src[0]          = a;
  result->src[1]          = b;
  return result;
}

v_tensor* v_get_rows_back(struct v_ctx* ctx,
                                    v_tensor* a,
                                    v_tensor* b,
                                    v_tensor* c) {
  V_ASSERT(v_is_matrix(a) && v_is_vector(b) && b->type == v_TYPE_I32);
  V_ASSERT(v_is_matrix(c) && (a->ne[0] == c->ne[0]));
  // TODO: implement non F32 return
  //struct v_tensor * result = v_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
  v_tensor* result = v_new_tensor_2d(ctx, v_TYPE_F32, c->ne[0], c->ne[1]);
  result->op              = V_OP_GET_ROWS_BACK;
  result->src[0]          = a;
  result->src[1]          = b;
  return result;
}

v_tensor* v_diag(struct v_ctx* ctx,
                        v_tensor* a) {
  V_ASSERT(a->ne[1] == 1);

  const int64_t ne[4]     = {a->ne[0], a->ne[0], a->ne[2], a->ne[3]};
  v_tensor* result = v_new_tensor(ctx, a->type, 4, ne);

  result->op     = V_OP_DIAG;
  result->src[0] = a;

  return result;
}

v_tensor* v_out_prod(struct v_ctx* ctx,
                            v_tensor* a,
                            v_tensor* b) {
  V_ASSERT(v_can_out_prod(a, b));
  V_ASSERT(!v_is_transposed(a));

  // a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
  const int64_t ne[4]     = {a->ne[0], b->ne[0], b->ne[2], b->ne[3]};
  v_tensor* result = v_new_tensor(ctx, v_TYPE_F32, 4, ne);

  result->op     = V_OP_OUT_PROD;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}


v_tensor* v_silu_back(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b) {
  v_tensor* result = v_dup_tensor(ctx, a);

  result->op     = v_OP_SILU_BACK;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

v_tensor* v_glu_impl(struct v_ctx* ctx,
                               v_tensor* a,
                               v_tensor* b,
                               enum v_glu_op op,
                               bool swapped) {
  V_ASSERT(v_is_contiguous_1(a));

  if (b) {
    V_ASSERT(v_is_contiguous_1(b));
    V_ASSERT(v_are_same_shape(a, b));
    V_ASSERT(a->type == b->type);
  }

  int64_t ne[V_MAX_DIMS] = {a->ne[0] / 2};
  for (int i = 1; i < V_MAX_DIMS; i++) ne[i] = a->ne[i];
  v_tensor* result = new_tensor_impl(ctx, a->type, V_MAX_DIMS, b
                                                                          ? a->ne
                                                                          : ne, NULL, 0);

  v_set_op_params_i32(result, 0, (int32_t)op);
  v_set_op_params_i32(result, 1, (int32_t)swapped);

  result->op     = v_OP_GLU;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

v_tensor* v_repeat_4d(struct v_ctx* ctx,
                                v_tensor* a,
                                int64_t ne0,
                                int64_t ne1,
                                int64_t ne2,
                                int64_t ne3) {
  const bool can_repeat = is_empty(a) || (
    (ne0 % a->ne[0] == 0) &&
    (ne1 % a->ne[1] == 0) &&
    (ne2 % a->ne[2] == 0) &&
    (ne3 % a->ne[3] == 0)
  );
  V_ASSERT(can_repeat);

  v_tensor* result = v_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
  result->op              = v_OP_REPEAT;
  result->src[0]          = a;
  return result;
}


v_tensor* v_permute(struct v_ctx* ctx,
                           v_tensor* a,
                           int axis0,
                           int axis1,
                           int axis2,
                           int axis3) {
  V_ASSERT(axis0 >= 0 && axis0 < V_MAX_DIMS);
  V_ASSERT(axis1 >= 0 && axis1 < V_MAX_DIMS);
  V_ASSERT(axis2 >= 0 && axis2 < V_MAX_DIMS);
  V_ASSERT(axis3 >= 0 && axis3 < V_MAX_DIMS);

  V_ASSERT(axis0 != axis1);
  V_ASSERT(axis0 != axis2);
  V_ASSERT(axis0 != axis3);
  V_ASSERT(axis1 != axis2);
  V_ASSERT(axis1 != axis3);
  V_ASSERT(axis2 != axis3);

  v_tensor* result = v_tensor_view(ctx, a);
  v_format_name(result, "%s (permuted)", a->name);

  int ne[V_MAX_DIMS];
  int nb[V_MAX_DIMS];

  ne[axis0] = a->ne[0];
  ne[axis1] = a->ne[1];
  ne[axis2] = a->ne[2];
  ne[axis3] = a->ne[3];

  nb[axis0] = a->nb[0];
  nb[axis1] = a->nb[1];
  nb[axis2] = a->nb[2];
  nb[axis3] = a->nb[3];

  result->ne[0] = ne[0];
  result->ne[1] = ne[1];
  result->ne[2] = ne[2];
  result->ne[3] = ne[3];

  result->nb[0] = nb[0];
  result->nb[1] = nb[1];
  result->nb[2] = nb[2];
  result->nb[3] = nb[3];

  result->op     = V_OP_PERMUTE;
  result->src[0] = a;

  int32_t params[] = {axis0, axis1, axis2, axis3};
  v_set_op_params(result, params, sizeof(params));

  return result;
}

v_tensor* v_set_rows(struct v_ctx* ctx,
                            v_tensor* a,
                            v_tensor* b,
                            v_tensor* c) {
  V_ASSERT(a->ne[0] == b->ne[0]);
  V_ASSERT(a->ne[2] == b->ne[2]);
  V_ASSERT(a->ne[3] == b->ne[3]);
  V_ASSERT(b->ne[1] == c->ne[0]);
  V_ASSERT(b->ne[2] % c->ne[1] == 0);
  V_ASSERT(b->ne[3] % c->ne[2] == 0);
  V_ASSERT(c->ne[3] == 1);
  V_ASSERT(b->type == v_TYPE_F32);
  V_ASSERT(c->type == v_TYPE_I64 || c->type == v_TYPE_I32);

  V_ASSERT(v_is_contiguous_rows(a));
  V_ASSERT(v_is_contiguous_rows(b));

  v_tensor* result = v_tensor_view(ctx, a);

  result->op     = v_OP_SET_ROWS;
  result->src[0] = b;
  result->src[1] = c;
  result->src[2] = a;
  // note: order is weird due to legacy reasons (https://github.com/ggml-org/llama.cpp/pull/16063#discussion_r2385795931)

  return result;
}

v_tensor* v_transpose(struct v_ctx* ctx,
                             v_tensor* a) {
  v_tensor* result = v_tensor_view(ctx, a);
  v_format_name(result, "%s (transposed)", a->name);

  result->ne[0] = a->ne[1];
  result->ne[1] = a->ne[0];

  result->nb[0] = a->nb[1];
  result->nb[1] = a->nb[0];

  result->op     = v_OP_TRANSPOSE;
  result->src[0] = a;

  return result;
}


v_tensor* v_div_impl(struct v_ctx* ctx,
                            v_tensor* a,
                            v_tensor* b,
                            bool inplace) {
  V_ASSERT(can_repeat(b, a));
  v_tensor* result = inplace
                              ? v_tensor_view(ctx, a)
                              : v_dup_tensor(ctx, a);
  result->op     = v_OP_DIV;
  result->src[0] = a;
  result->src[1] = b;
  return result;
}

v_tensor* v_sqr_impl(struct v_ctx* ctx,
                               v_tensor* a,
                               bool inplace) {
  v_tensor* result = inplace
                              ? v_tensor_view(ctx, a)
                              : v_dup_tensor(ctx, a);

  result->op     = v_OP_SQR;
  result->src[0] = a;

  return result;
}


v_tensor* v_sqrt_impl(struct v_ctx* ctx,
                                v_tensor* a,
                                bool inplace) {
  v_tensor* result = inplace
                              ? v_tensor_view(ctx, a)
                              : v_dup_tensor(ctx, a);

  result->op     = v_OP_SQRT;
  result->src[0] = a;

  return result;
}

v_tensor* v_sum_rows(struct v_ctx* ctx,
                               v_tensor* a) {
  int64_t ne[V_MAX_DIMS] = {1};
  for (int i = 1; i < V_MAX_DIMS; ++i) {
    ne[i] = a->ne[i];
  }

  v_tensor* result = v_new_tensor(ctx, a->type, V_MAX_DIMS, ne);

  result->op     = v_OP_SUM_ROWS;
  result->src[0] = a;

  return result;
}


v_tensor* v_mean(struct v_ctx* ctx,
                           v_tensor* a) {
  int64_t ne[4]           = {1, a->ne[1], a->ne[2], a->ne[3]};
  v_tensor* result = v_new_tensor(ctx, v_TYPE_F32, 4, ne);

  result->op     = V_OP_MEAN;
  result->src[0] = a;

  return result;
}

v_tensor* v_sum(struct v_ctx* ctx,
                       v_tensor* a) {
  v_tensor* result = v_new_tensor_1d(ctx, a->type, 1);

  result->op     = v_OP_SUM;
  result->src[0] = a;

  return result;
}

v_tensor* mmlSubImpl(struct v_ctx* ctx,
                            v_tensor* a,
                            v_tensor* b,
                            bool inplace) {
  V_ASSERT(can_repeat(b, a));

  v_tensor* result = inplace
                              ? v_tensor_view(ctx, a)
                              : v_dup_tensor(ctx, a);

  result->op     = v_OP_SUB;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}


v_tensor* add_cast_impl(struct v_ctx* ctx,
                               v_tensor* a,
                               v_tensor* b,
                               enum v_data_type type) {
  // TODO: support less-strict constraint
  //       v_ASSERT(v_can_repeat(b, a));
  V_ASSERT(can_repeat_rows(b, a));

  // currently only supported for quantized input and f16
  V_ASSERT(v_is_quantized(a->type) ||
    a->type == v_TYPE_F16 ||
    a->type == v_TYPE_BF16);

  v_tensor* result = v_new_tensor(ctx, type, V_MAX_DIMS, a->ne);

  result->op     = v_OP_ADD;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

v_tensor* v_mul_impl(struct v_ctx* ctx,
                               v_tensor* a,
                               v_tensor* b,
                               bool inplace) {
  V_ASSERT(can_repeat(b, a));
  v_tensor* result = inplace
                              ? v_tensor_view(ctx, a)
                              : v_dup_tensor(ctx, a);
  result->op     = v_OP_MUL;
  result->src[0] = a;
  result->src[1] = b;
  return result;
}

v_tensor* v_matmul(struct v_ctx* ctx,
                          v_tensor* a,
                          v_tensor* b) {
  V_ASSERT(can_mul_mat(a, b));
  V_ASSERT(!v_is_transposed(a));

  const int64_t ne[4]     = {a->ne[1], b->ne[1], b->ne[2], b->ne[3]};
  v_tensor* result = v_new_tensor(ctx, v_TYPE_F32, 4, ne);
  result->op              = v_OP_MUL_MAT;
  result->src[0]          = a;
  result->src[1]          = b;
  return result;
}

v_tensor* v_l2_norm_impl(struct v_ctx* ctx,
                                   v_tensor* a,
                                   float eps,
                                   bool inplace) {
  v_tensor* result = inplace
                              ? v_tensor_view(ctx, a)
                              : v_dup_tensor(ctx, a);

  v_set_op_params_f32(result, 0, eps);

  result->op     = v_OP_L2_NORM;
  result->src[0] = a;

  return result;
}


v_tensor* v_dup_impl(struct v_ctx* ctx,
                               v_tensor* a,
                               bool inplace) {
  v_tensor* result = inplace
                              ? v_tensor_view(ctx, a)
                              : v_dup_tensor(ctx, a);

  result->op     = v_OP_DUP;
  result->src[0] = a;

  return result;
}

v_tensor* add_id(struct v_ctx* ctx,
                        v_tensor* a,
                        v_tensor* b,
                        v_tensor* ids) {
  V_ASSERT(a->ne[0] == b->ne[0]);
  V_ASSERT(a->ne[1] == ids->ne[0]);
  V_ASSERT(a->ne[2] == ids->ne[1]);
  V_ASSERT(ids->type == v_TYPE_I32);
  v_tensor* result = v_dup_tensor(ctx, a);
  result->op              = v_OP_ADD_ID;
  result->src[0]          = a;
  result->src[1]          = b;
  result->src[2]          = ids;
  return result;
}

v_tensor* add1_impl(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  bool inplace) {
  V_ASSERT(v_is_scalar(b));
  V_ASSERT(v_is_padded_1d(a));
  v_tensor* result = inplace
                              ? v_tensor_view(ctx, a)
                              : v_dup_tensor(ctx, a);

  result->op     = v_OP_ADD1;
  result->src[0] = a;
  result->src[1] = b;
  return result;
}

v_tensor* v_sin_impl(struct v_ctx* ctx,
                               v_tensor* a,
                               bool inplace) {
  v_tensor* result = inplace
                              ? v_tensor_view(ctx, a)
                              : v_dup_tensor(ctx, a);
  result->op     = v_OP_SIN;
  result->src[0] = a;

  return result;
}


// v_cos

v_tensor* v_cos_impl(
  struct v_ctx* ctx,
  v_tensor* a,
  bool inplace) {
  v_tensor* result = inplace
                              ? v_tensor_view(ctx, a)
                              : v_dup_tensor(ctx, a);

  result->op     = v_OP_COS;
  result->src[0] = a;

  return result;
}


v_tensor* v_log_impl(struct v_ctx* ctx,
                               v_tensor* a,
                               bool inplace) {
  v_tensor* result = inplace
                              ? v_tensor_view(ctx, a)
                              : v_dup_tensor(ctx, a);

  result->op     = v_OP_LOG;
  result->src[0] = a;

  return result;
}

v_tensor* v_argmax(struct v_ctx* ctx,
                          v_tensor* a) {
  V_ASSERT(v_is_matrix(a));
  V_ASSERT(a->ne[0] <= INT32_MAX);

  v_tensor* result = v_new_tensor_1d(ctx, v_TYPE_I32, a->ne[1]);
  result->op              = V_OP_ARGMAX;
  result->src[0]          = a;
  return result;
}

v_tensor* v_count_equal(struct v_ctx* ctx,
                                  v_tensor* a,
                                  v_tensor* b) {
  V_ASSERT(v_are_same_shape(a, b));

  v_tensor* result = v_new_tensor_1d(ctx, v_TYPE_I64, 1);

  result->op     = v_OP_COUNT_EQUAL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

v_tensor* v_repeat(struct v_ctx* ctx,
                             v_tensor* a,
                             v_tensor* b) {
  V_ASSERT(can_repeat(a, b));

  v_tensor* result = v_new_tensor(ctx, a->type, V_MAX_DIMS, b->ne);

  result->op     = v_OP_REPEAT;
  result->src[0] = a;

  return result;
}

v_tensor* v_repeat_back(struct v_ctx* ctx,
                                  v_tensor* a,
                                  v_tensor* b) {
  V_ASSERT(can_repeat(b, a));

  v_tensor* result = v_new_tensor(ctx, a->type, V_MAX_DIMS, b->ne);

  result->op     = v_OP_REPEAT_BACK;
  result->src[0] = a;

  return result;
}

// v_concat

v_tensor* v_concat(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  int dim) {
  V_ASSERT(dim >= 0 && dim < V_MAX_DIMS);
  V_ASSERT(a->type == b->type);

  int64_t ne[V_MAX_DIMS];
  for (int d = 0; d < V_MAX_DIMS; ++d) {
    if (d == dim) {
      ne[d] = a->ne[d] + b->ne[d];
      continue;
    }
    V_ASSERT(a->ne[d] == b->ne[d]);
    ne[d] = a->ne[d];
  }

  v_tensor* result = v_new_tensor(ctx, a->type, V_MAX_DIMS, ne);

  v_set_op_params_i32(result, 0, dim);

  result->op     = v_OP_CONCAT;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

v_tensor* v_leaky_relu(struct v_ctx* ctx,
                                 v_tensor* a,
                                 float negative_slope,
                                 bool inplace) {
  v_tensor* result = inplace
                              ? v_tensor_view(ctx, a)
                              : v_dup_tensor(ctx, a);

  v_set_op_params(result, &negative_slope, sizeof(negative_slope));

  result->op     = v_OP_LEAKY_RELU;
  result->src[0] = a;

  return result;
}

v_tensor* v_xielu(struct v_ctx* ctx,
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

v_tensor* v_swiglu_oai(struct v_ctx* ctx,
                                 v_tensor* a,
                                 v_tensor* b,
                                 float alpha,
                                 float limit) {
  v_tensor* result = v_glu_impl(ctx, a, b, v_GLU_OP_SWIGLU_OAI, false);
  v_set_op_params_f32(result, 2, alpha);
  v_set_op_params_f32(result, 3, limit);
  return result;
}

v_tensor* v_norm_impl(struct v_ctx* ctx,
                                v_tensor* a,
                                float eps,
                                bool inplace) {
  v_tensor* result = inplace
                              ? v_tensor_view(ctx, a)
                              : v_dup_tensor(ctx, a);

  v_set_op_params(result, &eps, sizeof(eps));

  result->op     = v_OP_NORM;
  result->src[0] = a;

  return result;
}

v_tensor* v_rms_norm_impl(struct v_ctx* ctx,
                                    v_tensor* a,
                                    float eps,
                                    bool inplace) {
  v_tensor* result = inplace
                              ? v_tensor_view(ctx, a)
                              : v_dup_tensor(ctx, a);

  v_set_op_params(result, &eps, sizeof(eps));

  result->op     = v_OP_RMS_NORM;
  result->src[0] = a;

  return result;
}

v_tensor* v_rms_norm_back(struct v_ctx* ctx,
                                    v_tensor* a,
                                    v_tensor* b,
                                    float eps) {
  v_tensor* result = v_dup_tensor(ctx, a);

  v_set_op_params(result, &eps, sizeof(eps));

  result->op     = v_OP_RMS_NORM_BACK;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

v_tensor* v_group_norm_impl(struct v_ctx* ctx,
                                   v_tensor* a,
                                   int n_groups,
                                   float eps,
                                   bool inplace) {
  v_tensor* result = inplace
                              ? v_tensor_view(ctx, a)
                              : v_dup_tensor(ctx, a);

  v_set_op_params_i32(result, 0, n_groups);
  v_set_op_params_f32(result, 1, eps);

  result->op     = v_OP_GROUP_NORM;
  result->src[0] = a;

  return result;
}

// v_mul_mat_id
/*
    c = v_mul_mat_id(ctx, as, b, ids);
    as  -> [cols, rows, n_expert]
    b   -> [cols, n_expert_used, n_tokens]
    ids -> [n_expert_used, n_tokens] (i32)
    c   -> [rows, n_expert_used, n_tokens]
    in b, n_expert_used can be broadcasted to match the n_expert_used of ids
    c ~= as[:,:,i] @ b[:,i%r,t], i = ids[e,t] for all e,t in ids
*/
v_tensor* mmlMatrixMulId(struct v_ctx* ctx,
                                v_tensor* as,
                                v_tensor* b,
                                v_tensor* ids) {
  V_ASSERT(!v_is_transposed(as));
  V_ASSERT(ids->type == v_TYPE_I32);

  V_ASSERT(as->ne[3] == 1); // as is 3d (one matrix per expert)
  V_ASSERT(b->ne[3] == 1); // b is 3d
  V_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1); // ids is 2d
  V_ASSERT(ids->ne[1] == b->ne[2]); // must have an expert list per b row
  V_ASSERT(as->ne[0] == b->ne[0]); // can_mul_mat
  V_ASSERT(ids->ne[0] % b->ne[1] == 0); // can broadcast

  const int64_t ne[4]     = {as->ne[1], ids->ne[0], b->ne[2], 1};
  v_tensor* result = v_new_tensor(ctx, v_TYPE_F32, 4, ne);

  result->op     = v_OP_MUL_MAT_ID;
  result->src[0] = as;
  result->src[1] = b;
  result->src[2] = ids;

  return result;
}

v_tensor* v_scale_impl(struct v_ctx* ctx,
                              v_tensor* a,
                              float s,
                              float b,
                              bool inplace) {
  V_ASSERT(v_is_padded_1d(a));

  v_tensor* result = inplace
                              ? v_tensor_view(ctx, a)
                              : v_dup_tensor(ctx, a);

  float params[2] = {s, b};
  v_set_op_params(result, &params, sizeof(params));

  result->op     = V_OP_SCALE;
  result->src[0] = a;

  return result;
}

v_tensor* v_cast(struct v_ctx* ctx,
                        v_tensor* a,
                        enum v_data_type type) {
  v_tensor* result = v_new_tensor(ctx, type, V_MAX_DIMS, a->ne);
  v_format_name(result, "%s (copy)", a->name);
  result->op     = v_OP_CPY;
  result->src[0] = a;
  result->src[1] = result;
  return result;
}

v_tensor* v_mem_cont(struct v_ctx* ctx,
                            v_tensor* a) {
  v_tensor* result = v_dup_tensor(ctx, a);
  v_format_name(result, "%s (cont)", a->name);
  result->op     = v_OP_CONT;
  result->src[0] = a;
  return result;
}
