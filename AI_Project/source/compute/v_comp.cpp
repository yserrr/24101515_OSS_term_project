#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC
#include "v_backend.hpp"
#include "ggml-impl.hpp"
#include "v_hash.hpp"
#include "v_util.hpp"



v_tensor* v_set_zero(v_tensor* tensor) {
  if (tensor->is_empty()) {
    return tensor;
  }
  if (tensor->buffer) {
    v_backend_tensor_memset(tensor, 0, 0, num_bytes(tensor));
  } else {
    V_ASSERT(tensor->data);
    memset(tensor->data, 0, num_bytes(tensor));
  }
  return tensor;
}


v_tensor* v_get_rows(v_ctx* ctx,
                     v_tensor* a, v_tensor* b) {
  V_ASSERT(a->ne[2] == b->ne[1]);
  V_ASSERT(a->ne[3] == b->ne[2]);
  V_ASSERT(b->ne[3] == 1);
  V_ASSERT(b->type == v_TYPE_I32);
  // TODO: implement non F32 return
  v_data_type type = v_TYPE_F32;
  if (a->type == v_TYPE_I32) type = a->type;
  v_tensor* result = v_new_tensor_4d(ctx, type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);
  result->op       = v_OP_GET_ROWS;
  result->src[0]   = a;
  result->src[1]   = b;
  return result;
}


v_tensor* v_set_rows(v_ctx* ctx,
                     v_tensor* a, v_tensor* b, v_tensor* c) {
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


v_tensor* v_sqr_impl(v_ctx* ctx,
                     v_tensor* a,
                     bool inplace) {
  v_tensor* result = inplace ? v_tensor_view(ctx, a) : v_dup_tensor(ctx, a);
  result->op       = V_OP_SQR;
  result->src[0]   = a;
  return result;
}


v_tensor* v_sqrt_impl(v_ctx* ctx,
                      v_tensor* a,
                      bool inplace) {
  v_tensor* result = inplace ? v_tensor_view(ctx, a) : v_dup_tensor(ctx, a);

  result->op     = v_OP_SQRT;
  result->src[0] = a;

  return result;
}

v_tensor* v_sum_rows(v_ctx* ctx,
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


v_tensor* v_mean(v_ctx* ctx,
                 v_tensor* a) {
  int64_t ne[4]    = {1, a->ne[1], a->ne[2], a->ne[3]};
  v_tensor* result = v_new_tensor(ctx, v_TYPE_F32, 4, ne);

  result->op     = V_OP_MEAN;
  result->src[0] = a;

  return result;
}

v_tensor* v_sum(v_ctx* ctx,
                v_tensor* a) {
  v_tensor* result = v_new_tensor_1d(ctx, a->type, 1);

  result->op     = v_OP_SUM;
  result->src[0] = a;

  return result;
}



v_tensor* v_add_cast_impl(struct v_ctx* ctx,
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

  v_tensor* result = v_new_tensor(ctx, type, V_MAX_DIMS, a->ne.data());

  result->op     = v_OP_ADD;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}


v_tensor* v_l2_norm_impl(v_ctx* ctx,
                         v_tensor* a,
                         float eps,
                         bool inplace) {
  v_tensor* result = inplace ? v_tensor_view(ctx, a) : v_dup_tensor(ctx, a);

  v_set_op_params_f32(result, 0, eps);

  result->op     = v_OP_L2_NORM;
  result->src[0] = a;

  return result;
}


v_tensor* v_argmax(v_ctx* ctx,
                   v_tensor* a) {
  V_ASSERT(a->is_matrix());
  V_ASSERT(a->ne[0] <= INT32_MAX);

  v_tensor* result = v_new_tensor_1d(ctx, v_TYPE_I32, a->ne[1]);
  result->op       = V_OP_ARGMAX;
  result->src[0]   = a;
  return result;
}

v_tensor* v_count_equal(v_ctx* ctx,
                        v_tensor* a,
                        v_tensor* b) {
  V_ASSERT(v_are_same_shape(a, b));

  v_tensor* result = v_new_tensor_1d(ctx, v_TYPE_I64, 1);

  result->op     = v_OP_COUNT_EQUAL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

v_tensor* v_repeat(v_ctx* ctx,
                   v_tensor* a, v_tensor* b) {
  V_ASSERT(v_can_repeat(a, b));

  v_tensor* result = v_new_tensor(ctx, a->type, V_MAX_DIMS, b->ne.data());

  result->op     = v_OP_REPEAT;
  result->src[0] = a;

  return result;
}

v_tensor* v_repeat_back(struct v_ctx* ctx,
                        v_tensor* a,
                        v_tensor* b) {
  V_ASSERT(v_can_repeat(b, a));

  v_tensor* result = v_new_tensor(ctx, a->type, V_MAX_DIMS, b->ne.data());

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


v_tensor* v_norm_impl(v_ctx* ctx,
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

v_tensor* v_rms_norm_back(v_ctx* ctx,
                          v_tensor* a, v_tensor* b,
                          float eps) {
  v_tensor* result = v_dup_tensor(ctx, a);
  v_set_op_params(result, &eps, sizeof(eps));
  result->op     = v_OP_RMS_NORM_BACK;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

v_tensor* v_group_norm_impl(v_ctx* ctx,
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
v_tensor* v_mat_mul_id(v_ctx* ctx,
                       v_tensor* as,
                       v_tensor* b,
                       v_tensor* ids) {
  V_ASSERT(!as->is_transposed());
  V_ASSERT(ids->type == v_TYPE_I32);

  V_ASSERT(as->ne[3] == 1);                     // as is 3d (one matrix per expert)
  V_ASSERT(b->ne[3] == 1);                      // b is 3d
  V_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1); // ids is 2d
  V_ASSERT(ids->ne[1] == b->ne[2]);             // must have an expert list per b row
  V_ASSERT(as->ne[0] == b->ne[0]);              // can_mul_mat
  V_ASSERT(ids->ne[0] % b->ne[1] == 0);         // can broadcast

  const int64_t ne[4] = {as->ne[1], ids->ne[0], b->ne[2], 1};
  v_tensor* result    = v_new_tensor(ctx, v_TYPE_F32, 4, ne);

  result->op     = v_OP_MUL_MAT_ID;
  result->src[0] = as;
  result->src[1] = b;
  result->src[2] = ids;

  return result;
}

v_tensor* v_scale_impl(v_ctx* ctx,
                       v_tensor* a,
                       float s, float b,
                       bool inplace) {
  V_ASSERT(v_is_padded_1d(a));

  v_tensor* result = inplace ? v_tensor_view(ctx, a) : v_dup_tensor(ctx, a);

  float params[2] = {s, b};
  v_set_op_params(result, &params, sizeof(params));

  result->op     = V_OP_SCALE;
  result->src[0] = a;

  return result;
}

v_tensor* v_cast(v_ctx* ctx,
                 v_tensor* a,
                 v_data_type type) {
  v_tensor* result = v_new_tensor(ctx, type, V_MAX_DIMS, a->ne.data());
  v_format_name(result, "%s (copy)", a->name);
  result->op     = V_OP_CPY;
  result->src[0] = a;
  result->src[1] = result;
  return result;
}
