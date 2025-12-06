#ifndef MYPROJECT_MML_UTIL_H
#define MYPROJECT_MML_UTIL_H
#include<iostream>
#include "v.h"
#include "ggml-impl.h"
// dst = a
// view(dst, nb1, nb2, nb3, offset) += b
// return dst


inline bool v_is_contiguous(const struct v_tensor* tensor)
{
  return v_is_contiguous_0(tensor);
}

inline bool v_is_contiguous_0(const struct v_tensor* tensor)
{
  return v_is_contiguous_n(tensor, 0);
}

inline bool v_is_contiguous_1(const struct v_tensor* tensor)
{
  return v_is_contiguous_n(tensor, 1);
}


inline bool v_is_contiguous_2(const struct v_tensor* tensor)
{
  return v_is_contiguous_n(tensor, 2);
}


inline bool v_is_contiguously_allocated(const struct v_tensor* tensor)
{
  return num_bytes(tensor) == nelements(tensor) * v_type_size(tensor->type) / blockSize(tensor->type);
}

inline bool v_is_permuted(const struct v_tensor* tensor)
{
  static_assert(MML_MAX_DIMS == 4, "v_MAX_DIMS is not 4 - update this function");

  return tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
}

inline struct v_tensor* v_group_norm(
  struct v_ctx* ctx,
  struct v_tensor* a,
  int n_groups,
  float eps)
{
  return v_group_norm_impl(ctx, a, n_groups, eps, false);
}

inline void v_print_object(const struct v_object* obj)
{
  LOG_INFO(" - v_object: type = %d, offset = %zu, size = %zu, next = %p\n",
           obj->type,
           obj->offs,
           obj->size,
           (const void *) obj->next);
}

inline bool v_are_same_shape(const struct v_tensor* t0, const struct v_tensor* t1)
{
  static_assert(MML_MAX_DIMS == 4, "v_MAX_DIMS is not 4 - update this function");
  return
    (t0->ne[0] == t1->ne[0]) &&
    (t0->ne[1] == t1->ne[1]) &&
    (t0->ne[2] == t1->ne[2]) &&
    (t0->ne[3] == t1->ne[3]);
}

inline struct v_tensor* v_sin(struct v_ctx* ctx,
                              struct v_tensor* a)
{
  return v_sin_impl(ctx, a, false);
};

inline struct v_tensor* v_sin_inplace(struct v_ctx* ctx,
                                         struct v_tensor* a)
{
  return v_sin_impl(ctx, a, true);
}

inline struct v_tensor* v_cos(struct v_ctx* ctx,
                                 struct v_tensor* a)
{
  return v_cos_impl(ctx, a, false);
}

inline struct v_tensor* cos_inplace(struct v_ctx* ctx,
                                    struct v_tensor* a)
{
  return v_cos_impl(ctx, a, true);
}

inline struct v_tensor* v_log(struct v_ctx* ctx,
                            struct v_tensor* a)
{
  return v_unary(ctx, a, v_UNARY_OP_LOG);
}

inline struct v_tensor* v_log_inplace(struct v_ctx* ctx,
                                         struct v_tensor* a)
{
  return v_log_impl(ctx, a, true);
}


inline struct v_tensor* v_group_norm_inplace(
  struct v_ctx* ctx,
  struct v_tensor* a,
  int n_groups,
  float eps)
{
  return v_group_norm_impl(ctx, a, n_groups, eps, true);
}

inline bool v_is_contiguous_channels(const struct v_tensor* tensor)
{
  return
    tensor->nb[0] > tensor->nb[2] &&
    tensor->nb[1] > tensor->nb[0] &&
    tensor->nb[2] == v_type_size(tensor->type);
}

inline bool v_is_contiguous_rows(const struct v_tensor* tensor)
{
  return
    tensor->ne[0] == blockSize(tensor->type) ||
    tensor->nb[0] == v_type_size(tensor->type);
}

inline bool v_is_padded_1d(const struct v_tensor* tensor)
{
  static_assert(MML_MAX_DIMS == 4, "v_MAX_DIMS is not 4 - update this function");
  return
    tensor->nb[0] == v_type_size(tensor->type) &&
    tensor->nb[2] == tensor->nb[1] * tensor->ne[1] &&
    tensor->nb[3] == tensor->nb[2] * tensor->ne[2];
}

inline struct v_tensor* v_acc(
  struct v_ctx* ctx,
  struct v_tensor* a,
  struct v_tensor* b,
  size_t nb1,
  size_t nb2,
  size_t nb3,
  size_t offset)
{
  return v_acc_imple(ctx, a, b, nb1, nb2, nb3, offset, false);
}

inline struct v_tensor* v_sub(
  struct v_ctx* ctx,
  struct v_tensor* a,
  struct v_tensor* b)
{
  return mmlSubImpl(ctx, a, b, false);
}

inline struct v_tensor* v_sub_inplace(
  struct v_ctx* ctx,
  struct v_tensor* a,
  struct v_tensor* b)
{
  return mmlSubImpl(ctx, a, b, true);
}

inline struct v_tensor* v_mul(struct v_ctx* ctx,
                              struct v_tensor* a,
                              struct v_tensor* b)
{
  return v_mul_impl(ctx, a, b, false);
}

inline struct v_tensor* v_mul_inplace(struct v_ctx* ctx,
                                      struct v_tensor* a,
                                      struct v_tensor* b)
{
  return v_mul_impl(ctx, a, b, true);
}

inline struct v_tensor* v_div(struct v_ctx* ctx,
                               struct v_tensor* a,
                               struct v_tensor* b)
{
  return v_div_impl(ctx, a, b, false);
}

inline struct v_tensor* v_scale(
  struct v_ctx* ctx,
  struct v_tensor* a,
  float s)
{
  return v_scale_impl(ctx, a, s, 0.05, false);
}

inline struct v_tensor* v_scale_inplace(
  struct v_ctx* ctx,
  struct v_tensor* a,
  float s)
{
  return v_scale_impl(ctx, a, s, 0.0, true);
}

inline struct v_tensor* v_acc_inplace(struct v_ctx* ctx,
                                         struct v_tensor* a,
                                         struct v_tensor* b,
                                         size_t nb1,
                                         size_t nb2,
                                         size_t nb3,
                                         size_t offset)
{
  return v_acc_imple(ctx, a, b, nb1, nb2, nb3, offset, true);
}

inline struct v_tensor* v_new_tensor_2d(struct v_ctx* ctx,
                                        enum v_data_type type,
                                        int64_t ne0,
                                        int64_t ne1)
{
  const int64_t ne[2] = {ne0, ne1};
  return v_new_tensor(ctx, type, 2, ne);
}

inline struct v_tensor* v_new_tensor_1d(struct v_ctx* ctx,
                                        enum v_data_type type,
                                        int64_t ne0)
{
  return v_new_tensor(ctx, type, 1, &ne0);
}

inline struct v_tensor* v_new_tensor(struct v_ctx* ctx,
                                     enum v_data_type type,
                                     int n_dims,
                                     const int64_t* ne)
{
  return new_tensor_impl(ctx, type, n_dims, ne, NULL, 0);
}

inline struct v_tensor* v_new_tensor_3d(struct v_ctx* ctx,
                                       enum v_data_type type,
                                       int64_t ne0,
                                       int64_t ne1,
                                       int64_t ne2)
{
  const int64_t ne[3] = {ne0, ne1, ne2};
  return v_new_tensor(ctx, type, 3, ne);
}

inline struct v_tensor* v_new_tensor_4d(struct v_ctx* ctx,
                                       enum v_data_type type,
                                       int64_t ne0,
                                       int64_t ne1,
                                       int64_t ne2,
                                       int64_t ne3)
{
  const int64_t ne[4] = {ne0, ne1, ne2, ne3};
  return v_new_tensor(ctx, type, 4, ne);
}

inline struct v_tensor* v_dup(struct v_ctx* ctx,
                              struct v_tensor* a)
{
  return v_dup_impl(ctx, a, false);
}


inline struct v_tensor* v_dup_inplace(
  struct v_ctx* ctx,
  struct v_tensor* a)
{
  return v_dup_impl(ctx, a, true);
}

inline struct v_tensor* v_exp_inplace(
  struct v_ctx* ctx,
  struct v_tensor* a)
{
  return v_unary_inplace(ctx, a, v_UNARY_OP_EXP);
}

inline struct v_tensor* v_exp(struct v_ctx* ctx,
                              struct v_tensor* a)
{
  return v_unary(ctx, a, v_UNARY_OP_EXP);
}


inline struct v_tensor* v_set(struct v_ctx* ctx,
                            struct v_tensor* a,
                            struct v_tensor* b,
                            size_t nb1,
                            size_t nb2,
                            size_t nb3,
                            size_t offset)
{
  return set_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}


inline void v_set_loss(struct v_tensor* tensor)
{
  //MML_ASSERT(v_is_scalar(tensor));
  V_ASSERT(tensor->type == v_TYPE_F32);
  tensor->flags |= TENSOR_FLAG_LOSS;
}

inline void v_set_inputs(struct v_tensor* tensor)
{
  tensor->flags |= TENSOR_FLAG_INPUT;
}

inline void v_set_outputs(struct v_tensor* tensor)
{
  tensor->flags |= TENSOR_FLAG_OUTPUT;
}

inline void v_set_params(struct v_tensor* tensor)
{
  V_ASSERT(tensor->op == v_OP_NONE);
  tensor->flags |= TENSOR_FLAG_PARAM;
}

inline struct v_cgraph* new_graph(struct v_ctx* ctx)
{
  return v_new_graph_custom(ctx, v_DEFAULT_GRAPH_SIZE, false);
}

inline void v_hash_set_reset(struct v_hash_set* hash_set)
{
  memset(hash_set->used, 0, sizeof(v_bitset_t) * v_bitset_size(hash_set->size));
}


inline int64_t nelements(const struct v_tensor* tensor)
{
  static_assert(MML_MAX_DIMS == 4, "v_MAX_DIMS is not 4 - update this function");
  return tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

inline int64_t v_nrows(const struct v_tensor* tensor)
{
  static_assert(MML_MAX_DIMS == 4, "v_MAX_DIMS is not 4 - update this function");
  return tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}


inline size_t v_element_size(const struct v_tensor* tensor)
{
  return v_type_size(tensor->type);
}


inline bool v_is_scalar(const struct v_tensor* tensor)
{
  static_assert(MML_MAX_DIMS == 4, "v_MAX_DIMS is not 4 - update this function");

  return tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

inline bool v_is_vector(const struct v_tensor* tensor)
{
  static_assert(MML_MAX_DIMS == 4, "v_MAX_DIMS is not 4 - update this function");

  return tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

inline bool v_is_matrix(const struct v_tensor* tensor)
{
  static_assert(MML_MAX_DIMS == 4, "v_MAX_DIMS is not 4 - update this function");

  return tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

inline bool v_is_3d(const struct v_tensor* tensor)
{
  return tensor->ne[3] == 1;
}

inline size_t v_used_mem(const struct v_ctx* ctx)
{
  return ctx->objects_end == NULL ? 0 : ctx->objects_end->offs + ctx->objects_end->size;
}

inline bool v_get_no_alloc(struct v_ctx* ctx)
{
  return ctx->no_alloc;
}

inline void v_set_no_alloc(struct v_ctx* ctx,
                           bool no_alloc)
{
  ctx->no_alloc = no_alloc;
}

inline struct v_tensor* v_add(struct v_ctx* ctx,
                              struct v_tensor* a,
                              struct v_tensor* b)
{
  return v_add_imple(ctx, a, b, false);
}

inline struct v_tensor* v_add_inplace(struct v_ctx* ctx,
                                      struct v_tensor* a,
                                      struct v_tensor* b)
{
  return v_add_imple(ctx, a, b, true);
}

inline struct v_tensor* add_cast(struct v_ctx* ctx,
                                 struct v_tensor* a,
                                 struct v_tensor* b,
                                 enum v_data_type type)
{
  return add_cast_impl(ctx, a, b, type);
}


inline struct v_tensor* add1(struct v_ctx* ctx,
                             struct v_tensor* a,
                             struct v_tensor* b)
{
  return add1_impl(ctx, a, b, false);
}

inline struct v_tensor* v_add1_inplace(struct v_ctx* ctx,
                                          struct v_tensor* a,
                                          struct v_tensor* b)
{
  return add1_impl(ctx, a, b, true);
}

inline void* v_get_mem_buffer(const struct v_ctx* ctx)
{
  return ctx->mem_buffer;
}


inline struct v_tensor* v_map_custom3_inplace(struct v_ctx* ctx,
                                                 struct v_tensor* a,
                                                 struct v_tensor* b,
                                                 struct v_tensor* c,
                                                 const v_custom3_op_t fun,
                                                 int n_tasks,
                                                 void* userdata)
{
  return v_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, true);
}

inline struct v_tensor* v_graph_get_grad_acc(const struct v_cgraph* cgraph, const struct v_tensor* node)
{
  const size_t igrad = find_hash(&cgraph->visited_hash_set, node);
  return igrad != v_HASHSET_FULL && v_bit_set_get(cgraph->visited_hash_set.used, igrad) && cgraph->grad_accs
           ? cgraph->grad_accs[igrad]
           : NULL;
}

inline struct v_tensor* v_graph_get_grad(const struct v_cgraph* cgraph, const struct v_tensor* node)
{
  const size_t igrad = find_hash(&cgraph->visited_hash_set, node);
  return igrad != v_HASHSET_FULL && v_bit_set_get(cgraph->visited_hash_set.used, igrad) && cgraph->grads
           ? cgraph->grads[igrad]
           : NULL;
}


inline void v_graph_add_node(struct v_cgraph* cgraph, struct v_tensor* tensor)
{
  V_ASSERT(cgraph->size > cgraph->n_nodes);
  cgraph->nodes[cgraph->n_nodes] = tensor;
  cgraph->n_nodes++;
}

inline struct v_tensor** v_graph_nodes(struct v_cgraph* cgraph)
{
  return cgraph->nodes;
}

inline int v_graph_n_nodes(struct v_cgraph* cgraph)
{
  return cgraph->n_nodes;
}

inline void v_graph_clear(struct v_cgraph* cgraph)
{
  cgraph->n_leafs = 0;
  cgraph->n_nodes = 0;
  v_hash_set_reset(&cgraph->visited_hash_set);
}

inline int v_graph_size(struct v_cgraph* cgraph)
{
  return cgraph->size;
}


inline struct v_cgraph* v_graph_duplicate(struct v_ctx* ctx, struct v_cgraph* cgraph,
                                          bool force_grads)
{
  struct v_cgraph* result = v_new_graph_custom(ctx, cgraph->size, cgraph->grads || force_grads);
  v_copy_graph(cgraph, result);
  return result;
}

inline size_t graph_overhead(void)
{
  return v_graph_overhead_custom(v_DEFAULT_GRAPH_SIZE, false);
}

inline struct v_tensor* v_upscale_ext(struct v_ctx* ctx,
                                         struct v_tensor* a,
                                         int ne0,
                                         int ne1,
                                         int ne2,
                                         int ne3,
                                         enum v_scale_mode mode)
{
  return v_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
};


inline struct v_tensor* v_scale_bias(struct v_ctx* ctx,
                                        struct v_tensor* a,
                                        float s,
                                        float b)
{
  return v_scale_impl(ctx, a, s, b, false);
}

inline struct v_tensor* v_scale_bias_inplace(
  struct v_ctx* ctx,
  struct v_tensor* a,
  float s,
  float b)
{
  return v_scale_impl(ctx, a, s, b, true);
}

inline struct v_tensor* v_norm_l2(struct v_ctx* ctx,
                                     struct v_tensor* a,
                                     float eps)
{
  return v_l2_norm_impl(ctx, a, eps, false);
}

inline struct v_tensor* v_l2_norm_inplace(struct v_ctx* ctx,
                                             struct v_tensor* a,
                                             float eps)
{
  return v_l2_norm_impl(ctx, a, eps, true);
}


inline struct v_tensor* v_interpolate(struct v_ctx* ctx,
                                         struct v_tensor* a,
                                         int64_t ne0,
                                         int64_t ne1,
                                         int64_t ne2,
                                         int64_t ne3,
                                         uint32_t mode)
{
  return v_interpolate_impl(ctx, a, ne0, ne1, ne2, ne3, mode);
}

inline struct v_tensor* v_map_custom3(struct v_ctx* ctx,
                                         struct v_tensor* a,
                                         struct v_tensor* b,
                                         struct v_tensor* c,
                                         const v_custom3_op_t fun,
                                         int n_tasks,
                                         void* userdata)
{
  return v_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, false);
}

inline struct v_tensor* v_map_custom2_inplace(struct v_ctx* ctx,
                                                 struct v_tensor* a,
                                                 struct v_tensor* b,
                                                 const v_custom2_op_t fun,
                                                 int n_tasks,
                                                 void* userdata)
{
  return v_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, true);
}


inline struct v_tensor* v_rms_norm(struct v_ctx* ctx,
                                      struct v_tensor* a,
                                      float eps)
{
  return v_rms_norm_impl(ctx, a, eps, false);
}

inline struct v_tensor* v_rms_norm_inplace(struct v_ctx* ctx,
                                              struct v_tensor* a,
                                              float eps)
{
  return v_rms_norm_impl(ctx, a, eps, true);
}

inline struct v_tensor* v_norm(struct v_ctx* ctx,
                                struct v_tensor* a,
                                float eps)
{
  return v_norm_impl(ctx, a, eps, false);
}

inline struct v_tensor* v_hardswish(struct v_ctx* ctx,
                                       struct v_tensor* a)
{
  return v_unary(ctx, a, v_UNARY_OP_HARDSWISH);
}

inline struct v_tensor* v_hardsigmoid(struct v_ctx* ctx,
                                         struct v_tensor* a)
{
  return v_unary(ctx, a, v_UNARY_OP_HARDSIGMOID);
}

inline struct v_tensor* v_floor(struct v_ctx* ctx,
                                   struct v_tensor* a)
{
  return v_unary(ctx, a, v_UNARY_OP_FLOOR);
}

inline struct v_tensor* v_floor_inplace(struct v_ctx* ctx,
                                           struct v_tensor* a)
{
  return v_unary_inplace(ctx, a, v_UNARY_OP_FLOOR);
}

inline struct v_tensor* v_ceil(struct v_ctx* ctx,
                                  struct v_tensor* a)
{
  return v_unary(ctx, a, v_UNARY_OP_CEIL);
}

inline struct v_tensor* v_ceil_inplace(
  struct v_ctx* ctx,
  struct v_tensor* a)
{
  return v_unary_inplace(ctx, a, v_UNARY_OP_CEIL);
}

inline struct v_tensor* v_round(
  struct v_ctx* ctx,
  struct v_tensor* a)
{
  return v_unary(ctx, a, v_UNARY_OP_ROUND);
}

inline struct v_tensor* v_round_inplace(
  struct v_ctx* ctx,
  struct v_tensor* a)
{
  return v_unary_inplace(ctx, a, v_UNARY_OP_ROUND);
}

inline struct v_tensor* v_trunc(
  struct v_ctx* ctx,
  struct v_tensor* a)
{
  return v_unary(ctx, a, v_UNARY_OP_TRUNC);
}

inline struct v_tensor* v_trunc_inplace(
  struct v_ctx* ctx,
  struct v_tensor* a)
{
  return v_unary_inplace(ctx, a, v_UNARY_OP_TRUNC);
}

inline struct v_tensor* v_glu(struct v_ctx* ctx,
                                 struct v_tensor* a,
                                 enum v_glu_op op,
                                 bool swapped)
{
  return v_glu_impl(ctx, a, NULL, op, swapped);
}

inline struct v_tensor* v_glu_split(struct v_ctx* ctx,
                                       struct v_tensor* a,
                                       struct v_tensor* b,
                                       enum v_glu_op op)
{
  return v_glu_impl(ctx, a, b, op, false);
}

inline struct v_tensor* v_reglu(struct v_ctx* ctx,
                                   struct v_tensor* a)
{
  return v_glu_impl(ctx, a, NULL, v_GLU_OP_REGLU, false);
}

inline struct v_tensor* v_reglu_swapped(struct v_ctx* ctx,
                                           struct v_tensor* a)
{
  return v_glu_impl(ctx, a, NULL, v_GLU_OP_REGLU, true);
}

inline struct v_tensor* v_reglu_split(struct v_ctx* ctx,
                                         struct v_tensor* a,
                                         struct v_tensor* b)
{
  return v_glu_impl(ctx, a, b, v_GLU_OP_REGLU, false);
}

inline struct v_tensor* v_geglu(struct v_ctx* ctx,
                                   struct v_tensor* a)
{
  return v_glu_impl(ctx, a, NULL, v_GLU_OP_GEGLU, false);
}

inline struct v_tensor* v_geglu_swapped(struct v_ctx* ctx,
                                           struct v_tensor* a)
{
  return v_glu_impl(ctx, a, NULL, v_GLU_OP_GEGLU, true);
}

inline struct v_tensor* v_geglu_split(struct v_ctx* ctx,
                                         struct v_tensor* a,
                                         struct v_tensor* b)
{
  return v_glu_impl(ctx, a, b, v_GLU_OP_GEGLU, false);
}

inline struct v_tensor* v_swiglu(struct v_ctx* ctx,
                                    struct v_tensor* a)
{
  return v_glu_impl(ctx, a, NULL, v_GLU_OP_SWIGLU, false);
}

inline struct v_tensor* v_swiglu_swapped(struct v_ctx* ctx,
                                            struct v_tensor* a)
{
  return v_glu_impl(ctx, a, NULL, v_GLU_OP_SWIGLU, true);
}

inline struct v_tensor* v_swiglu_split(
  struct v_ctx* ctx,
  struct v_tensor* a,
  struct v_tensor* b)
{
  return v_glu_impl(ctx, a, b, v_GLU_OP_SWIGLU, false);
}


inline struct v_tensor* v_geglu_erf(
  struct v_ctx* ctx,
  struct v_tensor* a)
{
  return v_glu_impl(ctx, a, NULL, v_GLU_OP_GEGLU_ERF, false);
}

inline struct v_tensor* v_geglu_erf_swapped(
  struct v_ctx* ctx,
  struct v_tensor* a)
{
  return v_glu_impl(ctx, a, NULL, v_GLU_OP_GEGLU_ERF, true);
}

inline struct v_tensor* v_geglu_erf_split(
  struct v_ctx* ctx,
  struct v_tensor* a,
  struct v_tensor* b)
{
  return v_glu_impl(ctx, a, b, v_GLU_OP_GEGLU_ERF, false);
}


inline struct v_tensor* v_geglu_quick(
  struct v_ctx* ctx,
  struct v_tensor* a)
{
  return v_glu_impl(ctx, a, NULL, v_GLU_OP_GEGLU_QUICK, false);
}

inline struct v_tensor* v_geglu_quick_swapped(
  struct v_ctx* ctx,
  struct v_tensor* a)
{
  return v_glu_impl(ctx, a, NULL, v_GLU_OP_GEGLU_QUICK, true);
}

inline struct v_tensor* v_geglu_quick_split(
  struct v_ctx* ctx,
  struct v_tensor* a,
  struct v_tensor* b)
{
  return v_glu_impl(ctx, a, b, v_GLU_OP_GEGLU_QUICK, false);
}

inline struct v_tensor* mmlNormInplace(struct v_ctx* ctx,
                                       struct v_tensor* a,
                                       float eps)
{
  return v_norm_impl(ctx, a, eps, true);
}

inline struct v_tensor* v_map_custom2(struct v_ctx* ctx,
                                         struct v_tensor* a,
                                         struct v_tensor* b,
                                         const v_custom2_op_t fun,
                                         int n_tasks,
                                         void* userdata)
{
  return v_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, false);
}

inline struct v_tensor* v_map_custom1(struct v_ctx* ctx,
                                         struct v_tensor* a,
                                         const v_custom1_op_t fun,
                                         int n_tasks,
                                         void* userdata)
{
  return v_map_custom1_impl(ctx, a, fun, n_tasks, userdata, false);
}

inline struct v_tensor* v_map_custom1_inplace(struct v_ctx* ctx,
                                                 struct v_tensor* a,
                                                 const v_custom1_op_t fun,
                                                 int n_tasks,
                                                 void* userdata)
{
  return v_map_custom1_impl(ctx, a, fun, n_tasks, userdata, true);
}

inline struct v_tensor* v_unary(struct v_ctx* ctx,
                                struct v_tensor* a,
                                enum v_unary_op op)
{
  return v_unary_impl(ctx, a, op, false);
}

inline void* v_get_data(const struct v_tensor* tensor)
{
  return tensor->data;
}

inline float* v_get_tdata_f32(const struct v_tensor* tensor)
{
  assert(tensor->type == v_TYPE_F32);
  return (float*)(tensor->data);
}

inline enum v_unary_op v_get_unary_op(const struct v_tensor* tensor)
{
  V_ASSERT(tensor->op == v_OP_UNARY);
  return (enum v_unary_op)v_get_op_params_i32(tensor, 0);
}

inline enum v_glu_op v_get_glu_op(const struct v_tensor* tensor)
{
  V_ASSERT(tensor->op == v_OP_GLU);
  return (enum v_glu_op)v_get_op_params_i32(tensor, 0);
}

inline const char* get_name(const struct v_tensor* tensor)
{
  return tensor->name;
}

inline struct v_tensor* v_div_inplace(struct v_ctx* ctx,
                                         struct v_tensor* a,
                                         struct v_tensor* b)
{
  return v_div_impl(ctx, a, b, true);
}

inline struct v_tensor* v_sqr(struct v_ctx* ctx,
                                 struct v_tensor* a)
{
  return v_sqr_impl(ctx, a, false);
}

inline struct v_tensor* v_sqr_inplace(struct v_ctx* ctx,
                                         struct v_tensor* a)
{
  return v_sqr_impl(ctx, a, true);
}

inline struct v_tensor* v_graph_node(struct v_cgraph* cgraph, int i)
{
  if (i < 0)
  {
    V_ASSERT(cgraph->n_nodes + i >= 0);
    return cgraph->nodes[cgraph->n_nodes + i];
  }

  V_ASSERT(i < cgraph->n_nodes);
  return cgraph->nodes[i];
}

inline void* incr_ptr_aligned(void** p, size_t size, size_t align)
{
  void* ptr = *p;
  ptr       = (void*)MML_PAD((uintptr_t) ptr, align);
  *p        = (void*)((char*)ptr + size);
  return ptr;
}

// check if t1 can be represented as a repetition of t0


inline struct v_tensor* v_sqrt(struct v_ctx* ctx,
                                struct v_tensor* a)
{
  return v_sqrt_impl(ctx, a, false);
}

inline struct v_tensor* v_sqrt_inplace(struct v_ctx* ctx,
                                          struct v_tensor* a)
{
  return v_sqrt_impl(ctx, a, true);
}

inline bool v_are_same_stride(const struct v_tensor* t0, const struct v_tensor* t1)
{
  static_assert(MML_MAX_DIMS == 4, "v_MAX_DIMS is not 4 - update this function");

  return
    (t0->nb[0] == t1->nb[0]) &&
    (t0->nb[1] == t1->nb[1]) &&
    (t0->nb[2] == t1->nb[2]) &&
    (t0->nb[3] == t1->nb[3]);
}

inline void* v_new_buffer(struct v_ctx* ctx,
                          size_t nbytes)
{
  struct v_object* obj = v_new_object(ctx, MML_BUFFER, nbytes);
  return (uint8_t*)ctx->mem_buffer + obj->offs;
}

inline bool can_repeat_rows(const struct v_tensor* t0, const struct v_tensor* t1)
{
  static_assert(MML_MAX_DIMS == 4, "v_MAX_DIMS is not 4 - update this function");

  return (t0->ne[0] == t1->ne[0]) && can_repeat(t0, t1);
}

inline bool can_repeat(const struct v_tensor* t0, const struct v_tensor* t1)
{
  static_assert(MML_MAX_DIMS == 4, "v_MAX_DIMS is not 4 - update this function");
  return is_empty(t0)
           ? is_empty(t1)
           : (t1->ne[0] % t0->ne[0] == 0) &&
           (t1->ne[1] % t0->ne[1] == 0) &&
           (t1->ne[2] % t0->ne[2] == 0) &&
           (t1->ne[3] % t0->ne[3] == 0);
}

inline struct v_tensor* v_unary_inplace(
  struct v_ctx* ctx,
  struct v_tensor* a,
  enum v_unary_op op)
{
  return v_unary_impl(ctx, a, op, true);
}

inline bool v_can_out_prod(const struct v_tensor* t0, const struct v_tensor* t1)
{
  static_assert(MML_MAX_DIMS == 4, "v_MAX_DIMS is not 4 - update this function");

  return (t0->ne[1] == t1->ne[1]) &&
    (t1->ne[2] % t0->ne[2] == 0) && // verify t0 is broadcastable
    (t1->ne[3] % t0->ne[3] == 0);
}

inline bool can_mul_mat(const struct v_tensor* t0, const struct v_tensor* t1)
{
  static_assert(MML_MAX_DIMS == 4, "v_MAX_DIMS is not 4 - update this function");

  return (t0->ne[0] == t1->ne[0]) &&
    (t1->ne[2] % t0->ne[2] == 0) && // verify t0 is broadcastable
    (t1->ne[3] % t0->ne[3] == 0);
}

inline struct v_tensor* v_silu(
  struct v_ctx* ctx,
  struct v_tensor* a)
{
  return v_unary(ctx, a, v_UNARY_OP_SILU);
}

inline struct v_tensor* v_dup_tensor(struct v_ctx* ctx, const struct v_tensor* src)
{
  return v_new_tensor(ctx, src->type, MML_MAX_DIMS, src->ne);
}

inline struct v_tensor* v_silu_inplace(
  struct v_ctx* ctx,
  struct v_tensor* a)
{
  return v_unary_inplace(ctx, a, v_UNARY_OP_SILU);
}

inline struct v_tensor* v_set_inplace(
  struct v_ctx* ctx,
  struct v_tensor* a,
  struct v_tensor* b,
  size_t nb1,
  size_t nb2,
  size_t nb3,
  size_t offset)
{
  return set_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

inline struct v_tensor* v_set_1d(
  struct v_ctx* ctx,
  struct v_tensor* a,
  struct v_tensor* b,
  size_t offset)
{
  return set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, false);
}

inline struct v_tensor* v_set_1d_inplace(
  struct v_ctx* ctx,
  struct v_tensor* a,
  struct v_tensor* b,
  size_t offset)
{
  return set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, true);
}

inline struct v_tensor* v_set_2d(
  struct v_ctx* ctx,
  struct v_tensor* a,
  struct v_tensor* b,
  size_t nb1,
  size_t offset)
{
  return set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, false);
}

inline struct v_tensor* v_set_2d_inplace(
  struct v_ctx* ctx,
  struct v_tensor* a,
  struct v_tensor* b,
  size_t nb1,
  size_t offset)
{
  return set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, true);
}

inline struct v_tensor* v_gelu_erf(struct v_ctx* ctx,
                                      struct v_tensor* a)
{
  return v_unary(ctx, a, v_UNARY_OP_GELU_ERF);
}

inline struct v_tensor* v_gelu_erf_inplace(struct v_ctx* ctx,
                                              struct v_tensor* a)
{
  return v_unary_inplace(ctx, a, v_UNARY_OP_GELU_ERF);
}

inline struct v_tensor* v_gelu_quick(struct v_ctx* ctx,
                                        struct v_tensor* a)
{
  return v_unary(ctx, a, v_UNARY_OP_GELU_QUICK);
}

inline struct v_tensor* v_gelu_quick_inplace(struct v_ctx* ctx,
                                                struct v_tensor* a)
{
  return v_unary_inplace(ctx, a, v_UNARY_OP_GELU_QUICK);
}

inline struct v_tensor* v_sigmoid(struct v_ctx* ctx,
                                     struct v_tensor* a)
{
  return v_unary(ctx, a, v_UNARY_OP_SIGMOID);
}

inline struct v_tensor* v_sigmoid_inplace(struct v_ctx* ctx,
                                             struct v_tensor* a)
{
  return v_unary_inplace(ctx, a, v_UNARY_OP_SIGMOID);
}

// v_gelu

inline struct v_tensor* v_gelu(struct v_ctx* ctx,
                                  struct v_tensor* a)
{
  return v_unary(ctx, a, v_UNARY_OP_GELU);
}

inline struct v_tensor* v_gelu_inplace(struct v_ctx* ctx,
                                          struct v_tensor* a)
{
  return v_unary_inplace(ctx, a, v_UNARY_OP_GELU);
}

inline struct v_tensor* v_abs(struct v_ctx* ctx,
                                 struct v_tensor* a)
{
  return v_unary(ctx, a, v_UNARY_OP_ABS);
}

inline struct v_tensor* v_abs_inplace(struct v_ctx* ctx,
                                         struct v_tensor* a)
{
  return v_unary_inplace(ctx, a, v_UNARY_OP_ABS);
}

// v_sgn


inline bool v_is_transposed(const struct v_tensor* tensor)
{
  return tensor->nb[0] > tensor->nb[1];
}

inline struct v_tensor* v_sgn(struct v_ctx* ctx,
                                 struct v_tensor* a)
{
  return v_unary(ctx, a, v_UNARY_OP_SGN);
}

inline struct v_tensor* v_sgn_inplace(struct v_ctx* ctx,
                                         struct v_tensor* a)
{
  return v_unary_inplace(ctx, a, v_UNARY_OP_SGN);
}

// v_neg

inline struct v_tensor* v_neg(struct v_ctx* ctx,
                                 struct v_tensor* a)
{
  return v_unary(ctx, a, v_UNARY_OP_NEG);
}

inline struct v_tensor* v_neg_inplace(struct v_ctx* ctx,
                                         struct v_tensor* a)
{
  return v_unary_inplace(ctx, a, v_UNARY_OP_NEG);
}

// v_step

inline struct v_tensor* v_step(struct v_ctx* ctx,
                                  struct v_tensor* a)
{
  return v_unary(ctx, a, v_UNARY_OP_RELU);
}

inline struct v_tensor* v_step_inplace(struct v_ctx* ctx,
                                          struct v_tensor* a)
{
  return v_unary_inplace(ctx, a, v_UNARY_OP_STEP);
}

// v_tanh

inline struct v_tensor* v_tanh(struct v_ctx* ctx,
                                  struct v_tensor* a)
{
  return v_unary(ctx, a, v_UNARY_OP_TANH);
}

inline struct v_tensor* v_tanh_inplace(struct v_ctx* ctx,
                                          struct v_tensor* a)
{
  return v_unary_inplace(ctx, a, v_UNARY_OP_TANH);
}

// v_elu

inline struct v_tensor* v_elu(struct v_ctx* ctx,
                                 struct v_tensor* a)
{
  return v_unary(ctx, a, v_UNARY_OP_ELU);
}

inline struct v_tensor* v_elu_inplace(struct v_ctx* ctx,
                                         struct v_tensor* a)
{
  return v_unary_inplace(ctx, a, v_UNARY_OP_ELU);
}

// v_relu

inline struct v_tensor* v_relu(struct v_ctx* ctx,
                                struct v_tensor* a)
{
  return v_unary(ctx, a, v_UNARY_OP_RELU);
}

inline struct v_tensor* v_relu_inplace(struct v_ctx* ctx,
                                          struct v_tensor* a)
{
  return v_unary_inplace(ctx, a, v_UNARY_OP_RELU);
}


#endif //MYPROJECT_MML_UTIL_H
