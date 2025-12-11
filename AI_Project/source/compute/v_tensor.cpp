#include "ggml-impl.hpp"
#include "v_header.hpp"
#include "v.hpp"

void v_tensor::set_inputs() {
  flags |= TENSOR_FLAG_INPUT;
}

void v_tensor::set_outputs() {
  flags |= TENSOR_FLAG_OUTPUT;
}

void v_tensor::set_params() {
  V_ASSERT(this->op == v_OP_NONE);
  flags |= TENSOR_FLAG_PARAM;
}

void v_tensor::set_name(const char* name) {
  size_t i;
  for (i = 0; i < sizeof(name) - 1 && name[i] != '\0'; i++) {
    this->name[i] = name[i];
  }
  this->name[i] = '\0';
}

bool v_tensor::is_empty() const {
  for (int i = 0; i < V_MAX_DIMS; ++i) {
    if (ne[i] == 0) {
      // empty if any dimension has no elements
      return true;
    }
  }
  return false;
}

bool v_tensor::is_transposed() {
  return nb[0] > nb[1];
}

v_tensor* v_new_tensor(v_ctx* ctx,
                       v_data_type type,
                       int n_dims,
                       const int64_t* ne) {
  return v_new_tensor_impl(ctx, type, n_dims, ne, nullptr, 0);
}


v_tensor* v_new_tensor_1d(v_ctx* ctx,
                          v_data_type type,
                          int64_t ne0) {
  return v_new_tensor(ctx, type, 1, &ne0);
}

v_tensor* v_new_tensor_2d(v_ctx* ctx,
                          v_data_type type,
                          int64_t ne0,
                          int64_t ne1) {
  const int64_t ne[2] = {ne0, ne1};
  return v_new_tensor(ctx, type, 2, ne);
}


v_tensor* v_new_tensor_3d(v_ctx* ctx,
                          v_data_type type,
                          int64_t ne0, int64_t ne1, int64_t ne2) {
  const int64_t ne[3] = {ne0, ne1, ne2};
  return v_new_tensor(ctx, type, 3, ne);
}

v_tensor* v_new_tensor_4d(v_ctx* ctx,
                          v_data_type type,
                          int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
  const int64_t ne[4] = {ne0, ne1, ne2, ne3};
  return v_new_tensor(ctx, type, 4, ne);
}

v_tensor* v_new_tensor_impl(v_ctx* ctx,
                            v_data_type type,
                            int n_dims,
                            const int64_t* ne,
                            v_tensor* view_src, size_t view_offs) {
  V_ASSERT(type >= 0 && type < V_TYPE_COUNT);
  V_ASSERT(n_dims >= 1 && n_dims <= V_MAX_DIMS);
  if (view_src != nullptr && view_src->view_src != nullptr) {
    view_offs += view_src->view_offs; // find the base tensor and absolute offset
    view_src = view_src->view_src;
  }
  size_t data_size = v_row_size(type, ne[0]);
  for (int i = 1; i < n_dims; i++) {
    data_size *= ne[i];
  }
  V_ASSERT(view_src == nullptr || data_size == 0 || data_size + view_offs <= num_bytes(view_src));
  void* data = view_src != nullptr ? view_src->data : nullptr;
  if (data != nullptr) {
    data = static_cast<std::byte*>(data) + view_offs;
  }
  size_t obj_alloc_size = 0;
  if (view_src == nullptr && !ctx->no_alloc) {
    obj_alloc_size = data_size; // allocate tensor data in the context's memory pool
  }
  v_object* const obj_new = v_new_object(ctx, V_TENSOR, V_TENSOR_SIZE + obj_alloc_size);
  V_ASSERT(obj_new);

  v_tensor* result  = reinterpret_cast<v_tensor*>(static_cast<std::byte*>(ctx->mem_buffer) + obj_new->offs);
  result->type      = type;
  result->buffer    = nullptr;
  result->ne[0]     = 1;
  result->ne[1]     = 1;
  result->ne[2]     = 1;
  result->ne[3]     = 1;
  result->nb[0]     = 0;
  result->nb[1]     = 0;
  result->nb[2]     = 0;
  result->nb[3]     = 0;
  result->op        = v_OP_NONE;
  result->flags     = 0;
  result->view_src  = view_src;
  result->view_offs = view_offs;
  result->data      = obj_alloc_size > 0 ? static_cast<void*>(result + 1) : data;
  result->src.fill(nullptr);
  result->name.fill(NULL);
  result->op_params.fill(NULL);
  for (int i = 0; i < n_dims; i++) {
    result->ne[i] = ne[i];
  }
  result->nb[0] = v_type_size(type);
  result->nb[1] = result->nb[0] * (result->ne[0] / block_size(type));
  for (int i = 2; i < V_MAX_DIMS; i++) {
    result->nb[i] = result->nb[i - 1] * result->ne[i - 1];
  }
  ctx->n_objects++;
  return result;
}

v_tensor* v_dup_tensor(v_ctx* ctx, const v_tensor* src) {
  return v_new_tensor(ctx, src->type, V_MAX_DIMS, src->ne.data());
}


v_tensor* v_reshape(v_ctx* ctx,
                    v_tensor* a, v_tensor* b) {
  V_ASSERT(v_is_contiguous(a));
  // as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
  V_ASSERT(nelements(a) == nelements(b));
  v_tensor* result = v_new_tensor_impl(ctx, a->type, V_MAX_DIMS, b->ne.data(), a, 0);
  v_format_name(result, "%s (reshaped)", a->name);
  result->op     = V_OP_RESHAPE;
  result->src[0] = a;
  return result;
}

v_tensor* v_reshape_1d(v_ctx* ctx,
                       v_tensor* a,
                       int64_t ne0) {
  V_ASSERT(v_is_contiguous(a));
  V_ASSERT(nelements(a) == ne0);
  const int64_t ne[1] = {ne0};
  v_tensor* result    = v_new_tensor_impl(ctx, a->type, 1, ne, a, 0);
  v_format_name(result, "%s (reshaped)", a->name);
  result->op     = V_OP_RESHAPE;
  result->src[0] = a;
  return result;
}

v_tensor* v_reshape_2d(v_ctx* ctx,
                       v_tensor* a,
                       int64_t ne0,
                       int64_t ne1) {
  V_ASSERT(v_is_contiguous(a));
  V_ASSERT(nelements(a) == ne0*ne1);

  const int64_t ne[2] = {ne0, ne1};
  v_tensor* result    = v_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
  v_format_name(result, "%s (reshaped)", a->name);

  result->op     = V_OP_RESHAPE;
  result->src[0] = a;

  return result;
}

v_tensor* v_reshape_3d(v_ctx* ctx,
                       v_tensor* a,
                       int64_t ne0,
                       int64_t ne1,
                       int64_t ne2) {
  V_ASSERT(v_is_contiguous(a));
  V_ASSERT(nelements(a) == ne0*ne1*ne2);

  const int64_t ne[3] = {ne0, ne1, ne2};
  v_tensor* result    = v_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
  v_format_name(result, "%s (reshaped)", a->name);

  result->op     = V_OP_RESHAPE;
  result->src[0] = a;

  return result;
}

v_tensor* v_reshape_4d(v_ctx* ctx,
                       v_tensor* a,
                       int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
  V_ASSERT(v_is_contiguous(a));
  V_ASSERT(nelements(a) == ne0*ne1*ne2*ne3);
  const int64_t ne[4] = {ne0, ne1, ne2, ne3};
  v_tensor* result    = v_new_tensor_impl(ctx, a->type, 4, ne, a, 0);
  v_format_name(result, "%s (reshaped)", a->name);
  result->op     = V_OP_RESHAPE;
  result->src[0] = a;
  return result;
}

v_tensor* v_tensor_view(v_ctx* ctx,
                        v_tensor* src) {
  v_tensor* result = v_new_tensor_impl(ctx, src->type, V_MAX_DIMS, src->ne.data(), src, 0);
  v_format_name(result, "%s (view)", src->name);
  for (int i = 0; i < V_MAX_DIMS; i++) {
    result->nb[i] = src->nb[i];
  }
  return result;
}

v_tensor* v_view_impl(v_ctx* ctx,
                      v_tensor* a,
                      int n_dims, const int64_t* ne,
                      size_t offset) {
  v_tensor* result = v_new_tensor_impl(ctx, a->type, n_dims, ne, a, offset);
  v_format_name(result, "%s (view)", a->name);
  v_set_op_params(result, &offset, sizeof(offset));
  result->op     = V_OP_VIEW;
  result->src[0] = a;
  return result;
}

v_tensor* v_view_1d(v_ctx* ctx,
                    v_tensor* a,
                    int64_t ne0,
                    size_t offset) {
  v_tensor* result = v_view_impl(ctx, a, 1, &ne0, offset);
  return result;
}

v_tensor* v_view_2d(v_ctx* ctx,
                    v_tensor* a,
                    int64_t ne0, int64_t ne1,
                    size_t nb1,
                    size_t offset) {
  const int64_t ne[2] = {ne0, ne1};
  v_tensor* result    = v_view_impl(ctx, a, 2, ne, offset);
  result->nb[1]       = nb1;
  result->nb[2]       = result->nb[1] * ne1;
  result->nb[3]       = result->nb[2];
  return result;
}

v_tensor* v_view_3d(v_ctx* ctx,
                    v_tensor* a,
                    int64_t ne0, int64_t ne1, int64_t ne2,
                    size_t nb1, size_t nb2,
                    size_t offset) {
  const int64_t ne[3] = {ne0, ne1, ne2};
  v_tensor* result    = v_view_impl(ctx, a, 3, ne, offset);
  result->nb[1]       = nb1;
  result->nb[2]       = nb2;
  result->nb[3]       = result->nb[2] * ne2;
  return result;
}

v_tensor* v_view_4d(v_ctx* ctx,
                    v_tensor* a,
                    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                    size_t nb1, size_t nb2, size_t nb3,
                    size_t offset) {
  const int64_t ne[4] = {ne0, ne1, ne2, ne3};
  v_tensor* result    = v_view_impl(ctx, a, 4, ne, offset);
  result->nb[1]       = nb1;
  result->nb[2]       = nb2;
  result->nb[3]       = nb3;
  return result;
}

bool v_are_same_shape(const v_tensor* t0, const v_tensor* t1) {
  return
    (t0->ne[0] == t1->ne[0]) &&
    (t0->ne[1] == t1->ne[1]) &&
    (t0->ne[2] == t1->ne[2]) &&
    (t0->ne[3] == t1->ne[3]);
}


bool v_are_same_stride(const v_tensor* t0, const v_tensor* t1) {
  return
    (t0->nb[0] == t1->nb[0]) &&
    (t0->nb[1] == t1->nb[1]) &&
    (t0->nb[2] == t1->nb[2]) &&
    (t0->nb[3] == t1->nb[3]);
}

v_tensor* v_cont(v_ctx* ctx,
                 v_tensor* a) {
  v_tensor* result = v_dup_tensor(ctx, a);
  v_format_name(result, "%s (cont)", a->name);
  result->op     = V_OP_CONT;
  result->src[0] = a;
  return result;
}

v_tensor* v_cont_1d(v_ctx* ctx,
                    v_tensor* a,
                    int64_t ne0) {
  return v_cont_4d(ctx, a, ne0, 1, 1, 1);
}

v_tensor* v_cont_2d(v_ctx* ctx,
                    v_tensor* a,
                    int64_t ne0, int64_t ne1) {
  return v_cont_4d(ctx, a, ne0, ne1, 1, 1);
}

v_tensor* v_cont_3d(v_ctx* ctx,
                    v_tensor* a,
                    int64_t ne0, int64_t ne1, int64_t ne2) {
  return v_cont_4d(ctx, a, ne0, ne1, ne2, 1);
}

v_tensor* v_cont_4d(v_ctx* ctx,
                    v_tensor* a,
                    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
  V_ASSERT(nelements(a) == (ne0*ne1*ne2*ne3));
  v_tensor* result = v_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
  v_format_name(result, "%s (cont)", a->name);
  result->op     = V_OP_CONT;
  result->src[0] = a;
  return result;
}


v_tensor* v_permute(v_ctx* ctx,
                    v_tensor* a,
                    int axis0, int axis1, int axis2, int axis3) {
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

  result->op       = V_OP_PERMUTE;
  result->src[0]   = a;
  int32_t params[] = {axis0, axis1, axis2, axis3};
  v_set_op_params(result, params, sizeof(params));
  return result;
}

v_tensor* v_transpose(v_ctx* ctx,
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
