#include "ggml-impl.hpp"
#include "v.hpp"

struct v_object* v_new_object(struct v_ctx* ctx,
                              enum v_object_type type,
                              size_t size) {
  struct v_object* obj_cur = ctx->objects_end;
  const size_t cur_offs    = obj_cur == nullptr
                               ? 0
                               : obj_cur->offs;
  const size_t cur_size = obj_cur == nullptr
                            ? 0
                            : obj_cur->size;
  const size_t cur_end     = cur_offs + cur_size;
  size_t size_needed       = V_PAD(size, v_MEM_ALIGN);
  char* const mem_buffer   = (char* const)ctx->mem_buffer;
  struct v_object* obj_new = (struct v_object*)(mem_buffer + cur_end);
  if (cur_end + size_needed + MML_OBJECT_SIZE > ctx->mem_size) {
    v_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                  __func__,
                  cur_end + size_needed + MML_OBJECT_SIZE,
                  ctx->mem_size);
    #ifndef NDEBUG
    v_ABORT("not enough space in the context's memory pool");
    #endif
    return nullptr;
  }
  obj_new           = (struct v_object*)obj_new;
  (*obj_new).offs   = cur_end + MML_OBJECT_SIZE,
    (*obj_new).size = size_needed,
    (*obj_new).next = nullptr,
    (*obj_new).type = type;

  v_ASSERT_ALIGNED(mem_buffer + obj_new->offs);
  if (obj_cur != nullptr) { obj_cur->next = obj_new; }
  else { ctx->objects_begin = obj_new; }
  ctx->objects_end = obj_new;
  //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);
  return obj_new;
}

struct v_ctx* v_ctx_init(struct v_init_param params) {
  static bool is_first_call = true;
  //v_critical_section_start();
  if (is_first_call) {
    v_time_init();
    is_first_call = false;
  }
  //=v_critical_section_end();
  struct v_ctx* ctx = new v_ctx();
  // allow to call v_init with 0 size
  if (params.mem_size == 0) { params.mem_size = v_MEM_ALIGN; }
  const size_t mem_size = params.mem_buffer
                            ? params.mem_size
                            : V_PAD(params.mem_size, v_MEM_ALIGN);
  (*ctx).mem_size   = mem_size;
  (*ctx).mem_buffer = params.mem_buffer
                        ? params.mem_buffer
                        : v_aligned_malloc(mem_size);
  (*ctx).mem_buffer_owned = params.mem_buffer
                              ? false
                              : true;
  (*ctx).no_alloc      = params.no_alloc;
  (*ctx).n_objects     = 0;
  (*ctx).objects_begin = nullptr;
  (*ctx).objects_end   = nullptr;
  V_ASSERT(ctx->mem_buffer != nullptr);
  v_ASSERT_ALIGNED(ctx->mem_buffer);
  //PRINT_DEBUG("%s: context initialized\n", __func__);
  return ctx;
}


void free_ctx(struct v_ctx* ctx) {
  if (ctx == nullptr) { return; }

  if (ctx->mem_buffer_owned) { v_aligned_free(ctx->mem_buffer, ctx->mem_size); }

  delete (ctx);
}


size_t v_get_max_tensor_size(const struct v_ctx* ctx) {
  size_t max_size = 0;
  for (v_tensor* tensor = v_get_first_tensor(ctx); tensor != nullptr; tensor =
       v_get_next_tensor(ctx, tensor)) {
    size_t bytes = num_bytes(tensor);
    max_size     = MAX(max_size, bytes);
  }

  return max_size;
}

v_tensor* v_get_first_tensor(const struct v_ctx* ctx) {
  struct v_object* obj   = ctx->objects_begin;
  char* const mem_buffer = static_cast<char* const>(ctx->mem_buffer);
  while (obj != nullptr) {
    if (obj->type == V_TENSOR) { return reinterpret_cast<v_tensor*>(mem_buffer + obj->offs); }

    obj = obj->next;
  }

  return nullptr;
}

v_tensor* v_get_tensor_name(struct v_ctx* ctx, const char* name) {
  struct v_object* obj   = ctx->objects_begin;
  char* const mem_buffer = (char* const)ctx->mem_buffer;
  while (obj != nullptr) {
    if (obj->type == V_TENSOR) {
      v_tensor* cur = (v_tensor*)(mem_buffer + obj->offs);
      if (strcmp(cur->name.data(), name) == 0) { return cur; }
    }
    obj = obj->next;
  }
  return nullptr;
}

v_tensor* v_get_next_tensor(const struct v_ctx* ctx, v_tensor* tensor) {
  struct v_object* obj   = (struct v_object*)((char*)tensor - MML_OBJECT_SIZE);
  obj                    = obj->next;
  char* const mem_buffer = (char* const)ctx->mem_buffer;
  while (obj != nullptr) {
    if (obj->type == V_TENSOR) { return (v_tensor*)(mem_buffer + obj->offs); }

    obj = obj->next;
  }
  return nullptr;
}
