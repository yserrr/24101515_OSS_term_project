#pragma once
#include "v.h"
#include "v_allocator.h"


typedef struct v_backend_buffer_type* v_backend_buffer_type_t;
typedef struct v_backend_buffer* v_backend_buffer_t;
typedef struct v_backend* v_backend_t;

const char* v_get_backend_buffer_name(v_backend_buffer_type_t buft);
v_backend_buffer_t v_backend_buffer_alloc(v_backend_buffer_type_t buft, size_t size);
size_t v_get_backend_buffer_align(v_backend_buffer_type_t buft);
size_t v_get_backend_buffer_max_size(v_backend_buffer_type_t buft);
size_t v_get_backend_buffer_alloc_size(v_backend_buffer_type_t buft,const v_tensor*  tensor);
bool mmlBackendBufferIsHostVisible(v_backend_buffer_type_t buft);
struct vk_device_ctx* v_backend_buft_get_device(v_backend_buffer_type_t buft);
void vk_device_buffer_memset_tensor(v_backend_buffer_t buffer, v_tensor* tensor, uint8_t value,
                                    size_t offset, size_t size);

const char* v_backend_buffer_name(v_backend_buffer_t buffer);
void v_backend_buffer_free(v_backend_buffer_t buffer);
void* v_backend_buffer_get_base(v_backend_buffer_t buffer);
size_t v_backend_buffer_get_size(v_backend_buffer_t buffer);
size_t v_backend_buffer_get_alignment(v_backend_buffer_t buffer);
size_t v_backend_buffer_get_max_size(v_backend_buffer_t buffer);
void v_backend_buffer_clear(v_backend_buffer_t buffer, uint8_t value);
bool v_backend_buffer_is_host(v_backend_buffer_t buffer);

v_backend_buffer_type_t v_backend_buffer_get_type(v_backend_buffer_t buffer);
void v_backend_buffer_reset(v_backend_buffer_t buffer);

// tensor copy between different backends
void v_backend_tensor_cpy(v_tensor* src, v_tensor* dst);
void v_backend_free(v_backend_t backend);
void v_set_backend_tensor(v_tensor* tensor, const void* data, size_t offset, size_t size);
void v_get_backend_tensor(const v_tensor* tensor, void* data, size_t offset, size_t size);
void v_backend_tensor_memset(v_tensor* tensor, uint8_t value, size_t offset, size_t size);
const char* vk_device_buffer_name(v_backend_buffer_type_t buft);
v_backend_buffer_t vk_device_buffer_alloc(v_backend_buffer_type_t buft, size_t size);
size_t vk_device_buffer_get_align(v_backend_buffer_type_t buft);
size_t vk_device_buffer_get_max_size(v_backend_buffer_type_t buft);
size_t vk_device_buffer_get_alloc_size(v_backend_buffer_type_t buft, v_tensor* const tensor);
// (optional) check if tensor data is in host memory and uses standard ggml tensor layout (defaults to false)


typedef struct v_backend_sched* v_backend_sched_t;
v_backend_sched_t v_sched_new(v_backend_t backend, v_backend_buffer_type_t* bufts,
                              int n_backends, size_t graph_size, bool parallel, bool op_offload);
void v_sched_free(v_backend_sched_t sched);
v_backend_t v_sched_get_backend(v_backend_sched_t sched, int i);
bool v_sched_alloc_graph(v_backend_sched_t sched, struct v_cgraph* graph);
enum v_status v_sched_graph_compute(v_backend_sched_t sched, struct v_cgraph* graph);
v_API void v_sched_reset(v_backend_sched_t sched);
v_API enum v_status v_backend_tensor_alloc(v_backend_buffer_t buffer, v_tensor* tensor, void* addr);
v_API enum v_status v_backend_tensor_view_init(v_tensor* tensor);
v_API v_backend_buffer_t v_backend_cpu_buffer_from_ptr(void* ptr, size_t size);
v_API v_backend_buffer_type_t v_backend_cpu_buffer_type(void);
struct vk_device_ctx;
struct vk_buffer_type_context;
struct vk_backend_ctx;
void* vk_device_buffer_memset_tensor(v_backend_buffer_t buffer);
void vk_device_buffer_memset_tensor(v_backend_buffer_t buffer, v_tensor* tensor, uint8_t value, size_t offset, size_t size);
void vk_device_buffer_set_tensor(v_backend_buffer_t buffer, v_tensor* tensor, const void* data, size_t offset, size_t size);
void vk_device_buffer_get_tensor(v_backend_buffer_t buffer, v_tensor*  tensor, void* data, size_t offset, size_t size);
bool vk_buffer_cpy_tensor(v_backend_buffer_t buffer, v_tensor*  src, v_tensor* dst);
void vk_buffer_clear(v_backend_buffer_t buffer, uint8_t value);

struct v_backend_vk_buffer_ctx;
enum v_backend_buffer_usage {
  v_BACKEND_BUFFER_USAGE_ANY     = 0,
  v_BACKEND_BUFFER_USAGE_WEIGHTS = 1,
  v_BACKEND_BUFFER_USAGE_COMPUTE = 2,
};

struct v_backend_buffer_type {
  struct vk_device_ctx* device;
  vk_buffer_type_context* context;
  bool host;
};

struct v_backend_buffer {
  v_backend_buffer_type_t buft;
  void* context;
  size_t size;
  enum v_backend_buffer_usage usage;
};

struct v_backend {
  vk_device_ctx* device;
  vk_backend_ctx* context;
};

v_API v_backend_buffer_t v_backend_buffer_init(
  v_backend_buffer_type_t buft,
  void* context,
  size_t size);


bool vk_device_supports_buft(struct vk_device_ctx* dev, v_backend_buffer_type_t buft);
struct vk_device_ctx* v_backend_vk_reg_get_device(size_t index);
void v_backend_buffer_set_usage(v_backend_buffer_t buffer, enum v_backend_buffer_usage usage);
enum v_backend_buffer_usage v_backend_buffer_get_usage(v_backend_buffer_t buffer);
