#include "v_backend.hpp"
#include "v_allocator.hpp"
#include "ggml-impl.hpp"
#include <cassert>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <vector>
#include "vk_buffer.h"
#include "v_vk.hpp"

v_backend_sched_t v_sched_new(v_backend_t backend,
                              v_backend_buffer_type_t* bufts,
                              size_t graph_size) {
  v_backend_sched* sched = new v_backend_sched(graph_size);
  // initialize hash table
  sched->backend  = *backend;
  sched->bufts[0] = bufts ? bufts[0] : vk_device_buffer_type(backend->device->device);
  V_ASSERT(vk_device_supports_buft(backend->device, sched->bufts[0]));
  sched->galloc = v_gallocr_new_n(sched->bufts, 1);
  v_sched_reset(sched);
  return sched;
}

bool v_sched_alloc_graph(v_backend_sched_t sched, v_cgraph* graph) {
  V_ASSERT(sched);
  V_ASSERT(!sched->is_alloc);
  sched->is_reset = false;
  if (!v_gallocr_alloc_graph(sched->galloc, graph)) {
    // the re-allocation may cause the split inputs to be moved to a different address
    // synchronize without v_backend_sched_synchronize to avoid changing cur_copy
    v_gallocr_reserve_n(sched->galloc, graph);
    if (!v_gallocr_alloc_graph(sched->galloc, graph)) {
      V_LOG_ERROR("%s: failed to allocate graph\n", __func__);
      return false;
    }
  }
  sched->is_alloc = true;
  return true;
}

const char* vk_host_buffer_name(v_backend_buffer_type_t buft);

const char* v_get_backend_buffer_name(v_backend_buffer_type_t buft) {
  V_ASSERT(buft);
  return buft->host ? vk_host_buffer_name(buft) : vk_device_buffer_name(buft);
}

v_backend_buffer_t vk_host_buffer_alloc(v_backend_buffer_type_t buft, size_t size);

v_backend_buffer_t v_backend_buffer_alloc(v_backend_buffer_type_t buft, size_t size) {
  if (size == 0) {
    // return a dummy buffer for zero-sized allocations
    return v_backend_buffer_init(buft, nullptr, 0);
  }
  V_ASSERT(buft);
  return buft->host ? vk_host_buffer_alloc(buft, size) : vk_device_buffer_alloc(buft, size);
}

size_t v_get_backend_buffer_align(v_backend_buffer_type_t buft) {
  return buft->host ? vk_host_buffer_get_align(buft) : vk_device_buffer_get_align(buft);
}

size_t v_get_backend_buffer_max_size(v_backend_buffer_type_t buft) {
  V_ASSERT(buft);
  return buft->host ? SIZE_MAX : vk_device_buffer_get_max_size(buft);
}

size_t v_get_backend_buffer_alloc_size(v_backend_buffer_type_t buft,
                                       const v_tensor* tensor) {
  V_ASSERT(buft);
  return num_bytes(tensor);
}


v_backend_buffer_t v_backend_buffer_init(v_backend_buffer_type_t buft,
                                         void* context,
                                         size_t size) {
  v_backend_buffer_t buffer = new v_backend_buffer{
    /* .buft      = */ buft,
    /* .context   = */ static_cast<v_backend_vk_buffer_ctx*>(context),
    /* .size      = */ size,
    /* .usage     = */ V_BACKEND_BUFFER_USAGE_ANY
  };
  return buffer;
}


void v_backend_buffer_free(v_backend_buffer_t buffer) {
  if (buffer == nullptr) { return; }
  vk_free_buffer(buffer);
  delete buffer;
}

size_t v_backend_buffer_get_size(v_backend_buffer_t buffer) {
  V_ASSERT(buffer);
  return buffer->size;
}

void* vk_host_buffer_get_base(v_backend_buffer_t buffer);

void* v_backend_buffer_get_base(v_backend_buffer_t buffer) {
  V_ASSERT(buffer);
  if (buffer->size == 0) { return nullptr; }
  void* base = buffer->buft->host ? vk_host_buffer_get_base(buffer) : vk_device_buffer_get_base(buffer);
  V_ASSERT(base != nullptr && "backend buffer base cannot be nullptr");
  return base;
}

bool v_backend_buffer_is_host(v_backend_buffer_t buffer) {
  printf("host call checked \n");
  return buffer->buft->host;
}

void v_backend_buffer_set_usage(v_backend_buffer_t buffer, v_backend_buffer_usage usage) {
  V_ASSERT(buffer);
  buffer->usage = usage;
}


void v_backend_free(v_backend_t backend) {
  if (backend == nullptr) return;
  delete backend;
}


void v_sched_free(v_backend_sched_t sched) {
  if (sched == nullptr) return;
  v_gallocr_free(sched->galloc);
  free(sched);
}

void v_sched_reset(v_backend_sched_t sched) {
  V_ASSERT(sched);
  if (!sched->is_reset) {
    sched->hash_set.clear();
    sched->is_reset = true;
  }
  sched->is_alloc = false;
}


v_status v_sched_graph_compute(v_backend_sched_t sched, v_cgraph* graph) {
  V_ASSERT(sched);
  if (!sched->is_reset && !sched->is_alloc) { v_sched_reset(sched); }
  if (!sched->is_alloc) { if (!v_sched_alloc_graph(sched, graph)) { throw std::runtime_error("allocation fail"); } }
  V_ASSERT(sched);
  v_status ec = vk_graph_compute(&sched->backend, graph);
  return ec;
}


v_backend_t v_sched_get_backend(v_backend_sched_t sched, int i) {
  V_ASSERT(sched);
  return &sched->backend;
}


v_status v_backend_tensor_view_init(v_tensor* tensor) {
  V_ASSERT(tensor);
  V_ASSERT(tensor->buffer == nullptr);
  V_ASSERT(tensor->view_src != nullptr);
  V_ASSERT(tensor->view_src->buffer != nullptr);
  V_ASSERT(tensor->view_src->data != nullptr);

  tensor->buffer = tensor->view_src->buffer;
  tensor->data   = static_cast<std::byte*>(tensor->view_src->data) + tensor->view_offs;
  return vk_buffer_init_tensor(tensor->buffer, tensor);
}

v_status v_backend_tensor_alloc(v_backend_buffer_t buffer,
                                v_tensor* tensor,
                                void* addr) {
  V_ASSERT(tensor);
  V_ASSERT(tensor->buffer == nullptr);
  V_ASSERT(tensor->data == nullptr);
  V_ASSERT(tensor->view_src == nullptr);
  V_ASSERT(addr >= v_backend_buffer_get_base(buffer));
  V_ASSERT(static_cast<char*>(addr) + num_bytes(tensor) <=
    static_cast<char*>(v_backend_buffer_get_base(buffer)) + v_backend_buffer_get_size(buffer));

  tensor->buffer = buffer;
  tensor->data   = addr;
  return vk_buffer_init_tensor(buffer, tensor);
}


void* vk_host_buffer_get_base(v_backend_buffer_t buffer) {
  V_ASSERT(buffer);
  uintptr_t data = (uintptr_t)buffer->context;
  if (data % TENSOR_ALIGNMENT != 0) { data = V_PAD(data, TENSOR_ALIGNMENT); }
  return (void*)data;
}

void vk_host_buffer_memset_tensor(v_backend_buffer_t buffer,
                                  v_tensor* tensor,
                                  uint8_t value, size_t offset, size_t size) {
  V_ASSERT(tensor);
  memset((char*)tensor->data + offset, value, size);
  V_UNUSED(buffer);
}

void vk_host_buffer_set_tensor(v_backend_buffer_t buffer,
                               v_tensor* tensor,
                               const void* data, size_t offset, size_t size) {
  V_ASSERT(tensor);
  memcpy((char*)tensor->data + offset, data, size);
  V_UNUSED(buffer);
}

void vk_host_buffer_get_tensor(v_backend_buffer_t buffer,
                               const v_tensor* tensor,
                               void* data, size_t offset, size_t size) {
  V_ASSERT(tensor);
  memcpy(data, (const char*)tensor->data + offset, size);
  V_UNUSED(buffer);
}


v_backend_buffer_type_t v_backend_cpu_buffer_type(void) {
  static v_backend_buffer_type v_backend_cpu_buffer_type = {
    /* .device  = */ nullptr, // FIXME v_backend_reg_dev_get(v_backend_cpu_reg(), 0),
    /* .context = */ nullptr,
  };

  return &v_backend_cpu_buffer_type;
}


static v_backend_buffer_type_t v_backend_cpu_buffer_from_ptr_type(void) {
  static v_backend_buffer_type v_backend_cpu_buffer_type = {
    /* .device  = */ nullptr, // FIXME v_backend_reg_dev_get(v_backend_cpu_reg(), 0),
    /* .context = */ nullptr,
  };
  return &v_backend_cpu_buffer_type;
}

v_backend_buffer_t v_backend_cpu_buffer_from_ptr(void* ptr, size_t size) {
  V_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
  return v_backend_buffer_init(v_backend_cpu_buffer_from_ptr_type(), ptr, size);
}
