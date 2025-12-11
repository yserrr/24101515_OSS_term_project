#include "v_backend.hpp"
#include "v_allocator.hpp"
#include "ggml-impl.hpp"
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include "vk_buffer.h"
#include "v_vk.hpp"
#define hash_id(tensor) sched->hash_set.find_or_insert(tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]


v_backend_sched_t v_sched_new(v_backend_t backend,
                              v_backend_buffer_type_t* bufts,
                              int n_backends, size_t graph_size,
                              bool op_offload) {
  V_ASSERT(n_backends > 0);
  V_ASSERT(n_backends <= v_SCHED_MAX_BACKENDS);
  v_backend_sched* sched    = new v_backend_sched(graph_size);
  const char* v_SCHED_DEBUG = getenv("v_SCHED_DEBUG");
  sched->debug              = v_SCHED_DEBUG ? atoi(v_SCHED_DEBUG) : 0;
  sched->n_backends         = 1;
  sched->n_copies           = 1;
  // initialize hash table
  // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
  sched->hv_tensor_backend_ids = (int*)malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
  sched->hv_tensor_copies      = (v_tensor**)malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(v_tensor*));

  const size_t v_sched_max_splits = graph_size; // at most there is one split for each node in the graph
  const size_t nodes_size         = graph_size + v_sched_max_splits * v_SCHED_MAX_SPLIT_INPUTS * 2;
  sched->node_backend_ids         = (int*)calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
  sched->leaf_backend_ids         = (int*)calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
  sched->context_buffer_size      = v_sched_max_splits * v_SCHED_MAX_SPLIT_INPUTS * 2 * sizeof(v_tensor) +
    v_graph_overhead_custom(graph_size, false);
  sched->context_buffer = (char*)malloc(sched->context_buffer_size);

  const int initial_splits_capacity = 16;
  sched->splits_capacity            = initial_splits_capacity;
  sched->backend                    = *backend;
  sched->bufts[0]                   = bufts ? bufts[0] : vk_device_buffer_type(backend->device->device);
  V_ASSERT(vk_device_supports_buft(backend->device, sched->bufts[0]));
  sched->galloc     = v_gallocr_new_n(sched->bufts, n_backends);
  sched->op_offload = op_offload;
  v_sched_reset(sched);
  return sched;
}

bool v_sched_alloc_graph(v_backend_sched_t sched, v_cgraph* graph) {
  V_ASSERT(sched);
  V_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
  V_ASSERT(!sched->is_alloc);

  sched->cur_copy       = sched->next_copy;
  sched->next_copy      = (sched->next_copy + 1) % sched->n_copies;
  sched->n_splits       = 0;
  sched->n_graph_inputs = 0;
  sched->is_reset       = false;
  v_free_ctx(sched->ctx);
  v_init_param params = {
    /* .mem_size =   */ sched->context_buffer_size,
    /* .mem_buffer = */ sched->context_buffer,
    /* .no_alloc =   */ true
  };
  sched->ctx = v_ctx_init(params);
  if (sched->ctx == nullptr) { v_ABORT("%s: failed to initialize context\n", __func__); }
  for (int i = 0; i < graph->n_leafs; i++) {
    v_tensor* leaf       = graph->leafs[i];
    int* leaf_backend_id = &tensor_backend_id(leaf);
    if (*leaf_backend_id == -1) { *leaf_backend_id = 0; }
  }
  for (int i = 0; i < graph->n_nodes; i++) {
    v_tensor* node       = graph->nodes[i];
    int* node_backend_id = &tensor_backend_id(node);
    // do not overwrite user assignments
    if (*node_backend_id == -1) { *node_backend_id = 0; }
  }
  if (!v_gallocr_alloc_graph(sched->galloc, graph)) {
    // the re-allocation may cause the split inputs to be moved to a different address
    // synchronize without v_backend_sched_synchronize to avoid changing cur_copy
    v_gallocr_reserve_n(sched->galloc, graph, sched->node_backend_ids, sched->leaf_backend_ids);
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
  return buft->host
           ? vk_host_buffer_get_align(buft)
           : vk_device_buffer_get_align(buft);
}

size_t v_get_backend_buffer_max_size(v_backend_buffer_type_t buft) {
  V_ASSERT(buft);
  return buft->host
           ? SIZE_MAX
           : vk_device_buffer_get_max_size(buft);
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

void v_backend_buffer_set_usage(v_backend_buffer_t buffer, enum v_backend_buffer_usage usage) {
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
  v_free_ctx(sched->ctx);
  free(sched->hv_tensor_backend_ids);
  free(sched->hv_tensor_copies);
  free(sched->node_backend_ids);
  free(sched->leaf_backend_ids);;
  free(sched->context_buffer);
  free(sched);
}

void v_sched_reset(v_backend_sched_t sched) {
  V_ASSERT(sched);
  // reset state for the next run
  if (!sched->is_reset) {
    v_hash_set_reset(&sched->hash_set);
    memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    memset(sched->hv_tensor_copies, 0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(v_tensor*));
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
  V_ASSERT(i >= 0 && i < sched->n_backends);
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

void vk_host_buffer_memset_tensor(v_backend_buffer_t buffer, v_tensor* tensor,
                                  uint8_t value, size_t offset, size_t size) {
  V_ASSERT(tensor);
  memset((char*)tensor->data + offset, value, size);
  V_UNUSED(buffer);
}

void vk_host_buffer_set_tensor(v_backend_buffer_t buffer, v_tensor* tensor,
                               const void* data, size_t offset, size_t size) {
  V_ASSERT(tensor);
  memcpy((char*)tensor->data + offset, data, size);
  V_UNUSED(buffer);
}

void vk_host_buffer_get_tensor(v_backend_buffer_t buffer, const v_tensor* tensor,
                               void* data, size_t offset, size_t size) {
  V_ASSERT(tensor);
  memcpy(data, (const char*)tensor->data + offset, size);
  V_UNUSED(buffer);
}


v_backend_buffer_type_t v_backend_cpu_buffer_type(void) {
  static struct v_backend_buffer_type v_backend_cpu_buffer_type = {
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
