#pragma once
#include "v-backend.h"
#include "v.h"
#include "v_hash.h"

struct v_tensor_alloc {
  struct v_backend_buffer* buffer;
  void* base;
  size_t alignment;
  size_t offset;
};


struct free_block {
  size_t offset;
  size_t size;
};

#define MAX_FREE_BLOCKS 256

struct tallocr_chunk {
  struct free_block free_blocks[MAX_FREE_BLOCKS];
  int n_free_blocks;
  size_t max_size;
};

struct buffer_address {
  int chunk; // index of a backend buffer
  size_t offset; // local memory offset within the buffer
};

// dynamic tensor allocator
#define v_VBUFFER_MAX_CHUNKS 16

struct dyn_tensor_alloc {
  size_t alignment;
  size_t max_chunk_size;
  struct tallocr_chunk* chunks[v_VBUFFER_MAX_CHUNKS];
  int n_chunks;
};

struct vbuffer {
  struct v_backend_buffer* chunks[v_VBUFFER_MAX_CHUNKS];
};

struct hash_node {
  int n_children;
  int n_views;
  int buffer_id;
  struct buffer_address addr;
  bool allocated;
};

struct tensor_alloc {
  int buffer_id;
  struct buffer_address addr;
  size_t size_max; // 0 = pre-allocated, unused, or view
};

struct leaf_alloc {
  struct tensor_alloc leaf;
};

struct node_alloc {
  struct tensor_alloc dst;
  struct tensor_alloc src[v_MAX_SRC];
};

struct v_graph_alloc {
  struct v_backend_buffer_type** bufts; // [n_buffers]
  struct vbuffer** buffers; // [n_buffers]
  struct dyn_tensor_alloc** buf_tallocs; // [n_buffers]
  int n_buffers;
  struct v_hash_set hash_set;
  struct hash_node* hash_values; // [hash_set.size]
  struct node_alloc* node_allocs; // [n_nodes]
  int n_nodes;
  struct leaf_alloc* leaf_allocs; // [n_leafs]
  int n_leafs;
};


v_API struct v_tensor_alloc v_tallocr_new(struct v_backend_buffer* buffer);
v_API enum v_status tensorAlloc(struct v_tensor_alloc* talloc, struct v_tensor* tensor);
typedef struct v_graph_alloc* v_graph_allocator_t;
v_API v_graph_allocator_t v_gallocr_new(struct v_backend_buffer_type* buft);
v_API v_graph_allocator_t v_gallocr_new_n(struct v_backend_buffer_type** bufts, int n_bufs);
v_API void v_gallocr_free(struct v_graph_alloc* galloc);
v_API bool v_gallocr_reserve_n(struct v_graph_alloc* galloc,
                                  struct v_cgraph* graph,
                                  const int* node_buffer_ids,
                                  const int* leaf_buffer_ids);

// automatic reallocation if the topology changes when using a single buffer
// returns false if using multiple buffers and a re-allocation is needed (call v_gallocr_reserve_n first to set the node buffers)
v_API bool v_gallocr_alloc_graph(v_graph_allocator_t galloc, struct v_cgraph* graph);
v_API size_t v_gallocr_get_buffer_size(v_graph_allocator_t galloc, int buffer_id);
// Utils
// Create a buffer and allocate all the tensors in a v_context
v_API struct v_backend_buffer* v_backend_alloc_ctx_tensor_from_buffer_t(
  struct v_ctx* ctx, struct v_backend_buffer_type* buft);
v_API struct v_backend_buffer* v_backend_alloc_ctx_tensors(struct v_ctx* ctx, struct v_backend* backend);
