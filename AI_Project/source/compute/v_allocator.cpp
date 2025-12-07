#include <assert.h>
#include <filesystem>
#include <limits.h>
#include <ranges>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "v_allocator.h"
#include "v.h"
#include "ggml-impl.h"
#include "v_vk.h"
#include "v_util.h"
#define MAX(a, b) ((a) > (b) ? (a) : (b))
//#define v_ALLOCATOR_DEBUG
//#define AT_PRINTF(...) v_LOG_DEBUG(__VA_ARGS__)
#define AT_PRINTF(...)
/// 12/08 need to check refactor
static bool v_is_view(const v_tensor* t) { return t->view_src != NULL; }

bool op_can_inplace(enum v_operation op) {
  switch (op) {
    case V_OP_SCALE:
    case V_OP_DIAG_MASK_ZERO:
    case V_OP_DIAG_MASK_INF:
    case v_OP_ADD:
    case v_OP_ADD_ID:
    case v_OP_ADD1:
    case v_OP_SUB:
    case v_OP_MUL:
    case v_OP_DIV:
    case v_OP_SQR:
    case v_OP_SQRT:
    case v_OP_LOG:
    case v_OP_UNARY:
    case V_OP_ROPE:
    case v_OP_ROPE_BACK:
    case v_OP_SILU_BACK:
    case v_OP_RMS_NORM:
    case v_OP_RMS_NORM_BACK:
    case V_OP_SOFT_MAX:
    case v_OP_SOFT_MAX_BACK:
      return true;

    default:
      return false;
  }
}

static size_t aligned_offset(const void* buffer, size_t offset, size_t alignment) {
  assert(alignment && !(alignment & (alignment - 1))); // power of 2
  size_t align = (alignment - (((uintptr_t)buffer + offset) % alignment)) % alignment;
  return offset + align;
}

// tallocr

v_tensor_alloc v_tallocr_new(v_backend_buffer_t buffer) {
  void* base   = v_backend_buffer_get_base(buffer);
  size_t align = v_get_backend_buffer_align(buffer->buft);;
  assert(align && !(align & (align - 1))); // power of 2
  v_tensor_alloc talloc = v_tensor_alloc{
    .buffer = buffer,
    .base = base,
    .alignment = align,
    .offset = aligned_offset(base, 0, align),
  };
  return talloc;
}

enum v_status tensorAlloc(struct v_tensor_alloc* talloc, v_tensor* tensor) {
  size_t size = num_bytes(tensor);
  size        = MML_PAD(size, talloc->alignment);
  if (talloc->offset + size > v_backend_buffer_get_size(talloc->buffer)) {
    v_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                __func__,
                tensor->name,
                size,
                v_backend_buffer_get_size(talloc->buffer) - talloc->offset);
    v_ABORT("not enough space in the buffer");
  }
  //void* addr = (char*)v_backend_buffer_get_base(talloc->buffer) + talloc->offset;
  void* addr = (char*)v_backend_buffer_get_base(talloc->buffer) + talloc->offset;

  talloc->offset += size;
  assert(((uintptr_t)addr % talloc->alignment) == 0);
  return v_backend_tensor_alloc(talloc->buffer, tensor, addr);
}

// relative memory address within an allocation that can be split into multiple buffers (chunks)


static const struct buffer_address v_BUFFER_ADDRESS_INVALID = {-1, SIZE_MAX};

static bool v_buffer_address_less(struct buffer_address a, struct buffer_address b) {
  return a.chunk != b.chunk ? a.chunk < b.chunk : a.offset < b.offset;
}


static void v_dyn_tallocr_remove_block(struct tallocr_chunk* chunk, int idx) {
  // shift all elements after idx by 1 to the left, overwriting the element at idx
  for (int i = idx; i < chunk->n_free_blocks; i++) { chunk->free_blocks[i] = chunk->free_blocks[i + 1]; }
  chunk->n_free_blocks--;
}

static int v_dyn_tallocr_new_chunk(struct dyn_tensor_alloc* alloc, size_t min_size) {
  if (alloc->n_chunks >= v_VBUFFER_MAX_CHUNKS) { return -1; }
  struct tallocr_chunk* chunk  = reinterpret_cast<struct tallocr_chunk*>(calloc(1, sizeof(struct tallocr_chunk)));
  chunk->n_free_blocks         = 1;
  chunk->free_blocks[0].offset = 0;
  // available space in a chunk is limited to max_chunk_size, but can be higher if:
  // 1. a single tensor exceeds the maximum, and cannot fit any other way
  // 2. we are running out of chunks
  // backends will either manage to allocate the larger size, or report an error.
  chunk->free_blocks[0].size = MAX(min_size, alloc->max_chunk_size);
  if (alloc->n_chunks == v_VBUFFER_MAX_CHUNKS - 1) { chunk->free_blocks[0].size = SIZE_MAX / 2; }
  alloc->chunks[alloc->n_chunks] = chunk;
  alloc->n_chunks++;
  return alloc->n_chunks - 1;
}

#ifdef v_ALLOCATOR_DEBUG
static void add_allocated_tensor(struct dyn_tensor_alloc* alloc, struct buffer_address addr,
                                 const v_tensor* tensor) {
  for (int i = 0; i < 1024; i++) {
    if (alloc->allocated_tensors[i].tensor == NULL) {
      alloc->allocated_tensors[i].tensor = tensor;
      alloc->allocated_tensors[i].addr   = addr;
      return;
    }
  }
  v_ABORT("out of allocated_tensors");
}
static void remove_allocated_tensor(struct dyn_tensor_alloc* alloc, struct buffer_address addr,
                                    const v_tensor* tensor) {
  for (int i = 0; i < 1024; i++) {
    if (alloc->allocated_tensors[i].addr.chunk == addr.chunk && alloc->allocated_tensors[i].addr.offset == addr.
      offset) {
      alloc->allocated_tensors[i].tensor = NULL;
      return;
    }
  }
  v_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#endif

static struct buffer_address v_dyn_tallocr_alloc(struct dyn_tensor_alloc* alloc, size_t size,
                                                 const v_tensor* tensor) {
  size = aligned_offset(NULL, size, alloc->alignment);

  AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

  int best_fit_chunk = -1;
  int best_fit_block = -1;
  size_t max_avail   = 0;

  // find the best fitting free block besides the last block, within any chunk
  for (int c = 0; c < alloc->n_chunks; ++c) {
    struct tallocr_chunk* chunk = alloc->chunks[c];
    size_t best_fit_size        = SIZE_MAX;
    for (int i = 0; i < chunk->n_free_blocks - 1; i++) {
      struct free_block* block = &chunk->free_blocks[i];
      max_avail                = MAX(max_avail, block->size);
      if (block->size >= size && block->size <= best_fit_size) {
        best_fit_chunk = c;
        best_fit_block = i;
        best_fit_size  = block->size;
      }
    }
  }

  if (best_fit_block == -1) {
    // no suitable block found, try the last block (this may grow a chunks size)
    int64_t best_reuse = INT64_MIN;
    for (int c = 0; c < alloc->n_chunks; ++c) {
      struct tallocr_chunk* chunk = alloc->chunks[c];
      if (chunk->n_free_blocks > 0) {
        struct free_block* block = &chunk->free_blocks[chunk->n_free_blocks - 1];
        max_avail                = MAX(max_avail, block->size);
        int64_t reuse_factor     = chunk->max_size - block->offset - size;
        // reuse_factor < 0 : amount of extra memory that needs to be allocated
        // reuse_factor = 0 : allocated free space exactly matches tensor size
        // reuse_factor > 0 : superfluous memory that will remain unused
        bool better_reuse = best_reuse < 0 && reuse_factor > best_reuse;
        bool better_fit   = reuse_factor >= 0 && reuse_factor < best_reuse;
        if (block->size >= size && (better_reuse || better_fit)) {
          best_fit_chunk = c;
          best_fit_block = chunk->n_free_blocks - 1;
          best_reuse     = reuse_factor;
        }
      }
    }
  }

  if (best_fit_block == -1) {
    // none of the existing chunks have enough space left
    best_fit_chunk = v_dyn_tallocr_new_chunk(alloc, size);
    best_fit_block = 0;
  }
  if (best_fit_chunk == -1) {
    // since the last chunk always has virtually endless memory, this should never happen
    v_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
                __func__,
                size,
                max_avail);
    v_ABORT("graph allocation: failed to reserve memory");
  }

  struct tallocr_chunk* chunk = alloc->chunks[best_fit_chunk];
  struct free_block* block    = &chunk->free_blocks[best_fit_block];
  struct buffer_address addr  = {.chunk = best_fit_chunk, .offset = block->offset};
  block->offset += size;
  block->size -= size;
  if (block->size == 0) {
    // remove block if empty
    v_dyn_tallocr_remove_block(chunk, best_fit_block);
  }

  AT_PRINTF("block %d, offset %zu, chunk %d\n", best_fit_block, addr.offset, addr.chunk);

  #ifdef v_ALLOCATOR_DEBUG
  add_allocated_tensor(alloc, addr, tensor);
  size_t cur_max = addr.offset + size;
  if (cur_max > chunk->max_size) {
    // sort allocated_tensors by chunk/offset
    for (int i = 0; i < 1024; i++) {
      for (int j = i + 1; j < 1024; j++) {
        if (v_buffer_address_less(alloc->allocated_tensors[j].addr, alloc->allocated_tensors[i].addr)) {
          const v_tensor* tmp_tensor         = alloc->allocated_tensors[i].tensor;
          struct buffer_address tmp_addr     = alloc->allocated_tensors[i].addr;
          alloc->allocated_tensors[i].tensor = alloc->allocated_tensors[j].tensor;
          alloc->allocated_tensors[i].addr   = alloc->allocated_tensors[j].addr;
          alloc->allocated_tensors[j].tensor = tmp_tensor;
          alloc->allocated_tensors[j].addr   = tmp_addr;
        }
      }
    }
    v_LOG_DEBUG("max_size[%d] = %.2f MB: tensors: ", addr.chunk, cur_max / 1024.0 / 1024.0);
    for (int i = 0; i < 1024; i++) {
      if (alloc->allocated_tensors[i].tensor) {
        v_LOG_DEBUG("%s [%d: %zx-%zx] (%.2f MB) ",
                    alloc->allocated_tensors[i].tensor->name,
                    alloc->allocated_tensors[i].addr.chunk,
                    alloc->allocated_tensors[i].addr.offset,
                    alloc->allocated_tensors[i].addr.offset + num_bytes(alloc->allocated_tensors[i].tensor),
                    num_bytes(alloc->allocated_tensors[i].tensor) / 1024.0 / 1024.0);
      }
    }
    v_LOG_DEBUG("\n");
  }
  #endif

  chunk->max_size = MAX(chunk->max_size, addr.offset + size);

  return addr;

  v_UNUSED(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
static void v_dyn_tallocr_free_tensor(struct dyn_tensor_alloc* alloc, struct buffer_address addr, size_t size,
                                      const v_tensor* tensor) {
  size = aligned_offset(NULL, size, alloc->alignment);

  AT_PRINTF("%s: freeing %s at {chunk=%d, offset=%zu} (%zu bytes) - n_free_blocks = %d\n",
            __func__,
            tensor->name,
            addr.chunk,
            addr.offset,
            size,
            alloc->chunks[addr.chunk]->n_free_blocks);

  #ifdef v_ALLOCATOR_DEBUG
  remove_allocated_tensor(alloc, addr, tensor);
  #endif

  struct tallocr_chunk* chunk = alloc->chunks[addr.chunk];

  // see if we can merge with an existing block
  for (int i = 0; i < chunk->n_free_blocks; i++) {
    struct free_block* block = &chunk->free_blocks[i];
    // check if ptr is at the end of the block
    if (block->offset + block->size == addr.offset) {
      block->size += size;
      // check if we can merge with the next block
      if (i < chunk->n_free_blocks - 1) {
        struct free_block* next = &chunk->free_blocks[i + 1];
        if (block->offset + block->size == next->offset) {
          block->size += next->size;
          v_dyn_tallocr_remove_block(chunk, i + 1);
        }
      }
      return;
    }
    // check if ptr is at the beginning of the block
    if (addr.offset + size == block->offset) {
      block->offset = addr.offset;
      block->size += size;
      // check if we can merge with the previous block
      if (i > 0) {
        struct free_block* prev = &chunk->free_blocks[i - 1];
        if (prev->offset + prev->size == block->offset) {
          prev->size += block->size;
          v_dyn_tallocr_remove_block(chunk, i);
        }
      }
      return;
    }
  }
  // otherwise, add a new block
  V_ASSERT(chunk->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
  // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
  int insert_pos = 0;
  while (insert_pos < chunk->n_free_blocks && chunk->free_blocks[insert_pos].offset < addr.offset) { insert_pos++; }
  // shift all blocks from insert_pos onward to make room for the new block
  for (int i = chunk->n_free_blocks; i > insert_pos; i--) { chunk->free_blocks[i] = chunk->free_blocks[i - 1]; }
  // insert the new block
  chunk->free_blocks[insert_pos].offset = addr.offset;
  chunk->free_blocks[insert_pos].size   = size;
  chunk->n_free_blocks++;
  v_UNUSED(tensor);
}

static void v_dyn_tallocr_reset(struct dyn_tensor_alloc* alloc) {
  for (int i = 0; i < v_VBUFFER_MAX_CHUNKS; i++) {
    free(alloc->chunks[i]);
    alloc->chunks[i] = NULL;
  }
  alloc->n_chunks = 0;

  #ifdef v_ALLOCATOR_DEBUG
  for (int i = 0; i < 1024; i++) { alloc->allocated_tensors[i].tensor = NULL; }
  #endif
}

static struct dyn_tensor_alloc* v_dyn_tallocr_new(size_t alignment, size_t max_buffer_size) {
  struct dyn_tensor_alloc* alloc = new struct dyn_tensor_alloc();

  alloc->alignment      = alignment;
  alloc->max_chunk_size = MIN(max_buffer_size, SIZE_MAX/2); // clamp to avoid overflows
  alloc->n_chunks       = 0;
  v_dyn_tallocr_reset(alloc);
  return alloc;
}

static void v_dyn_tallocr_free(struct dyn_tensor_alloc* alloc) {
  for (int i = 0; i < alloc->n_chunks; ++i) { free(alloc->chunks[i]); }
  free(alloc);
}

static size_t v_dyn_tallocr_max_size(struct dyn_tensor_alloc* alloc, int chunk) {
  return chunk < alloc->n_chunks ? alloc->chunks[chunk]->max_size : 0;
}


// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)


static void v_vbuffer_free(struct vbuffer* buf) {
  if (buf == NULL) { return; }
  for (int i = 0; i < v_VBUFFER_MAX_CHUNKS; ++i) { v_backend_buffer_free(buf->chunks[i]); }
  free(buf);
}

static size_t v_vbuffer_chunk_size(struct vbuffer* buf, int chunk) {
  return buf->chunks[chunk] ? v_backend_buffer_get_size(buf->chunks[chunk]) : 0;
}

static size_t v_vbuffer_size(struct vbuffer* buf) {
  size_t size = 0;
  for (int i = 0; i < v_VBUFFER_MAX_CHUNKS && buf->chunks[i]; ++i) {
    size += v_backend_buffer_get_size(buf->chunks[i]);
  }
  return size;
}

static struct vbuffer* v_vbuffer_alloc(v_backend_buffer_type_t buft, const struct dyn_tensor_alloc* talloc,
                                       enum v_backend_buffer_usage usage) {
  struct vbuffer* buf = (struct vbuffer*)calloc(1, sizeof(struct vbuffer));
  if (buf == NULL) { return NULL; }

  for (int n = 0; n < talloc->n_chunks; n++) {
    size_t chunk_size = talloc->chunks[n]->max_size;
    buf->chunks[n]    = v_backend_buffer_alloc(buft, chunk_size);
    if (buf->chunks[n] == NULL) {
      v_vbuffer_free(buf);
      return NULL;
    }
    v_backend_buffer_set_usage(buf->chunks[n], usage);
  }
  return buf;
}

static void v_vbuffer_tensor_alloc(struct vbuffer* buf, v_tensor* tensor, struct buffer_address buf_addr) {
  void* base = v_backend_buffer_get_base(buf->chunks[buf_addr.chunk]);
  void* addr = (char*)base + buf_addr.offset;
  v_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}


/////////////////////////////////////

// graph allocator


v_graph_allocator_t v_gallocr_new_n(v_backend_buffer_type_t* bufts, int n_bufs) {
  v_graph_allocator_t galloc = (v_graph_allocator_t)calloc(1, sizeof(struct v_graph_alloc));
  V_ASSERT(galloc != NULL);

  galloc->bufts = (v_backend_buffer_type_t*)calloc(n_bufs, sizeof(v_backend_buffer_type_t));
  V_ASSERT(galloc->bufts != NULL);

  galloc->buffers = (struct vbuffer**)calloc(n_bufs, sizeof(struct vbuffer*));
  V_ASSERT(galloc->buffers != NULL);

  galloc->buf_tallocs = (struct dyn_tensor_alloc**)calloc(n_bufs, sizeof(struct dyn_tensor_alloc*));
  V_ASSERT(galloc->buf_tallocs != NULL);

  for (int i = 0; i < n_bufs; i++) {
    galloc->bufts[i]   = bufts[i];
    galloc->buffers[i] = NULL;

    // check if the same buffer type is used multiple times and reuse the same allocator
    for (int j = 0; j < i; j++) {
      if (bufts[i] == bufts[j]) {
        galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
        break;
      }
    }

    if (galloc->buf_tallocs[i] == NULL) {
      size_t alignment       = v_get_backend_buffer_align(bufts[i]);
      size_t max_size        = v_get_backend_buffer_max_size(bufts[i]);
      galloc->buf_tallocs[i] = v_dyn_tallocr_new(alignment, max_size);
    }
  }
  galloc->n_buffers = n_bufs;

  return galloc;
}

v_graph_allocator_t v_gallocr_new(v_backend_buffer_type_t buft) { return v_gallocr_new_n(&buft, 1); }

void v_gallocr_free(v_graph_allocator_t galloc) {
  if (galloc == NULL) { return; }

  for (int i = 0; i < galloc->n_buffers; i++) {
    if (galloc->buffers != NULL) {
      // skip if already freed
      bool freed = false;
      for (int j = 0; j < i; j++) {
        if (galloc->buffers[j] == galloc->buffers[i]) {
          freed = true;
          break;
        }
      }
      if (!freed) { v_vbuffer_free(galloc->buffers[i]); }
    }
    if (galloc->buf_tallocs != NULL) {
      // skip if already freed
      bool freed = false;
      for (int j = 0; j < i; j++) {
        if (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
          freed = true;
          break;
        }
      }
      if (!freed) { v_dyn_tallocr_free(galloc->buf_tallocs[i]); }
    }
  }

  v_hash_set_free(&galloc->hash_set);
  free(galloc->hash_values);
  free(galloc->bufts);
  free(galloc->buffers);
  free(galloc->buf_tallocs);
  free(galloc->node_allocs);
  free(galloc->leaf_allocs);
  free(galloc);
}

typedef struct v_graph_alloc* v_graph_allocator_t;

static struct hash_node* v_gallocr_hash_get(v_graph_allocator_t galloc, v_tensor* t) {
  size_t i = v_hash_find_or_insert(&galloc->hash_set, t);
  return &galloc->hash_values[i];
}

static bool v_gallocr_is_own(v_graph_allocator_t galloc, v_tensor* t) {
  return v_gallocr_hash_get(galloc, t)->allocated;
}

static bool v_gallocr_is_allocated(v_graph_allocator_t galloc, v_tensor* t) {
  return t->data != NULL || v_gallocr_hash_get(galloc, t)->allocated;
}

// free the extra space at the end if the new tensor is smaller
static void v_gallocr_free_extra_space(v_graph_allocator_t galloc, v_tensor* node, v_tensor* parent) {
  struct hash_node* hn   = v_gallocr_hash_get(galloc, node);
  struct hash_node* p_hn = v_gallocr_hash_get(galloc, parent);

  size_t parent_size = v_get_backend_buffer_alloc_size(galloc->bufts[p_hn->buffer_id], parent);
  size_t node_size   = v_get_backend_buffer_alloc_size(galloc->bufts[hn->buffer_id], node);

  V_ASSERT(parent_size >= node_size);

  if (parent_size > node_size) {
    struct dyn_tensor_alloc* p_alloc = galloc->buf_tallocs[p_hn->buffer_id];
    struct buffer_address p_addr     = p_hn->addr;
    p_addr.offset += node_size;
    size_t extra_size = parent_size - node_size;
    AT_PRINTF("freeing extra %zu bytes from parent %s for %s\n", extra_size, parent->name, node->name);
    v_dyn_tallocr_free_tensor(p_alloc, p_addr, extra_size, parent);
  }
}

static void v_graph_allocate_node(v_graph_allocator_t galloc,
                                  v_tensor* node,
                                  int buffer_id) {
  V_ASSERT(buffer_id >= 0);
  struct hash_node* hn = v_gallocr_hash_get(galloc, node);

  if (!v_gallocr_is_allocated(galloc, node) && !v_is_view(node)) {
    hn->allocated = true;
    assert(hn->addr.offset == 0);
    // try to reuse a parent's buffer (inplace)
    if (op_can_inplace(node->op)) {
      for (int i = 0; i < v_MAX_SRC; i++) {
        v_tensor* parent = node->src[i];
        if (parent == NULL) { continue; }
        // if the node's data is external, then we cannot re-use it
        if (!v_gallocr_is_own(galloc, parent)) {
          AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, node->name, parent->data);
          continue;
        }

        // outputs cannot be reused
        if (parent->flags & TENSOR_FLAG_OUTPUT || (parent->view_src != NULL && parent->view_src->flags &
          TENSOR_FLAG_OUTPUT)) {
          AT_PRINTF("not reusing parent %s for %s as it is an output\n", parent->name, node->name);
          continue;
        }

        if (!v_is_same_layout(node, parent)) {
          AT_PRINTF("not reusing parent %s for %s as layouts are different\n", parent->name, node->name);
          continue;
        }

        struct hash_node* p_hn = v_gallocr_hash_get(galloc, parent);
        if (p_hn->n_children == 1 && p_hn->n_views == 0) {
          if (v_is_view(parent)) {
            v_tensor* view_src            = parent->view_src;
            struct hash_node* view_src_hn = v_gallocr_hash_get(galloc, view_src);
            if (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
              AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, node->name);
              assert(view_src_hn->addr.chunk == p_hn->addr.chunk && view_src_hn->addr.offset == p_hn->addr.offset);
              hn->buffer_id          = p_hn->buffer_id;
              hn->addr               = p_hn->addr;
              p_hn->allocated        = false; // avoid freeing the parent
              view_src_hn->allocated = false;
              v_gallocr_free_extra_space(galloc, node, view_src);
              return;
            }
          }
          else {
            AT_PRINTF("reusing parent %s for %s\n", parent->name, node->name);
            hn->buffer_id   = p_hn->buffer_id;
            hn->addr        = p_hn->addr;
            p_hn->allocated = false; // avoid freeing the parent
            v_gallocr_free_extra_space(galloc, node, parent);
            return;
          }
        }
      }
    }
    // allocate tensor from the buffer
    struct dyn_tensor_alloc* alloc = galloc->buf_tallocs[buffer_id];
    v_backend_buffer_type_t buft   = galloc->bufts[buffer_id];
    size_t size                    = v_get_backend_buffer_alloc_size(buft, node);
    hn->buffer_id                  = buffer_id;
    hn->addr                       = v_dyn_tallocr_alloc(alloc, size, node);
  }
}

static void v_gallocr_free_node(v_graph_allocator_t galloc, v_tensor* node) {
  // graph outputs are never freed
  if (node->flags & TENSOR_FLAG_OUTPUT) {
    AT_PRINTF("not freeing output %s\n", node->name);
    return;
  }

  struct hash_node* hn           = v_gallocr_hash_get(galloc, node);
  int buffer_id                  = hn->buffer_id;
  struct dyn_tensor_alloc* alloc = galloc->buf_tallocs[buffer_id];
  v_backend_buffer_type_t buft   = galloc->bufts[buffer_id];
  size_t size                    = v_get_backend_buffer_alloc_size(buft, node);
  v_dyn_tallocr_free_tensor(alloc, hn->addr, size, node);
  hn->allocated = false;
}

static int get_node_buffer_id(const int* node_buffer_ids, int i) { return node_buffer_ids ? node_buffer_ids[i] : 0; }

//todo :
bool v_gallocr_reserve_n(v_graph_allocator_t galloc,
                         struct v_cgraph* graph,
                         const int* node_buffer_ids,
                         const int* leaf_buffer_ids) {
  size_t min_hash_size = graph->n_nodes + graph->n_leafs;
  // add 25% margin to avoid hash collisions
  min_hash_size += min_hash_size / 4;

  // initialize hash table
  if (galloc->hash_set.size < min_hash_size) {
    v_hash_set_free(&galloc->hash_set);
    galloc->hash_set = v_hash_set_new(min_hash_size);
    V_ASSERT(galloc->hash_set.keys != NULL);
    free(galloc->hash_values);
    galloc->hash_values = (struct hash_node*)malloc(sizeof(struct hash_node) * galloc->hash_set.size);
    V_ASSERT(galloc->hash_values != NULL);
  }

  // reset allocators
  for (int i = 0; i < galloc->n_buffers; i++) { v_dyn_tallocr_reset(galloc->buf_tallocs[i]); }
  v_hash_set_reset(&galloc->hash_set);
  memset(galloc->hash_values, 0, sizeof(struct hash_node) * galloc->hash_set.size);

  // allocate leafs
  // these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
  for (int i = 0; i < graph->n_leafs; i++) {
    v_tensor* leaf = graph->leafs[i];
    v_graph_allocate_node(galloc, leaf, get_node_buffer_id(leaf_buffer_ids, i));
  }
  // count number of children and views
  // allocate other graph inputs and leafs first to avoid overwriting them
  for (int i = 0; i < graph->n_nodes; i++) {
    v_tensor* node = graph->nodes[i];
    // TODO: better way to add external dependencies
    // v_OP_NONE does not appear normally in the graph nodes, but is used by ggml-backend to add dependencies to
    // control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
    // itself is never used and should not be considered a dependency
    if (v_is_view(node) && node->op != v_OP_NONE) {
      v_tensor* view_src = node->view_src;
      v_gallocr_hash_get(galloc, view_src)->n_views += 1;
    }

    if (node->flags & TENSOR_FLAG_INPUT) {
      v_graph_allocate_node(galloc, graph->nodes[i], get_node_buffer_id(node_buffer_ids, i));
    }

    for (int j = 0; j < v_MAX_SRC; j++) {
      v_tensor* src = node->src[j];
      if (src == NULL) { continue; }

      v_gallocr_hash_get(galloc, src)->n_children += 1;

      // allocate explicit inputs
      if (src->flags & TENSOR_FLAG_INPUT) {
        v_graph_allocate_node(galloc, src, get_node_buffer_id(node_buffer_ids, i));
      }
    }
  }

  // allocate tensors
  for (int i = 0; i < graph->n_nodes; i++) {
    v_tensor* node = graph->nodes[i];
    int buffer_id  = get_node_buffer_id(node_buffer_ids, i);

    // allocate parents (only leafs need to be allocated at this point)
    for (int j = 0; j < v_MAX_SRC; j++) {
      v_tensor* parent = node->src[j];
      if (parent == NULL) { continue; }
      v_graph_allocate_node(galloc, parent, buffer_id);
    }

    // allocate node
    v_graph_allocate_node(galloc, node, buffer_id);

    AT_PRINTF("exec: %s (%s) <= ", v_op_desc(node), node->name);
    for (int j = 0; j < v_MAX_SRC; j++) {
      v_tensor* parent = node->src[j];
      if (parent == NULL) { continue; }
      AT_PRINTF("%s", parent->name);
      if (j < v_MAX_SRC - 1 && node->src[j + 1] != NULL) { AT_PRINTF(", "); }
    }
    AT_PRINTF("\n");

    // update parents
    for (int j = 0; j < v_MAX_SRC; j++) {
      v_tensor* parent = node->src[j];
      if (parent == NULL) { continue; }
      struct hash_node* p_hn = v_gallocr_hash_get(galloc, parent);
      p_hn->n_children -= 1;

      AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n",
                parent->name,
                p_hn->n_children,
                p_hn->n_views,
                p_hn->allocated);

      if (p_hn->n_children == 0 && p_hn->n_views == 0) {
        if (v_is_view(parent)) {
          v_tensor* view_src            = parent->view_src;
          struct hash_node* view_src_hn = v_gallocr_hash_get(galloc, view_src);
          view_src_hn->n_views -= 1;
          AT_PRINTF("view_src %s: %d children, %d views\n",
                    view_src->name,
                    view_src_hn->n_children,
                    view_src_hn->n_views);
          if (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
            v_gallocr_free_node(galloc, view_src);
          }
        }
        else if (p_hn->allocated) { v_gallocr_free_node(galloc, parent); }
      }
      AT_PRINTF("\n");
    }
  }

  // set the node_allocs from the hash table
  if (galloc->n_nodes < graph->n_nodes) {
    free(galloc->node_allocs);
    galloc->node_allocs = (struct node_alloc*)calloc(graph->n_nodes, sizeof(struct node_alloc));
    V_ASSERT(galloc->node_allocs != NULL);
  }
  galloc->n_nodes = graph->n_nodes;
  for (int i = 0; i < graph->n_nodes; i++) {
    v_tensor* node                = graph->nodes[i];
    struct node_alloc* node_alloc = &galloc->node_allocs[i];
    if (node->view_src || node->data) {
      node_alloc->dst.buffer_id = -1;
      node_alloc->dst.addr      = v_BUFFER_ADDRESS_INVALID;
      node_alloc->dst.size_max  = 0;
    }
    else {
      struct hash_node* hn      = v_gallocr_hash_get(galloc, node);
      node_alloc->dst.buffer_id = hn->buffer_id;
      node_alloc->dst.addr      = hn->addr;
      node_alloc->dst.size_max  = v_get_backend_buffer_alloc_size(galloc->bufts[hn->buffer_id], node);
    }
    for (int j = 0; j < v_MAX_SRC; j++) {
      v_tensor* src = node->src[j];
      if (!src || src->view_src || src->data) {
        node_alloc->src[j].buffer_id = -1;
        node_alloc->src[j].addr      = v_BUFFER_ADDRESS_INVALID;
        node_alloc->src[j].size_max  = 0;
      }
      else {
        struct hash_node* hn         = v_gallocr_hash_get(galloc, src);
        node_alloc->src[j].buffer_id = hn->buffer_id;
        node_alloc->src[j].addr      = hn->addr;
        node_alloc->src[j].size_max  = v_get_backend_buffer_alloc_size(galloc->bufts[hn->buffer_id], src);
      }
    }
  }
  if (galloc->n_leafs < graph->n_leafs) {
    free(galloc->leaf_allocs);
    galloc->leaf_allocs = (struct leaf_alloc*)calloc(graph->n_leafs, sizeof(galloc->leaf_allocs[0]));
    V_ASSERT(galloc->leaf_allocs != NULL);
  }
  galloc->n_leafs = graph->n_leafs;
  for (int i = 0; i < graph->n_leafs; i++) {
    v_tensor* leaf       = graph->leafs[i];
    struct hash_node* hn = v_gallocr_hash_get(galloc, leaf);
    if (leaf->view_src || leaf->data) {
      galloc->leaf_allocs[i].leaf.buffer_id = -1;
      galloc->leaf_allocs[i].leaf.addr      = v_BUFFER_ADDRESS_INVALID;
      galloc->leaf_allocs[i].leaf.size_max  = 0;
    }
    else {
      galloc->leaf_allocs[i].leaf.buffer_id = hn->buffer_id;
      galloc->leaf_allocs[i].leaf.addr      = hn->addr;
      galloc->leaf_allocs[i].leaf.size_max  = v_get_backend_buffer_alloc_size(galloc->bufts[hn->buffer_id], leaf);
    }
  }

  // reallocate buffers if needed
  for (int i = 0; i < galloc->n_buffers; i++) {
    // if the buffer type is used multiple times, we reuse the same buffer
    for (int j = 0; j < i; j++) {
      if (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
        galloc->buffers[i] = galloc->buffers[j];
        break;
      }
    }

    // even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
    bool realloc    = galloc->buffers[i] == NULL;
    size_t new_size = 0;
    for (int c = 0; c < galloc->buf_tallocs[i]->n_chunks; c++) {
      size_t cur_chunk_size = galloc->buffers[i] ? v_vbuffer_chunk_size(galloc->buffers[i], c) : 0;
      size_t new_chunk_size = v_dyn_tallocr_max_size(galloc->buf_tallocs[i], c);
      new_size += new_chunk_size;
      if (new_chunk_size > cur_chunk_size) { realloc = true; }
    }
    if (realloc) {
      #ifndef NDEBUG
      //size_t cur_size = galloc->buffers[i] ? v_vbuffer_size(galloc->buffers[i]) : 0;
      //v_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n", __func__,
      //               mmlBackendBufferTypeName(galloc->bufts[i]), cur_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
      #endif

      v_vbuffer_free(galloc->buffers[i]);
      galloc->buffers[i] = v_vbuffer_alloc(galloc->bufts[i],
                                           galloc->buf_tallocs[i],
                                           v_BACKEND_BUFFER_USAGE_COMPUTE);
      if (galloc->buffers[i] == NULL) {
        v_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n",
                    __func__,
                    v_get_backend_buffer_name(galloc->bufts[i]),
                    new_size);
        return false;
      }
    }
  }

  return true;
}


void v_graph_alloc_init_tensor(v_graph_allocator_t galloc,
                               v_tensor* tensor,
                               struct tensor_alloc* tensor_alloc) {
  int buffer_id = tensor_alloc->buffer_id;
  assert(tensor->data ||
    tensor->view_src ||
    v_get_backend_buffer_alloc_size(galloc->bufts[buffer_id], tensor) <=
    tensor_alloc->size_max);
  if (tensor->view_src != NULL) {
    if (tensor->buffer == NULL) {
      assert(tensor_alloc->addr.offset == SIZE_MAX);
      if (tensor->view_src->buffer == NULL) {
        // this tensor was allocated without ggml-backend
        return;
      }
      v_backend_tensor_view_init(tensor);
    }
  }
  else {
    if (tensor->data == NULL) {
      assert(tensor_alloc->addr.offset != SIZE_MAX);
      assert(v_get_backend_buffer_alloc_size(galloc->bufts[buffer_id], tensor) <= tensor_alloc->size_max);
      v_vbuffer_tensor_alloc(galloc->buffers[buffer_id], tensor, tensor_alloc->addr);
    }
    else {
      if (tensor->buffer == NULL) {
        // this tensor was allocated without ggml-backend
        return;
      }
    }
  }
}

static bool v_gallocr_node_needs_realloc(v_graph_allocator_t galloc,
                                         v_tensor* node,
                                         struct tensor_alloc* talloc) {
  size_t node_size = 0;
  if (!node->data && !node->view_src) {
    // If we previously had data but don't now then reallocate
    if (talloc->buffer_id < 0) { return false; }
    node_size = v_get_backend_buffer_alloc_size(galloc->bufts[talloc->buffer_id], node);
  }
  return talloc->size_max >= node_size;
}

static bool v_gallocr_needs_realloc(v_graph_allocator_t galloc, struct v_cgraph* graph) {
  if (galloc->n_nodes != graph->n_nodes) {
    #ifndef NDEBUG
    //v_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
    #endif
    return true;
  }

  if (galloc->n_leafs != graph->n_leafs) {
    #ifndef NDEBUG
    v_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
    #endif
    return true;
  }

  for (int i = 0; i < graph->n_nodes; i++) {
    v_tensor* node                = graph->nodes[i];
    struct node_alloc* node_alloc = &galloc->node_allocs[i];

    if (!v_gallocr_node_needs_realloc(galloc, node, &node_alloc->dst)) {
      #ifndef NDEBUG
      v_LOG_DEBUG("%s: node %s is not valid\n", __func__, node->name);
      #endif
      return true;
    }

    for (int j = 0; j < v_MAX_SRC; j++) {
      v_tensor* src = node->src[j];
      if (src == NULL) { continue; }
      if (!v_gallocr_node_needs_realloc(galloc, src, &node_alloc->src[j])) {
        #ifndef NDEBUG
        v_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, node->name);
        #endif
        return true;
      }
    }
  }

  return false;
}

bool v_gallocr_alloc_graph(v_graph_allocator_t galloc, struct v_cgraph* graph) {
  if (v_gallocr_needs_realloc(galloc, graph)) {
    if (galloc->n_buffers == 1) {
      #ifndef NDEBUG
      //v_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
      #endif
      if (!v_gallocr_reserve_n(galloc, graph, 0, 0)) { return false; }
    }
    else {
      #ifndef NDEBUG
      v_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
      #endif
      return false;
    }
  }


  // allocate the graph tensors from the previous assignments
  // leafs
  for (int i = 0; i < graph->n_leafs; i++) {
    v_tensor* leaf                = graph->leafs[i];
    struct leaf_alloc* leaf_alloc = &galloc->leaf_allocs[i];
    v_graph_alloc_init_tensor(galloc, leaf, &leaf_alloc->leaf);
  }
  // nodes
  for (int i = 0; i < graph->n_nodes; i++) {
    v_tensor* node                = graph->nodes[i];
    struct node_alloc* node_alloc = &galloc->node_allocs[i];
    for (int j = 0; j < v_MAX_SRC; j++) {
      v_tensor* src = node->src[j];
      if (src == NULL) { continue; }
      v_graph_alloc_init_tensor(galloc, src, &node_alloc->src[j]);
    }
    v_graph_alloc_init_tensor(galloc, node, &node_alloc->dst);
  }

  return true;
}


// utils

static void free_buffers(v_backend_buffer_t** buffers, const size_t* n_buffers) {
  for (size_t i = 0; i < *n_buffers; i++) { v_backend_buffer_free((*buffers)[i]); }
  free(*buffers);
}

//todo:
static bool alloc_tensor_range(struct v_ctx* ctx,
                               v_tensor* first,
                               v_tensor* last,
                               v_backend_buffer_type_t buft,
                               size_t size,
                               v_backend_buffer_t** buffers,
                               size_t* n_buffers) {
  v_backend_buffer_t buffer = v_backend_buffer_alloc(buft, size);
  if (buffer == NULL) {
    v_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, v_get_backend_buffer_name(buft), size);
    free_buffers(buffers, n_buffers);
    return false;
  }
  *buffers                   = (v_backend_buffer_t*)realloc(*buffers, sizeof(v_backend_buffer_t) * (*n_buffers + 1));
  (*buffers)[(*n_buffers)++] = buffer;

  struct v_tensor_alloc tallocr = v_tallocr_new(buffer);

  for (v_tensor* t = first; t != last; t = v_get_next_tensor(ctx, t)) {
    enum v_status status = v_STATUS_SUCCESS;
    if (t->data == NULL) {
      if (t->view_src == NULL) { status = tensorAlloc(&tallocr, t); }
      else if (t->buffer == NULL) { status = v_backend_tensor_view_init(t); }
    }
    else {
      if (t->view_src != NULL && t->buffer == NULL) {
        // view of a pre-allocated tensor
        status = v_backend_tensor_view_init(t);
      }
    }
    if (status != v_STATUS_SUCCESS) {
      v_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t->name);
      free_buffers(buffers, n_buffers);
      return false;
    }
  }

  return true;
}

v_backend_buffer_t v_backend_alloc_ctx_tensor_from_buffer_t(struct v_ctx* ctx,
                                                            v_backend_buffer_type_t buft) {
  V_ASSERT(v_get_no_alloc(ctx) == true);
  size_t alignment = v_get_backend_buffer_align(buft);
  size_t max_size  = v_get_backend_buffer_max_size(buft);

  v_backend_buffer_t* buffers = NULL;

  size_t n_buffers    = 0;
  size_t cur_buf_size = 0;
  v_tensor* first     = v_get_first_tensor(ctx);
  for (v_tensor* t = first; t != NULL; t = v_get_next_tensor(ctx, t)) {
    size_t this_size = 0;
    if (t->data == NULL && t->view_src == NULL) {
      this_size = MML_PAD(v_get_backend_buffer_alloc_size(buft, t), alignment);
    }

    if (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
      // allocate tensors in the current buffer
      if (!alloc_tensor_range(ctx,
                              first,
                              t,
                              buft,
                              cur_buf_size,
                              &buffers,
                              &n_buffers)) { return NULL; }
      first        = t;
      cur_buf_size = this_size;
    }
    else { cur_buf_size += this_size; }
  }

  // allocate remaining tensors
  if (cur_buf_size > 0) {
    if (!alloc_tensor_range(ctx, first, NULL, buft, cur_buf_size, &buffers, &n_buffers)) { return NULL; }
  }

  if (n_buffers == 0) {
    #ifndef NDEBUG
    v_LOG_DEBUG("%s: all tensors in the context are already allocated\n", __func__);
    #endif
    return NULL;
  }
  v_backend_buffer_t buffer;
  if (n_buffers == 1) { buffer = buffers[0]; }
  else { throw std::runtime_error("2 buffers "); }
  free(buffers);
  return buffer;
}

v_backend_buffer_t v_backend_alloc_ctx_tensors(struct v_ctx* ctx, v_backend_t backend) {
  return v_backend_alloc_ctx_tensor_from_buffer_t(ctx, vk_device_buffer_type(backend->device->device));
}
