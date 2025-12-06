//
// Created by dlwog on 25. 11. 14..
//

#include "v_graph.hpp"
#include  "v_util.h"
#include "ggml-impl.h"

size_t v_graph_nbyte(size_t size, bool grads) {
  size_t hash_size = v_hash_size(size * 2);
  void* p          = 0;
  incr_ptr_aligned(&p, sizeof(struct v_cgraph), 1);
  incr_ptr_aligned(&p, size * sizeof(struct v_tensor*), sizeof(struct v_tensor*)); // nodes
  incr_ptr_aligned(&p, size * sizeof(struct v_tensor*), sizeof(struct v_tensor*)); // leafs
  incr_ptr_aligned(&p, hash_size * sizeof(int32_t), sizeof(int32_t)); // use_counts
  incr_ptr_aligned(&p, hash_size * sizeof(struct v_tensor*), sizeof(struct v_tensor*)); // hash keys
  if (grads) {
    incr_ptr_aligned(&p, hash_size * sizeof(struct v_tensor*), sizeof(struct v_tensor*)); // grads
    incr_ptr_aligned(&p, hash_size * sizeof(struct v_tensor*), sizeof(struct v_tensor*)); // grad_accs
  }
  incr_ptr_aligned(&p, v_bitset_size(hash_size) * sizeof(v_bitset_t), sizeof(v_bitset_t));
  size_t nbytes = (size_t)p;
  return nbytes;
}


struct v_cgraph* v_new_graph_custom(struct v_ctx* ctx, size_t size, bool grads) {
  const size_t obj_size   = v_graph_nbyte(size, grads);
  struct v_object* obj    = v_new_object(ctx, MML_GRAPH, obj_size);
  struct v_cgraph* cgraph = (struct v_cgraph*)((char*)ctx->mem_buffer + obj->offs);
  // the size of the hash table is doubled since it needs to hold both nodes and leafs
  size_t hash_size = v_hash_size(size * 2);
  void* p          = cgraph + 1;

  struct v_tensor** nodes_ptr =
    (struct v_tensor**)incr_ptr_aligned(&p, size * sizeof(struct v_tensor*), sizeof(struct v_tensor*));
  struct v_tensor** leafs_ptr =
    (struct v_tensor**)incr_ptr_aligned(&p, size * sizeof(struct v_tensor*), sizeof(struct v_tensor*));
  int32_t* use_counts_ptr         = (int32_t*)incr_ptr_aligned(&p, hash_size * sizeof(int32_t), sizeof(int32_t));
  struct v_tensor** hash_keys_ptr = (struct v_tensor**)incr_ptr_aligned(&p,
                                                                        hash_size * sizeof(struct v_tensor*),
                                                                        sizeof(struct v_tensor*));
  void* grads_ptr = (struct v_tensor**)grads
                      ? incr_ptr_aligned(&p,
                                         hash_size * sizeof(struct v_tensor*),
                                         sizeof(struct v_tensor*))
                      : NULL;
  void* grad_accs_ptr = grads
                          ? incr_ptr_aligned(&p,
                                             hash_size * sizeof(struct v_tensor*),
                                             sizeof(struct v_tensor*))
                          : NULL;

  v_bitset_t* hash_used = (v_bitset_t*)incr_ptr_aligned(&p,
                                                              v_bitset_size(hash_size) * sizeof(v_bitset_t),
                                                              sizeof(v_bitset_t));

  // check that we allocated the correct amount of memory
  assert(obj_size == (size_t)((char *)p - (char *)cgraph));

  cgraph                     = (struct v_cgraph*)cgraph;
  (*cgraph).size             = (int)size;
  (*cgraph).n_nodes          = 0;
  (*cgraph).n_leafs          = 0;
  (*cgraph).nodes            = nodes_ptr;
  (*cgraph).grads            = (v_tensor**)grads_ptr;
  (*cgraph).grad_accs        = (v_tensor**)grad_accs_ptr;
  (*cgraph).leafs            = leafs_ptr;
  (*cgraph).use_counts       = use_counts_ptr;
  (*cgraph).visited_hash_set = {hash_size, hash_used, hash_keys_ptr};
  (*cgraph).order            = v_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT;


  v_hash_set_reset(&cgraph->visited_hash_set);
  if (grads) {
    memset(cgraph->grads, 0, hash_size * sizeof(struct v_tensor*));
    memset(cgraph->grad_accs, 0, hash_size * sizeof(struct v_tensor*));
  }

  return cgraph;
}

void v_copy_graph(struct v_cgraph* src,
                  struct v_cgraph* dst) {
  V_ASSERT(dst->size >= src->n_leafs);
  V_ASSERT(dst->size >= src->n_nodes);
  V_ASSERT(dst->visited_hash_set.size >= src->visited_hash_set.size);

  dst->n_leafs = src->n_leafs;
  dst->n_nodes = src->n_nodes;
  dst->order   = src->order;

  for (int i = 0; i < src->n_leafs; ++i) { dst->leafs[i] = src->leafs[i]; }

  for (int i = 0; i < src->n_nodes; ++i) { dst->nodes[i] = src->nodes[i]; }

  for (size_t i = 0; i < src->visited_hash_set.size; ++i) {
    // copy all hashset keys (tensors) that are in use
    if (v_bit_set_get(src->visited_hash_set.used, i)) {
      size_t new_hash_pos           = v_hash_insert(&dst->visited_hash_set, src->visited_hash_set.keys[i]);
      dst->use_counts[new_hash_pos] = src->use_counts[i];
    }
  }

  if (dst->grads) {
    memset(dst->grads, 0, dst->visited_hash_set.size * sizeof(struct v_tensor*));
    memset(dst->grad_accs, 0, dst->visited_hash_set.size * sizeof(struct v_tensor*));
  }

  if (src->grads) {
    V_ASSERT(dst->grads != NULL);
    V_ASSERT(dst->grad_accs != NULL);
    for (int i = 0; i < src->n_nodes; ++i) {
      const size_t igrad_src = find_hash(&src->visited_hash_set, src->nodes[i]);
      const size_t igrad_dst = find_hash(&dst->visited_hash_set, dst->nodes[i]);

      V_ASSERT(igrad_src != v_HASHSET_FULL);
      V_ASSERT(v_bit_set_get(src->visited_hash_set.used, igrad_src));
      V_ASSERT(igrad_dst != v_HASHSET_FULL);
      V_ASSERT(v_bit_set_get(dst->visited_hash_set.used, igrad_dst));

      dst->grads[igrad_dst]     = src->grads[igrad_src];
      dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
    }
  }
}
