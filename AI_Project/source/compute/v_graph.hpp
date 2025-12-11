#pragma once
#include <cstdint>
#include <set>
#include <unordered_map>
#include <vector>
#include "v_hash.hpp"
#include "v_header.hpp"

/**
 * @brief compute graph structure.
 *
 * Represents the tensors and their relationships for graph-based computation.
 * Tracks nodes, leafs, gradients, and accumulators for automatic differentiation.
 */

struct v_cgraph {
  v_cgraph(size_t size) :
    size(size), n_nodes(0), n_leafs(0),
    nodes(size, 0), leafs(size, 0), grads(size, 0), grad_accs(size, 0),
    use_cnt(size, 0), visited_hash_set(size), order(V_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT) {}

  int size;                         ///< Maximum number of nodes / leafs / grads / grad_accs
  int n_nodes;                      ///< current nodes size
  int n_leafs;                      ///< current leafs size
  std::vector<v_tensor*> nodes;     ///< Tensors with data that can change if the graph is evaluated
  std::vector<v_tensor*> leafs;     ///< Tensors with constant data
  std::vector<v_tensor*> grads;     ///< Gradients of the nodes
  std::vector<v_tensor*> grad_accs; ///< Accumulators for node gradients
  std::vector<int32_t> use_cnt;     ///< Number of uses of each tensor, indexed by hash table slot
  v_hash_set visited_hash_set;      ///< Number of nodes currently in use
  v_cgraph_eval_order order;        ///< Number of leafs currently in use
  v_tensor** get_nodes_data() { return nodes.data(); }
};

size_t v_graph_overhead_custom(size_t size, bool grads);

void v_copy_graph(v_cgraph* src, v_cgraph* dst);
v_cgraph* v_new_graph_custom(v_ctx* ctx, size_t size, bool grads);
