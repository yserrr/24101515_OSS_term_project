//
// Created by dlwog on 25. 11. 14..
//

#ifndef MYPROJECT_MML_GRAPH_H
#define MYPROJECT_MML_GRAPH_H
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "v_hash.hpp"

enum v_cgraph_eval_order {
  v_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,
  v_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
  v_CGRAPH_EVAL_ORDER_COUNT
};

struct v_cgraph {
  int size; // maximum number of nodes/leafs/grads/grad_accs
  int n_nodes; // number of nodes currently in use
  int n_leafs; // number of leafs currently in use
  v_tensor** nodes;
  v_tensor** grads;
  v_tensor** grad_accs;
  v_tensor** leafs;
  std::vector<v_tensor*> nodes__; // tensors with data that can change if the graph is evaluated
  std::vector<v_tensor*> grads__; // the outputs of these tensors are the gradients of the nodes
  std::vector<v_tensor*> grad_accs__; // accumulators for node gradients
  std::vector<v_tensor*> leafs__; // tensors with constant data
  int32_t* use_counts; // number of uses of each tensor, indexed by hash table slot
  v_hash_set visited_hash_set;
  v_cgraph_eval_order order;
  std::unordered_map<v_tensor*, v_tensor*> tensor_maps;
};

size_t v_graph_overhead_custom(size_t size, bool grads);
void v_copy_graph(v_cgraph* src, v_cgraph* dst);
v_cgraph* v_new_graph_custom(struct v_ctx* ctx, size_t size, bool grads);
size_t v_graph_nbyte(size_t size, bool grads);
#endif //MYPROJECT_MML_GRAPH_H
