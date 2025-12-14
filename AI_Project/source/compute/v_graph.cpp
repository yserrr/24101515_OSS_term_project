#include <cinttypes>
#include "ggml-impl.hpp"
#include "v_graph.hpp"
#include "v_util.hpp"
#include "v_backend.hpp"


v_cgraph* v_new_graph_custom(v_ctx* ctx, size_t size, bool grads) {
  v_object* obj    = v_new_object(ctx, V_GRAPH, sizeof(v_cgraph));
  void* buf        = static_cast<std::byte*>(ctx->mem_buffer) + obj->offs;
  v_cgraph* cgraph = new(buf) v_cgraph(size);
  if (grads) {
    cgraph->grads.resize(size, nullptr);
    cgraph->grad_accs.resize(size, nullptr);
  }
  return cgraph;
}

void v_copy_graph(v_cgraph* src, v_cgraph* dst) {
  V_ASSERT(dst->size >= src->n_leafs);
  V_ASSERT(dst->size >= src->n_nodes);
  V_ASSERT(dst->visited_hash_set.size >= src->visited_hash_set.size);
  dst->n_leafs = src->n_leafs;
  dst->n_nodes = src->n_nodes;
  dst->order   = src->order;
  for (int i = 0; i < src->n_leafs; ++i) dst->leafs[i] = src->leafs[i];
  for (int i = 0; i < src->n_nodes; ++i) dst->nodes[i] = src->nodes[i];
  for (size_t i = 0; i < src->visited_hash_set.size; ++i) {
    // copy all hashset keys__ (tensors) that are in use
    if (src->visited_hash_set.is_contains(i)) {
      size_t new_hash_pos        = dst->visited_hash_set.insert(src->visited_hash_set.keys__[i]);
      dst->use_cnt[new_hash_pos] = src->use_cnt[i];
    }
  }

  if (dst->grads.data()) {
    dst->grads.resize(src->grads.size(), nullptr);
    dst->grad_accs.resize(src->grad_accs.size(), nullptr);
  }

  if (src->grads.data()) {
    V_ASSERT(dst->grads.data() != nullptr);
    V_ASSERT(dst->grad_accs.data() != nullptr);
    for (int i = 0; i < src->n_nodes; ++i) {
      const size_t igrad_src = src->visited_hash_set.find_hash(src->nodes[i]);
      const size_t igrad_dst = dst->visited_hash_set.find_hash(dst->nodes[i]);
      V_ASSERT(igrad_src != V_HASHSET_FULL);
      //V_ASSERT(is_contains(src->visited_hash_set.used_bits__, igrad_src));
      V_ASSERT(igrad_dst != V_HASHSET_FULL);
      //V_ASSERT(is_contains(dst->visited_hash_set.used_bits__, igrad_dst));
      dst->grads[igrad_dst]     = src->grads[igrad_src];
      dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
    }
  }
}

void v_graph_reset(v_cgraph* cgraph) {
  if (!cgraph) {
    return;
  }
  //V_ASSERT(cgraph->grads != nullptr);
  for (int i = 0; i < cgraph->n_nodes; i++) {
    v_tensor* node     = cgraph->nodes[i];
    v_tensor* grad_acc = v_graph_get_grad_acc(cgraph, node);
    if (node->op == v_OP_OPT_STEP_ADAMW) {
      // clear momenta
      v_set_zero(node->src[2]);
      v_set_zero(node->src[3]);
    }

    // initial gradients of loss should be 1, 0 otherwise
    if (grad_acc) {
      if (node->flags & TENSOR_FLAG_LOSS) {
        V_ASSERT(grad_acc->type == v_TYPE_F32);
        V_ASSERT(grad_acc->is_scalar());
        const float onef = 1.0f;
        if (grad_acc->buffer) {
          v_set_backend_tensor(grad_acc, &onef, 0, sizeof(float));
        } else {
          V_ASSERT(grad_acc->data);
          *static_cast<float*>(grad_acc->data) = onef;
        }
      } else {
        v_set_zero(grad_acc);
      }
    }
  }
}


v_tensor* v_graph_get_tensor(const v_cgraph* cgraph, const char* name) {
  for (int i = 0; i < cgraph->n_leafs; i++) {
    v_tensor* leaf = cgraph->leafs[i];
    if (strcmp(leaf->name.data(), name) == 0) {
      return leaf;
    }
  }
  for (int i = 0; i < cgraph->n_nodes; i++) {
    v_tensor* node = cgraph->nodes[i];
    if (strcmp(node->name.data(), name) == 0) {
      return node;
    }
  }
  return nullptr;
}


void v_print_graph(const v_cgraph* cgraph) {
  V_LOG_INFO("=== GRAPH ===\n");
  V_LOG_INFO("n_nodes = %d\n", cgraph->n_nodes);
  for (int i = 0; i < cgraph->n_nodes; i++) {
    v_tensor* node = cgraph->nodes[i];
    V_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
               i,
               node->ne[0],
               node->ne[1],
               node->ne[2],
               v_op_name(node->op),
               (node->flags & TENSOR_FLAG_PARAM) ? "x" :
               v_graph_get_grad(cgraph, node) ? "g" : " ");
  }

  V_LOG_INFO("n_leafs = %d\n", cgraph->n_leafs);
  for (int i = 0; i < cgraph->n_leafs; i++) {
    v_tensor* node = cgraph->leafs[i];

    V_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
               i,
               node->ne[0],
               node->ne[1],
               v_op_name(node->op),
               get_name(node));
  }
  V_LOG_INFO("========================================\n");
}

// check if node is part of the graph
static bool v_graph_find(const v_cgraph* cgraph, const v_tensor* node) {
  if (cgraph == nullptr) {
    return true;
  }

  for (int i = 0; i < cgraph->n_nodes; i++) {
    if (cgraph->nodes[i] == node) {
      return true;
    }
  }

  return false;
}


void v_graph_dump_dot_leaf_edge(FILE* fp, v_tensor* node, v_tensor* parent,
                                const char* label) {
  fprintf(fp,
          "  \"%p\" -> \"%p\" [ label = \"%s\"; ]\n",
          static_cast<void*>(parent),
          (void*)node,
          label);
}

static v_tensor* v_graph_get_parent(const struct v_cgraph* cgraph, const v_tensor* node) {
  for (int i = 0; i < cgraph->n_nodes; i++) {
    v_tensor* parent = cgraph->nodes[i];
    v_tensor* grad   = v_graph_get_grad(cgraph, parent);

    if (grad == node) {
      return parent;
    }
  }

  return nullptr;
}


static void v_graph_dump_dot_node_edge(FILE* fp, const struct v_cgraph* gb, v_tensor* node,
                                       v_tensor* parent, const char* label) {
  v_tensor* gparent  = v_graph_get_parent(gb, node);
  v_tensor* gparent0 = v_graph_get_parent(gb, parent);
  fprintf(fp,
          "  \"%p\" -> \"%p\" [ arrowhead = %s; style = %s; label = \"%s\"; ]\n",
          gparent0
            ? (void*)gparent0
            : (void*)parent,
          gparent
            ? (void*)gparent
            : (void*)node,
          gparent
            ? "empty"
            : "vee",
          gparent
            ? "dashed"
            : "solid",
          label);
}

void v_graph_dump_dot(const v_cgraph* gb, const v_cgraph* gf, const char* filename) {
  char color[16];
  FILE* fp = v_fopen(filename, "w");
  V_ASSERT(fp);
  fprintf(fp, "digraph G {\n");
  fprintf(fp, "  newrank = true;\n");
  fprintf(fp, "  rankdir = TB;\n");
  for (int i = 0; i < gb->n_nodes; i++) {
    v_tensor* node = gb->nodes[i];
    v_tensor* grad = v_graph_get_grad(gb, node);

    if (v_graph_get_parent(gb, node) != nullptr) {
      continue;
    }

    if (node->flags & TENSOR_FLAG_PARAM) {
      snprintf(color, sizeof(color), "yellow");
    } else if (grad) {
      if (v_graph_find(gf, node)) {
        snprintf(color, sizeof(color), "green");
      } else {
        snprintf(color, sizeof(color), "lightblue");
      }
    } else {
      snprintf(color, sizeof(color), "white");
    }

    fprintf(fp,
            "  \"%p\" [ "
            "style = filled; fillcolor = %s; shape = record; "
            "label=\"",
            static_cast<void*>(node), color);

    if (strlen(node->name.data()) > 0) {
      fprintf(fp, "%s (%s)|", node->name.data(), v_type_name(node->type));
    } else {
      fprintf(fp, "(%s)|", v_type_name(node->type));
    }

    if (node->is_matrix()) {
      fprintf(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, node->ne[0], node->ne[1], v_op_symbol(node->op));
    } else {
      fprintf(fp,
              "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s",
              i,
              node->ne[0],
              node->ne[1],
              node->ne[2],
              v_op_symbol(node->op));
    }

    if (grad) {
      fprintf(fp, " | <g>%s\"; ]\n", v_op_symbol(grad->op));
    } else {
      fprintf(fp, "\"; ]\n");
    }
  }

  for (int i = 0; i < gb->n_leafs; i++) {
    v_tensor* node = gb->leafs[i];

    snprintf(color, sizeof(color), "pink");

    fprintf(fp,
            "  \"%p\" [ "
            "style = filled; fillcolor = %s; shape = record; "
            "label=\"<x>",
            (void*)node,
            color);

    if (strlen(node->name.data()) > 0) {
      fprintf(fp, "%s (%s)|", node->name.data(), v_type_name(node->type));
    } else {
      fprintf(fp, "(%s)|", v_type_name(node->type));
    }

    fprintf(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, node->ne[0], node->ne[1]);
    if (nelements(node) < 5 && node->data != nullptr) {
      fprintf(fp, " | (");
      for (int j = 0; j < nelements(node); j++) {
        // FIXME: use ggml-backend to obtain the tensor data
        //if (node->type == v_TYPE_I8 || node->type == v_TYPE_I16 || node->type == v_TYPE_I32) {
        //    fprintf(fp, "%d", v_get_i32_1d(node, j));
        //}
        //else if (node->type == v_TYPE_F32 ||
        //         node->type == v_TYPE_F16 ||
        //         node->type == v_TYPE_BF16) {
        //    fprintf(fp, "%.1e", (double)v_get_f32_1d(node, j));
        //}
        //else
        {
          fprintf(fp, "#");
        }
        if (j < nelements(node) - 1) {
          fprintf(fp, ", ");
        }
      }
      fprintf(fp, ")");
    }
    fprintf(fp, "\"; ]\n");
  }

  for (int i = 0; i < gb->n_nodes; i++) {
    v_tensor* node = gb->nodes[i];

    for (int j = 0; j < V_MAX_SRC; j++) {
      if (node->src[j]) {
        char label[16];
        snprintf(label, sizeof(label), "src %d", j);
        v_graph_dump_dot_node_edge(fp, gb, node, node->src[j], label);
      }
    }
  }

  for (int i = 0; i < gb->n_leafs; i++) {
    v_tensor* node = gb->leafs[i];

    for (int j = 0; j < V_MAX_SRC; j++) {
      if (node->src[j]) {
        char label[16];
        snprintf(label, sizeof(label), "src %d", j);
        v_graph_dump_dot_leaf_edge(fp, node, node->src[j], label);
      }
    }
  }

  fprintf(fp, "}\n");

  fclose(fp);

  V_LOG_INFO("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

static int v_node_list_find_tensor(const v_cgraph* cgraph,
                                   const int* idxs,
                                   int count,
                                   const v_tensor* tensor) {
  V_ASSERT(cgraph && idxs);
  for (int i = 0; i < count; ++i) {
    const int node_idx = idxs[i];
    if (node_idx >= cgraph->n_nodes) {
      return -1;
    }
    if (cgraph->nodes[node_idx] == tensor) {
      return i;
    }
  }
  return -1;
}

bool v_can_fuse_subgraph_ext(const v_cgraph* cgraph,
                             const int* node_idxs,
                             int count,
                             const v_operation* ops,
                             const int* outputs,
                             int num_outputs) {
  V_ASSERT(outputs && num_outputs > 0);

  for (int i = 0; i < count; ++i) {
    if (node_idxs[i] >= cgraph->n_nodes) {
      return false;
    }

    const v_tensor* node = cgraph->nodes[node_idxs[i]];

    if (node->op != ops[i]) {
      return false;
    }

    if (v_node_list_find_tensor(cgraph, outputs, num_outputs, node) != -1) {
      continue;
    }

    if (node->flags & TENSOR_FLAG_OUTPUT) {
      return false;
    }

    int subgraph_uses = 0;
    for (int j = i + 1; j < count; ++j) {
      const v_tensor* other_node = cgraph->nodes[node_idxs[j]];
      for (int src_idx = 0; src_idx < V_MAX_SRC; src_idx++) {
        if (other_node->src[src_idx] == node) {
          subgraph_uses++;
        }
      }
    }

    if (subgraph_uses != v_node_get_use_count(cgraph, node_idxs[i])) {
      return false;
    }

    // if node is a view, check if the view_src and all it's parent view_srcs are within the subgraph
    v_tensor* view_src = node->view_src;
    while (view_src) {
      if (v_node_list_find_tensor(cgraph, node_idxs, count, view_src) == -1) {
        return false;
      }
      view_src = view_src->view_src;
    }
  }

  return true;
}
