#include "vk_context.h"
#include "vk_pipeline.hpp"
#include "vk_common.h"
#include "vk_device.hpp"
#include "vk_util.hpp"
#include "vk_op_f32.hpp"
#include "vk_vision_comp.hpp"
#include "vk_comp.hpp"


uint32_t v_vk_rms_partials_size(vk_backend_ctx* ctx, const v_tensor* node);
// Returns true if node has enqueued work into the queue, false otherwise
// If submit is true the current all operations queued so far are being submitted to Vulkan to overlap cmdlist creation and GPU execution.
bool v_vk_build_graph(vk_backend_ctx* ctx, v_cgraph* cgraph, int node_idx,
                    v_tensor* node_begin, int node_idx_begin, bool dryrun, bool last_node,
                    bool almost_ready, bool submit) {
  v_tensor* node = cgraph->nodes[node_idx];
  if ((node)->is_empty() || !node->buffer) { return false; }

  VK_LOG_DEBUG("v_vk_build_graph(" << node << ", " << op_name(node->op) << ")");
  ctx->semaphore_idx = 0;

  v_tensor* src0 = node->src[0];
  v_tensor* src1 = node->src[1];
  v_tensor* src2 = node->src[2];
  v_tensor* src3 = node->src[3];

  switch (node->op) {
    // Return on empty ops to avoid generating a compute_ctx and setting exit_tensor
    case V_OP_RESHAPE:
    case V_OP_VIEW:
    case V_OP_PERMUTE:
    case v_OP_TRANSPOSE:
    case v_OP_NONE:
      return false;
    case v_OP_UNARY:
      switch (v_get_unary_op(node)) {
        case v_UNARY_OP_EXP:
        case v_UNARY_OP_SILU:
        case v_UNARY_OP_GELU:
        case V_UNARY_OP_LOG:
        case v_UNARY_OP_GELU_ERF:
        case v_UNARY_OP_GELU_QUICK:
        case v_UNARY_OP_RELU:
        case v_UNARY_OP_TANH:
        case v_UNARY_OP_SIGMOID:
        case v_UNARY_OP_HARDSIGMOID:
        case v_UNARY_OP_HARDSWISH:
          break;
        default:
          std::cout << " false case operation :: " << v_get_unary_op(node) << std::endl;
          throw std::runtime_error("fail to operate ");
          return false;
      }
      break;
    case v_OP_GLU:
      switch (v_get_glu_op(node)) {
        case v_GLU_OP_GEGLU:
        case v_GLU_OP_REGLU:
        case v_GLU_OP_SWIGLU:
        case v_GLU_OP_SWIGLU_OAI:
        case v_GLU_OP_GEGLU_ERF:
        case v_GLU_OP_GEGLU_QUICK:
          break;
        default:
          return false;
      }
      break;
    case v_OP_ADD: {
      int next_node_idx = node_idx + 1 + ctx->num_additional_fused_ops;
      if (next_node_idx < cgraph->n_nodes &&
        cgraph->nodes[next_node_idx]->op == v_OP_RMS_NORM &&
        cgraph->nodes[next_node_idx]->src[0] == cgraph->nodes[next_node_idx - 1] &&
        v_nrows(cgraph->nodes[next_node_idx]) == 1 &&
        ctx->device->add_rms_fusion) {
        if (dryrun) { ctx->prealloc_size_add_rms_partials += v_vk_rms_partials_size(ctx, cgraph->nodes[node_idx]); }
        ctx->do_add_rms_partials = true;
      }
    }
    break;
    case v_OP_REPEAT:
    case v_OP_REPEAT_BACK:
    case v_OP_GET_ROWS:
    case v_OP_ADD_ID:
    case v_OP_ACC:
    case v_OP_SUB:
    case v_OP_MUL:
    case v_OP_DIV:
    case v_OP_CONCAT:
    case v_OP_UPSCALE:
    case V_OP_SCALE:
    case V_OP_SQR:
    case v_OP_SQRT:
    case V_OP_SIN:
    case V_OP_COS:
    case V_OP_CLAMP:
    case v_OP_PAD:
    case v_OP_ROLL:
    case V_OP_CPY:
    case v_OP_SET_ROWS:
    case V_OP_CONT:
    case v_OP_DUP:
    case v_OP_SILU_BACK:
    case v_OP_NORM:
    case v_OP_GROUP_NORM:
    case v_OP_RMS_NORM:
    case v_OP_RMS_NORM_BACK:
    case v_OP_L2_NORM:
    case V_OP_DIAG_MASK_INF:
    case V_OP_SOFT_MAX:
    case v_OP_SOFT_MAX_BACK:
    case V_OP_ROPE:
    case v_OP_ROPE_BACK:
    case V_OP_MUL_MAT:
    case v_OP_MUL_MAT_ID:
    case v_OP_ARGSORT:
    case v_OP_SUM:
    case v_OP_SUM_ROWS:
    case V_OP_MEAN:
    case V_OP_ARGMAX:
    case v_OP_COUNT_EQUAL:
    case V_OP_IM2COL:
    case v_OP_IM2COL_3D:
    case v_OP_TIMESTEP_EMBEDDING:
    case v_OP_CONV_TRANSPOSE_1D:
    case V_OP_POOL_2D:
    case V_OP_POOL_2D_BACK:
    case v_OP_CONV_2D:
    case v_OP_CONV_TRANSPOSE_2D:
    case v_OP_CONV_2D_DW:
    case v_OP_RWKV_WKV6:
    case v_OP_RWKV_WKV7:
    case v_OP_SSM_SCAN:
    case v_OP_SSM_CONV:
    case V_OP_LEAKY_RELU:
    case v_OP_FLASH_ATTN_EXT:
    case v_OP_OPT_STEP_ADAMW:
    case v_OP_OPT_STEP_SGD:
      break;
    default:
      std::cerr << "v_vulkan: Error: Missing op: " << v_op_name(node->op) << std::endl;
      //v_ABORT("fatal error");
  }

  vk_context compute_ctx;

  if (!dryrun) {
    if (ctx->compute_ctx.expired()) {
      compute_ctx      = vk_create_context(ctx, ctx->compute_cmd_pool);
      ctx->compute_ctx = compute_ctx;
      vk_begin_ctx(ctx->device, compute_ctx);
    }
    else { compute_ctx = ctx->compute_ctx.lock(); }
  }
  else {
    switch (node->op) {
      case v_OP_REPEAT:
      case v_OP_REPEAT_BACK:
      case v_OP_ACC:
      case v_OP_GET_ROWS:
      case v_OP_ADD:
      case v_OP_SUB:
      case v_OP_MUL:
      case v_OP_DIV:
      case v_OP_CONCAT:
      case v_OP_UPSCALE:
      case V_OP_SCALE:
      case V_OP_SQR:
      case v_OP_SQRT:
      case V_OP_SIN:
      case V_OP_COS:
      case V_OP_CLAMP:
      case v_OP_PAD:
      case V_OP_CPY:
      case v_OP_SET_ROWS:
      case V_OP_CONT:
      case v_OP_DUP:
      case v_OP_SILU_BACK:
      case v_OP_NORM:
      case v_OP_GROUP_NORM:
      case v_OP_RMS_NORM:
      case v_OP_RMS_NORM_BACK:
      case v_OP_L2_NORM:
      case v_OP_UNARY:
      case v_OP_GLU:
      case V_OP_DIAG_MASK_INF:
      case V_OP_SOFT_MAX:
      case v_OP_SOFT_MAX_BACK:
      case V_OP_ROPE:
      case v_OP_ROPE_BACK:
      case v_OP_ARGSORT:
      case v_OP_SUM:
      case v_OP_SUM_ROWS:
      case V_OP_MEAN:
      case V_OP_ARGMAX:
      case v_OP_COUNT_EQUAL:
      case V_OP_IM2COL:
      case v_OP_IM2COL_3D:
      case v_OP_TIMESTEP_EMBEDDING:
      case v_OP_CONV_TRANSPOSE_1D:
      case V_OP_POOL_2D:
      case V_OP_POOL_2D_BACK:
      case v_OP_CONV_2D:
      case v_OP_CONV_TRANSPOSE_2D:
      case v_OP_CONV_2D_DW:
      case V_OP_LEAKY_RELU:
      case v_OP_OPT_STEP_SGD: {
        // These operations all go through v_vk_op_f32, so short-circuit and
        // do the only thing needed for the dryrun.
        vk_pipeline pipeline = v_vk_op_get_pipeline(ctx, src0, src1, src2, node, node->op);
        v_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        if (node->op == v_OP_RMS_NORM) { ctx->do_add_rms_partials = false; }
        return false;
      }
      default:
        break;
    }
  }

  if (!dryrun) {
    // This logic detects dependencies between modes in the graph and calls v_vk_sync_buffers
    // to synchronize them. This handles most "normal" synchronization when computing the graph, and when
    // there is no auxiliary memory use, it shouldn't be necessary to call v_vk_sync_buffers
    // outside of this logic. When a node uses one of the prealloc buffers for something like
    // dequantization or split_k, additional synchronization is needed between those passes.
    bool need_sync = false;
    // Check whether "node" requires synchronization. The node requires synchronization if it
    // overlaps in memory with another unsynchronized node and at least one of them is a write.
    // Destination nodes are checked against both the written/read lists. Source nodes are only
    // checked against the written list. Two nodes overlap in memory if they come from the same
    // buffer and the tensor or view ranges overlap.
    auto const& overlaps_unsynced = [&](const v_tensor* node,
                                        const std::vector<const v_tensor*>& unsynced_nodes) -> bool {
      if (unsynced_nodes.size() == 0) { return false; }
      auto n_base                        = vk_tensor_offset(node) + node->view_offs;
      auto n_size                        = num_bytes(node);
      v_backend_vk_buffer_ctx* a_buf_ctx = (v_backend_vk_buffer_ctx*)node->buffer->context;
      vk_buffer a_buf                    = a_buf_ctx->dev_buffer;

      for (auto& other : unsynced_nodes) {
        v_backend_vk_buffer_ctx* o_buf_ctx = (v_backend_vk_buffer_ctx*)other->buffer->context;
        vk_buffer o_buf                    = o_buf_ctx->dev_buffer;
        if (a_buf == o_buf) {
          auto o_base = vk_tensor_offset(other) + other->view_offs;
          auto o_size = num_bytes(other);

          if ((o_base <= n_base && n_base < o_base + o_size) ||
            (n_base <= o_base && o_base < n_base + n_size)) { return true; }
        }
      }
      return false;
    };

    // For all fused ops, check if the destination node or any of the source
    // nodes require synchronization.
    for (int32_t i = 0; i < ctx->num_additional_fused_ops + 1 && !need_sync; ++i) {
      const v_tensor* cur_node = cgraph->nodes[node_idx + i];
      if (overlaps_unsynced(cur_node, ctx->unsynced_nodes_read) || overlaps_unsynced(
        cur_node,
        ctx->unsynced_nodes_written)) {
        need_sync = true;
        break;
      }
      for (uint32_t j = 0; j < V_MAX_SRC; ++j) {
        if (!cur_node->src[j]) { continue; }
        if (overlaps_unsynced(cur_node->src[j], ctx->unsynced_nodes_written)) {
          need_sync = true;
          break;
        }
      }
    }
    if (need_sync) {
      ctx->unsynced_nodes_written.clear();
      ctx->unsynced_nodes_read.clear();
      vk_sync_buffers(ctx, compute_ctx);
    }
    // Add all fused nodes to the unsynchronized lists.
    for (int32_t i = 0; i < ctx->num_additional_fused_ops + 1; ++i) {
      const v_tensor* cur_node = cgraph->nodes[node_idx + i];
      // Multiple outputs could be written, e.g. in topk_moe. Add them all to the list.
      ctx->unsynced_nodes_written.push_back(cur_node);
      for (uint32_t j = 0; j < V_MAX_SRC; ++j) {
        if (!cur_node->src[j]) { continue; }
        ctx->unsynced_nodes_read.push_back(cur_node->src[j]);
      }
    }
  }

  switch (node->op) {
    case v_OP_REPEAT:
      v_vk_repeat(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_REPEAT_BACK:
      v_vk_repeat_back(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_ACC:
      v_vk_acc(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_GET_ROWS:
      v_vk_get_rows(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_ADD:
      if (ctx->num_additional_fused_ops) { v_vk_multi_add(ctx, compute_ctx, cgraph, node_idx, dryrun); }
      else { v_vk_add(ctx, compute_ctx, src0, src1, node, dryrun); }
      break;
    case v_OP_SUB:
      v_vk_sub(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_MUL:
      v_vk_mul(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_DIV:
      v_vk_div(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_ADD_ID:
      v_vk_add_id(ctx, compute_ctx, src0, src1, src2, node, dryrun);

      break;
    case v_OP_CONCAT:
      v_vk_concat(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_UPSCALE:
      v_vk_upscale(ctx, compute_ctx, src0, node, dryrun);

      break;
    case V_OP_SCALE:
      v_vk_scale(ctx, compute_ctx, src0, node, dryrun);

      break;
    case V_OP_SQR:
      v_vk_sqr(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_SQRT:
      v_vk_sqrt(ctx, compute_ctx, src0, node, dryrun);

      break;
    case V_OP_SIN:
      v_vk_sin(ctx, compute_ctx, src0, node, dryrun);

      break;
    case V_OP_COS:
      v_vk_cos(ctx, compute_ctx, src0, node, dryrun);

      break;
    case V_OP_CLAMP:
      v_vk_clamp(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_PAD:
      v_vk_pad(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_ROLL:
      v_vk_roll(ctx, compute_ctx, src0, node, dryrun);

      break;
    case V_OP_CPY:
    case V_OP_CONT:
    case v_OP_DUP:
      v_vk_cpy(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_SET_ROWS:
      v_vk_set_rows(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_SILU_BACK: v_vk_silu_back(ctx, compute_ctx, src0, src1, node, dryrun);
      break;
    case v_OP_NORM: v_vk_norm(ctx, compute_ctx, src0, node, dryrun);
      break;
    case v_OP_GROUP_NORM:
      v_vk_group_norm(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_RMS_NORM:
      if (ctx->num_additional_fused_ops > 0) {
        // fused rms_norm + mul
        v_tensor* mul       = cgraph->nodes[node_idx + 1];
        v_tensor* other_src = mul->src[0] == node
                                ? mul->src[1]
                                : mul->src[0];
        v_vk_rms_norm(ctx, compute_ctx, src0, other_src, mul, reinterpret_cast<float*>(node->op_params.data()), dryrun);
      }
      else { v_vk_rms_norm(ctx, compute_ctx, src0, src0, node, reinterpret_cast<float*>(node->op_params.data()), dryrun); }
      break;
    case v_OP_RMS_NORM_BACK:
      v_vk_rms_norm_back(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_L2_NORM:
      v_vk_l2_norm(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_UNARY:
      switch (v_get_unary_op(node)) {
        case v_UNARY_OP_EXP:
        case v_UNARY_OP_SILU:
        case v_UNARY_OP_GELU:
        case v_UNARY_OP_GELU_ERF:
        case v_UNARY_OP_GELU_QUICK:
        case v_UNARY_OP_RELU:
        case v_UNARY_OP_TANH:
        case v_UNARY_OP_SIGMOID:
        case V_UNARY_OP_LOG:
        case v_UNARY_OP_HARDSIGMOID:
        case v_UNARY_OP_HARDSWISH:
          v_vk_unary(ctx, compute_ctx, src0, node, dryrun);
          break;
        default:
          std::cout << " false case operation :: " << v_get_unary_op(node) << std::endl;
          throw std::runtime_error("fail to operate ");

          return false;
      }
      break;
    case v_OP_GLU:
      switch (v_get_glu_op(node)) {
        case v_GLU_OP_GEGLU:
        case v_GLU_OP_REGLU:
        case v_GLU_OP_SWIGLU:
        case v_GLU_OP_SWIGLU_OAI:
        case v_GLU_OP_GEGLU_ERF:
        case v_GLU_OP_GEGLU_QUICK:
          v_vk_glu(ctx, compute_ctx, src0, src1, node, dryrun);
          break;
        default:
          return false;
      }
      break;
    case V_OP_DIAG_MASK_INF:
      v_vk_diag_mask_inf(ctx, compute_ctx, src0, node, dryrun);

      break;
    case V_OP_SOFT_MAX:
      if (ctx->num_additional_fused_ops) { v_vk_topk_moe(ctx, compute_ctx, cgraph, node_idx, dryrun); }
      else { v_vk_soft_max(ctx, compute_ctx, src0, src1, src2, node, dryrun); }

      break;
    case v_OP_SOFT_MAX_BACK:
      v_vk_soft_max_back(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case V_OP_ROPE:
      v_vk_rope(ctx, compute_ctx, src0, src1, src2, node, false, dryrun);

      break;
    case v_OP_ROPE_BACK:
      v_vk_rope(ctx, compute_ctx, src0, src1, src2, node, true, dryrun);

      break;
    case v_OP_ARGSORT:
      v_vk_argsort(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_SUM:
      v_vk_sum(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_SUM_ROWS:
      v_vk_sum_rows(ctx, compute_ctx, src0, node, dryrun);

      break;
    case V_OP_MEAN:
      v_vk_mean(ctx, compute_ctx, src0, node, dryrun);

      break;
    case V_OP_ARGMAX:
      v_vk_argmax(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_COUNT_EQUAL:
      v_vk_count_equal(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case V_OP_IM2COL:
      v_vk_im2col(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_IM2COL_3D:
      v_vk_im2col_3d(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_TIMESTEP_EMBEDDING:
      v_vk_timestep_embedding(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_CONV_TRANSPOSE_1D:
      v_vk_conv_transpose_1d(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case V_OP_POOL_2D:
      v_vk_pool_2d(ctx, compute_ctx, src0, node, dryrun);
      break;

    case V_OP_POOL_2D_BACK:
      v_vk_pool_2d_back(ctx, compute_ctx, src0, node, dryrun);
      break;
    case v_OP_CONV_2D:
      v_vk_conv_2d(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_CONV_TRANSPOSE_2D:
      v_vk_conv_transpose_2d(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_CONV_2D_DW:
      v_vk_conv_2d_dw(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case V_OP_LEAKY_RELU:
      v_vk_leaky_relu(ctx, compute_ctx, src0, node, dryrun);

      break;
    case V_OP_MUL_MAT:
      v_vk_mul_mat(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_MUL_MAT_ID:
      v_vk_mul_mat_id(ctx, compute_ctx, src0, src1, src2, node, dryrun);

      break;

    case v_OP_FLASH_ATTN_EXT:
      v_vk_flash_attn(ctx, compute_ctx, src0, src1, src2, src3, node->src[4], node, dryrun);

      break;

    case v_OP_RWKV_WKV6:
      v_vk_rwkv_wkv6(ctx, compute_ctx, node, dryrun);

      break;

    case v_OP_RWKV_WKV7:
      v_vk_rwkv_wkv7(ctx, compute_ctx, node, dryrun);

      break;

    case v_OP_SSM_SCAN:
      v_vk_ssm_scan(ctx, compute_ctx, node, dryrun);

      break;

    case v_OP_SSM_CONV:
      v_vk_ssm_conv(ctx, compute_ctx, node, dryrun);

      break;

    case v_OP_OPT_STEP_ADAMW:
      v_vk_opt_step_adamw(ctx, compute_ctx, node, dryrun);

      break;

    case v_OP_OPT_STEP_SGD:
      v_vk_opt_step_sgd(ctx, compute_ctx, src0, src1, src2, node, dryrun);

      break;
    default:
      return false;
  }

  if (dryrun) { return false; }

  ctx->tensor_ctxs[node_idx] = compute_ctx;

  #if defined(v_VULKAN_CHECK_RESULTS)
  // Force context reset on each node so that each tensor ends up in its own context
  // and can be run and compared to its CPU equivalent separately
  last_node = true;
  #endif

  if (submit || last_node) {
    vk_ctx_end(compute_ctx);

    // TODO probably it'd be better to pass a exit_node flag to v_vk_compute_forward
    if (last_node) { compute_ctx->exit_tensor_idx = node_idx_begin; }
    else { compute_ctx->exit_tensor_idx = -1; }

    ctx->compute_ctx.reset();

    bool ok = v_vk_compute_forward(ctx, cgraph, node_begin, node_idx_begin, false, almost_ready);
    if (!ok) {
      if (node->op == v_OP_UNARY) {
        std::cerr << __func__ << ": error: op not supported UNARY " << node->name.data() << " (" << v_unary_op_name(
          static_cast<v_unary_op>(node->op_params[0])) << ")" << std::endl;
      }
      else if (node->op == v_OP_GLU) {
        std::cerr << __func__ << ": error: op not supported GLU " << node->name.data() << " (" << v_glu_op_name(
          static_cast<v_glu_op>(node->op_params[0])) << ")" << std::endl;
      }
      else {
        std::cerr << __func__ << ": error: op not supported " << node->name.data() << " (" << v_op_name(node->op) << ")" <<
          std::endl;
      }
    }
  }
  return true;
}
// Sort the graph for improved parallelism.
void vk_graph_optimize(v_backend_t backend, struct v_cgraph* graph) {
  VK_LOG_DEBUG("v_vk_graph_optimize(" << graph->n_nodes << " nodes)");
  vk_backend_ctx* ctx = (vk_backend_ctx*)backend->context;
  if (ctx->device->disable_graph_optimize) { return; }
  auto const& is_empty = [](v_tensor* node) -> bool {
    return node->op == v_OP_NONE || node->op == V_OP_RESHAPE || node->op == v_OP_TRANSPOSE || node->op ==
      V_OP_VIEW || node->op == V_OP_PERMUTE;
  };

  auto const& is_src_of = [](const v_tensor* dst, const v_tensor* src) -> bool {
    for (uint32_t s = 0; s < V_MAX_SRC; ++s) { if (dst->src[s] == src) { return true; } }
    // implicit dependency if they view the same tensor
    const v_tensor* dst2 = dst->view_src
                             ? dst->view_src
                             : dst;
    const v_tensor* src2 = src->view_src
                             ? src->view_src
                             : src;
    if (dst2 == src2) { return true; }
    return false;
  };

  // This function tries to reorder the graph to allow nodes to run in parallel.
  // This helps with small batches, but for large batches its a slowdown, probably
  // due to cache contention. So only reorder if the majority of nodes have few rows.
  int num_small_nodes   = 0;
  int num_counted_nodes = 0;
  for (int i = 0; i < graph->n_nodes; ++i) {
    if (!is_empty(graph->nodes[i]) &&
      graph->nodes[i]->op != v_OP_SET_ROWS) {
      if (v_nrows(graph->nodes[i]) <= 8) { num_small_nodes++; }
      num_counted_nodes++;
    }
  }
  if (num_small_nodes < num_counted_nodes / 2) { return; }

  std::vector<v_tensor*> new_order;
  std::vector<bool> used(graph->n_nodes, false);
  int first_unused = 0;
  while (first_unused < graph->n_nodes) {
    std::vector<int> current_set;

    // Avoid reordering topk_moe_norm
    if (first_unused + (int)topk_moe_norm.size() <= graph->n_nodes) {
      bool is_topk_moe_norm = true;
      for (size_t j = 0; j < topk_moe_norm.size(); ++j) { if (graph->nodes[first_unused + j]->op != topk_moe_norm[j] || used[first_unused + j]) { is_topk_moe_norm = false; } }
      if (is_topk_moe_norm) {
        for (size_t j = 0; j < topk_moe_norm.size(); ++j) {
          new_order.push_back(graph->nodes[first_unused + j]);
          used[first_unused + j] = true;
        }
        while (first_unused < graph->n_nodes && used[first_unused]) { first_unused++; }
        continue;
      }
    }
    // First, grab the next unused node.
    current_set.push_back(first_unused);

    // Loop through the next N nodes. Grab any that don't depend on other nodes that
    // haven't already been run. Nodes that have already been run have used_bits__[i] set
    // to true. Allow nodes that depend on the previous node if it's a fusion pattern
    // that we support (e.g. RMS_NORM + MUL).
    // This first pass only grabs "real" (non-view nodes). Second pass grabs view nodes.
    // The goal is to not interleave real and view nodes in a way that breaks fusion.
    const int NUM_TO_CHECK = 20;
    for (int j = first_unused + 1; j < std::min(first_unused + NUM_TO_CHECK, graph->n_nodes); ++j) {
      if (used[j]) { continue; }
      if (is_empty(graph->nodes[j])) { continue; }
      bool ok = true;
      for (int c = first_unused; c < j; ++c) {
        if (!used[c] &&
          is_src_of(graph->nodes[j], graph->nodes[c]) &&
          !(j == c + 1 && c == current_set.back() && graph->nodes[c]->op == v_OP_RMS_NORM && graph->nodes[j]->op ==
            v_OP_MUL)) {
          ok = false;
          break;
        }
      }
      if (ok) { current_set.push_back(j); }
    }
    // Second pass grabs view nodes.
    // Skip this if it would break a fusion optimization (don't split up add->rms_norm or add->add).
    if (graph->nodes[current_set.back()]->op != v_OP_ADD) {
      for (int j = first_unused + 1; j < std::min(first_unused + NUM_TO_CHECK, graph->n_nodes); ++j) {
        if (used[j]) { continue; }
        if (!is_empty(graph->nodes[j])) { continue; }
        bool ok = true;
        for (int c = first_unused; c < j; ++c) {
          bool c_in_current_set = std::find(current_set.begin(), current_set.end(), c) != current_set.end();
          // skip views whose srcs haven't been processed.
          if (!used[c] &&
            is_src_of(graph->nodes[j], graph->nodes[c]) &&
            !c_in_current_set) {
            ok = false;
            break;
          }
        }
        if (ok) { current_set.push_back(j); }
      }
    }

    // Push the current set into new_order
    for (auto c : current_set) {
      new_order.push_back(graph->nodes[c]);
      used[c] = true;
    }
    while (first_unused < graph->n_nodes && used[first_unused]) { first_unused++; }
  }
  // Replace the graph with the new order.
  for (int i = 0; i < graph->n_nodes; ++i) { graph->nodes[i] = new_order[i]; }
}
