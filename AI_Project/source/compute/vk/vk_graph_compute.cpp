#include "vk_common.h"
#include "vk_pipeline.h"
#include "vk_queue.h"
#include "vk_device.h"
#include "vk_constant.h"
#include "vk_buffer.h"
#include "vk_context.h"
#include "vk_util.h"
#include "v_util.h"
#include "vk_op_f32.hpp"
#include "vk_comp.hpp"
bool vk_is_empty(v_tensor* node) {
  return is_empty(node) || node->op == v_OP_NONE || node->op == v_OP_RESHAPE || node->op == v_OP_TRANSPOSE
    || node->op == V_OP_VIEW || node->op == V_OP_PERMUTE;
}

bool v_vk_can_fuse(const struct v_cgraph* cgraph, int node_idx,
                   std::initializer_list<enum v_operation> ops) {
  if (!v_can_fuse(cgraph, node_idx, ops)) { return false; }
  if (ops.size() == 2 && ops.begin()[0] == v_OP_RMS_NORM && ops.begin()[1] == v_OP_MUL) {
    // additional constraints specific to this fusion
    const v_tensor* rms_norm = cgraph->nodes[node_idx];
    const v_tensor* mul      = cgraph->nodes[node_idx + 1];

    V_ASSERT(rms_norm->src[0]->type == v_TYPE_F32);
    V_ASSERT(rms_norm->type == v_TYPE_F32);
    // rms_norm only supports f32
    if (mul->src[0]->type != v_TYPE_F32 ||
      mul->src[1]->type != v_TYPE_F32 ||
      mul->type != v_TYPE_F32) { return false; }
    // if rms_norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] &&
      !v_are_same_shape(mul->src[0], rms_norm)) { return false; }
    // rms_norm shader assumes contiguous rows
    if (!v_is_contiguous_rows(mul->src[0]) || !v_is_contiguous_rows(mul->src[1])) { return false; }
  }
  return true;
}

bool v_vk_can_fuse_topk_moe(vk_backend_ctx* ctx, const struct v_cgraph* cgraph,
                            int node_idx, bool with_norm) {
  if (with_norm) {
    if (node_idx + (int)topk_moe_norm.size() > cgraph->n_nodes) { return false; }
    for (size_t i = 0; i < topk_moe_norm.size(); ++i) { if (cgraph->nodes[node_idx + i]->op != topk_moe_norm[i]) { return false; } }
  }
  else {
    if (node_idx + (int)topk_moe.size() > cgraph->n_nodes) { return false; }
    for (size_t i = 0; i < topk_moe.size(); ++i) { if (cgraph->nodes[node_idx + i]->op != topk_moe[i]) { return false; } }
  }

  const v_tensor* softmax = cgraph->nodes[node_idx + 0];
  const v_tensor* weights = with_norm
                              ? cgraph->nodes[node_idx + 8]
                              : cgraph->nodes[node_idx + 4];

  const float* op_params = (const float*)softmax->op_params;

  float scale    = op_params[0];
  float max_bias = op_params[1];

  if (!v_is_contiguous(softmax->src[0]) || !v_is_contiguous(weights)) { return false; }

  if (scale != 1.0f || max_bias != 0.0f) { return false; }

  // don't fuse when masks or sinks are present
  if (softmax->src[1] || softmax->src[2]) { return false; }

  const int n_expert = softmax->ne[0];
  // n_expert must be a power of 2
  if (!is_pow2(n_expert) || n_expert > (1 << (num_topk_moe_pipelines - 1))) { return false; }

  // Check that the nodes don't have any unexpected uses
  const v_tensor* reshape1 = cgraph->nodes[node_idx + 1];
  const v_tensor* argsort  = cgraph->nodes[node_idx + 2];
  const v_tensor* view     = cgraph->nodes[node_idx + 3];
  const v_tensor* get_rows = cgraph->nodes[node_idx + 4];
  const v_tensor* reshape5 = with_norm
                               ? cgraph->nodes[node_idx + 5]
                               : nullptr;
  const v_tensor* sum_rows = with_norm
                               ? cgraph->nodes[node_idx + 6]
                               : nullptr;
  const v_tensor* div = with_norm
                          ? cgraph->nodes[node_idx + 7]
                          : nullptr;
  const v_tensor* reshape8 = with_norm
                               ? cgraph->nodes[node_idx + 8]
                               : nullptr;

  // softmax is used by reshape and argsort
  if (v_node_get_use_count(cgraph, node_idx) != 2 ||
    reshape1->src[0] != softmax ||
    argsort->src[0] != softmax) { return false; }
  // reshape is used by get_rows
  if (v_node_get_use_count(cgraph, node_idx + 1) != 1 ||
    get_rows->src[0] != reshape1) { return false; }
  // argsort is used by view
  if (v_node_get_use_count(cgraph, node_idx + 2) != 1 ||
    view->src[0] != argsort) { return false; }
  // view is written (via argsort), we can skip checking it

  if (with_norm) {
    // get_rows is used by reshape
    if (v_node_get_use_count(cgraph, node_idx + 4) != 1 ||
      reshape5->src[0] != get_rows) { return false; }

    // reshape is used by sum_rows and div
    if (v_node_get_use_count(cgraph, node_idx + 5) != 2 ||
      sum_rows->src[0] != reshape5 ||
      div->src[0] != reshape5) { return false; }

    // sum_rows is used by div
    if (v_node_get_use_count(cgraph, node_idx + 6) != 1 ||
      div->src[1] != sum_rows) { return false; }

    // div/reshape are written
    if (reshape8->src[0] != div) { return false; }
  }

  if (!ctx->device->subgroup_arithmetic ||
    !ctx->device->subgroup_shuffle ||
    !ctx->device->subgroup_require_full_support ||
    ctx->device->disable_fusion) { return false; }

  return true;
}

v_status vk_graph_compute(v_backend_t backend, v_cgraph* cgraph) {
  VK_LOG_DEBUG("v_backend_vk_graph_compute(" << cgraph->n_nodes << " nodes)");
  vk_backend_ctx* ctx = (vk_backend_ctx*)backend->context;

  if (vk_instance.debug_utils_support) {
    vk::DebugUtilsLabelEXT dul = {};
    dul.pLabelName             = "v_backend_vk_graph_compute";
    dul.color                  = std::array<float, 4>{1.0f, 1.0f, 1.0f, 1.0f};
    vk_instance.pfn_vkQueueBeginDebugUtilsLabelEXT(ctx->device->compute_queue.queue,
                                                   reinterpret_cast<VkDebugUtilsLabelEXT*>(&dul));
  }

  ctx->prealloc_size_add_rms_partials        = 0;
  ctx->prealloc_size_add_rms_partials_offset = 0;
  ctx->do_add_rms_partials                   = false;

  uint64_t total_mat_mul_bytes = 0;
  for (int i = 0; i < cgraph->n_nodes; i++) {
    if (!ctx->device->disable_fusion) {
      uint32_t num_adds = v_vk_fuse_multi_add(ctx, cgraph, i);
      if (num_adds) { ctx->num_additional_fused_ops = num_adds - 1; }
      else if (v_vk_can_fuse(cgraph, i, {v_OP_RMS_NORM, v_OP_MUL})) { ctx->num_additional_fused_ops = 1; }
      else if (v_vk_can_fuse_topk_moe(ctx, cgraph, i, true)) { ctx->num_additional_fused_ops = topk_moe_norm.size() - 1; }
      else if (v_vk_can_fuse_topk_moe(ctx, cgraph, i, false)) { ctx->num_additional_fused_ops = topk_moe.size() - 1; }
    }
    vk_build_graph(ctx, cgraph, i, nullptr, 0, true, false, false, false);
    if (cgraph->nodes[i]->op == v_OP_MUL_MAT || cgraph->nodes[i]->op == v_OP_MUL_MAT_ID) { total_mat_mul_bytes += num_bytes(cgraph->nodes[i]->src[0]); }
    else if (cgraph->nodes[i]->op == v_OP_CONV_2D || cgraph->nodes[i]->op == v_OP_CONV_TRANSPOSE_2D) {
      // Return CRSxNPQxsizeof(*) to account as many bytes as mul_mat has in im2col->mul_mat mode.
      auto CRS_size =
        cgraph->nodes[i]->src[0]->ne[0] * cgraph->nodes[i]->src[0]->ne[1] * cgraph->nodes[i]->src[1]->ne[2];
      auto NPQ_size = cgraph->nodes[i]->ne[0] * cgraph->nodes[i]->ne[1] * cgraph->nodes[i]->ne[3];
      total_mat_mul_bytes += NPQ_size * CRS_size * v_type_size(cgraph->nodes[i]->type);
    }
    i += ctx->num_additional_fused_ops;
    ctx->num_additional_fused_ops = 0;
  }
  if (ctx->device->need_compiles) { vk_load_shaders(ctx->device); }
  vk_ctx_pre_alloc_buffers(ctx);
  vk_pipeline_allocate_descriptor_sets(ctx);

  int last_node = cgraph->n_nodes - 1;

  // If the last op in the cgraph isn't backend GPU, the command buffer doesn't get closed properly
  while (last_node > 0 && vk_is_empty(cgraph->nodes[last_node])) { last_node -= 1; }

  // Reserve tensor context space for all nodes
  ctx->tensor_ctxs.resize(cgraph->n_nodes);

  bool first_node_in_batch = true; // true if next node will be first node in a batch
  int submit_node_idx      = 0; // index to first node in a batch

  vk_context compute_ctx;
  if (vk_perf_logger_enabled) {
    // allocate/resize the query pool
    if (ctx->device->num_queries < cgraph->n_nodes + 1) {
      if (ctx->device->query_pool) { ctx->device->device.destroyQueryPool(ctx->device->query_pool); }
      vk::QueryPoolCreateInfo query_create_info;
      query_create_info.queryType  = vk::QueryType::eTimestamp;
      query_create_info.queryCount = cgraph->n_nodes + 100;
      ctx->device->query_pool      = ctx->device->device.createQueryPool(query_create_info);
      ctx->device->num_queries     = query_create_info.queryCount;
    }

    ctx->device->device.resetQueryPool(ctx->device->query_pool, 0, cgraph->n_nodes + 1);

    V_ASSERT(ctx->compute_ctx.expired());
    compute_ctx      = vk_create_context(ctx, ctx->compute_cmd_pool);
    ctx->compute_ctx = compute_ctx;
    vk_begin_ctx(ctx->device, compute_ctx);
    compute_ctx->s->buffer.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->device->query_pool, 0);
  }

  ctx->prealloc_y_last_pipeline_used = nullptr;
  ctx->prealloc_y_last_tensor_used   = nullptr;

  if (ctx->prealloc_size_add_rms_partials) {
    if (ctx->compute_ctx.expired()) {
      compute_ctx      = vk_create_context(ctx, ctx->compute_cmd_pool);
      ctx->compute_ctx = compute_ctx;
      vk_begin_ctx(ctx->device, compute_ctx);
    }
    else { compute_ctx = ctx->compute_ctx.lock(); }
    // initialize partial sums to zero.
    vk_buffer_memset_async(compute_ctx, ctx->prealloc_add_rms_partials, 0, 0, ctx->prealloc_size_add_rms_partials);
    vk_sync_buffers(ctx, compute_ctx);
  }

  // Submit after enough work has accumulated, to overlap CPU cmdbuffer generation with GPU execution.
  // Estimate the amount of matmul work by looking at the weight matrix size, and submit every 100MB
  // (and scaled down based on model size, so smaller models submit earlier).
  // Also submit at least every 100 nodes, in case there are workloads without as much matmul.
  int nodes_per_submit              = 100;
  int submitted_nodes               = 0;
  int submit_count                  = 0;
  uint64_t mul_mat_bytes            = 0;
  uint64_t mul_mat_bytes_per_submit = std::min(uint64_t(100 * 1000 * 1000), total_mat_mul_bytes / 40u);
  for (int i = 0; i < cgraph->n_nodes; i++) {
    if (first_node_in_batch) { submit_node_idx = i; }

    if (cgraph->nodes[i]->op == v_OP_MUL_MAT || cgraph->nodes[i]->op == v_OP_MUL_MAT_ID) { mul_mat_bytes += num_bytes(cgraph->nodes[i]->src[0]); }

    if (!ctx->device->disable_fusion) {
      uint32_t num_adds = v_vk_fuse_multi_add(ctx, cgraph, i);
      if (num_adds) { ctx->num_additional_fused_ops = num_adds - 1; }
      else if (v_vk_can_fuse(cgraph, i, {v_OP_RMS_NORM, v_OP_MUL})) { ctx->num_additional_fused_ops = 1; }
      else if (v_vk_can_fuse_topk_moe(ctx, cgraph, i, true)) { ctx->num_additional_fused_ops = topk_moe_norm.size() - 1; }
      else if (v_vk_can_fuse_topk_moe(ctx, cgraph, i, false)) { ctx->num_additional_fused_ops = topk_moe.size() - 1; }
    }

    // Signal the almost_ready fence when the graph is mostly complete (< 20% remaining)
    bool almost_ready = (cgraph->n_nodes - i) < cgraph->n_nodes / 5;
    bool submit       = (submitted_nodes >= nodes_per_submit) ||
      (mul_mat_bytes >= mul_mat_bytes_per_submit) ||
      (i + ctx->num_additional_fused_ops >= last_node) ||
      (almost_ready && !ctx->almost_ready_fence_pending);

    bool enqueued = vk_build_graph(ctx,
                                   cgraph,
                                   i,
                                   cgraph->nodes[submit_node_idx],
                                   submit_node_idx,
                                   false,
                                   i + ctx->num_additional_fused_ops >= last_node,
                                   almost_ready,
                                   submit);

    if (vk_perf_logger_enabled) {
      if (ctx->compute_ctx.expired()) {
        compute_ctx      = vk_create_context(ctx, ctx->compute_cmd_pool);
        ctx->compute_ctx = compute_ctx;
        vk_begin_ctx(ctx->device, compute_ctx);
      }
      else { compute_ctx = ctx->compute_ctx.lock(); }
      // If there are fused ops, just write out timestamps for all nodes to keep the accounting simple
      for (int j = 0; j < ctx->num_additional_fused_ops + 1; ++j) {
        compute_ctx->s->buffer.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands,
                                              ctx->device->query_pool,
                                              i + j + 1);
      }
    }

    if (enqueued) {
      ++submitted_nodes;

      #ifndef v_VULKAN_CHECK_RESULTS
      if (first_node_in_batch) { first_node_in_batch = false; }
      #endif
    }

    if (submit && enqueued) {
      first_node_in_batch = true;
      submitted_nodes     = 0;
      mul_mat_bytes       = 0;
      if (submit_count < 3) { mul_mat_bytes_per_submit *= 2; }
      submit_count++;
    }
    i += ctx->num_additional_fused_ops;
    ctx->num_additional_fused_ops = 0;
  }

  if (vk_perf_logger_enabled) {
    // End the command buffer and submit/wait
    V_ASSERT(!ctx->compute_ctx.expired());
    compute_ctx = ctx->compute_ctx.lock();
    vk_ctx_end(compute_ctx);

    vk_submit(compute_ctx, ctx->device->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->device->fence }, true, UINT64_MAX),
             "v_VULKAN_PERF waitForFences");
    ctx->device->device.resetFences({ctx->device->fence});

    // Get the results and pass them to the logger
    std::vector<uint64_t> timestamps(cgraph->n_nodes + 1);
    VK_CHECK(
      ctx->device->device.getQueryPoolResults(ctx->device->query_pool, 0, cgraph->n_nodes + 1, (cgraph->n_nodes + 1)*
        sizeof(uint64_t), timestamps.data(), sizeof(uint64_t), vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::
        eWait),
      "get timestamp results");
    for (int i = 0; i < cgraph->n_nodes; i++) {
      if (!vk_is_empty(cgraph->nodes[i])) {
        ctx->device->perf_logger->log_timing(cgraph->nodes[i],
                                             uint64_t(
                                               (timestamps[i + 1] - timestamps[i]) * ctx->device->properties.limits.
                                                                                          timestampPeriod));
      }
    }

    ctx->device->perf_logger->print_timings();
  }

  vk_graph_cleanup(ctx);

  return v_STATUS_SUCCESS;

  UNUSED(backend);
}


bool v_vk_compute_forward(vk_backend_ctx* ctx, v_cgraph* cgraph, v_tensor* tensor,
                          int tensor_idx, bool use_fence = true, bool almost_ready = false) {
  v_UNUSED(cgraph);
  v_backend_buffer* buf = nullptr;

  switch (tensor->op) {
    case v_OP_ADD:
    case v_OP_ACC:
    case v_OP_GET_ROWS:
    case v_OP_SUB:
    case v_OP_MUL:
    case v_OP_DIV:
    case v_OP_ADD_ID:
    case v_OP_CONCAT:
    case v_OP_UPSCALE:
    case V_OP_SCALE:
    case v_OP_SQR:
    case v_OP_SQRT:
    case v_OP_SIN:
    case v_OP_COS:
    case v_OP_CLAMP:
    case v_OP_PAD:
    case v_OP_ROLL:
    case v_OP_CPY:
    case v_OP_SET_ROWS:
    case v_OP_CONT:
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
    case v_OP_RESHAPE:
    case V_OP_VIEW:
    case V_OP_PERMUTE:
    case v_OP_TRANSPOSE:
    case v_OP_NONE:
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
    case v_OP_LEAKY_RELU:
    case v_OP_REPEAT:
    case v_OP_REPEAT_BACK:
    case v_OP_OPT_STEP_ADAMW:
    case v_OP_OPT_STEP_SGD:
      buf = tensor->buffer;
      break;
    case v_OP_UNARY:
      switch (v_get_unary_op(tensor)) {
        case v_UNARY_OP_EXP:
        case v_UNARY_OP_LOG:
        case v_UNARY_OP_SILU:
        case v_UNARY_OP_GELU:
        case v_UNARY_OP_GELU_ERF:
        case v_UNARY_OP_GELU_QUICK:
        case v_UNARY_OP_RELU:
        case v_UNARY_OP_TANH:
        case v_UNARY_OP_SIGMOID:
        case v_UNARY_OP_HARDSIGMOID:
        case v_UNARY_OP_HARDSWISH:
          buf = tensor->buffer;
          break;
        default:
          return false;
      }
      break;
    case v_OP_GLU:
      switch (v_get_glu_op(tensor)) {
        case v_GLU_OP_GEGLU:
        case v_GLU_OP_REGLU:
        case v_GLU_OP_SWIGLU:
        case v_GLU_OP_SWIGLU_OAI:
        case v_GLU_OP_GEGLU_ERF:
        case v_GLU_OP_GEGLU_QUICK:
          buf = tensor->buffer;
          break;
        default:
          return false;
      }
      break;
    case v_OP_MUL_MAT:
    case v_OP_MUL_MAT_ID:
    case v_OP_FLASH_ATTN_EXT:
      buf = tensor->buffer;

      break;
    default:
      return false;
  }

  if (buf == nullptr) { return false; }

  VK_LOG_DEBUG(
    "v_vk_compute_forward(" << tensor << ", name=" << tensor->name << ", op=" << op_name(tensor->op) <<
    ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] <<
    ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] <<
    ", nb3=" << tensor->nb[3] << ", view_src=" << tensor->view_src << ", view_offs=" << tensor->view_offs << ")");

  vk_context subctx = ctx->tensor_ctxs[tensor_idx].lock();
  // always wait for the GPU work to be done for the last submit
  if (tensor_idx == subctx->exit_tensor_idx) { use_fence = true; }

  // Only run if ctx hasn't been submitted yet
  if (!subctx->seqs.empty()) {
    #ifdef v_VULKAN_CHECK_RESULTS
    vk_check_results_0(ctx, cgraph, tensor_idx);
    use_fence = true;
    #endif

    // Do staging buffer copies
    for (auto& cpy : subctx->in_memcpys) { memcpy(cpy.dst, cpy.src, cpy.n); }

    for (auto& mset : subctx->memsets) { memset(mset.dst, mset.val, mset.n); }

    if (almost_ready && !ctx->almost_ready_fence_pending && !use_fence) {
      vk_submit(subctx, ctx->almost_ready_fence);
      ctx->almost_ready_fence_pending = true;
    }
    else {
      vk_submit(subctx, use_fence
                          ? ctx->fence
                          : vk::Fence{});
    }

    if (use_fence) { v_vk_wait_for_fence(ctx); }
    #ifdef v_VULKAN_CHECK_RESULTS
    vk_check_results_1(ctx, cgraph, tensor_idx);
    #endif
  }

  if (tensor_idx == subctx->exit_tensor_idx) {
    // Do staging buffer copies
    for (auto& cpy : subctx->out_memcpys) { memcpy(cpy.dst, cpy.src, cpy.n); }
    subctx->in_memcpys.clear();
    subctx->out_memcpys.clear();
    subctx->memsets.clear();
  }

  return true;
}

// Clean up after graph processing is done
void vk_graph_cleanup(vk_backend_ctx* ctx) {
  VK_LOG_DEBUG("v_vk_graph_cleanup()");
  ctx->prealloc_y_last_pipeline_used = {};

  ctx->unsynced_nodes_written.clear();
  ctx->unsynced_nodes_read.clear();
  ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_split_k_need_sync = false;

  v_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);
  v_vk_command_pool_cleanup(ctx->device, ctx->transfer_cmd_pool);

  for (size_t i = 0; i < ctx->gc.semaphores.size(); i++) { ctx->device->device.destroySemaphore({ctx->gc.semaphores[i].s}); }
  ctx->gc.semaphores.clear();

  for (size_t i = 0; i < ctx->gc.tl_semaphores.size(); i++) { ctx->device->device.destroySemaphore({ctx->gc.tl_semaphores[i].s}); }
  ctx->gc.tl_semaphores.clear();
  ctx->semaphore_idx = 0;

  ctx->event_idx = 0;

  for (auto& event : ctx->gc.events) { ctx->device->device.resetEvent(event); }

  ctx->tensor_ctxs.clear();
  ctx->gc.contexts.clear();
  ctx->pipeline_descriptor_set_requirements = 0;
  ctx->descriptor_set_idx                   = 0;
}