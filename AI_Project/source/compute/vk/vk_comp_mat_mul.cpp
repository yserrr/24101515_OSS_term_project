#include <vk_constant.h>
#include <vk_context.h>
#include <vk_util.h>
#include "vk_op_f32.hpp"
#include "vk_pipeline.h"
#include "vk_comp.hpp"


void v_vk_matmul(vk_backend_ctx* ctx,
                 vk_context& subctx,
                 vk_pipeline& pipeline,
                 vk_sub_buffer&& a,
                 vk_sub_buffer&& b,
                 vk_sub_buffer&& d,
                 vk_sub_buffer&& split_k_buffer,
                 uint32_t m,
                 uint32_t n,
                 uint32_t k,
                 uint32_t stride_a,
                 uint32_t stride_b,
                 uint32_t stride_d,
                 uint32_t batch_stride_a,
                 uint32_t batch_stride_b, uint32_t batch_stride_d,
                 uint32_t split_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2,
                 uint32_t broadcast3,
                 uint32_t padded_n) {
  VK_LOG_DEBUG(
    "v_vk_matmul(a: (" << a.buffer->buffer << ", " << a.offset << ", " << a.size << "), b: (" << b.buffer->buffer <<
    ", " << b.offset << ", " << b.size << "), d: (" << d.buffer->buffer << ", " << d.offset << ", " << d.size <<
    "), split_k: (" << (split_k_buffer.buffer != nullptr ? split_k_buffer.buffer->buffer : VK_NULL_HANDLE) << ", " <<
    split_k_buffer.offset << ", " << split_k_buffer.size << "), m: " << m << ", n: " << n << ", k: " << k <<
    ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", batch_stride_a: " <<
    batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", split_k: "
    << split_k << ", batch: " << batch << ", ne02: " << ne02 << ", ne12: " << ne12 << ", broadcast2: " << broadcast2 <<
    ", broadcast3: " << broadcast3 << ", padded_n: " << padded_n << ")");
  if (split_k == 1) {
    const vk_mat_mat_push_constants pc = {
      m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, k, ne02, ne12, broadcast2,
      broadcast3, padded_n
    };
    v_vk_dispatch_pipeline(ctx, subctx, pipeline, {a, b, d}, pc, {m, n, batch});
    return;
  }

  if (ctx->prealloc_split_k_need_sync) { vk_sync_buffers(ctx, subctx); }

  V_ASSERT(batch_stride_d == m * n);

  // Round the split size up to a multiple of 256 (k-quant alignment)
  uint32_t k_split = CEIL_DIV(k, split_k);
  k_split          = ROUNDUP_POW2(k_split, 256);

  const vk_mat_mat_push_constants pc1 = {
    m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, k_split, ne02, ne12,
    broadcast2, broadcast3, padded_n
  };
  // Make sure enough workgroups get assigned for split k to work
  v_vk_dispatch_pipeline(ctx,
                         subctx,
                         pipeline,
                         {a, b, split_k_buffer},
                         pc1,
                         {
                           (CEIL_DIV(m, pipeline->wg_denoms[0]) * pipeline->wg_denoms[0]) * split_k, n, batch
                         });
  vk_sync_buffers(ctx, subctx);
  const std::array<uint32_t, 2> pc2 = {(uint32_t)(m * n * batch), split_k};
  v_vk_dispatch_pipeline(ctx,
                         subctx,
                         ctx->device->pipeline_matmul_split_k_reduce,
                         {split_k_buffer, d},
                         pc2,
                         {m * n * batch, 1, 1});
  ctx->prealloc_split_k_need_sync = true;
}

vk_pipeline v_vk_guess_matmul_id_pipeline(vk_backend_ctx* ctx, vk_matmul_pipeline& mmp, uint32_t m,
                                          uint32_t n, bool aligned, v_data_type src0_type) {
  VK_LOG_DEBUG(
    "v_vk_guess_matmul_id_pipeline(" << m << ", " << n << ", " << aligned << ", " << v_type_name(src0_type) <<
    ")");

  if (ctx->device->coopmat2) {
    // Use large shader when the N dimension is greater than the medium shader's tile size
    uint32_t crossover_large = mmp->m->wg_denoms[1];
    if ((ctx->device->mul_mat_id_l[src0_type] && (n > crossover_large)) || (!ctx->device->mul_mat_id_m[src0_type] && !
      ctx->device->mul_mat_id_s[src0_type])) {
      return aligned
               ? mmp->a_l
               : mmp->l;
    }
    // Use medium shader when the N dimension is greater than the small shader's tile size
    uint32_t crossover_medium = mmp->s->wg_denoms[1];
    if ((ctx->device->mul_mat_id_m[src0_type] && (n > crossover_medium)) || !ctx->device->mul_mat_id_s[src0_type]) {
      return aligned
               ? mmp->a_m
               : mmp->m;
    }
    return aligned
             ? mmp->a_s
             : mmp->s;
  }

  if ((ctx->device->mul_mat_id_s[src0_type] && (m <= 32 || n <= 32)) || (!ctx->device->mul_mat_id_m[src0_type] && !ctx->
                                                                                                                   device
                                                                                                                   ->
                                                                                                                   mul_mat_id_l
    [src0_type])) {
    return aligned
             ? mmp->a_s
             : mmp->s;
  }
  if ((ctx->device->mul_mat_id_m[src0_type] && (m <= 64 || n <= 64)) || !ctx->device->mul_mat_id_l[src0_type]) {
    return aligned
             ? mmp->a_m
             : mmp->m;
  }
  return aligned
           ? mmp->a_l
           : mmp->l;
}
bool v_vk_should_use_mmvq(const vk_device& device, uint32_t m, uint32_t n, uint32_t k, v_data_type src0_type) {
  if (device->mmvq_mode == 1) { return true; }
  else if (device->mmvq_mode == -1) { return false; }

  // MMVQ is generally good for batches
  if (n > 1) { return true; }

  switch (device->vendor_id) {
    case VK_VENDOR_ID_NVIDIA:
      switch (src0_type) {
      case v_TYPE_Q8_0:
          return device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
      default:
          return true;
      }
    case VK_VENDOR_ID_AMD:
      switch (src0_type) {
      case v_TYPE_Q8_0:
          return device->architecture == vk_device_architecture::AMD_GCN;
      default:
          return true;
      }
    case VK_VENDOR_ID_INTEL:
      switch (src0_type) {
        // From tests on A770 Linux, may need more tuning
      case v_TYPE_Q4_0:
      case v_TYPE_Q5_1:
          return false;
      default:
          return true;
      }
    default:
      return true;
  }

  v_UNUSED(m);
  v_UNUSED(k);
}


uint32_t v_vk_guess_matmul_id_pipeline_align(vk_backend_ctx* ctx, vk_matmul_pipeline& mmp, int m,
                                             int n, v_data_type src0_type) {
  VK_LOG_DEBUG("v_vk_guess_matmul_pipeline_align(" << m << ", " << n << ", " << v_type_name(src0_type) << ")");
  return v_vk_guess_matmul_id_pipeline(ctx, mmp, m, n, true, src0_type)->align;
}



uint32_t v_vk_fuse_multi_add(vk_backend_ctx* ctx, const struct v_cgraph* cgraph,
           int node_idx) {
  const v_tensor* first_node = cgraph->nodes[node_idx];
  if (first_node->op != v_OP_ADD) { return 0; }

  if (!ctx->device->multi_add) { return 0; }

  int32_t num_adds = 1;
  while (node_idx + num_adds < cgraph->n_nodes &&
    cgraph->nodes[node_idx + num_adds]->op == v_OP_ADD &&
    num_adds < MAX_FUSED_ADDS) { num_adds++; }

  // The shader currently requires same shapes (but different strides are allowed),
  // everything f32, and no misalignment
  for (int32_t i = 0; i < num_adds; ++i) {
    const v_tensor* next_node = cgraph->nodes[node_idx + i];
    if (!v_are_same_shape(first_node, next_node->src[0]) ||
      !v_are_same_shape(first_node, next_node->src[1]) ||
      next_node->type != v_TYPE_F32 ||
      next_node->src[0]->type != v_TYPE_F32 ||
      next_node->src[1]->type != v_TYPE_F32 ||
      get_misalign_bytes(ctx, next_node) ||
      get_misalign_bytes(ctx, next_node->src[0]) ||
      get_misalign_bytes(ctx, next_node->src[1])) { num_adds = i; }
  }

  // Verify we can fuse these
  v_operation adds[MAX_FUSED_ADDS];
  for (int32_t i = 0; i < num_adds; ++i) { adds[i] = v_OP_ADD; }

  // decrease num_adds if they can't all be fused
  while (num_adds > 1 && !v_can_fuse(cgraph, node_idx, adds, num_adds)) { num_adds--; }

  // a single add is not "fused", so just return zero
  if (num_adds == 1) { return 0; }
  return num_adds;
}
void v_vk_quantize_q8_1(vk_backend_ctx* ctx, vk_context& subctx, vk_sub_buffer&& in,
                        vk_sub_buffer&& out, uint32_t ne, bool use_x4_blocks = false) {
  VK_LOG_DEBUG(
    "v_vk_quantize_q8_1(" << "buffer in size=" << in.buffer->size << ", buffer out size=" << out.buffer->size << ", "
    << ne << ")");

  vk_pipeline pipeline = use_x4_blocks
                           ? v_vk_get_quantize_pipeline(ctx, v_TYPE_Q8_1, true)
                           : v_vk_get_quantize_pipeline(ctx, v_TYPE_Q8_1, false);

  v_vk_dispatch_pipeline(ctx, subctx, pipeline, {in, out}, std::array<uint32_t, 1>{ne}, {ne, 1, 1});
  vk_sync_buffers(ctx, subctx);
}

void v_vk_mul_mat_q_f16(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                        const v_tensor* src1, v_tensor* dst, bool disable_split_k, bool dryrun) {
  VK_LOG_DEBUG(
    "v_vk_mul_mat_q_f16((" << src0 << ", name=" << src0->name << ", type=" << v_type_name(src0->type) << ", ne0="
    << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0
    ->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << v_type_name(src1->type) << ", ne0=" <<
    src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb
    [0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << v_type_name(dst->type) << ", ne0=" << dst->
    ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] <<
    ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << (dryrun ? "dryrun" : "") << ")");
  V_ASSERT(v_vk_dim01_contiguous(src0) || src0->type == v_TYPE_F32 || src0->type == v_TYPE_F16 || src0->type == v_TYPE_BF16); // NOLINT
  V_ASSERT(v_vk_dim01_contiguous(src1) || src1->type == v_TYPE_F32 || src1->type == v_TYPE_F16); // NOLINT

  const uint64_t ne00 = src0->ne[0];
  const uint64_t ne01 = src0->ne[1];
  const uint64_t ne02 = src0->ne[2];
  const uint64_t ne03 = src0->ne[3];

  const uint64_t ne10 = src1->ne[0];
  const uint64_t ne11 = src1->ne[1];
  const uint64_t ne12 = src1->ne[2];
  const uint64_t ne13 = src1->ne[3];

  const uint64_t ne21           = dst->ne[1];
  const uint32_t stride_d       = dst->nb[1] / v_type_size(dst->type);
  const uint32_t stride_batch_d = stride_d * ne21;

  const uint64_t r2 = ne12 / ne02;
  const uint64_t r3 = ne13 / ne03;

  v_backend_vk_buffer_ctx* dst_buf_ctx  = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  v_backend_vk_buffer_ctx* src0_buf_ctx = (v_backend_vk_buffer_ctx*)src0->buffer->context;
  v_backend_vk_buffer_ctx* src1_buf_ctx = (v_backend_vk_buffer_ctx*)src1->buffer->context;

  vk_buffer d_Qx       = nullptr;
  size_t qx_buf_offset = 0;
  vk_buffer d_Qy       = nullptr;
  size_t qy_buf_offset = 0;

  bool src0_uma = false;
  bool src1_uma = false;

  if (ctx->device->uma) {
    vk_get_host_buffer(ctx->device, src0->data, d_Qx, qx_buf_offset);
    vk_get_host_buffer(ctx->device, src1->data, d_Qy, qy_buf_offset);
    src0_uma = d_Qx != nullptr;
    src1_uma = d_Qy != nullptr;
  }

  // Reformat and convert to fp16 if non-contiguous, or for coopmat2 for better perf
  const bool x_non_contig = (ctx->device->coopmat2 && src0->type == v_TYPE_F32) ||
    !v_vk_dim01_contiguous(src0);
  const bool y_non_contig = (ctx->device->coopmat2 && src1->type == v_TYPE_F32) ||
    (src0->type == v_TYPE_BF16 && src1->type != v_TYPE_BF16) ||
    !v_vk_dim01_contiguous(src1);

  // If src0 is BF16, try to use a BF16 x BF16 multiply
  v_data_type f16_type = src0->type == v_TYPE_BF16
                           ? v_TYPE_BF16
                           : v_TYPE_F16;

  const bool y_f32_kernel = src1->type == v_TYPE_F32 && !y_non_contig;

  bool quantize_y = ctx->device->integer_dot_product && src1->type == v_TYPE_F32 && v_is_contiguous(src1) && (ne11
      * ne10)
    % 4 == 0;

  // Check for mmq first
  vk_matmul_pipeline mmp = quantize_y
                             ? v_vk_get_mul_mat_mat_pipeline(ctx,
                                                             src0->type,
                                                             v_TYPE_Q8_1,
                                                             (v_prec)dst->op_params[0])
                             : nullptr;

  if (mmp == nullptr) {
    // Fall back to f16 dequant mul mat
    mmp = v_vk_get_mul_mat_mat_pipeline(ctx,
                                        src0->type,
                                        y_non_contig
                                          ? f16_type
                                          : src1->type,
                                        (v_prec)dst->op_params[0]);
    quantize_y = false;
  }

  const bool qx_needs_dequant = mmp == nullptr || x_non_contig;
  const bool qy_needs_dequant = !quantize_y && ((src1->type != f16_type && !y_f32_kernel) || y_non_contig);

  if (qx_needs_dequant) {
    // Fall back to dequant + f16 mulmat
    mmp = v_vk_get_mul_mat_mat_pipeline(ctx,
                                        f16_type,
                                        y_f32_kernel
                                          ? v_TYPE_F32
                                          : f16_type,
                                        (v_prec)dst->op_params[0]);
  }

  // Not implemented
  V_ASSERT(y_non_contig || !qy_needs_dequant); // NOLINT

  const uint32_t kpad = quantize_y
                          ? 0
                          : vk_align_size(ne10,
                                          v_vk_guess_matmul_pipeline_align(
                                            ctx,
                                            mmp,
                                            ne01,
                                            ne11,
                                            qx_needs_dequant
                                              ? f16_type
                                              : src0->type,
                                            quantize_y
                                              ? v_TYPE_Q8_1
                                              : (y_f32_kernel
                                                   ? v_TYPE_F32
                                                   : src1->type)));
  const bool aligned = !quantize_y && ne10 == kpad && ne01 > 8 && ne11 > 8;

  vk_pipeline pipeline = v_vk_guess_matmul_pipeline(ctx,
                                                    mmp,
                                                    ne01,
                                                    ne11,
                                                    aligned,
                                                    qx_needs_dequant
                                                      ? f16_type
                                                      : src0->type,
                                                    quantize_y
                                                      ? v_TYPE_Q8_1
                                                      : (y_f32_kernel
                                                           ? v_TYPE_F32
                                                           : src1->type));

  // Reserve extra storage in the N dimension for the Y matrix, so we can avoid bounds-checking
  uint32_t padded_n = qy_needs_dequant
                        ? ROUNDUP_POW2(ne11, pipeline->wg_denoms[1])
                        : ne11;
  const int x_ne = ne01 * ne00;
  const int y_ne = padded_n * ne10;
  const int d_ne = ne11 * ne01;

  const uint32_t split_k = v_vk_guess_split_k(ctx, ne01, ne11, ne10, disable_split_k, pipeline);

  const uint64_t qx_sz = v_type_size(src0->type) * x_ne / block_size(src0->type);
  const uint64_t qy_sz = v_type_size(src1->type) * y_ne / block_size(src1->type);
  const uint64_t x_sz  = !qx_needs_dequant
                           ? qx_sz
                           : sizeof(v_fp16_t) * x_ne;
  const uint64_t y_sz = quantize_y
                          ? (y_ne * v_type_size(v_TYPE_Q8_1) / block_size(v_TYPE_Q8_1))
                          : (y_f32_kernel
                               ? sizeof(float) * y_ne
                               : sizeof(v_fp16_t) * y_ne);
  const uint64_t d_sz = sizeof(float) * d_ne;

  vk_pipeline to_fp16_vk_0 = nullptr;
  vk_pipeline to_fp16_vk_1 = nullptr;
  vk_pipeline to_q8_1      = nullptr;

  if (x_non_contig) { to_fp16_vk_0 = v_vk_get_cpy_pipeline(ctx, src0, nullptr, f16_type); }
  else { to_fp16_vk_0 = v_vk_get_to_fp16(ctx, src0->type); }
  if (y_non_contig) { to_fp16_vk_1 = v_vk_get_cpy_pipeline(ctx, src1, nullptr, f16_type); }
  else { to_fp16_vk_1 = v_vk_get_to_fp16(ctx, src1->type); }
  V_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr); // NOLINT
  V_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr); // NOLINT

  if (quantize_y) { to_q8_1 = v_vk_get_quantize_pipeline(ctx, v_TYPE_Q8_1, true); }

  if (dryrun) {
    const uint64_t x_sz_upd = x_sz * ne02 * ne03;
    uint64_t y_sz_upd       = y_sz * ne12 * ne13;
    if (quantize_y) { y_sz_upd = CEIL_DIV(y_sz_upd, 144) * 144; }
    const uint64_t split_k_size = split_k > 1
                                    ? d_sz * ne12 * ne13 * split_k
                                    : 0;
    if (
      (qx_needs_dequant && x_sz_upd > ctx->device->properties.limits.maxStorageBufferRange) ||
      (qy_needs_dequant && y_sz_upd > ctx->device->properties.limits.maxStorageBufferRange) ||
      (split_k > 1 && split_k_size > ctx->device->properties.limits.maxStorageBufferRange)) { v_ABORT("Requested preallocation size is too large"); }
    if (qx_needs_dequant && ctx->prealloc_size_x < x_sz_upd) { ctx->prealloc_size_x = x_sz_upd; }
    if ((qy_needs_dequant || quantize_y) && ctx->prealloc_size_y < y_sz_upd) { ctx->prealloc_size_y = y_sz_upd; }
    if (split_k > 1 && ctx->prealloc_size_split_k < split_k_size) { ctx->prealloc_size_split_k = split_k_size; }

    // Request descriptor sets
    v_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    if (qx_needs_dequant) { v_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1); }
    if (qy_needs_dequant) { v_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1); }
    if (quantize_y) { v_pipeline_request_descriptor_sets(ctx, to_q8_1, 1); }
    if (split_k > 1) { v_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_split_k_reduce, 1); }
    return;
  }

  vk_buffer d_D               = dst_buf_ctx->dev_buffer;
  const uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
  V_ASSERT(d_D != nullptr);
  V_ASSERT(d_D->size >= d_buf_offset + d_sz * ne02 * ne03);
  vk_buffer d_X;
  uint64_t x_buf_offset = 0;
  vk_buffer d_Y;
  uint64_t y_buf_offset = 0;
  if (!src0_uma) {
    d_Qx          = src0_buf_ctx->dev_buffer;
    qx_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
    V_ASSERT(d_Qx != nullptr);
  }
  if (!src1_uma) {
    d_Qy          = src1_buf_ctx->dev_buffer;
    qy_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
    V_ASSERT(d_Qy != nullptr);
  }
  if (qx_needs_dequant) {
    d_X = ctx->prealloc_x;
    V_ASSERT(d_X->size >= x_sz * ne02 * ne03);
  }
  else {
    d_X          = d_Qx;
    x_buf_offset = qx_buf_offset;
    V_ASSERT(qx_sz == x_sz);
  }
  if (qy_needs_dequant) {
    d_Y = ctx->prealloc_y;
    V_ASSERT(d_Y->size >= y_sz * ne12 * ne13);
  }
  else if (quantize_y) {
    d_Y = ctx->prealloc_y;
    V_ASSERT(d_Y->size >= CEIL_DIV(y_sz * ne12 * ne13, 144) * 144);
  }
  else {
    d_Y          = d_Qy;
    y_buf_offset = qy_buf_offset;
    V_ASSERT(qy_sz == y_sz);
  }

  if (x_non_contig || qx_needs_dequant) { if (ctx->prealloc_x_need_sync) { vk_sync_buffers(ctx, subctx); } }

  if (x_non_contig) {
    v_vk_cpy_to_contiguous(ctx,
                           subctx,
                           to_fp16_vk_0,
                           src0,
                           v_vk_subbuffer(ctx, d_Qx, qx_buf_offset),
                           v_vk_subbuffer(ctx, d_X, 0));
  }
  else if (qx_needs_dequant) {
    const std::vector<uint32_t> pc = {
      (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(nelements(src0))
    };
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           to_fp16_vk_0,
                           {
                             vk_sub_buffer{d_Qx, qx_buf_offset, qx_sz * ne02 * ne03},
                             vk_sub_buffer{d_X, 0, x_sz * ne02 * ne03}
                           },
                           pc,
                           {(uint32_t)(x_ne * ne02 * ne03), 1, 1});
    vk_sync_buffers(ctx, subctx);
  }
  if (y_non_contig) {
    if (ctx->prealloc_y_last_pipeline_used != to_fp16_vk_1.get() ||
      ctx->prealloc_y_last_tensor_used != src1) {
      if (ctx->prealloc_y_need_sync) { vk_sync_buffers(ctx, subctx); }
      v_vk_cpy_to_contiguous(ctx,
                             subctx,
                             to_fp16_vk_1,
                             src1,
                             v_vk_subbuffer(ctx, d_Qy, qy_buf_offset),
                             v_vk_subbuffer(ctx, d_Y, 0));
      ctx->prealloc_y_last_pipeline_used = to_fp16_vk_1.get();
      ctx->prealloc_y_last_tensor_used   = src1;
    }
  }
  if (quantize_y) {
    if (ctx->prealloc_y_last_pipeline_used != to_q8_1.get() ||
      ctx->prealloc_y_last_tensor_used != src1) {
      if (ctx->prealloc_y_need_sync) { vk_sync_buffers(ctx, subctx); }
      v_vk_quantize_q8_1(ctx,
                         subctx,
                         v_vk_subbuffer(ctx, d_Qy, qy_buf_offset),
                         v_vk_subbuffer(ctx, d_Y, 0),
                         y_ne * ne12 * ne13,
                         true);
      ctx->prealloc_y_last_pipeline_used = to_q8_1.get();
      ctx->prealloc_y_last_tensor_used   = src1;
    }
  }

  uint32_t stride_batch_x = ne00 * ne01;
  uint32_t stride_batch_y = ne10 * ne11;

  if (!v_vk_dim01_contiguous(src0) && !qx_needs_dequant) { stride_batch_x = src0->nb[0] / v_type_size(src0->type); }

  if (!v_vk_dim01_contiguous(src1) && !qy_needs_dequant && !quantize_y) { stride_batch_y = src1->nb[0] / v_type_size(src1->type); }

  uint32_t y_sz_total = y_sz * ne12 * ne13;
  if (quantize_y) { y_sz_total = CEIL_DIV(y_sz_total, 144) * 144; }

  // compute
  v_vk_matmul(
    ctx,
    subctx,
    pipeline,
    {d_X, x_buf_offset, x_sz * ne02 * ne03},
    {d_Y, y_buf_offset, y_sz_total},
    v_vk_subbuffer(ctx, d_D, d_buf_offset),
    {ctx->prealloc_split_k, 0, d_sz * ne12 * ne13 * split_k},
    ne01,
    ne11,
    ne10,
    ne10,
    ne10,
    stride_d,
    stride_batch_x,
    stride_batch_y,
    stride_batch_d,
    split_k,
    ne12 * ne13,
    ne02,
    ne12,
    r2,
    r3,
    padded_n
  ); // NOLINT

  if (x_non_contig || qx_needs_dequant) { ctx->prealloc_x_need_sync = true; }
  if (y_non_contig || quantize_y) { ctx->prealloc_y_need_sync = true; }
}

void v_vk_matmul_id(
  vk_backend_ctx* ctx, vk_context& subctx, vk_pipeline& pipeline,
  vk_sub_buffer&& a, vk_sub_buffer&& b, vk_sub_buffer&& d, vk_sub_buffer&& ids,
  uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
  uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
  uint32_t n_as, uint32_t nei0, uint32_t nei1, uint32_t nbi1, uint32_t ne11,
  uint32_t padded_n) {
  VK_LOG_DEBUG(
    "v_vk_matmul_id(a: (" << a.buffer->buffer << ", " << a.offset << ", " << a.size << "), b: (" << b.buffer->buffer
    << ", " << b.offset << ", " << b.size << "), d: (" << d.buffer->buffer << ", " << d.offset << ", " << d.size <<
    "), ids: (" << ids.buffer->buffer << ", " << ids.offset << ", " << ids.size << "), " <<
    "m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b <<
    ", stride_d: " << stride_d << ", " <<
    "batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " <<
    batch_stride_d << ", " <<
    "n_as: " << n_as << ", nei0: " << nei0 << ", nei1: " << nei1 << ", nbi1: " << nbi1 << ", ne11: " << ne11 << ")");
  const vk_mat_mat_id_push_constants pc = {
    m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d,
    nei0, nei1, nbi1, ne11, padded_n
  };
  v_vk_dispatch_pipeline(ctx, subctx, pipeline, {a, b, d, ids}, pc, {m, nei1, n_as});
}

void v_vk_mul_mat_vec_q_f16(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                            const v_tensor* src1, v_tensor* dst, bool dryrun) {
  VK_LOG_DEBUG(
    "v_vk_mul_mat_vec_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[
      0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] <<
    ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] <<
    ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1="
    << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1="
    << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1
    ] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << (dryrun ? "dryrun" : "") << "),)");
  V_ASSERT(v_vk_dim01_contiguous(src0) || src0->type == v_TYPE_F32 || src0->type == v_TYPE_F16 || src0->type == v_TYPE_BF16); // NOLINT
  V_ASSERT(v_vk_dim01_contiguous(src1) || src1->type == v_TYPE_F32 || src1->type == v_TYPE_F16); // NOLINT

  const uint64_t ne00 = src0->ne[0];
  const uint64_t ne01 = src0->ne[1];
  const uint64_t ne02 = src0->ne[2];
  const uint64_t ne03 = src0->ne[3];

  const uint64_t ne10 = src1->ne[0];
  const uint64_t ne11 = src1->ne[1];
  const uint64_t ne12 = src1->ne[2];
  const uint64_t ne13 = src1->ne[3];

  const uint64_t ne20 = dst->ne[0];
  const uint64_t ne21 = dst->ne[1];
  const uint64_t ne22 = dst->ne[2];
  const uint64_t ne23 = dst->ne[3];

  const uint64_t r2 = ne12 / ne02;
  const uint64_t r3 = ne13 / ne03;

  // batch_n indicates that we need to compute a few vector results, and this assumes
  // ne12 and ne13 are 1. It overloads the batch_strides to hold the row strides.
  V_ASSERT(ne11 == 1 || ne12 * ne13 == 1);
  bool batch_n = ne11 > 1;

  v_backend_vk_buffer_ctx* dst_buf_ctx  = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  v_backend_vk_buffer_ctx* src0_buf_ctx = (v_backend_vk_buffer_ctx*)src0->buffer->context;
  v_backend_vk_buffer_ctx* src1_buf_ctx = (v_backend_vk_buffer_ctx*)src1->buffer->context;

  vk_buffer d_Qx       = nullptr;
  size_t qx_buf_offset = 0;
  vk_buffer d_Qy       = nullptr;
  size_t qy_buf_offset = 0;

  bool src0_uma = false;
  bool src1_uma = false;

  if (ctx->device->uma) {
    vk_get_host_buffer(ctx->device, src0->data, d_Qx, qx_buf_offset);
    vk_get_host_buffer(ctx->device, src1->data, d_Qy, qy_buf_offset);
    src0_uma = d_Qx != nullptr;
    src1_uma = d_Qy != nullptr;
  }

  const bool x_non_contig = !v_vk_dim01_contiguous(src0);
  const bool y_non_contig = !v_vk_dim01_contiguous(src1);

  const bool f16_f32_kernel = src1->type == v_TYPE_F32;
  bool quantize_y           = ctx->device->integer_dot_product && src1->type == v_TYPE_F32 && v_is_contiguous(src1) && (ne11
      * ne10)
    % 4 == 0 && v_vk_should_use_mmvq(ctx->device, ne01, ne11, ne10, src0->type);

  vk_pipeline to_fp16_vk_0 = nullptr;
  vk_pipeline to_fp16_vk_1 = nullptr;
  if (x_non_contig) { to_fp16_vk_0 = v_vk_get_cpy_pipeline(ctx, src0, nullptr, src0->type); }
  if (y_non_contig) { to_fp16_vk_1 = v_vk_get_cpy_pipeline(ctx, src1, nullptr, src1->type); }
  else { to_fp16_vk_1 = v_vk_get_to_fp16(ctx, src1->type); }

  // Check for mmq first
  vk_pipeline dmmv = quantize_y
                       ? v_vk_get_dequantize_mul_mat_vec(ctx, src0->type, v_TYPE_Q8_1, ne11, ne20, ne00)
                       : nullptr;
  vk_pipeline to_q8_1 = nullptr;

  if (dmmv == nullptr) {
    // Fall back to f16 dequant mul mat
    dmmv       = v_vk_get_dequantize_mul_mat_vec(ctx, src0->type, src1->type, ne11, ne20, ne00);
    quantize_y = false;
  }

  if (quantize_y) { to_q8_1 = v_vk_get_quantize_pipeline(ctx, v_TYPE_Q8_1, true); }

  const bool qx_needs_dequant = x_non_contig;
  const bool qy_needs_dequant = !quantize_y && ((src1->type != v_TYPE_F16 && !f16_f32_kernel) || y_non_contig);

  // Not implemented
  V_ASSERT(y_non_contig || !qy_needs_dequant); // NOLINT

  V_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr); // NOLINT
  V_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr); // NOLINT
  V_ASSERT(dmmv != nullptr);

  const uint64_t x_ne = ne01 * ne00;
  const uint64_t y_ne = ne11 * ne10;
  const uint64_t d_ne = ne11 * ne01;

  const uint64_t qx_sz = vk_align_size(v_type_size(src0->type) * x_ne / block_size(src0->type),
                                       ctx->device->properties.limits.minStorageBufferOffsetAlignment);
  const uint64_t qy_sz = v_type_size(src1->type) * y_ne / block_size(src1->type);
  const uint64_t x_sz  = x_non_contig
                           ? vk_align_size(v_type_size(src0->type) * x_ne,
                                           ctx->device->properties.limits.minStorageBufferOffsetAlignment)
                           : qx_sz;
  const uint64_t y_sz = quantize_y
                          ? (y_ne * v_type_size(v_TYPE_Q8_1) / block_size(v_TYPE_Q8_1))
                          : (f16_f32_kernel
                               ? sizeof(float) * y_ne
                               : sizeof(v_fp16_t) * y_ne);
  const uint64_t d_sz = sizeof(float) * d_ne;

  if (dryrun) {
    const uint64_t x_sz_upd = x_sz * ne02 * ne03;
    uint64_t y_sz_upd       = y_sz * ne12 * ne13;
    if (quantize_y) { y_sz_upd = CEIL_DIV(y_sz_upd, 144) * 144; }
    if (
      (qx_needs_dequant && x_sz_upd > ctx->device->properties.limits.maxStorageBufferRange) ||
      (qy_needs_dequant && y_sz_upd > ctx->device->properties.limits.maxStorageBufferRange)) { v_ABORT("Requested preallocation size is too large"); }
    if (qx_needs_dequant && ctx->prealloc_size_x < x_sz_upd) { ctx->prealloc_size_x = x_sz_upd; }
    if ((qy_needs_dequant || quantize_y) && ctx->prealloc_size_y < y_sz_upd) { ctx->prealloc_size_y = y_sz_upd; }

    // Request descriptor sets
    if (qx_needs_dequant) { v_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1); }
    if (qy_needs_dequant) { v_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1); }
    if (quantize_y) { v_pipeline_request_descriptor_sets(ctx, to_q8_1, 1); }
    v_pipeline_request_descriptor_sets(ctx, dmmv, 1);
    return;
  }

  vk_buffer d_D               = dst_buf_ctx->dev_buffer;
  const uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
  V_ASSERT(d_D != nullptr);
  vk_buffer d_X;
  uint64_t x_buf_offset = 0;
  vk_buffer d_Y;
  uint64_t y_buf_offset = 0;
  if (!src0_uma) {
    d_Qx          = src0_buf_ctx->dev_buffer;
    qx_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
    V_ASSERT(d_Qx != nullptr);
  }
  if (!src1_uma) {
    d_Qy          = src1_buf_ctx->dev_buffer;
    qy_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
    V_ASSERT(d_Qy != nullptr);
  }
  if (qx_needs_dequant) { d_X = ctx->prealloc_x; }
  else {
    d_X          = d_Qx;
    x_buf_offset = qx_buf_offset;
    V_ASSERT(qx_sz == x_sz);
  }
  if (qy_needs_dequant) { d_Y = ctx->prealloc_y; }
  else if (quantize_y) {
    d_Y = ctx->prealloc_y;
    V_ASSERT(d_Y->size >= CEIL_DIV(y_sz * ne12 * ne13, 144) * 144);
  }
  else {
    d_Y          = d_Qy;
    y_buf_offset = qy_buf_offset;
    V_ASSERT(qy_sz == y_sz);
  }

  if (x_non_contig) {
    if (ctx->prealloc_x_need_sync) { vk_sync_buffers(ctx, subctx); }

    V_ASSERT(
      x_sz == vk_align_size(v_type_size(src0->type) * x_ne, ctx->device->properties.limits.
        minStorageBufferOffsetAlignment));
    v_vk_cpy_to_contiguous(ctx,
                           subctx,
                           to_fp16_vk_0,
                           src0,
                           v_vk_subbuffer(ctx, d_Qx, qx_buf_offset),
                           v_vk_subbuffer(ctx, d_X, 0));
  }
  if (y_non_contig) {
    V_ASSERT(y_sz == v_type_size(src1->type) * y_ne);
    if (ctx->prealloc_y_last_pipeline_used != to_fp16_vk_1.get() ||
      ctx->prealloc_y_last_tensor_used != src1) {
      if (ctx->prealloc_y_need_sync) { vk_sync_buffers(ctx, subctx); }
      v_vk_cpy_to_contiguous(ctx,
                             subctx,
                             to_fp16_vk_1,
                             src1,
                             v_vk_subbuffer(ctx, d_Qy, qy_buf_offset),
                             v_vk_subbuffer(ctx, d_Y, 0));
      ctx->prealloc_y_last_pipeline_used = to_fp16_vk_1.get();
      ctx->prealloc_y_last_tensor_used   = src1;
    }
  }
  if (quantize_y) {
    if (ctx->prealloc_y_last_pipeline_used != to_q8_1.get() ||
      ctx->prealloc_y_last_tensor_used != src1) {
      if (ctx->prealloc_y_need_sync) { vk_sync_buffers(ctx, subctx); }
      v_vk_quantize_q8_1(ctx,
                         subctx,
                         v_vk_subbuffer(ctx, d_Qy, qy_buf_offset),
                         v_vk_subbuffer(ctx, d_Y, 0),
                         y_ne * ne12 * ne13,
                         true);
      ctx->prealloc_y_last_pipeline_used = to_q8_1.get();
      ctx->prealloc_y_last_tensor_used   = src1;
    }
  }

  // For batch_n, the A matrix is the same for each batch, and B/D use the row stride as the batch stride
  uint32_t stride_batch_x = batch_n
                              ? 0
                              : ne00 * ne01;
  uint32_t stride_batch_y = batch_n
                              ? ne10
                              : (ne10 * ne11);
  uint32_t stride_batch_d = batch_n
                              ? ne20
                              : (ne20 * ne21);

  if (!v_vk_dim01_contiguous(src0) && !qx_needs_dequant) { stride_batch_x = src0->nb[0] / v_type_size(src0->type); }

  if (!v_vk_dim01_contiguous(src1) && !qy_needs_dequant) { stride_batch_y = src1->nb[0] / v_type_size(src1->type); }

  const uint32_t max_groups_x = ctx->device->properties.limits.maxComputeWorkGroupCount[0];

  uint32_t groups_x = ne01;
  uint32_t groups_z = 1;

  if (ne01 > max_groups_x) {
    groups_z = 64;
    groups_x = CEIL_DIV(groups_x, groups_z);
  }

  // TODO: Clean up this whole sz * ne_2 * ne_3 thing, it hasn't been necessary for a long time
  uint32_t y_sz_total = y_sz * ne12 * ne13;
  if (quantize_y) { y_sz_total = CEIL_DIV(y_sz_total, 144) * 144; }

  // compute
  const vk_mat_vec_push_constants pc = {
    (uint32_t)ne00, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne01,
    stride_batch_x, stride_batch_y, stride_batch_d,
    (uint32_t)ne02, (uint32_t)ne12, (uint32_t)r2, (uint32_t)r3,
  };
  v_vk_dispatch_pipeline(ctx,
                         subctx,
                         dmmv,
                         {
                           vk_sub_buffer{d_X, x_buf_offset, x_sz * ne02 * ne03},
                           vk_sub_buffer{d_Y, y_buf_offset, y_sz_total},
                           vk_sub_buffer{d_D, d_buf_offset, d_sz * ne22 * ne23}
                         },
                         pc,
                         {groups_x, (uint32_t)(ne12 * ne13), groups_z});

  if (x_non_contig) { ctx->prealloc_x_need_sync = true; }
  if (y_non_contig || quantize_y) { ctx->prealloc_y_need_sync = true; }
}

void v_vk_mul_mat_vec_p021_f16_f32(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                                   v_tensor* const src1, v_tensor* dst, bool dryrun) {
  VK_LOG_DEBUG(
    "v_vk_mul_mat_p021_f16_f32(" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->
    ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] <<
    ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] <<
    ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1="
    << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1="
    << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1
    ] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << (dryrun ? "dryrun" : "") << ")");
  V_ASSERT(v_is_permuted(src0) && v_is_permuted(src1));
  V_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // NOLINT
  V_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // NOLINT
  V_ASSERT(src0->type == v_TYPE_F16);
  V_ASSERT(src1->type == v_TYPE_F32);

  const uint64_t ne00 = src0->ne[0];
  const uint64_t ne01 = src0->ne[1];
  const uint64_t ne02 = src0->ne[2];
  // const uint64_t ne03 = src0->ne[3];

  const uint64_t ne10 = src1->ne[0];
  const uint64_t ne11 = src1->ne[1];
  const uint64_t ne12 = src1->ne[2];
  // const uint64_t ne13 = src1->ne[3];

  V_ASSERT(ne11 == 1);

  v_backend_vk_buffer_ctx* dst_buf_ctx  = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  v_backend_vk_buffer_ctx* src0_buf_ctx = (v_backend_vk_buffer_ctx*)src0->buffer->context;
  v_backend_vk_buffer_ctx* src1_buf_ctx = (v_backend_vk_buffer_ctx*)src1->buffer->context;

  vk_buffer d_Qy       = nullptr;
  size_t qy_buf_offset = 0;

  bool src1_uma = false;

  if (ctx->device->uma) {
    vk_get_host_buffer(ctx->device, src1->data, d_Qy, qy_buf_offset);
    src1_uma = d_Qy != nullptr;
  }

  const uint64_t x_ne = ne00 * ne01 * ne02;
  const uint64_t y_ne = ne10 * ne11 * ne12;
  const uint64_t d_ne = ne01 * ne11 * ne12;

  const uint64_t qx_sz = vk_align_size(v_type_size(src0->type) * x_ne / block_size(src0->type),
                                       ctx->device->properties.limits.minStorageBufferOffsetAlignment);
  const uint64_t qy_sz = v_type_size(src1->type) * y_ne / block_size(src1->type);
  const uint64_t d_sz  = sizeof(float) * d_ne;

  // With grouped query attention there are > 1 Q matrices per K, V matrix.
  uint32_t gqa_ratio = (uint32_t)ne12 / (uint32_t)ne02;
  if (gqa_ratio > 8 || gqa_ratio == 0 || ne12 != ne02 * gqa_ratio) { gqa_ratio = 1; }

  if (dryrun) {
    // Request descriptor sets
    v_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_mul_mat_vec_p021_f16_f32[gqa_ratio - 1], 1);
    return;
  }

  vk_buffer d_D               = dst_buf_ctx->dev_buffer;
  const uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
  V_ASSERT(d_D != nullptr);
  vk_buffer d_Qx               = src0_buf_ctx->dev_buffer;
  const uint64_t qx_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
  V_ASSERT(d_Qx != nullptr);
  if (!src1_uma) {
    d_Qy          = src1_buf_ctx->dev_buffer;
    qy_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
    V_ASSERT(d_Qx != nullptr);
  }

  const uint64_t qy_buffer_offset = (qy_buf_offset / ctx->device->properties.limits.minStorageBufferOffsetAlignment) *
    ctx->device->properties.limits.minStorageBufferOffsetAlignment;
  const uint64_t qy_shader_offset = qy_buf_offset - qy_buffer_offset;

  const uint64_t d_buffer_offset = (d_buf_offset / ctx->device->properties.limits.minStorageBufferOffsetAlignment) * ctx
                                                                                                                     ->
                                                                                                                     device
                                                                                                                     ->
                                                                                                                     properties
                                                                                                                     .limits
                                                                                                                     .minStorageBufferOffsetAlignment;
  const uint64_t d_shader_offset = d_buf_offset - d_buffer_offset;

  // compute
  const std::array<uint32_t, 6> pc = {
    (uint32_t)ne00, (uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne12,
    (uint32_t)(qy_shader_offset / v_type_size(src1->type)), (uint32_t)(d_shader_offset / v_type_size(dst->type))
  };

  uint32_t workgroups_z = (uint32_t)ne12;
  // When gqa_ratio > 1, each invocation does multiple rows and we can launch fewer workgroups
  if (gqa_ratio > 1) { workgroups_z /= gqa_ratio; }

  v_vk_dispatch_pipeline(ctx,
                         subctx,
                         ctx->device->pipeline_mul_mat_vec_p021_f16_f32[gqa_ratio - 1],
                         {
                           vk_sub_buffer{d_Qx, qx_buf_offset, qx_sz},
                           vk_sub_buffer{d_Qy, qy_buffer_offset, qy_sz + qy_shader_offset},
                           vk_sub_buffer{d_D, d_buffer_offset, d_sz + d_shader_offset}
                         },
                         pc,
                         {1, (uint32_t)ne01, workgroups_z});
}

void v_vk_mul_mat_vec_nc_f16_f32(vk_backend_ctx* ctx, vk_context& subctx, v_tensor* const src0,
                                 v_tensor* const src1, v_tensor* dst, bool dryrun) {
  VK_LOG_DEBUG(
    "v_vk_mul_mat_nc_f16_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne
    [0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] <<
    ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] <<
    ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1="
    << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1="
    << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1
    ] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << (dryrun ? "dryrun" : "") << ")");
  V_ASSERT(!v_is_transposed(src0));
  V_ASSERT(!v_is_transposed(src1));
  V_ASSERT(!v_is_permuted(src0));
  V_ASSERT(src0->type == v_TYPE_F16);
  V_ASSERT(src1->type == v_TYPE_F32);

  const uint64_t ne00 = src0->ne[0];
  const uint64_t ne01 = src0->ne[1];
  const uint64_t ne02 = src0->ne[2];
  const uint64_t ne03 = src0->ne[3];

  const uint64_t nb01 = src0->nb[1];
  const uint64_t nb02 = src0->nb[2];

  const uint64_t nb12 = src1->nb[2];

  // const uint64_t ne10 = src1->ne[0];
  const uint64_t ne11 = src1->ne[1];
  const uint64_t ne12 = src1->ne[2];
  // const uint64_t ne13 = src1->ne[3];

  const uint32_t nb03 = (uint32_t)(src0->nb[3] / sizeof(v_fp16_t));
  const uint32_t nb13 = (uint32_t)(src1->nb[3] / sizeof(float));
  const uint32_t nb23 = (uint32_t)(dst->nb[3] / sizeof(float));

  V_ASSERT(ne11 == 1);
  V_ASSERT(src0->ne[3] == src1->ne[3]); // checked in supports_op

  v_backend_vk_buffer_ctx* dst_buf_ctx  = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  v_backend_vk_buffer_ctx* src0_buf_ctx = (v_backend_vk_buffer_ctx*)src0->buffer->context;
  v_backend_vk_buffer_ctx* src1_buf_ctx = (v_backend_vk_buffer_ctx*)src1->buffer->context;

  vk_buffer d_Qy       = nullptr;
  size_t qy_buf_offset = 0;

  bool src1_uma = false;

  if (ctx->device->uma) {
    vk_get_host_buffer(ctx->device, src1->data, d_Qy, qy_buf_offset);
    src1_uma = d_Qy != nullptr;
  }

  const uint64_t d_ne = ne01 * ne11 * ne12 * ne03;

  const uint32_t row_stride_x     = nb01 / sizeof(v_fp16_t);
  const uint32_t channel_stride_x = nb02 / sizeof(v_fp16_t);
  const uint32_t channel_stride_y = nb12 / sizeof(float);

  const uint64_t qx_sz = num_bytes(src0);
  const uint64_t qy_sz = num_bytes(src1);
  const uint64_t d_sz  = sizeof(float) * d_ne;

  if (dryrun) {
    // Request descriptor sets
    v_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_mul_mat_vec_nc_f16_f32, 1);
    return;
  }

  vk_buffer d_D               = dst_buf_ctx->dev_buffer;
  const uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
  V_ASSERT(d_D != nullptr);
  vk_buffer d_Qx               = src0_buf_ctx->dev_buffer;
  const uint64_t qx_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
  V_ASSERT(d_Qx != nullptr);
  if (!src1_uma) {
    d_Qy          = src1_buf_ctx->dev_buffer;
    qy_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
    V_ASSERT(d_Qx != nullptr);
  }

  const uint64_t qy_buffer_offset = (qy_buf_offset / ctx->device->properties.limits.minStorageBufferOffsetAlignment) *
    ctx->device->properties.limits.minStorageBufferOffsetAlignment;
  const uint64_t qy_shader_offset = qy_buf_offset - qy_buffer_offset;

  const uint64_t d_buffer_offset = (d_buf_offset / ctx->device->properties.limits.minStorageBufferOffsetAlignment) * ctx
                                                                                                                     ->
                                                                                                                     device
                                                                                                                     ->
                                                                                                                     properties
                                                                                                                     .limits
                                                                                                                     .minStorageBufferOffsetAlignment;
  const uint64_t d_shader_offset = d_buf_offset - d_buffer_offset;

  // compute
  const std::array<uint32_t, 12> pc = {
    (uint32_t)ne00, (uint32_t)ne01, row_stride_x, channel_stride_x, channel_stride_y, (uint32_t)(ne12 / ne02),
    (uint32_t)ne12, (uint32_t)(qy_shader_offset / v_type_size(src1->type)),
    (uint32_t)(d_shader_offset / v_type_size(dst->type)), nb03, nb13, nb23
  };
  v_vk_dispatch_pipeline(ctx,
                         subctx,
                         ctx->device->pipeline_mul_mat_vec_nc_f16_f32,
                         {
                           vk_sub_buffer{d_Qx, qx_buf_offset, qx_sz},
                           vk_sub_buffer{d_Qy, qy_buffer_offset, qy_sz + qy_shader_offset},
                           vk_sub_buffer{d_D, d_buffer_offset, d_sz + d_shader_offset}
                         },
                         pc,
                         {(uint32_t)ne03, (uint32_t)ne01, (uint32_t)ne12});
}

void v_vk_mul_mat(vk_backend_ctx* ctx, vk_context& subctx, v_tensor* src0, v_tensor* src1,
                  v_tensor* dst, bool dryrun) {
  VK_LOG_DEBUG("v_vk_mul_mat(" << src0 << ", " << src1 << ", " << dst << ")");

  // Handle huge A matrix by splitting the M dimensions. This works well for convolution use cases
  // where the M dimension is very large.
  // Split_k doesn't work with M splitting.
  const size_t nbytes    = num_bytes(src0);
  const bool needs_split = nbytes > ctx->device->properties.limits.maxStorageBufferRange;
  if (needs_split) {
    // Choose the number of rows that can fit (and divide by two, to allow for any additional offsets)
    const uint32_t M_split = ctx->device->properties.limits.maxStorageBufferRange / (2 * src0->nb[1]);
    uint32_t m_offset      = 0;
    while (m_offset < dst->ne[0]) {
      const uint32_t cur_M_size = std::min(M_split, (uint32_t)(dst->ne[0] - m_offset));
      v_tensor dst2             = *dst;
      v_tensor src02            = *src0;

      dst2.view_src = dst->view_src
                        ? dst->view_src
                        : dst;
      src02.view_src = src0->view_src
                         ? src0->view_src
                         : src0;

      dst2.view_offs += m_offset * dst->nb[0];
      src02.view_offs += m_offset * src0->nb[1];
      dst2.ne[0]  = cur_M_size;
      src02.ne[1] = cur_M_size;

      v_vk_mul_mat_q_f16(ctx, subctx, &src02, src1, &dst2, true, dryrun);

      m_offset += cur_M_size;
    }
  }
  else if (src0->type == v_TYPE_F16 && v_is_permuted(src0) && v_is_permuted(src1) && dst->ne[1] == 1 &&
    // detect 0213 permutation, and batch size of 1
    src0->nb[0] <= src0->nb[2] &&
    src0->nb[2] <= src0->nb[1] &&
    src0->nb[1] <= src0->nb[3] &&
    src1->nb[0] <= src1->nb[2] &&
    src1->nb[2] <= src1->nb[1] &&
    src1->nb[1] <= src1->nb[3] &&
    src0->ne[3] == 1 &&
    src1->ne[3] == 1) { v_vk_mul_mat_vec_p021_f16_f32(ctx, subctx, src0, src1, dst, dryrun); }
  else if (src0->type == v_TYPE_F16 && !v_is_contiguous(src0) && !v_is_transposed(src1) && dst->ne[1] == 1 &&
    !v_is_permuted(src0) && !v_is_permuted(src1)) {
    v_vk_mul_mat_vec_nc_f16_f32(ctx, subctx, src0, src1, dst, dryrun);
    // mul_mat_vec supports batching ne12*ne13 when ne11==1, or treating ne11 as the batch size (up to four)
    // when ne12 and ne13 are one.
  }
  else if ((dst->ne[1] == 1 || (dst->ne[1] <= mul_mat_vec_max_cols && src1->ne[2] * src1->ne[3] == 1)) &&
    (src0->type == v_TYPE_F32 || src0->type == v_TYPE_F16 || src0->type == v_TYPE_BF16 ||
      v_is_quantized(src0->type))) { v_vk_mul_mat_vec_q_f16(ctx, subctx, src0, src1, dst, dryrun); }
  else { v_vk_mul_mat_q_f16(ctx, subctx, src0, src1, dst, false, dryrun); }
}

void v_vk_mul_mat_id_q_f16(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                           const v_tensor* src1, const v_tensor* ids, v_tensor* dst,
                           bool dryrun) {
  VK_LOG_DEBUG(
    "v_vk_mul_mat_id_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0
    ] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] <<
    ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] <<
    ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1="
    << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << ids << ", name=" << ids->name << ", type=" << ids->type << ", ne0=" << ids->ne[0] << ", ne1="
    << ids->ne[1] << ", ne2=" << ids->ne[2] << ", ne3=" << ids->ne[3] << ", nb0=" << ids->nb[0] << ", nb1=" << ids->nb[1
    ] << ", nb2=" << ids->nb[2] << ", nb3=" << ids->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1="
    << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1
    ] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3] << "),)");
  V_ASSERT(v_vk_dim01_contiguous(src1) || src1->type == v_TYPE_F32 || src1->type == v_TYPE_F16); // NOLINT
  V_ASSERT(ids->type == v_TYPE_I32);

  const uint64_t ne00 = src0->ne[0];
  const uint64_t ne01 = src0->ne[1];
  const uint64_t ne02 = src0->ne[2];
  const uint64_t ne03 = src0->ne[3];

  const uint64_t ne10 = src1->ne[0];
  const uint64_t ne11 = src1->ne[1];
  const uint64_t ne12 = src1->ne[2];
  const uint64_t ne13 = src1->ne[3];

  const uint64_t nei0 = ids->ne[0];
  const uint64_t nei1 = ids->ne[1];

  const uint32_t nbi1 = ids->nb[1];
  const uint32_t nbi2 = ids->nb[2];

  const uint64_t ne20 = dst->ne[0];
  const uint64_t ne21 = dst->ne[1];
  const uint64_t ne22 = dst->ne[2];
  const uint64_t ne23 = dst->ne[3];

  const uint64_t n_as = ne02;

  v_backend_vk_buffer_ctx* dst_buf_ctx  = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  v_backend_vk_buffer_ctx* src0_buf_ctx = (v_backend_vk_buffer_ctx*)src0->buffer->context;
  v_backend_vk_buffer_ctx* src1_buf_ctx = (v_backend_vk_buffer_ctx*)src1->buffer->context;
  v_backend_vk_buffer_ctx* ids_buf_ctx  = (v_backend_vk_buffer_ctx*)ids->buffer->context;

  vk_buffer d_Qx        = nullptr;
  size_t qx_buf_offset  = 0;
  vk_buffer d_Qy        = nullptr;
  size_t qy_buf_offset  = 0;
  vk_buffer d_ids       = nullptr;
  size_t ids_buf_offset = 0;

  bool src0_uma = false;
  bool src1_uma = false;
  bool ids_uma  = false;

  if (ctx->device->uma) {
    vk_get_host_buffer(ctx->device, src0->data, d_Qx, qx_buf_offset);
    vk_get_host_buffer(ctx->device, src1->data, d_Qy, qy_buf_offset);
    vk_get_host_buffer(ctx->device, ids->data, d_ids, ids_buf_offset);
    src0_uma = d_Qx != nullptr;
    src1_uma = d_Qy != nullptr;
    ids_uma  = d_ids != nullptr;
  }

  // Reformat and convert to fp16 if non-contiguous, or for coopmat2 for better perf
  const bool x_non_contig = (ctx->device->coopmat2 && src0->type == v_TYPE_F32) ||
    !v_vk_dim01_contiguous(src0);
  const bool y_non_contig = (ctx->device->coopmat2 && src1->type == v_TYPE_F32) ||
    (src0->type == v_TYPE_BF16 && src1->type != v_TYPE_BF16) ||
    !v_vk_dim01_contiguous(src1);

  // If src0 is BF16, try to use a BF16 x BF16 multiply
  v_data_type f16_type = src0->type == v_TYPE_BF16
                           ? v_TYPE_BF16
                           : v_TYPE_F16;

  const bool y_f32_kernel = src1->type == v_TYPE_F32 && !y_non_contig;

  vk_matmul_pipeline mmp = v_vk_get_mul_mat_mat_id_pipeline(ctx,
                                                            src0->type,
                                                            y_non_contig
                                                              ? f16_type
                                                              : src1->type,
                                                            (v_prec)dst->op_params[0]);

  const bool qx_needs_dequant = mmp == nullptr || x_non_contig;
  const bool qy_needs_dequant = (src1->type != f16_type && !y_f32_kernel) || y_non_contig;

  if (qx_needs_dequant) {
    // Fall back to dequant + f16 mulmat
    mmp = v_vk_get_mul_mat_mat_id_pipeline(ctx,
                                           f16_type,
                                           y_f32_kernel
                                             ? v_TYPE_F32
                                             : f16_type,
                                           (v_prec)dst->op_params[0]);
  }

  // Not implemented
  V_ASSERT(y_non_contig || !qy_needs_dequant); // NOLINT

  const uint32_t kpad = vk_align_size(ne10,
                                      v_vk_guess_matmul_id_pipeline_align(ctx, mmp, ne01, nei1, qx_needs_dequant
                                                                                                  ? f16_type
                                                                                                  : src0->type));
  const bool aligned = ne10 == kpad && ne01 > 8 && nei1 > 8;

  vk_pipeline pipeline = v_vk_guess_matmul_id_pipeline(ctx,
                                                       mmp,
                                                       ne01,
                                                       nei1,
                                                       aligned,
                                                       qx_needs_dequant
                                                         ? f16_type
                                                         : src0->type);

  // Reserve extra storage in the N dimension for the Y matrix, so we can avoid bounds-checking
  uint32_t padded_n = qy_needs_dequant
                        ? ROUNDUP_POW2(ne11, pipeline->wg_denoms[1])
                        : ne11;
  const uint64_t x_ne = ne01 * ne00;
  const uint64_t y_ne = padded_n * ne10;
  const uint64_t d_ne = ne21 * ne20;

  const uint64_t qx_sz = v_type_size(src0->type) * x_ne / block_size(src0->type);
  const uint64_t qy_sz = v_type_size(src1->type) * y_ne / block_size(src1->type);
  const uint64_t x_sz  = !qx_needs_dequant
                           ? qx_sz
                           : sizeof(v_fp16_t) * x_ne;
  const uint64_t y_sz = y_f32_kernel
                          ? sizeof(float) * y_ne
                          : sizeof(v_fp16_t) * y_ne;
  const uint64_t ids_sz = nbi2;
  const uint64_t d_sz   = sizeof(float) * d_ne;

  vk_pipeline to_fp16_vk_0 = nullptr;
  vk_pipeline to_fp16_vk_1 = nullptr;

  if (x_non_contig) { to_fp16_vk_0 = v_vk_get_cpy_pipeline(ctx, src0, nullptr, f16_type); }
  else { to_fp16_vk_0 = v_vk_get_to_fp16(ctx, src0->type); }
  if (y_non_contig) { to_fp16_vk_1 = v_vk_get_cpy_pipeline(ctx, src1, nullptr, f16_type); }
  else { to_fp16_vk_1 = v_vk_get_to_fp16(ctx, src1->type); }
  V_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr); // NOLINT
  V_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr); // NOLINT

  if (dryrun) {
    const uint64_t x_sz_upd = x_sz * ne02 * ne03;
    const uint64_t y_sz_upd = y_sz * ne12 * ne13;
    if (
      (qx_needs_dequant && x_sz_upd > ctx->device->properties.limits.maxStorageBufferRange) ||
      (qy_needs_dequant && y_sz_upd > ctx->device->properties.limits.maxStorageBufferRange)) { v_ABORT("Requested preallocation size is too large"); }
    if (qx_needs_dequant && ctx->prealloc_size_x < x_sz_upd) { ctx->prealloc_size_x = x_sz_upd; }
    if (qy_needs_dequant && ctx->prealloc_size_y < y_sz_upd) { ctx->prealloc_size_y = y_sz_upd; }

    // Request descriptor sets
    v_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    if (qx_needs_dequant) { v_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1); }
    if (qy_needs_dequant) { v_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1); }
    return;
  }

  vk_buffer d_D               = dst_buf_ctx->dev_buffer;
  const uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
  V_ASSERT(d_D != nullptr);
  vk_buffer d_X;
  uint64_t x_buf_offset = 0;
  vk_buffer d_Y;
  uint64_t y_buf_offset = 0;
  if (!src0_uma) {
    d_Qx          = src0_buf_ctx->dev_buffer;
    qx_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
    V_ASSERT(d_Qx != nullptr);
  }
  if (!src1_uma) {
    d_Qy          = src1_buf_ctx->dev_buffer;
    qy_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
    V_ASSERT(d_Qy != nullptr);
  }
  if (!ids_uma) {
    d_ids          = ids_buf_ctx->dev_buffer;
    ids_buf_offset = vk_tensor_offset(ids) + ids->view_offs;
    V_ASSERT(d_ids != nullptr);
  }
  if (qx_needs_dequant) {
    d_X = ctx->prealloc_x;
    V_ASSERT(d_X->size >= x_sz * ne02 * ne03);
  }
  else {
    d_X          = d_Qx;
    x_buf_offset = qx_buf_offset;
    V_ASSERT(qx_sz == x_sz);
  }
  if (qy_needs_dequant) {
    d_Y = ctx->prealloc_y;
    V_ASSERT(d_Y->size >= y_sz * ne12 * ne13);
  }
  else {
    d_Y          = d_Qy;
    y_buf_offset = qy_buf_offset;
    V_ASSERT(qy_sz == y_sz);
  }

  if (x_non_contig || qx_needs_dequant) { if (ctx->prealloc_x_need_sync) { vk_sync_buffers(ctx, subctx); } }

  if (x_non_contig) {
    v_vk_cpy_to_contiguous(ctx,
                           subctx,
                           to_fp16_vk_0,
                           src0,
                           v_vk_subbuffer(ctx, d_Qx, qx_buf_offset),
                           v_vk_subbuffer(ctx, d_X, 0));
  }
  else if (qx_needs_dequant) {
    const std::vector<uint32_t> pc = {
      (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(nelements(src0))
    };
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           to_fp16_vk_0,
                           {
                             vk_sub_buffer{d_Qx, qx_buf_offset, qx_sz * ne02 * ne03},
                             vk_sub_buffer{d_X, 0, x_sz * ne02 * ne03}
                           },
                           pc,
                           {(uint32_t)(x_ne * ne02 * ne03), 1, 1});
    vk_sync_buffers(ctx, subctx);
  }
  if (y_non_contig) {
    if (ctx->prealloc_y_last_pipeline_used != to_fp16_vk_1.get() ||
      ctx->prealloc_y_last_tensor_used != src1) {
      if (ctx->prealloc_y_need_sync) { vk_sync_buffers(ctx, subctx); }
      v_vk_cpy_to_contiguous(ctx,
                             subctx,
                             to_fp16_vk_1,
                             src1,
                             v_vk_subbuffer(ctx, d_Qy, qy_buf_offset),
                             v_vk_subbuffer(ctx, d_Y, 0));
      ctx->prealloc_y_last_pipeline_used = to_fp16_vk_1.get();
      ctx->prealloc_y_last_tensor_used   = src1;
    }
  }

  uint32_t stride_batch_x = ne00 * ne01;
  uint32_t stride_batch_y = ne10 * ne11;

  if (!v_vk_dim01_contiguous(src0) && !qx_needs_dequant) { stride_batch_x = src0->nb[0] / v_type_size(src0->type); }

  if (!v_vk_dim01_contiguous(src1) && !qy_needs_dequant) { stride_batch_y = src1->nb[0] / v_type_size(src1->type); }

  // compute
  v_vk_matmul_id(
    ctx,
    subctx,
    pipeline,
    {d_X, x_buf_offset, x_sz * ne02 * ne03},
    {d_Y, y_buf_offset, y_sz * ne12 * ne13},
    {d_D, d_buf_offset, d_sz * ne22 * ne23},
    {d_ids, ids_buf_offset, ids_sz},
    ne01,
    ne21,
    ne10,
    ne10,
    ne10,
    ne01,
    stride_batch_x,
    stride_batch_y,
    ne20 * ne21,
    n_as,
    nei0,
    nei1,
    nbi1 / v_type_size(ids->type),
    ne11,
    padded_n
  ); // NOLINT

  if (x_non_contig || qx_needs_dequant) { ctx->prealloc_x_need_sync = true; }
  if (y_non_contig) { ctx->prealloc_y_need_sync = true; }
}

void v_vk_mul_mat_vec_id_q_f16(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                               const v_tensor* src1, const v_tensor* ids, v_tensor* dst,
                               bool dryrun) {
  VK_LOG_DEBUG(
    "v_vk_mul_mat_vec_id_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->
    ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] <<
    ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] <<
    ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1="
    << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << ids << ", name=" << ids->name << ", type=" << ids->type << ", ne0=" << ids->ne[0] << ", ne1="
    << ids->ne[1] << ", ne2=" << ids->ne[2] << ", ne3=" << ids->ne[3] << ", nb0=" << ids->nb[0] << ", nb1=" << ids->nb[1
    ] << ", nb2=" << ids->nb[2] << ", nb3=" << ids->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1="
    << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1
    ] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << (dryrun ? "dryrun" : "") << ")");
  V_ASSERT(v_vk_dim01_contiguous(src0) || src0->type == v_TYPE_F32 || src0->type == v_TYPE_F16 || src0->type == v_TYPE_BF16); // NOLINT
  V_ASSERT(v_vk_dim01_contiguous(src1) || src1->type == v_TYPE_F32 || src1->type == v_TYPE_F16); // NOLINT
  V_ASSERT(ids->type == v_TYPE_I32);

  const uint64_t ne00 = src0->ne[0];
  const uint64_t ne01 = src0->ne[1];
  const uint64_t ne02 = src0->ne[2];
  const uint64_t ne03 = src0->ne[3];

  const uint64_t ne10 = src1->ne[0];
  const uint64_t ne11 = src1->ne[1];
  const uint64_t ne12 = src1->ne[2];
  const uint64_t ne13 = src1->ne[3];

  const uint64_t nei0 = ids->ne[0];
  const uint64_t nei1 = ids->ne[1];

  const uint64_t nbi2 = ids->nb[2];

  V_ASSERT(nei1 == 1);

  const uint64_t ne20 = dst->ne[0];
  const uint64_t ne21 = dst->ne[1];
  const uint64_t ne22 = dst->ne[2];
  const uint64_t ne23 = dst->ne[3];

  v_backend_vk_buffer_ctx* dst_buf_ctx  = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  v_backend_vk_buffer_ctx* src0_buf_ctx = (v_backend_vk_buffer_ctx*)src0->buffer->context;
  v_backend_vk_buffer_ctx* src1_buf_ctx = (v_backend_vk_buffer_ctx*)src1->buffer->context;
  v_backend_vk_buffer_ctx* ids_buf_ctx  = (v_backend_vk_buffer_ctx*)ids->buffer->context;

  vk_buffer d_Qx        = nullptr;
  size_t qx_buf_offset  = 0;
  vk_buffer d_Qy        = nullptr;
  size_t qy_buf_offset  = 0;
  vk_buffer d_ids       = nullptr;
  size_t ids_buf_offset = 0;

  bool src0_uma = false;
  bool src1_uma = false;
  bool ids_uma  = false;

  if (ctx->device->uma) {
    vk_get_host_buffer(ctx->device, src0->data, d_Qx, qx_buf_offset);
    vk_get_host_buffer(ctx->device, src1->data, d_Qy, qy_buf_offset);
    vk_get_host_buffer(ctx->device, ids->data, d_ids, ids_buf_offset);
    src0_uma = d_Qx != nullptr;
    src1_uma = d_Qy != nullptr;
    ids_uma  = d_ids != nullptr;
  }

  const bool x_non_contig = !v_vk_dim01_contiguous(src0);
  const bool y_non_contig = !v_vk_dim01_contiguous(src1);

  const bool f16_f32_kernel = src1->type == v_TYPE_F32;

  const bool qx_needs_dequant = x_non_contig;
  const bool qy_needs_dequant = (src1->type != v_TYPE_F16 && !f16_f32_kernel) || y_non_contig;

  // Not implemented
  V_ASSERT(y_non_contig || !qy_needs_dequant); // NOLINT

  const uint64_t x_ne = ne01 * ne00;
  const uint64_t y_ne = ne11 * ne10;
  const uint64_t d_ne = ne21 * ne20;

  const uint64_t qx_sz = vk_align_size(v_type_size(src0->type) * x_ne / block_size(src0->type),
                                       ctx->device->properties.limits.minStorageBufferOffsetAlignment);
  const uint64_t qy_sz = v_type_size(src1->type) * y_ne / block_size(src1->type);
  const uint64_t x_sz  = x_non_contig
                           ? vk_align_size(v_type_size(src0->type) * x_ne,
                                           ctx->device->properties.limits.minStorageBufferOffsetAlignment)
                           : qx_sz;
  const uint64_t y_sz = f16_f32_kernel
                          ? sizeof(float) * y_ne
                          : sizeof(v_fp16_t) * y_ne;
  const uint64_t ids_sz = nbi2;
  const uint64_t d_sz   = sizeof(float) * d_ne;

  vk_pipeline to_fp16_vk_0 = nullptr;
  vk_pipeline to_fp16_vk_1 = nullptr;
  if (x_non_contig) { to_fp16_vk_0 = v_vk_get_cpy_pipeline(ctx, src0, nullptr, src0->type); }
  if (y_non_contig) { to_fp16_vk_1 = v_vk_get_cpy_pipeline(ctx, src1, nullptr, src1->type); }
  else { to_fp16_vk_1 = v_vk_get_to_fp16(ctx, src1->type); }
  vk_pipeline dmmv = v_vk_get_dequantize_mul_mat_vec_id(ctx, src0->type, src1->type);
  V_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr); // NOLINT
  V_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr); // NOLINT
  V_ASSERT(dmmv != nullptr);

  if (dryrun) {
    const uint64_t x_sz_upd = x_sz * ne02 * ne03;
    const uint64_t y_sz_upd = y_sz * ne12 * ne13;
    if (
      (qx_needs_dequant && x_sz_upd > ctx->device->properties.limits.maxStorageBufferRange) ||
      (qy_needs_dequant && y_sz_upd > ctx->device->properties.limits.maxStorageBufferRange)) { v_ABORT("Requested preallocation size is too large"); }
    if (qx_needs_dequant && ctx->prealloc_size_x < x_sz_upd) { ctx->prealloc_size_x = x_sz_upd; }
    if (qy_needs_dequant && ctx->prealloc_size_y < y_sz_upd) { ctx->prealloc_size_y = y_sz_upd; }

    // Request descriptor sets
    if (qx_needs_dequant) { v_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1); }
    if (qy_needs_dequant) { v_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1); }
    v_pipeline_request_descriptor_sets(ctx, dmmv, 1);
    return;
  }

  vk_buffer d_D               = dst_buf_ctx->dev_buffer;
  const uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
  V_ASSERT(d_D != nullptr);
  vk_buffer d_X;
  uint64_t x_buf_offset = 0;
  vk_buffer d_Y;
  uint64_t y_buf_offset = 0;
  if (!src0_uma) {
    d_Qx          = src0_buf_ctx->dev_buffer;
    qx_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
    V_ASSERT(d_Qx != nullptr);
  }
  if (!src1_uma) {
    d_Qy          = src1_buf_ctx->dev_buffer;
    qy_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
    V_ASSERT(d_Qy != nullptr);
  }
  if (!ids_uma) {
    d_ids          = ids_buf_ctx->dev_buffer;
    ids_buf_offset = vk_tensor_offset(ids) + ids->view_offs;
    V_ASSERT(d_ids != nullptr);
  }
  if (qx_needs_dequant) { d_X = ctx->prealloc_x; }
  else {
    d_X          = d_Qx;
    x_buf_offset = qx_buf_offset;
    V_ASSERT(qx_sz == x_sz);
  }
  if (qy_needs_dequant) { d_Y = ctx->prealloc_y; }
  else {
    d_Y          = d_Qy;
    y_buf_offset = qy_buf_offset;
    V_ASSERT(qy_sz == y_sz);
  }

  if (x_non_contig) { if (ctx->prealloc_x_need_sync) { vk_sync_buffers(ctx, subctx); } }

  if (x_non_contig) {
    V_ASSERT(
      x_sz == vk_align_size(v_type_size(src0->type) * x_ne, ctx->device->properties.limits.
        minStorageBufferOffsetAlignment));
    v_vk_cpy_to_contiguous(ctx,
                           subctx,
                           to_fp16_vk_0,
                           src0,
                           v_vk_subbuffer(ctx, d_Qx, qx_buf_offset),
                           v_vk_subbuffer(ctx, d_X, 0));
  }
  if (y_non_contig) {
    V_ASSERT(y_sz == v_type_size(src1->type) * y_ne);
    if (ctx->prealloc_y_last_pipeline_used != to_fp16_vk_1.get() ||
      ctx->prealloc_y_last_tensor_used != src1) {
      if (ctx->prealloc_y_need_sync) { vk_sync_buffers(ctx, subctx); }
      v_vk_cpy_to_contiguous(ctx,
                             subctx,
                             to_fp16_vk_1,
                             src1,
                             v_vk_subbuffer(ctx, d_Qy, qy_buf_offset),
                             v_vk_subbuffer(ctx, d_Y, 0));
      ctx->prealloc_y_last_pipeline_used = to_fp16_vk_1.get();
      ctx->prealloc_y_last_tensor_used   = src1;
    }
  }

  uint32_t stride_batch_y = ne10 * ne11;

  if (!v_vk_dim01_contiguous(src1) && !qy_needs_dequant) { stride_batch_y = src1->nb[0] / v_type_size(src1->type); }

  const uint32_t max_groups_x = ctx->device->properties.limits.maxComputeWorkGroupCount[0];

  uint32_t groups_x = ne01;
  uint32_t groups_z = 1;

  if (ne01 > max_groups_x) {
    groups_z = 64;
    groups_x = CEIL_DIV(groups_x, groups_z);
  }

  // compute
  const vk_mat_vec_id_push_constants pc = {
    (uint32_t)ne00, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne01,
    (uint32_t)x_ne, stride_batch_y, (uint32_t)(ne20 * ne21),
    (uint32_t)nei0, (uint32_t)ne11,
  };
  v_vk_dispatch_pipeline(ctx,
                         subctx,
                         dmmv,
                         {
                           vk_sub_buffer{d_X, x_buf_offset, x_sz * ne02 * ne03},
                           vk_sub_buffer{d_Y, y_buf_offset, y_sz * ne12 * ne13},
                           vk_sub_buffer{d_D, d_buf_offset, d_sz * ne22 * ne23},
                           vk_sub_buffer{d_ids, ids_buf_offset, ids_sz}
                         },
                         pc,
                         {groups_x, (uint32_t)nei0, groups_z});

  if (x_non_contig) { ctx->prealloc_x_need_sync = true; }
  if (y_non_contig) { ctx->prealloc_y_need_sync = true; }
}

void v_vk_mul_mat_id(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                     const v_tensor* src1, const v_tensor* src2, v_tensor* dst, bool dryrun) {
  VK_LOG_DEBUG("v_vk_mul_mat_id(" << src0 << ", " << src1 << ", " << src2 << ", " << dst << ")");
  if (src2->ne[1] == 1 && (src0->type == v_TYPE_F32 || src0->type == v_TYPE_F16 ||
    v_is_quantized(src0->type))) { v_vk_mul_mat_vec_id_q_f16(ctx, subctx, src0, src1, src2, dst, dryrun); }
  else { v_vk_mul_mat_id_q_f16(ctx, subctx, src0, src1, src2, dst, dryrun); }
}
