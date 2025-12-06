#ifndef MYPROJECT_VK_OP_F32_HPP
#define MYPROJECT_VK_OP_F32_HPP
#include "vk_context.h"
#include "vk_pipeline.h"
#include "v_vk.h"
#include "vk_common.h"
#include "vk_util.h"
#include "vk_vision_comp.hpp"
#include "vk_device.h"
#include "vk_constant.h"
template <typename T>
inline void v_vk_dispatch_pipeline(vk_backend_ctx* ctx, vk_context& subctx, vk_pipeline& pipeline,
                                   std::initializer_list<vk::DescriptorBufferInfo> const& descriptor_buffer_infos,
                                   const T& push_constants, std::array<uint32_t, 3> elements) {
  const uint32_t wg0 = CEIL_DIV(elements[0], pipeline->wg_denoms[0]);
  const uint32_t wg1 = CEIL_DIV(elements[1], pipeline->wg_denoms[1]);
  const uint32_t wg2 = CEIL_DIV(elements[2], pipeline->wg_denoms[2]);
  VK_LOG_DEBUG("v_vk_dispatch_pipeline(" << pipeline->name << ", {";
    for (auto& buffer : descriptor_buffer_infos) {
    std::cerr << "(" << buffer.buffer << ", " << buffer.offset << ", " << buffer.range << "), ";
    }
    std::cerr << "}, (" << wg0 << "," << wg1 << "," << wg2 << "))");
  V_ASSERT(ctx->descriptor_set_idx < ctx->descriptor_sets.size());
  V_ASSERT(descriptor_buffer_infos.size() <= MAX_PARAMETER_COUNT);
  V_ASSERT(pipeline->parameter_count == descriptor_buffer_infos.size());

  vk::DescriptorSet& descriptor_set = ctx->descriptor_sets[ctx->descriptor_set_idx++];
  vk::WriteDescriptorSet write_descriptor_set{
    descriptor_set, 0, 0, pipeline->parameter_count, vk::DescriptorType::eStorageBuffer, nullptr,
    descriptor_buffer_infos.begin()
  };
  ctx->device->device.updateDescriptorSets({write_descriptor_set}, {});

  subctx->s->buffer.pushConstants(pipeline->layout,
                                  vk::ShaderStageFlagBits::eCompute,
                                  0,
                                  push_constant_size(push_constants),
                                  push_constant_data(push_constants));
  subctx->s->buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
  subctx->s->buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                       pipeline->layout,
                                       0,
                                       {descriptor_set},
                                       {});
  subctx->s->buffer.dispatch(wg0, wg1, wg2);
}

template <typename PC>
inline void v_vk_op_f32(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, const v_tensor* src1, const v_tensor* src2, v_tensor* dst, v_operation op, PC&& pc, bool dryrun ) {
  VK_LOG_DEBUG(
    "v_vk_op_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] <<
    ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1="
    << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    if (src1 != nullptr) {
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] <<
    ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1="
    << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    }
    if (src2 != nullptr) {
    std::cerr << "), (" << src2 << ", name=" << src2->name << ", type=" << src2->type << ", ne0=" << src2->ne[0] <<
    ", ne1=" << src2->ne[1] << ", ne2=" << src2->ne[2] << ", ne3=" << src2->ne[3] << ", nb0=" << src2->nb[0] << ", nb1="
    << src2->nb[1] << ", nb2=" << src2->nb[2] << ", nb3=" << src2->nb[3];
    }
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1="
    << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1
    ] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << op_name(op) << ", " << (dryrun ? "dryrun" : "") << ")");
  V_ASSERT(op == v_OP_GET_ROWS || op == v_OP_CPY || (!v_is_quantized(src0->type) && (src1 == nullptr || !v_is_quantized(src1->type)))); // NOLINT
  V_ASSERT(v_vk_op_supports_incontiguous(op) || v_vk_dim01_contiguous(src0)); // NOLINT
  V_ASSERT(dst->buffer != nullptr);
  const uint64_t ne00 = src0->ne[0];
  const uint64_t ne01 = src0->ne[1];
  const uint64_t ne02 = src0->ne[2];
  const uint64_t ne03 = src0->ne[3];
  const uint64_t ne0  = ne00 * ne01;

  const bool use_src1 = src1 != nullptr;
  const uint64_t ne10 = use_src1? src1->ne[0]: 0;
  const uint64_t ne11 = use_src1? src1->ne[1]: 0;
  const uint64_t ne12 = use_src1? src1->ne[2]: 0;
  const uint64_t ne13 = use_src1? src1->ne[3]: 0;
  const uint64_t ne1 = ne10 * ne11;
  // const uint64_t nb10 = use_src1 ? src1->nb[0] : 0;

  const bool use_src2 = src2 != nullptr;
  const uint64_t ne20 = use_src2? src2->ne[0]: 0;
  const uint64_t ne21 = use_src2? src2->ne[1]: 0;
  const uint64_t ne22 = use_src2? src2->ne[2]: 0;
  const uint64_t ne23 = use_src2? src2->ne[3]: 0;
  const uint64_t ne2 = ne20 * ne21;

  const uint64_t ned0 = dst->ne[0];
  const uint64_t ned1 = dst->ne[1];
  const uint64_t ned2 = dst->ne[2];
  const uint64_t ned3 = dst->ne[3];
  const uint64_t ned  = ned0 * ned1;

  init_pushconst_fastdiv(pc);

  vk_pipeline pipeline = v_vk_op_get_pipeline(ctx, src0, src1, src2, dst, op);

  if (pipeline == nullptr) {
    std::cerr << "v_vulkan: Error: Missing op: " << v_op_name(op) << " for " << v_type_name(src0->type);
    if (src1 != nullptr) { std::cerr << " and " << v_type_name(src1->type); }
    std::cerr << " to " << v_type_name(dst->type) << std::endl;
    v_ABORT("fatal error");
  }

  if (dryrun) {
    v_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    return;
  }

  const bool op_supports_incontiguous = v_vk_op_supports_incontiguous(op);

  v_backend_vk_buffer_ctx* dst_buf_ctx  = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  v_backend_vk_buffer_ctx* src0_buf_ctx = (v_backend_vk_buffer_ctx*)src0->buffer->context;
  v_backend_vk_buffer_ctx* src1_buf_ctx = use_src1? (v_backend_vk_buffer_ctx*)src1->buffer->context: nullptr;
  v_backend_vk_buffer_ctx* src2_buf_ctx = use_src2
                                            ? (v_backend_vk_buffer_ctx*)src2->buffer->context
                                            : nullptr;

  vk_buffer d_X       = nullptr;
  size_t x_buf_offset = 0;
  vk_buffer d_Y       = nullptr;
  size_t y_buf_offset = 0;
  vk_buffer d_Z       = nullptr;
  size_t z_buf_offset = 0;

  bool src0_uma = false;
  bool src1_uma = false;
  bool src2_uma = false;

  if (ctx->device->uma) {
    vk_get_host_buffer(ctx->device, src0->data, d_X, x_buf_offset);
    src0_uma = d_X != nullptr;
    if (use_src1) {
      vk_get_host_buffer(ctx->device, src1->data, d_Y, y_buf_offset);
      src1_uma = d_Y != nullptr;
    }
    if (use_src2) {
      vk_get_host_buffer(ctx->device, src2->data, d_Z, z_buf_offset);
      src2_uma = d_Z != nullptr;
    }
  }

  vk_buffer d_D = dst_buf_ctx->dev_buffer;

  V_ASSERT(d_D != nullptr);
  uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
  if (!src0_uma) {
    d_X          = src0_buf_ctx->dev_buffer;
    x_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
    V_ASSERT(d_X != nullptr);
  }
  if (use_src1 && !src1_uma) {
    d_Y          = src1_buf_ctx->dev_buffer;
    y_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
    V_ASSERT(d_Y != nullptr);
  }
  if (use_src2 && !src2_uma) {
    d_Z          = src2_buf_ctx->dev_buffer;
    z_buf_offset = vk_tensor_offset(src2) + src2->view_offs;
    V_ASSERT(d_Z != nullptr);
  }
  // Compute misalignment offset for descriptors and store it in in push constants, then align the descriptor offsets.
  init_pushconst_tensor_offsets(ctx, pc, src0, src1, src2, dst);
  x_buf_offset &= ~(ctx->device->properties.limits.minStorageBufferOffsetAlignment - 1);
  y_buf_offset &= ~(ctx->device->properties.limits.minStorageBufferOffsetAlignment - 1);
  z_buf_offset &= ~(ctx->device->properties.limits.minStorageBufferOffsetAlignment - 1);
  d_buf_offset &= ~(ctx->device->properties.limits.minStorageBufferOffsetAlignment - 1);

  std::array<uint32_t, 3> elements;

  // Single call if dimension 2 is contiguous
  V_ASSERT(op_supports_incontiguous || (v_is_contiguous(src0) && (src1 == nullptr || v_is_contiguous(src1))));

  switch (op) {
    case v_OP_NORM:
    case v_OP_RMS_NORM_BACK:
    case v_OP_L2_NORM:
    case V_OP_SOFT_MAX:
    case v_OP_SOFT_MAX_BACK:
    case v_OP_SUM_ROWS:
    case V_OP_MEAN:
    case V_OP_ARGMAX: {
      const uint32_t nr = v_nrows(src0);
      if (nr > 262144) { elements = {512, 512, CEIL_DIV(nr, 262144)}; }
      else if (nr > 512) { elements = {512, CEIL_DIV(nr, 512), 1}; }
      else { elements = {nr, 1, 1}; }
    }
    break;
    case v_OP_RMS_NORM:
      if (ctx->do_add_rms_partials) {
        // Run one element per thread, 128 threads per workgroup
        elements = {(uint32_t)CEIL_DIV(ne00, 128), 1, 1};
      }
      else { elements = {(uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne03}; }
      break;

    case v_OP_SUM:
      // We use v_OP_SUM_ROWS with 1 row.
      elements = {1, 1, 1};
      break;
    case v_OP_GROUP_NORM: {
      const uint32_t num_groups = dst->op_params[0];
      elements                  = {num_groups * (uint32_t)src0->ne[3], 1, 1};
    }
    break;
    case V_OP_DIAG_MASK_INF:
    case V_OP_ROPE:
    case v_OP_ROPE_BACK:
      elements = {(uint32_t)v_nrows(src0), (uint32_t)ne00, 1};
      break;
    case v_OP_GET_ROWS:
      elements = {(uint32_t)ne00, (uint32_t)ne10, (uint32_t)(ne11 * ne12)};
      elements[1] = std::min(elements[1], ctx->device->properties.limits.maxComputeWorkGroupCount[1]);
      elements[2] = std::min(elements[2], ctx->device->properties.limits.maxComputeWorkGroupCount[2]);
      break;
    case v_OP_ARGSORT:
      elements = {(uint32_t)ne00, (uint32_t)v_nrows(src0), 1};
      break;
    case V_OP_IM2COL: {
      const bool is_2D = dst->op_params[6] == 1;

      const uint32_t IC = src1->ne[is_2D
                                     ? 2
                                     : 1];

      const uint32_t KH = is_2D
                            ? src0->ne[1]
                            : 1;
      const uint32_t KW = src0->ne[0];

      const uint32_t OH = is_2D
                            ? dst->ne[2]
                            : 1;
      const uint32_t OW = dst->ne[1];

      const uint32_t batch = src1->ne[is_2D
                                        ? 3
                                        : 2];

      elements = {OW * KW * KH, OH, batch * IC};
    }
    break;
    case v_OP_IM2COL_3D: {
      const uint32_t IC = ((const uint32_t*)(dst->op_params))[9];

      const uint32_t N = ne13 / IC;

      const uint32_t KD = ne02;
      const uint32_t KH = ne01;
      const uint32_t KW = ne00;

      const uint32_t OD = ned3 / N;
      const uint32_t OH = ned2;
      const uint32_t OW = ned1;

      const uint32_t IC_KD_KH_KW = IC * KD * KH * KW;
      const uint32_t N_OD_OH     = N * OD * OH;

      elements    = {IC_KD_KH_KW, OW, N_OD_OH};
      elements[2] = std::min(elements[2], ctx->device->properties.limits.maxComputeWorkGroupCount[2]);
    }
    break;
    case v_OP_TIMESTEP_EMBEDDING: {
      const uint32_t dim = dst->op_params[0];
      uint32_t half_ceil = (dim + 1) / 2;
      elements           = {half_ceil, (uint32_t)src0->ne[0], 1};
    }
    break;
    case v_OP_CONV_TRANSPOSE_1D: {
      elements = {uint32_t(src0->ne[1]), 1, 1}; // parallelize in {Cout, 1, 1}
    }
    break;
    case V_OP_POOL_2D: {
      const uint32_t N  = dst->ne[3];
      const uint32_t OC = dst->ne[2];
      const uint32_t OH = dst->ne[1];
      const uint32_t OW = dst->ne[0];
      elements          = {N * OC * OH * OW, 1, 1};
    }
    case V_OP_POOL_2D_BACK: {
      const uint32_t N  = dst->ne[3];
      const uint32_t OC = dst->ne[2];
      const uint32_t OH = dst->ne[1];
      const uint32_t OW = dst->ne[0];
      elements          = {N * OC * OH * OW, 1, 1};
    }
    break;
    case v_OP_CONV_2D: { elements = v_vk_get_conv_elements(dst); }
    break;
    case v_OP_CONV_TRANSPOSE_2D: { elements = v_vk_get_conv_transpose_2d_elements(dst); }
    break;
    case v_OP_ADD:
    case v_OP_SUB:
    case v_OP_DIV:
    case v_OP_MUL:
    case V_OP_SCALE:
    case v_OP_SQR:
    case v_OP_SQRT:
    case v_OP_SIN:
    case v_OP_COS:
    case v_OP_CLAMP:
    case v_OP_PAD:
    case v_OP_ROLL:
    case v_OP_REPEAT:
    case v_OP_REPEAT_BACK:
    case v_OP_CPY:
    case v_OP_CONCAT:
    case v_OP_UPSCALE:
    case v_OP_UNARY:
    case v_OP_GLU:
    case v_OP_CONV_2D_DW: {
      uint32_t ne = nelements(dst);
      if (op == v_OP_CPY && v_is_quantized(src0->type) && v_is_quantized(dst->type)) {
        // Convert from number of logical elements to 2- or 4-byte units.
        ne /= block_size(src0->type);
        if ((v_type_size(src0->type) % 4) == 0) { ne *= v_type_size(src0->type) / 4; }
        else { ne *= v_type_size(src0->type) / 2; }
      }
      // copy_to_quant has block size of 32, and each thread does QUANT_K elements.
      // Splitting into 512x512xZ wouldn't work well since each workgroup does 1024 elements.
      // So divide by block size here before splitting into 512x512 groups.
      if (op == v_OP_CPY && !v_is_quantized(src0->type) && v_is_quantized(dst->type)) { ne = CEIL_DIV(ne, block_size(dst->type)); }
      if (ne > 262144) { elements = {512, 512, CEIL_DIV(ne, 262144)}; }
      else if (ne > 512) { elements = {512, CEIL_DIV(ne, 512), 1}; }
      else { elements = {ne, 1, 1}; }
    }
    break;
    case v_OP_ADD_ID: { elements = {(uint32_t)ne01, (uint32_t)ne02, 1}; }
    break;
    case v_OP_SET_ROWS: {
      uint32_t ne = nelements(src0);
      if (v_is_quantized(dst->type)) {
        // quants run 32 threads each doing QUANT_K elements
        ne = CEIL_DIV(ne, 32 * block_size(dst->type));
      }
      else {
        // scalar types do one element per thread, running 512 threads
        ne = CEIL_DIV(ne, 512);
      }
      if (ne > 262144) { elements = {512, 512, CEIL_DIV(ne, 262144)}; }
      else if (ne > 512) { elements = {512, CEIL_DIV(ne, 512), 1}; }
      else { elements = {ne, 1, 1}; }
    }
    break;
    case v_OP_SSM_CONV: {
      const uint32_t nr  = src0->ne[1];
      const uint32_t n_t = dst->ne[1];
      const uint32_t n_s = dst->ne[2];
      elements           = {nr, n_t, n_s};
    }
    break;
    default:
      elements = {(uint32_t)nelements(src0), 1, 1};
      break;
  }

  uint64_t x_sz, y_sz, z_sz, d_sz;

  if (op_supports_incontiguous) {
    x_sz = num_bytes(src0) + get_misalign_bytes(ctx, src0);
    y_sz = use_src1
             ? num_bytes(src1) + get_misalign_bytes(ctx, src1)
             : 0;
    z_sz = use_src2
             ? num_bytes(src2) + get_misalign_bytes(ctx, src2)
             : 0;
    d_sz = num_bytes(dst) + get_misalign_bytes(ctx, dst);

    if (x_buf_offset + x_sz >= d_X->size) { x_sz = v_vk_get_max_buffer_range(ctx, d_X, x_buf_offset); }
    if (use_src1 && y_buf_offset + y_sz >= d_Y->size) { y_sz = v_vk_get_max_buffer_range(ctx, d_Y, y_buf_offset); }
    if (use_src2 && z_buf_offset + z_sz >= d_Z->size) { z_sz = v_vk_get_max_buffer_range(ctx, d_Z, z_buf_offset); }
    if (d_buf_offset + d_sz >= d_D->size) { d_sz = v_vk_get_max_buffer_range(ctx, d_D, d_buf_offset); }
  }
  else {
    x_sz = v_type_size(src0->type) / block_size(src0->type) * ne0 * ne02 * ne03;
    y_sz = use_src1
             ? v_type_size(src1->type) * ne1 * ne12 * ne13
             : 0;
    z_sz = use_src2
             ? v_type_size(src2->type) * ne2 * ne22 * ne23
             : 0;
    d_sz = v_type_size(dst->type) * ned * ned2 * ned3;
  }

  if (op == v_OP_ADD || op == v_OP_RMS_NORM) {
    vk_buffer d_A = ctx->do_add_rms_partials
                      ? ctx->prealloc_add_rms_partials
                      : d_X;
    size_t a_buf_offset = ctx->do_add_rms_partials
                            ? ctx->prealloc_size_add_rms_partials_offset
                            : 0;
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             vk_sub_buffer{d_X, x_buf_offset, x_sz},
                             vk_sub_buffer{d_Y, y_buf_offset, y_sz},
                             vk_sub_buffer{d_D, d_buf_offset, d_sz},
                             v_vk_subbuffer(ctx, d_A, a_buf_offset),
                           },
                           pc,
                           elements);
  }
  else if (op == v_OP_GLU) {
    // Empty src1 is possible in glu, but the shader needs a buffer
    vk_sub_buffer subbuf_y;
    if (use_src1) { subbuf_y = {d_Y, y_buf_offset, y_sz}; }
    else { subbuf_y = {d_X, 0, x_sz}; }

    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             vk_sub_buffer{d_X, x_buf_offset, x_sz}, subbuf_y, vk_sub_buffer{d_D, d_buf_offset, d_sz}
                           },
                           pc,
                           elements);
  }
  else if (op == V_OP_SOFT_MAX) {
    // Empty src1 and src2 is possible in soft_max, but the shader needs a buffer
    vk_sub_buffer subbuf_y;
    if (use_src1) { subbuf_y = {d_Y, y_buf_offset, y_sz}; }
    else { subbuf_y = {d_X, 0, x_sz}; }

    vk_sub_buffer subbuf_z;
    if (use_src2) { subbuf_z = {d_Z, z_buf_offset, z_sz}; }
    else { subbuf_z = {d_X, 0, x_sz}; }

    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             vk_sub_buffer{d_X, x_buf_offset, x_sz}, subbuf_y, subbuf_z,
                             vk_sub_buffer{d_D, d_buf_offset, d_sz}
                           },
                           pc,
                           elements);
  }
  else if (op == V_OP_ROPE || op == v_OP_ROPE_BACK) {
    // Empty src2 is possible in rope, but the shader needs a buffer
    vk_sub_buffer subbuf_z;
    if (use_src2) { subbuf_z = {d_Z, z_buf_offset, z_sz}; }
    else { subbuf_z = {d_X, 0, x_sz}; }

    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             vk_sub_buffer{d_X, x_buf_offset, x_sz}, vk_sub_buffer{d_Y, y_buf_offset, y_sz}, subbuf_z,
                             vk_sub_buffer{d_D, d_buf_offset, d_sz}
                           },
                           pc,
                           elements);
  }
  else if (op == V_OP_IM2COL || op == v_OP_IM2COL_3D) {
    if (ctx->device->shader_int64 && ctx->device->buffer_device_address) {
      // buffer device address path doesn't use dst buffer
      d_sz = 1;
    }
    // im2col uses only src1 and dst buffers
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {vk_sub_buffer{d_Y, y_buf_offset, y_sz}, vk_sub_buffer{d_D, d_buf_offset, d_sz}},
                           pc,
                           elements);
  }
  else if (op == v_OP_COUNT_EQUAL) {
    // count_equal assumes that destination buffer is initialized with zeroes
    vk_buffer_memset_async(subctx, d_D, d_buf_offset, 0, d_sz);
    vk_sync_buffers(ctx, subctx);
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             vk_sub_buffer{d_X, x_buf_offset, x_sz}, vk_sub_buffer{d_Y, y_buf_offset, y_sz},
                             vk_sub_buffer{d_D, d_buf_offset, d_sz}
                           },
                           pc,
                           elements);
  }
  else if (op == v_OP_OPT_STEP_SGD) {
    // OPT_STEP_SGD works on src0, it does not need dst
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             vk_sub_buffer{d_X, x_buf_offset, x_sz}, vk_sub_buffer{d_Y, y_buf_offset, y_sz},
                             vk_sub_buffer{d_Z, z_buf_offset, z_sz}
                           },
                           pc,
                           elements);
  }
  else if (use_src2) {
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             vk_sub_buffer{d_X, x_buf_offset, x_sz}, vk_sub_buffer{d_Y, y_buf_offset, y_sz},
                             vk_sub_buffer{d_Z, z_buf_offset, z_sz}, vk_sub_buffer{d_D, d_buf_offset, d_sz}
                           },
                           pc,
                           elements);
  }
  else if (use_src1) {
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             vk_sub_buffer{d_X, x_buf_offset, x_sz}, vk_sub_buffer{d_Y, y_buf_offset, y_sz},
                             vk_sub_buffer{d_D, d_buf_offset, d_sz}
                           },
                           pc,
                           elements);
  }
  else {
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {vk_sub_buffer{d_X, x_buf_offset, x_sz}, vk_sub_buffer{d_D, d_buf_offset, d_sz}},
                           pc,
                           elements);
  }
}



#endif //MYPROJECT_VK_OP_F32_HPP
