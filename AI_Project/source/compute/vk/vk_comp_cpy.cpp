#include <vk_op_f32.hpp>
#include "vk_comp.hpp"
#include "vk_constant.h"

void v_vk_cpy(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst, bool dryrun) {
  auto ne = static_cast<uint32_t>(nelements(src0));
  if (v_is_quantized(src0->type) && v_is_quantized(dst->type)) {
    // Convert from number of logical elements to 2- or 4-byte units.
    ne /= block_size(src0->type);
    if ((v_type_size(src0->type) % 4) == 0) { ne *= v_type_size(src0->type) / 4; }
    else { ne *= v_type_size(src0->type) / 2; }
  }
  vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst, ne);
  v_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, V_OP_CPY, std::move(p), dryrun);
}

void v_vk_cpy_to_contiguous(vk_backend_ctx* ctx, vk_context& subctx, vk_pipeline pipeline,
                            const v_tensor* tensor, vk_sub_buffer&& in, vk_sub_buffer&& out) {
  VK_LOG_DEBUG(
    "v_vk_cpy_to_contiguous((" << tensor << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" <<
    tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" <<
    tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << "), ";
    std::cerr << "buffer in size=" << in.buffer->size << ", buffer out size=" << out.buffer->size << ")");
  const int tensor_type_size = v_type_size(tensor->type);

  const uint32_t ne = nelements(tensor);
  std::array<uint32_t, 3> elements;

  if (ne > 262144) { elements = {512, 512, CEIL_DIV(ne, 262144)}; }
  else if (ne > 512) { elements = {512, CEIL_DIV(ne, 512), 1}; }
  else { elements = {ne, 1, 1}; }

  vk_op_unary_push_constants pc = {
    (uint32_t)ne,
    (uint32_t)tensor->ne[0], (uint32_t)tensor->ne[1], (uint32_t)tensor->ne[2], (uint32_t)tensor->ne[3],
    (uint32_t)tensor->nb[0] / tensor_type_size, (uint32_t)tensor->nb[1] / tensor_type_size,
    (uint32_t)tensor->nb[2] / tensor_type_size, (uint32_t)tensor->nb[3] / tensor_type_size,
    (uint32_t)tensor->ne[0], (uint32_t)tensor->ne[1], (uint32_t)tensor->ne[2], (uint32_t)tensor->ne[3], 1,
    (uint32_t)tensor->ne[0], (uint32_t)(tensor->ne[0] * tensor->ne[1]),
    (uint32_t)(tensor->ne[0] * tensor->ne[1] * tensor->ne[2]),
    0,
    0.0f, 0.0f,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  init_pushconst_fastdiv(pc);
  v_vk_dispatch_pipeline(ctx, subctx, pipeline, {in, out}, pc, elements);
  vk_sync_buffers(ctx, subctx);
}

vk_pipeline v_vk_get_cpy_pipeline(vk_backend_ctx* ctx, const v_tensor* src,
                                  const v_tensor* dst, v_data_type to) {
  // Choose "contiguous copy" shader if src/dst are contiguous
  bool contig = v_is_contiguous(src) && (!dst || v_is_contiguous(dst));

  if (src->type == v_TYPE_F32 && to == v_TYPE_F32) {
    if (contig) { return ctx->device->pipeline_contig_cpy_f32_f32; }
    else { return ctx->device->pipeline_cpy_f32_f32; }
  }
  if (src->type == v_TYPE_F32 && to == v_TYPE_F16) {
    if (contig) { return ctx->device->pipeline_contig_cpy_f32_f16; }
    else { return ctx->device->pipeline_cpy_f32_f16; }
  }
  if (src->type == v_TYPE_F16 && to == v_TYPE_F16) {
    if (contig) { return ctx->device->pipeline_contig_cpy_f16_f16; }
    else { return ctx->device->pipeline_cpy_f16_f16; }
  }
  if (src->type == v_TYPE_F16 && to == v_TYPE_F32) {
    if (contig) { return ctx->device->pipeline_contig_cpy_f16_f32; }
    else { return ctx->device->pipeline_cpy_f16_f32; }
  }
  if (src->type == v_TYPE_F32 && to == v_TYPE_BF16) {
    if (contig) { return ctx->device->pipeline_contig_cpy_f32_bf16; }
    else { return ctx->device->pipeline_cpy_f32_bf16; }
  }
  if (src->type == v_TYPE_F32 && to == v_TYPE_I32) {
    if (contig) { return ctx->device->pipeline_contig_cpy_f32_i32; }
    else { return ctx->device->pipeline_cpy_f32_i32; }
  }
  if (src->type == v_TYPE_I32 && to == v_TYPE_F32) {
    if (contig) { return ctx->device->pipeline_contig_cpy_i32_f32; }
    else { return ctx->device->pipeline_cpy_i32_f32; }
  }
  if (src->type == v_TYPE_F32) {
    switch (to) {
      case v_TYPE_Q4_0:
      case v_TYPE_Q4_1:
      case v_TYPE_Q5_0:
      case v_TYPE_Q5_1:
      case v_TYPE_Q8_0:
      case v_TYPE_IQ4_NL:
        return ctx->device->pipeline_cpy_f32_quant[to];
      default:
        break;
    }
  }

  if (to == v_TYPE_F32) {
    switch (src->type) {
      case v_TYPE_Q4_0:
      case v_TYPE_Q4_1:
      case v_TYPE_Q5_0:
      case v_TYPE_Q5_1:
      case v_TYPE_Q8_0:
      case v_TYPE_IQ4_NL:
        return ctx->device->pipeline_cpy_quant_f32[src->type];
      default:
        break;
    }
  }

  if (src->type == to) {
    // Copy two or four bytes at a time, depending on block size.
    // For quantized types, we scale by block size/type size. But
    // this path is also used_bits__ for bf16->bf16 for example, where the
    // type size must be exactly 2 or 4.
    V_ASSERT(v_is_quantized(to) || v_type_size(src->type) == 2 || v_type_size(src->type) == 4);
    if ((v_type_size(src->type) % 4) == 0) {
      if (contig) { return ctx->device->pipeline_contig_cpy_f32_f32; }
      else { return ctx->device->pipeline_cpy_f32_f32; }
    }
    else {
      if (contig) { return ctx->device->pipeline_contig_cpy_f16_f16; }
      else { return ctx->device->pipeline_cpy_f16_f16; }
    }
  }

  std::cerr << "Missing CPY op for types: " << v_type_name(src->type) << " " << v_type_name(to) << std::endl;
  v_ABORT("fatal error");
}
