#include "vk_pipeline.hpp"
#include "vk_device.hpp"
#include "vk_context.h"
#include "vk_vision_comp.hpp"
#include "v_util.hpp"

void v_pipeline_request_descriptor_sets(vk_backend_ctx* ctx, vk_pipeline& pipeline, uint32_t n) {
  VK_LOG_DEBUG("v_pipeline_request_descriptor_sets(" << pipeline->name << ", " << n << ")");
  ctx->pipeline_descriptor_set_requirements += n;
  if (!pipeline->compiled) {
    pipeline->needed           = true;
    ctx->device->need_compiles = true;
  }
}
void vk_pipeline_allocate_descriptor_sets(vk_backend_ctx* ctx) {
  if (ctx->descriptor_sets.size() >= ctx->pipeline_descriptor_set_requirements) {
    // Enough descriptors are available
    return;
  }

  vk_device& device = ctx->device;

  uint32_t to_alloc       = ctx->pipeline_descriptor_set_requirements - ctx->descriptor_sets.size();
  uint32_t pool_remaining = VK_DEVICE_DESCRIPTOR_POOL_SIZE - ctx->descriptor_sets.size() %
    VK_DEVICE_DESCRIPTOR_POOL_SIZE;
  uint32_t pool_idx = ctx->descriptor_sets.size() / VK_DEVICE_DESCRIPTOR_POOL_SIZE;

  while (to_alloc > 0) {
    const uint32_t alloc_count = std::min(pool_remaining, to_alloc);
    to_alloc -= alloc_count;
    pool_remaining = VK_DEVICE_DESCRIPTOR_POOL_SIZE;

    if (pool_idx >= ctx->descriptor_pools.size()) {
      vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer,
                                                  MAX_PARAMETER_COUNT * VK_DEVICE_DESCRIPTOR_POOL_SIZE);
      vk::DescriptorPoolCreateInfo
        descriptor_pool_create_info({}, VK_DEVICE_DESCRIPTOR_POOL_SIZE, descriptor_pool_size);
      ctx->descriptor_pools.push_back(device->device.createDescriptorPool(descriptor_pool_create_info));
    }

    std::vector<vk::DescriptorSetLayout> layouts(alloc_count);
    for (uint32_t i = 0; i < alloc_count; i++) { layouts[i] = device->dsl; }
    vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(ctx->descriptor_pools[pool_idx],
                                                            alloc_count,
                                                            layouts.data());
    std::vector<vk::DescriptorSet> sets = device->device.allocateDescriptorSets(descriptor_set_alloc_info);
    ctx->descriptor_sets.insert(ctx->descriptor_sets.end(), sets.begin(), sets.end());

    pool_idx++;
  }
}

vk_matmul_pipeline v_vk_get_mul_mat_mat_pipeline(vk_backend_ctx* ctx, v_data_type src0_type, v_data_type src1_type, v_prec prec) {
  VK_LOG_DEBUG(
    "v_vk_get_mul_mat_mat_pipeline(" << v_type_name(src0_type) << ", " << v_type_name(src1_type) << ", " <<
    prec << ")");
  if (src0_type == v_TYPE_F32 && src1_type == v_TYPE_F32) { return ctx->device->pipeline_matmul_f32; }
  if (src0_type == v_TYPE_F32 && src1_type == v_TYPE_F16) { return ctx->device->pipeline_matmul_f32_f16; }
  if (src0_type == v_TYPE_BF16 && src1_type == v_TYPE_BF16) { return ctx->device->pipeline_matmul_bf16; }
  if (prec == v_PREC_DEFAULT && ctx->device->fp16 && !(ctx->device->coopmat_support && !ctx->device->
                                                                                             coopmat_acc_f16_support)) {
    if (src0_type == v_TYPE_F16 && src1_type == v_TYPE_F32) { return ctx->device->pipeline_matmul_f16_f32.f16acc; }
    if (src0_type == v_TYPE_F16 && src1_type == v_TYPE_F16) { return ctx->device->pipeline_matmul_f16.f16acc; }
  }
  else {
    if (src0_type == v_TYPE_F16 && src1_type == v_TYPE_F32) { return ctx->device->pipeline_matmul_f16_f32.f32acc; }
    if (src0_type == v_TYPE_F16 && src1_type == v_TYPE_F16) { return ctx->device->pipeline_matmul_f16.f32acc; }
  }

  // MMQ
  if (src1_type == v_TYPE_Q8_1) {
    vk_matmul_pipeline pipelines = (ctx->device->fp16 && prec == v_PREC_DEFAULT)
                                     ? ctx->device->pipeline_dequant_mul_mat_mat_q8_1[src0_type].f16acc
                                     : ctx->device->pipeline_dequant_mul_mat_mat_q8_1[src0_type].f32acc;

    if (pipelines->s == nullptr && pipelines->m == nullptr && pipelines->l == nullptr) { return nullptr; }

    return pipelines;
  }

  if (src1_type != v_TYPE_F32 && !ctx->device->coopmat2) { return nullptr; }

  switch (src0_type) {
    case v_TYPE_Q4_0:
    case v_TYPE_Q4_1:
    case v_TYPE_Q5_0:
    case v_TYPE_Q5_1:
    case v_TYPE_Q8_0:
    case v_TYPE_Q2_K:
    case v_TYPE_Q3_K:
    case v_TYPE_Q4_K:
    case v_TYPE_Q5_K:
    case v_TYPE_Q6_K:
    case v_TYPE_IQ1_S:
    case v_TYPE_IQ1_M:
    case v_TYPE_IQ2_XXS:
    case v_TYPE_IQ2_XS:
    case v_TYPE_IQ2_S:
    case v_TYPE_IQ3_XXS:
    case v_TYPE_IQ3_S:
    case v_TYPE_IQ4_XS:
    case v_TYPE_IQ4_NL:
    case v_TYPE_MXFP4:
      break;
    default:
      return nullptr;
  }

  if (ctx->device->coopmat2) {
    assert(src1_type == v_TYPE_F16);
    return prec == v_PREC_DEFAULT
             ? ctx->device->pipeline_dequant_mul_mat_mat_f16[src0_type].f16acc
             : ctx->device->pipeline_dequant_mul_mat_mat_f16[src0_type].f32acc;
  }
  if (ctx->device->coopmat_support) {
    return (ctx->device->fp16 && ctx->device->coopmat_acc_f16_support && prec == v_PREC_DEFAULT)
             ? ctx->device->pipeline_dequant_mul_mat_mat[src0_type].f16acc
             : ctx->device->pipeline_dequant_mul_mat_mat[src0_type].f32acc;
  }
  return (ctx->device->fp16 && prec == v_PREC_DEFAULT)
           ? ctx->device->pipeline_dequant_mul_mat_mat[src0_type].f16acc
           : ctx->device->pipeline_dequant_mul_mat_mat[src0_type].f32acc;
}
vk_pipeline v_vk_get_dequantize_mul_mat_vec(vk_backend_ctx* ctx, v_data_type a_type, v_data_type b_type, uint32_t num_cols, uint32_t m, uint32_t k) {
  VK_LOG_DEBUG("v_vk_get_dequantize_mul_mat_vec()");
  V_ASSERT(b_type == v_TYPE_F32 || b_type == v_TYPE_F16 || b_type == v_TYPE_Q8_1);
  V_ASSERT(num_cols >= 1 && num_cols <= mul_mat_vec_max_cols);

  if (b_type == v_TYPE_Q8_1) {
    switch (a_type) {
      case v_TYPE_Q4_0:
      case v_TYPE_Q4_1:
      case v_TYPE_Q5_0:
      case v_TYPE_Q5_1:
      case v_TYPE_Q8_0:
        break;
      default:
        return nullptr;
    }
  }

  switch (a_type) {
    case v_TYPE_F32:
    case v_TYPE_F16:
    case v_TYPE_BF16:
    case v_TYPE_Q4_0:
    case v_TYPE_Q4_1:
    case v_TYPE_Q5_0:
    case v_TYPE_Q5_1:
    case v_TYPE_Q8_0:
    case v_TYPE_Q2_K:
    case v_TYPE_Q3_K:
    case v_TYPE_Q4_K:
    case v_TYPE_Q5_K:
    case v_TYPE_Q6_K:
    case v_TYPE_IQ1_S:
    case v_TYPE_IQ1_M:
    case v_TYPE_IQ2_XXS:
    case v_TYPE_IQ2_XS:
    case v_TYPE_IQ2_S:
    case v_TYPE_IQ3_XXS:
    case v_TYPE_IQ3_S:
    case v_TYPE_IQ4_XS:
    case v_TYPE_IQ4_NL:
    case v_TYPE_MXFP4:
      break;
    default:
      return nullptr;
  }

  // heuristic to choose workgroup size
  uint32_t dmmv_wg = DMMV_WG_SIZE_SUBGROUP;
  if ((ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA && ctx->device->architecture !=
    vk_device_architecture::NVIDIA_PRE_TURING) || ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
    // Prefer larger workgroups when M is small, to spread the work out more
    // and keep more SMs busy.
    // q6_k seems to prefer small workgroup size even for "medium" values of M.
    if (a_type == v_TYPE_Q6_K) { if (m < 4096 && k >= 1024) { dmmv_wg = DMMV_WG_SIZE_LARGE; } }
    else { if (m <= 8192 && k >= 1024) { dmmv_wg = DMMV_WG_SIZE_LARGE; } }
  }

  if (b_type == v_TYPE_Q8_1) {
    if (ctx->device->vendor_id == VK_VENDOR_ID_INTEL) { dmmv_wg = DMMV_WG_SIZE_SUBGROUP; }
    return ctx->device->pipeline_dequant_mul_mat_vec_q8_1_f32[dmmv_wg][a_type][num_cols - 1];
  }

  return b_type == v_TYPE_F32
           ? ctx->device->pipeline_dequant_mul_mat_vec_f32_f32[dmmv_wg][a_type][num_cols - 1]
           : ctx->device->pipeline_dequant_mul_mat_vec_f16_f32[dmmv_wg][a_type][num_cols - 1];
}
vk_matmul_pipeline v_vk_get_mul_mat_mat_id_pipeline(vk_backend_ctx* ctx, v_data_type src0_type, v_data_type src1_type, v_prec prec) {
  VK_LOG_DEBUG("v_vk_get_mul_mat_mat_id_pipeline()");
  if (src0_type == v_TYPE_F32 && src1_type == v_TYPE_F32) { return ctx->device->pipeline_matmul_id_f32; }
  if (src0_type == v_TYPE_BF16 && src1_type == v_TYPE_BF16) { return ctx->device->pipeline_matmul_id_bf16; }
  if (prec == v_PREC_DEFAULT && ctx->device->fp16 && !(ctx->device->coopmat_support && !ctx->device->
                                                                                             coopmat_acc_f16_support)) {
    if (src0_type == v_TYPE_F16 && src1_type == v_TYPE_F32) { return ctx->device->pipeline_matmul_id_f16_f32.f16acc; }
    if (src0_type == v_TYPE_F16 && src1_type == v_TYPE_F16) { return ctx->device->pipeline_matmul_id_f16.f16acc; }
  }
  else {
    if (src0_type == v_TYPE_F16 && src1_type == v_TYPE_F32) { return ctx->device->pipeline_matmul_id_f16_f32.f32acc; }
    if (src0_type == v_TYPE_F16 && src1_type == v_TYPE_F16) { return ctx->device->pipeline_matmul_id_f16.f32acc; }
  }

  V_ASSERT(src1_type == v_TYPE_F32 || (ctx->device->coopmat2 && src1_type == v_TYPE_F16));

  switch (src0_type) {
    case v_TYPE_Q4_0:
    case v_TYPE_Q4_1:
    case v_TYPE_Q5_0:
    case v_TYPE_Q5_1:
    case v_TYPE_Q8_0:
    case v_TYPE_Q2_K:
    case v_TYPE_Q3_K:
    case v_TYPE_Q4_K:
    case v_TYPE_Q5_K:
    case v_TYPE_Q6_K:
    case v_TYPE_IQ1_S:
    case v_TYPE_IQ1_M:
    case v_TYPE_IQ2_XXS:
    case v_TYPE_IQ2_XS:
    case v_TYPE_IQ2_S:
    case v_TYPE_IQ3_XXS:
    case v_TYPE_IQ3_S:
    case v_TYPE_IQ4_XS:
    case v_TYPE_IQ4_NL:
    case v_TYPE_MXFP4:
      break;
    default:
      return nullptr;
  }

  // XXX TODO 'prec' is not actually allowed in mul_mat_id.
  bool prefer_fp16acc  = ctx->device->fp16 /*&& prec == v_PREC_DEFAULT*/;
  bool support_fp16acc = ctx->device->pipeline_dequant_mul_mat_mat_id[src0_type].f16acc != nullptr;
  bool support_fp32acc = ctx->device->pipeline_dequant_mul_mat_mat_id[src0_type].f32acc != nullptr;

  if (support_fp16acc && (prefer_fp16acc || !support_fp32acc)) { return ctx->device->pipeline_dequant_mul_mat_mat_id[src0_type].f16acc; }
  else {
    V_ASSERT(support_fp32acc);
    return ctx->device->pipeline_dequant_mul_mat_mat_id[src0_type].f32acc;
  }
}
vk_pipeline v_vk_get_dequantize_mul_mat_vec_id(vk_backend_ctx* ctx, v_data_type a_type, v_data_type b_type) {
  VK_LOG_DEBUG("v_vk_get_dequantize_mul_mat_vec_id()");
  V_ASSERT(b_type == v_TYPE_F32);

  switch (a_type) {
    case v_TYPE_F32:
    case v_TYPE_F16:
    case v_TYPE_BF16:
    case v_TYPE_Q4_0:
    case v_TYPE_Q4_1:
    case v_TYPE_Q5_0:
    case v_TYPE_Q5_1:
    case v_TYPE_Q8_0:
    case v_TYPE_Q2_K:
    case v_TYPE_Q3_K:
    case v_TYPE_Q4_K:
    case v_TYPE_Q5_K:
    case v_TYPE_Q6_K:
    case v_TYPE_IQ1_S:
    case v_TYPE_IQ1_M:
    case v_TYPE_IQ2_XXS:
    case v_TYPE_IQ2_XS:
    case v_TYPE_IQ2_S:
    case v_TYPE_IQ3_XXS:
    case v_TYPE_IQ3_S:
    case v_TYPE_IQ4_XS:
    case v_TYPE_IQ4_NL:
    case v_TYPE_MXFP4:
      break;
    default:
      return nullptr;
  }

  return ctx->device->pipeline_dequant_mul_mat_vec_id_f32[a_type];
}
uint32_t v_vk_guess_split_k(vk_backend_ctx* ctx, uint32_t m, uint32_t n, uint32_t k, bool disable_split_k, const vk_pipeline& pipeline) {
  VK_LOG_DEBUG("v_vk_guess_split_k(" << m << ", " << n << ", " << k << ", " << disable_split_k << ")");

  if (disable_split_k) { return 1; }

  uint32_t split_k = 1;
  if (ctx->device->shader_core_count != 0 && m >= pipeline->wg_denoms[0] && n >= pipeline->wg_denoms[1]) {
    // If k is 'large' and the SMs will fill less than halfway, use split_k.
    uint32_t m_tiles = CEIL_DIV(m, pipeline->wg_denoms[0]);
    uint32_t n_tiles = CEIL_DIV(n, pipeline->wg_denoms[1]);

    if (k >= 2048) {
      if (m_tiles * n_tiles <= ctx->device->shader_core_count / 2) { split_k = ctx->device->shader_core_count / (m_tiles * n_tiles); }
      else if (m_tiles * n_tiles <= ctx->device->shader_core_count * 2 / 3) { split_k = 3; }
      // Cap the split at 8x. Unless k is huge this is a lot of overhead.
      split_k = std::min(split_k, 8u);

      // v_vk_matmul will align the splits to be a multiple of 256.
      // If this rounded up size would cause the last split to be empty,
      // then reduce the split count.
      while (true) {
        if (split_k == 1) { break; }
        uint32_t k_split = CEIL_DIV(k, split_k);
        k_split          = ROUNDUP_POW2(k_split, 256);
        if (k_split * (split_k - 1) < k) { break; }
        split_k--;
      }
    }
  }

  return split_k;
}
vk_pipeline v_vk_guess_matmul_pipeline(vk_backend_ctx* ctx, vk_matmul_pipeline& mmp, uint32_t m, uint32_t n, bool aligned, v_data_type src0_type, v_data_type src1_type) {
  VK_LOG_DEBUG(
    "v_vk_guess_matmul_pipeline(" << m << ", " << n << ", " << aligned << ", " << v_type_name(src0_type) << ", "
    << v_type_name(src1_type) << ")");

  if (ctx->device->coopmat2) {
    const uint32_t shader_core_count = ctx->device->shader_core_count;
    const uint32_t tiles_l           = CEIL_DIV(m, mmp->a_l->wg_denoms[0]) * CEIL_DIV(n, mmp->a_l->wg_denoms[1]);
    const uint32_t tiles_m           = CEIL_DIV(m, mmp->a_m->wg_denoms[0]) * CEIL_DIV(n, mmp->a_m->wg_denoms[1]);

    // Use large shader when the N dimension is greater than the medium shader's tile size
    uint32_t crossover_large = mmp->m->wg_denoms[1];

    // Prefer large over medium if either:
    // - medium or large tiles would overfill the GPU
    // - large tiles with a split_k==3 fits in the GPU and medium tiles with split_k==2 does not
    //   (medium with split_k==2 is probably better if it fits - more workgroups running and less split_k overhead)
    bool prefer_large = tiles_m > shader_core_count || tiles_l > shader_core_count ||
      // split_k==3 with large tiles likely better than medium tiles with no split_k.
      (tiles_l <= shader_core_count / 3 && tiles_m > shader_core_count / 2);

    if ((ctx->device->mul_mat_l[src0_type] && (n > crossover_large && prefer_large)) || (!ctx->device->mul_mat_m[
        src0_type] && !ctx->device->
                            mul_mat_s
      [src0_type])) {
      return aligned
               ? mmp->a_l
               : mmp->l;
    }
    // Use medium shader when the N dimension is greater than the small shader's tile size
    uint32_t crossover_medium = mmp->s->wg_denoms[1];
    if ((ctx->device->mul_mat_m[src0_type] && (n > crossover_medium)) || !ctx->device->mul_mat_s[src0_type]) {
      return aligned
               ? mmp->a_m
               : mmp->m;
    }
    return aligned
             ? mmp->a_s
             : mmp->s;
  }

  if ((ctx->device->mul_mat_s[src0_type] && (m <= 32 || n <= 32)) || (!ctx->device->mul_mat_m[src0_type] && !ctx->device
                                                                                                                ->
                                                                                                                mul_mat_l
    [src0_type])) {
    return aligned
             ? mmp->a_s
             : mmp->s;
  }
  if ((ctx->device->mul_mat_m[src0_type] && (m <= 64 || n <= 64)) || !ctx->device->mul_mat_l[src0_type]) {
    return aligned
             ? mmp->a_m
             : mmp->m;
  }
  return aligned
           ? mmp->a_l
           : mmp->l;

  V_UNUSED(src1_type);
}
uint32_t v_vk_guess_matmul_pipeline_align(vk_backend_ctx* ctx, vk_matmul_pipeline& mmp, int m, int n, v_data_type src0_type, v_data_type src1_type) {
  VK_LOG_DEBUG(
    "v_vk_guess_matmul_pipeline_align(" << m << ", " << n << ", " << v_type_name(src0_type) << ", " <<
    v_type_name(src1_type) << ")");
  return v_vk_guess_matmul_pipeline(ctx, mmp, m, n, true, src0_type, src1_type)->align;
}

vk_pipeline v_vk_get_quantize_pipeline(vk_backend_ctx* ctx, v_data_type type, bool use_x4_blocks) {
  switch (type) {
    case v_TYPE_Q8_1:
      return use_x4_blocks
               ? ctx->device->pipeline_quantize_q8_1_x4
               : ctx->device->pipeline_quantize_q8_1;
    default:
      std::cerr << "Missing quantize pipeline for type: " << v_type_name(type) << std::endl;
      V_ABORT("fatal error");
  }
}

vk_pipeline v_vk_op_get_pipeline(vk_backend_ctx* ctx, const v_tensor* src0,
                                 const v_tensor* src1, const v_tensor* src2, const v_tensor* dst,
                                 v_operation op) {
  switch (op) {
    case v_OP_GET_ROWS:
      V_ASSERT(src1->type == v_TYPE_I32);
      if (dst->type == v_TYPE_F16) { return ctx->device->pipeline_get_rows[src0->type]; }
      if (dst->type == v_TYPE_F32) { return ctx->device->pipeline_get_rows_f32[src0->type]; }
      return nullptr;
    case v_OP_ACC:
      if (src0->type == v_TYPE_F32 && src1->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_acc_f32; }
      return nullptr;
    case v_OP_ADD:
    case v_OP_SUB:
    case v_OP_MUL:
    case v_OP_DIV:
      if ((src0->type != v_TYPE_F32 && src0->type != v_TYPE_F16) ||
        (src1->type != v_TYPE_F32 && src1->type != v_TYPE_F16) ||
        (dst->type != v_TYPE_F32 && dst->type != v_TYPE_F16)) { return nullptr; }
      switch (op) {
        case v_OP_ADD: {
          if (ctx->num_additional_fused_ops > 0) {
            if (ctx->do_add_rms_partials) { return ctx->device->pipeline_multi_add_rms[ctx->num_additional_fused_ops]; }
            else { return ctx->device->pipeline_multi_add[ctx->num_additional_fused_ops]; }
          }
          if (ctx->do_add_rms_partials) {
            auto pipelines = v_are_same_shape(src0, src1) ? ctx->device->pipeline_add_rms_norepeat : ctx->device->pipeline_add_rms;
            return pipelines[src0->type == v_TYPE_F16][src1->type == v_TYPE_F16][dst->type == v_TYPE_F16];
          }
          else {
            auto pipelines = v_are_same_shape(src0, src1) ? ctx->device->pipeline_add_norepeat : ctx->device->pipeline_add;
            return pipelines[src0->type == v_TYPE_F16][src1->type == v_TYPE_F16][dst->type == v_TYPE_F16];
          }
        }
        case v_OP_SUB: {
          auto pipelines = v_are_same_shape(src0, src1)
                             ? ctx->device->pipeline_sub_norepeat
                             : ctx->device->pipeline_sub;
          return pipelines[src0->type == v_TYPE_F16][src1->type == v_TYPE_F16][dst->type == v_TYPE_F16];
        }
        case v_OP_MUL: {
          auto pipelines = v_are_same_shape(src0, src1)
                             ? ctx->device->pipeline_mul_norepeat
                             : ctx->device->pipeline_mul;
          return pipelines[src0->type == v_TYPE_F16][src1->type == v_TYPE_F16][dst->type == v_TYPE_F16];
        }
        case v_OP_DIV: {
          auto pipelines = v_are_same_shape(src0, src1)
                             ? ctx->device->pipeline_div_norepeat
                             : ctx->device->pipeline_div;
          return pipelines[src0->type == v_TYPE_F16][src1->type == v_TYPE_F16][dst->type == v_TYPE_F16];
        }
        default:
          break;
      }
      return nullptr;
    case v_OP_ADD_ID:
      if (src0->type == v_TYPE_F32 && src1->type == v_TYPE_F32 && src2->type == v_TYPE_I32 && dst->type ==
        v_TYPE_F32) { return ctx->device->pipeline_add_id_f32; }
      return nullptr;
    case v_OP_CONCAT:
      if (src0->type == v_TYPE_F32 && src1->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_concat_f32; }
      if (src0->type == v_TYPE_F16 && src1->type == v_TYPE_F16 && dst->type == v_TYPE_F16) { return ctx->device->pipeline_concat_f16; }
      if (src0->type == v_TYPE_I32 && src1->type == v_TYPE_I32 && dst->type == v_TYPE_I32) { return ctx->device->pipeline_concat_i32; }
      return nullptr;
    case v_OP_UPSCALE:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) {
        switch (static_cast<v_scale_mode>(v_get_op_params_i32(dst, 0) & 0xFF)) {
          case v_SCALE_MODE_NEAREST:
            return ctx->device->pipeline_upscale_nearest_f32;
          case v_SCALE_MODE_BILINEAR:
            return ctx->device->pipeline_upscale_bilinear_f32;
          default:
            return nullptr;
        }
      }
      return nullptr;
    case V_OP_SCALE:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_scale_f32; }
      return nullptr;
    case V_OP_SQR:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_sqr_f32; }
      return nullptr;
    case v_OP_SQRT:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_sqrt_f32; }
      return nullptr;
    case V_OP_SIN:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_sin_f32; }
      return nullptr;
    case V_OP_COS:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_cos_f32; }
      return nullptr;
    case V_OP_CLAMP:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_clamp_f32; }
      return nullptr;
    case v_OP_PAD:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_pad_f32; }
      return nullptr;
    case v_OP_ROLL:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_roll_f32; }
      return nullptr;
    case v_OP_REPEAT:
      if (v_type_size(src0->type) == sizeof(float) && v_type_size(dst->type) == sizeof(float)) { return ctx->device->pipeline_repeat_f32; }
      return nullptr;
    case v_OP_REPEAT_BACK:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_repeat_back_f32; }
      return nullptr;
    case V_OP_CPY:
    case V_OP_CONT:
    case v_OP_DUP:
      return v_vk_get_cpy_pipeline(ctx, src0, dst, dst->type);
    case v_OP_SET_ROWS:
      if (src1->type == v_TYPE_I64) { return ctx->device->pipeline_set_rows_i64[dst->type]; }
      else { return ctx->device->pipeline_set_rows_i32[dst->type]; }
    case v_OP_SILU_BACK:
      if (src0->type == v_TYPE_F32 && src1->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_silu_back_f32; }
      return nullptr;
    case v_OP_NORM:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_norm_f32; }
      return nullptr;
    case v_OP_GROUP_NORM:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_group_norm_f32; }
      return nullptr;
    case v_OP_RMS_NORM:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) {
        if (ctx->do_add_rms_partials) {
          return ctx->num_additional_fused_ops > 0
                   ? ctx->device->pipeline_rms_norm_mul_partials_f32
                   : ctx->device->pipeline_rms_norm_partials_f32;
        }
        else {
          return ctx->num_additional_fused_ops > 0
                   ? ctx->device->pipeline_rms_norm_mul_f32
                   : ctx->device->pipeline_rms_norm_f32;
        }
      }
      return nullptr;
    case v_OP_RMS_NORM_BACK:
      if (src0->type == v_TYPE_F32 && src1->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_rms_norm_back_f32; }
      return nullptr;
    case v_OP_L2_NORM:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_l2_norm_f32; }
      return nullptr;
    case v_OP_UNARY:
      if ((src0->type != v_TYPE_F32 && src0->type != v_TYPE_F16) ||
        (dst->type != v_TYPE_F32 && dst->type != v_TYPE_F16) ||
        (src0->type != dst->type)) { return nullptr; }

      switch (v_get_unary_op(dst)) {
        case v_UNARY_OP_EXP:
          return ctx->device->pipeline_exp[dst->type == v_TYPE_F16];

        case V_UNARY_OP_LOG:
          return ctx->device->pipeline_log[dst->type == v_TYPE_F16];

        case v_UNARY_OP_SILU:
          return ctx->device->pipeline_silu[dst->type == v_TYPE_F16];
        case v_UNARY_OP_GELU:
          return ctx->device->pipeline_gelu[dst->type == v_TYPE_F16];
        case v_UNARY_OP_GELU_ERF:
          return ctx->device->pipeline_gelu_erf[dst->type == v_TYPE_F16];
        case v_UNARY_OP_GELU_QUICK:
          return ctx->device->pipeline_gelu_quick[dst->type == v_TYPE_F16];
        case v_UNARY_OP_RELU:
          return ctx->device->pipeline_relu[dst->type == v_TYPE_F16];
        case v_UNARY_OP_TANH:
          return ctx->device->pipeline_tanh[dst->type == v_TYPE_F16];
        case v_UNARY_OP_SIGMOID:
          return ctx->device->pipeline_sigmoid[dst->type == v_TYPE_F16];
        case v_UNARY_OP_HARDSIGMOID:
          return ctx->device->pipeline_hardsigmoid[dst->type == v_TYPE_F16];
        case v_UNARY_OP_HARDSWISH:
          return ctx->device->pipeline_hardswish[dst->type == v_TYPE_F16];
        default:
          break;
      }
      return nullptr;
    case v_OP_GLU:
      if ((src0->type != v_TYPE_F32 && src0->type != v_TYPE_F16) ||
        (dst->type != v_TYPE_F32 && dst->type != v_TYPE_F16) ||
        (src0->type != dst->type)) { return nullptr; }

      switch (v_get_glu_op(dst)) {
        case v_GLU_OP_GEGLU:
          return ctx->device->pipeline_geglu[dst->type == v_TYPE_F16];
        case v_GLU_OP_REGLU:
          return ctx->device->pipeline_reglu[dst->type == v_TYPE_F16];
        case v_GLU_OP_SWIGLU:
          return ctx->device->pipeline_swiglu[dst->type == v_TYPE_F16];
        case v_GLU_OP_SWIGLU_OAI:
          return ctx->device->pipeline_swiglu_oai[dst->type == v_TYPE_F16];
        case v_GLU_OP_GEGLU_ERF:
          return ctx->device->pipeline_geglu_erf[dst->type == v_TYPE_F16];
        case v_GLU_OP_GEGLU_QUICK:
          return ctx->device->pipeline_geglu_quick[dst->type == v_TYPE_F16];
        default:
          break;
      }
      return nullptr;
    case V_OP_DIAG_MASK_INF:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_diag_mask_inf_f32; }
      return nullptr;
    case V_OP_SOFT_MAX:
      V_ASSERT(!src1 || src1->type == v_TYPE_F32 || src1->type == v_TYPE_F16);
      V_ASSERT(!src2 || src2->type == v_TYPE_F32);

      if (ctx->num_additional_fused_ops) {
        uint32_t idx = static_cast<uint32_t>(ceilf(log2f(static_cast<float>(dst->ne[0]))));
        V_ASSERT(idx < num_topk_moe_pipelines);
        bool with_norm = ctx->num_additional_fused_ops == topk_moe_norm.size() - 1;
        return ctx->device->pipeline_topk_moe[idx][with_norm];
      }

      if (src0->type == v_TYPE_F32 && (src1 == nullptr || src1->type == v_TYPE_F32) && dst->type == v_TYPE_F32) {
        return src0->ne[0] > 1024
                 ? ctx->device->pipeline_soft_max_f32_wg512
                 : ctx->device->pipeline_soft_max_f32;
      }
      if (src0->type == v_TYPE_F32 && src1->type == v_TYPE_F16 && dst->type == v_TYPE_F32) {
        return src0->ne[0] > 1024
                 ? ctx->device->pipeline_soft_max_f32_f16_wg512
                 : ctx->device->pipeline_soft_max_f32_f16;
      }
      return nullptr;
    case v_OP_SOFT_MAX_BACK:
      if (src0->type == v_TYPE_F32 && src1->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_soft_max_back_f32; }
      return nullptr;
    case V_OP_ROPE:
    case v_OP_ROPE_BACK: {
      const int mode       = dst->op_params[2];
      const bool is_neox   = mode & V_ROPE_TYPE_NEOX;
      const bool is_mrope  = mode & V_ROPE_TYPE_MROPE;
      const bool is_vision = mode == V_ROPE_TYPE_VISION;

      if (is_neox) {
        if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_rope_neox_f32; }
        if (src0->type == v_TYPE_F16 && dst->type == v_TYPE_F16) { return ctx->device->pipeline_rope_neox_f16; }
      }
      else if (is_mrope && !is_vision) {
        if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_rope_multi_f32; }
        if (src0->type == v_TYPE_F16 && dst->type == v_TYPE_F16) { return ctx->device->pipeline_rope_multi_f16; }
      }
      else if (is_vision) {
        if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_rope_vision_f32; }
        if (src0->type == v_TYPE_F16 && dst->type == v_TYPE_F16) { return ctx->device->pipeline_rope_vision_f16; }
      }
      else {
        if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_rope_norm_f32; }
        if (src0->type == v_TYPE_F16 && dst->type == v_TYPE_F16) { return ctx->device->pipeline_rope_norm_f16; }
      }
      return nullptr;
    }
    case v_OP_ARGSORT:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_I32) {
        uint32_t idx = static_cast<uint32_t>(ceilf(log2f(static_cast<float>(dst->ne[0]))));
        return ctx->device->pipeline_argsort_f32[idx];
      }
      return nullptr;
    case v_OP_SUM:
    case v_OP_SUM_ROWS:
    case V_OP_MEAN:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_sum_rows_f32; }
      return nullptr;
    case V_OP_ARGMAX:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_I32) { return ctx->device->pipeline_argmax_f32; }
      return nullptr;
    case v_OP_COUNT_EQUAL:
      if (src0->type == v_TYPE_I32 && src1->type == v_TYPE_I32 && dst->type == v_TYPE_I64) { return ctx->device->pipeline_count_equal_i32; }
      return nullptr;
    case V_OP_IM2COL:
      if (src1->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_im2col_f32; }
      if (src1->type == v_TYPE_F32 && dst->type == v_TYPE_F16) { return ctx->device->pipeline_im2col_f32_f16; }
      return nullptr;
    case v_OP_IM2COL_3D:
      if (src1->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_im2col_3d_f32; }
      if (src1->type == v_TYPE_F32 && dst->type == v_TYPE_F16) { return ctx->device->pipeline_im2col_3d_f32_f16; }
      return nullptr;
    case v_OP_TIMESTEP_EMBEDDING:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_timestep_embedding_f32; }
      return nullptr;
    case v_OP_CONV_TRANSPOSE_1D:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_conv_transpose_1d_f32; }
      return nullptr;
    case V_OP_POOL_2D:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_pool2d_f32; }
      return nullptr;
    case V_OP_POOL_2D_BACK:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_pool2d_back_f32; }
      return nullptr;

    case v_OP_RWKV_WKV6:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_rwkv_wkv6_f32; }
      return nullptr;
    case v_OP_RWKV_WKV7:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_rwkv_wkv7_f32; }
      return nullptr;
    case v_OP_SSM_SCAN:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) {
        const uint32_t d_state = src0->ne[0];
        if (d_state == 128) { return ctx->device->pipeline_ssm_scan_f32_d128; }
        else if (d_state == 256) { return ctx->device->pipeline_ssm_scan_f32_d256; }
      }
      return nullptr;
    case V_OP_SSM_CONV:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_ssm_conv_f32; }
      return nullptr;
    case v_OP_OPT_STEP_ADAMW:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_opt_step_adamw_f32; }
      return nullptr;
    case v_OP_OPT_STEP_SGD:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_opt_step_sgd_f32; }
      return nullptr;
    case V_OP_LEAKY_RELU:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_leaky_relu_f32; }
      return nullptr;
    case v_OP_CONV_2D:
    case v_OP_CONV_TRANSPOSE_2D:
      if (src1->type == v_TYPE_F32 && dst->type == v_TYPE_F32 &&
        v_is_contiguous(src0) && v_is_contiguous(src1) && v_is_contiguous(dst)) {
        std::array<uint32_t, 3> elements;
        if (op == v_OP_CONV_2D) elements = v_vk_get_conv_elements(dst);
        else if (op == v_OP_CONV_TRANSPOSE_2D) elements = v_vk_get_conv_transpose_2d_elements(dst);
        vk_conv_shapes shape;

        uint32_t tiles[CONV_SHAPE_COUNT];
        for (uint32_t i = 0; i < CONV_SHAPE_COUNT; ++i) {
          tiles[i] = CEIL_DIV(elements[0], ctx->device->pipeline_conv2d_f32[i]->wg_denoms[0]) * CEIL_DIV(
            elements[1],
            ctx->device->pipeline_conv2d_f32[i]->wg_denoms[1]);
        }

        // We can't query number of shader cores on Intel, use 32 as a placeholder
        // so small convolutions will still choose a smaller tile.
        const uint32_t shader_core_count = ctx->device->shader_core_count > 0
                                             ? ctx->device->shader_core_count
                                             : 32;

        if (elements[0] > 64 && tiles[CONV_SHAPE_128x128] >= shader_core_count * 2) { shape = CONV_SHAPE_128x128; }
        else if (elements[0] <= 32 && tiles[CONV_SHAPE_32x256] >= shader_core_count * 2) { shape = CONV_SHAPE_32x256; }
        else { shape = CONV_SHAPE_64x32; }

        if (op == v_OP_CONV_2D) {
          if (src0->type == v_TYPE_F32) { return ctx->device->pipeline_conv2d_f32[shape]; }
          else if (src0->type == v_TYPE_F16) { return ctx->device->pipeline_conv2d_f16_f32[shape]; }
        }
        else if (op == v_OP_CONV_TRANSPOSE_2D) {
          if (src0->type == v_TYPE_F32) { return ctx->device->pipeline_conv_transpose_2d_f32[shape]; }
          else if (src0->type == v_TYPE_F16) { return ctx->device->pipeline_conv_transpose_2d_f16_f32[shape]; }
        }
      }
      return nullptr;
    case v_OP_CONV_2D_DW:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) {
        if (v_is_contiguous(src1)) { return ctx->device->pipeline_conv2d_dw_whcn_f32; }
        else if (v_is_contiguous_channels(src1)) { return ctx->device->pipeline_conv2d_dw_cwhn_f32; }
      }
      else if (src0->type == v_TYPE_F16 && dst->type == v_TYPE_F32) {
        if (v_is_contiguous(src1)) { return ctx->device->pipeline_conv2d_dw_whcn_f16_f32; }
        else if (v_is_contiguous_channels(src1)) { return ctx->device->pipeline_conv2d_dw_cwhn_f16_f32; }
      }
      return nullptr;
    default:
      return nullptr;
  }
  V_UNUSED(src2);
}
