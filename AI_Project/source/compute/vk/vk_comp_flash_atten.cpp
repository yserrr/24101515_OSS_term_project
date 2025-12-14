#include <vk_constant.h>
#include <vk_util.hpp>
#include "vk_op_f32.hpp"
#include "vk_comp.hpp"

bool v_vk_flash_attn_scalar_shmem_support(const vk_device& device, const uint32_t hsk, uint32_t hsv) {
  // Needs to be kept up to date on shader changes
  V_UNUSED(hsv);
  const uint32_t wg_size = scalar_flash_attention_workgroup_size;
  const uint32_t Br      = get_fa_scalar_num_large_rows(hsv);
  const uint32_t Bc      = scalar_flash_attention_Bc;

  const uint32_t tmpsh   = wg_size * sizeof(float);
  const uint32_t tmpshv4 = wg_size * 4 * sizeof(float);

  const uint32_t masksh = Bc * Br * sizeof(float);

  const uint32_t Qf = Br * (hsk / 4 + 2) * 4 * sizeof(float);

  const uint32_t total_size = tmpsh + tmpshv4 + masksh + Qf;
  const bool supported      = total_size <= device->properties.limits.maxComputeSharedMemorySize;

  VK_LOG_DEBUG(
    "v_vk_flash_attn_coopmat_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", total_size=" << total_size <<
    ", supported=" << supported);

  return supported;
}

bool v_vk_flash_attn_coopmat_shmem_support(const vk_device& device, const uint32_t hsk, uint32_t hsv,
                                           bool f32acc) {
  // Needs to be kept up to date on shader changes
  V_UNUSED(hsv);
  const uint32_t wg_size = scalar_flash_attention_workgroup_size;
  const uint32_t Br      = coopmat1_flash_attention_num_large_rows;
  const uint32_t Bc      = scalar_flash_attention_Bc;

  const uint32_t hsk_pad = ROUNDUP_POW2(hsk, 16);

  const uint32_t acctype = f32acc
                             ? 4
                             : 2;
  const uint32_t f16vec4 = 8;

  const uint32_t tmpsh   = wg_size * sizeof(float);
  const uint32_t tmpshv4 = wg_size * 4 * acctype;

  const uint32_t qstride = hsk_pad / 4 + 2;
  const uint32_t Qf      = Br * qstride * f16vec4;

  const uint32_t sfshstride = (hsk <= 128)
                                ? (Br + 8)
                                : Br;
  const uint32_t sfsh = Bc * sfshstride * acctype;

  const uint32_t kshstride = hsk_pad / 4 + 2;
  const uint32_t ksh       = Bc * kshstride * f16vec4;

  const uint32_t slope = Br * sizeof(float);

  const uint32_t total_size = tmpsh + tmpshv4 + Qf + sfsh + ksh + slope;
  const bool supported      = total_size <= device->properties.limits.maxComputeSharedMemorySize;

  VK_LOG_DEBUG(
    "v_vk_flash_attn_coopmat_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", f32acc=" << f32acc <<
    ", total_size=" << total_size << ", supported=" << supported);

  return supported;
}

void v_vk_flash_attn(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* q,
                     const v_tensor* k, const v_tensor* v, const v_tensor* mask,
                     const v_tensor* sinks, v_tensor* dst, bool dryrun) {
  VK_LOG_DEBUG(
    "v_vk_flash_attn((" << q << ", name=" << q->name << ", type=" << q->type << ", ne0=" << q->ne[0] << ", ne1=" << q
    ->ne[1] << ", ne2=" << q->ne[2] << ", ne3=" << q->ne[3] << ", nb0=" << q->nb[0] << ", nb1=" << q->nb[1] << ", nb2="
    << q->nb[2] << ", nb3=" << q->nb[3];
    std::cerr << "), (" << k << ", name=" << k->name << ", type=" << k->type << ", ne0=" << k->ne[0] << ", ne1=" << k->
    ne[1] << ", ne2=" << k->ne[2] << ", ne3=" << k->ne[3] << ", nb0=" << k->nb[0] << ", nb1=" << k->nb[1] << ", nb2=" <<
    k->nb[2] << ", nb3=" << k->nb[3];
    std::cerr << "), (" << v << ", name=" << v->name << ", type=" << v->type << ", ne0=" << v->ne[0] << ", ne1=" << v->
    ne[1] << ", ne2=" << v->ne[2] << ", ne3=" << v->ne[3] << ", nb0=" << v->nb[0] << ", nb1=" << v->nb[1] << ", nb2=" <<
    v->nb[2] << ", nb3=" << v->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1="
    << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1
    ] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    if (sinks) {
    std::cerr << "), (" << sinks << ", name=" << sinks->name << ", type=" << sinks->type << ", ne0=" << sinks->ne[0] <<
    ", ne1=" << sinks->ne[1] << ", ne2=" << sinks->ne[2] << ", ne3=" << sinks->ne[3] << ", nb0=" << sinks->nb[0] <<
    ", nb1=" << sinks->nb[1] << ", nb2=" << sinks->nb[2] << ", nb3=" << sinks->nb[3];
    }
    std::cerr << "), " << (dryrun ? "dryrun" : "") << ")");

  v_TENSOR_LOCALS(int64_t, neq, q, ne)
  v_TENSOR_LOCALS(size_t, nbq, q, nb)
  v_TENSOR_LOCALS(int64_t, nek, k, ne)
  v_TENSOR_LOCALS(size_t, nbk, k, nb)
  v_TENSOR_LOCALS(int64_t, nev, v, ne)
  v_TENSOR_LOCALS(size_t, nbv, v, nb)
  v_TENSOR_LOCALS(int64_t, ne, dst, ne)
  v_TENSOR_LOCALS(size_t, nb, dst, nb)

  const uint32_t nem1 = mask
                          ? mask->ne[1]
                          : 0;
  const uint32_t nem2 = mask
                          ? mask->ne[2]
                          : 0;
  const uint32_t nem3 = mask
                          ? mask->ne[3]
                          : 0;

  const uint32_t HSK = nek0;
  const uint32_t HSV = nev0;
  uint32_t N         = neq1;
  const uint32_t KV  = nek1;

  V_ASSERT(ne0 == HSV);
  V_ASSERT(ne2 == N);

  // input tensor rows must be contiguous
  V_ASSERT(nbq0 == v_type_size(q->type));
  V_ASSERT(nbk0 == v_type_size(k->type));
  V_ASSERT(nbv0 == v_type_size(v->type));

  V_ASSERT(neq0 == HSK);

  V_ASSERT(neq1 == N);

  V_ASSERT(nev1 == nek1);

  // dst cannot be transposed or permuted
  V_ASSERT(nb0 == sizeof(float));
  V_ASSERT(nb0 <= nb1);
  V_ASSERT(nb1 <= nb2);
  V_ASSERT(nb2 <= nb3);

  assert(dst->type == v_TYPE_F32);
  assert(q->type == v_TYPE_F32);
  assert(k->type == v->type);

  FaCodePath path = ctx->device->coopmat2
                      ? FA_COOPMAT2
                      : ctx->device->coopmat1_fa_support
                      ? FA_COOPMAT1
                      : FA_SCALAR;

  if (path == FA_COOPMAT1) {
    const bool coopmat_shape_supported = (dst->op_params[3] == v_PREC_F32 && ctx->device->
                                                                                  coopmat_support_16x16x16_f32acc) ||
    (dst->op_params[3] != v_PREC_F32 && ctx->device->
                                             coopmat_support_16x16x16_f16acc);

    const bool coopmat_shmem_supported = v_vk_flash_attn_coopmat_shmem_support(
      ctx->device,
      HSK,
      HSV,
      dst->op_params[3] == v_PREC_F32);

    if (!coopmat_shape_supported || !coopmat_shmem_supported) { path = FA_SCALAR; }
  }

  uint32_t gqa_ratio    = 1;
  uint32_t qk_ratio     = neq2 / nek2;
  uint32_t workgroups_x = (uint32_t)neq1;
  uint32_t workgroups_y = (uint32_t)neq2;
  uint32_t workgroups_z = (uint32_t)neq3;

  // For scalar/coopmat1 FA, we can use the "large" size to accommodate qga.
  // For coopmat2 FA, we always use the small size (which is still pretty large for gqa).
  uint32_t max_gqa;
  switch (path) {
    case FA_SCALAR:
    case FA_COOPMAT1:
      // We may switch from coopmat1 to scalar, so use the scalar limit for both
      max_gqa = get_fa_scalar_num_large_rows(HSV);
      break;
    case FA_COOPMAT2:
      max_gqa = get_fa_num_small_rows(FA_COOPMAT2);
      break;
    default:
      V_ASSERT(0);
  }

  if (N == 1 && qk_ratio > 1 && qk_ratio <= max_gqa &&
    qk_ratio * nek2 == neq2 && nek2 == nev2 && nem2 <= 1) {
    // grouped query attention - make the N dimension equal to gqa_ratio, reduce
    // workgroups proportionally in y dimension. The shader will detect gqa_ratio > 1
    // and change addressing calculations to index Q's dimension 2.
    gqa_ratio = qk_ratio;
    N         = gqa_ratio;
    workgroups_y /= N;
  }

  bool small_rows = N <= get_fa_num_small_rows(path);

  // coopmat1 does not actually support "small rows" (it needs 16 rows).
  // So use scalar instead.
  if (small_rows && path == FA_COOPMAT1) { path = FA_SCALAR; }

  // scalar is faster than coopmat2 when N==1
  if (N == 1 && path == FA_COOPMAT2) { path = FA_SCALAR; }

  // with large hsk/hsv, scalar path may need to use small_rows to fit in shared memory
  if (path == FA_SCALAR &&
    !v_vk_flash_attn_scalar_shmem_support(ctx->device, HSK, HSV)) { small_rows = true; }

  const uint32_t q_stride = (uint32_t)(nbq1 / v_type_size(q->type));
  uint32_t k_stride       = (uint32_t)(nbk1 / v_type_size(k->type));
  uint32_t v_stride       = (uint32_t)(nbv1 / v_type_size(v->type));

  // For F32, the shader treats it as a block of size 4 (for vec4 loads)
  if (k->type == v_TYPE_F32) { k_stride /= 4; }
  if (v->type == v_TYPE_F32) { v_stride /= 4; }

  uint32_t alignment = fa_align(path, HSK, HSV, k->type, small_rows);
  bool aligned       = (KV % alignment) == 0 &&
    // the "aligned" shader variant will forcibly align strides, for performance
    (q_stride & 7) == 0 && (k_stride & 7) == 0 && (v_stride & 7) == 0;

  // Need to use the coopmat2 variant that clamps loads when HSK/HSV aren't sufficiently aligned.
  if (((HSK | HSV) % 16) != 0 && path == FA_COOPMAT2) { aligned = false; }

  bool f32acc = path == FA_SCALAR || dst->op_params[3] == v_PREC_F32;

  vk_fa_pipeline_state fa_pipeline_state(HSK, HSV, small_rows, path, aligned, f32acc);

  vk_pipeline pipeline = nullptr;

  auto& pipelines = ctx->device->pipeline_flash_attn_f32_f16[k->type];
  auto it         = pipelines.find(fa_pipeline_state);
  if (it != pipelines.end()) { pipeline = it->second; }
  else { pipelines[fa_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>(); }

  assert(pipeline);

  uint32_t split_kv = KV;
  uint32_t split_k  = 1;

  // Use a placeholder core count if one isn't available. split_k is a big help for perf.
  const uint32_t shader_core_count = ctx->device->shader_core_count
                                       ? ctx->device->shader_core_count
                                       : 16;

  // Try to use split_k when KV is large enough to be worth the overhead
  if (workgroups_x == 1 && shader_core_count > 0) {
    // Try to run two workgroups per SM.
    split_k = shader_core_count * 2 / (workgroups_y * workgroups_z);
    if (split_k > 1) {
      // Try to evenly split KV into split_k chunks, but it needs to be a multiple
      // of "align", so recompute split_k based on that.
      split_kv     = ROUNDUP_POW2(std::max(1u, KV / split_k), alignment);
      split_k      = CEIL_DIV(KV, split_kv);
      workgroups_x = split_k;
    }
  }

  // Reserve space for split_k temporaries. For each split x batch, we need to store the O matrix (D x ne1)
  // and the per-row m and L values (ne1 rows). We store all the matrices first, followed by the rows.
  const uint64_t split_k_size = split_k > 1
                                  ? (HSV * ne1 * sizeof(float) + ne1 * sizeof(float) * 2) * split_k * ne3
                                  : 0;
  if (split_k_size > ctx->device->properties.limits.maxStorageBufferRange) { V_ABORT("Requested preallocation size is too large"); }
  if (ctx->prealloc_size_split_k < split_k_size) { ctx->prealloc_size_split_k = split_k_size; }

  if (dryrun) {
    // Request descriptor sets
    v_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    if (split_k > 1) { v_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_flash_attn_split_k_reduce, 1); }
    return;
  }

  float scale         = 1.0f;
  float max_bias      = 0.0f;
  float logit_softcap = 0.0f;

  memcpy(&scale, (const float*)dst->op_params.data() + 0, sizeof(float));
  memcpy(&max_bias, (const float*)dst->op_params.data() + 1, sizeof(float));
  memcpy(&logit_softcap, (const float*)dst->op_params.data() + 2, sizeof(float));

  if (logit_softcap != 0) { scale /= logit_softcap; }

  const uint32_t n_head_kv   = neq2;
  const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head_kv));
  const float m0             = powf(2.0f, -(max_bias) / n_head_log2);
  const float m1             = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

  vk_buffer d_Q       = nullptr, d_K    = nullptr, d_V    = nullptr, d_D    = nullptr, d_M    = nullptr, d_S    = nullptr;
  size_t q_buf_offset = 0, k_buf_offset = 0, v_buf_offset = 0, d_buf_offset = 0, m_buf_offset = 0, s_buf_offset = 0;

  bool Q_uma = false, K_uma = false, V_uma = false, D_uma = false, M_uma = false, S_uma = false;

  if (ctx->device->uma) {
    vk_get_host_buffer(ctx->device, q->data, d_Q, q_buf_offset);
    vk_get_host_buffer(ctx->device, k->data, d_K, k_buf_offset);
    vk_get_host_buffer(ctx->device, v->data, d_V, v_buf_offset);
    vk_get_host_buffer(ctx->device, dst->data, d_D, d_buf_offset);
    Q_uma = d_Q != nullptr;
    K_uma = d_K != nullptr;
    V_uma = d_V != nullptr;
    D_uma = d_D != nullptr;
    if (mask) {
      vk_get_host_buffer(ctx->device, mask->data, d_M, m_buf_offset);
      M_uma = d_M != nullptr;
    }
    if (sinks) {
      vk_get_host_buffer(ctx->device, sinks->data, d_S, s_buf_offset);
      S_uma = d_S != nullptr;
    }
  }


  v_backend_vk_buffer_ctx* d_buf_ctx = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  v_backend_vk_buffer_ctx* q_buf_ctx = (v_backend_vk_buffer_ctx*)q->buffer->context;
  v_backend_vk_buffer_ctx* k_buf_ctx = (v_backend_vk_buffer_ctx*)k->buffer->context;
  v_backend_vk_buffer_ctx* v_buf_ctx = (v_backend_vk_buffer_ctx*)v->buffer->context;

  if (!Q_uma) {
    d_Q          = q_buf_ctx->dev_buffer;
    q_buf_offset = vk_tensor_offset(q) + q->view_offs;
  }
  if (!K_uma) {
    d_K          = k_buf_ctx->dev_buffer;
    k_buf_offset = vk_tensor_offset(k) + k->view_offs;
  }
  if (!V_uma) {
    d_V          = v_buf_ctx->dev_buffer;
    v_buf_offset = vk_tensor_offset(v) + v->view_offs;
  }
  if (!D_uma) {
    d_D          = d_buf_ctx->dev_buffer;
    d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
  }

  if (!M_uma) {
    d_M          = d_Q;
    m_buf_offset = q_buf_offset;
    if (mask) {
      v_backend_vk_buffer_ctx* m_buf_ctx = (v_backend_vk_buffer_ctx*)mask->buffer->context;
      d_M                                = m_buf_ctx->dev_buffer;
      m_buf_offset                       = vk_tensor_offset(mask) + mask->view_offs;
    }
  }

  if (!S_uma) {
    d_S          = d_Q;
    s_buf_offset = q_buf_offset;
    if (sinks) {
      v_backend_vk_buffer_ctx* s_buf_ctx = static_cast<v_backend_vk_buffer_ctx*>(sinks->buffer->context);
      d_S                                = s_buf_ctx->dev_buffer;
      s_buf_offset                       = vk_tensor_offset(sinks) + sinks->view_offs;
    }
  }

  uint32_t mask_n_head_log2 = ((sinks != nullptr) << 24) | ((mask != nullptr) << 16) | n_head_log2;

  const vk_flash_attn_push_constants pc = {
    N, KV,
    (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
    (uint32_t)neq2, (uint32_t)neq3,
    (uint32_t)nek2, (uint32_t)nek3,
    (uint32_t)nev2, (uint32_t)nev3,
    nem1, nem2, nem3,
    q_stride, (uint32_t)nbq2, (uint32_t)nbq3,
    k_stride, (uint32_t)nbk2, (uint32_t)nbk3,
    v_stride, (uint32_t)nbv2, (uint32_t)nbv3,
    scale, max_bias, logit_softcap,
    mask_n_head_log2, m0, m1,
    gqa_ratio, split_kv, split_k
  };

  if (split_k > 1) {
    if (ctx->prealloc_split_k_need_sync) { vk_sync_buffers(ctx, subctx); }

    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             v_vk_subbuffer(ctx, d_Q, q_buf_offset),
                             v_vk_subbuffer(ctx, d_K, k_buf_offset),
                             v_vk_subbuffer(ctx, d_V, v_buf_offset),
                             v_vk_subbuffer(ctx, d_M, m_buf_offset),
                             v_vk_subbuffer(ctx, d_S, s_buf_offset),
                             v_vk_subbuffer(ctx, ctx->prealloc_split_k, 0),
                           },
                           // We only use split_k when group query attention is enabled, which means
                           // there's no more than one tile of rows (i.e. workgroups_x would have been
                           // one). We reuse workgroups_x to mean the number of splits, so we need to
                           // cancel out the divide by wg_denoms[0].
                           pc,
                           {workgroups_x * pipeline->wg_denoms[0], workgroups_y, workgroups_z});

    vk_sync_buffers(ctx, subctx);
    const std::array<uint32_t, 5> pc2 = {HSV, (uint32_t)ne1, (uint32_t)ne3, split_k, (sinks != nullptr)};
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           ctx->device->pipeline_flash_attn_split_k_reduce,
                           {
                             v_vk_subbuffer(ctx, ctx->prealloc_split_k, 0),
                             v_vk_subbuffer(ctx, d_S, s_buf_offset),
                             v_vk_subbuffer(ctx, d_D, d_buf_offset),
                           },
                           pc2,
                           {(uint32_t)ne1, HSV, (uint32_t)ne3});
    ctx->prealloc_split_k_need_sync = true;
  }
  else {
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             v_vk_subbuffer(ctx, d_Q, q_buf_offset),
                             v_vk_subbuffer(ctx, d_K, k_buf_offset),
                             v_vk_subbuffer(ctx, d_V, v_buf_offset),
                             v_vk_subbuffer(ctx, d_M, m_buf_offset),
                             v_vk_subbuffer(ctx, d_S, s_buf_offset),
                             v_vk_subbuffer(ctx, d_D, d_buf_offset),
                           },
                           pc,
                           {workgroups_x, workgroups_y, workgroups_z});
  }
}


