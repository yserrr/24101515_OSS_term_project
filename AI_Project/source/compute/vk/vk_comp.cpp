#include "vk_comp.hpp"

#include <vk_op_f32.hpp>


void v_vk_timestep_embedding(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                             v_tensor* dst, bool dryrun) {
  const uint32_t dim        = dst->op_params[0];
  const uint32_t max_period = dst->op_params[1];
  const uint32_t nb1        = dst->nb[1] / v_type_size(dst->type);

  v_vk_op_f32<vk_op_timestep_embedding_push_constants>(ctx, subctx,
                                                       src0, nullptr, nullptr,
                                                       dst,
                                                       v_OP_TIMESTEP_EMBEDDING,
                                                       {nb1, dim, max_period,}, dryrun);
}

void v_vk_add(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
              const v_tensor* src1, v_tensor* dst, bool dryrun) {
  const uint32_t src0_type_size = v_type_size(src0->type);
  const uint32_t src1_type_size = v_type_size(src1->type);
  const uint32_t dst_type_size  = v_type_size(dst->type);

  v_vk_op_f32<vk_op_binary_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           src1,
                                           nullptr,
                                           dst,
                                           v_OP_ADD,
                                           {
                                             (uint32_t)nelements(src0),
                                             (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
                                             (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size,
                                             (uint32_t)src0->nb[1] / src0_type_size,
                                             (uint32_t)src0->nb[2] / src0_type_size,
                                             (uint32_t)src0->nb[3] / src0_type_size,
                                             (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],
                                             (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size,
                                             (uint32_t)src1->nb[1] / src1_type_size,
                                             (uint32_t)src1->nb[2] / src1_type_size,
                                             (uint32_t)src1->nb[3] / src1_type_size,
                                             (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], (uint32_t)dst->ne[2],
                                             (uint32_t)dst->ne[3], (uint32_t)dst->nb[0] / dst_type_size,
                                             (uint32_t)dst->nb[1] / dst_type_size,
                                             (uint32_t)dst->nb[2] / dst_type_size,
                                             (uint32_t)dst->nb[3] / dst_type_size,
                                             0,
                                             0.0f, 0.0f, ctx->do_add_rms_partials,
                                           },
                                           dryrun);
}

void v_vk_sub(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
              const v_tensor* src1, v_tensor* dst, bool dryrun) {
  const uint32_t src0_type_size = v_type_size(src0->type);
  const uint32_t src1_type_size = v_type_size(src1->type);
  const uint32_t dst_type_size  = v_type_size(dst->type);

  v_vk_op_f32<vk_op_binary_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           src1,
                                           nullptr,
                                           dst,
                                           v_OP_SUB,
                                           {
                                             (uint32_t)nelements(src0),
                                             (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
                                             (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size,
                                             (uint32_t)src0->nb[1] / src0_type_size,
                                             (uint32_t)src0->nb[2] / src0_type_size,
                                             (uint32_t)src0->nb[3] / src0_type_size,
                                             (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],
                                             (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size,
                                             (uint32_t)src1->nb[1] / src1_type_size,
                                             (uint32_t)src1->nb[2] / src1_type_size,
                                             (uint32_t)src1->nb[3] / src1_type_size,
                                             (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], (uint32_t)dst->ne[2],
                                             (uint32_t)dst->ne[3], (uint32_t)dst->nb[0] / dst_type_size,
                                             (uint32_t)dst->nb[1] / dst_type_size,
                                             (uint32_t)dst->nb[2] / dst_type_size,
                                             (uint32_t)dst->nb[3] / dst_type_size,
                                             0,
                                             0.0f, 0.0f, 0,
                                           },
                                           dryrun);
}

void v_vk_mul(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
              const v_tensor* src1, v_tensor* dst, bool dryrun) {
  const uint32_t src0_type_size = v_type_size(src0->type);
  const uint32_t src1_type_size = v_type_size(src1->type);
  const uint32_t dst_type_size  = v_type_size(dst->type);

  v_vk_op_f32<vk_op_binary_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           src1,
                                           nullptr,
                                           dst,
                                           v_OP_MUL,
                                           {
                                             (uint32_t)nelements(src0),
                                             (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
                                             (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size,
                                             (uint32_t)src0->nb[1] / src0_type_size,
                                             (uint32_t)src0->nb[2] / src0_type_size,
                                             (uint32_t)src0->nb[3] / src0_type_size,
                                             (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],
                                             (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size,
                                             (uint32_t)src1->nb[1] / src1_type_size,
                                             (uint32_t)src1->nb[2] / src1_type_size,
                                             (uint32_t)src1->nb[3] / src1_type_size,
                                             (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], (uint32_t)dst->ne[2],
                                             (uint32_t)dst->ne[3], (uint32_t)dst->nb[0] / dst_type_size,
                                             (uint32_t)dst->nb[1] / dst_type_size,
                                             (uint32_t)dst->nb[2] / dst_type_size,
                                             (uint32_t)dst->nb[3] / dst_type_size,
                                             0,
                                             0.0f, 0.0f, 0,
                                           },
                                           dryrun);
}

void v_vk_div(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
              const v_tensor* src1, v_tensor* dst, bool dryrun) {
  const uint32_t src0_type_size = v_type_size(src0->type);
  const uint32_t src1_type_size = v_type_size(src1->type);
  const uint32_t dst_type_size  = v_type_size(dst->type);

  v_vk_op_f32<vk_op_binary_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           src1,
                                           nullptr,
                                           dst,
                                           v_OP_DIV,
                                           {
                                             (uint32_t)nelements(src0),
                                             (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
                                             (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size,
                                             (uint32_t)src0->nb[1] / src0_type_size,
                                             (uint32_t)src0->nb[2] / src0_type_size,
                                             (uint32_t)src0->nb[3] / src0_type_size,
                                             (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],
                                             (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size,
                                             (uint32_t)src1->nb[1] / src1_type_size,
                                             (uint32_t)src1->nb[2] / src1_type_size,
                                             (uint32_t)src1->nb[3] / src1_type_size,
                                             (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], (uint32_t)dst->ne[2],
                                             (uint32_t)dst->ne[3], (uint32_t)dst->nb[0] / dst_type_size,
                                             (uint32_t)dst->nb[1] / dst_type_size,
                                             (uint32_t)dst->nb[2] / dst_type_size,
                                             (uint32_t)dst->nb[3] / dst_type_size,
                                             0,
                                             0.0f, 0.0f, 0,
                                           },
                                           dryrun);
}

void v_vk_add_id(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                 const v_tensor* src1, const v_tensor* src2, v_tensor* dst, bool dryrun) {
  const uint32_t src0_type_size = v_type_size(src0->type);
  const uint32_t src1_type_size = v_type_size(src1->type);
  const uint32_t src2_type_size = v_type_size(src2->type);

  v_vk_op_f32<vk_op_add_id_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           src1,
                                           src2,
                                           dst,
                                           v_OP_ADD_ID,
                                           {
                                             (uint32_t)dst->ne[0],
                                             (uint32_t)dst->ne[1],
                                             (uint32_t)src0->nb[1] / src0_type_size,
                                             (uint32_t)src0->nb[2] / src0_type_size,
                                             (uint32_t)src1->nb[1] / src1_type_size,
                                             (uint32_t)src2->nb[1] / src2_type_size,
                                           },
                                           dryrun);
}

void v_vk_op_f32_wkv(vk_backend_ctx* ctx, vk_context& subctx, v_tensor* dst,
                     const vk_op_rwkv_wkv6_push_constants&& pc, int version, bool dryrun) {
  V_ASSERT(version == 6 || version == 7);
  int num_srcs = version == 6
                   ? 6
                   : 7;

  for (int i = 0; i < num_srcs; i++) { V_ASSERT(!v_is_quantized(dst->src[i]->type)); }

  V_ASSERT(dst->buffer != nullptr);

  vk_pipeline pipeline = v_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
  V_ASSERT(pipeline != nullptr);

  if (dryrun) {
    v_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    return;
  }

  v_backend_vk_buffer_ctx* dst_buf_ctx     = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  v_backend_vk_buffer_ctx* src_buf_ctxs[7] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  for (int i = 0; i < num_srcs; i++) { src_buf_ctxs[i] = (v_backend_vk_buffer_ctx*)dst->src[i]->buffer->context; }

  vk_buffer d_D     = nullptr, d_srcs[7] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  size_t dst_offset = 0, src_offsets[7]  = {0, 0, 0, 0, 0, 0, 0};
  bool dst_uma      = false, srcs_uma[7] = {false, false, false, false, false, false, false};

  if (ctx->device->uma) {
    for (int i = 0; i < num_srcs; i++) {
      vk_get_host_buffer(ctx->device, dst->src[i]->data, d_srcs[i], src_offsets[i]);
      srcs_uma[i] = d_srcs[i] != nullptr;
    }

    vk_get_host_buffer(ctx->device, dst->data, d_D, dst_offset);
    dst_uma = d_D != nullptr;
  }

  uint64_t src_sizes[7] = {0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < num_srcs; i++) {
    src_sizes[i] = num_bytes(dst->src[i]);
    if (!srcs_uma[i]) {
      d_srcs[i]      = src_buf_ctxs[i]->dev_buffer;
      src_offsets[i] = vk_tensor_offset(dst->src[i]) + dst->src[i]->view_offs;
    }
  }

  const uint64_t dst_size = num_bytes(dst);
  if (!dst_uma) {
    d_D        = dst_buf_ctx->dev_buffer;
    dst_offset = vk_tensor_offset(dst) + dst->view_offs;
  }

  std::array<uint32_t, 3> elements = {
    (uint32_t)(pc.B * pc.H),
    1,
    1
  };

  if (version == 6) {
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             vk_sub_buffer{d_srcs[0], src_offsets[0], src_sizes[0]},
                             vk_sub_buffer{d_srcs[1], src_offsets[1], src_sizes[1]},
                             vk_sub_buffer{d_srcs[2], src_offsets[2], src_sizes[2]},
                             vk_sub_buffer{d_srcs[3], src_offsets[3], src_sizes[3]},
                             vk_sub_buffer{d_srcs[4], src_offsets[4], src_sizes[4]},
                             vk_sub_buffer{d_srcs[5], src_offsets[5], src_sizes[5]},
                             vk_sub_buffer{d_D, dst_offset, dst_size}
                           },
                           pc,
                           elements);
  }
  else if (version == 7) {
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             vk_sub_buffer{d_srcs[0], src_offsets[0], src_sizes[0]},
                             vk_sub_buffer{d_srcs[1], src_offsets[1], src_sizes[1]},
                             vk_sub_buffer{d_srcs[2], src_offsets[2], src_sizes[2]},
                             vk_sub_buffer{d_srcs[3], src_offsets[3], src_sizes[3]},
                             vk_sub_buffer{d_srcs[4], src_offsets[4], src_sizes[4]},
                             vk_sub_buffer{d_srcs[5], src_offsets[5], src_sizes[5]},
                             vk_sub_buffer{d_srcs[6], src_offsets[6], src_sizes[6]},
                             vk_sub_buffer{d_D, dst_offset, dst_size}
                           },
                           pc,
                           elements);
  }
  else {
    // shouldn't happen
    V_ASSERT(false);
  }
}

void v_vk_rwkv_wkv6(vk_backend_ctx* ctx, vk_context& subctx, v_tensor* dst, bool dryrun) {
  const size_t seq_length = dst->src[0]->ne[2];
  const size_t n_embed    = dst->ne[0];
  const size_t n_heads    = dst->src[0]->ne[1];
  const size_t n_seqs     = dst->src[5]->ne[1];

  v_vk_op_f32_wkv(
    ctx,
    subctx,
    dst,
    {
      (uint32_t)n_seqs,
      (uint32_t)seq_length,
      (uint32_t)n_embed,
      (uint32_t)n_heads,
    },
    6,
    dryrun
  );
}

void v_vk_rwkv_wkv7(vk_backend_ctx* ctx, vk_context& subctx, v_tensor* dst, bool dryrun) {
  const size_t seq_length = dst->src[0]->ne[2];
  const size_t n_embed    = dst->ne[0];
  const size_t n_heads    = dst->src[0]->ne[1];
  const size_t n_seqs     = dst->src[6]->ne[1];

  v_vk_op_f32_wkv(
    ctx,
    subctx,
    dst,
    {
      (uint32_t)n_seqs,
      (uint32_t)seq_length,
      (uint32_t)n_embed,
      (uint32_t)n_heads,
    },
    7,
    dryrun
  );
}

void v_vk_ssm_scan(vk_backend_ctx* ctx, vk_context& subctx, v_tensor* dst, bool dryrun) {
  const v_tensor* src0 = dst->src[0];
  const v_tensor* src1 = dst->src[1];
  const v_tensor* src2 = dst->src[2];
  const v_tensor* src3 = dst->src[3];
  const v_tensor* src4 = dst->src[4];
  const v_tensor* src5 = dst->src[5];

  V_ASSERT(dst->buffer != nullptr);

  const uint32_t head_dim = src0->ne[1];
  const uint32_t n_head   = src1->ne[1];
  const uint32_t n_group  = src4->ne[1];
  const uint32_t n_tok    = src1->ne[2];
  const uint32_t n_seq    = src1->ne[3];

  bool is_mamba2 = (src3->nb[1] == sizeof(float));
  V_ASSERT(is_mamba2);

  vk_pipeline pipeline = v_vk_op_get_pipeline(ctx, src0, src1, src2, dst, dst->op);
  V_ASSERT(pipeline != nullptr);

  if (dryrun) {
    v_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    return;
  }

  const int64_t s_off = nelements(src1) * sizeof(float);

  const vk_op_ssm_scan_push_constants pc = {
    (uint32_t)src0->nb[2], (uint32_t)src0->nb[3],
    (uint32_t)src1->nb[2], (uint32_t)src1->nb[3],
    (uint32_t)src2->nb[1], (uint32_t)src2->nb[2],
    (uint32_t)src3->nb[1],
    (uint32_t)src4->nb[2], (uint32_t)src4->nb[3],
    (uint32_t)src5->nb[2], (uint32_t)src5->nb[3],
    (uint32_t)s_off,
    n_head, head_dim, n_group, n_tok
  };

  v_backend_vk_buffer_ctx* dst_buf_ctx = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  v_backend_vk_buffer_ctx* src_buf_ctxs[V_MAX_SRC];
  for (int i = 0; i < V_MAX_SRC && dst->src[i] != nullptr; i++) { src_buf_ctxs[i] = (v_backend_vk_buffer_ctx*)dst->src[i]->buffer->context; }

  vk_buffer d_D     = nullptr, d_srcs[V_MAX_SRC] = {nullptr};
  size_t dst_offset = 0, src_offsets[V_MAX_SRC]  = {0};
  bool dst_uma      = false, srcs_uma[V_MAX_SRC] = {false};

  if (ctx->device->uma) {
    for (int i = 0; i < V_MAX_SRC && dst->src[i] != nullptr; i++) {
      vk_get_host_buffer(ctx->device, dst->src[i]->data, d_srcs[i], src_offsets[i]);
      srcs_uma[i] = d_srcs[i] != nullptr;
    }
    vk_get_host_buffer(ctx->device, dst->data, d_D, dst_offset);
    dst_uma = d_D != nullptr;
  }

  if (!dst_uma) {
    d_D        = dst_buf_ctx->dev_buffer;
    dst_offset = vk_tensor_offset(dst) + dst->view_offs;
  }
  for (int i = 0; i < V_MAX_SRC && dst->src[i] != nullptr; i++) {
    if (!srcs_uma[i]) {
      d_srcs[i]      = src_buf_ctxs[i]->dev_buffer;
      src_offsets[i] = vk_tensor_offset(dst->src[i]) + dst->src[i]->view_offs;
    }
  }

  size_t dst_size = num_bytes(dst);
  size_t src_sizes[V_MAX_SRC];
  for (int i = 0; i < V_MAX_SRC && dst->src[i] != nullptr; i++) { src_sizes[i] = num_bytes(dst->src[i]); }

  std::array<uint32_t, 3> elements;

  const int splitH                = 16;
  const uint32_t num_workgroups_x = CEIL_DIV(n_head * head_dim, splitH);
  const uint32_t num_workgroups_y = n_seq;
  elements                        = {num_workgroups_x, num_workgroups_y, 1};

  v_vk_dispatch_pipeline(ctx,
                         subctx,
                         pipeline,
                         {
                           vk_sub_buffer{d_srcs[0], src_offsets[0], src_sizes[0]},
                           vk_sub_buffer{d_srcs[1], src_offsets[1], src_sizes[1]},
                           vk_sub_buffer{d_srcs[2], src_offsets[2], src_sizes[2]},
                           vk_sub_buffer{d_srcs[3], src_offsets[3], src_sizes[3]},
                           vk_sub_buffer{d_srcs[4], src_offsets[4], src_sizes[4]},
                           vk_sub_buffer{d_srcs[5], src_offsets[5], src_sizes[5]},
                           vk_sub_buffer{d_srcs[6], src_offsets[6], src_sizes[6]},
                           vk_sub_buffer{d_D, dst_offset, dst_size}
                         },
                         pc,
                         elements);
}

void v_vk_ssm_conv(vk_backend_ctx* ctx, vk_context& subctx, v_tensor* dst, bool dryrun) {
  const v_tensor* src0 = dst->src[0];
  const v_tensor* src1 = dst->src[1];

  v_vk_op_f32<vk_op_ssm_conv_push_constants>(ctx,
                                             subctx,
                                             src0,
                                             src1,
                                             nullptr,
                                             dst,
                                             v_OP_SSM_CONV,
                                             {
                                               (uint32_t)src0->nb[1], (uint32_t)src0->nb[2],
                                               (uint32_t)src1->nb[1],
                                               (uint32_t)dst->nb[0], (uint32_t)dst->nb[1], (uint32_t)dst->nb[2],
                                               (uint32_t)src1->ne[0],
                                               (uint32_t)src0->ne[0],
                                               (uint32_t)src0->ne[1],
                                               (uint32_t)dst->ne[1],
                                               (uint32_t)dst->ne[2],
                                             },
                                             dryrun);
}

void v_vk_op_f32_opt_step_adamw(vk_backend_ctx* ctx, vk_context& subctx, v_tensor* dst,
                                const vk_op_push_constants&& pc, bool dryrun) {
  const v_tensor* x  = dst->src[0];
  const v_tensor* g  = dst->src[1];
  const v_tensor* gm = dst->src[2];
  const v_tensor* gv = dst->src[3];
  const v_tensor* p  = dst->src[4];

  V_ASSERT(x->type == v_TYPE_F32);
  V_ASSERT(g->type == v_TYPE_F32);
  V_ASSERT(gm->type == v_TYPE_F32);
  V_ASSERT(gv->type == v_TYPE_F32);
  V_ASSERT(p->type == v_TYPE_F32);
  V_ASSERT(dst->buffer != nullptr);
  V_ASSERT(v_is_contiguous(x));
  V_ASSERT(v_is_contiguous(g));
  V_ASSERT(v_is_contiguous(gm));
  V_ASSERT(v_is_contiguous(gv));
  V_ASSERT(v_is_contiguous(p));
  V_ASSERT(v_are_same_shape(x, g));
  V_ASSERT(v_are_same_shape(x, gm));
  V_ASSERT(v_are_same_shape(x, gv));
  V_ASSERT(nelements(p) == 7);

  vk_pipeline pipeline = v_vk_op_get_pipeline(ctx, g, gm, gv, dst, v_OP_OPT_STEP_ADAMW);
  V_ASSERT(pipeline != nullptr);

  if (dryrun) {
    v_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    return;
  }

  v_backend_vk_buffer_ctx* x_buf_ctx  = (v_backend_vk_buffer_ctx*)x->buffer->context;
  v_backend_vk_buffer_ctx* g_buf_ctx  = (v_backend_vk_buffer_ctx*)g->buffer->context;
  v_backend_vk_buffer_ctx* gm_buf_ctx = (v_backend_vk_buffer_ctx*)gm->buffer->context;
  v_backend_vk_buffer_ctx* gv_buf_ctx = (v_backend_vk_buffer_ctx*)gv->buffer->context;
  v_backend_vk_buffer_ctx* p_buf_ctx  = (v_backend_vk_buffer_ctx*)p->buffer->context;

  vk_buffer d_X   = nullptr, d_G = nullptr, d_GM = nullptr, d_GV = nullptr, d_P = nullptr;
  size_t x_offset = 0, g_offset  = 0, gm_offset  = 0, gv_offset  = 0, p_offset  = 0;
  bool X_uma      = false, G_uma = false, GM_uma = false, GV_uma = false, P_uma = false;

  if (ctx->device->uma) {
    vk_get_host_buffer(ctx->device, x->data, d_X, x_offset);
    vk_get_host_buffer(ctx->device, g->data, d_G, g_offset);
    vk_get_host_buffer(ctx->device, gm->data, d_GM, gm_offset);
    vk_get_host_buffer(ctx->device, gv->data, d_GV, gv_offset);
    vk_get_host_buffer(ctx->device, p->data, d_P, p_offset);

    X_uma  = d_X != nullptr;
    G_uma  = d_G != nullptr;
    GM_uma = d_GM != nullptr;
    GV_uma = d_GV != nullptr;
    P_uma  = d_P != nullptr;
  }

  if (!X_uma) {
    d_X      = x_buf_ctx->dev_buffer;
    x_offset = vk_tensor_offset(x) + x->view_offs;
  }
  if (!G_uma) {
    d_G      = g_buf_ctx->dev_buffer;
    g_offset = vk_tensor_offset(g) + g->view_offs;
  }
  if (!GM_uma) {
    d_GM      = gm_buf_ctx->dev_buffer;
    gm_offset = vk_tensor_offset(gm) + gm->view_offs;
  }
  if (!GV_uma) {
    d_GV      = gv_buf_ctx->dev_buffer;
    gv_offset = vk_tensor_offset(gv) + gv->view_offs;
  }
  if (!P_uma) {
    d_P      = p_buf_ctx->dev_buffer;
    p_offset = vk_tensor_offset(p) + p->view_offs;
  }

  const uint64_t x_size  = num_bytes(x);
  const uint64_t g_size  = num_bytes(g);
  const uint64_t gm_size = num_bytes(gm);
  const uint64_t gv_size = num_bytes(gv);
  const uint64_t p_size  = num_bytes(p);

  std::array<uint32_t, 3> elements = {(uint32_t)nelements(x), 1, 1};

  v_vk_dispatch_pipeline(ctx,
                         subctx,
                         pipeline,
                         {
                           vk_sub_buffer{d_X, x_offset, x_size},
                           vk_sub_buffer{d_G, g_offset, g_size},
                           vk_sub_buffer{d_GM, gm_offset, gm_size},
                           vk_sub_buffer{d_GV, gv_offset, gv_size},
                           vk_sub_buffer{d_P, p_offset, p_size},
                         },
                         pc,
                         elements);
}

void v_vk_opt_step_adamw(vk_backend_ctx* ctx, vk_context& subctx, v_tensor* dst,
                         bool dryrun) {
  const size_t n = nelements(dst->src[0]);

  v_vk_op_f32_opt_step_adamw(
    ctx,
    subctx,
    dst,
    {(uint32_t)n, 0, 0.0f, 0.0f},
    dryrun
  );
}

void v_vk_opt_step_sgd(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                       const v_tensor* src1, const v_tensor* src2, v_tensor* dst,
                       bool dryrun) {
  const size_t n = nelements(dst->src[0]);

  v_vk_op_f32<vk_op_push_constants>(ctx,
                                    subctx,
                                    src0,
                                    src1,
                                    src2,
                                    dst,
                                    v_OP_OPT_STEP_SGD,
                                    {(uint32_t)n, 0, 0.0f, 0.0f},
                                    dryrun);
}

void v_vk_concat(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                 const v_tensor* src1, v_tensor* dst, bool dryrun) {
  int* op_params = dst->op_params.data();

  const uint32_t src0_type_size = v_type_size(src0->type);
  const uint32_t src1_type_size = v_type_size(src1->type);
  const uint32_t dst_type_size  = v_type_size(dst->type);

  v_vk_op_f32<vk_op_binary_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           src1,
                                           nullptr,
                                           dst,
                                           v_OP_CONCAT,
                                           {
                                             (uint32_t)nelements(dst),
                                             (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
                                             (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size,
                                             (uint32_t)src0->nb[1] / src0_type_size,
                                             (uint32_t)src0->nb[2] / src0_type_size,
                                             (uint32_t)src0->nb[3] / src0_type_size,
                                             (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],
                                             (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size,
                                             (uint32_t)src1->nb[1] / src1_type_size,
                                             (uint32_t)src1->nb[2] / src1_type_size,
                                             (uint32_t)src1->nb[3] / src1_type_size,
                                             (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], (uint32_t)dst->ne[2],
                                             (uint32_t)dst->ne[3], (uint32_t)dst->nb[0] / dst_type_size,
                                             (uint32_t)dst->nb[1] / dst_type_size,
                                             (uint32_t)dst->nb[2] / dst_type_size,
                                             (uint32_t)dst->nb[3] / dst_type_size,
                                             0,
                                             0.0f, 0.0f, op_params[0],
                                           },
                                           dryrun);
}

void v_vk_upscale(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
                  bool dryrun) {
  const uint32_t src0_type_size = v_type_size(src0->type);
  const uint32_t mode           = (uint32_t)v_get_op_params_i32(dst, 0);

  v_TENSOR_UNARY_OP_LOCALS

  float sf0          = (float)ne0 / ne00;
  float sf1          = (float)ne1 / ne01;
  float sf2          = (float)ne2 / ne02;
  float sf3          = (float)ne3 / ne03;
  float pixel_offset = 0.5f;

  if (mode & v_SCALE_FLAG_ALIGN_CORNERS) {
    sf0 = ne0 > 1 && ne00 > 1
            ? (float)(ne0 - 1) / (ne00 - 1)
            : sf0;
    sf1 = ne1 > 1 && ne01 > 1
            ? (float)(ne1 - 1) / (ne01 - 1)
            : sf1;
    pixel_offset = 0.0f;
  }

  v_vk_op_f32<vk_op_upscale_push_constants>(ctx,
                                            subctx,
                                            src0,
                                            nullptr,
                                            nullptr,
                                            dst,
                                            v_OP_UPSCALE,
                                            {
                                              (uint32_t)nelements(dst), 0, 0,
                                              (uint32_t)ne00, (uint32_t)ne01,
                                              (uint32_t)nb00 / src0_type_size, (uint32_t)nb01 / src0_type_size,
                                              (uint32_t)nb02 / src0_type_size, (uint32_t)nb03 / src0_type_size,
                                              (uint32_t)ne0, (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
                                              sf0, sf1, sf2, sf3, pixel_offset
                                            },
                                            dryrun);
}

void v_vk_scale(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
                bool dryrun) {
  vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst);
  p.param1                     = v_get_op_params_f32(dst, 0);
  p.param2                     = v_get_op_params_f32(dst, 1);

  v_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, V_OP_SCALE, std::move(p), dryrun);
}

void v_vk_sqr(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
              bool dryrun) {
  v_vk_op_f32(ctx,
              subctx,
              src0,
              nullptr,
              nullptr,
              dst,
              v_OP_SQR,
              vk_op_unary_push_constants_init(src0, dst),
              dryrun);
}

void v_vk_sqrt(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
               bool dryrun) {
  v_vk_op_f32(ctx,
              subctx,
              src0,
              nullptr,
              nullptr,
              dst,
              v_OP_SQRT,
              vk_op_unary_push_constants_init(src0, dst),
              dryrun);
}

void v_vk_sin(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
              bool dryrun) {
  v_vk_op_f32(ctx,
              subctx,
              src0,
              nullptr,
              nullptr,
              dst,
              v_OP_SIN,
              vk_op_unary_push_constants_init(src0, dst),
              dryrun);
}

void v_vk_cos(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
              bool dryrun) {
  v_vk_op_f32(ctx,
              subctx,
              src0,
              nullptr,
              nullptr,
              dst,
              v_OP_COS,
              vk_op_unary_push_constants_init(src0, dst),
              dryrun);
}

void v_vk_clamp(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
                bool dryrun) {
  vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst);
  p.param1                     = v_get_op_params_f32(dst, 0);
  p.param2                     = v_get_op_params_f32(dst, 1);

  v_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, v_OP_CLAMP, std::move(p), dryrun);
}

void v_vk_pad(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
              bool dryrun) {
  vk_op_pad_push_constants p = vk_op_pad_push_constants_init(src0, dst);
  v_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, v_OP_PAD, std::move(p), dryrun);
}

void v_vk_roll(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
               bool dryrun) {
  const int32_t s0          = v_get_op_params_i32(dst, 0);
  const int32_t s1          = v_get_op_params_i32(dst, 1);
  const int32_t s2          = v_get_op_params_i32(dst, 2);
  const int32_t s3          = v_get_op_params_i32(dst, 3);
  const uint32_t s01_packed = ((s0 + 0x8000) << 16) | (s1 + 0x8000);
  const uint32_t s23_packed = ((s2 + 0x8000) << 16) | (s3 + 0x8000);

  vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst);
  memcpy(&p.param1, &s01_packed, sizeof(float));
  memcpy(&p.param2, &s23_packed, sizeof(float));

  v_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, v_OP_ROLL, std::move(p), dryrun);
}

void v_vk_repeat(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst, bool dryrun) {
  vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst, nelements(dst));
  v_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, v_OP_REPEAT, std::move(p), dryrun);
}

void v_vk_repeat_back(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst, bool dryrun) {
  vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst, nelements(dst));
  v_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, v_OP_REPEAT_BACK, std::move(p), dryrun);
}


void v_vk_set_rows(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                   const v_tensor* src1, v_tensor* dst, bool dryrun) {
  const uint32_t src0_type_size = v_type_size(src0->type);
  const uint32_t src1_type_size = v_type_size(src1->type);
  const uint32_t dst_type_size  = v_type_size(dst->type);

  // Skip empty skip_rows operations. For most ops the empty check at the start
  // of v_vk_build_graph is sufficient, but set_rows can have a nonempty dst
  // with empty srcs.
  if (is_empty(src0) || is_empty(src1)) { return; }

  v_vk_op_f32<vk_op_binary_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           src1,
                                           nullptr,
                                           dst,
                                           v_OP_SET_ROWS,
                                           {
                                             (uint32_t)nelements(src0),
                                             (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
                                             (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size,
                                             (uint32_t)src0->nb[1] / src0_type_size,
                                             (uint32_t)src0->nb[2] / src0_type_size,
                                             (uint32_t)src0->nb[3] / src0_type_size,
                                             (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],
                                             (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size,
                                             (uint32_t)src1->nb[1] / src1_type_size,
                                             (uint32_t)src1->nb[2] / src1_type_size,
                                             (uint32_t)src1->nb[3] / src1_type_size,
                                             (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], (uint32_t)dst->ne[2],
                                             (uint32_t)dst->ne[3], (uint32_t)dst->nb[0] / dst_type_size,
                                             (uint32_t)dst->nb[1] / dst_type_size,
                                             (uint32_t)dst->nb[2] / dst_type_size,
                                             (uint32_t)dst->nb[3] / dst_type_size,
                                             0,
                                             0.0f, 0.0f, 0,
                                           },
                                           dryrun);
}

void v_vk_silu_back(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                    const v_tensor* src1, v_tensor* dst, bool dryrun) {
  v_vk_op_f32<vk_op_push_constants>(ctx,
                                    subctx,
                                    src0,
                                    src1,
                                    nullptr,
                                    dst,
                                    v_OP_SILU_BACK,
                                    {(uint32_t)nelements(src0), 0, 0.0f, 0.0f},
                                    dryrun);
}

void v_vk_norm(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
               bool dryrun) {
  float* op_params = (float*)dst->op_params.data();

  v_vk_op_f32<vk_op_push_constants>(ctx,
                                    subctx,
                                    src0,
                                    nullptr,
                                    nullptr,
                                    dst,
                                    v_OP_NORM,
                                    {(uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0], 0.0f},
                                    dryrun);
}

void v_vk_group_norm(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                     v_tensor* dst, bool dryrun) {
  const int* int_op_params     = dst->op_params.data();
  const float* float_op_params = reinterpret_cast<const float*>(dst->op_params.data());

  const uint32_t num_groups = int_op_params[0];
  const float eps           = float_op_params[1];
  const uint32_t group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);

  v_vk_op_f32<vk_op_push_constants>(ctx,
                                    subctx,
                                    src0,
                                    nullptr,
                                    nullptr,
                                    dst,
                                    v_OP_GROUP_NORM,
                                    {group_size, 0, eps, 0.0f},
                                    dryrun);
}

uint32_t v_vk_rms_num_partials(vk_backend_ctx* ctx, const v_tensor* node) {
  const uint32_t ne           = (uint32_t)node->ne[0];
  const uint32_t denom        = ctx->device->pipeline_add_rms[0][0][0]->wg_denoms[0];
  const uint32_t num_partials = CEIL_DIV(ne, denom);
  return num_partials;
}

uint32_t v_vk_rms_partials_size(vk_backend_ctx* ctx, const v_tensor* node) {
  const uint32_t num_partials = v_vk_rms_num_partials(ctx, node);
  const uint32_t num_bytes    = ROUNDUP_POW2(num_partials * sizeof(uint32_t), ctx->device->partials_binding_alignment);
  return num_bytes;
}

void v_vk_rms_norm(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                   const v_tensor* src1, v_tensor* dst, float* op_params, bool dryrun) {
  const uint32_t src0_type_size = v_type_size(src0->type);
  const uint32_t src1_type_size = v_type_size(src1->type);
  const uint32_t dst_type_size  = v_type_size(dst->type);

  uint32_t param3 = ctx->do_add_rms_partials
                      ? v_vk_rms_num_partials(ctx, dst)
                      : 0;

  v_vk_op_f32<vk_op_binary_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           src1,
                                           nullptr,
                                           dst,
                                           v_OP_RMS_NORM,
                                           {
                                             (uint32_t)nelements(src0),
                                             (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
                                             (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size,
                                             (uint32_t)src0->nb[1] / src0_type_size,
                                             (uint32_t)src0->nb[2] / src0_type_size,
                                             (uint32_t)src0->nb[3] / src0_type_size,
                                             (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],
                                             (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size,
                                             (uint32_t)src1->nb[1] / src1_type_size,
                                             (uint32_t)src1->nb[2] / src1_type_size,
                                             (uint32_t)src1->nb[3] / src1_type_size,
                                             (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], (uint32_t)dst->ne[2],
                                             (uint32_t)dst->ne[3], (uint32_t)dst->nb[0] / dst_type_size,
                                             (uint32_t)dst->nb[1] / dst_type_size,
                                             (uint32_t)dst->nb[2] / dst_type_size,
                                             (uint32_t)dst->nb[3] / dst_type_size,
                                             0,
                                             op_params[0], 0.0f, (int32_t)param3,
                                           },
                                           dryrun);

  if (ctx->do_add_rms_partials) {
    ctx->prealloc_size_add_rms_partials_offset += v_vk_rms_partials_size(ctx, src0);
    ctx->do_add_rms_partials = false;
  }
}

void v_vk_rms_norm_back(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                        const v_tensor* src1, v_tensor* dst, bool dryrun) {
  float* op_params = (float*)dst->op_params.data();
  v_vk_op_f32<vk_op_push_constants>(ctx,
                                    subctx,
                                    src0,
                                    src1,
                                    nullptr,
                                    dst,
                                    v_OP_RMS_NORM_BACK,
                                    {(uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0], 0.0f},
                                    dryrun);
}

void v_vk_l2_norm(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
                  bool dryrun) {
  float* op_params = reinterpret_cast<float*>(dst->op_params.data());
  v_vk_op_f32<vk_op_push_constants>(ctx,
                                    subctx,
                                    src0,
                                    nullptr,
                                    nullptr,
                                    dst,
                                    v_OP_L2_NORM,
                                    {(uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0], 0.0f},
                                    dryrun);
}

void v_vk_unary(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
                bool dryrun) {
  v_vk_op_f32<vk_op_push_constants>(ctx,
                                    subctx,
                                    src0,
                                    nullptr,
                                    nullptr,
                                    dst,
                                    v_OP_UNARY,
                                    {(uint32_t)nelements(src0), 0, 0.0f, 0.0f},
                                    dryrun);
}

void v_vk_glu(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
              const v_tensor* src1, v_tensor* dst, bool dryrun) {
  const float* op_params_f = reinterpret_cast<const float*>(dst->op_params.data());

  const bool swapped = (bool)dst->op_params[1];
  const bool split   = src1 != nullptr;
  const float alpha  = op_params_f[2];
  const float limit  = op_params_f[3];

  V_ASSERT(v_is_contiguous(src0));

  if (!split) { V_ASSERT(src0->ne[0] / 2 == dst->ne[0]); }
  else {
    V_ASSERT(src0->ne[0] == src1->ne[0]);
    V_ASSERT(src0->ne[0] == dst->ne[0]);
    V_ASSERT(src0->type == src1->type);
  }

  const uint32_t mode = split
                          ? 2
                          : (swapped
                               ? 1
                               : 0);

  v_vk_op_f32<vk_op_glu_push_constants>(ctx,
                                        subctx,
                                        src0,
                                        src1,
                                        nullptr,
                                        dst,
                                        v_OP_GLU,
                                        {
                                          (uint32_t)nelements(dst),
                                          (uint32_t)src0->ne[0],
                                          (uint32_t)dst->ne[0],
                                          mode,
                                          alpha,
                                          limit
                                        },
                                        dryrun);
}

void v_vk_diag_mask_inf(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                        v_tensor* dst, bool dryrun) {
  int32_t* op_params = dst->op_params.data();
  v_vk_op_f32<vk_op_diag_mask_push_constants>(ctx,
                                              subctx,
                                              src0,
                                              nullptr,
                                              nullptr,
                                              dst,
                                              V_OP_DIAG_MASK_INF,
                                              {(uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0]},
                                              dryrun);
}

void v_vk_soft_max(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                   const v_tensor* src1, const v_tensor* src2, v_tensor* dst, bool dryrun) {
  float* op_params = reinterpret_cast<float*>(dst->op_params.data());

  float scale    = op_params[0];
  float max_bias = op_params[1];

  const uint32_t ncols   = (uint32_t)src0->ne[0];
  const uint32_t nrows_x = (uint32_t)v_nrows(src0);
  const uint32_t nrows_y = (uint32_t)src0->ne[1];

  const uint32_t ne12 = src1
                          ? (uint32_t)(src1->ne[2])
                          : 0u;
  const uint32_t ne13 = src1
                          ? (uint32_t)(src1->ne[3])
                          : 0u;
  const uint32_t nb11 = src1
                          ? (uint32_t)(src1->nb[1] / src1->nb[0])
                          : 0u;
  const uint32_t nb12 = src1
                          ? (uint32_t)(src1->nb[2] / src1->nb[0])
                          : 0u;
  const uint32_t nb13 = src1
                          ? (uint32_t)(src1->nb[3] / src1->nb[0])
                          : 0u;

  const uint32_t n_head_kv   = src0->ne[2];
  const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head_kv));

  const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
  const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

  v_vk_op_f32<vk_op_soft_max_push_constants>(ctx,
                                             subctx,
                                             src0,
                                             src1,
                                             src2,
                                             dst,
                                             V_OP_SOFT_MAX,
                                             {
                                               ncols,
                                               src1 != nullptr
                                                 ? nrows_y
                                                 : (uint32_t)0,
                                               (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
                                               ne12, ne13,
                                               nb11, nb12, nb13,
                                               scale, max_bias,
                                               m0, m1,
                                               n_head_log2,
                                               nrows_x,
                                               src2 != nullptr
                                             },
                                             dryrun);
}

void v_vk_soft_max_back(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                        const v_tensor* src1, v_tensor* dst, bool dryrun) {
  float* op_params = reinterpret_cast<float*>(dst->op_params.data());
  v_vk_op_f32<vk_op_push_constants>(ctx,
                                    subctx,
                                    src0,
                                    src1,
                                    nullptr,
                                    dst,
                                    v_OP_SOFT_MAX_BACK,
                                    {
                                      (uint32_t)src0->ne[0], (uint32_t)v_nrows(src0), op_params[0], op_params[1]
                                    },
                                    dryrun);
}

void v_vk_topk_moe(vk_backend_ctx* ctx, vk_context& subctx, v_cgraph* cgraph, int node_idx,
                   bool dryrun) {
  bool with_norm    = ctx->num_additional_fused_ops == topk_moe_norm.size() - 1;
  v_tensor* logits  = cgraph->nodes[node_idx + 0]->src[0];
  v_tensor* weights = with_norm
                        ? cgraph->nodes[node_idx + 8]
                        : cgraph->nodes[node_idx + 4];
  v_tensor* ids = cgraph->nodes[node_idx + 3];

  V_ASSERT(logits->type == v_TYPE_F32);
  V_ASSERT(weights->type == v_TYPE_F32);
  V_ASSERT(ids->type == v_TYPE_I32);

  const int n_experts     = logits->ne[0];
  const int n_rows        = logits->ne[1];
  const int n_expert_used = weights->ne[1];

  V_ASSERT(ids->nb[1] / v_type_size(ids->type) == (size_t) n_experts);

  vk_pipeline pipeline = v_vk_op_get_pipeline(ctx,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              cgraph->nodes[node_idx],
                                              V_OP_SOFT_MAX);

  if (dryrun) {
    v_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    return;
  }

  v_backend_vk_buffer_ctx* logits_buf_ctx  = (v_backend_vk_buffer_ctx*)logits->buffer->context;
  v_backend_vk_buffer_ctx* weights_buf_ctx = (v_backend_vk_buffer_ctx*)weights->buffer->context;
  v_backend_vk_buffer_ctx* ids_buf_ctx     = (v_backend_vk_buffer_ctx*)ids->buffer->context;

  vk_buffer d_logits        = nullptr;
  size_t logits_buf_offset  = 0;
  vk_buffer d_weights       = nullptr;
  size_t weights_buf_offset = 0;
  vk_buffer d_ids           = nullptr;
  size_t ids_buf_offset     = 0;

  bool logits_uma  = false;
  bool weights_uma = false;
  bool ids_uma     = false;

  if (ctx->device->uma) {
    vk_get_host_buffer(ctx->device, logits->data, d_logits, logits_buf_offset);
    vk_get_host_buffer(ctx->device, weights->data, d_weights, weights_buf_offset);
    vk_get_host_buffer(ctx->device, ids->data, d_ids, ids_buf_offset);
    logits_uma  = d_logits != nullptr;
    weights_uma = d_weights != nullptr;
    ids_uma     = d_ids != nullptr;
  }

  if (!logits_uma) {
    d_logits          = logits_buf_ctx->dev_buffer;
    logits_buf_offset = vk_tensor_offset(logits) + logits->view_offs;
    V_ASSERT(d_logits != nullptr);
  }
  if (!weights_uma) {
    d_weights          = weights_buf_ctx->dev_buffer;
    weights_buf_offset = vk_tensor_offset(weights) + weights->view_offs;
    V_ASSERT(d_weights != nullptr);
  }
  if (!ids_uma) {
    d_ids          = ids_buf_ctx->dev_buffer;
    ids_buf_offset = vk_tensor_offset(ids) + ids->view_offs;
    V_ASSERT(d_ids != nullptr);
  }

  vk_op_topk_moe_push_constants pc;
  pc.n_rows        = n_rows;
  pc.n_expert_used = n_expert_used;

  V_ASSERT(n_expert_used <= n_experts);

  const uint32_t rows_per_block    = 4;
  std::array<uint32_t, 3> elements = {CEIL_DIV(n_rows, rows_per_block), 1, 1};

  v_vk_dispatch_pipeline(ctx,
                         subctx,
                         pipeline,
                         {
                           v_vk_subbuffer(ctx, d_logits, logits_buf_offset),
                           v_vk_subbuffer(ctx, d_weights, weights_buf_offset),
                           v_vk_subbuffer(ctx, d_ids, ids_buf_offset),
                         },
                         pc,
                         elements);
}

void v_vk_rope(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
               const v_tensor* src1, const v_tensor* src2, v_tensor* dst, bool backprop,
               bool dryrun) {
  const int n_dims = dst->op_params[1];
  const int mode   = dst->op_params[2];
  // const int n_ctx         = ((int32_t *) dst->op_params)[3];
  const int n_ctx_orig    = dst->op_params[4];
  const float freq_base   = ((float*)dst->op_params.data())[5];
  const float freq_scale  = ((float*)dst->op_params.data())[6];
  const float ext_factor  = ((float*)dst->op_params.data())[7];
  const float attn_factor = ((float*)dst->op_params.data())[8];
  const float beta_fast   = ((float*)dst->op_params.data())[9];
  const float beta_slow   = ((float*)dst->op_params.data())[10];
  int sections[4]{};
  if (mode & v_ROPE_TYPE_MROPE) { memcpy(sections, dst->op_params.data() + 11, sizeof(int) * 4); }

  float corr_dims[2];
  v_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

  const float theta_scale = powf(freq_base, -2.0f / n_dims);

  uint32_t s1 = src0->nb[1] / v_type_size(src0->type);
  uint32_t s2 = src0->nb[2] / v_type_size(src0->type);

  v_vk_op_f32<vk_op_rope_push_constants>(ctx,
                                         subctx,
                                         src0,
                                         src1,
                                         src2,
                                         dst,
                                         V_OP_ROPE,
                                         {
                                           (uint32_t)src0->ne[0], (uint32_t)n_dims, freq_scale,
                                           (uint32_t)src0->ne[1],
                                           freq_base, ext_factor, attn_factor, {corr_dims[0], corr_dims[1]},
                                           theta_scale,
                                           src2 != nullptr, (uint32_t)src0->ne[2], s1, s2,
                                           {sections[0], sections[1], sections[2], sections[3]}, backprop
                                         },
                                         dryrun);
}

void v_vk_argsort(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
                  bool dryrun) {
  int32_t* op_params = dst->op_params.data();
  uint32_t ncols     = src0->ne[0];

  v_vk_op_f32<vk_op_argsort_push_constants>(ctx,
                                            subctx,
                                            src0,
                                            nullptr,
                                            nullptr,
                                            dst,
                                            v_OP_ARGSORT,
                                            {
                                              ncols,
                                              op_params[0],
                                            },
                                            dryrun);
}

void v_vk_sum(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
              bool dryrun) {
  vk_op_sum_rows_push_constants p = vk_op_sum_rows_push_constants_init(src0, dst, nelements(src0));
  v_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, v_OP_SUM, p, dryrun);
}

void v_vk_sum_rows(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                   v_tensor* dst, bool dryrun) {
  vk_op_sum_rows_push_constants p = vk_op_sum_rows_push_constants_init(src0, dst, src0->ne[0]);
  v_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, v_OP_SUM_ROWS, p, dryrun);
}

void v_vk_mean(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
               bool dryrun) {
  vk_op_sum_rows_push_constants p = vk_op_sum_rows_push_constants_init(src0, dst, src0->ne[0]);
  p.weight                        = 1.0f / (float)src0->ne[0];
  v_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, V_OP_MEAN, p, dryrun);
}

void v_vk_argmax(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
                 bool dryrun) {
  v_vk_op_f32<vk_op_push_constants>(ctx,
                                    subctx,
                                    src0,
                                    nullptr,
                                    nullptr,
                                    dst,
                                    V_OP_ARGMAX,
                                    {(uint32_t)src0->ne[0], (uint32_t)src0->ne[1], 0.0f, 0.0f},
                                    dryrun);
}

void v_vk_count_equal(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                      const v_tensor* src1, v_tensor* dst, bool dryrun) {
  v_vk_op_f32<vk_op_push_constants>(ctx,
                                    subctx,
                                    src0,
                                    src1,
                                    nullptr,
                                    dst,
                                    v_OP_COUNT_EQUAL,
                                    {(uint32_t)nelements(src0), 0, 0.0f, 0.0f},
                                    dryrun);
}


void v_vk_leaky_relu(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                     v_tensor* dst, bool dryrun) {
  const float* op_params = reinterpret_cast<const float*>(dst->op_params.data());
  v_vk_op_f32<vk_op_push_constants>(ctx,
                                    subctx,
                                    src0,
                                    nullptr,
                                    nullptr,
                                    dst,
                                    v_OP_LEAKY_RELU,
                                    {static_cast<uint32_t>(nelements(src0)), 0, op_params[0], 0.0f},
                                    dryrun);
}

void v_vk_get_rows(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                   const v_tensor* src1, v_tensor* dst, bool dryrun) {
  const uint32_t src0_type_size = v_type_size(src0->type);
  const uint32_t src1_type_size = v_type_size(src1->type);
  const uint32_t dst_type_size  = v_type_size(dst->type);

  v_vk_op_f32<vk_op_binary_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           src1,
                                           nullptr,
                                           dst,
                                           v_OP_GET_ROWS,
                                           {
                                             (uint32_t)nelements(src0),
                                             (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
                                             (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size,
                                             (uint32_t)src0->nb[1] / src0_type_size,
                                             (uint32_t)src0->nb[2] / src0_type_size,
                                             (uint32_t)src0->nb[3] / src0_type_size,
                                             (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],
                                             (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size,
                                             (uint32_t)src1->nb[1] / src1_type_size,
                                             (uint32_t)src1->nb[2] / src1_type_size,
                                             (uint32_t)src1->nb[3] / src1_type_size,
                                             (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], (uint32_t)dst->ne[2],
                                             (uint32_t)dst->ne[3], (uint32_t)dst->nb[0] / dst_type_size,
                                             (uint32_t)dst->nb[1] / dst_type_size,
                                             (uint32_t)dst->nb[2] / dst_type_size,
                                             (uint32_t)dst->nb[3] / dst_type_size,
                                             0,
                                             0.0f, 0.0f, 0,
                                           },
                                           dryrun);
}

void v_vk_acc(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
              const v_tensor* src1, v_tensor* dst, bool dryrun) {
  const uint32_t src0_type_size = v_type_size(src0->type);
  const uint32_t src1_type_size = v_type_size(src1->type);
  const uint32_t dst_type_size  = v_type_size(dst->type);

  int nb1 = dst->op_params[0] / 4; // 4 bytes of float32
  int nb2 = dst->op_params[1] / 4; // 4 bytes of float32
  // int nb3 = dst->op_params[2] / 4; // 4 bytes of float32 - unused
  int offset = dst->op_params[3] / 4; // offset in bytes

  v_vk_op_f32<vk_op_binary_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           src1,
                                           nullptr,
                                           dst,
                                           v_OP_ACC,
                                           {
                                             (uint32_t)nelements(src0),
                                             (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
                                             (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size,
                                             (uint32_t)nb1, (uint32_t)nb2, (uint32_t)src0->nb[3] / src0_type_size,
                                             (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],
                                             (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size,
                                             (uint32_t)src1->nb[1] / src1_type_size,
                                             (uint32_t)src1->nb[2] / src1_type_size,
                                             (uint32_t)src1->nb[3] / src1_type_size,
                                             (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], (uint32_t)dst->ne[2],
                                             (uint32_t)dst->ne[3], (uint32_t)dst->nb[0] / dst_type_size,
                                             (uint32_t)nb1, (uint32_t)nb2, (uint32_t)dst->nb[3] / dst_type_size,
                                             0,
                                             0.0f, 0.0f, offset,
                                           },
                                           dryrun);
}

void v_vk_multi_add(vk_backend_ctx* ctx, vk_context& subctx, v_cgraph* cgraph, int node_idx,
                    bool dryrun) {
  const v_tensor* first_node = cgraph->nodes[node_idx];
  const v_tensor* dst        = cgraph->nodes[node_idx + ctx->num_additional_fused_ops];

  // Make a list of all the tensors used_bits__ by the op.
  // Last element of the list is the dest tensor.
  const v_tensor* tensors[MAX_PARAMETER_COUNT];
  uint32_t num_srcs    = ctx->num_additional_fused_ops + 2;
  uint32_t num_tensors = num_srcs + 1;
  V_ASSERT(num_tensors + ctx->do_add_rms_partials <= MAX_PARAMETER_COUNT);

  tensors[0] = first_node->src[0];
  tensors[1] = first_node->src[1];
  for (int32_t i = 0; i < ctx->num_additional_fused_ops; ++i) {
    // check whether the previous result is src[0] or src[1]
    if (cgraph->nodes[node_idx + i] == cgraph->nodes[node_idx + i + 1]->src[0]) { tensors[i + 2] = cgraph->nodes[node_idx + i + 1]->src[1]; }
    else { tensors[i + 2] = cgraph->nodes[node_idx + i + 1]->src[0]; }
  }
  tensors[num_srcs] = dst;

  vk_op_multi_add_push_constants pc;
  pc.ne20 = (uint32_t)dst->ne[0];
  pc.ne21 = (uint32_t)dst->ne[1];
  pc.ne22 = (uint32_t)dst->ne[2];
  pc.ne23 = (uint32_t)dst->ne[3];

  for (uint32_t i = 0; i < num_tensors; ++i) {
    const v_tensor* t = tensors[i];
    pc.nb[i][0]       = (uint32_t)t->nb[0] / sizeof(float);
    pc.nb[i][1]       = (uint32_t)t->nb[1] / sizeof(float);
    pc.nb[i][2]       = (uint32_t)t->nb[2] / sizeof(float);
    pc.nb[i][3]       = (uint32_t)t->nb[3] / sizeof(float);
  }
  pc.rms_partials = ctx->do_add_rms_partials;

  vk_pipeline pipeline = v_vk_op_get_pipeline(ctx, tensors[0], tensors[1], nullptr, dst, dst->op);

  if (pipeline == nullptr) {
    std::cerr << "v_vulkan: Error: Missing multi_add";
    v_ABORT("fatal error");
  }

  if (dryrun) {
    v_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    return;
  }

  v_backend_vk_buffer_ctx* buf_ctx[MAX_PARAMETER_COUNT];
  vk_buffer buf[MAX_PARAMETER_COUNT];
  size_t offset[MAX_PARAMETER_COUNT];
  bool uma[MAX_PARAMETER_COUNT];

  for (uint32_t i = 0; i < num_tensors; ++i) {
    buf_ctx[i] = (v_backend_vk_buffer_ctx*)tensors[i]->buffer->context;
    buf[i]     = nullptr;
    offset[i]  = 0;
    uma[i]     = false;

    if (ctx->device->uma) {
      vk_get_host_buffer(ctx->device, tensors[i]->data, buf[i], offset[i]);
      uma[i] = buf[i] != nullptr;
    }
    if (!uma[i]) {
      buf[i]    = buf_ctx[i]->dev_buffer;
      offset[i] = vk_tensor_offset(tensors[i]) + tensors[i]->view_offs;
    }
    V_ASSERT(buf[i] != nullptr);
  }
  // If any remaining descriptors are unused, just point them at src[0]
  for (uint32_t i = num_tensors; i < MAX_PARAMETER_COUNT; ++i) {
    buf[i]    = buf[0];
    offset[i] = 0;
  }
  if (ctx->do_add_rms_partials) {
    buf[num_tensors]    = ctx->prealloc_add_rms_partials;
    offset[num_tensors] = ctx->prealloc_size_add_rms_partials_offset;
  }

  std::array<uint32_t, 3> elements;

  uint32_t ne = nelements(dst);
  if (ne > 262144) { elements = {512, 512, CEIL_DIV(ne, 262144)}; }
  else if (ne > 512) { elements = {512, CEIL_DIV(ne, 512), 1}; }
  else { elements = {ne, 1, 1}; }

  static_assert(MAX_PARAMETER_COUNT == 12);
  v_vk_dispatch_pipeline(ctx,
                         subctx,
                         pipeline,
                         {
                           v_vk_subbuffer(ctx, buf[0], offset[0]),
                           v_vk_subbuffer(ctx, buf[1], offset[1]),
                           v_vk_subbuffer(ctx, buf[2], offset[2]),
                           v_vk_subbuffer(ctx, buf[3], offset[3]),
                           v_vk_subbuffer(ctx, buf[4], offset[4]),
                           v_vk_subbuffer(ctx, buf[5], offset[5]),
                           v_vk_subbuffer(ctx, buf[6], offset[6]),
                           v_vk_subbuffer(ctx, buf[7], offset[7]),
                           v_vk_subbuffer(ctx, buf[8], offset[8]),
                           v_vk_subbuffer(ctx, buf[9], offset[9]),
                           v_vk_subbuffer(ctx, buf[10], offset[10]),
                           v_vk_subbuffer(ctx, buf[11], offset[11]),
                         },
                         pc,
                         elements);
}
