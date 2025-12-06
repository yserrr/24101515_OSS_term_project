#include "vk_vision_comp.hpp"
#include "vk_constant.h"
#include "vk_op_f32.hpp"

std::array<uint32_t, 3> v_vk_get_conv_elements(const v_tensor* dst) {
  const v_tensor* src0 = dst->src[0];
  const v_tensor* src1 = dst->src[1];

  // src0 - kernel:   [KW, KH, Cin, Cout]
  // src1 - input:    [W, H, Cin, N]
  // dst - result:    [OW, OH, Cout, N]
  // Copied from ggml.c: int64_t v_calc_conv_output_size(int64_t ins, int64_t ks, int s, int p, int d)
  auto calc_conv_output_size = [](int64_t ins, int64_t ks, int s, int p, int d) -> int64_t {
    return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
  };
  // parallelize in {OW/BS_K, OH/BS_NPQ, 1}
  int64_t W    = src1->ne[0];
  int64_t H    = src1->ne[1];
  int64_t KW   = src0->ne[0];
  int64_t KH   = src0->ne[1];
  int64_t Cout = src0->ne[3];
  int64_t N    = src1->ne[3];
  int64_t OH   = calc_conv_output_size(H, KH, dst->op_params[1], dst->op_params[3], dst->op_params[5]);
  int64_t OW   = calc_conv_output_size(W, KW, dst->op_params[0], dst->op_params[2], dst->op_params[4]);
  int64_t NPQ  = N * OW * OH;

  // Tile output matrix to (K/NB_K, NPQ/NB_NPQ, 1) workgroups
  std::array<uint32_t, 3> elements = {static_cast<uint32_t>(Cout), static_cast<uint32_t>(NPQ), 1};
  return elements;
}
void v_vk_conv_transpose_1d(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                            const v_tensor* src1, v_tensor* dst, bool dryrun) {
  // src0: (K, Cout, Cin, 1) -- kernel
  // src1: (L, Cin, 1, 1) -- input
  // dst: (*, Cout, 1, 1)

  V_ASSERT(src0->type == v_TYPE_F32);
  V_ASSERT(src1->type == v_TYPE_F32);
  V_ASSERT(dst->type == v_TYPE_F32);

  v_TENSOR_BINARY_OP_LOCALS

  V_ASSERT(nb00 == sizeof(float));
  V_ASSERT(nb10 == sizeof(float));

  const int32_t s0 = dst->op_params[0];

  vk_op_conv_transpose_1d_push_constants p{};
  p.Cout = static_cast<uint32_t>(ne01);
  p.Cin  = static_cast<uint32_t>(ne02);
  p.K    = static_cast<uint32_t>(ne00);
  p.L    = static_cast<uint32_t>(ne10);
  p.KL   = static_cast<uint32_t>(ne0);
  p.nb01 = static_cast<uint32_t>(nb01 / nb00);
  p.nb02 = static_cast<uint32_t>(nb02 / nb00);
  p.nb11 = static_cast<uint32_t>(nb11 / nb10);
  p.nb1  = static_cast<uint32_t>(nb1 / nb0);
  p.s0   = static_cast<uint32_t>(s0);

  v_vk_op_f32(ctx, subctx, src0, src1, nullptr, dst, v_OP_CONV_TRANSPOSE_1D, std::move(p), dryrun);
}

void v_vk_conv_transpose_2d(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                            const v_tensor* src1, v_tensor* dst, bool dryrun) {
  V_ASSERT(src0->type == v_TYPE_F32 || src0->type == v_TYPE_F16);
  V_ASSERT(src1->type == v_TYPE_F32);
  V_ASSERT(dst->type == v_TYPE_F32);

  v_TENSOR_BINARY_OP_LOCALS

  V_ASSERT(nb00 == sizeof(float) || nb00 == sizeof(v_fp16_t));
  V_ASSERT(nb10 == sizeof(float));
  V_ASSERT(nb0 == sizeof(float));

  vk_op_conv_transpose_2d_push_constants p{};
  p.Cout = static_cast<uint32_t>(ne02);
  p.Cin  = static_cast<uint32_t>(ne03);
  p.N    = static_cast<uint32_t>(ne13);

  p.KW = static_cast<uint32_t>(ne00);
  p.KH = static_cast<uint32_t>(ne01);
  p.W  = static_cast<uint32_t>(ne10);
  p.H  = static_cast<uint32_t>(ne11);
  p.OW = static_cast<uint32_t>(ne0);
  p.OH = static_cast<uint32_t>(ne1);

  p.s0 = static_cast<uint32_t>(dst->op_params[0]);
  p.s1 = static_cast<uint32_t>(dst->op_params[0]);
  p.p0 = 0;
  p.p1 = 0;
  p.d0 = 1;
  p.d1 = 1;

  p.nb01 = static_cast<uint32_t>(nb01 / nb00);
  p.nb02 = static_cast<uint32_t>(nb02 / nb00);
  p.nb03 = static_cast<uint32_t>(nb03 / nb00);

  p.nb11 = static_cast<uint32_t>(nb11 / nb10);
  p.nb12 = static_cast<uint32_t>(nb12 / nb10);
  p.nb13 = static_cast<uint32_t>(nb13 / nb10);

  p.nb1 = static_cast<uint32_t>(nb1 / nb0);
  p.nb2 = static_cast<uint32_t>(nb2 / nb0);
  p.nb3 = static_cast<uint32_t>(nb3 / nb0);

  V_ASSERT(ne02 == ne2);
  V_ASSERT(ne03 == ne12);

  v_vk_op_f32(ctx, subctx, src0, src1, nullptr, dst, v_OP_CONV_TRANSPOSE_2D, std::move(p), dryrun);
}

std::array<uint32_t, 3> v_vk_get_conv_transpose_2d_elements(const v_tensor* dst) {
  const v_tensor* src0 = dst->src[0];
  const v_tensor* src1 = dst->src[1];

  // src0 - kernel:   [KW, KH, Cout, Cin]
  // src1 - input:    [W, H, Cin, N]
  // dst - result:    [OW, OH, Cout, N]

  auto calc_conv_output_size = [](int64_t ins, int64_t ks, int s, int p, int d) -> int64_t { return (ins - 1) * s - 2 * p + (ks - 1) * d + 1; };
  // parallelize in {OW/BS_K, OH/BS_NPQ, 1}
  int64_t W    = src1->ne[0];
  int64_t H    = src1->ne[1];
  int64_t KW   = src0->ne[0];
  int64_t KH   = src0->ne[1];
  int64_t Cout = src0->ne[2];
  int64_t N    = src1->ne[3];
  int64_t OH   = calc_conv_output_size(H, KH, dst->op_params[0], 0, 1);
  int64_t OW   = calc_conv_output_size(W, KW, dst->op_params[0], 0, 1);
  int64_t NPQ  = N * OW * OH;

  // Tile output matrix to (K/NB_K, NPQ/NB_NPQ, 1) workgroups
  std::array<uint32_t, 3> elements = {static_cast<uint32_t>(Cout), static_cast<uint32_t>(NPQ), 1};
  return elements;
}

void v_vk_conv_2d_dw(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                     const v_tensor* src1, v_tensor* dst, bool dryrun) {
  vk_op_conv2d_dw_push_constants p{};
  p.ne         = nelements(dst);
  p.channels   = dst->ne[2];
  p.batches    = dst->ne[3];
  p.dst_w      = dst->ne[0];
  p.dst_h      = dst->ne[1];
  p.src_w      = src1->ne[0];
  p.src_h      = src1->ne[1];
  p.knl_w      = src0->ne[0];
  p.knl_h      = src0->ne[1];
  p.stride_x   = dst->op_params[0];
  p.stride_y   = dst->op_params[1];
  p.pad_x      = dst->op_params[2];
  p.pad_y      = dst->op_params[3];
  p.dilation_x = dst->op_params[4];
  p.dilation_y = dst->op_params[5];

  V_ASSERT(src0->ne[3] == p.channels);
  V_ASSERT(src1->ne[3] == p.batches);

  v_vk_op_f32(ctx, subctx, src0, src1, nullptr, dst, v_OP_CONV_2D_DW, std::move(p), dryrun);
}


void v_vk_conv_2d(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                  const v_tensor* src1, v_tensor* dst, bool dryrun) {
  V_ASSERT(src0->type == v_TYPE_F32 || src0->type == v_TYPE_F16);
  V_ASSERT(src1->type == v_TYPE_F32);
  V_ASSERT(dst->type == v_TYPE_F32);

  v_TENSOR_BINARY_OP_LOCALS

  V_ASSERT(nb00 == sizeof(float) || nb00 == sizeof(v_fp16_t));
  V_ASSERT(nb10 == sizeof(float));
  V_ASSERT(nb0 == sizeof(float));

  vk_op_conv2d_push_constants p{};
  p.Cout = static_cast<uint32_t>(ne03);
  p.Cin  = static_cast<uint32_t>(ne02);
  p.N    = static_cast<uint32_t>(ne13);

  p.KW = static_cast<uint32_t>(ne00);
  p.KH = static_cast<uint32_t>(ne01);
  p.W  = static_cast<uint32_t>(ne10);
  p.H  = static_cast<uint32_t>(ne11);
  p.OW = static_cast<uint32_t>(ne0);
  p.OH = static_cast<uint32_t>(ne1);

  p.s0 = static_cast<uint32_t>(dst->op_params[0]);
  p.s1 = static_cast<uint32_t>(dst->op_params[1]);
  p.p0 = static_cast<uint32_t>(dst->op_params[2]);
  p.p1 = static_cast<uint32_t>(dst->op_params[3]);
  p.d0 = static_cast<uint32_t>(dst->op_params[4]);
  p.d1 = static_cast<uint32_t>(dst->op_params[5]);

  p.nb01 = static_cast<uint32_t>(nb01 / nb00);
  p.nb02 = static_cast<uint32_t>(nb02 / nb00);
  p.nb03 = static_cast<uint32_t>(nb03 / nb00);

  p.nb11 = static_cast<uint32_t>(nb11 / nb10);
  p.nb12 = static_cast<uint32_t>(nb12 / nb10);
  p.nb13 = static_cast<uint32_t>(nb13 / nb10);

  p.nb1 = static_cast<uint32_t>(nb1 / nb0);
  p.nb2 = static_cast<uint32_t>(nb2 / nb0);
  p.nb3 = static_cast<uint32_t>(nb3 / nb0);

  V_ASSERT(ne03 == ne2);
  V_ASSERT(ne02 == ne12);

  v_vk_op_f32(ctx, subctx, src0, src1, nullptr, dst, v_OP_CONV_2D, std::move(p), dryrun);
}

void v_vk_pool_2d(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
                  bool dryrun) {
  uint32_t op      = static_cast<uint32_t>(dst->op_params[0]);
  const int32_t k1 = dst->op_params[1];
  const int32_t k0 = dst->op_params[2];
  const int32_t s1 = dst->op_params[3];
  const int32_t s0 = dst->op_params[4];
  const int32_t p1 = dst->op_params[5];
  const int32_t p0 = dst->op_params[6];

  const uint32_t IH = src0->ne[1];
  const uint32_t IW = src0->ne[0];

  const uint32_t N = dst->ne[3];

  const uint32_t OC = dst->ne[2];
  const uint32_t OH = dst->ne[1];
  const uint32_t OW = dst->ne[0];

  const uint32_t parallel_elements = N * OC * OH * OW;

  v_vk_op_f32<vk_op_pool2d_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           nullptr,
                                           nullptr,
                                           dst,
                                           V_OP_POOL_2D,
                                           {
                                             IW, IH, OW, OH, OC,
                                             parallel_elements,
                                             op,
                                             k0, k1, s0, s1, p0, p1,
                                           },
                                           dryrun);
}

void v_vk_pool_2d_back(vk_backend_ctx* ctx,
                       vk_context& subctx,
                       const v_tensor* src0, v_tensor* dst,
                       bool dryrun) {
  uint32_t op      = static_cast<uint32_t>(dst->op_params[0]);
  const int32_t k1 = dst->op_params[1];
  const int32_t k0 = dst->op_params[2];
  const int32_t s1 = dst->op_params[3];
  const int32_t s0 = dst->op_params[4];
  const int32_t p1 = dst->op_params[5];
  const int32_t p0 = dst->op_params[6];

  const uint32_t IH = src0->ne[1];
  const uint32_t IW = src0->ne[0];

  const uint32_t N = dst->ne[3];

  const uint32_t OC = dst->ne[2];
  const uint32_t OH = dst->ne[1];
  const uint32_t OW = dst->ne[0];

  const uint32_t parallel_elements = N * OC * OH * OW;

  v_vk_op_f32<vk_op_pool2d_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           nullptr,
                                           nullptr,
                                           dst,
                                           V_OP_POOL_2D_BACK,
                                           {
                                             IW, IH, OW, OH, OC,
                                             parallel_elements,
                                             op,
                                             k0, k1, s0, s1, p0, p1,
                                           },
                                           dryrun);
}

void v_vk_im2col_3d(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                    const v_tensor* src1, v_tensor* dst, bool dryrun) {
  v_TENSOR_BINARY_OP_LOCALS

  const int32_t s0 = static_cast<const int32_t*>(dst->op_params)[0];
  const int32_t s1 = static_cast<const int32_t*>(dst->op_params)[1];
  const int32_t s2 = static_cast<const int32_t*>(dst->op_params)[2];
  const int32_t p0 = static_cast<const int32_t*>(dst->op_params)[3];
  const int32_t p1 = static_cast<const int32_t*>(dst->op_params)[4];
  const int32_t p2 = static_cast<const int32_t*>(dst->op_params)[5];
  const int32_t d0 = static_cast<const int32_t*>(dst->op_params)[6];
  const int32_t d1 = static_cast<const int32_t*>(dst->op_params)[7];
  const int32_t d2 = static_cast<const int32_t*>(dst->op_params)[8];
  const int32_t IC = static_cast<const int32_t*>(dst->op_params)[9];

  const int64_t N  = ne13 / IC;
  const int64_t ID = ne12;
  const int64_t IH = ne11;
  const int64_t IW = ne10;

  const int64_t KD = ne02;
  const int64_t KH = ne01;
  const int64_t KW = ne00;

  const int64_t OD = ne3 / N;
  const int64_t OH = ne2;
  const int64_t OW = ne1;

  const v_backend_vk_buffer_ctx* d_buf_ctx = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  const vk_buffer d_buf                    = d_buf_ctx->dev_buffer;

  const vk::DeviceAddress dst_addr = d_buf->bda_addr + vk_tensor_offset(dst) + dst->view_offs;

  vk_op_im2col_3d_push_constants pc{};

  pc.dst_addr             = dst_addr;
  pc.nb10                 = nb10 / v_type_size(src1->type);
  pc.nb11                 = nb11 / v_type_size(src1->type);
  pc.nb12                 = nb12 / v_type_size(src1->type);
  pc.nb13                 = nb13 / v_type_size(src1->type);
  pc.s0                   = s0;
  pc.s1                   = s1;
  pc.s2                   = s2;
  pc.p0                   = p0;
  pc.p1                   = p1;
  pc.p2                   = p2;
  pc.d0                   = d0;
  pc.d1                   = d1;
  pc.d2                   = d2;
  pc.IW                   = IW;
  pc.IH                   = IH;
  pc.ID                   = ID;
  pc.IC                   = IC;
  pc.KW                   = KW;
  pc.OH                   = OH;
  pc.KD_KH_KW             = KD * KH * KW;
  pc.KH_KW                = KH * KW;
  pc.IC_KD_KH_KW          = IC * KD * KH * KW;
  pc.N_OD_OH              = N * OD * OH;
  pc.OD_OH                = OD * OH;
  pc.OD_OH_OW_IC_KD_KH_KW = OD * OH * OW * IC * KD * KH * KW;
  pc.OH_OW_IC_KD_KH_KW    = OH * OW * IC * KD * KH * KW;
  pc.OW_IC_KD_KH_KW       = OW * IC * KD * KH * KW;

  v_vk_op_f32<vk_op_im2col_3d_push_constants>(ctx,
                                              subctx,
                                              src0,
                                              src1,
                                              nullptr,
                                              dst,
                                              v_OP_IM2COL_3D,
                                              std::move(pc),
                                              dryrun);
}

void v_vk_im2col(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                 const v_tensor* src1, v_tensor* dst, bool dryrun) {
  const int32_t s0 = dst->op_params[0];
  const int32_t s1 = dst->op_params[1];
  const int32_t p0 = dst->op_params[2];
  const int32_t p1 = dst->op_params[3];
  const int32_t d0 = dst->op_params[4];
  const int32_t d1 = dst->op_params[5];

  const bool is_2D = dst->op_params[6] == 1;

  const uint32_t IC = src1->ne[is_2D
                                 ? 2
                                 : 1];
  const uint32_t IH = is_2D
                        ? src1->ne[1]
                        : 1;
  const uint32_t IW = src1->ne[0];

  const uint32_t KH = is_2D
                        ? src0->ne[1]
                        : 1;
  const uint32_t KW = src0->ne[0];

  const uint32_t OH = is_2D
                        ? dst->ne[2]
                        : 1;
  const uint32_t OW = dst->ne[1];

  const uint32_t offset_delta = src1->nb[is_2D
                                           ? 2
                                           : 1] / 4; // nb is byte offset, src is type float32
  const uint32_t batch_offset = src1->nb[is_2D
                                           ? 3
                                           : 2] / 4; // nb is byte offset, src is type float32

  const uint32_t pelements = OW * KW * KH;

  const v_backend_vk_buffer_ctx* d_buf_ctx = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  const vk_buffer d_buf                    = d_buf_ctx->dev_buffer;

  const vk::DeviceAddress dst_addr = d_buf->bda_addr + vk_tensor_offset(dst) + dst->view_offs;

  v_vk_op_f32<vk_op_im2col_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           src1,
                                           nullptr,
                                           dst,
                                           V_OP_IM2COL,
                                           {
                                             dst_addr,
                                             batch_offset, offset_delta,
                                             IC, IW, IH, OW, OH, KW, KH,
                                             pelements,
                                             IC * KH * KW,
                                             s0, s1, p0, p1, d0, d1,
                                           },
                                           dryrun);
}


