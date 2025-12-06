#include "v_vision.hpp"
#include "ggml-impl.h"
#include "vk_common.h"
#include "vk_util.h"
static int64_t v_calc_conv_output_size(int64_t ins, int64_t ks, int s, int p, int d) {
  return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

struct v_tensor* v_conv_1d(struct v_ctx* ctx,
                           struct v_tensor* a,
                           struct v_tensor* b,
                           int s0,
                           int p0,
                           int d0) {
  struct v_tensor* im2col = v_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, v_TYPE_F16); // [N, OL, IC * K]

  struct v_tensor* result =
    v_matmul(ctx,
             v_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])),
             // [N, OL, IC * K] => [N*OL, IC * K]
             v_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2])); // [OC，IC, K] => [OC, IC * K]

  result = v_reshape_3d(ctx, result, im2col->ne[1], a->ne[2], im2col->ne[2]); // [N, OC, OL]

  return result;
}

// v_conv_1d_ph
struct v_tensor* v_conv_1d_ph(
  struct v_ctx* ctx,
  struct v_tensor* a,
  struct v_tensor* b,
  int s,
  int d) {
  return v_conv_1d(ctx, a, b, s, a->ne[0] / 2, d);
}

// v_conv_1d_dw

struct v_tensor* v_conv_1d_dw(
  struct v_ctx* ctx,
  struct v_tensor* a,
  struct v_tensor* b,
  int s0,
  int p0,
  int d0) {
  struct v_tensor* new_b = v_reshape_4d(ctx, b, b->ne[0], 1, b->ne[1], b->ne[2]);

  struct v_tensor* im2col = v_im2col(ctx, a, new_b, s0, 0, p0, 0, d0, 0, false, v_TYPE_F16);

  struct v_tensor* result = v_matmul(ctx, im2col, a);

  result = v_reshape_3d(ctx, result, result->ne[0], result->ne[2], 1);

  return result;
}

// v_conv_1d_dw_ph

struct v_tensor* v_conv_1d_dw_ph(
  struct v_ctx* ctx,
  struct v_tensor* a,
  struct v_tensor* b,
  int s0,
  int d0) {
  return v_conv_1d_dw(ctx, a, b, s0, a->ne[0] / 2, d0);
}

// v_conv_transpose_1d

static int64_t v_calc_conv_transpose_1d_output_size(int64_t ins, int64_t ks, int s, int p, int d) {
  return (ins - 1) * s - 2 * p + d * (ks - 1) + 1;
}

v_API struct v_tensor* v_conv_transpose_1d(
  struct v_ctx* ctx,
  struct v_tensor* a,
  struct v_tensor* b,
  int s0,
  int p0,
  int d0) {
  V_ASSERT(v_is_matrix(b));
  V_ASSERT(a->ne[2] == b->ne[1]);
  V_ASSERT(a->ne[3] == 1);

  V_ASSERT(p0 == 0);
  V_ASSERT(d0 == 1);

  const int64_t ne[4] = {
    v_calc_conv_transpose_1d_output_size(b->ne[0], a->ne[0], s0, 0 /*p0*/, 1 /*d0*/),
    a->ne[1], b->ne[2], 1,
  };
  struct v_tensor* result = v_new_tensor(ctx, v_TYPE_F32, 4, ne);

  int32_t params[] = {s0, p0, d0};
  v_set_op_params(result, params, sizeof(params));

  result->op     = v_OP_CONV_TRANSPOSE_1D;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

// v_conv_2d

// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OC, OH, OW]
struct v_tensor* v_conv_2d(
  struct v_ctx* ctx,
  struct v_tensor* a,
  struct v_tensor* b,
  int s0,
  int s1,
  int p0,
  int p1,
  int d0,
  int d1) {
  struct v_tensor* im2col = v_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, a->type);
  // [N, OH, OW, IC * KH * KW]

  struct v_tensor* result =
    v_matmul(ctx,
             v_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]),
             // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
             v_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2]), a->ne[3]));
  // [OC，IC, KH, KW] => [OC, IC * KH * KW]
  result = v_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3]); // [OC, N, OH, OW]
  result = v_mem_cont(ctx, v_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OH, OW]
  return result;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OD, OH, OW, IC * KD * KH * KW]
struct v_tensor* v_im2col_3d(
  struct v_ctx* ctx,
  struct v_tensor* a,
  struct v_tensor* b,
  int64_t IC,
  int s0, // stride width
  int s1, // stride height
  int s2, // stride depth
  int p0, // padding width
  int p1, // padding height
  int p2, // padding depth
  int d0, // dilation width
  int d1, // dilation height
  int d2, // dilation depth
  enum v_data_type dst_type) {
  const int64_t N  = b->ne[3] / IC;
  const int64_t ID = b->ne[2];
  const int64_t IH = b->ne[1];
  const int64_t IW = b->ne[0];

  const int64_t OC = a->ne[3] / IC;
  UNUSED(OC);
  const int64_t KD = a->ne[2];
  const int64_t KH = a->ne[1];
  const int64_t KW = a->ne[0];
  const int64_t OD = v_calc_conv_output_size(ID, KD, s2, p2, d2);
  const int64_t OH = v_calc_conv_output_size(IH, KH, s1, p1, d1);
  const int64_t OW = v_calc_conv_output_size(IW, KW, s0, p0, d0);

  V_ASSERT((OD > 0) && "b too small compared to a");
  V_ASSERT((OH > 0) && "b too small compared to a");
  V_ASSERT((OW > 0) && "b too small compared to a");


  const int64_t ne[4] = {KW * KH * KD * IC, OW, OH, OD * N};

  struct v_tensor* result = v_new_tensor(ctx, dst_type, 4, ne);
  int32_t params[]        = {s0, s1, s2, p0, p1, p2, d0, d1, d2, (int32_t)IC};
  v_set_op_params(result, params, sizeof(params));

  result->op     = v_OP_IM2COL_3D;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OC, OD, OH, OW]
struct v_tensor* v_conv_3d(
  struct v_ctx* ctx,
  struct v_tensor* a,
  struct v_tensor* b,
  int64_t IC,
  int s0, // stride width
  int s1, // stride height
  int s2, // stride depth
  int p0, // padding width
  int p1, // padding height
  int p2, // padding depth
  int d0, // dilation width
  int d1, // dilation height
  int d2 // dilation depth
) {
  struct v_tensor* im2col = v_im2col_3d(ctx, a, b, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, a->type);
  // [N*OD, OH, OW, IC * KD * KH * KW]

  int64_t OC              = a->ne[3] / IC;
  int64_t N               = b->ne[3] / IC;
  struct v_tensor* result =
    v_matmul(ctx,
             v_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]),
             // [N*OD, OH, OW, IC * KD * KH * KW] => [N*OD*OH*OW, IC * KD * KH * KW]
             v_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2] * IC), OC));
  // [OC*IC, KD, KH, KW] => [OC, IC * KD * KH * KW]

  int64_t OD = im2col->ne[3] / N;
  result     = v_reshape_4d(ctx, result, im2col->ne[1] * im2col->ne[2], OD, N, OC);
  // [OC, N*OD*OH*OW] => [OC, N, OD, OH*OW]
  result = v_mem_cont(ctx, v_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OD, OH*OW]
  result = v_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], OD, OC * N); // [N*OC, OD, OH, OW]

  return result;
}

// v_conv_2d_sk_p0

struct v_tensor* v_conv_2d_sk_p0(
  struct v_ctx* ctx,
  struct v_tensor* a,
  struct v_tensor* b) {
  return v_conv_2d(ctx, a, b, a->ne[0], a->ne[1], 0, 0, 1, 1);
}

// v_conv_2d_s1_ph

struct v_tensor* v_conv_2d_s1_ph(struct v_ctx* ctx,
                                 struct v_tensor* a,
                                 struct v_tensor* b) {
  return v_conv_2d(ctx, a, b, 1, 1, a->ne[0] / 2, a->ne[1] / 2, 1, 1);
}

// v_conv_2d_dw
struct v_tensor* v_conv_2d_dw(struct v_ctx* ctx,
                              struct v_tensor* a,
                              struct v_tensor* b,
                              int s0,
                              int s1,
                              int p0,
                              int p1,
                              int d0,
                              int d1) {
  struct v_tensor* new_a  = v_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
  struct v_tensor* im2col = v_im2col(ctx,
                                     new_a,
                                     v_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]),
                                     s0,
                                     s1,
                                     p0,
                                     p1,
                                     d0,
                                     d1,
                                     true,
                                     v_TYPE_F16); // [N * IC, OH, OW, KH * KW]
  struct v_tensor* new_b = v_reshape_4d(ctx,
                                        im2col,
                                        im2col->ne[0],
                                        im2col->ne[2] * im2col->ne[1],
                                        b->ne[2],
                                        b->ne[3]); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

  new_a = v_reshape_4d(ctx, new_a, (new_a->ne[0] * new_a->ne[1]), new_a->ne[2], new_a->ne[3], 1);
  // [OC，1, KH, KW] => [1, OC, 1, KH * KW]
  struct v_tensor* result = v_matmul(ctx, new_a, new_b);
  result                  = v_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]); // [N, OC, OH, OW]

  return result;
}

// v_conv_2d_dw_direct

struct v_tensor* v_conv_2d_dw_direct(
  struct v_ctx* ctx,
  struct v_tensor* a,
  struct v_tensor* b,
  int stride0,
  int stride1,
  int pad0,
  int pad1,
  int dilation0,
  int dilation1) {
  V_ASSERT(a->ne[2] == 1);
  V_ASSERT(a->ne[3] == b->ne[2]);
  int64_t ne[4];
  ne[0]                   = v_calc_conv_output_size(b->ne[0], a->ne[0], stride0, pad0, dilation0);
  ne[1]                   = v_calc_conv_output_size(b->ne[1], a->ne[1], stride1, pad1, dilation1);
  ne[2]                   = b->ne[2];
  ne[3]                   = b->ne[3];
  struct v_tensor* result = v_new_tensor(ctx, b->type, 4, ne);

  if (v_is_contiguous_channels(b)) {
    // Result will be permuted the same way as input (CWHN order)
    const int64_t type_size = v_type_size(result->type);
    V_ASSERT(block_size(result->type) == 1);
    result->nb[0] = result->ne[2] * type_size;
    result->nb[1] = result->ne[0] * result->nb[0];
    result->nb[2] = type_size;
  }

  int32_t params[] = {stride0, stride1, pad0, pad1, dilation0, dilation1};
  v_set_op_params(result, params, sizeof(params));

  result->op     = v_OP_CONV_2D_DW;
  result->src[0] = a;
  result->src[1] = b;
  return result;
}

// v_conv_2d_direct

struct v_tensor* v_conv_2d_direct(struct v_ctx* ctx,
                                  struct v_tensor* a, // convolution kernel [KW, KH, IC, OC]
                                  struct v_tensor* b, // input data [W, H, C, N]
                                  int s0, // stride dimension 0
                                  int s1, // stride dimension 1
                                  int p0, // padding dimension 0
                                  int p1, // padding dimension 1
                                  int d0, // dilation dimension 0
                                  int d1) {
  // dilation dimension 1
  V_ASSERT(a->ne[2] == b->ne[2]);
  V_ASSERT(a->type == b->type);

  int64_t ne[4];
  ne[0] = v_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);
  ne[1] = v_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1);
  ne[2] = a->ne[3];
  ne[3] = b->ne[3];

  struct v_tensor* result = v_new_tensor(ctx, b->type, 4, ne);

  v_set_op_params_i32(result, 0, s0);
  v_set_op_params_i32(result, 1, s1);
  v_set_op_params_i32(result, 2, p0);
  v_set_op_params_i32(result, 3, p1);
  v_set_op_params_i32(result, 4, d0);
  v_set_op_params_i32(result, 5, d1);

  result->op     = v_OP_CONV_2D;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

struct v_tensor* v_conv_3d_direct(
  struct v_ctx* ctx,
  struct v_tensor* a,
  struct v_tensor* b,
  int s0,
  int s1,
  int s2,
  int p0,
  int p1,
  int p2,
  int d0,
  int d1,
  int d2,
  int c,
  int n,
  int oc) {
  V_ASSERT(a->ne[3] == (int64_t) c * oc);
  V_ASSERT(b->ne[3] == (int64_t) c * n);

  int64_t ne[4];
  ne[0] = v_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);
  ne[1] = v_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1);
  ne[2] = v_calc_conv_output_size(b->ne[2], a->ne[2], s2, p2, d2);
  ne[3] = (int64_t)oc * n;

  struct v_tensor* result = v_new_tensor(ctx, v_TYPE_F32, 4, ne);

  v_set_op_params_i32(result, 0, s0);
  v_set_op_params_i32(result, 1, s1);
  v_set_op_params_i32(result, 2, s2);
  v_set_op_params_i32(result, 3, p0);
  v_set_op_params_i32(result, 4, p1);
  v_set_op_params_i32(result, 5, p2);
  v_set_op_params_i32(result, 6, d0);
  v_set_op_params_i32(result, 7, d1);
  v_set_op_params_i32(result, 8, d2);
  v_set_op_params_i32(result, 9, c);
  v_set_op_params_i32(result, 10, n);
  v_set_op_params_i32(result, 11, oc);

  result->op     = v_OP_CONV_3D;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

// v_conv_transpose_2d_p0

static int64_t v_calc_conv_transpose_output_size(int64_t ins, int64_t ks, int s, int p) {
  return (ins - 1) * s - 2 * p + ks;
}

struct v_tensor* v_conv_transpose_2d_p0(
  struct v_ctx* ctx,
  struct v_tensor* a,
  struct v_tensor* b,
  int stride) {
  V_ASSERT(a->ne[3] == b->ne[2]);

  const int64_t ne[4] = {
    v_calc_conv_transpose_output_size(b->ne[0], a->ne[0], stride, 0 /*p0*/),
    v_calc_conv_transpose_output_size(b->ne[1], a->ne[1], stride, 0 /*p1*/),
    a->ne[2], b->ne[3],
  };

  struct v_tensor* result = v_new_tensor(ctx, v_TYPE_F32, 4, ne);

  v_set_op_params_i32(result, 0, stride);

  result->op     = v_OP_CONV_TRANSPOSE_2D;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

// v_pool_*
static int64_t v_calc_pool_output_size(int64_t ins, int ks, int s, float p) {
  return (ins + 2 * p - ks) / s + 1;
}

// v_pool_1d

struct v_tensor* v_pool_1d(
  struct v_ctx* ctx,
  struct v_tensor* a,
  enum v_op_pool op,
  int k0,
  int s0,
  int p0) {
  const int64_t ne[4] = {
    v_calc_pool_output_size(a->ne[0], k0, s0, p0),
    a->ne[1],
    a->ne[2],
    a->ne[3],
  };
  struct v_tensor* result = v_new_tensor(ctx, v_TYPE_F32, 4, ne);

  int32_t params[] = {op, k0, s0, p0};
  v_set_op_params(result, params, sizeof(params));

  result->op     = v_OP_POOL_1D;
  result->src[0] = a;

  return result;
}

// v_pool_2d

struct v_tensor* v_pool_2d(struct v_ctx* ctx,
                           struct v_tensor* a,
                           enum v_op_pool op,
                           int k0, int k1,
                           int s0, int s1,
                           float p0, float p1) {
  struct v_tensor* result;
  const int64_t ne[4] = {
    v_calc_pool_output_size(a->ne[0], k0, s0, p0),
    v_calc_pool_output_size(a->ne[1], k1, s1, p1),
    a->ne[2],
    a->ne[3],
  };
  result = v_new_tensor(ctx, v_TYPE_F32, 4, ne);

  int32_t params[] = {
    static_cast<int32_t>(op),
    static_cast<int32_t>(k0),
    static_cast<int32_t>(k1),
    static_cast<int32_t>(s0),
    static_cast<int32_t>(s1),
    static_cast<int32_t>(p0),
    static_cast<int32_t>(p1)
  };
  v_set_op_params(result, params, sizeof(params));

  result->op     = V_OP_POOL_2D;
  result->src[0] = a;

  return result;
}

struct v_tensor* v_pool_2d_back(struct v_ctx* ctx,
                                struct v_tensor* a,
                                struct v_tensor* af,
                                enum v_op_pool op,
                                int k0,
                                int k1,
                                int s0,
                                int s1,
                                float p0,
                                float p1) {
  struct v_tensor* result;
  auto t_a         = v_mem_cont(ctx, a);
  auto t_af        = v_mem_cont(ctx, af);
  result           = v_new_tensor(ctx, v_TYPE_F32, 4, af->ne);
  int32_t params[] = {
    static_cast<int32_t>(op),
    static_cast<int32_t>(k0),
    static_cast<int32_t>(k1),
    static_cast<int32_t>(s0),
    static_cast<int32_t>(s1),
    static_cast<int32_t>(p0),
    static_cast<int32_t>(p1)
  };
  v_set_op_params(result, params, sizeof(params));
  result->op     = V_OP_POOL_2D_BACK;
  result->src[0] = t_af;
  result->src[1] = t_a;
  return result;
}

// v_upscale / v_interpolate
struct v_tensor* v_interpolate_impl(struct v_ctx* ctx,
                                    struct v_tensor* a,
                                    int64_t ne0,
                                    int64_t ne1,
                                    int64_t ne2,
                                    int64_t ne3,
                                    uint32_t mode) {
  V_ASSERT((mode & 0xFF) < v_SCALE_MODE_COUNT);

  struct v_tensor* result = v_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

  v_set_op_params_i32(result, 0, (int32_t)mode);

  result->op     = v_OP_UPSCALE;
  result->src[0] = a;

  return result;
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OH, OW, IC*KH*KW]
struct v_tensor* v_im2col(struct v_ctx* ctx,
                          struct v_tensor* a,
                          struct v_tensor* b,
                          int s0, int s1,
                          int p0, int p1,
                          int d0, int d1,
                          bool is_2D,
                          enum v_data_type dst_type) {
  if (is_2D) {
    V_ASSERT(a->ne[2] == b->ne[2]);
  }
  else {
    //v_ASSERT(b->ne[1] % a->ne[1] == 0);
    V_ASSERT(b->ne[1] == a->ne[1]);
    V_ASSERT(b->ne[3] == 1);
  }

  const int64_t OH = is_2D
                       ? v_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1)
                       : 0;
  const int64_t OW = v_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);
  V_ASSERT((!is_2D || OH > 0) && "b too small compared to a");
  V_ASSERT((OW > 0) && "b too small compared to a");

  const int64_t ne[4] = {
    is_2D
      ? (a->ne[2] * a->ne[1] * a->ne[0])
      : a->ne[1] * a->ne[0],
    OW,
    is_2D
      ? OH
      : b->ne[2],
    is_2D
      ? b->ne[3]
      : 1,
  };

  struct v_tensor* result = v_new_tensor(ctx, dst_type, 4, ne);
  int32_t params[]        = {
    s0, s1, p0, p1, d0, d1, (is_2D
                               ? 1
                               : 0)
  };
  v_set_op_params(result, params, sizeof(params));

  result->op     = V_OP_IM2COL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

///im2col:
///Y = X*W
/// t_a = dY(grad)
/// t_b = forward Input X
/// dX =
struct v_tensor* v_im2col_back(struct v_ctx* ctx,
                               struct v_tensor* a,
                               struct v_tensor* b,
                               int64_t* ne,
                               int s0, int s1,
                               int p0, int p1,
                               int d0, int d1,
                               bool is_2D) {

  //t_b= src0
  //t_W=  im2col (grad^T, dY_col)
  struct v_tensor* result = v_new_tensor(ctx, v_TYPE_F32, 4, ne);
  v_tensor_t t_a = v_mem_cont(ctx, v_transpose(ctx,a));
  v_tensor_t t_b = v_mem_cont(ctx, v_transpose(ctx,b));

  int32_t params[]        = {
    s0, s1, p0, p1, d0, d1, (is_2D
                               ? 1
                               : 0)
  };
  v_set_op_params(result, params, sizeof(params));
  result->op     = V_OP_IM2COL;
  result->src[0] = t_a;
  result->src[1] = t_b;
  return result;
}
