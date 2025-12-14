#ifndef MYPROJECT_V_VISION_HPP
#define MYPROJECT_V_VISION_HPP
#include "v.hpp"
#include "v_header.hpp"
// converts data into a format that effectively results in a convolution when combined with matrix multiplication
V_API v_tensor* v_im2col(v_ctx* ctx,
                         v_tensor* a, // convolution kernel
                         v_tensor* b, // data
                         int s0,      // stride dimension 0
                         int s1,      // stride dimension 1
                         int p0,      // padding dimension 0
                         int p1,      // padding dimension 1
                         int d0,      // dilation dimension 0
                         int d1,      // dilation dimension 1
                         bool is_2D,
                         enum v_data_type dst_type);

V_API v_tensor* v_im2col_back(
  struct v_ctx* ctx,
  v_tensor* a, // convolution kernel
  v_tensor* b, // gradient of im2col output
  int64_t* ne, // shape of im2col input
  int s0,      // stride dimension 0
  int s1,      // stride dimension 1
  int p0,      // padding dimension 0
  int p1,      // padding dimension 1
  int d0,      // dilation dimension 0
  int d1,      // dilation dimension 1
  bool is_2D);

V_API v_tensor* v_conv_1d(
  struct v_ctx* ctx,
  v_tensor* a, // convolution kernel
  v_tensor* b, // data
  int s0,      // stride
  int p0,      // padding
  int d0);     // dilation

// conv_1d with padding = half
// alias for v_conv_1d(a, b, s, a->ne[0]/2, d)
V_API v_tensor* v_conv_1d_ph(
  struct v_ctx* ctx,
  v_tensor* a, // convolution kernel
  v_tensor* b, // data
  int s,       // stride
  int d);      // dilation

// depthwise
// TODO: this is very likely wrong for some cases! - needs more testing
V_API v_tensor* v_conv_1d_dw(
  struct v_ctx* ctx,
  v_tensor* a, // convolution kernel
  v_tensor* b, // data
  int s0,      // stride
  int p0,      // padding
  int d0);     // dilation

V_API v_tensor* v_conv_1d_dw_ph(
  struct v_ctx* ctx,
  v_tensor* a, // convolution kernel
  v_tensor* b, // data
  int s0,      // stride
  int d0);     // dilation

V_API v_tensor* v_conv_transpose_1d(
  struct v_ctx* ctx,
  v_tensor* a, // convolution kernel
  v_tensor* b, // data
  int s0,      // stride
  int p0,      // padding
  int d0);     // dilation
v_tensor* v_log_impl(struct v_ctx* ctx,
                     v_tensor* a,
                     bool inplace);

/**
 * @brief convolution operation with 2d
 *
 *
 * @param ctx       ptr to the computation context
 * @param a         convolution kernel
 * @param b         data
 * @param s0        stride dimension 0
 * @param s1        stride dimension 1
 * @param p0        padding dimension 0
 * @param p1        padding dimension 1
 * @param d0        dilation dimension 0
 * @param d1        dilation dimension 1
 * @return          Pointer to the view tensor (does not own the data)
 */
V_API v_tensor* v_conv_2d(v_ctx* ctx,v_tensor* a,v_tensor* b,int s0,int s1,int p0,int p1,int d0,int d1);

V_API v_tensor* v_im2col_3d(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
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
  enum v_data_type dst_type);

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OC, OD, OH, OW]
V_API v_tensor* v_conv_3d(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  int64_t IC,
  int s0, // stride width
  int s1, // stride height
  int s2, // stride depth
  int p0, // padding width
  int p1, // padding height
  int p2, // padding depth
  int d0, // dilation width
  int d1, // dilation height
  int d2  // dilation depth
);

// kernel size is a->ne[0] x a->ne[1]
// stride is equal to kernel size
// padding is zero
// example:
// a:     16   16    3  768
// b:   1024 1024    3    1
// res:   64   64  768    1
// used_bits__ in sam
V_API v_tensor* v_conv_2d_sk_p0(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b);

// kernel size is a->ne[0] x a->ne[1]
// stride is 1
// padding is half
// example:
// a:      3    3    256  256
// b:     64   64    256    1
// res:   64   64    256    1
// used_bits__ in sam
V_API v_tensor* v_conv_2d_s1_ph(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b);

// depthwise (via im2col and mul_mat)
V_API v_tensor* v_conv_2d_dw(
  struct v_ctx* ctx,
  v_tensor* a, // convolution kernel
  v_tensor* b, // data
  int s0,      // stride dimension 0
  int s1,      // stride dimension 1
  int p0,      // padding dimension 0
  int p1,      // padding dimension 1
  int d0,      // dilation dimension 0
  int d1);     // dilation dimension 1

// Depthwise 2D convolution
// may be faster than v_conv_2d_dw, but not available in all backends
// a:   KW    KH    1    C    convolution kernel
// b:   W     H     C    N    input data
// res: W_out H_out C    N
v_tensor* v_conv_2d_dw_direct(v_ctx* ctx, v_tensor* a, v_tensor* b, int stride0, int stride1, int pad0, int pad1, int dilation0, int dilation1);
v_tensor* v_conv_transpose_2d_p0(v_ctx* ctx, v_tensor* a, v_tensor* b, int stride);

// convolution kernel [KW, KH, IC, OC]
// input data [W, H, C, N]
// stride dimension 0
// stride dimension 1
// padding dimension 0
// padding dimension 1
// dilation dimension 0
// dilation dimension 1
V_API v_tensor* v_conv_2d_direct(v_ctx* ctx, v_tensor* a, v_tensor* b, int s0, int s1, int p0, int p1, int d0, int d1);
// kernel [KW, KH, KD, IC * OC]
// input  [W, H, D, C * N]
V_API v_tensor* v_conv_3d_direct(v_ctx* ctx, v_tensor* a, v_tensor* b, int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2, int n_channels, int n_batch, int n_channels_out);

V_API v_tensor* v_pool_1d(
  struct v_ctx* ctx,
  v_tensor* a,
  enum v_op_pool op,
  int k0,  // kernel size
  int s0,  // stride
  int p0); // padding

// the result will have 2*p0 padding for the first dimension
// and 2*p1 padding for the second dimension
V_API v_tensor* v_pool_2d(
  struct v_ctx* ctx,
  v_tensor* a,
  enum v_op_pool op,
  int k0,
  int k1,
  int s0,
  int s1,
  float p0,
  float p1);

V_API v_tensor* v_pool_2d_back(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* af, // "a"/input used_bits__ in forward pass
  enum v_op_pool op,
  int k0,
  int k1,
  int s0,
  int s1,
  float p0,
  float p1);

#endif //MYPROJECT_V_VISION_HPP
