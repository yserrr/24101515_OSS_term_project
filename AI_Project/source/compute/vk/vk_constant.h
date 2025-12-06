#ifndef MYPROJECT_VK_CONSTANT_H
#define MYPROJECT_VK_CONSTANT_H
#include "vk_common.h"

struct vk_mat_mat_push_constants
{
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t stride_a;
  uint32_t stride_b;
  uint32_t stride_d;
  uint32_t batch_stride_a;
  uint32_t batch_stride_b;
  uint32_t batch_stride_d;
  uint32_t k_split;
  uint32_t ne02;
  uint32_t ne12;
  uint32_t broadcast2;
  uint32_t broadcast3;
  uint32_t padded_N;
};

struct vk_mat_vec_push_constants
{
  uint32_t ncols;
  uint32_t stride_a;
  uint32_t stride_b;
  uint32_t stride_d;
  uint32_t batch_stride_a;
  uint32_t batch_stride_b;
  uint32_t batch_stride_d;
  uint32_t ne02;
  uint32_t ne12;
  uint32_t broadcast2;
  uint32_t broadcast3;
};

struct vk_mat_mat_id_push_constants
{
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t stride_a;
  uint32_t stride_b;
  uint32_t stride_d;
  uint32_t batch_stride_a;
  uint32_t batch_stride_b;
  uint32_t batch_stride_d;
  uint32_t nei0;
  uint32_t nei1;
  uint32_t nbi1;
  uint32_t ne11;
  uint32_t padded_N;
};

struct vk_mat_vec_id_push_constants
{
  uint32_t ncols;
  uint32_t stride_a;
  uint32_t stride_b;
  uint32_t stride_d;
  uint32_t batch_stride_a;
  uint32_t batch_stride_b;
  uint32_t batch_stride_d;
  uint32_t nei0;
  uint32_t ne11;
};

struct vk_flash_attn_push_constants
{
  uint32_t N;
  uint32_t KV;

  uint32_t ne1;
  uint32_t ne2;
  uint32_t ne3;

  uint32_t neq2;
  uint32_t neq3;
  uint32_t nek2;
  uint32_t nek3;
  uint32_t nev2;
  uint32_t nev3;
  uint32_t nem1;
  uint32_t nem2;
  uint32_t nem3;

  uint32_t nb01;
  uint32_t nb02;
  uint32_t nb03;
  uint32_t nb11;
  uint32_t nb12;
  uint32_t nb13;
  uint32_t nb21;
  uint32_t nb22;
  uint32_t nb23;

  float scale;
  float max_bias;
  float logit_softcap;

  uint32_t mask_n_head_log2;
  float m0;
  float m1;

  uint32_t gqa_ratio;
  uint32_t split_kv;
  uint32_t k_num;
};

static_assert(sizeof(vk_flash_attn_push_constants) <= 128, "sizeof(vk_flash_attn_push_constants) must be <= 128");


struct vk_op_push_constants
{
  uint32_t KX;
  uint32_t KY;
  float param1;
  float param2;
};


struct vk_op_glu_push_constants
{
  uint32_t N;
  uint32_t ne00;
  uint32_t ne20;
  uint32_t mode; // 0: default, 1: swapped, 2: split
  float alpha; // for swiglu_oai
  float limit;
};

struct vk_op_unary_push_constants
{
  uint32_t ne;
  uint32_t ne00;
  uint32_t ne01;
  uint32_t ne02;
  uint32_t ne03;
  uint32_t nb00;
  uint32_t nb01;
  uint32_t nb02;
  uint32_t nb03;
  uint32_t ne10;
  uint32_t ne11;
  uint32_t ne12;
  uint32_t ne13;
  uint32_t nb10;
  uint32_t nb11;
  uint32_t nb12;
  uint32_t nb13;
  uint32_t misalign_offsets;
  float param1;
  float param2;
  uint32_t ne0_012mp;
  uint32_t ne0_012L;
  uint32_t ne0_01mp;
  uint32_t ne0_01L;
  uint32_t ne0_0mp;
  uint32_t ne0_0L;
  uint32_t ne1_012mp;
  uint32_t ne1_012L;
  uint32_t ne1_01mp;
  uint32_t ne1_01L;
  uint32_t ne1_0mp;
  uint32_t ne1_0L;
};

static_assert(sizeof(vk_op_unary_push_constants) <= 128, "sizeof(vk_op_unary_push_constants) must be <= 128");

struct vk_op_pad_push_constants
{
  uint32_t ne;
  uint32_t ne00;
  uint32_t ne01;
  uint32_t ne02;
  uint32_t ne03;
  uint32_t nb00;
  uint32_t nb01;
  uint32_t nb02;
  uint32_t nb03;
  uint32_t ne10;
  uint32_t ne11;
  uint32_t ne12;
  uint32_t ne13;
  uint32_t nb10;
  uint32_t nb11;
  uint32_t nb12;
  uint32_t nb13;
  uint32_t misalign_offsets;

  uint32_t lp0;
  uint32_t rp0;
  uint32_t lp1;
  uint32_t rp1;
  uint32_t lp2;
  uint32_t rp2;
  uint32_t lp3;
  uint32_t rp3;
};


struct vk_op_binary_push_constants
{
  uint32_t ne;
  uint32_t ne00;
  uint32_t ne01;
  uint32_t ne02;
  uint32_t ne03;
  uint32_t nb00;
  uint32_t nb01;
  uint32_t nb02;
  uint32_t nb03;
  uint32_t ne10;
  uint32_t ne11;
  uint32_t ne12;
  uint32_t ne13;
  uint32_t nb10;
  uint32_t nb11;
  uint32_t nb12;
  uint32_t nb13;
  uint32_t ne20;
  uint32_t ne21;
  uint32_t ne22;
  uint32_t ne23;
  uint32_t nb20;
  uint32_t nb21;
  uint32_t nb22;
  uint32_t nb23;
  uint32_t misalign_offsets;
  float param1;
  float param2;
  int32_t param3;
};

struct vk_op_multi_add_push_constants
{
  // shape for dst
  uint32_t ne20;
  uint32_t ne21;
  uint32_t ne22;
  uint32_t ne23;

  // strides for srcs+dst
  uint32_t nb[MAX_PARAMETER_COUNT][4];

  uint32_t rms_partials;
};

struct vk_op_topk_moe_push_constants
{
  uint32_t n_rows;
  uint32_t n_expert_used;
};

struct vk_op_add_id_push_constants
{
  uint32_t ne0;
  uint32_t ne1;
  uint32_t s01;
  uint32_t s02;
  uint32_t s11;
  uint32_t s21;
};

struct vk_op_diag_mask_push_constants
{
  uint32_t ncols;
  uint32_t rows_per_channel;
  int32_t n_past;
};

struct vk_op_rope_push_constants
{
  uint32_t ncols;
  uint32_t n_dims;
  float freq_scale;
  uint32_t p_delta_rows;
  float freq_base;
  float ext_factor;
  float attn_factor;
  float corr_dims[2];
  float theta_scale;
  uint32_t has_ff;
  uint32_t ne02;
  uint32_t s1;
  uint32_t s2;
  int32_t sections[4];
  uint32_t is_back;
};

struct vk_op_soft_max_push_constants
{
  uint32_t KX;
  uint32_t KY;
  uint32_t ne00;
  uint32_t ne01;
  uint32_t ne02;
  uint32_t ne12;
  uint32_t ne13;
  uint32_t nb11;
  uint32_t nb12;
  uint32_t nb13;
  float scale;
  float max_bias;
  float m0;
  float m1;
  uint32_t n_head_log2;
  uint32_t nrows_x;
  uint32_t has_sinks;
};

struct vk_op_argsort_push_constants
{
  uint32_t ncols;
  int32_t order;
};

struct vk_op_im2col_push_constants
{
  uint64_t dst_addr;
  uint32_t batch_offset;
  uint32_t offset_delta;
  uint32_t IC;
  uint32_t IW;
  uint32_t IH;
  uint32_t OW;
  uint32_t OH;
  uint32_t KW;
  uint32_t KH;
  uint32_t pelements;
  uint32_t CHW;
  int32_t s0;
  int32_t s1;
  int32_t p0;
  int32_t p1;
  int32_t d0;
  int32_t d1;
};

struct vk_op_im2col_3d_push_constants
{
  uint64_t dst_addr;
  uint32_t nb10;
  uint32_t nb11;
  uint32_t nb12;
  uint32_t nb13;
  uint32_t s0;
  uint32_t s1;
  uint32_t s2;
  uint32_t p0;
  uint32_t p1;
  uint32_t p2;
  uint32_t d0;
  uint32_t d1;
  uint32_t d2;
  uint32_t IW;
  uint32_t IH;
  uint32_t ID;
  uint32_t IC;
  uint32_t KW;
  uint32_t OH;
  uint32_t KD_KH_KW;
  uint32_t KH_KW;
  uint32_t IC_KD_KH_KW;
  uint32_t N_OD_OH;
  uint32_t OD_OH;
  uint32_t OD_OH_OW_IC_KD_KH_KW;
  uint32_t OH_OW_IC_KD_KH_KW;
  uint32_t OW_IC_KD_KH_KW;
  uint32_t misalign_offsets;
};

struct vk_op_timestep_embedding_push_constants
{
  uint32_t nb1;
  uint32_t dim;
  uint32_t max_period;
};

struct vk_op_conv_transpose_1d_push_constants
{
  uint32_t Cout;
  uint32_t Cin;
  uint32_t K;
  uint32_t L;
  uint32_t KL;

  uint32_t nb01;
  uint32_t nb02;
  uint32_t nb11;
  uint32_t nb1;

  int32_t s0;
};

struct vk_op_pool2d_push_constants
{
  uint32_t IW;
  uint32_t IH;
  uint32_t OW;
  uint32_t OH;
  uint32_t OC;
  uint32_t pelements;
  uint32_t op;
  int32_t k0;
  int32_t k1;
  int32_t s0;
  int32_t s1;
  int32_t p0;
  int32_t p1;
};

struct vk_op_rwkv_wkv6_push_constants
{
  uint32_t B;
  uint32_t T;
  uint32_t C;
  uint32_t H;
};

struct vk_op_rwkv_wkv7_push_constants
{
  uint32_t B;
  uint32_t T;
  uint32_t C;
  uint32_t H;
};

struct vk_op_ssm_scan_push_constants
{
  uint32_t nb02, nb03, nb12, nb13;
  uint32_t nb21, nb22, nb31;
  uint32_t nb42, nb43, nb52, nb53;
  uint32_t s_off;
  uint32_t n_head, d_head, n_group, n_tok;
};

struct vk_op_ssm_conv_push_constants
{
  uint32_t nb01, nb02;
  uint32_t nb11;
  uint32_t dst_nb0, dst_nb1, dst_nb2;
  uint32_t nc, ncs, nr, n_t, n_s;
};

struct vk_op_conv2d_push_constants
{
  uint32_t Cout;
  uint32_t Cin;
  uint32_t N;

  uint32_t KW;
  uint32_t KH;
  uint32_t W;
  uint32_t H;
  uint32_t OW;
  uint32_t OH;

  uint32_t s0;
  uint32_t s1;
  uint32_t p0;
  uint32_t p1;
  uint32_t d0;
  uint32_t d1;

  uint32_t nb01;
  uint32_t nb02;
  uint32_t nb03;

  uint32_t nb11;
  uint32_t nb12;
  uint32_t nb13;

  uint32_t nb1;
  uint32_t nb2;
  uint32_t nb3;

  // init_fastdiv_values constants for dividing by KW, KW*KH, OW, OW*OH
  uint32_t KWmp;
  uint32_t KWL;
  uint32_t KWKHmp;
  uint32_t KWKHL;
  uint32_t OWmp;
  uint32_t OWL;
  uint32_t OWOHmp;
  uint32_t OWOHL;
};

struct vk_op_conv_transpose_2d_push_constants
{
  uint32_t Cout;
  uint32_t Cin;
  uint32_t N;

  uint32_t KW;
  uint32_t KH;
  uint32_t W;
  uint32_t H;
  uint32_t OW;
  uint32_t OH;

  uint32_t s0;
  uint32_t s1;
  uint32_t p0;
  uint32_t p1;
  uint32_t d0;
  uint32_t d1;

  uint32_t nb01;
  uint32_t nb02;
  uint32_t nb03;

  uint32_t nb11;
  uint32_t nb12;
  uint32_t nb13;

  uint32_t nb1;
  uint32_t nb2;
  uint32_t nb3;

  // init_fastdiv_values constants for dividing by KW, KW*KH, OW, OW*OH, s0, s1
  uint32_t KWmp;
  uint32_t KWL;
  uint32_t KWKHmp;
  uint32_t KWKHL;
  uint32_t OWmp;
  uint32_t OWL;
  uint32_t OWOHmp;
  uint32_t OWOHL;
  uint32_t s0mp;
  uint32_t s0L;
  uint32_t s1mp;
  uint32_t s1L;
};

struct vk_op_conv2d_dw_push_constants
{
  uint32_t ne;
  uint32_t batches;
  uint32_t channels;
  uint32_t dst_w;
  uint32_t dst_h;
  uint32_t src_w;
  uint32_t src_h;
  uint32_t knl_w;
  uint32_t knl_h;
  int32_t stride_x;
  int32_t stride_y;
  int32_t pad_x;
  int32_t pad_y;
  int32_t dilation_x;
  int32_t dilation_y;
};

struct vk_op_upscale_push_constants
{
  uint32_t ne;
  uint32_t a_offset;
  uint32_t d_offset;
  uint32_t ne00;
  uint32_t ne01;
  uint32_t nb00;
  uint32_t nb01;
  uint32_t nb02;
  uint32_t nb03;
  uint32_t ne10;
  uint32_t ne11;
  uint32_t ne12;
  uint32_t ne13;
  float sf0;
  float sf1;
  float sf2;
  float sf3;
  float pixel_offset;
};

struct vk_op_sum_rows_push_constants
{
  uint32_t n_cols;
  uint32_t ne01, ne02;
  uint32_t nb01, nb02, nb03;
  uint32_t nb11, nb12, nb13;
  float weight;
  uint32_t misalign_offsets;
  uint32_t ne0_12mp, ne0_12L;
  uint32_t ne0_1mp, ne0_1L;
};

#endif //MYPROJECT_VK_CONSTANT_H
