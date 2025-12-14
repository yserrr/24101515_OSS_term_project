#include "v.hpp"
#include "v_quants.hpp"
#include "v_type.hpp"
#include <array>
#include <cstring>

v_type_trait::v_type_trait() {
  traits[v_TYPE_I8].type_name    = "i8";
  traits[v_TYPE_I8].blck_size    = 1;
  traits[v_TYPE_I8].type_size    = sizeof(int8_t);
  traits[v_TYPE_I8].is_quantized = false;

  traits[v_TYPE_I16] = {
      .type_name = "i16",
      .blck_size = 1,
      .type_size = sizeof(int16_t),
      .is_quantized = false,
    },
    traits[v_TYPE_I32] = {
      .type_name = "i32",
      .blck_size = 1,
      .type_size = sizeof(int32_t),
      .is_quantized = false,
    },
    traits[v_TYPE_I64] = {
      .type_name = "i64",
      .blck_size = 1,
      .type_size = sizeof(int64_t),
      .is_quantized = false,
    },
    traits[v_TYPE_F64] = {
      .type_name = "f64",
      .blck_size = 1,
      .type_size = sizeof(double),
      .is_quantized = false,
    },
    traits[v_TYPE_F32] = {
      .type_name = "f32",
      .blck_size = 1,
      .type_size = sizeof(float),
      .is_quantized = false,
    },
    traits[v_TYPE_F16] = {
      .type_name = "f16",
      .blck_size = 1,
      .type_size = sizeof(v_fp16_t),
      .is_quantized = false,
      .to_float = nullptr,
      .from_float_ref = nullptr,
    },
    traits[v_TYPE_Q4_0] = {
      .type_name = "q4_0",
      .blck_size = QK4_0,
      .type_size = sizeof(block_q4_0),
      .is_quantized = true,
      .to_float = (v_to_float_t)dequantize_row_q4_0,
      .from_float_ref = (v_from_float_t)quantize_row_q4_0_ref,
    },
    traits[v_TYPE_Q4_1] = {
      .type_name = "q4_1",
      .blck_size = QK4_1,
      .type_size = sizeof(block_q4_1),
      .is_quantized = true,
      .to_float = (v_to_float_t)dequantize_row_q4_1,
      .from_float_ref = (v_from_float_t)quantize_row_q4_1_ref,
    },
    traits[4] = {
      // v_TYPE_Q4_2
      .type_name = "DEPRECATED",
      .blck_size = 0,
      .type_size = 0,
      .is_quantized = false,
    },
    traits[5] = {
      // v_TYPE_Q4_3
      .type_name = "DEPRECATED",
      .blck_size = 0,
      .type_size = 0,
      .is_quantized = false,
    },
    traits[v_TYPE_Q5_0] = {
      .type_name = "q5_0",
      .blck_size = QK5_0,
      .type_size = sizeof(block_q5_0),
      .is_quantized = true,
      .to_float = (v_to_float_t)dequantize_row_q5_0,
      .from_float_ref = (v_from_float_t)quantize_row_q5_0_ref,
    },
    traits[v_TYPE_Q5_1] = {
      .type_name = "q5_1",
      .blck_size = QK5_1,
      .type_size = sizeof(block_q5_1),
      .is_quantized = true,
      .to_float = (v_to_float_t)dequantize_row_q5_1,
      .from_float_ref = (v_from_float_t)quantize_row_q5_1_ref,
    },
    traits[v_TYPE_Q8_0] = {
      .type_name = "q8_0",
      .blck_size = QK8_0,
      .type_size = sizeof(block_q8_0),
      .is_quantized = true,
      .to_float = (v_to_float_t)dequantize_row_q8_0,
      .from_float_ref = (v_from_float_t)quantize_row_q8_0_ref,
    },
    traits[v_TYPE_Q8_1] = {
      .type_name = "q8_1",
      .blck_size = QK8_1,
      .type_size = sizeof(block_q8_1),
      .is_quantized = true,
      .from_float_ref = (v_from_float_t)quantize_row_q8_1_ref,
    },
    traits[v_TYPE_MXFP4] = {
      .type_name = "mxfp4",
      .blck_size = QK_MXFP4,
      .type_size = sizeof(block_mxfp4),
      .is_quantized = true,
      .to_float = (v_to_float_t)dequantize_row_mxfp4,
      .from_float_ref = (v_from_float_t)quantize_row_mxfp4_ref,
    },
    traits[v_TYPE_Q2_K] = {
      .type_name = "q2_K",
      .blck_size = QK_K,
      .type_size = sizeof(block_q2_K),
      .is_quantized = true,
      .to_float = (v_to_float_t)dequantize_row_q2_K,
      .from_float_ref = (v_from_float_t)quantize_row_q2_K_ref,
    },
    traits[v_TYPE_Q3_K] = {
      .type_name = "q3_K",
      .blck_size = QK_K,
      .type_size = sizeof(block_q3_K),
      .is_quantized = true,
      .to_float = (v_to_float_t)dequantize_row_q3_K,
      .from_float_ref = (v_from_float_t)quantize_row_q3_K_ref,
    },
    traits[v_TYPE_Q4_K] = {
      .type_name = "q4_K",
      .blck_size = QK_K,
      .type_size = sizeof(block_q4_K),
      .is_quantized = true,
      .to_float = (v_to_float_t)dequantize_row_q4_K,
      .from_float_ref = (v_from_float_t)quantize_row_q4_K_ref,
    },
    traits[v_TYPE_Q5_K] = {
      .type_name = "q5_K",
      .blck_size = QK_K,
      .type_size = sizeof(block_q5_K),
      .is_quantized = true,
      .to_float = (v_to_float_t)dequantize_row_q5_K,
      .from_float_ref = (v_from_float_t)quantize_row_q5_K_ref,
    },
    traits[v_TYPE_Q6_K] = {
      .type_name = "q6_K",
      .blck_size = QK_K,
      .type_size = sizeof(block_q6_K),
      .is_quantized = true,
      .to_float = (v_to_float_t)dequantize_row_q6_K,
      .from_float_ref = (v_from_float_t)quantize_row_q6_K_ref,
    },
    traits[v_TYPE_IQ2_XXS] = {
      .type_name = "iq2_xxs",
      .blck_size = QK_K,
      .type_size = sizeof(block_iq2_xxs),
      .is_quantized = true,
      .to_float = (v_to_float_t)dequantize_row_iq2_xxs,
      .from_float_ref = NULL,
    },
    traits[v_TYPE_IQ2_XS] = {
      .type_name = "iq2_xs",
      .blck_size = QK_K,
      .type_size = sizeof(block_iq2_xs),
      .is_quantized = true,
      .to_float = (v_to_float_t)dequantize_row_iq2_xs,
      .from_float_ref = NULL,
    },
    traits[v_TYPE_IQ3_XXS] = {
      .type_name = "iq3_xxs",
      .blck_size = QK_K,
      .type_size = sizeof(block_iq3_xxs),
      .is_quantized = true,
      .to_float = (v_to_float_t)dequantize_row_iq3_xxs,
      .from_float_ref = (v_from_float_t)quantize_row_iq3_xxs_ref,
    },
    traits[v_TYPE_IQ3_S] = {
      .type_name = "iq3_s",
      .blck_size = QK_K,
      .type_size = sizeof(block_iq3_s),
      .is_quantized = true,
      .to_float = (v_to_float_t)dequantize_row_iq3_s,
      .from_float_ref = (v_from_float_t)quantize_row_iq3_s_ref,
    },
    traits[v_TYPE_IQ2_S] = {
      .type_name = "iq2_s",
      .blck_size = QK_K,
      .type_size = sizeof(block_iq2_s),
      .is_quantized = true,
      .to_float = (v_to_float_t)dequantize_row_iq2_s,
      .from_float_ref = (v_from_float_t)quantize_row_iq2_s_ref,
    },
    traits[v_TYPE_IQ1_S] = {
      .type_name = "iq1_s",
      .blck_size = QK_K,
      .type_size = sizeof(block_iq1_s),
      .is_quantized = true,
      .to_float = (v_to_float_t)dequantize_row_iq1_s,
      .from_float_ref = NULL,
    },
    traits[v_TYPE_IQ1_M] = {
      .type_name = "iq1_m",
      .blck_size = QK_K,
      .type_size = sizeof(block_iq1_m),
      .is_quantized = true,
      .to_float = (v_to_float_t)dequantize_row_iq1_m,
      .from_float_ref = NULL,
    },
    traits[v_TYPE_IQ4_NL] = {
      .type_name = "iq4_nl",
      .blck_size = QK4_NL,
      .type_size = sizeof(block_iq4_nl),
      .is_quantized = true,
      .to_float = (v_to_float_t)dequantize_row_iq4_nl,
      .from_float_ref = (v_from_float_t)quantize_row_iq4_nl_ref,
    },
    traits[v_TYPE_IQ4_XS] = {
      .type_name = "iq4_xs",
      .blck_size = QK_K,
      .type_size = sizeof(block_iq4_xs),
      .is_quantized = true,
      .to_float = (v_to_float_t)dequantize_row_iq4_xs,
      .from_float_ref = (v_from_float_t)quantize_row_iq4_xs_ref,
    },
    traits[v_TYPE_Q8_K] = {
      .type_name = "q8_K",
      .blck_size = QK_K,
      .type_size = sizeof(block_q8_K),
      .is_quantized = true,
    },
    traits[v_TYPE_BF16] = {
      .type_name = "bf16",
      .blck_size = 1,
      .type_size = sizeof(v_bf16_t),
      .is_quantized = false,
      .to_float = nullptr,
      .from_float_ref = nullptr,
    },
    traits[31] = {
      // v_TYPE_Q4_0_4_4
      .type_name = "TYPE_Q4_0_4_4 REMOVED, use Q4_0 with runtime repacking",
      .blck_size = 0,
      .type_size = 0,
      .is_quantized = false,
    },
    traits[32] = {
      // v_TYPE_Q4_0_4_8
      .type_name = "TYPE_Q4_0_4_8 REMOVED, use Q4_0 with runtime repacking",
      .blck_size = 0,
      .type_size = 0,
      .is_quantized = false,
    },
    traits[33] = {
      // v_TYPE_Q4_0_8_8
      .type_name = "TYPE_Q4_0_8_8 REMOVED, use Q4_0 with runtime repacking",
      .blck_size = 0,
      .type_size = 0,
      .is_quantized = false,
    },
    traits[v_TYPE_TQ1_0] = {
      .type_name = "tq1_0",
      .blck_size = QK_K,
      .type_size = sizeof(block_tq1_0),
      .is_quantized = true,
      .to_float = (v_to_float_t)dequantize_row_tq1_0,
      .from_float_ref = (v_from_float_t)quantize_row_tq1_0_ref,
    },
    traits[v_TYPE_TQ2_0] = {
      .type_name = "tq2_0",
      .blck_size = QK_K,
      .type_size = sizeof(block_tq2_0),
      .is_quantized = true,
      .to_float = (v_to_float_t)dequantize_row_tq2_0,
      .from_float_ref = (v_from_float_t)quantize_row_tq2_0_ref,
    },
    traits[36] = {
      // v_TYPE_IQ4_NL_4_4
      .type_name = "TYPE_IQ4_NL_4_4 REMOVED, use IQ4_NL with runtime repacking",
      .blck_size = 0,
      .type_size = 0,
      .is_quantized = false,
    },
    traits[37] = {
      // v_TYPE_IQ4_NL_4_8
      .type_name = "TYPE_IQ4_NL_4_8 REMOVED, use IQ4_NL with runtime repacking",
      .blck_size = 0,
      .type_size = 0,
      .is_quantized = false,
    },
    traits[38] = {
      // v_TYPE_IQ4_NL_8_8
      .type_name = "TYPE_IQ4_NL_8_8 REMOVED, use IQ4_NL with runtime repacking",
      .blck_size = 0,
      .type_size = 0,
      .is_quantized = false,
    };
}

static v_type_trait trait;

static_assert(V_OP_COUNT == 90, "V_OP_COUNT != 90");
static const char* v_UNARY_OP_NAME[v_UNARY_OP_COUNT] = {
  "ABS",
  "SGN",
  "NEG",

  "LOG"
  "STEP",
  "TANH",
  "ELU",
  "RELU",
  "SIGMOID",
  "GELU",
  "GELU_QUICK",
  "SILU",
  "HARDSWISH",
  "HARDSIGMOID",
  "EXP",
  "GELU_ERF",
  "XIELU",

  "FLOOR",
  "CEIL",
  "ROUND",
  "TRUNC",
};

const char* v_unary_op_name(enum v_unary_op op) {
  return v_UNARY_OP_NAME[op];
}

static const char* v_OP_SYMBOL[V_OP_COUNT] = {
  "none",

  "x",
  "x+y",
  "x[i]+y",
  "x+y",
  "view(x,nb,offset)+=y->x",
  "x-y",
  "x*y",
  "x/y",
  "x^2",
  "√x",
  "log(x)",
  "sin(x)",
  "cos(x)",
  "Σx",
  "Σx_k",
  "Σx/n",
  "argmax(x)",
  "count_equal(x)",
  "repeat(x)",
  "repeat_back(x)",
  "concat(x, y)",
  "silu_back(x)",
  "norm(x)",
  "rms_norm(x)",
  "rms_norm_back(x)",
  "group_norm(x)",
  "l2_norm(x)",

  "X*Y",
  "X[i]*Y",
  "X*Y",

  "x*v",
  "y-\\>view(x)",
  "x-\\>y",
  "cont(x)",
  "reshape(x)",
  "view(x)",
  "permute(x)",
  "transpose(x)",
  "get_rows(x)",
  "get_rows_back(x)",
  "set_rows(x)",
  "diag(x)",
  "diag_mask_inf(x)",
  "diag_mask_zero(x)",
  "soft_max(x)",
  "soft_max_back(x)",
  "rope(x)",
  "rope_back(x)",
  "clamp(x)",
  "conv_transpose_1d(x)",
  "im2col(x)",
  "im2col_back(x)",
  "im2col_3d(x)",
  "conv_2d(x)",
  "conv_3d(x)",
  "conv_2d_dw(x)",
  "conv_transpose_2d(x)",
  "pool_1d(x)",
  "pool_2d(x)",
  "pool_2d_back(x)",
  "upscale(x)",
  "pad(x)",
  "pad_reflect_1d(x)",
  "roll(x)",
  "arange(start, stop, step)",
  "timestep_embedding(timesteps, dim, max_period)",
  "argsort(x)",
  "leaky_relu(x)",

  "flash_attn_ext(x)",
  "flash_attn_back(x)",
  "ssm_conv(x)",
  "ssm_scan(x)",
  "win_part(x)",
  "win_unpart(x)",
  "get_rel_pos(x)",
  "add_rel_pos(x)",
  "rwkv_wkv6(k, v, r, tf, td, s)",
  "gated_linear_attn(k, v, q, gate, s)",
  "rwkv_wkv7(r, w, k, v, a, b, s)",

  "unary(x)",

  "map_custom(x)",
  "map_custom(x,y)",
  "map_custom(x,y,z)",

  "custom(x)",

  "cross_entropy_loss(x,y)",
  "cross_entropy_loss_back(x,y)",
  "adamw(x)",
  "sgd(x)",

  "glu(x)",
};

const char* v_OP_NAME[V_OP_COUNT] = {
  "NONE",

  "DUP",
  "ADD",
  "ADD_ID",
  "ADD1",
  "ACC",
  "SUB",
  "MUL",
  "DIV",
  "SQR",
  "SQRT",
  "LOG",
  "SIN",
  "COS",
  "SUM",
  "SUM_ROWS",
  "MEAN",
  "ARGMAX",
  "COUNT_EQUAL",
  "REPEAT",
  "REPEAT_BACK",
  "CONCAT",
  "SILU_BACK",
  "NORM",
  "RMS_NORM",
  "RMS_NORM_BACK",
  "GROUP_NORM",
  "L2_NORM",

  "MUL_MAT",
  "MUL_MAT_ID",
  "OUT_PROD",

  "SCALE",
  "SET",
  "CPY",
  "CONT",
  "RESHAPE",
  "VIEW",
  "PERMUTE",
  "TRANSPOSE",
  "GET_ROWS",
  "GET_ROWS_BACK",
  "SET_ROWS",
  "DIAG",
  "DIAG_MASK_INF",
  "DIAG_MASK_ZERO",
  "SOFT_MAX",
  "SOFT_MAX_BACK",
  "ROPE",
  "ROPE_BACK",
  "CLAMP",
  "CONV_TRANSPOSE_1D",
  "IM2COL",
  "IM2COL_BACK",
  "IM2COL_3D",
  "CONV_2D",
  "CONV_3D",
  "CONV_2D_DW",
  "CONV_TRANSPOSE_2D",
  "POOL_1D",
  "POOL_2D",
  "POOL_2D_BACK",
  "UPSCALE",
  "PAD",
  "PAD_REFLECT_1D",
  "ROLL",
  "ARANGE",
  "TIMESTEP_EMBEDDING",
  "ARGSORT",
  "LEAKY_RELU",

  "FLASH_ATTN_EXT",
  "FLASH_ATTN_BACK",
  "SSM_CONV",
  "SSM_SCAN",
  "WIN_PART",
  "WIN_UNPART",
  "GET_REL_POS",
  "ADD_REL_POS",
  "RWKV_WKV6",
  "GATED_LINEAR_ATTN",
  "RWKV_WKV7",

  "UNARY",

  "MAP_CUSTOM1",
  "MAP_CUSTOM2",
  "MAP_CUSTOM3",

  "CUSTOM",

  "CROSS_ENTROPY_LOSS",
  "CROSS_ENTROPY_LOSS_BACK",
  "OPT_STEP_ADAMW",
  "OPT_STEP_SGD",

  "GLU",
};
const char* v_GLU_OP_NAME[V_GLU_OP_COUNT] = {
  "REGLU",
  "GEGLU",
  "SWIGLU",
  "SWIGLU_OAI",
  "GEGLU_ERF",
  "GEGLU_QUICK",
};


const char* v_op_name(enum v_operation op) {
  return v_OP_NAME[op];
}


enum v_data_type v_ftype_to_v_type(enum v_ftype ftype) {
  enum v_data_type wtype = V_TYPE_COUNT;

  switch (ftype) {
    case v_FTYPE_ALL_F32: wtype = v_TYPE_F32;
      break;
    case v_FTYPE_MOSTLY_F16: wtype = v_TYPE_F16;
      break;
    case v_FTYPE_MOSTLY_BF16: wtype = v_TYPE_BF16;
      break;
    case v_FTYPE_MOSTLY_Q4_0: wtype = v_TYPE_Q4_0;
      break;
    case v_FTYPE_MOSTLY_Q4_1: wtype = v_TYPE_Q4_1;
      break;
    case v_FTYPE_MOSTLY_Q5_0: wtype = v_TYPE_Q5_0;
      break;
    case v_FTYPE_MOSTLY_Q5_1: wtype = v_TYPE_Q5_1;
      break;
    case v_FTYPE_MOSTLY_Q8_0: wtype = v_TYPE_Q8_0;
      break;
    case v_FTYPE_MOSTLY_MXFP4: wtype = v_TYPE_MXFP4;
      break;
    case v_FTYPE_MOSTLY_Q2_K: wtype = v_TYPE_Q2_K;
      break;
    case v_FTYPE_MOSTLY_Q3_K: wtype = v_TYPE_Q3_K;
      break;
    case v_FTYPE_MOSTLY_Q4_K: wtype = v_TYPE_Q4_K;
      break;
    case v_FTYPE_MOSTLY_Q5_K: wtype = v_TYPE_Q5_K;
      break;
    case v_FTYPE_MOSTLY_Q6_K: wtype = v_TYPE_Q6_K;
      break;
    case v_FTYPE_MOSTLY_IQ2_XXS: wtype = v_TYPE_IQ2_XXS;
      break;
    case v_FTYPE_MOSTLY_IQ2_XS: wtype = v_TYPE_IQ2_XS;
      break;
    case v_FTYPE_MOSTLY_IQ3_XXS: wtype = v_TYPE_IQ3_XXS;
      break;
    case v_FTYPE_MOSTLY_IQ1_S: wtype = v_TYPE_IQ1_S;
      break;
    case v_FTYPE_MOSTLY_IQ1_M: wtype = v_TYPE_IQ1_M;
      break;
    case v_FTYPE_MOSTLY_IQ4_NL: wtype = v_TYPE_IQ4_NL;
      break;
    case v_FTYPE_MOSTLY_IQ4_XS: wtype = v_TYPE_IQ4_XS;
      break;
    case v_FTYPE_MOSTLY_IQ3_S: wtype = v_TYPE_IQ3_S;
      break;
    case v_FTYPE_MOSTLY_IQ2_S: wtype = v_TYPE_IQ2_S;
      break;
    case v_FTYPE_UNKNOWN: wtype = V_TYPE_COUNT;
      break;
    case v_FTYPE_MOSTLY_Q4_1_SOME_F16: wtype = V_TYPE_COUNT;
      break;
  }

  V_ASSERT(wtype != V_TYPE_COUNT);

  return wtype;
}


const char* v_glu_op_name(v_glu_op op) {
  return v_GLU_OP_NAME[op];
}

const char* v_op_symbol(v_operation op) {
  return v_OP_SYMBOL[op];
}


const v_type_traits* v_get_type_traits(v_data_type type) {
  V_ASSERT(type < V_TYPE_COUNT);
  return &trait.traits[type];
}

int64_t block_size(v_data_type type) {
  return trait.traits[type].blck_size;
}

size_t v_type_size(v_data_type type) {
  return trait.traits[type].type_size;
}

double v_type_sizef(v_data_type type) {
  return ((double)(trait.traits[type].type_size)) / trait.traits[type].blck_size;
}

const char* v_type_name(v_data_type type) {
  return type < V_TYPE_COUNT ? trait.traits[type].type_name : "NONE";
}


bool v_is_quantized(enum v_data_type type) {
  return trait.traits[type].is_quantized;
}

size_t v_quantize_chunk(enum v_data_type type,
                        const float* src,
                        void* dst,
                        int64_t start,
                        int64_t nrows,
                        int64_t n_per_row,
                        const float* imatrix) {
  const int64_t n = (int64_t)nrows * n_per_row;

  if (v_quantize_requires_imatrix(type)) {
    V_ASSERT(imatrix != NULL);
  }

  //MML_ASSERT(start % type_traits[type].blck_size == 0);
  V_ASSERT(start % n_per_row == 0);

  const size_t start_row = start / n_per_row;
  const size_t row_size  = v_row_size(type, n_per_row);

  size_t result = 0;

  switch (type) {
    case v_TYPE_Q4_0: result =
        quantize_q4_0(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix);
      break;
    case v_TYPE_Q4_1: result =
        quantize_q4_1(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix);
      break;
    case v_TYPE_Q5_0: result =
        quantize_q5_0(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix);
      break;
    case v_TYPE_Q5_1: result =
        quantize_q5_1(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix);
      break;
    case v_TYPE_Q8_0: result =
        quantize_q8_0(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix);
      break;
    case v_TYPE_MXFP4: result = quantize_mxfp4(src + start,
                                               (char*)dst + start_row * row_size,
                                               nrows,
                                               n_per_row,
                                               imatrix);
      break;
    case v_TYPE_Q2_K: result =
        quantize_q2_K(src + start, (block_q2_K*)dst + start_row * row_size, nrows, n_per_row, imatrix);
      break;
    case v_TYPE_Q3_K: result =
        quantize_q3_K(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix);
      break;
    case v_TYPE_Q4_K: result =
        quantize_q4_K(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix);
      break;
    case v_TYPE_Q5_K: result =
        quantize_q5_K(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix);
      break;
    case v_TYPE_Q6_K: result =
        quantize_q6_K(src + start, (char*)dst + start_row * row_size, nrows, n_per_row, imatrix);
      break;
    case v_TYPE_TQ1_0: result = quantize_tq1_0(src + start,
                                               (char*)dst + start_row * row_size,
                                               nrows,
                                               n_per_row,
                                               imatrix);
      break;
    case v_TYPE_TQ2_0: result = quantize_tq2_0(src + start,
                                               (char*)dst + start_row * row_size,
                                               nrows,
                                               n_per_row,
                                               imatrix);
      break;
    case v_TYPE_IQ2_XXS: result = quantize_iq2_xxs(src + start,
                                                   (char*)dst + start_row * row_size,
                                                   nrows,
                                                   n_per_row,
                                                   imatrix);
      break;
    case v_TYPE_IQ2_XS: result = quantize_iq2_xs(src + start,
                                                 (char*)dst + start_row * row_size,
                                                 nrows,
                                                 n_per_row,
                                                 imatrix);
      break;
    case v_TYPE_IQ3_XXS: result = quantize_iq3_xxs(src + start,
                                                   (char*)dst + start_row * row_size,
                                                   nrows,
                                                   n_per_row,
                                                   imatrix);
      break;
    case v_TYPE_IQ3_S: result = quantize_iq3_s(src + start,
                                               (char*)dst + start_row * row_size,
                                               nrows,
                                               n_per_row,
                                               imatrix);
      break;
    case v_TYPE_IQ2_S: result = quantize_iq2_s(src + start,
                                               (char*)dst + start_row * row_size,
                                               nrows,
                                               n_per_row,
                                               imatrix);
      break;
    case v_TYPE_IQ1_S: result = quantize_iq1_s(src + start,
                                               (char*)dst + start_row * row_size,
                                               nrows,
                                               n_per_row,
                                               imatrix);
      break;
    case v_TYPE_IQ1_M: result = quantize_iq1_m(src + start,
                                               (char*)dst + start_row * row_size,
                                               nrows,
                                               n_per_row,
                                               imatrix);
      break;
    case v_TYPE_IQ4_NL: result = quantize_iq4_nl(src + start,
                                                 (char*)dst + start_row * row_size,
                                                 nrows,
                                                 n_per_row,
                                                 imatrix);
      break;
    case v_TYPE_IQ4_XS: result = quantize_iq4_xs(src + start,
                                                 (char*)dst + start_row * row_size,
                                                 nrows,
                                                 n_per_row,
                                                 imatrix);
      break;
    case v_TYPE_F16: {
      size_t elemsize = sizeof(v_fp16_t);
      result          = n * elemsize;
    }
    break;
    case v_TYPE_BF16: {
      size_t elemsize = sizeof(v_bf16_t);
      result          = n * elemsize;
    }
    break;
    case v_TYPE_F32: {
      size_t elemsize = sizeof(float);
      result          = n * elemsize;
      std::memcpy((uint8_t*)dst + start * elemsize, src + start, result);
    }
    break;
    default:
      V_ASSERT(false);
  }

  V_ASSERT(result == nrows * row_size);

  return result;
}

bool v_quantize_requires_imatrix(enum v_data_type type) {
  return
    type == v_TYPE_IQ2_XXS ||
    type == v_TYPE_IQ2_XS ||
    type == v_TYPE_IQ1_S; //   ||
  //type == v_TYPE_IQ1_M;
}
