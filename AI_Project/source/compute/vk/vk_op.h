#ifndef MYPROJECT_VK_OP_H
#define MYPROJECT_VK_OP_H
#include <array>
#include "vk_common.h"

enum vk_conv_shapes
{
  CONV_SHAPE_128x128,
  CONV_SHAPE_64x32,
  CONV_SHAPE_32x256,
  CONV_SHAPE_COUNT,
};

enum dmmv_wg_sizes
{
  DMMV_WG_SIZE_SUBGROUP,
  DMMV_WG_SIZE_LARGE,
  DMMV_WG_SIZE_COUNT,
};

enum FaCodePath
{
  FA_SCALAR,
  FA_COOPMAT1,
  FA_COOPMAT2,
};

enum shader_reduction_mode
{
  SHADER_REDUCTION_MODE_SHMEM,
  SHADER_REDUCTION_MODE_HYBRID,
  SHADER_REDUCTION_MODE_SUBGROUP,
  SHADER_REDUCTION_MODE_COUNT,
};


constexpr std::array topk_moe_norm{
  V_OP_SOFT_MAX,
  v_OP_RESHAPE,
  v_OP_ARGSORT,
  V_OP_VIEW,
  v_OP_GET_ROWS,
  v_OP_RESHAPE,
  v_OP_SUM_ROWS,
  v_OP_DIV,
  v_OP_RESHAPE
};

constexpr std::array topk_moe{
  V_OP_SOFT_MAX,
  v_OP_RESHAPE,
  v_OP_ARGSORT,
  V_OP_VIEW,
  v_OP_GET_ROWS
};


#endif //MYPROJECT_VK_OP_H
