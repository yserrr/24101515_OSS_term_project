#ifndef MYPROJECT_VK_VISION_COMP_HPP
#define MYPROJECT_VK_VISION_COMP_HPP
#include "v_vk.hpp"
#include "vk_common.h"

std::array<uint32_t, 3> v_vk_get_conv_elements(const v_tensor* dst);
std::array<uint32_t, 3> v_vk_get_conv_transpose_2d_elements(const v_tensor* dst);
void v_vk_im2col(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, const v_tensor* src1, v_tensor* dst, bool dryrun = false);
void v_vk_im2col_3d(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, const v_tensor* src1, v_tensor* dst, bool dryrun = false);
void v_vk_pool_2d(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst, bool dryrun = false);
void v_vk_pool_2d_back(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst, bool dryrun = false);
void v_vk_conv_2d(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, const v_tensor* src1, v_tensor* dst, bool dryrun = false);
void v_vk_conv_2d_dw(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, const v_tensor* src1, v_tensor* dst, bool dryrun = false);
void v_vk_conv_transpose_2d(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, const v_tensor* src1, v_tensor* dst, bool dryrun = false);
void v_vk_ssm_conv(vk_backend_ctx* ctx, vk_context& subctx, v_tensor* dst, bool dryrun = false);
std::array<uint32_t, 3> v_vk_get_conv_transpose_2d_elements(const v_tensor* dst);


#endif //MYPROJECT_VK_VISION_COMP_HPP
