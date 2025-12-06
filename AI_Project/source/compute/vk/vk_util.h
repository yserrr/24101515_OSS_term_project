//
// Created by dlwog on 25. 11. 11..
//

#ifndef MYPROJECT_MML_VK_UTIL_H
#define MYPROJECT_MML_VK_UTIL_H
#include "vk_device.h"
#include "vk_context.h"
uint64_t vk_tensor_offset(const v_tensor* tensor);
uint32_t get_misalign_bytes(vk_backend_ctx* ctx, const v_tensor* t);
uint32_t get_subgroup_size(const std::string& pipeline_name, const vk_device_architecture& arch);
uint32_t get_fa_num_small_rows(FaCodePath path);
uint32_t get_fa_scalar_num_large_rows(uint32_t hsv);
bool v_vk_op_supports_incontiguous(v_operation op);
vk_pipeline v_vk_get_to_fp16(vk_backend_ctx* ctx, v_data_type type);
std::array<uint32_t, 2> fa_rows_cols(FaCodePath path, uint32_t hsk, uint32_t hsv, uint32_t clamp, v_data_type type, bool small_rows);
bool v_vk_dim01_contiguous(const v_tensor* tensor);
uint32_t fa_align(FaCodePath path, uint32_t hsk, uint32_t hsv, v_data_type type, bool small_rows);
uint32_t find_properties(const vk::PhysicalDeviceMemoryProperties* mem_props, vk::MemoryRequirements* mem_req, vk::MemoryPropertyFlags flags);
void vk_sync_buffers(vk_backend_ctx* ctx, vk_context& subctx);
void v_vk_print_gpu_info(size_t idx);
bool v_vk_instance_validation_ext_available();
bool v_vk_device_is_supported(const vk::PhysicalDevice& vkdev);
bool v_vk_instance_debug_utils_ext_available(const std::vector<vk::ExtensionProperties>& instance_extensions);
bool v_vk_instance_portability_enumeration_ext_available(const std::vector<vk::ExtensionProperties>& instance_extensions);
size_t vk_align_size(size_t width, size_t align);


#endif //MYPROJECT_MML_VK_UTIL_H
