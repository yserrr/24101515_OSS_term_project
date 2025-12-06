#ifndef MYPROJECT_VK_PIPELINE_H
#define MYPROJECT_VK_PIPELINE_H
#include "vk_common.h"
void vk_destory_pipeline(vk::Device& device, vk_pipeline& pipeline);
vk_pipeline v_vk_op_get_pipeline(vk_backend_ctx* ctx, const v_tensor* src0, const v_tensor* src1, const v_tensor* src2, const v_tensor* dst, v_operation op);
vk_pipeline v_vk_get_cpy_pipeline(vk_backend_ctx* ctx, const v_tensor* src, const v_tensor* dst, v_data_type to);
void vk_pipeline_allocate_descriptor_sets(vk_backend_ctx* ctx);
void v_pipeline_request_descriptor_sets(vk_backend_ctx* ctx, vk_pipeline& pipeline, uint32_t n);
void vk_pipeline_allocate_descriptor_sets(vk_backend_ctx* ctx);
vk_matmul_pipeline v_vk_get_mul_mat_mat_pipeline(vk_backend_ctx* ctx, v_data_type src0_type, v_data_type src1_type, v_prec prec);
vk_pipeline v_vk_get_dequantize_mul_mat_vec(vk_backend_ctx* ctx, v_data_type a_type, v_data_type b_type, uint32_t num_cols, uint32_t m, uint32_t k);
vk_matmul_pipeline v_vk_get_mul_mat_mat_id_pipeline(vk_backend_ctx* ctx, v_data_type src0_type, v_data_type src1_type, v_prec prec);
vk_pipeline v_vk_get_dequantize_mul_mat_vec_id(vk_backend_ctx* ctx, v_data_type a_type, v_data_type b_type);
uint32_t v_vk_guess_split_k(vk_backend_ctx* ctx, uint32_t m, uint32_t n, uint32_t k, bool disable_split_k, const vk_pipeline& pipeline);
vk_pipeline v_vk_guess_matmul_pipeline(vk_backend_ctx* ctx, vk_matmul_pipeline& mmp, uint32_t m, uint32_t n, bool aligned, v_data_type src0_type, v_data_type src1_type);
uint32_t v_vk_guess_matmul_pipeline_align(vk_backend_ctx* ctx, vk_matmul_pipeline& mmp, int m, int n, v_data_type src0_type, v_data_type src1_type);
vk_pipeline v_vk_get_quantize_pipeline(vk_backend_ctx* ctx, v_data_type type, bool use_x4_blocks);
template <typename T> void v_vk_dispatch_pipeline(vk_backend_ctx* ctx, vk_context& subctx, vk_pipeline& pipeline, std::initializer_list<vk::DescriptorBufferInfo> const& descriptor_buffer_infos, const T& push_constants, std::array<uint32_t, 3> elements);

/// @p_init: true if fields have been set by v_vk_create_pipeline
/// @p_need: compiled ->true;
/// @register_cnt: number of registers used, extracted from pipeline executable properties

struct vk_pipeline_struct {
  std::string name;
  vk::ShaderModule shader_module;
  vk::PipelineLayout layout;
  vk::Pipeline pipeline;
  uint32_t push_constant_size;
  uint32_t parameter_count;
  std::array<uint32_t, 3> wg_denoms;
  uint32_t align;
  bool initialized{};
  bool needed{};
  bool compiled{};
  uint32_t register_count{};
};

struct vk_matmul_pipeline_struct {
  vk_pipeline l;
  vk_pipeline m;
  vk_pipeline s;
  vk_pipeline a_l;
  vk_pipeline a_m;
  vk_pipeline a_s;
};


struct vk_matmul_pipeline2 {
  vk_matmul_pipeline2() {
    f16acc = std::make_shared<vk_matmul_pipeline_struct>();
    f32acc = std::make_shared<vk_matmul_pipeline_struct>();
  }

  vk_matmul_pipeline f32acc;
  vk_matmul_pipeline f16acc;
};



#endif //MYPROJECT_VK_PIPELINE_H
