#ifndef LLAMA_CPP_VK_DEVICE_H
#define LLAMA_CPP_VK_DEVICE_H
#include <mutex>
#include <map>
#include "vk_config.h"
#include "vk_queue.h"
#include "vk_pipeline.h"
#include "vk_perp_logger.hpp"
enum vk_device_architecture {
  OTHER,
  AMD_GCN,
  AMD_RDNA1,
  AMD_RDNA2,
  AMD_RDNA3,
  INTEL_XE2,
  NVIDIA_PRE_TURING,
};

vk_device_architecture get_device_architecture(const vk::PhysicalDevice& device);
void vk_load_shaders(vk_device& device);
void vk_instance_init();
vk_device v_vk_get_device(size_t idx);
void vk_init(vk_backend_ctx* ctx, size_t idx);
bool vk_khr_cooperative_matrix_support(const vk::PhysicalDeviceProperties& props,
                                         const vk::PhysicalDeviceDriverProperties& driver_props,
                                            vk_device_architecture arch);

vk_buffer v_vk_create_buffer_device(vk_device& device, size_t size) ;
int v_vk_get_device_count();

void v_vk_get_device_description(int device, char* description, size_t description_size);

struct vk_fa_pipeline_state
{
  vk_fa_pipeline_state(uint32_t HSK, uint32_t HSV, bool small_rows, FaCodePath path, bool aligned, bool f32acc)
    : HSK(HSK), HSV(HSV), small_rows(small_rows), path(path), aligned(aligned), f32acc(f32acc)
  {
  }

  uint32_t HSK, HSV;
  bool small_rows;
  FaCodePath path;
  bool aligned;
  bool f32acc;

  bool operator<(const vk_fa_pipeline_state& b) const
  {
    return std::tie(HSK, HSV, small_rows, path, aligned, f32acc) <
           std::tie(b.HSK, b.HSV, b.small_rows, b.path, b.aligned, b.f32acc);
  }
};




struct vk_device_struct
{
  std::recursive_mutex mutex;
  vk::PhysicalDevice physical_device;
  vk::PhysicalDeviceProperties properties;
  std::string name;
  uint64_t max_memory_allocation_size;
  uint64_t max_buffer_size;
  uint64_t suballocation_block_size;
  bool fp16;
  bool bf16;
  bool pipeline_robustness;
  vk::Device device;
  uint32_t vendor_id;
  vk::DriverId driver_id;
  vk_device_architecture architecture;
  vk_queue compute_queue;
  vk_queue transfer_queue;
  bool single_queue;
  uint32_t subgroup_size;
  uint32_t shader_core_count;
  bool uma;
  bool prefer_host_memory;
  bool float_controls_rte_fp16;
  bool subgroup_arithmetic;
  bool subgroup_shuffle;
  bool subgroup_ballot;
  bool subgroup_clustered;
  bool multi_add;
  bool shader_int64;
  bool buffer_device_address;

  bool add_rms_fusion;
  uint32_t partials_binding_alignment;

  bool integer_dot_product;
  // 0: default, 1: force mmvq, -1: disable mmvq
  int32_t mmvq_mode;

  bool subgroup_size_control;
  uint32_t subgroup_min_size;
  uint32_t subgroup_max_size;
  bool subgroup_require_full_support;

  bool coopmat_support;
  bool coopmat_acc_f32_support{};
  bool coopmat_acc_f16_support{};
  bool coopmat_bf16_support{};
  bool coopmat_support_16x16x16_f16acc{};
  bool coopmat_support_16x16x16_f32acc{};
  bool coopmat1_fa_support{};
  uint32_t coopmat_m;
  uint32_t coopmat_n;
  uint32_t coopmat_k;

  bool coopmat_int_support;
  uint32_t coopmat_int_m;
  uint32_t coopmat_int_n;
  uint32_t coopmat_int_k;

  bool coopmat2;

  bool pipeline_executable_properties_support{};

  size_t idx;

  bool mul_mat_l[v_TYPE_COUNT];
  bool mul_mat_m[v_TYPE_COUNT];
  bool mul_mat_s[v_TYPE_COUNT];
  bool mul_mat_id_l[v_TYPE_COUNT];
  bool mul_mat_id_m[v_TYPE_COUNT];
  bool mul_mat_id_s[v_TYPE_COUNT];

  // set to true to indicate that some shaders need to be compiled after the dryrun
  bool need_compiles{};

  vk::DescriptorSetLayout dsl;

  vk_matmul_pipeline pipeline_matmul_f32{};
  vk_matmul_pipeline pipeline_matmul_f32_f16{};
  vk_matmul_pipeline pipeline_matmul_bf16{};
  vk_matmul_pipeline2 pipeline_matmul_f16;
  vk_matmul_pipeline2 pipeline_matmul_f16_f32;

  vk_matmul_pipeline2 pipeline_dequant_mul_mat_mat[v_TYPE_COUNT];
  vk_matmul_pipeline2 pipeline_dequant_mul_mat_mat_f16[v_TYPE_COUNT];
  vk_matmul_pipeline2 pipeline_dequant_mul_mat_mat_q8_1[v_TYPE_COUNT];

  vk_matmul_pipeline pipeline_matmul_id_f32{};
  vk_matmul_pipeline pipeline_matmul_id_bf16{};
  vk_matmul_pipeline2 pipeline_matmul_id_f16;
  vk_matmul_pipeline2 pipeline_matmul_id_f16_f32;

  vk_matmul_pipeline2 pipeline_dequant_mul_mat_mat_id[v_TYPE_COUNT];

  vk_pipeline pipeline_matmul_split_k_reduce;
  vk_pipeline pipeline_quantize_q8_1;
  vk_pipeline pipeline_quantize_q8_1_x4;

  vk_pipeline pipeline_dequant[v_TYPE_COUNT];
  vk_pipeline pipeline_dequant_mul_mat_vec_f32_f32[DMMV_WG_SIZE_COUNT][v_TYPE_COUNT][mul_mat_vec_max_cols];
  vk_pipeline pipeline_dequant_mul_mat_vec_f16_f32[DMMV_WG_SIZE_COUNT][v_TYPE_COUNT][mul_mat_vec_max_cols];
  vk_pipeline pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_COUNT];

  vk_pipeline pipeline_dequant_mul_mat_vec_q8_1_f32[DMMV_WG_SIZE_COUNT][v_TYPE_COUNT][mul_mat_vec_max_cols];

  vk_pipeline pipeline_mul_mat_vec_p021_f16_f32[p021_max_gqa_ratio];
  vk_pipeline pipeline_mul_mat_vec_nc_f16_f32;
  vk_pipeline pipeline_get_rows[v_TYPE_COUNT];
  vk_pipeline pipeline_get_rows_f32[v_TYPE_COUNT];
  vk_pipeline pipeline_acc_f32;

  // [src0 0=fp32,1=fp16][src1 0=fp32,1=fp16][dst 0=fp32,1=fp16]
  vk_pipeline pipeline_add[2][2][2];
  vk_pipeline pipeline_add_norepeat[2][2][2];
  vk_pipeline pipeline_sub[2][2][2];
  vk_pipeline pipeline_sub_norepeat[2][2][2];
  vk_pipeline pipeline_mul[2][2][2];
  vk_pipeline pipeline_mul_norepeat[2][2][2];
  vk_pipeline pipeline_div[2][2][2];
  vk_pipeline pipeline_div_norepeat[2][2][2];
  vk_pipeline pipeline_add_rms[2][2][2];
  vk_pipeline pipeline_add_rms_norepeat[2][2][2];

  // indexed by num_additional_fused_ops == num_adds - 1
  vk_pipeline pipeline_multi_add[MAX_FUSED_ADDS];
  vk_pipeline pipeline_multi_add_rms[MAX_FUSED_ADDS];

  vk_pipeline pipeline_add_id_f32;

  vk_pipeline pipeline_concat_f32, pipeline_concat_f16, pipeline_concat_i32;
  vk_pipeline pipeline_upscale_nearest_f32, pipeline_upscale_bilinear_f32;
  vk_pipeline pipeline_scale_f32;
  vk_pipeline pipeline_sqr_f32;
  vk_pipeline pipeline_sqrt_f32;
  vk_pipeline pipeline_rand_f32;

  vk_pipeline pipeline_sin_f32;
  vk_pipeline pipeline_cos_f32;
  vk_pipeline pipeline_clamp_f32;
  vk_pipeline pipeline_pad_f32;
  vk_pipeline pipeline_roll_f32;
  vk_pipeline pipeline_repeat_f32, pipeline_repeat_back_f32;
  vk_pipeline pipeline_cpy_f32_f32, pipeline_cpy_f32_f16, pipeline_cpy_f16_f16, pipeline_cpy_f16_f32,
              pipeline_cpy_f32_bf16, pipeline_cpy_f32_i32, pipeline_cpy_i32_f32;
  vk_pipeline pipeline_contig_cpy_f32_f32, pipeline_contig_cpy_f32_f16, pipeline_contig_cpy_f16_f16,
              pipeline_contig_cpy_f16_f32, pipeline_contig_cpy_f32_bf16, pipeline_contig_cpy_f32_i32,
              pipeline_contig_cpy_i32_f32;
  vk_pipeline pipeline_cpy_f32_quant[v_TYPE_COUNT];
  vk_pipeline pipeline_cpy_quant_f32[v_TYPE_COUNT];
  vk_pipeline pipeline_set_rows_i32[v_TYPE_COUNT];
  vk_pipeline pipeline_set_rows_i64[v_TYPE_COUNT];
  vk_pipeline pipeline_norm_f32;
  vk_pipeline pipeline_group_norm_f32;
  vk_pipeline pipeline_rms_norm_f32;
  vk_pipeline pipeline_rms_norm_mul_f32;
  vk_pipeline pipeline_rms_norm_partials_f32;
  vk_pipeline pipeline_rms_norm_mul_partials_f32;
  vk_pipeline pipeline_rms_norm_back_f32;
  vk_pipeline pipeline_l2_norm_f32;

  // [src/dst 0=fp32,1=fp16]
  vk_pipeline pipeline_exp[2];
  vk_pipeline pipeline_log[2];
  vk_pipeline pipeline_gelu[2];
  vk_pipeline pipeline_gelu_erf[2];
  vk_pipeline pipeline_gelu_quick[2];
  vk_pipeline pipeline_silu[2];
  vk_pipeline pipeline_relu[2];
  vk_pipeline pipeline_tanh[2];
  vk_pipeline pipeline_sigmoid[2];
  vk_pipeline pipeline_hardsigmoid[2];
  vk_pipeline pipeline_hardswish[2];

  vk_pipeline pipeline_geglu[2];
  vk_pipeline pipeline_reglu[2];
  vk_pipeline pipeline_swiglu[2];
  vk_pipeline pipeline_swiglu_oai[2];
  vk_pipeline pipeline_geglu_erf[2];
  vk_pipeline pipeline_geglu_quick[2];

  vk_pipeline pipeline_leaky_relu_f32;
  vk_pipeline pipeline_silu_back_f32;
  vk_pipeline pipeline_diag_mask_inf_f32;
  vk_pipeline pipeline_soft_max_f32, pipeline_soft_max_f32_f16;
  vk_pipeline pipeline_soft_max_f32_wg512, pipeline_soft_max_f32_f16_wg512;
  vk_pipeline pipeline_soft_max_back_f32;
  vk_pipeline pipeline_rope_norm_f32, pipeline_rope_norm_f16;
  vk_pipeline pipeline_rope_neox_f32, pipeline_rope_neox_f16;
  vk_pipeline pipeline_rope_multi_f32, pipeline_rope_multi_f16;
  vk_pipeline pipeline_rope_vision_f32, pipeline_rope_vision_f16;
  vk_pipeline pipeline_argsort_f32[num_argsort_pipelines];
  vk_pipeline pipeline_sum_rows_f32;
  vk_pipeline pipeline_argmax_f32;
  vk_pipeline pipeline_count_equal_i32;
  vk_pipeline pipeline_im2col_f32, pipeline_im2col_f32_f16;
  vk_pipeline pipeline_im2col_3d_f32, pipeline_im2col_3d_f32_f16;
  vk_pipeline pipeline_timestep_embedding_f32;
  vk_pipeline pipeline_conv_transpose_1d_f32;
  vk_pipeline pipeline_pool2d_f32;
  vk_pipeline pipeline_pool2d_back_f32;
  vk_pipeline pipeline_rwkv_wkv6_f32;
  vk_pipeline pipeline_rwkv_wkv7_f32;
  vk_pipeline pipeline_ssm_scan_f32_d128;
  vk_pipeline pipeline_ssm_scan_f32_d256;
  vk_pipeline pipeline_ssm_conv_f32;
  vk_pipeline pipeline_opt_step_adamw_f32;
  vk_pipeline pipeline_opt_step_sgd_f32;
  vk_pipeline pipeline_conv2d_f32[CONV_SHAPE_COUNT];
  vk_pipeline pipeline_conv2d_f16_f32[CONV_SHAPE_COUNT];
  vk_pipeline pipeline_conv_transpose_2d_f32[CONV_SHAPE_COUNT];
  vk_pipeline pipeline_conv_transpose_2d_f16_f32[CONV_SHAPE_COUNT];
  vk_pipeline pipeline_conv2d_dw_whcn_f32, pipeline_conv2d_dw_whcn_f16_f32;
  vk_pipeline pipeline_conv2d_dw_cwhn_f32, pipeline_conv2d_dw_cwhn_f16_f32;

  std::map<vk_fa_pipeline_state, vk_pipeline> pipeline_flash_attn_f32_f16[v_TYPE_COUNT];
  vk_pipeline pipeline_flash_attn_split_k_reduce;
  // [2] is {!norm, norm}
  vk_pipeline pipeline_topk_moe[num_topk_moe_pipelines][2];
  std::vector<vk_pipeline_ref> all_pipelines;
  std::vector<std::tuple<void*, size_t, vk_buffer>> pinned_memory;
  vk::Fence fence;
  vk_buffer sync_staging;
  v_backend_buffer_type buffer_type;
  bool disable_fusion;
  bool disable_host_visible_vidmem;
  bool allow_sysmem_fallback;
  bool disable_graph_optimize;
#ifdef v_VULKAN_MEMORY_DEBUG
  std::unique_ptr<vk_memory_logger> memory_logger;
#endif
  // for v_VK_PERF_LOGGER
  std::unique_ptr<vk_perf_logger> perf_logger;
  vk::QueryPool query_pool;
  int32_t num_queries;
  ~vk_device_struct();
};


// GPU architecture identifier.
// Mapping of pipeline names to their specific subgroup sizes.
// Example: {"soft_max_f32", 64}
// Default subgroup size for this GPU.
// Defaults to 0 if not explicitly provided.
struct GpuPipelineConfig {
  vk_device_architecture arch;
  std::unordered_map<std::string, uint32_t> pipelines;
  uint32_t default_subgroup_size = 0;
};

extern std::vector<GpuPipelineConfig> gpu_pipeline_configs;


#endif //LLAMA_CPP_VK_DEVICE_H
