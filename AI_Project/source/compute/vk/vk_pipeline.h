#ifndef MYPROJECT_VK_PIPELINE_H
#define MYPROJECT_VK_PIPELINE_H
#include "vk_common.h"

struct vk_pipeline_struct
{
  std::string name;
  vk::ShaderModule shader_module;
  vk::PipelineLayout layout;
  vk::Pipeline pipeline;
  uint32_t push_constant_size;
  uint32_t parameter_count;
  std::array<uint32_t, 3> wg_denoms;
  uint32_t align;
  // true if fields have been set by v_vk_create_pipeline
  bool initialized{};
  // set to true to request the pipeline is compiled after the dryrun
  bool needed{};
  // set to true when the shader has been compiled
  bool compiled{};
  // number of registers used, extracted from pipeline executable properties
  uint32_t register_count{};
};

using vk_pipeline = std::shared_ptr<vk_pipeline_struct>;

struct vk_matmul_pipeline_struct
{
  vk_pipeline l;
  vk_pipeline m;
  vk_pipeline s;
  vk_pipeline a_l;
  vk_pipeline a_m;
  vk_pipeline a_s;
};

using vk_pipeline_ref = std::weak_ptr<vk_pipeline_struct>;
using vk_matmul_pipeline = std::shared_ptr<vk_matmul_pipeline_struct>;


struct vk_matmul_pipeline2
{
  vk_matmul_pipeline2()
  {
    f16acc = std::make_shared<vk_matmul_pipeline_struct>();
    f32acc = std::make_shared<vk_matmul_pipeline_struct>();
  }


  vk_matmul_pipeline f32acc;
  vk_matmul_pipeline f16acc;
};

enum vk_device_architecture
{
  OTHER,
  AMD_GCN,
  AMD_RDNA1,
  AMD_RDNA2,
  AMD_RDNA3,
  INTEL_XE2,
  NVIDIA_PRE_TURING,
};


struct GpuPipelineConfig
{
  // GPU architecture identifier.
  // Example: vk_device_architecture::AMD_GCN
  vk_device_architecture arch;

  // Mapping of pipeline names to their specific subgroup sizes.
  // Example: {"soft_max_f32", 64}
  std::unordered_map<std::string, uint32_t> pipelines;

  // Default subgroup size for this GPU.
  // Defaults to 0 if not explicitly provided.
  uint32_t default_subgroup_size = 0;
};

extern std::vector<GpuPipelineConfig> gpu_pipeline_configs;


#endif //MYPROJECT_VK_PIPELINE_H
