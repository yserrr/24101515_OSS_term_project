#include <array>

#include "vk_device.h"

#include "v-vulkan-kernels.hpp"
#include "vk_buffer.h"
#include "vk_context.h"
#include "vk_constant.h"
#include "vk_util.h"


void vk_destory_pipeline(vk::Device& device, vk_pipeline& pipeline);
void vk_destroy_buffer(vk_buffer& buf);
// variables to track number of compiles in progress
static uint32_t compile_count = 0;
static std::mutex compile_count_mutex;
static std::condition_variable compile_count_cond;
// The FA coopmat1 shader assumes 16x16x16 matrix multiply support.
// 128 threads split into four subgroups, each subgroup does 1/4
// of the Bc dimension.


static void v_vk_create_pipeline_func(vk_device& device,
                                         vk_pipeline& pipeline,
                                         size_t spv_size,
                                         const void* spv_data,
                                         const std::string entrypoint,
                                         uint32_t parameter_count,
                                         std::array<uint32_t, 3> wg_denoms,
                                         std::vector<uint32_t> specialization_constants,
                                         bool disable_robustness,
                                         bool require_full_subgroups,
                                         uint32_t required_subgroup_size) {
  VK_LOG_DEBUG(
    "v_vk_create_pipeline(" << device->name << ", " << pipeline->name << ", " << entrypoint << ", " <<
    parameter_count <<
    ", (" << wg_denoms[0] << "," << wg_denoms[1] << "," << wg_denoms[2] << "), specialization_constants, " <<
    disable_robustness << ", " << require_full_subgroups << ", " << required_subgroup_size << ")");
  V_ASSERT(parameter_count > 0);
  V_ASSERT(parameter_count <= MAX_PARAMETER_COUNT);
  V_ASSERT(wg_denoms[0] > 0 && wg_denoms[1] > 0 && wg_denoms[2] > 0); // NOLINT

  vk::ShaderModuleCreateInfo shader_module_create_info({}, spv_size, reinterpret_cast<const uint32_t*>(spv_data));
  pipeline->shader_module = device->device.createShaderModule(shader_module_create_info);

  vk::PushConstantRange pcr(
    vk::ShaderStageFlagBits::eCompute,
    0,
    pipeline->push_constant_size
  );

  vk::PipelineLayoutCreateInfo pipeline_layout_create_info(vk::PipelineLayoutCreateFlags(), device->dsl, pcr);
  pipeline->layout = device->device.createPipelineLayout(pipeline_layout_create_info);

  std::vector<vk::SpecializationMapEntry> specialization_entries(specialization_constants.size());

  for (size_t i = 0; i < specialization_constants.size(); i++) {
    specialization_entries[i].constantID = i;
    specialization_entries[i].offset     = i * sizeof(uint32_t);
    specialization_entries[i].size       = sizeof(uint32_t);
  }

  vk::SpecializationInfo specialization_info(
    specialization_entries.size(),
    specialization_entries.data(),
    specialization_constants.size() * sizeof(uint32_t),
    specialization_constants.data()
  );

  vk::PipelineShaderStageCreateFlags pipeline_shader_stage_create_flags{};

  if (device->subgroup_require_full_support && require_full_subgroups) {
    pipeline_shader_stage_create_flags |= vk::PipelineShaderStageCreateFlagBits::eRequireFullSubgroupsEXT;
  }

  vk::PipelineShaderStageCreateInfo pipeline_shader_create_info(
    pipeline_shader_stage_create_flags,
    vk::ShaderStageFlagBits::eCompute,
    pipeline->shader_module,
    entrypoint.c_str(),
    &specialization_info);

  vk::PipelineShaderStageRequiredSubgroupSizeCreateInfoEXT pipeline_shader_stage_required_subgroup_size_create_info;
  pipeline_shader_stage_required_subgroup_size_create_info.requiredSubgroupSize = required_subgroup_size;
  if (device->subgroup_size_control && required_subgroup_size > 0) {
    V_ASSERT(
      device->subgroup_min_size <= required_subgroup_size && required_subgroup_size <= device->subgroup_max_size);
    pipeline_shader_create_info.setPNext(&pipeline_shader_stage_required_subgroup_size_create_info);
  }

  vk::ComputePipelineCreateInfo compute_pipeline_create_info(
    device->pipeline_executable_properties_support
      ? vk::PipelineCreateFlagBits::eCaptureStatisticsKHR
      : vk::PipelineCreateFlags{},
    pipeline_shader_create_info,
    pipeline->layout);

  vk::PipelineRobustnessCreateInfoEXT rci;

  if (device->pipeline_robustness && disable_robustness) {
    rci.storageBuffers = vk::PipelineRobustnessBufferBehaviorEXT::eDisabled;
    rci.uniformBuffers = vk::PipelineRobustnessBufferBehaviorEXT::eDisabled;
    compute_pipeline_create_info.setPNext(&rci);
  }

  try { pipeline->pipeline = device->device.createComputePipeline(VK_NULL_HANDLE, compute_pipeline_create_info).value; }
  catch (const vk::SystemError& e) {
    std::cerr << "v_vulkan: Compute pipeline creation failed for " << pipeline->name << std::endl;
    std::cerr << "v_vulkan: " << e.what() << std::endl;
    throw e;
  }
  pipeline->compiled = true;

  if (vk_instance.debug_utils_support) {
    vk::DebugUtilsObjectNameInfoEXT duoni;
    duoni.objectType   = vk::ObjectType::ePipeline;
    duoni.pObjectName  = pipeline->name.c_str();
    duoni.objectHandle = /*reinterpret_cast*/(uint64_t)(static_cast<VkPipeline>(pipeline->pipeline));
    vk_instance.pfn_vkSetDebugUtilsObjectNameEXT(device->device, &static_cast<VkDebugUtilsObjectNameInfoEXT&>(duoni));
  }

  if (device->pipeline_executable_properties_support) {
    vk::PipelineExecutableInfoKHR executableInfo;
    executableInfo.pipeline = pipeline->pipeline;

    auto statistics = device->device.getPipelineExecutableStatisticsKHR(executableInfo);
    for (auto& s : statistics) {
      // "Register Count" is reported by NVIDIA drivers.
      if (strcmp(s.name, "Register Count") == 0) {
        VK_LOG_DEBUG(pipeline->name << " " << s.name << ": " << s.value.u64 << " registers");
        pipeline->register_count = (uint32_t)s.value.u64;
      }
    }
  }

  {
    std::lock_guard<std::recursive_mutex> guard(device->mutex);
    device->all_pipelines.push_back(pipeline);
  }

  {
    std::lock_guard<std::mutex> guard(compile_count_mutex);
    assert(compile_count > 0);
    compile_count--;
  }
  compile_count_cond.notify_all();
}


static bool v_vk_matmul_shmem_support(const vk_device& device, const std::vector<uint32_t>& warptile,
                                         bool mul_mat_id, v_data_type src0_type) {
  uint32_t lut_size = 0;
  switch (src0_type) {
  case v_TYPE_IQ1_S:
  case v_TYPE_IQ1_M:
    lut_size = 2 * 2048;
    break;
  case v_TYPE_IQ2_XXS:
    lut_size = 8 * 256;
    break;
  case v_TYPE_IQ2_XS:
    lut_size = 8 * 512;
    break;
  case v_TYPE_IQ2_S:
    lut_size = 8 * 1024;
    break;
  case v_TYPE_IQ3_XXS:
    lut_size = 4 * 256;
    break;
  case v_TYPE_IQ3_S:
    lut_size = 4 * 512;
    break;
  case v_TYPE_IQ4_NL:
  case v_TYPE_IQ4_XS:
  case v_TYPE_MXFP4:
    lut_size = 4 * 16;
    break;
  default:
    break;
  }

  // Needs to be kept up to date on shader changes
  const uint32_t bank_conflict_offset = device->coopmat_support ? 8 : 1;
  const uint32_t type_size            = device->fp16 ? sizeof(v_fp16_t) : sizeof(float);
  const uint32_t warps                = warptile[0] / warptile[10];

  const uint32_t load_bufs     = (warptile[1] + warptile[2]) * (warptile[3] + bank_conflict_offset) * type_size;
  const uint32_t mmid_row_ids  = mul_mat_id ? (warptile[2] * 2 * sizeof(uint16_t)) : 0;
  const uint32_t coopmat_stage = device->coopmat_support ? warptile[7] * warptile[8] / warps * sizeof(float) : 0;
  const uint32_t ballots_sh    = mul_mat_id ? (warps * 4 * sizeof(uint32_t)) : 0;

  const uint32_t total_size = load_bufs + mmid_row_ids + coopmat_stage + lut_size + ballots_sh;
  const bool supported      = total_size <= device->properties.limits.maxComputeSharedMemorySize;

  VK_LOG_DEBUG(
    "v_vk_matmul_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
    "mul_mat_id=" << mul_mat_id << ", src0_type=" << v_type_name(src0_type) << ", supported=" << supported);
  return supported;
}

// Pipeline configuration for RDNA1 GPUs.
static const std::unordered_map<std::string, uint32_t> rdna1_pipelines = {
  {"soft_max", 64}, {"im2col", 64},
  {"argmax", 64}, {"mul_mat_vec", 64},
  {"mul_mat_vec_f16", 32}, {"mul_mat_vec_f32_f16", 32}
};

// Pipeline configuration for RDNA2 GPUs.
static const std::unordered_map<std::string, uint32_t> rdna2_pipelines = {
  {"soft_max", 64}, {"im2col", 64},
};

static constexpr uint32_t RDNA_DEFAULT_SUBGROUP_SIZE = 32;
// Define configurations for different GPUs.
std::vector<GpuPipelineConfig> gpu_pipeline_configs = {
  {
    vk_device_architecture::AMD_RDNA1,
    {
      rdna1_pipelines,
    },
    RDNA_DEFAULT_SUBGROUP_SIZE
  },
  {
    vk_device_architecture::AMD_RDNA2,
    {
      rdna2_pipelines,
    },
    RDNA_DEFAULT_SUBGROUP_SIZE
  },
};


vk_device_architecture get_device_architecture(const vk::PhysicalDevice& device) {
  vk::PhysicalDeviceProperties props = device.getProperties();

  if (props.vendorID == VK_VENDOR_ID_AMD) {
    const std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

    bool amd_shader_core_properties = false;
    bool integer_dot_product        = false;
    bool subgroup_size_control      = false;

    for (const auto& properties : ext_props) {
      if (strcmp("VK_AMD_shader_core_properties", properties.extensionName) == 0) { amd_shader_core_properties = true; }
      else if (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0) {
        integer_dot_product = true;
      }
      else if (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) { subgroup_size_control = true; }
    }

    if (!amd_shader_core_properties || !integer_dot_product || !subgroup_size_control) {
      return vk_device_architecture::OTHER;
    }

    vk::PhysicalDeviceProperties2 props2;
    vk::PhysicalDeviceShaderCorePropertiesAMD shader_core_props_amd;
    vk::PhysicalDeviceShaderIntegerDotProductPropertiesKHR integer_dot_props;
    vk::PhysicalDeviceSubgroupSizeControlPropertiesEXT subgroup_size_control_props;

    props2.pNext                = &shader_core_props_amd;
    shader_core_props_amd.pNext = &integer_dot_props;
    integer_dot_props.pNext     = &subgroup_size_control_props;

    device.getProperties2(&props2);

    if (subgroup_size_control_props.maxSubgroupSize == 64 && subgroup_size_control_props.minSubgroupSize == 64) {
      return vk_device_architecture::AMD_GCN;
    }
    if (subgroup_size_control_props.maxSubgroupSize == 64 && subgroup_size_control_props.minSubgroupSize == 32) {
      // RDNA
      if (shader_core_props_amd.wavefrontsPerSimd == 20) { return vk_device_architecture::AMD_RDNA1; }
      if (integer_dot_props.integerDotProduct4x8BitPackedMixedSignednessAccelerated) {
        return vk_device_architecture::AMD_RDNA3;
      }
      return vk_device_architecture::AMD_RDNA2;
    }
  }
  else if (props.vendorID == VK_VENDOR_ID_INTEL) {
    const std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

    bool subgroup_size_control = false;

    for (const auto& properties : ext_props) {
      if (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) { subgroup_size_control = true; }
    }

    if (!subgroup_size_control) { return vk_device_architecture::OTHER; }
    vk::PhysicalDeviceProperties2 props2;
    vk::PhysicalDeviceSubgroupSizeControlPropertiesEXT subgroup_size_control_props;
    props2.pNext = &subgroup_size_control_props;
    device.getProperties2(&props2);
    if (subgroup_size_control_props.minSubgroupSize == 16) {
      // Xe2 architecture uses SIMD16 while previous Xe and Gen architecture uses SIMD8.
      // Minimum subgroup size matches the SIMD width so we distinguish architecture by checking this value.
      // https://www.intel.com/content/www/us/en/content-details/824434/2024-intel-tech-tour-xe2-and-lunar-lake-s-gpu.html
      // https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html
      return vk_device_architecture::INTEL_XE2;
    }
  }
  else if (props.vendorID == VK_VENDOR_ID_NVIDIA) {
    const std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();
    bool cooperative_matrix                              = false;
    // Detect "pre-turing" based on lack of coopmat support.
    for (const auto& properties : ext_props) {
      if (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0) {
        cooperative_matrix = true;
        break;
      }
    }

    if (!cooperative_matrix) { return vk_device_architecture::NVIDIA_PRE_TURING; }
  }
  return vk_device_architecture::OTHER;
}

vk_device_struct::~vk_device_struct() {
  VK_LOG_DEBUG("destroy device " << name);

  device.destroyFence(fence);

  vk_destroy_buffer(sync_staging);

  compute_queue.cmd_pool.destroy(device);
  transfer_queue.cmd_pool.destroy(device);

  for (auto& pipeline : all_pipelines) {
    if (pipeline.expired()) { continue; }

    vk_pipeline pl = pipeline.lock();
    vk_destory_pipeline(device, pl);
  }
  all_pipelines.clear();

  device.destroyDescriptorSetLayout(dsl);

  device.destroy();
}


void vk_destory_pipeline(vk::Device& device, vk_pipeline& pipeline) {
  VK_LOG_DEBUG("v_pipeline_destroy_pipeline(" << pipeline->name << ")");
  device.destroyPipelineLayout(pipeline->layout);
  device.destroyShaderModule(pipeline->shader_module);
  device.destroyPipeline(pipeline->pipeline);
}

static DispatchLoaderDynamic vk_default_dispatcher_instance;

DispatchLoaderDynamic& v_vk_default_dispatcher() { return vk_default_dispatcher_instance; }

void vk_create_queue(vk_device& device, vk_queue& q, uint32_t queue_family_index, uint32_t queue_index,
                     vk::PipelineStageFlags&& stage_flags, bool transfer_only) {
  VK_LOG_DEBUG("v_vk_create_queue()");
  std::lock_guard<std::recursive_mutex> guard(device->mutex);

  q.queue_family_index = queue_family_index;
  q.transfer_only      = transfer_only;

  q.cmd_pool.init(device, &q);

  q.queue = device->device.getQueue(queue_family_index, queue_index);

  q.stage_flags = stage_flags;
}

vk_device v_vk_get_device(size_t idx) {
  VK_LOG_DEBUG("v_vk_get_device(" << idx << ")");

  if (vk_instance.devices[idx] == nullptr) {
    VK_LOG_DEBUG("Initializing new vk_device");
    vk_device device         = std::make_shared<vk_device_struct>();
    vk_instance.devices[idx] = device;

    #ifdef V_VULKAN_MEMORY_DEBUG
    device->memory_logger = std::unique_ptr<vk_memory_logger>(new vk_memory_logger());
    #endif
    if (vk_perf_logger_enabled) { device->perf_logger = std::unique_ptr<vk_perf_logger>(new vk_perf_logger()); }

    size_t dev_num = vk_instance.device_indices[idx];

    std::vector<vk::PhysicalDevice> physical_devices = vk_instance.instance.enumeratePhysicalDevices();

    if (dev_num >= physical_devices.size()) {
      std::cerr << "v_vulkan: Device with index " << dev_num << " does not exist." << std::endl;
      throw std::runtime_error("Device not found");
    }

    device->physical_device                              = physical_devices[dev_num];
    const std::vector<vk::ExtensionProperties> ext_props = device->physical_device.enumerateDeviceExtensionProperties();

    device->architecture = get_device_architecture(device->physical_device);

    const char* v_VK_PREFER_HOST_MEMORY = getenv("v_VK_PREFER_HOST_MEMORY");
    device->prefer_host_memory             = v_VK_PREFER_HOST_MEMORY != nullptr;

    const char* v_VK_DISABLE_HOST_VISIBLE_VIDMEM = getenv("v_VK_DISABLE_HOST_VISIBLE_VIDMEM");
    device->disable_host_visible_vidmem             = v_VK_DISABLE_HOST_VISIBLE_VIDMEM != nullptr;

    const char* v_VK_ALLOW_SYSMEM_FALLBACK = getenv("v_VK_ALLOW_SYSMEM_FALLBACK");
    device->allow_sysmem_fallback             = v_VK_ALLOW_SYSMEM_FALLBACK != nullptr;

    const char* v_VK_DISABLE_GRAPH_OPTIMIZE = getenv("v_VK_DISABLE_GRAPH_OPTIMIZE");
    device->disable_graph_optimize             = v_VK_DISABLE_GRAPH_OPTIMIZE != nullptr;

    bool fp16_storage                           = false;
    bool fp16_compute                           = false;
    bool maintenance4_support                   = false;
    bool sm_builtins                            = false;
    bool amd_shader_core_properties2            = false;
    bool pipeline_robustness                    = false;
    bool coopmat2_support                       = false;
    bool pipeline_executable_properties_support = false;
    device->coopmat_support                     = false;
    device->integer_dot_product                 = false;
    bool bfloat16_support                       = false;

    for (const auto& properties : ext_props) {
      if (strcmp("VK_KHR_maintenance4", properties.extensionName) == 0) { maintenance4_support = true; }
      else if (strcmp("VK_KHR_16bit_storage", properties.extensionName) == 0) { fp16_storage = true; }
      else if (strcmp("VK_KHR_shader_float16_int8", properties.extensionName) == 0) { fp16_compute = true; }
      else if (strcmp("VK_NV_shader_sm_builtins", properties.extensionName) == 0) { sm_builtins = true; }
      else if (strcmp("VK_AMD_shader_core_properties2", properties.extensionName) == 0) {
        amd_shader_core_properties2 = true;
      }
      else if (strcmp("VK_EXT_pipeline_robustness", properties.extensionName) == 0) { pipeline_robustness = true; }
      else if (strcmp("VK_EXT_subgroup_size_control", properties.extensionName) == 0) {
        device->subgroup_size_control = true;
        #if defined(V_VULKAN_COOPMAT_GLSLC_SUPPORT)
      }
      else if (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
        !getenv("v_VK_DISABLE_COOPMAT")) {
        device->coopmat_support = true;
        device->coopmat_m       = 0;
        device->coopmat_n       = 0;
        device->coopmat_k       = 0;
        #endif
        #if defined(V_VULKAN_COOPMAT2_GLSLC_SUPPORT)
      }
      else if (strcmp("VK_NV_cooperative_matrix2", properties.extensionName) == 0 &&
        !getenv("v_VK_DISABLE_COOPMAT2")) {
        coopmat2_support = true;
        #endif
        #if defined(V_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
      }
      else if (strcmp("VK_KHR_shader_integer_dot_product", properties.extensionName) == 0 &&
        !getenv("v_VK_DISABLE_INTEGER_DOT_PRODUCT")) {
        device->integer_dot_product = true;
        #endif
        #if defined(V_VULKAN_BFLOAT16_GLSLC_SUPPORT)
      }
      else if (strcmp("VK_KHR_shader_bfloat16", properties.extensionName) == 0 &&
        !getenv("v_VK_DISABLE_BFLOAT16")) {
        bfloat16_support = true;
        #endif
      }
      else if (strcmp("VK_KHR_pipeline_executable_properties", properties.extensionName) == 0) {
        pipeline_executable_properties_support = true;
      }
    }

    vk::PhysicalDeviceProperties2 props2;
    vk::PhysicalDeviceMaintenance3Properties props3;
    vk::PhysicalDeviceMaintenance4Properties props4;
    vk::PhysicalDeviceSubgroupProperties subgroup_props;
    vk::PhysicalDeviceDriverProperties driver_props;
    vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV sm_props;
    vk::PhysicalDeviceShaderCoreProperties2AMD amd_shader_core_properties2_props;
    vk::PhysicalDeviceVulkan11Properties vk11_props;
    vk::PhysicalDeviceVulkan12Properties vk12_props;
    vk::PhysicalDeviceSubgroupSizeControlPropertiesEXT subgroup_size_control_props;
    vk::PhysicalDeviceShaderIntegerDotProductPropertiesKHR shader_integer_dot_product_props;

    props2.pNext         = &props3;
    props3.pNext         = &subgroup_props;
    subgroup_props.pNext = &driver_props;
    driver_props.pNext   = &vk11_props;
    vk11_props.pNext     = &vk12_props;

    VkBaseOutStructure* last_struct = (VkBaseOutStructure*)&vk12_props;

    if (maintenance4_support) {
      last_struct->pNext = (VkBaseOutStructure*)&props4;
      last_struct        = (VkBaseOutStructure*)&props4;
    }
    if (sm_builtins) {
      last_struct->pNext = (VkBaseOutStructure*)&sm_props;
      last_struct        = (VkBaseOutStructure*)&sm_props;
    }
    if (amd_shader_core_properties2) {
      last_struct->pNext = (VkBaseOutStructure*)&amd_shader_core_properties2_props;
      last_struct        = (VkBaseOutStructure*)&amd_shader_core_properties2_props;
    }
    if (device->subgroup_size_control) {
      last_struct->pNext = (VkBaseOutStructure*)&subgroup_size_control_props;
      last_struct        = (VkBaseOutStructure*)&subgroup_size_control_props;
    }

    #if defined(VK_NV_cooperative_matrix2)
    vk::PhysicalDeviceCooperativeMatrix2PropertiesNV coopmat2_props;
    if (coopmat2_support) {
      last_struct->pNext = (VkBaseOutStructure*)&coopmat2_props;
      last_struct        = (VkBaseOutStructure*)&coopmat2_props;
    }
    #endif

    if (device->integer_dot_product) {
      last_struct->pNext = (VkBaseOutStructure*)&shader_integer_dot_product_props;
      last_struct        = (VkBaseOutStructure*)&shader_integer_dot_product_props;
    }

    device->physical_device.getProperties2(&props2);
    device->properties = props2.properties;
    device->vendor_id  = device->properties.vendorID;
    device->driver_id  = driver_props.driverID;

    const char* v_VK_FORCE_MAX_ALLOCATION_SIZE = getenv("v_VK_FORCE_MAX_ALLOCATION_SIZE");

    if (v_VK_FORCE_MAX_ALLOCATION_SIZE != nullptr) {
      device->max_memory_allocation_size = std::stoull(v_VK_FORCE_MAX_ALLOCATION_SIZE);
    }
    else if (maintenance4_support) {
      device->max_memory_allocation_size = std::min(props3.maxMemoryAllocationSize, props4.maxBufferSize);
    }
    else { device->max_memory_allocation_size = props3.maxMemoryAllocationSize; }

    const char* v_VK_FORCE_MAX_BUFFER_SIZE = getenv("v_VK_FORCE_MAX_BUFFER_SIZE");

    if (v_VK_FORCE_MAX_BUFFER_SIZE != nullptr) {
      device->max_buffer_size = std::stoull(v_VK_FORCE_MAX_BUFFER_SIZE);
    }
    else if (maintenance4_support) { device->max_buffer_size = props4.maxBufferSize; }
    else { device->max_buffer_size = device->max_memory_allocation_size; }

    const char* v_VK_SUBALLOCATION_BLOCK_SIZE = getenv("v_VK_SUBALLOCATION_BLOCK_SIZE");

    if (v_VK_SUBALLOCATION_BLOCK_SIZE != nullptr) {
      device->suballocation_block_size = std::stoull(v_VK_SUBALLOCATION_BLOCK_SIZE);
    }
    else {
      // Limit batching of allocations to 1GB by default to avoid fragmentation issues
      device->suballocation_block_size = 1024 * 1024 * 1024;
    }
    device->suballocation_block_size = std::min(device->suballocation_block_size, device->max_memory_allocation_size);

    device->subgroup_size = subgroup_props.subgroupSize;
    device->uma           = device->properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;
    if (sm_builtins) { device->shader_core_count = sm_props.shaderSMCount; }
    else if (amd_shader_core_properties2) {
      device->shader_core_count = amd_shader_core_properties2_props.activeComputeUnitCount;
    }
    else { device->shader_core_count = 0; }
    device->float_controls_rte_fp16 = vk12_props.shaderRoundingModeRTEFloat16;

    device->subgroup_arithmetic = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
      (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eArithmetic);
    #ifdef __APPLE__
    // Workaround for subgroup arithmetic failing on MoltenVK with AMD GPUs (issue 15846)
    if (device->vendor_id == VK_VENDOR_ID_AMD) { device->subgroup_arithmetic = false; }
    #endif
    device->subgroup_shuffle = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
      (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eShuffle);
    device->subgroup_clustered = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
      (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eClustered);

    device->subgroup_ballot = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
      (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eBallot);

    const bool force_disable_f16 = getenv("v_VK_DISABLE_F16") != nullptr;

    device->fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

    if (!vk_khr_cooperative_matrix_support(device->properties, driver_props, device->architecture)) {
      device->coopmat_support = false;
    }
    device->coopmat_support = true;
    device->integer_dot_product = device->integer_dot_product && shader_integer_dot_product_props.
      integerDotProduct4x8BitPackedSignedAccelerated;

    std::vector<vk::QueueFamilyProperties> queue_family_props = device->physical_device.getQueueFamilyProperties();

    // Try to find a non-graphics compute queue and transfer-focused queues
    const uint32_t compute_queue_family_index = v_vk_find_queue_family_index(
      queue_family_props,
      vk::QueueFlagBits::eCompute,
      vk::QueueFlagBits::eGraphics,
      -1,
      1);
    const uint32_t transfer_queue_family_index = v_vk_find_queue_family_index(
      queue_family_props,
      vk::QueueFlagBits::eTransfer,
      vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eGraphics,
      compute_queue_family_index,
      1);

    const float priorities[] = {1.0f, 1.0f};
    device->single_queue     = compute_queue_family_index == transfer_queue_family_index && queue_family_props[
      compute_queue_family_index].queueCount == 1;

    std::vector<vk::DeviceQueueCreateInfo> device_queue_create_infos;
    if (compute_queue_family_index != transfer_queue_family_index) {
      device_queue_create_infos.push_back({vk::DeviceQueueCreateFlags(), compute_queue_family_index, 1, priorities});
      device_queue_create_infos.push_back(
        {vk::DeviceQueueCreateFlags(), transfer_queue_family_index, 1, priorities + 1});
    }
    else if (!device->single_queue) {
      device_queue_create_infos.push_back({vk::DeviceQueueCreateFlags(), compute_queue_family_index, 2, priorities});
    }
    else {
      device_queue_create_infos.push_back({vk::DeviceQueueCreateFlags(), compute_queue_family_index, 1, priorities});
    }
    vk::DeviceCreateInfo device_create_info;
    std::vector<const char*> device_extensions;
    vk::PhysicalDeviceFeatures device_features = device->physical_device.getFeatures();

    VkPhysicalDeviceFeatures2 device_features2;
    device_features2.sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    device_features2.pNext    = nullptr;
    device_features2.features = (VkPhysicalDeviceFeatures)device_features;

    VkPhysicalDeviceVulkan11Features vk11_features;
    vk11_features.pNext    = nullptr;
    vk11_features.sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    device_features2.pNext = &vk11_features;

    VkPhysicalDeviceVulkan12Features vk12_features;
    vk12_features.pNext = nullptr;
    vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk11_features.pNext = &vk12_features;

    last_struct = (VkBaseOutStructure*)&vk12_features;

    VkPhysicalDevicePipelineRobustnessFeaturesEXT pl_robustness_features;
    pl_robustness_features.pNext              = nullptr;
    pl_robustness_features.sType              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES_EXT;
    pl_robustness_features.pipelineRobustness = VK_FALSE;

    if (pipeline_robustness) {
      last_struct->pNext = (VkBaseOutStructure*)&pl_robustness_features;
      last_struct        = (VkBaseOutStructure*)&pl_robustness_features;
      device_extensions.push_back("VK_EXT_pipeline_robustness");
    }

    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroup_size_control_features;
    subgroup_size_control_features.pNext = nullptr;
    subgroup_size_control_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
    subgroup_size_control_features.computeFullSubgroups = false;
    subgroup_size_control_features.subgroupSizeControl = false;

    if (device->subgroup_size_control) {
      last_struct->pNext = (VkBaseOutStructure*)&subgroup_size_control_features;
      last_struct        = (VkBaseOutStructure*)&subgroup_size_control_features;
    }

    #if defined(VK_KHR_cooperative_matrix)
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmat_features;
    coopmat_features.pNext             = nullptr;
    coopmat_features.sType             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    coopmat_features.cooperativeMatrix = VK_FALSE;

    if (device->coopmat_support) {
      last_struct->pNext = (VkBaseOutStructure*)&coopmat_features;
      last_struct        = (VkBaseOutStructure*)&coopmat_features;
    }
    #endif

    #if defined(VK_NV_cooperative_matrix2)
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopmat2_features{};
    coopmat2_features.pNext = nullptr;
    coopmat2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV;
    if (coopmat2_support) {
      last_struct->pNext = (VkBaseOutStructure*)&coopmat2_features;
      last_struct        = (VkBaseOutStructure*)&coopmat2_features;
      device_extensions.push_back("VK_NV_cooperative_matrix2");
    }
    #endif

    #if defined(VK_KHR_shader_bfloat16)
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features{};
    bfloat16_features.pNext = nullptr;
    bfloat16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
    if (bfloat16_support) {
      last_struct->pNext = (VkBaseOutStructure*)&bfloat16_features;
      last_struct        = (VkBaseOutStructure*)&bfloat16_features;
      device_extensions.push_back("VK_KHR_shader_bfloat16");
    }
    #endif

    VkPhysicalDeviceMaintenance4Features maint4_features{};
    maint4_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
    if (maintenance4_support) {
      last_struct->pNext = (VkBaseOutStructure*)&maint4_features;
      last_struct        = (VkBaseOutStructure*)&maint4_features;
      device_extensions.push_back("VK_KHR_maintenance4");
    }

    VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR shader_integer_dot_product_features{};
    shader_integer_dot_product_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
    if (device->integer_dot_product) {
      last_struct->pNext = (VkBaseOutStructure*)&shader_integer_dot_product_features;
      last_struct        = (VkBaseOutStructure*)&shader_integer_dot_product_features;
      device_extensions.push_back("VK_KHR_shader_integer_dot_product");
    }

    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pep_features{};
    pep_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR;
    if (pipeline_executable_properties_support) {
      last_struct->pNext = (VkBaseOutStructure*)&pep_features;
      last_struct        = (VkBaseOutStructure*)&pep_features;
      device_extensions.push_back("VK_KHR_pipeline_executable_properties");
    }

    vkGetPhysicalDeviceFeatures2(device->physical_device, &device_features2);

    device->pipeline_executable_properties_support = pipeline_executable_properties_support;

    device->fp16 = device->fp16 && vk12_features.shaderFloat16;

    #if defined(VK_KHR_shader_bfloat16)
    device->bf16 = bfloat16_support && bfloat16_features.shaderBFloat16Type;
    #else
    device->bf16 = false;
    #endif

    device->pipeline_robustness = pl_robustness_features.pipelineRobustness;

    device->multi_add = vk12_props.shaderRoundingModeRTEFloat16 &&
      device->properties.limits.maxPushConstantsSize >= sizeof(vk_op_multi_add_push_constants) &&
      vk12_features.runtimeDescriptorArray &&
      device->vendor_id != VK_VENDOR_ID_INTEL &&
      getenv("v_VK_DISABLE_MULTI_ADD") == nullptr;

    device->shader_int64          = device_features2.features.shaderInt64;
    device->buffer_device_address = vk12_features.bufferDeviceAddress;

    if (device->subgroup_size_control) {
      device->subgroup_min_size = subgroup_size_control_props.minSubgroupSize;
      device->subgroup_max_size = subgroup_size_control_props.maxSubgroupSize;
      device_extensions.push_back("VK_EXT_subgroup_size_control");
    }

    device->subgroup_size_control = device->subgroup_size_control &&
      (subgroup_size_control_props.requiredSubgroupSizeStages &
        vk::ShaderStageFlagBits::eCompute) &&
      subgroup_size_control_features.subgroupSizeControl;

    device->subgroup_require_full_support = subgroup_size_control_features.computeFullSubgroups;

    #if defined(VK_KHR_cooperative_matrix)
    device->coopmat_support = device->coopmat_support && coopmat_features.cooperativeMatrix;

    // coopmat1 fa shader currently assumes 32 invocations per subgroup
    device->coopmat1_fa_support = device->coopmat_support && device->subgroup_require_full_support &&
      device->subgroup_size_control && device->subgroup_min_size <= 32 &&
      device->subgroup_max_size >= 32;
    #endif

    if (coopmat2_support) {
      #if defined(VK_NV_cooperative_matrix2) && defined(V_VULKAN_COOPMAT2_GLSLC_SUPPORT)
      if (coopmat2_features.cooperativeMatrixWorkgroupScope &&
        coopmat2_features.cooperativeMatrixFlexibleDimensions &&
        coopmat2_features.cooperativeMatrixReductions &&
        coopmat2_features.cooperativeMatrixConversions &&
        coopmat2_features.cooperativeMatrixPerElementOperations &&
        coopmat2_features.cooperativeMatrixTensorAddressing &&
        coopmat2_features.cooperativeMatrixBlockLoads &&
        vk12_features.bufferDeviceAddress) {
        std::vector<VkCooperativeMatrixFlexibleDimensionsPropertiesNV> flexible_dimensions;
        uint32_t count = 0;

        PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV
          _vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV =
            (PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV)
            vk_instance.instance.getProcAddr("vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV");

        _vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV(device->physical_device, &count, nullptr);

        VkCooperativeMatrixFlexibleDimensionsPropertiesNV empty_prop{};
        empty_prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_FLEXIBLE_DIMENSIONS_PROPERTIES_NV;
        flexible_dimensions.resize(count, empty_prop);

        _vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV(
          device->physical_device,
          &count,
          flexible_dimensions.data());

        bool found_fp16_128 = false,
             found_fp16_256 = false,
             found_fp32_128 = false,
             found_fp32_256 = false;
        // need to support fp16*fp16 with fp16/fp32 accumulator, for workgroupsize 128
        // with 32x16x16 and 256 with 32x32x16.
        for (auto& prop : flexible_dimensions) {
          if (prop.saturatingAccumulation == VK_FALSE &&
            prop.scope == VK_SCOPE_WORKGROUP_KHR &&
            prop.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
            prop.BType == VK_COMPONENT_TYPE_FLOAT16_KHR) {
            if (prop.workgroupInvocations == 128 &&
              prop.MGranularity <= 32 &&
              prop.NGranularity <= 16 &&
              prop.KGranularity <= 16) {
              if (prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                prop.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR) { found_fp16_128 = true; }
              if (prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                prop.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR) { found_fp32_128 = true; }
            }
            if (prop.workgroupInvocations == 256 &&
              prop.MGranularity <= 32 &&
              prop.NGranularity <= 32 &&
              prop.KGranularity <= 16) {
              if (prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                prop.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR) { found_fp16_256 = true; }
              if (prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                prop.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR) { found_fp32_256 = true; }
            }
          }
        }
        if (found_fp16_128 && found_fp16_256 &&
          found_fp32_128 && found_fp32_256 &&
          coopmat2_props.cooperativeMatrixFlexibleDimensionsMaxDimension >= 512) { device->coopmat2 = true; }
      }
      #endif
    }

    if (!vk11_features.storageBuffer16BitAccess) {
      std::cerr << "v_vulkan: device " << v_VK_NAME << idx << " does not support 16-bit storage." << std::endl;
      throw std::runtime_error("Unsupported device");
    }

    device_extensions.push_back("VK_KHR_16bit_storage");

    #ifdef V_VULKAN_VALIDATE
    device_extensions.push_back("VK_KHR_shader_non_semantic_info");
    #endif

    if (device->fp16) { device_extensions.push_back("VK_KHR_shader_float16_int8"); }

    #if defined(VK_KHR_cooperative_matrix)
    if (device->coopmat_support) {
      // Query supported shapes
      std::vector<VkCooperativeMatrixPropertiesKHR> cm_props;

      PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR =
        (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(
          vk_instance.instance,
          "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");

      uint32_t cm_props_num;

      pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device->physical_device, &cm_props_num, nullptr);

      cm_props.resize(cm_props_num);

      for (auto& prop : cm_props) { prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR; }

      pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device->physical_device, &cm_props_num, cm_props.data());

      VK_LOG_DEBUG("v_vulkan: Cooperative Matrix Shapes: " << cm_props.size());

      for (auto& prop : cm_props) {
        VK_LOG_DEBUG(
          "v_vulkan: M: " << prop.MSize << " N: " << prop.NSize << " K: " << prop.KSize << " A: " << vk::to_string((
            vk::ComponentTypeKHR)prop.AType) << " B: " << vk::to_string((vk::ComponentTypeKHR)prop.BType) << " C: " <<
          vk::to_string((vk::ComponentTypeKHR)prop.CType) << " Result: " << vk::to_string((vk::ComponentTypeKHR)prop.
            ResultType) << " saturatingAccumulation: " << prop.saturatingAccumulation << " scope: " << vk::to_string((vk
            ::ScopeKHR)prop.scope));

        if ((vk::ComponentTypeKHR)prop.AType == vk::ComponentTypeKHR::eFloat16 &&
          (vk::ComponentTypeKHR)prop.BType == vk::ComponentTypeKHR::eFloat16 &&
          (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup
        ) {
          if ((vk::ComponentTypeKHR)prop.CType == vk::ComponentTypeKHR::eFloat32 &&
            (vk::ComponentTypeKHR)prop.ResultType == vk::ComponentTypeKHR::eFloat32) {
            // coopmat sizes not set yet
            if (device->coopmat_m == 0) {
              device->coopmat_acc_f32_support = true;
              device->coopmat_m               = prop.MSize;
              device->coopmat_n               = prop.NSize;
              device->coopmat_k               = prop.KSize;
            }
            else if (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.
              KSize) {
              // Only enable if shape is identical
              device->coopmat_acc_f32_support = true;
            }
            if (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16) {
              device->coopmat_support_16x16x16_f32acc = true;
            }
          }
          else if ((vk::ComponentTypeKHR)prop.CType == vk::ComponentTypeKHR::eFloat16 &&
            (vk::ComponentTypeKHR)prop.ResultType == vk::ComponentTypeKHR::eFloat16) {
            // coopmat sizes not set yet
            if (device->coopmat_m == 0) {
              device->coopmat_acc_f16_support = true;
              device->coopmat_m               = prop.MSize;
              device->coopmat_n               = prop.NSize;
              device->coopmat_k               = prop.KSize;
            }
            else if (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.
              KSize) {
              // Only enable if shape is identical
              device->coopmat_acc_f16_support = true;
            }
            if (prop.MSize == 16 && prop.NSize == 16 && prop.KSize == 16) {
              device->coopmat_support_16x16x16_f16acc = true;
            }
          }
        }
        else if ((vk::ComponentTypeKHR)prop.AType == vk::ComponentTypeKHR::eSint8 &&
          (vk::ComponentTypeKHR)prop.BType == vk::ComponentTypeKHR::eSint8 &&
          (vk::ComponentTypeKHR)prop.CType == vk::ComponentTypeKHR::eSint32 &&
          (vk::ComponentTypeKHR)prop.ResultType == vk::ComponentTypeKHR::eSint32 &&
          (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup &&
          device->coopmat_int_m == 0
        ) {
          device->coopmat_int_support = true;
          device->coopmat_int_m       = prop.MSize;
          device->coopmat_int_n       = prop.NSize;
          device->coopmat_int_k       = prop.KSize;
        }
        #if defined(VK_KHR_shader_bfloat16) && defined(V_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        if (prop.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
          prop.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
          prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
          prop.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
          (vk::ScopeKHR)prop.scope == vk::ScopeKHR::eSubgroup
        ) {
          // coopmat sizes not set yet
          if (device->coopmat_m == 0) {
            device->coopmat_bf16_support = true;
            device->coopmat_m            = prop.MSize;
            device->coopmat_n            = prop.NSize;
            device->coopmat_k            = prop.KSize;
          }
          else if (device->coopmat_m == prop.MSize && device->coopmat_n == prop.NSize && device->coopmat_k == prop.
            KSize) {
            // Only enable if shape is identical
            device->coopmat_bf16_support = true;
          }
        }
        #endif
      }

      if (device->coopmat_m == 0 || !device->coopmat_acc_f32_support) {
        // No suitable matmul mode found

        v_LOG_DEBUG("v_vulkan: WARNING: No suitable matrix core mode found. Disabling matrix cores.\n");
        device->coopmat_support = false;
      }
      if (getenv("v_VK_DISABLE_BFLOAT16")) { device->coopmat_bf16_support = false; }
    }

    if (device->coopmat_support) { device_extensions.push_back("VK_KHR_cooperative_matrix"); }
    #if defined(VK_KHR_shader_bfloat16)
    if (device->coopmat_bf16_support) { device_extensions.push_back("VK_KHR_shader_bfloat16"); }
    #endif
    #endif
    device->name = v_VK_NAME + std::to_string(idx);

    device_create_info = {
      vk::DeviceCreateFlags(),
      device_queue_create_infos,
      {},
      device_extensions
    };
    device_create_info.setPNext(&device_features2);
    device->device = device->physical_device.createDevice(device_create_info);

    // Queues
    vk_create_queue(device,
                    device->compute_queue,
                    compute_queue_family_index,
                    0,
                    {
                      vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer
                    },
                    false);

    // Shaders
    // Disable matmul tile sizes early if performance low or not supported
    for (uint32_t i = 0; i < v_TYPE_COUNT; ++i) {
      switch (device->vendor_id) {
        #ifndef V_VULKAN_RUN_TESTS
      case VK_VENDOR_ID_AMD:
      case VK_VENDOR_ID_INTEL:
        device->mul_mat_l[i] = false;
        device->mul_mat_m[i]    = true;
        device->mul_mat_s[i]    = true;
        device->mul_mat_id_l[i] = false;
        device->mul_mat_id_m[i] = true;
        device->mul_mat_id_s[i] = true;
        break;
      case VK_VENDOR_ID_APPLE:
        device->mul_mat_l[i] = false;
        device->mul_mat_m[i]    = true;
        device->mul_mat_s[i]    = false;
        device->mul_mat_id_l[i] = false;
        device->mul_mat_id_m[i] = true;
        device->mul_mat_id_s[i] = false;
        break;
        #endif
      default:
        device->mul_mat_l[i] = true;
        device->mul_mat_m[i]    = true;
        device->mul_mat_s[i]    = true;
        device->mul_mat_id_l[i] = true;
        device->mul_mat_id_m[i] = true;
        device->mul_mat_id_s[i] = true;
        break;
      }
    }


    std::vector<vk::DescriptorSetLayoutBinding> dsl_binding;
    std::vector<vk::DescriptorBindingFlags> dsl_binding_flags;
    for (uint32_t i = 0; i < MAX_PARAMETER_COUNT; i++) {
      dsl_binding.push_back({i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
      dsl_binding_flags.push_back({});
    }

    vk::DescriptorSetLayoutBindingFlagsCreateInfo dslbfci = {dsl_binding_flags};

    vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info(
      {},
      dsl_binding);
    descriptor_set_layout_create_info.setPNext(&dslbfci);
    device->dsl = device->device.createDescriptorSetLayout(descriptor_set_layout_create_info);

    vk_load_shaders(device);

    if (!device->single_queue) {
      const uint32_t transfer_queue_index = compute_queue_family_index == transfer_queue_family_index ? 1 : 0;
      vk_create_queue(device,
                      device->transfer_queue,
                      transfer_queue_family_index,
                      transfer_queue_index,
                      {vk::PipelineStageFlagBits::eTransfer},
                      true);
    }
    else {
      // TODO: Use pointer or reference to avoid copy
      device->transfer_queue.copyFrom(device->compute_queue);
      device->transfer_queue.cmd_pool.init(device, &device->transfer_queue);
    }

    device->buffer_type = {
      /* .device   = */ v_backend_vk_reg_get_device(idx),
      /* .context  = */ new vk_buffer_type_context{device->name, device},
      /* .host  = */false
    };

    device->fence          = device->device.createFence({});
    device->idx            = idx;
    device->disable_fusion = getenv("v_VK_DISABLE_FUSION") != nullptr;

    device->add_rms_fusion = !device->disable_fusion &&
      device->subgroup_arithmetic &&
      device->vendor_id != VK_VENDOR_ID_INTEL;
    device->partials_binding_alignment =
      std::max(4u, (uint32_t)device->properties.limits.minStorageBufferOffsetAlignment);

    device->mmvq_mode = 0;
    if (getenv("v_VK_DISABLE_MMVQ")) { device->mmvq_mode = -1; }
    else if (getenv("v_VK_FORCE_MMVQ")) { device->mmvq_mode = 1; }

    return device;
  }
  return vk_instance.devices[idx];
}

bool vk_khr_cooperative_matrix_support(const vk::PhysicalDeviceProperties& props,
                                       const vk::PhysicalDeviceDriverProperties& driver_props,
                                       vk_device_architecture arch) {
  switch (props.vendorID) {
  case VK_VENDOR_ID_INTEL:
    // Only allowing Xe2 GPU at the moment since Xe2 GPU can gain significant performance boost,
    // while some older hardware (ex. Arc A770) has performance regressions
    return arch == vk_device_architecture::INTEL_XE2;
  case VK_VENDOR_ID_AMD:
    if (driver_props.driverID == vk::DriverId::eAmdProprietary || driver_props.driverID ==
      vk::DriverId::eAmdOpenSource) {
      // Workaround for AMD proprietary driver reporting support on all GPUs
      return arch == vk_device_architecture::AMD_RDNA3;
    }
    return true;
  default:
    return true;
  }
}


void vk_instance_init() {
  if (vk_instance_initialized) { return; }
  VK_LOG_DEBUG("v_vk_instance_init()");

  // See https://github.com/KhronosGroup/Vulkan-Hpp?tab=readme-ov-file#extensions--per-device-function-pointers-
  vk_default_dispatcher_instance.init(vkGetInstanceProcAddr);

  uint32_t api_version = vk::enumerateInstanceVersion();

  if (api_version < VK_API_VERSION_1_2) {
    std::cerr << "v_vulkan: Error: Vulkan 1.2 required." << std::endl;
    throw vk::SystemError(vk::Result::eErrorFeatureNotPresent, "Vulkan 1.2 required");
  }

  vk::ApplicationInfo app_info{"v-vulkan", 1, nullptr, 0, api_version};

  const std::vector<vk::ExtensionProperties> instance_extensions = vk::enumerateInstanceExtensionProperties();
  const bool validation_ext                                      = v_vk_instance_validation_ext_available();
  #ifdef __APPLE__
  const bool portability_enumeration_ext = v_vk_instance_portability_enumeration_ext_available(instance_extensions);
  #endif
  const bool debug_utils_ext = v_vk_instance_debug_utils_ext_available(instance_extensions) && getenv(
    "v_VK_DEBUG_MARKERS") != nullptr;
  std::vector<const char*> layers;

  if (validation_ext) { layers.push_back("VK_LAYER_KHRONOS_validation"); }
  std::vector<const char*> extensions;
  if (validation_ext) { extensions.push_back("VK_EXT_validation_features"); }
  #ifdef __APPLE__
  if (portability_enumeration_ext) { extensions.push_back("VK_KHR_portability_enumeration"); }
  #endif
  if (debug_utils_ext) { extensions.push_back("VK_EXT_debug_utils"); }
  vk::InstanceCreateInfo instance_create_info(vk::InstanceCreateFlags{}, &app_info, layers, extensions);
  #ifdef __APPLE__
  if (portability_enumeration_ext) {
    instance_create_info.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
  }
  #endif

  std::vector<vk::ValidationFeatureEnableEXT> features_enable;
  vk::ValidationFeaturesEXT validation_features;

  if (validation_ext) {
    features_enable     = {vk::ValidationFeatureEnableEXT::eBestPractices};
    validation_features = {
      features_enable,
      {},
    };
    validation_features.setPNext(nullptr);
    instance_create_info.setPNext(&validation_features);
    v_LOG_DEBUG("v_vulkan: Validation layers enabled\n");
  }
  vk_instance.instance    = vk::createInstance(instance_create_info);
  vk_instance_initialized = true;

  if (debug_utils_ext) {
    vk_instance.debug_utils_support              = true;
    vk_instance.pfn_vkSetDebugUtilsObjectNameEXT = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetInstanceProcAddr(
      vk_instance.instance,
      "vkSetDebugUtilsObjectNameEXT");
    vk_instance.pfn_vkQueueBeginDebugUtilsLabelEXT = (PFN_vkQueueBeginDebugUtilsLabelEXT)vkGetInstanceProcAddr(
      vk_instance.instance,
      "vkQueueBeginDebugUtilsLabelEXT");
    vk_instance.pfn_vkQueueEndDebugUtilsLabelEXT = (PFN_vkQueueEndDebugUtilsLabelEXT)vkGetInstanceProcAddr(
      vk_instance.instance,
      "vkQueueEndDebugUtilsLabelEXT");
    vk_instance.pfn_vkCmdBeginDebugUtilsLabelEXT = (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetInstanceProcAddr(
      vk_instance.instance,
      "vkCmdBeginDebugUtilsLabelEXT");
    vk_instance.pfn_vkCmdEndDebugUtilsLabelEXT = (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetInstanceProcAddr(
      vk_instance.instance,
      "vkCmdEndDebugUtilsLabelEXT");
    vk_instance.pfn_vkCmdInsertDebugUtilsLabelEXT = (PFN_vkCmdInsertDebugUtilsLabelEXT)vkGetInstanceProcAddr(
      vk_instance.instance,
      "vkCmdInsertDebugUtilsLabelEXT");
  }

  vk_perf_logger_enabled = getenv("v_VK_PERF_LOGGER") != nullptr;

  // See https://github.com/KhronosGroup/Vulkan-Hpp?tab=readme-ov-file#extensions--per-device-function-pointers-
  VULKAN_HPP_DEFAULT_DISPATCHER.init(vk_instance.instance);

  std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

  // Emulate behavior of CUDA_VISIBLE_DEVICES for Vulkan
  char* devices_env = getenv("v_VK_VISIBLE_DEVICES");
  if (devices_env != nullptr) {
    size_t num_available_devices = devices.size();

    std::string devices(devices_env);
    std::replace(devices.begin(), devices.end(), ',', ' ');

    std::stringstream ss(devices);
    size_t tmp;
    while (ss >> tmp) {
      if (tmp >= num_available_devices) {
        std::cerr << "v_vulkan: Invalid device index " << tmp << " in v_VK_VISIBLE_DEVICES." << std::endl;
        throw std::runtime_error("Invalid Vulkan device index");
      }
      vk_instance.device_indices.push_back(tmp);
    }
  }
  else {
    // If no vulkan devices are found, return early
    if (devices.empty()) {
      LOG_INFO("v_vulkan: No devices found.\n");
      return;
    }

    // Default to using all dedicated GPUs
    for (size_t i = 0; i < devices.size(); i++) {
      vk::PhysicalDeviceProperties2 new_props;
      vk::PhysicalDeviceDriverProperties new_driver;
      vk::PhysicalDeviceIDProperties new_id;
      new_props.pNext  = &new_driver;
      new_driver.pNext = &new_id;
      devices[i].getProperties2(&new_props);

      if ((new_props.properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu || new_props.properties.deviceType ==
        vk::PhysicalDeviceType::eIntegratedGpu) && v_vk_device_is_supported(devices[i])) {
        // Check if there are two physical devices corresponding to the same GPU
        auto old_device = std::find_if(
          vk_instance.device_indices.begin(),
          vk_instance.device_indices.end(),
          [&devices, &new_id](const size_t k) {
            vk::PhysicalDeviceProperties2 old_props;
            vk::PhysicalDeviceIDProperties old_id;
            old_props.pNext = &old_id;
            devices[k].getProperties2(&old_props);

            bool equals = std::equal(std::begin(old_id.deviceUUID),
                                     std::end(old_id.deviceUUID),
                                     std::begin(new_id.deviceUUID));
            equals = equals || (
              old_id.deviceLUIDValid && new_id.deviceLUIDValid &&
              std::equal(std::begin(old_id.deviceLUID),
                         std::end(old_id.deviceLUID),
                         std::begin(new_id.deviceLUID))
            );

            return equals;
          }
        );
        if (old_device == vk_instance.device_indices.end()) { vk_instance.device_indices.push_back(i); }
        else {
          // There can be two physical devices corresponding to the same GPU if there are 2 different drivers
          // This can cause error when splitting layers aross the devices, need to keep only 1
          VK_LOG_DEBUG("Device " << i << " and device " << *old_device << " have the same deviceUUID");

          vk::PhysicalDeviceProperties2 old_props;
          vk::PhysicalDeviceDriverProperties old_driver;
          old_props.pNext = &old_driver;
          devices[*old_device].getProperties2(&old_props);

          std::map<vk::DriverId, int> driver_priorities{};
          int old_priority = std::numeric_limits<int>::max();
          int new_priority = std::numeric_limits<int>::max();

          // Check https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDriverId.html for the list of driver id
          // Smaller number -> higher priority
          switch (old_props.properties.vendorID) {
          case VK_VENDOR_ID_AMD:
            driver_priorities[vk::DriverId::eMesaRadv] = 1;
            driver_priorities[vk::DriverId::eAmdOpenSource]  = 2;
            driver_priorities[vk::DriverId::eAmdProprietary] = 3;
            break;
          case VK_VENDOR_ID_INTEL:
            driver_priorities[vk::DriverId::eIntelOpenSourceMESA] = 1;
            driver_priorities[vk::DriverId::eIntelProprietaryWindows] = 2;
            break;
          case VK_VENDOR_ID_NVIDIA:
            driver_priorities[vk::DriverId::eNvidiaProprietary] = 1;
            #if defined(VK_API_VERSION_1_3) && VK_HEADER_VERSION >= 235
            driver_priorities[vk::DriverId::eMesaNvk] = 2;
            #endif
            break;
          }
          driver_priorities[vk::DriverId::eMesaDozen] = 100;

          if (driver_priorities.count(old_driver.driverID)) { old_priority = driver_priorities[old_driver.driverID]; }
          if (driver_priorities.count(new_driver.driverID)) { new_priority = driver_priorities[new_driver.driverID]; }

          if (new_priority < old_priority) {
            auto r = std::remove(vk_instance.device_indices.begin(), vk_instance.device_indices.end(), *old_device);
            vk_instance.device_indices.erase(r, vk_instance.device_indices.end());
            vk_instance.device_indices.push_back(i);

            VK_LOG_DEBUG(
              "Prioritize device " << i << " driver " << new_driver.driverName << " over device " << *old_device <<
              " driver " << old_driver.driverName);
          }
          else {
            VK_LOG_DEBUG(
              "Prioritize device " << *old_device << " driver " << old_driver.driverName << " over device " << i <<
              " driver " << new_driver.driverName << std::endl);
          }
        }
      }
    }

    // If no GPUs found, fall back to the first non-CPU device.
    // If only CPU devices are available, return without devices.
    if (vk_instance.device_indices.empty()) {
      for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i].getProperties().deviceType != vk::PhysicalDeviceType::eCpu) {
          vk_instance.device_indices.push_back(i);
          break;
        }
      }
    }

    if (vk_instance.device_indices.empty()) {
      LOG_INFO("v_vulkan: No devices found.\n");
      return;
    }
  }
  v_LOG_DEBUG("v_vulkan: Found %zu Vulkan devices:\n", vk_instance.device_indices.size());

  for (size_t i = 0; i < vk_instance.device_indices.size(); i++) {
    vk::PhysicalDevice vkdev                            = devices[vk_instance.device_indices[i]];
    std::vector<vk::ExtensionProperties> extensionprops = vkdev.enumerateDeviceExtensionProperties();

    bool membudget_supported = false;
    for (const auto& ext : extensionprops) {
      if (strcmp(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME, ext.extensionName) == 0) {
        membudget_supported = true;
        break;
      }
    }

    vk_instance.device_supports_membudget.push_back(membudget_supported);

    v_vk_print_gpu_info(i);
  }
}

void vk_init(vk_backend_ctx* ctx, size_t idx) {
  VK_LOG_DEBUG("v_vk_init(" << ctx->name << ", " << idx << ")");
  vk_instance_init();
  V_ASSERT(idx < vk_instance.device_indices.size());

  ctx->name = v_VK_NAME + std::to_string(idx);

  ctx->device = v_vk_get_device(idx);

  ctx->semaphore_idx = 0;
  ctx->event_idx     = 0;

  ctx->prealloc_size_x       = 0;
  ctx->prealloc_size_y       = 0;
  ctx->prealloc_size_split_k = 0;

  ctx->fence              = ctx->device->device.createFence({});
  ctx->almost_ready_fence = ctx->device->device.createFence({});

  ctx->compute_cmd_pool.init(ctx->device, &ctx->device->compute_queue);
  ctx->transfer_cmd_pool.init(ctx->device, &ctx->device->transfer_queue);

  #ifdef V_VULKAN_CHECK_RESULTS
  const char* skip_checks   = getenv("V_VULKAN_SKIP_CHECKS");
  vk_skip_checks            = (skip_checks == NULL ? 0 : atoi(skip_checks));
  const char* output_tensor = getenv("V_VULKAN_OUTPUT_TENSOR");
  vk_output_tensor          = (output_tensor == NULL ? 0 : atoi(output_tensor));
  #endif
}


void vk_load_shaders(vk_device& device) {
  VK_LOG_DEBUG("v_vk_load_shaders(" << device->name << ")");

  // some shaders have a minimum subgroup size
  const uint32_t subgroup_size_8  = std::max(device->subgroup_size, 8u);
  const uint32_t subgroup_size_16 = std::max(device->subgroup_size, 16u);
  const uint32_t subgroup_size_32 = std::max(device->subgroup_size, 32u);

  const uint32_t mul_mat_subgroup_size = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_size_control)
                                           ? device->subgroup_min_size
                                           : device->subgroup_size;
  const uint32_t mul_mat_subgroup_size_8  = std::max(mul_mat_subgroup_size, 8u);
  const uint32_t mul_mat_subgroup_size_16 = std::max(mul_mat_subgroup_size, 16u);
  const uint32_t mul_mat_subgroup_size_32 = std::max(mul_mat_subgroup_size, 32u);

  const bool subgroup_min_size_16 = (!device->subgroup_size_control && device->subgroup_size >= 16) ||
    (device->subgroup_size_control && device->subgroup_max_size >= 16);

  // mulmat
  std::vector<uint32_t> l_warptile, m_warptile, s_warptile,
                        l_warptile_id, m_warptile_id, s_warptile_id,
                        l_warptile_mmq, m_warptile_mmq, s_warptile_mmq,
                        l_warptile_mmq_int, m_warptile_mmq_int, s_warptile_mmq_int,
                        l_warptile_mmq_k, m_warptile_mmq_k, s_warptile_mmq_k,
                        l_warptile_mmqid, m_warptile_mmqid, s_warptile_mmqid;
  std::array<uint32_t, 3> l_wg_denoms, m_wg_denoms, s_wg_denoms,
                          l_mmq_wg_denoms, m_mmq_wg_denoms, s_mmq_wg_denoms,
                          l_mmq_wg_denoms_k, m_mmq_wg_denoms_k, s_mmq_wg_denoms_k,
                          l_mmqid_wg_denoms, m_mmqid_wg_denoms, s_mmqid_wg_denoms;

  uint32_t l_align, m_align, s_align;
  if (device->coopmat2) {
    // spec constants and tile sizes for non-quant matmul/matmul_id
    l_warptile  = {256, 128, 256, 64, 1};
    m_warptile  = {256, 128, 128, 64, 0};
    s_warptile  = {128, 64, 64, 64, 0};
    l_wg_denoms = {128, 256, 1};
    m_wg_denoms = {128, 128, 1};
    s_wg_denoms = {64, 64, 1};

    // spec constants and tile sizes for quant matmul (non-Qi_K)
    l_warptile_mmq  = {256, 128, 256, 64, 1};
    m_warptile_mmq  = {256, 128, 128, 64, 1};
    s_warptile_mmq  = {256, 32, 64, 128, 0};
    l_mmq_wg_denoms = {128, 256, 1};
    m_mmq_wg_denoms = {128, 128, 1};
    s_mmq_wg_denoms = {32, 64, 1};

    // spec constants and tile sizes for quant matmul (Qi_K)
    l_warptile_mmq_k  = {256, 128, 256, 64, 1};
    m_warptile_mmq_k  = {256, 128, 128, 64, 1};
    s_warptile_mmq_k  = {256, 32, 64, 128, 0};
    l_mmq_wg_denoms_k = {128, 256, 1};
    m_mmq_wg_denoms_k = {128, 128, 1};
    s_mmq_wg_denoms_k = {32, 64, 1};

    // spec constants and tile sizes for quant matmul_id
    l_warptile_mmqid  = {256, 128, 128, 16, 1, device->subgroup_size};
    m_warptile_mmqid  = {256, 128, 64, 16, 0, device->subgroup_size};
    s_warptile_mmqid  = {256, 128, 64, 16, 0, device->subgroup_size};
    l_mmqid_wg_denoms = {128, 128, 1};
    m_mmqid_wg_denoms = {128, 64, 1};
    s_mmqid_wg_denoms = {128, 64, 1};

    l_align = 128;
    m_align = 64;
    s_align = 32;
  }
  else {
    // Matrix cores require different warp group sizes
    const uint32_t tm_l = device->coopmat_support ? device->coopmat_m : 4;
    const uint32_t tm_m = device->coopmat_support ? device->coopmat_m : 4;
    const uint32_t tm_s = device->coopmat_support ? device->coopmat_m : 2;
    const uint32_t tn_l = device->coopmat_support ? device->coopmat_n : 4;
    const uint32_t tn_m = device->coopmat_support ? device->coopmat_n : 2;
    const uint32_t tn_s = device->coopmat_support ? device->coopmat_n : 2;
    const uint32_t tk_l = device->coopmat_support ? device->coopmat_k : 1;
    const uint32_t tk_m = device->coopmat_support ? device->coopmat_k : 1;
    const uint32_t tk_s = device->coopmat_support ? device->coopmat_k : 1;

    l_warptile = {128, 128, 128, 16, subgroup_size_8 * 2, 64, 2, tm_l, tn_l, tk_l, subgroup_size_8};
    m_warptile = {128, 64, 64, 16, subgroup_size_8, 32, 2, tm_m, tn_m, tk_m, subgroup_size_8};
    s_warptile = {subgroup_size_16, 32, 32, 16, 32, 32, 2, tm_s, tn_s, tk_s, subgroup_size_8};

    l_warptile_mmq = {128, 128, 128, 32, subgroup_size_8 * 2, 64, 2, tm_l, tn_l, tk_l, subgroup_size_8};
    m_warptile_mmq = {128, 64, 64, 32, subgroup_size_8, 32, 2, tm_m, tn_m, tk_m, subgroup_size_8};
    s_warptile_mmq = {subgroup_size_32, 32, 32, 32, 32, 32, 2, tm_s, tn_s, tk_s, subgroup_size_8};

    l_warptile_mmq_int = {128, 128, 128, 32, subgroup_size_8 * 2, 64, 2, 4, 4, 1, subgroup_size_8};
    m_warptile_mmq_int = {128, 64, 64, 32, subgroup_size_8, 32, 2, 2, 2, 1, subgroup_size_8};
    s_warptile_mmq_int = {subgroup_size_32, 32, 32, 32, 32, 32, 2, 2, 1, 1, subgroup_size_8};

    l_warptile_id = {
      128, 128, 128, 16, mul_mat_subgroup_size_16 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_subgroup_size_16
    };
    m_warptile_id = {128, 64, 64, 16, mul_mat_subgroup_size_16, 32, 2, tm_m, tn_m, tk_m, mul_mat_subgroup_size_16};
    s_warptile_id = {mul_mat_subgroup_size_16, 32, 32, 16, 32, 32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_size_16};

    l_warptile_mmqid = {
      128, 128, 128, 32, mul_mat_subgroup_size_8 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_subgroup_size_8
    };
    m_warptile_mmqid = {128, 64, 64, 32, mul_mat_subgroup_size_8, 32, 2, tm_m, tn_m, tk_m, mul_mat_subgroup_size_8};
    s_warptile_mmqid = {mul_mat_subgroup_size_32, 32, 32, 32, 32, 32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_size_8};

    // chip specific tuning
    if ((device->architecture == AMD_GCN) && (device->driver_id != vk::DriverId::eAmdProprietary)) {
      m_warptile_mmq   = m_warptile_mmq_int = {256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16};
      m_warptile_mmqid = {256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16};
    }

    l_mmq_wg_denoms = l_wg_denoms = {128, 128, 1};
    m_mmq_wg_denoms = m_wg_denoms = {64, 64, 1};
    s_mmq_wg_denoms = s_wg_denoms = {32, 32, 1};
    l_align         = 128;
    m_align         = 64;
    s_align         = 32;

    for (uint32_t i = 0; i < v_TYPE_COUNT; ++i) {
      v_data_type t = (v_data_type)i;
      // Disable medium and large matrix multiplication if not enough shared memory is available
      // Check mmq warptiles as the largest configuration
      // Throw an error if not enough for any matrix multiplication is available
      if (!v_vk_matmul_shmem_support(device, s_warptile_mmq, false, t)) {
        std::cerr << "v_vulkan: Error: Shared memory size too small for matrix multiplication." << std::endl;
        throw std::runtime_error("Shared memory size too small for matrix multiplication.");
      }
      else if (!v_vk_matmul_shmem_support(device, m_warptile_mmq, false, t)) {
        device->mul_mat_m[i] = false;
        device->mul_mat_l[i] = false;
      }
      else if (!v_vk_matmul_shmem_support(device, l_warptile_mmq, false, t)) { device->mul_mat_l[i] = false; }

      // Disable mul_mat_id if not enough shared memory is available
      if (!v_vk_matmul_shmem_support(device, s_warptile_mmqid, true, t)) {
        device->mul_mat_id_s[i] = false;
        device->mul_mat_id_m[i] = false;
        device->mul_mat_id_l[i] = false;
      }
      else if (!v_vk_matmul_shmem_support(device, m_warptile_mmqid, true, t)) {
        device->mul_mat_id_m[i] = false;
        device->mul_mat_id_l[i] = false;
      }
      else if (!v_vk_matmul_shmem_support(device, l_warptile_mmqid, true, t)) { device->mul_mat_id_l[i] = false; }
    }
  }

  if (!device->pipeline_matmul_f32) { device->pipeline_matmul_f32 = std::make_shared<vk_matmul_pipeline_struct>(); }
  if (!device->pipeline_matmul_f32_f16) {
    device->pipeline_matmul_f32_f16 = std::make_shared<vk_matmul_pipeline_struct>();
  }
  if (!device->pipeline_matmul_id_f32) {
    device->pipeline_matmul_id_f32 = std::make_shared<vk_matmul_pipeline_struct>();
  }
  if (!device->pipeline_matmul_bf16) { device->pipeline_matmul_bf16 = std::make_shared<vk_matmul_pipeline_struct>(); }
  if (!device->pipeline_matmul_id_bf16) {
    device->pipeline_matmul_id_bf16 = std::make_shared<vk_matmul_pipeline_struct>();
  }

  std::vector<std::future<void>> compiles;
  auto const& v_vk_create_pipeline = [&](vk_device& device, vk_pipeline& pipeline, const char* name, size_t spv_size,
                                            const void* spv_data, const char* entrypoint,
                                            uint32_t parameter_count, uint32_t push_constant_size,
                                            std::array<uint32_t, 3> wg_denoms,
                                            const std::vector<uint32_t>& specialization_constants,
                                            uint32_t align, bool disable_robustness = false,
                                            bool require_full_subgroups = false, uint32_t required_subgroup_size = 0) {
    if (!require_full_subgroups && required_subgroup_size == 0) {
      required_subgroup_size = get_subgroup_size(name, device->architecture);
    }

    if (!pipeline) { pipeline = std::make_shared<vk_pipeline_struct>(); }
    if (!pipeline->initialized) {
      pipeline->name               = name;
      pipeline->parameter_count    = parameter_count;
      pipeline->push_constant_size = push_constant_size;
      pipeline->wg_denoms          = wg_denoms;
      pipeline->align              = align;
      pipeline->initialized        = true;
    }

    if (!pipeline->needed || pipeline->compiled) { return; }
    {
      // wait until fewer than N compiles are in progress
      uint32_t N = std::max(1u, std::thread::hardware_concurrency());
      std::unique_lock<std::mutex> guard(compile_count_mutex);
      while (compile_count >= N) { compile_count_cond.wait(guard); }
      compile_count++;
    }

    compiles.push_back(std::async(v_vk_create_pipeline_func,
                                  std::ref(device),
                                  std::ref(pipeline),
                                  spv_size,
                                  spv_data,
                                  entrypoint,
                                  parameter_count,
                                  wg_denoms,
                                  specialization_constants,
                                  disable_robustness,
                                  require_full_subgroups,
                                  required_subgroup_size));
  };

  auto const& v_vk_create_pipeline2 = [&](vk_device& device, vk_pipeline& pipeline, const std::string& name,
                                             size_t spv_size, const void* spv_data, const char* entrypoint,
                                             uint32_t parameter_count, uint32_t push_constant_size,
                                             std::array<uint32_t, 3> wg_denoms,
                                             const std::vector<uint32_t>& specialization_constants,
                                             uint32_t align, bool disable_robustness = false,
                                             bool require_full_subgroups = false, uint32_t required_subgroup_size = 0) {
    return v_vk_create_pipeline(device,
                                   pipeline,
                                   name.c_str(),
                                   spv_size,
                                   spv_data,
                                   entrypoint,
                                   parameter_count,
                                   push_constant_size,
                                   wg_denoms,
                                   specialization_constants,
                                   align,
                                   disable_robustness,
                                   require_full_subgroups,
                                   required_subgroup_size);
  };

  auto const& fa_wg_denoms = [&](FaCodePath path, uint32_t hsk, uint32_t hsv, uint32_t clamp, v_data_type type,
                                 bool small_rows) -> std::array<uint32_t, 3> {
    return {fa_rows_cols(path, hsk, hsv, clamp, type, small_rows)[0], 1, 1};
  };

  auto const& fa_spec_constants = [&](FaCodePath path, uint32_t hsk, uint32_t hsv, uint32_t clamp, v_data_type type,
                                      bool small_rows) -> std::vector<uint32_t> {
    // For large number of rows, 128 invocations seems to work best.
    // For small number of rows (e.g. N==1), 256 works better. But matrix granularity for 256 is 32, so we
    // can't use 256 for D==80.
    // For scalar, use 128 (arbitrary)
    // The same D_split value is used for both HSK and HSV, so just base it on the union of the LSBs.
    const uint32_t D = (hsk | hsv);
    uint32_t wg_size = (path == FA_SCALAR || path == FA_COOPMAT1)
                         ? scalar_flash_attention_workgroup_size
                         : ((small_rows && (D % 32) == 0) ? 256 : 128);
    auto rows_cols = fa_rows_cols(path, hsk, hsv, clamp, type, small_rows);

    // D_split can't be larger than a subgroup because we use subgroupShuffle to reduce it.
    // D_split can't be larger than the LSB of D divided by 4 due to vectorization in the shader.
    const uint32_t D_lsb = D ^ (D & (D - 1));
    uint32_t D_split     = std::min(std::min(device->subgroup_size, 8u), D_lsb / 4);

    return {wg_size, rows_cols[0], rows_cols[1], hsk, hsv, clamp, D_split};
  };

  #define CREATE_FA(TYPE, NAMELC, FAPATH, SUFFIX) \
        for (auto &fa : device->pipeline_flash_attn_f32_f16[TYPE]) { \
            uint32_t HSK = fa.first.HSK; \
            uint32_t HSV = fa.first.HSV; \
            bool small_rows = fa.first.small_rows; \
            FaCodePath path = fa.first.path; \
            bool aligned = fa.first.aligned; \
            bool f32acc = fa.first.f32acc; \
            if (path == FAPATH) { \
                if (aligned) { \
                    if (f32acc) { \
                        v_vk_create_pipeline(device, fa.second, "flash_attn_f32_f16_aligned_f32acc" #NAMELC, flash_attn_f32_f16_ ## NAMELC ##            SUFFIX ## _len,  flash_attn_f32_f16_ ## NAMELC ##            SUFFIX ## _data,  "main", 6, sizeof(vk_flash_attn_push_constants), fa_wg_denoms(FAPATH, HSK,HSV,0,TYPE,small_rows), fa_spec_constants(FAPATH, HSK,HSV,0,TYPE,small_rows), fa_align(FAPATH,HSK,HSV,TYPE,small_rows), true, FAPATH==FA_COOPMAT1, (FAPATH==FA_COOPMAT1 ? 32 : 0));     \
                    } else { \
                        v_vk_create_pipeline(device, fa.second, "flash_attn_f32_f16_aligned_f16acc" #NAMELC, flash_attn_f32_f16_ ## NAMELC ## _f16acc ## SUFFIX ## _len,  flash_attn_f32_f16_ ## NAMELC ## _f16acc ## SUFFIX ## _data,  "main", 6, sizeof(vk_flash_attn_push_constants), fa_wg_denoms(FAPATH, HSK,HSV,0,TYPE,small_rows), fa_spec_constants(FAPATH, HSK,HSV,0,TYPE,small_rows), fa_align(FAPATH,HSK,HSV,TYPE,small_rows), true, FAPATH==FA_COOPMAT1, (FAPATH==FA_COOPMAT1 ? 32 : 0));     \
                    } \
                } else { \
                    if (f32acc) { \
                        v_vk_create_pipeline(device, fa.second, "flash_attn_f32_f16_f32acc"         #NAMELC, flash_attn_f32_f16_ ## NAMELC ##            SUFFIX ## _len,  flash_attn_f32_f16_ ## NAMELC ##            SUFFIX ## _data,  "main", 6, sizeof(vk_flash_attn_push_constants), fa_wg_denoms(FAPATH, HSK,HSV,1,TYPE,small_rows), fa_spec_constants(FAPATH, HSK,HSV,1,TYPE,small_rows), 1,                                        true, FAPATH==FA_COOPMAT1, (FAPATH==FA_COOPMAT1 ? 32 : 0));     \
                    } else { \
                        v_vk_create_pipeline(device, fa.second, "flash_attn_f32_f16_f16acc"         #NAMELC, flash_attn_f32_f16_ ## NAMELC ## _f16acc ## SUFFIX ## _len,  flash_attn_f32_f16_ ## NAMELC ## _f16acc ## SUFFIX ## _data,  "main", 6, sizeof(vk_flash_attn_push_constants), fa_wg_denoms(FAPATH, HSK,HSV,1,TYPE,small_rows), fa_spec_constants(FAPATH, HSK,HSV,1,TYPE,small_rows), 1,                                        true, FAPATH==FA_COOPMAT1, (FAPATH==FA_COOPMAT1 ? 32 : 0));     \
                    } \
                } \
            } \
        }

  CREATE_FA(v_TYPE_F32, f32, FA_SCALAR,)
  CREATE_FA(v_TYPE_F16, f16, FA_SCALAR,)
  CREATE_FA(v_TYPE_Q4_0, q4_0, FA_SCALAR,)
  CREATE_FA(v_TYPE_Q8_0, q8_0, FA_SCALAR,)
  #if defined(VK_KHR_cooperative_matrix) && defined(V_VULKAN_COOPMAT_GLSLC_SUPPORT)
  if (device->coopmat1_fa_support) {
    CREATE_FA(v_TYPE_F32, f32, FA_COOPMAT1, _cm1)
    CREATE_FA(v_TYPE_F16, f16, FA_COOPMAT1, _cm1)
    CREATE_FA(v_TYPE_Q4_0, q4_0, FA_COOPMAT1, _cm1)
    CREATE_FA(v_TYPE_Q8_0, q8_0, FA_COOPMAT1, _cm1)
  }
  #endif
  #if defined(VK_NV_cooperative_matrix2) && defined(V_VULKAN_COOPMAT2_GLSLC_SUPPORT)
  if (device->coopmat2) {
    CREATE_FA(v_TYPE_F32, f32, FA_COOPMAT2, _cm2)
    CREATE_FA(v_TYPE_F16, f16, FA_COOPMAT2, _cm2)
    CREATE_FA(v_TYPE_Q4_0, q4_0, FA_COOPMAT2, _cm2)
    CREATE_FA(v_TYPE_Q4_1, q4_1, FA_COOPMAT2, _cm2)
    CREATE_FA(v_TYPE_Q5_0, q5_0, FA_COOPMAT2, _cm2)
    CREATE_FA(v_TYPE_Q5_1, q5_1, FA_COOPMAT2, _cm2)
    CREATE_FA(v_TYPE_Q8_0, q8_0, FA_COOPMAT2, _cm2)
    CREATE_FA(v_TYPE_IQ4_NL, iq4_nl, FA_COOPMAT2, _cm2)
  }
  #endif
  #undef CREATE_FA

  #if defined(VK_NV_cooperative_matrix2) && defined(V_VULKAN_COOPMAT2_GLSLC_SUPPORT)
  if (device->coopmat2) {
    // Create 6 variants, {s,m,l}x{unaligned,aligned}
    #define CREATE_MM(PIPELINE_NAME, NAMELC, F16ACC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT) \
        v_vk_create_pipeline(device, device-> PIPELINE_NAME ->l, #NAMELC #F16ACC "_l", NAMELC ## F16ACC ## _cm2_len, NAMELC ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1);   \
        v_vk_create_pipeline(device, device-> PIPELINE_NAME ->m, #NAMELC #F16ACC "_m", NAMELC ## F16ACC ## _cm2_len, NAMELC ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1);   \
        v_vk_create_pipeline(device, device-> PIPELINE_NAME ->s, #NAMELC #F16ACC "_s", NAMELC ## F16ACC ## _cm2_len, NAMELC ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1);   \
        v_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_l, #NAMELC #F16ACC "_aligned_l", NAMELC ## _aligned ## F16ACC ## _cm2_len, NAMELC ## _aligned ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, l_align);   \
        v_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_m, #NAMELC #F16ACC "_aligned_m", NAMELC ## _aligned ## F16ACC ## _cm2_len, NAMELC ## _aligned ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, m_align);   \
        v_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_s, #NAMELC #F16ACC "_aligned_s", NAMELC ## _aligned ## F16ACC ## _cm2_len, NAMELC ## _aligned ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, s_align);
    // Create 2 variants, {f16,f32} accumulator
    #define CREATE_MM2(PIPELINE_NAME, NAMELC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT) \
        CREATE_MM(PIPELINE_NAME . f16acc, NAMELC, _f16acc, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT)   \
        CREATE_MM(PIPELINE_NAME . f32acc, NAMELC, , WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT)
    CREATE_MM2(pipeline_matmul_f16, matmul_f16, wg_denoms, warptile, vk_mat_mat_push_constants, 3)
    #if defined(V_VULKAN_BFLOAT16_GLSLC_SUPPORT)
    if (device->coopmat_bf16_support) {
      CREATE_MM(pipeline_matmul_bf16, matmul_bf16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3)
    }
    #endif
    CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[v_TYPE_Q4_0],
               matmul_q4_0_f16,
               mmq_wg_denoms,
               warptile_mmq,
               vk_mat_mat_push_constants,
               3)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[v_TYPE_Q4_1],
               matmul_q4_1_f16,
               mmq_wg_denoms,
               warptile_mmq,
               vk_mat_mat_push_constants,
               3)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[v_TYPE_Q5_0],
               matmul_q5_0_f16,
               mmq_wg_denoms,
               warptile_mmq,
               vk_mat_mat_push_constants,
               3)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[v_TYPE_Q5_1],
               matmul_q5_1_f16,
               mmq_wg_denoms,
               warptile_mmq,
               vk_mat_mat_push_constants,
               3)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[v_TYPE_Q8_0],
               matmul_q8_0_f16,
               mmq_wg_denoms,
               warptile_mmq,
               vk_mat_mat_push_constants,
               3)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[v_TYPE_Q2_K],
               matmul_q2_k_f16,
               mmq_wg_denoms_k,
               warptile_mmq_k,
               vk_mat_mat_push_constants,
               3)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[v_TYPE_Q3_K],
               matmul_q3_k_f16,
               mmq_wg_denoms_k,
               warptile_mmq_k,
               vk_mat_mat_push_constants,
               3)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[v_TYPE_Q4_K],
               matmul_q4_k_f16,
               mmq_wg_denoms_k,
               warptile_mmq_k,
               vk_mat_mat_push_constants,
               3)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[v_TYPE_Q5_K],
               matmul_q5_k_f16,
               mmq_wg_denoms_k,
               warptile_mmq_k,
               vk_mat_mat_push_constants,
               3)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[v_TYPE_Q6_K],
               matmul_q6_k_f16,
               mmq_wg_denoms_k,
               warptile_mmq_k,
               vk_mat_mat_push_constants,
               3)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[v_TYPE_IQ1_S],
               matmul_iq1_s_f16,
               mmq_wg_denoms,
               warptile_mmq,
               vk_mat_mat_push_constants,
               3)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[v_TYPE_IQ1_M],
               matmul_iq1_m_f16,
               mmq_wg_denoms,
               warptile_mmq,
               vk_mat_mat_push_constants,
               3)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[v_TYPE_IQ2_XXS],
               matmul_iq2_xxs_f16,
               mmq_wg_denoms,
               warptile_mmq,
               vk_mat_mat_push_constants,
               3)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[v_TYPE_IQ2_XS],
               matmul_iq2_xs_f16,
               mmq_wg_denoms,
               warptile_mmq,
               vk_mat_mat_push_constants,
               3)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[v_TYPE_IQ2_S],
               matmul_iq2_s_f16,
               mmq_wg_denoms,
               warptile_mmq,
               vk_mat_mat_push_constants,
               3)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[v_TYPE_IQ3_XXS],
               matmul_iq3_xxs_f16,
               mmq_wg_denoms,
               warptile_mmq,
               vk_mat_mat_push_constants,
               3)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[v_TYPE_IQ3_S],
               matmul_iq3_s_f16,
               mmq_wg_denoms,
               warptile_mmq,
               vk_mat_mat_push_constants,
               3)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[v_TYPE_IQ4_XS],
               matmul_iq4_xs_f16,
               mmq_wg_denoms,
               warptile_mmq,
               vk_mat_mat_push_constants,
               3)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[v_TYPE_IQ4_NL],
               matmul_iq4_nl_f16,
               mmq_wg_denoms,
               warptile_mmq,
               vk_mat_mat_push_constants,
               3)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[v_TYPE_MXFP4],
               matmul_mxfp4_f16,
               mmq_wg_denoms,
               warptile_mmq,
               vk_mat_mat_push_constants,
               3)

    V_ASSERT(device->subgroup_ballot);

    CREATE_MM2(pipeline_matmul_id_f16, matmul_id_subgroup_f16, wg_denoms, warptile, vk_mat_mat_id_push_constants, 4)
    #if defined(V_VULKAN_BFLOAT16_GLSLC_SUPPORT)
    if (device->coopmat_bf16_support) {
      CREATE_MM(pipeline_matmul_id_bf16,
                matmul_id_subgroup_bf16, ,
                wg_denoms,
                warptile,
                vk_mat_mat_id_push_constants,
                4)
    }
    #endif
    CREATE_MM2(pipeline_dequant_mul_mat_mat_id[v_TYPE_Q4_0],
               matmul_id_subgroup_q4_0_f16,
               mmqid_wg_denoms,
               warptile_mmqid,
               vk_mat_mat_id_push_constants,
               4)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_id[v_TYPE_Q4_1],
               matmul_id_subgroup_q4_1_f16,
               mmqid_wg_denoms,
               warptile_mmqid,
               vk_mat_mat_id_push_constants,
               4)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_id[v_TYPE_Q5_0],
               matmul_id_subgroup_q5_0_f16,
               mmqid_wg_denoms,
               warptile_mmqid,
               vk_mat_mat_id_push_constants,
               4)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_id[v_TYPE_Q5_1],
               matmul_id_subgroup_q5_1_f16,
               mmqid_wg_denoms,
               warptile_mmqid,
               vk_mat_mat_id_push_constants,
               4)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_id[v_TYPE_Q8_0],
               matmul_id_subgroup_q8_0_f16,
               mmqid_wg_denoms,
               warptile_mmqid,
               vk_mat_mat_id_push_constants,
               4)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_id[v_TYPE_Q2_K],
               matmul_id_subgroup_q2_k_f16,
               mmqid_wg_denoms,
               warptile_mmqid,
               vk_mat_mat_id_push_constants,
               4)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_id[v_TYPE_Q3_K],
               matmul_id_subgroup_q3_k_f16,
               mmqid_wg_denoms,
               warptile_mmqid,
               vk_mat_mat_id_push_constants,
               4)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_id[v_TYPE_Q4_K],
               matmul_id_subgroup_q4_k_f16,
               mmqid_wg_denoms,
               warptile_mmqid,
               vk_mat_mat_id_push_constants,
               4)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_id[v_TYPE_Q5_K],
               matmul_id_subgroup_q5_k_f16,
               mmqid_wg_denoms,
               warptile_mmqid,
               vk_mat_mat_id_push_constants,
               4)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_id[v_TYPE_Q6_K],
               matmul_id_subgroup_q6_k_f16,
               mmqid_wg_denoms,
               warptile_mmqid,
               vk_mat_mat_id_push_constants,
               4)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ1_S],
               matmul_id_subgroup_iq1_s_f16,
               mmqid_wg_denoms,
               warptile_mmqid,
               vk_mat_mat_id_push_constants,
               4)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ1_M],
               matmul_id_subgroup_iq1_m_f16,
               mmqid_wg_denoms,
               warptile_mmqid,
               vk_mat_mat_id_push_constants,
               4)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ2_XXS],
               matmul_id_subgroup_iq2_xxs_f16,
               mmqid_wg_denoms,
               warptile_mmqid,
               vk_mat_mat_id_push_constants,
               4)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ2_XS],
               matmul_id_subgroup_iq2_xs_f16,
               mmqid_wg_denoms,
               warptile_mmqid,
               vk_mat_mat_id_push_constants,
               4)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ2_S],
               matmul_id_subgroup_iq2_s_f16,
               mmqid_wg_denoms,
               warptile_mmqid,
               vk_mat_mat_id_push_constants,
               4)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ3_XXS],
               matmul_id_subgroup_iq3_xxs_f16,
               mmqid_wg_denoms,
               warptile_mmqid,
               vk_mat_mat_id_push_constants,
               4)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ3_S],
               matmul_id_subgroup_iq3_s_f16,
               mmqid_wg_denoms,
               warptile_mmqid,
               vk_mat_mat_id_push_constants,
               4)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ4_XS],
               matmul_id_subgroup_iq4_xs_f16,
               mmqid_wg_denoms,
               warptile_mmqid,
               vk_mat_mat_id_push_constants,
               4)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ4_NL],
               matmul_id_subgroup_iq4_nl_f16,
               mmqid_wg_denoms,
               warptile_mmqid,
               vk_mat_mat_id_push_constants,
               4)
    CREATE_MM2(pipeline_dequant_mul_mat_mat_id[v_TYPE_MXFP4],
               matmul_id_subgroup_mxfp4_f16,
               mmqid_wg_denoms,
               warptile_mmqid,
               vk_mat_mat_id_push_constants,
               4)
    #undef CREATE_MM
    #undef CREATE_MM2
  }
  else
  #endif  // defined(VK_NV_cooperative_matrix2) && defined(V_VULKAN_COOPMAT2_GLSLC_SUPPORT)
  #if defined(VK_KHR_cooperative_matrix) && defined(V_VULKAN_COOPMAT_GLSLC_SUPPORT)
    if (device->coopmat_support) {
      // Create 6 variants, {s,m,l}x{unaligned,aligned}
      #define CREATE_MM(TYPE, PIPELINE_NAME, NAMELC, F16ACC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME ->l, #NAMELC #F16ACC "_l", NAMELC ## F16ACC ## _cm1_len, NAMELC ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1, false, true);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME ->m, #NAMELC #F16ACC "_m", NAMELC ## F16ACC ## _cm1_len, NAMELC ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1, false, true);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME ->s, #NAMELC #F16ACC "_s", NAMELC ## F16ACC ## _cm1_len, NAMELC ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1, false, true);   \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_l, #NAMELC #F16ACC "_aligned_l", NAMELC ## _aligned ## F16ACC ## _cm1_len, NAMELC ## _aligned ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, l_align, false, true);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_m, #NAMELC #F16ACC "_aligned_m", NAMELC ## _aligned ## F16ACC ## _cm1_len, NAMELC ## _aligned ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, m_align, false, true);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_s, #NAMELC #F16ACC "_aligned_s", NAMELC ## _aligned ## F16ACC ## _cm1_len, NAMELC ## _aligned ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, s_align, false, true);
      // Create 2 variants, {f16,f32} accumulator
      #define CREATE_MM2(TYPE, PIPELINE_NAME, NAMELC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        if (device->coopmat_acc_f16_support) { \
            CREATE_MM(TYPE, PIPELINE_NAME . f16acc, NAMELC, _f16acc, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        } \
        if (device->coopmat_acc_f32_support) { \
            CREATE_MM(TYPE, PIPELINE_NAME . f32acc, NAMELC, , WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        }
      CREATE_MM(v_TYPE_F32,
                pipeline_matmul_f32,
                matmul_f32_f32, ,
                wg_denoms,
                warptile,
                vk_mat_mat_push_constants,
                3,);
      CREATE_MM(v_TYPE_F32,
                pipeline_matmul_f32_f16,
                matmul_f32_f16, ,
                wg_denoms,
                warptile,
                vk_mat_mat_push_constants,
                3,);
      CREATE_MM2(v_TYPE_F16, pipeline_matmul_f16, matmul_f16, wg_denoms, warptile, vk_mat_mat_push_constants, 3,);
      CREATE_MM2(v_TYPE_F16,
                 pipeline_matmul_f16_f32,
                 matmul_f16_f32,
                 wg_denoms,
                 warptile,
                 vk_mat_mat_push_constants,
                 3,);
      #if defined(V_VULKAN_BFLOAT16_GLSLC_SUPPORT)
      if (device->coopmat_bf16_support) {
        CREATE_MM(v_TYPE_BF16,
                  pipeline_matmul_bf16,
                  matmul_bf16, ,
                  wg_denoms,
                  warptile,
                  vk_mat_mat_push_constants,
                  3,)
      }
      #endif

      if (device->coopmat_acc_f16_support) {
        CREATE_MM2(v_TYPE_Q4_0,
                   pipeline_dequant_mul_mat_mat[v_TYPE_Q4_0],
                   matmul_q4_0_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3,);
        CREATE_MM2(v_TYPE_Q4_1,
                   pipeline_dequant_mul_mat_mat[v_TYPE_Q4_1],
                   matmul_q4_1_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3,);
        CREATE_MM2(v_TYPE_Q5_0,
                   pipeline_dequant_mul_mat_mat[v_TYPE_Q5_0],
                   matmul_q5_0_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3,);
        CREATE_MM2(v_TYPE_Q5_1,
                   pipeline_dequant_mul_mat_mat[v_TYPE_Q5_1],
                   matmul_q5_1_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3,);
        CREATE_MM2(v_TYPE_Q8_0,
                   pipeline_dequant_mul_mat_mat[v_TYPE_Q8_0],
                   matmul_q8_0_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3,);

        CREATE_MM2(v_TYPE_Q2_K,
                   pipeline_dequant_mul_mat_mat[v_TYPE_Q2_K],
                   matmul_q2_k_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3,);
        CREATE_MM2(v_TYPE_Q3_K,
                   pipeline_dequant_mul_mat_mat[v_TYPE_Q3_K],
                   matmul_q3_k_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3,);
        CREATE_MM2(v_TYPE_Q4_K,
                   pipeline_dequant_mul_mat_mat[v_TYPE_Q4_K],
                   matmul_q4_k_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3,);
        CREATE_MM2(v_TYPE_Q5_K,
                   pipeline_dequant_mul_mat_mat[v_TYPE_Q5_K],
                   matmul_q5_k_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3,);
        CREATE_MM2(v_TYPE_Q6_K,
                   pipeline_dequant_mul_mat_mat[v_TYPE_Q6_K],
                   matmul_q6_k_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3,);
        CREATE_MM2(v_TYPE_IQ1_S,
                   pipeline_dequant_mul_mat_mat[v_TYPE_IQ1_S],
                   matmul_iq1_s_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3,);
        CREATE_MM2(v_TYPE_IQ1_M,
                   pipeline_dequant_mul_mat_mat[v_TYPE_IQ1_M],
                   matmul_iq1_m_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3,);
        CREATE_MM2(v_TYPE_IQ2_XXS,
                   pipeline_dequant_mul_mat_mat[v_TYPE_IQ2_XXS],
                   matmul_iq2_xxs_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3,);
        CREATE_MM2(v_TYPE_IQ2_XS,
                   pipeline_dequant_mul_mat_mat[v_TYPE_IQ2_XS],
                   matmul_iq2_xs_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3,);
        CREATE_MM2(v_TYPE_IQ2_S,
                   pipeline_dequant_mul_mat_mat[v_TYPE_IQ2_S],
                   matmul_iq2_s_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3,);
        CREATE_MM2(v_TYPE_IQ3_XXS,
                   pipeline_dequant_mul_mat_mat[v_TYPE_IQ3_XXS],
                   matmul_iq3_xxs_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3,);
        CREATE_MM2(v_TYPE_IQ3_S,
                   pipeline_dequant_mul_mat_mat[v_TYPE_IQ3_S],
                   matmul_iq3_s_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3,);
        CREATE_MM2(v_TYPE_IQ4_XS,
                   pipeline_dequant_mul_mat_mat[v_TYPE_IQ4_XS],
                   matmul_iq4_xs_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3,);
        CREATE_MM2(v_TYPE_IQ4_NL,
                   pipeline_dequant_mul_mat_mat[v_TYPE_IQ4_NL],
                   matmul_iq4_nl_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3,);
        CREATE_MM2(v_TYPE_MXFP4,
                   pipeline_dequant_mul_mat_mat[v_TYPE_MXFP4],
                   matmul_mxfp4_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3,);
      }
      else {
        CREATE_MM(v_TYPE_Q4_0,
                  pipeline_dequant_mul_mat_mat[v_TYPE_Q4_0].f32acc,
                  matmul_q4_0_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3,);
        CREATE_MM(v_TYPE_Q4_1,
                  pipeline_dequant_mul_mat_mat[v_TYPE_Q4_1].f32acc,
                  matmul_q4_1_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3,);
        CREATE_MM(v_TYPE_Q5_0,
                  pipeline_dequant_mul_mat_mat[v_TYPE_Q5_0].f32acc,
                  matmul_q5_0_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3,);
        CREATE_MM(v_TYPE_Q5_1,
                  pipeline_dequant_mul_mat_mat[v_TYPE_Q5_1].f32acc,
                  matmul_q5_1_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3,);
        CREATE_MM(v_TYPE_Q8_0,
                  pipeline_dequant_mul_mat_mat[v_TYPE_Q8_0].f32acc,
                  matmul_q8_0_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3,);

        CREATE_MM(v_TYPE_Q2_K,
                  pipeline_dequant_mul_mat_mat[v_TYPE_Q2_K].f32acc,
                  matmul_q2_k_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3,);
        CREATE_MM(v_TYPE_Q3_K,
                  pipeline_dequant_mul_mat_mat[v_TYPE_Q3_K].f32acc,
                  matmul_q3_k_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3,);
        CREATE_MM(v_TYPE_Q4_K,
                  pipeline_dequant_mul_mat_mat[v_TYPE_Q4_K].f32acc,
                  matmul_q4_k_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3,);
        CREATE_MM(v_TYPE_Q5_K,
                  pipeline_dequant_mul_mat_mat[v_TYPE_Q5_K].f32acc,
                  matmul_q5_k_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3,);
        CREATE_MM(v_TYPE_Q6_K,
                  pipeline_dequant_mul_mat_mat[v_TYPE_Q6_K].f32acc,
                  matmul_q6_k_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3,);
        CREATE_MM(v_TYPE_IQ1_S,
                  pipeline_dequant_mul_mat_mat[v_TYPE_IQ1_S].f32acc,
                  matmul_iq1_s_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3,);
        CREATE_MM(v_TYPE_IQ1_M,
                  pipeline_dequant_mul_mat_mat[v_TYPE_IQ1_M].f32acc,
                  matmul_iq1_m_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3,);
        CREATE_MM(v_TYPE_IQ2_XXS,
                  pipeline_dequant_mul_mat_mat[v_TYPE_IQ2_XXS].f32acc,
                  matmul_iq2_xxs_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3,);
        CREATE_MM(v_TYPE_IQ2_XS,
                  pipeline_dequant_mul_mat_mat[v_TYPE_IQ2_XS].f32acc,
                  matmul_iq2_xs_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3,);
        CREATE_MM(v_TYPE_IQ2_S,
                  pipeline_dequant_mul_mat_mat[v_TYPE_IQ2_S].f32acc,
                  matmul_iq2_s_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3,);
        CREATE_MM(v_TYPE_IQ3_XXS,
                  pipeline_dequant_mul_mat_mat[v_TYPE_IQ3_XXS].f32acc,
                  matmul_iq3_xxs_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3,);
        CREATE_MM(v_TYPE_IQ3_S,
                  pipeline_dequant_mul_mat_mat[v_TYPE_IQ3_S].f32acc,
                  matmul_iq3_s_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3,);
        CREATE_MM(v_TYPE_IQ4_XS,
                  pipeline_dequant_mul_mat_mat[v_TYPE_IQ4_XS].f32acc,
                  matmul_iq4_xs_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3,);
        CREATE_MM(v_TYPE_IQ4_NL,
                  pipeline_dequant_mul_mat_mat[v_TYPE_IQ4_NL].f32acc,
                  matmul_iq4_nl_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3,);
        CREATE_MM(v_TYPE_MXFP4,
                  pipeline_dequant_mul_mat_mat[v_TYPE_MXFP4].f32acc,
                  matmul_mxfp4_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3,);
      }

      V_ASSERT(device->subgroup_ballot);

      CREATE_MM(v_TYPE_F32,
                pipeline_matmul_id_f32,
                matmul_id_subgroup_f32_f32, ,
                wg_denoms,
                warptile,
                vk_mat_mat_push_constants,
                4,
                _id);
      CREATE_MM2(v_TYPE_F16,
                 pipeline_matmul_id_f16,
                 matmul_id_subgroup_f16,
                 wg_denoms,
                 warptile,
                 vk_mat_mat_push_constants,
                 4,
                 _id);
      CREATE_MM2(v_TYPE_F16,
                 pipeline_matmul_id_f16_f32,
                 matmul_id_subgroup_f16_f32,
                 wg_denoms,
                 warptile,
                 vk_mat_mat_push_constants,
                 4,
                 _id);
      #if defined(V_VULKAN_BFLOAT16_GLSLC_SUPPORT)
      if (device->coopmat_bf16_support) {
        CREATE_MM(v_TYPE_BF16,
                  pipeline_matmul_id_bf16,
                  matmul_id_subgroup_bf16, ,
                  wg_denoms,
                  warptile,
                  vk_mat_mat_push_constants,
                  4,
                  _id);
      }
      #endif

      CREATE_MM2(v_TYPE_Q4_0,
                 pipeline_dequant_mul_mat_mat_id[v_TYPE_Q4_0],
                 matmul_id_subgroup_q4_0_f32,
                 mmq_wg_denoms,
                 warptile_mmq,
                 vk_mat_mat_id_push_constants,
                 4,
                 _id);
      CREATE_MM2(v_TYPE_Q4_1,
                 pipeline_dequant_mul_mat_mat_id[v_TYPE_Q4_1],
                 matmul_id_subgroup_q4_1_f32,
                 mmq_wg_denoms,
                 warptile_mmq,
                 vk_mat_mat_id_push_constants,
                 4,
                 _id);
      CREATE_MM2(v_TYPE_Q5_0,
                 pipeline_dequant_mul_mat_mat_id[v_TYPE_Q5_0],
                 matmul_id_subgroup_q5_0_f32,
                 mmq_wg_denoms,
                 warptile_mmq,
                 vk_mat_mat_id_push_constants,
                 4,
                 _id);
      CREATE_MM2(v_TYPE_Q5_1,
                 pipeline_dequant_mul_mat_mat_id[v_TYPE_Q5_1],
                 matmul_id_subgroup_q5_1_f32,
                 mmq_wg_denoms,
                 warptile_mmq,
                 vk_mat_mat_id_push_constants,
                 4,
                 _id);
      CREATE_MM2(v_TYPE_Q8_0,
                 pipeline_dequant_mul_mat_mat_id[v_TYPE_Q8_0],
                 matmul_id_subgroup_q8_0_f32,
                 mmq_wg_denoms,
                 warptile_mmq,
                 vk_mat_mat_id_push_constants,
                 4,
                 _id);
      CREATE_MM2(v_TYPE_Q2_K,
                 pipeline_dequant_mul_mat_mat_id[v_TYPE_Q2_K],
                 matmul_id_subgroup_q2_k_f32,
                 mmq_wg_denoms,
                 warptile_mmq,
                 vk_mat_mat_id_push_constants,
                 4,
                 _id);
      CREATE_MM2(v_TYPE_Q3_K,
                 pipeline_dequant_mul_mat_mat_id[v_TYPE_Q3_K],
                 matmul_id_subgroup_q3_k_f32,
                 mmq_wg_denoms,
                 warptile_mmq,
                 vk_mat_mat_id_push_constants,
                 4,
                 _id);
      CREATE_MM2(v_TYPE_Q4_K,
                 pipeline_dequant_mul_mat_mat_id[v_TYPE_Q4_K],
                 matmul_id_subgroup_q4_k_f32,
                 mmq_wg_denoms,
                 warptile_mmq,
                 vk_mat_mat_id_push_constants,
                 4,
                 _id);
      CREATE_MM2(v_TYPE_Q5_K,
                 pipeline_dequant_mul_mat_mat_id[v_TYPE_Q5_K],
                 matmul_id_subgroup_q5_k_f32,
                 mmq_wg_denoms,
                 warptile_mmq,
                 vk_mat_mat_id_push_constants,
                 4,
                 _id);
      CREATE_MM2(v_TYPE_Q6_K,
                 pipeline_dequant_mul_mat_mat_id[v_TYPE_Q6_K],
                 matmul_id_subgroup_q6_k_f32,
                 mmq_wg_denoms,
                 warptile_mmq,
                 vk_mat_mat_id_push_constants,
                 4,
                 _id);
      CREATE_MM2(v_TYPE_IQ1_S,
                 pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ1_S],
                 matmul_id_subgroup_iq1_s_f32,
                 mmq_wg_denoms,
                 warptile_mmq,
                 vk_mat_mat_id_push_constants,
                 4,
                 _id);
      CREATE_MM2(v_TYPE_IQ1_M,
                 pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ1_M],
                 matmul_id_subgroup_iq1_m_f32,
                 mmq_wg_denoms,
                 warptile_mmq,
                 vk_mat_mat_id_push_constants,
                 4,
                 _id);
      CREATE_MM2(v_TYPE_IQ2_XXS,
                 pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ2_XXS],
                 matmul_id_subgroup_iq2_xxs_f32,
                 mmq_wg_denoms,
                 warptile_mmq,
                 vk_mat_mat_id_push_constants,
                 4,
                 _id);
      CREATE_MM2(v_TYPE_IQ2_XS,
                 pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ2_XS],
                 matmul_id_subgroup_iq2_xs_f32,
                 mmq_wg_denoms,
                 warptile_mmq,
                 vk_mat_mat_id_push_constants,
                 4,
                 _id);
      CREATE_MM2(v_TYPE_IQ2_S,
                 pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ2_S],
                 matmul_id_subgroup_iq2_s_f32,
                 mmq_wg_denoms,
                 warptile_mmq,
                 vk_mat_mat_id_push_constants,
                 4,
                 _id);
      CREATE_MM2(v_TYPE_IQ3_XXS,
                 pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ3_XXS],
                 matmul_id_subgroup_iq3_xxs_f32,
                 mmq_wg_denoms,
                 warptile_mmq,
                 vk_mat_mat_id_push_constants,
                 4,
                 _id);
      CREATE_MM2(v_TYPE_IQ3_S,
                 pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ3_S],
                 matmul_id_subgroup_iq3_s_f32,
                 mmq_wg_denoms,
                 warptile_mmq,
                 vk_mat_mat_id_push_constants,
                 4,
                 _id);
      CREATE_MM2(v_TYPE_IQ4_XS,
                 pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ4_XS],
                 matmul_id_subgroup_iq4_xs_f32,
                 mmq_wg_denoms,
                 warptile_mmq,
                 vk_mat_mat_id_push_constants,
                 4,
                 _id);
      CREATE_MM2(v_TYPE_IQ4_NL,
                 pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ4_NL],
                 matmul_id_subgroup_iq4_nl_f32,
                 mmq_wg_denoms,
                 warptile_mmq,
                 vk_mat_mat_id_push_constants,
                 4,
                 _id);
      CREATE_MM2(v_TYPE_MXFP4,
                 pipeline_dequant_mul_mat_mat_id[v_TYPE_MXFP4],
                 matmul_id_subgroup_mxfp4_f32,
                 mmq_wg_denoms,
                 warptile_mmq,
                 vk_mat_mat_id_push_constants,
                 4,
                 _id);
      #undef CREATE_MM2
      #undef CREATE_MM
    }
    else
    #endif  // defined(VK_KHR_cooperative_matrix) && defined(V_VULKAN_COOPMAT_GLSLC_SUPPORT)
      if (device->fp16) {
        // Create 6 variants, {s,m,l}x{unaligned,aligned}
        #define CREATE_MM(TYPE, PIPELINE_NAME, NAMELC, F16ACC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID, REQSUBGROUPSIZE) \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME ->l, #NAMELC #F16ACC "_l", NAMELC ## F16ACC ## _len, NAMELC ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME ->m, #NAMELC #F16ACC "_m", NAMELC ## F16ACC ## _len, NAMELC ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME ->s, #NAMELC #F16ACC "_s", NAMELC ## F16ACC ## _len, NAMELC ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_l, #NAMELC #F16ACC "_aligned_l", NAMELC ## _aligned ## F16ACC ## _len, NAMELC ## _aligned ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, l_align, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_m, #NAMELC #F16ACC "_aligned_m", NAMELC ## _aligned ## F16ACC ## _len, NAMELC ## _aligned ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, m_align, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_s, #NAMELC #F16ACC "_aligned_s", NAMELC ## _aligned ## F16ACC ## _len, NAMELC ## _aligned ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, s_align, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);
        #define CREATE_MMQ(TYPE, PIPELINE_NAME, NAMELC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        if (device->mul_mat ## ID ## _l[TYPE]) { \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME .f16acc->l, #NAMELC "_f16acc_l", NAMELC ## _f16acc_len, NAMELC ##  _f16acc_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1);   \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME .f32acc->l, #NAMELC        "_l", NAMELC ## _len,        NAMELC ##  _data,        "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1);   \
        } \
        if (device->mul_mat ## ID ## _m[TYPE]) { \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME .f16acc->m, #NAMELC "_f16acc_m", NAMELC ## _f16acc_len, NAMELC ##  _f16acc_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1);   \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME .f32acc->m, #NAMELC        "_m", NAMELC ## _len,        NAMELC ##  _data,        "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1);   \
        } \
        if (device->mul_mat ## ID ## _s[TYPE]) { \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME .f16acc->s, #NAMELC "_f16acc_s", NAMELC ## _f16acc_len, NAMELC ##  _f16acc_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1);   \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME .f32acc->s, #NAMELC        "_s", NAMELC ## _len,        NAMELC ##  _data,        "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1);   \
        }
        // Create 2 variants, {f16,f32} accumulator
        #define CREATE_MM2(TYPE, PIPELINE_NAME, NAMELC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID, REQSUBGROUPSIZE) \
        CREATE_MM(TYPE, PIPELINE_NAME . f16acc, NAMELC, _f16acc, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID, REQSUBGROUPSIZE) \
        CREATE_MM(TYPE, PIPELINE_NAME . f32acc, NAMELC, , WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID, REQSUBGROUPSIZE)
        CREATE_MM(v_TYPE_F32,
                  pipeline_matmul_f32,
                  matmul_f32_f32, ,
                  wg_denoms,
                  warptile,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);
        CREATE_MM(v_TYPE_F32,
                  pipeline_matmul_f32_f16,
                  matmul_f32_f16, ,
                  wg_denoms,
                  warptile,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);
        CREATE_MM2(v_TYPE_F16,
                   pipeline_matmul_f16,
                   matmul_f16,
                   wg_denoms,
                   warptile,
                   vk_mat_mat_push_constants,
                   3, ,
                   0);
        CREATE_MM2(v_TYPE_F16,
                   pipeline_matmul_f16_f32,
                   matmul_f16_f32,
                   wg_denoms,
                   warptile,
                   vk_mat_mat_push_constants,
                   3, ,
                   0);

        CREATE_MM(v_TYPE_BF16,
                  pipeline_matmul_bf16,
                  matmul_bf16, ,
                  wg_denoms,
                  warptile,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);

        CREATE_MM2(v_TYPE_Q4_0,
                   pipeline_dequant_mul_mat_mat[v_TYPE_Q4_0],
                   matmul_q4_0_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3, ,
                   0);
        CREATE_MM2(v_TYPE_Q4_1,
                   pipeline_dequant_mul_mat_mat[v_TYPE_Q4_1],
                   matmul_q4_1_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3, ,
                   0);
        CREATE_MM2(v_TYPE_Q5_0,
                   pipeline_dequant_mul_mat_mat[v_TYPE_Q5_0],
                   matmul_q5_0_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3, ,
                   0);
        CREATE_MM2(v_TYPE_Q5_1,
                   pipeline_dequant_mul_mat_mat[v_TYPE_Q5_1],
                   matmul_q5_1_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3, ,
                   0);
        CREATE_MM2(v_TYPE_Q8_0,
                   pipeline_dequant_mul_mat_mat[v_TYPE_Q8_0],
                   matmul_q8_0_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3, ,
                   0);

        CREATE_MM2(v_TYPE_Q2_K,
                   pipeline_dequant_mul_mat_mat[v_TYPE_Q2_K],
                   matmul_q2_k_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3, ,
                   0);
        CREATE_MM2(v_TYPE_Q3_K,
                   pipeline_dequant_mul_mat_mat[v_TYPE_Q3_K],
                   matmul_q3_k_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3, ,
                   0);
        CREATE_MM2(v_TYPE_Q4_K,
                   pipeline_dequant_mul_mat_mat[v_TYPE_Q4_K],
                   matmul_q4_k_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3, ,
                   0);
        CREATE_MM2(v_TYPE_Q5_K,
                   pipeline_dequant_mul_mat_mat[v_TYPE_Q5_K],
                   matmul_q5_k_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3, ,
                   0);
        CREATE_MM2(v_TYPE_Q6_K,
                   pipeline_dequant_mul_mat_mat[v_TYPE_Q6_K],
                   matmul_q6_k_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3, ,
                   0);
        CREATE_MM2(v_TYPE_IQ1_S,
                   pipeline_dequant_mul_mat_mat[v_TYPE_IQ1_S],
                   matmul_iq1_s_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3, ,
                   0);
        CREATE_MM2(v_TYPE_IQ1_M,
                   pipeline_dequant_mul_mat_mat[v_TYPE_IQ1_M],
                   matmul_iq1_m_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3, ,
                   0);
        CREATE_MM2(v_TYPE_IQ2_XXS,
                   pipeline_dequant_mul_mat_mat[v_TYPE_IQ2_XXS],
                   matmul_iq2_xxs_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3, ,
                   0);
        CREATE_MM2(v_TYPE_IQ2_XS,
                   pipeline_dequant_mul_mat_mat[v_TYPE_IQ2_XS],
                   matmul_iq2_xs_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3, ,
                   0);
        CREATE_MM2(v_TYPE_IQ2_S,
                   pipeline_dequant_mul_mat_mat[v_TYPE_IQ2_S],
                   matmul_iq2_s_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3, ,
                   0);
        CREATE_MM2(v_TYPE_IQ3_XXS,
                   pipeline_dequant_mul_mat_mat[v_TYPE_IQ3_XXS],
                   matmul_iq3_xxs_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3, ,
                   0);
        CREATE_MM2(v_TYPE_IQ3_S,
                   pipeline_dequant_mul_mat_mat[v_TYPE_IQ3_S],
                   matmul_iq3_s_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3, ,
                   0);
        CREATE_MM2(v_TYPE_IQ4_XS,
                   pipeline_dequant_mul_mat_mat[v_TYPE_IQ4_XS],
                   matmul_iq4_xs_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3, ,
                   0);
        CREATE_MM2(v_TYPE_IQ4_NL,
                   pipeline_dequant_mul_mat_mat[v_TYPE_IQ4_NL],
                   matmul_iq4_nl_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3, ,
                   0);
        CREATE_MM2(v_TYPE_MXFP4,
                   pipeline_dequant_mul_mat_mat[v_TYPE_MXFP4],
                   matmul_mxfp4_f32,
                   mmq_wg_denoms,
                   warptile_mmq,
                   vk_mat_mat_push_constants,
                   3, ,
                   0);

        #if defined(V_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        if (device->integer_dot_product) {
          CREATE_MMQ(v_TYPE_Q4_0,
                     pipeline_dequant_mul_mat_mat_q8_1[v_TYPE_Q4_0],
                     matmul_q4_0_q8_1,
                     mmq_wg_denoms,
                     warptile_mmq_int,
                     vk_mat_mat_push_constants,
                     3,);
          CREATE_MMQ(v_TYPE_Q4_1,
                     pipeline_dequant_mul_mat_mat_q8_1[v_TYPE_Q4_1],
                     matmul_q4_1_q8_1,
                     mmq_wg_denoms,
                     warptile_mmq_int,
                     vk_mat_mat_push_constants,
                     3,);
          CREATE_MMQ(v_TYPE_Q5_0,
                     pipeline_dequant_mul_mat_mat_q8_1[v_TYPE_Q5_0],
                     matmul_q5_0_q8_1,
                     mmq_wg_denoms,
                     warptile_mmq_int,
                     vk_mat_mat_push_constants,
                     3,);
          CREATE_MMQ(v_TYPE_Q5_1,
                     pipeline_dequant_mul_mat_mat_q8_1[v_TYPE_Q5_1],
                     matmul_q5_1_q8_1,
                     mmq_wg_denoms,
                     warptile_mmq_int,
                     vk_mat_mat_push_constants,
                     3,);
          CREATE_MMQ(v_TYPE_Q8_0,
                     pipeline_dequant_mul_mat_mat_q8_1[v_TYPE_Q8_0],
                     matmul_q8_0_q8_1,
                     mmq_wg_denoms,
                     warptile_mmq_int,
                     vk_mat_mat_push_constants,
                     3,);
        }
        #endif

        if (device->subgroup_ballot && device->subgroup_require_full_support && subgroup_min_size_16) {
          CREATE_MM(v_TYPE_F32,
                    pipeline_matmul_id_f32,
                    matmul_id_subgroup_f32_f32, ,
                    wg_denoms,
                    warptile_id,
                    vk_mat_mat_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size_16);
          CREATE_MM2(v_TYPE_F16,
                     pipeline_matmul_id_f16,
                     matmul_id_subgroup_f16,
                     wg_denoms,
                     warptile_id,
                     vk_mat_mat_push_constants,
                     4,
                     _id,
                     mul_mat_subgroup_size_16);
          CREATE_MM2(v_TYPE_F16,
                     pipeline_matmul_id_f16_f32,
                     matmul_id_subgroup_f16_f32,
                     wg_denoms,
                     warptile_id,
                     vk_mat_mat_push_constants,
                     4,
                     _id,
                     mul_mat_subgroup_size_16);
          CREATE_MM(v_TYPE_BF16,
                    pipeline_matmul_id_bf16,
                    matmul_id_subgroup_bf16, ,
                    wg_denoms,
                    warptile_id,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size_16);

          CREATE_MM2(v_TYPE_Q4_0,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_Q4_0],
                     matmul_id_subgroup_q4_0_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     mul_mat_subgroup_size);
          CREATE_MM2(v_TYPE_Q4_1,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_Q4_1],
                     matmul_id_subgroup_q4_1_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     mul_mat_subgroup_size);
          CREATE_MM2(v_TYPE_Q5_0,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_Q5_0],
                     matmul_id_subgroup_q5_0_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     mul_mat_subgroup_size);
          CREATE_MM2(v_TYPE_Q5_1,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_Q5_1],
                     matmul_id_subgroup_q5_1_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     mul_mat_subgroup_size);
          CREATE_MM2(v_TYPE_Q8_0,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_Q8_0],
                     matmul_id_subgroup_q8_0_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     mul_mat_subgroup_size);
          CREATE_MM2(v_TYPE_Q2_K,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_Q2_K],
                     matmul_id_subgroup_q2_k_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     mul_mat_subgroup_size);
          CREATE_MM2(v_TYPE_Q3_K,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_Q3_K],
                     matmul_id_subgroup_q3_k_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     mul_mat_subgroup_size);
          CREATE_MM2(v_TYPE_Q4_K,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_Q4_K],
                     matmul_id_subgroup_q4_k_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     mul_mat_subgroup_size);
          CREATE_MM2(v_TYPE_Q5_K,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_Q5_K],
                     matmul_id_subgroup_q5_k_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     mul_mat_subgroup_size);
          CREATE_MM2(v_TYPE_Q6_K,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_Q6_K],
                     matmul_id_subgroup_q6_k_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     mul_mat_subgroup_size);
          CREATE_MM2(v_TYPE_IQ1_S,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ1_S],
                     matmul_id_subgroup_iq1_s_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     mul_mat_subgroup_size);
          CREATE_MM2(v_TYPE_IQ1_M,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ1_M],
                     matmul_id_subgroup_iq1_m_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     mul_mat_subgroup_size);
          CREATE_MM2(v_TYPE_IQ2_XXS,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ2_XXS],
                     matmul_id_subgroup_iq2_xxs_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     mul_mat_subgroup_size);
          CREATE_MM2(v_TYPE_IQ2_XS,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ2_XS],
                     matmul_id_subgroup_iq2_xs_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     mul_mat_subgroup_size);
          CREATE_MM2(v_TYPE_IQ2_S,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ2_S],
                     matmul_id_subgroup_iq2_s_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     mul_mat_subgroup_size);
          CREATE_MM2(v_TYPE_IQ3_XXS,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ3_XXS],
                     matmul_id_subgroup_iq3_xxs_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     mul_mat_subgroup_size);
          CREATE_MM2(v_TYPE_IQ3_S,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ3_S],
                     matmul_id_subgroup_iq3_s_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     mul_mat_subgroup_size);
          CREATE_MM2(v_TYPE_IQ4_XS,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ4_XS],
                     matmul_id_subgroup_iq4_xs_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     mul_mat_subgroup_size);
          CREATE_MM2(v_TYPE_IQ4_NL,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ4_NL],
                     matmul_id_subgroup_iq4_nl_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     mul_mat_subgroup_size);
          CREATE_MM2(v_TYPE_MXFP4,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_MXFP4],
                     matmul_id_subgroup_mxfp4_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     mul_mat_subgroup_size);
        }
        else {
          CREATE_MM(v_TYPE_F32,
                    pipeline_matmul_id_f32,
                    matmul_id_f32_f32, ,
                    wg_denoms,
                    warptile,
                    vk_mat_mat_push_constants,
                    4,
                    _id,
                    0);
          CREATE_MM2(v_TYPE_F16,
                     pipeline_matmul_id_f16,
                     matmul_id_f16,
                     wg_denoms,
                     warptile,
                     vk_mat_mat_push_constants,
                     4,
                     _id,
                     0);
          CREATE_MM2(v_TYPE_F16,
                     pipeline_matmul_id_f16_f32,
                     matmul_id_f16_f32,
                     wg_denoms,
                     warptile,
                     vk_mat_mat_push_constants,
                     4,
                     _id,
                     0);
          CREATE_MM(v_TYPE_BF16,
                    pipeline_matmul_id_bf16,
                    matmul_id_bf16, ,
                    wg_denoms,
                    warptile,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    0);

          CREATE_MM2(v_TYPE_Q4_0,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_Q4_0],
                     matmul_id_q4_0_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     0);
          CREATE_MM2(v_TYPE_Q4_1,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_Q4_1],
                     matmul_id_q4_1_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     0);
          CREATE_MM2(v_TYPE_Q5_0,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_Q5_0],
                     matmul_id_q5_0_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     0);
          CREATE_MM2(v_TYPE_Q5_1,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_Q5_1],
                     matmul_id_q5_1_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     0);
          CREATE_MM2(v_TYPE_Q8_0,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_Q8_0],
                     matmul_id_q8_0_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     0);
          CREATE_MM2(v_TYPE_Q2_K,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_Q2_K],
                     matmul_id_q2_k_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     0);
          CREATE_MM2(v_TYPE_Q3_K,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_Q3_K],
                     matmul_id_q3_k_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     0);
          CREATE_MM2(v_TYPE_Q4_K,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_Q4_K],
                     matmul_id_q4_k_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     0);
          CREATE_MM2(v_TYPE_Q5_K,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_Q5_K],
                     matmul_id_q5_k_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     0);
          CREATE_MM2(v_TYPE_Q6_K,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_Q6_K],
                     matmul_id_q6_k_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     0);
          CREATE_MM2(v_TYPE_IQ1_S,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ1_S],
                     matmul_id_iq1_s_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     0);
          CREATE_MM2(v_TYPE_IQ1_M,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ1_M],
                     matmul_id_iq1_m_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     0);
          CREATE_MM2(v_TYPE_IQ2_XXS,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ2_XXS],
                     matmul_id_iq2_xxs_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     0);
          CREATE_MM2(v_TYPE_IQ2_XS,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ2_XS],
                     matmul_id_iq2_xs_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     0);
          CREATE_MM2(v_TYPE_IQ2_S,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ2_S],
                     matmul_id_iq2_s_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     0);
          CREATE_MM2(v_TYPE_IQ3_XXS,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ3_XXS],
                     matmul_id_iq3_xxs_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     0);
          CREATE_MM2(v_TYPE_IQ3_S,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ3_S],
                     matmul_id_iq3_s_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     0);
          CREATE_MM2(v_TYPE_IQ4_XS,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ4_XS],
                     matmul_id_iq4_xs_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     0);
          CREATE_MM2(v_TYPE_IQ4_NL,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ4_NL],
                     matmul_id_iq4_nl_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     0);
          CREATE_MM2(v_TYPE_MXFP4,
                     pipeline_dequant_mul_mat_mat_id[v_TYPE_MXFP4],
                     matmul_id_mxfp4_f32,
                     mmq_wg_denoms,
                     warptile_mmqid,
                     vk_mat_mat_id_push_constants,
                     4,
                     _id,
                     0);
        }
        #undef CREATE_MM2
        #undef CREATE_MMQ
        #undef CREATE_MM
      }
      else {
        // Create 6 variants, {s,m,l}x{unaligned,aligned}
        #define CREATE_MM(TYPE, PIPELINE_NAME, NAMELC, F16ACC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID, REQSUBGROUPSIZE) \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME ->l, #NAMELC #F16ACC "_l", NAMELC ## F16ACC ## _fp32_len, NAMELC ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME ->m, #NAMELC #F16ACC "_m", NAMELC ## F16ACC ## _fp32_len, NAMELC ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME ->s, #NAMELC #F16ACC "_s", NAMELC ## F16ACC ## _fp32_len, NAMELC ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_l, #NAMELC #F16ACC "_aligned_l", NAMELC ## _aligned ## F16ACC ## _fp32_len, NAMELC ## _aligned ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, l_align, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_m, #NAMELC #F16ACC "_aligned_m", NAMELC ## _aligned ## F16ACC ## _fp32_len, NAMELC ## _aligned ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, m_align, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_s, #NAMELC #F16ACC "_aligned_s", NAMELC ## _aligned ## F16ACC ## _fp32_len, NAMELC ## _aligned ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, s_align, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);
        #define CREATE_MMQ(TYPE, PIPELINE_NAME, NAMELC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME ->l, #NAMELC "_l", NAMELC ## _fp32_len, NAMELC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME ->m, #NAMELC "_m", NAMELC ## _fp32_len, NAMELC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            v_vk_create_pipeline(device, device-> PIPELINE_NAME ->s, #NAMELC "_s", NAMELC ## _fp32_len, NAMELC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1);
        CREATE_MM(v_TYPE_F32,
                  pipeline_matmul_f32,
                  matmul_f32_f32, ,
                  wg_denoms,
                  warptile,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);
        CREATE_MM(v_TYPE_F32,
                  pipeline_matmul_f32_f16,
                  matmul_f32_f16, ,
                  wg_denoms,
                  warptile,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);
        CREATE_MM(v_TYPE_F16,
                  pipeline_matmul_f16.f32acc,
                  matmul_f16, ,
                  wg_denoms,
                  warptile,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);
        CREATE_MM(v_TYPE_F16,
                  pipeline_matmul_f16_f32.f32acc,
                  matmul_f16_f32, ,
                  wg_denoms,
                  warptile,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);

        CREATE_MM(v_TYPE_BF16,
                  pipeline_matmul_bf16,
                  matmul_bf16, ,
                  wg_denoms,
                  warptile,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);

        CREATE_MM(v_TYPE_Q4_0,
                  pipeline_dequant_mul_mat_mat[v_TYPE_Q4_0].f32acc,
                  matmul_q4_0_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);
        CREATE_MM(v_TYPE_Q4_1,
                  pipeline_dequant_mul_mat_mat[v_TYPE_Q4_1].f32acc,
                  matmul_q4_1_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);
        CREATE_MM(v_TYPE_Q5_0,
                  pipeline_dequant_mul_mat_mat[v_TYPE_Q5_0].f32acc,
                  matmul_q5_0_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);
        CREATE_MM(v_TYPE_Q5_1,
                  pipeline_dequant_mul_mat_mat[v_TYPE_Q5_1].f32acc,
                  matmul_q5_1_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);
        CREATE_MM(v_TYPE_Q8_0,
                  pipeline_dequant_mul_mat_mat[v_TYPE_Q8_0].f32acc,
                  matmul_q8_0_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);

        CREATE_MM(v_TYPE_Q2_K,
                  pipeline_dequant_mul_mat_mat[v_TYPE_Q2_K].f32acc,
                  matmul_q2_k_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);
        CREATE_MM(v_TYPE_Q3_K,
                  pipeline_dequant_mul_mat_mat[v_TYPE_Q3_K].f32acc,
                  matmul_q3_k_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);
        CREATE_MM(v_TYPE_Q4_K,
                  pipeline_dequant_mul_mat_mat[v_TYPE_Q4_K].f32acc,
                  matmul_q4_k_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);
        CREATE_MM(v_TYPE_Q5_K,
                  pipeline_dequant_mul_mat_mat[v_TYPE_Q5_K].f32acc,
                  matmul_q5_k_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);
        CREATE_MM(v_TYPE_Q6_K,
                  pipeline_dequant_mul_mat_mat[v_TYPE_Q6_K].f32acc,
                  matmul_q6_k_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);
        CREATE_MM(v_TYPE_IQ1_S,
                  pipeline_dequant_mul_mat_mat[v_TYPE_IQ1_S].f32acc,
                  matmul_iq1_s_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);
        CREATE_MM(v_TYPE_IQ1_M,
                  pipeline_dequant_mul_mat_mat[v_TYPE_IQ1_M].f32acc,
                  matmul_iq1_m_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);
        CREATE_MM(v_TYPE_IQ2_XXS,
                  pipeline_dequant_mul_mat_mat[v_TYPE_IQ2_XXS].f32acc,
                  matmul_iq2_xxs_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);
        CREATE_MM(v_TYPE_IQ2_XS,
                  pipeline_dequant_mul_mat_mat[v_TYPE_IQ2_XS].f32acc,
                  matmul_iq2_xs_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);
        CREATE_MM(v_TYPE_IQ2_S,
                  pipeline_dequant_mul_mat_mat[v_TYPE_IQ2_S].f32acc,
                  matmul_iq2_s_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);
        CREATE_MM(v_TYPE_IQ3_XXS,
                  pipeline_dequant_mul_mat_mat[v_TYPE_IQ3_XXS].f32acc,
                  matmul_iq3_xxs_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);
        CREATE_MM(v_TYPE_IQ3_S,
                  pipeline_dequant_mul_mat_mat[v_TYPE_IQ3_S].f32acc,
                  matmul_iq3_s_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);
        CREATE_MM(v_TYPE_IQ4_XS,
                  pipeline_dequant_mul_mat_mat[v_TYPE_IQ4_XS].f32acc,
                  matmul_iq4_xs_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);
        CREATE_MM(v_TYPE_IQ4_NL,
                  pipeline_dequant_mul_mat_mat[v_TYPE_IQ4_NL].f32acc,
                  matmul_iq4_nl_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);
        CREATE_MM(v_TYPE_MXFP4,
                  pipeline_dequant_mul_mat_mat[v_TYPE_MXFP4].f32acc,
                  matmul_mxfp4_f32, ,
                  mmq_wg_denoms,
                  warptile_mmq,
                  vk_mat_mat_push_constants,
                  3, ,
                  0);

        #if defined(V_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        if (device->integer_dot_product) {
          CREATE_MMQ(v_TYPE_Q4_0,
                     pipeline_dequant_mul_mat_mat_q8_1[v_TYPE_Q4_0].f32acc,
                     matmul_q4_0_q8_1,
                     mmq_wg_denoms,
                     warptile_mmq_int,
                     vk_mat_mat_push_constants,
                     3,);
          CREATE_MMQ(v_TYPE_Q4_1,
                     pipeline_dequant_mul_mat_mat_q8_1[v_TYPE_Q4_1].f32acc,
                     matmul_q4_1_q8_1,
                     mmq_wg_denoms,
                     warptile_mmq_int,
                     vk_mat_mat_push_constants,
                     3,);
          CREATE_MMQ(v_TYPE_Q5_0,
                     pipeline_dequant_mul_mat_mat_q8_1[v_TYPE_Q5_0].f32acc,
                     matmul_q5_0_q8_1,
                     mmq_wg_denoms,
                     warptile_mmq_int,
                     vk_mat_mat_push_constants,
                     3,);
          CREATE_MMQ(v_TYPE_Q5_1,
                     pipeline_dequant_mul_mat_mat_q8_1[v_TYPE_Q5_1].f32acc,
                     matmul_q5_1_q8_1,
                     mmq_wg_denoms,
                     warptile_mmq_int,
                     vk_mat_mat_push_constants,
                     3,);
          CREATE_MMQ(v_TYPE_Q8_0,
                     pipeline_dequant_mul_mat_mat_q8_1[v_TYPE_Q8_0].f32acc,
                     matmul_q8_0_q8_1,
                     mmq_wg_denoms,
                     warptile_mmq_int,
                     vk_mat_mat_push_constants,
                     3,);
        }
        #endif

        if (device->subgroup_ballot && device->subgroup_require_full_support && subgroup_min_size_16) {
          CREATE_MM(v_TYPE_F32,
                    pipeline_matmul_id_f32,
                    matmul_id_subgroup_f32_f32, ,
                    wg_denoms,
                    warptile_id,
                    vk_mat_mat_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size_16);
          CREATE_MM(v_TYPE_F16,
                    pipeline_matmul_id_f16.f32acc,
                    matmul_id_subgroup_f16, ,
                    wg_denoms,
                    warptile_id,
                    vk_mat_mat_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size_16);
          CREATE_MM(v_TYPE_F16,
                    pipeline_matmul_id_f16_f32.f32acc,
                    matmul_id_subgroup_f16_f32, ,
                    wg_denoms,
                    warptile_id,
                    vk_mat_mat_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size_16);
          CREATE_MM(v_TYPE_BF16,
                    pipeline_matmul_id_bf16,
                    matmul_id_subgroup_bf16, ,
                    wg_denoms,
                    warptile_id,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size_16);

          CREATE_MM(v_TYPE_Q4_0,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_Q4_0].f32acc,
                    matmul_id_subgroup_q4_0_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size);
          CREATE_MM(v_TYPE_Q4_1,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_Q4_1].f32acc,
                    matmul_id_subgroup_q4_1_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size);
          CREATE_MM(v_TYPE_Q5_0,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_Q5_0].f32acc,
                    matmul_id_subgroup_q5_0_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size);
          CREATE_MM(v_TYPE_Q5_1,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_Q5_1].f32acc,
                    matmul_id_subgroup_q5_1_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size);
          CREATE_MM(v_TYPE_Q8_0,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_Q8_0].f32acc,
                    matmul_id_subgroup_q8_0_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size);
          CREATE_MM(v_TYPE_Q2_K,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_Q2_K].f32acc,
                    matmul_id_subgroup_q2_k_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size);
          CREATE_MM(v_TYPE_Q3_K,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_Q3_K].f32acc,
                    matmul_id_subgroup_q3_k_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size);
          CREATE_MM(v_TYPE_Q4_K,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_Q4_K].f32acc,
                    matmul_id_subgroup_q4_k_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size);
          CREATE_MM(v_TYPE_Q5_K,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_Q5_K].f32acc,
                    matmul_id_subgroup_q5_k_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size);
          CREATE_MM(v_TYPE_Q6_K,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_Q6_K].f32acc,
                    matmul_id_subgroup_q6_k_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size);
          CREATE_MM(v_TYPE_IQ1_S,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ1_S].f32acc,
                    matmul_id_subgroup_iq1_s_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size);
          CREATE_MM(v_TYPE_IQ1_M,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ1_M].f32acc,
                    matmul_id_subgroup_iq1_m_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size);
          CREATE_MM(v_TYPE_IQ2_XXS,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ2_XXS].f32acc,
                    matmul_id_subgroup_iq2_xxs_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size);
          CREATE_MM(v_TYPE_IQ2_XS,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ2_XS].f32acc,
                    matmul_id_subgroup_iq2_xs_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size);
          CREATE_MM(v_TYPE_IQ2_S,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ2_S].f32acc,
                    matmul_id_subgroup_iq2_s_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size);
          CREATE_MM(v_TYPE_IQ3_XXS,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ3_XXS].f32acc,
                    matmul_id_subgroup_iq3_xxs_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size);
          CREATE_MM(v_TYPE_IQ3_S,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ3_S].f32acc,
                    matmul_id_subgroup_iq3_s_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size);
          CREATE_MM(v_TYPE_IQ4_XS,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ4_XS].f32acc,
                    matmul_id_subgroup_iq4_xs_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size);
          CREATE_MM(v_TYPE_IQ4_NL,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ4_NL].f32acc,
                    matmul_id_subgroup_iq4_nl_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size);
          CREATE_MM(v_TYPE_MXFP4,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_MXFP4].f32acc,
                    matmul_id_subgroup_mxfp4_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    mul_mat_subgroup_size);
        }
        else {
          CREATE_MM(v_TYPE_F32,
                    pipeline_matmul_id_f32,
                    matmul_id_f32_f32, ,
                    wg_denoms,
                    warptile,
                    vk_mat_mat_push_constants,
                    4,
                    _id,
                    0);
          CREATE_MM(v_TYPE_F16,
                    pipeline_matmul_id_f16.f32acc,
                    matmul_id_f16, ,
                    wg_denoms,
                    warptile,
                    vk_mat_mat_push_constants,
                    4,
                    _id,
                    0);
          CREATE_MM(v_TYPE_F16,
                    pipeline_matmul_id_f16_f32.f32acc,
                    matmul_id_f16_f32, ,
                    wg_denoms,
                    warptile,
                    vk_mat_mat_push_constants,
                    4,
                    _id,
                    0);
          CREATE_MM(v_TYPE_BF16,
                    pipeline_matmul_id_bf16,
                    matmul_id_bf16, ,
                    wg_denoms,
                    warptile,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    0);

          CREATE_MM(v_TYPE_Q4_0,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_Q4_0].f32acc,
                    matmul_id_q4_0_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    0);
          CREATE_MM(v_TYPE_Q4_1,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_Q4_1].f32acc,
                    matmul_id_q4_1_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    0);
          CREATE_MM(v_TYPE_Q5_0,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_Q5_0].f32acc,
                    matmul_id_q5_0_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    0);
          CREATE_MM(v_TYPE_Q5_1,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_Q5_1].f32acc,
                    matmul_id_q5_1_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    0);
          CREATE_MM(v_TYPE_Q8_0,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_Q8_0].f32acc,
                    matmul_id_q8_0_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    0);
          CREATE_MM(v_TYPE_Q2_K,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_Q2_K].f32acc,
                    matmul_id_q2_k_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    0);
          CREATE_MM(v_TYPE_Q3_K,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_Q3_K].f32acc,
                    matmul_id_q3_k_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    0);
          CREATE_MM(v_TYPE_Q4_K,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_Q4_K].f32acc,
                    matmul_id_q4_k_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    0);
          CREATE_MM(v_TYPE_Q5_K,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_Q5_K].f32acc,
                    matmul_id_q5_k_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    0);
          CREATE_MM(v_TYPE_Q6_K,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_Q6_K].f32acc,
                    matmul_id_q6_k_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    0);
          CREATE_MM(v_TYPE_IQ1_S,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ1_S].f32acc,
                    matmul_id_iq1_s_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    0);
          CREATE_MM(v_TYPE_IQ1_M,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ1_M].f32acc,
                    matmul_id_iq1_m_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    0);
          CREATE_MM(v_TYPE_IQ2_XXS,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ2_XXS].f32acc,
                    matmul_id_iq2_xxs_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    0);
          CREATE_MM(v_TYPE_IQ2_XS,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ2_XS].f32acc,
                    matmul_id_iq2_xs_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    0);
          CREATE_MM(v_TYPE_IQ2_S,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ2_S].f32acc,
                    matmul_id_iq2_s_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    0);
          CREATE_MM(v_TYPE_IQ3_XXS,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ3_XXS].f32acc,
                    matmul_id_iq3_xxs_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    0);
          CREATE_MM(v_TYPE_IQ3_S,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ3_S].f32acc,
                    matmul_id_iq3_s_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    0);
          CREATE_MM(v_TYPE_IQ4_XS,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ4_XS].f32acc,
                    matmul_id_iq4_xs_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    0);
          CREATE_MM(v_TYPE_IQ4_NL,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_IQ4_NL].f32acc,
                    matmul_id_iq4_nl_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    0);
          CREATE_MM(v_TYPE_MXFP4,
                    pipeline_dequant_mul_mat_mat_id[v_TYPE_MXFP4].f32acc,
                    matmul_id_mxfp4_f32, ,
                    mmq_wg_denoms,
                    warptile_mmqid,
                    vk_mat_mat_id_push_constants,
                    4,
                    _id,
                    0);
        }
      }
  // reusing CREATE_MM from the fp32 path
  if ((device->coopmat2 || device->coopmat_support)
    #if defined(V_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    && !device->coopmat_bf16_support
    #endif
  ) {
    // use scalar tile sizes
    l_warptile = {128, 128, 128, 16, subgroup_size_8 * 2, 64, 2, 4, 4, 1, subgroup_size_8};
    m_warptile = {128, 64, 64, 16, subgroup_size_8, 32, 2, 4, 2, 1, subgroup_size_8};
    s_warptile = {subgroup_size_16, 32, 32, 16, 32, 32, 2, 2, 2, 1, subgroup_size_8};

    l_wg_denoms = {128, 128, 1};
    m_wg_denoms = {64, 64, 1};
    s_wg_denoms = {32, 32, 1};

    CREATE_MM(v_TYPE_BF16,
              pipeline_matmul_bf16,
              matmul_bf16, ,
              wg_denoms,
              warptile,
              vk_mat_mat_push_constants,
              3, ,
              0);
    CREATE_MM(v_TYPE_BF16,
              pipeline_matmul_id_bf16,
              matmul_id_bf16, ,
              wg_denoms,
              warptile,
              vk_mat_mat_id_push_constants,
              4,
              _id,
              0);
  }
  #undef CREATE_MM

  // mul mat vec
  // the number of rows computed per shader depends on GPU model and quant
  uint32_t rm_stdq = 1;
  uint32_t rm_kq   = 2;
  if (device->vendor_id == VK_VENDOR_ID_AMD) {
    if (device->architecture == AMD_GCN) {
      rm_stdq = 2;
      rm_kq   = 4;
    }
  }
  else if (device->vendor_id == VK_VENDOR_ID_INTEL)
    rm_stdq = 2;
  uint32_t rm_iq = 2 * rm_kq;

  const bool use_subgroups = device->subgroup_arithmetic && device->architecture != vk_device_architecture::AMD_GCN;
  // Ensure a subgroup size >= 16 is available
  const bool use_subgroups16 = use_subgroups && subgroup_min_size_16;

  const uint32_t subgroup_size = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_size_control && device->
                                   subgroup_min_size <= 16 && device->subgroup_max_size >= 16)
                                   ? 16
                                   : device->subgroup_size;
  const uint32_t subgroup_size16 = std::max(subgroup_size, 16u);

  const uint32_t force_subgroup_size   = use_subgroups ? subgroup_size : 0;
  const uint32_t force_subgroup_size16 = use_subgroups16 ? subgroup_size16 : 0;

  for (uint32_t w = 0; w < DMMV_WG_SIZE_COUNT; ++w) {
    const uint32_t wg_size_subgroup   = (w == DMMV_WG_SIZE_SUBGROUP) ? subgroup_size : (subgroup_size * 4);
    const uint32_t wg_size_subgroup16 = (w == DMMV_WG_SIZE_SUBGROUP) ? subgroup_size16 : (subgroup_size16 * 4);

    const shader_reduction_mode reduc = (use_subgroups && w == DMMV_WG_SIZE_SUBGROUP)
                                          ? SHADER_REDUCTION_MODE_SUBGROUP
                                          : (use_subgroups && w == DMMV_WG_SIZE_LARGE)
                                          ? SHADER_REDUCTION_MODE_HYBRID
                                          : SHADER_REDUCTION_MODE_SHMEM;

    const shader_reduction_mode reduc16 = (use_subgroups16 && w == DMMV_WG_SIZE_SUBGROUP)
                                            ? SHADER_REDUCTION_MODE_SUBGROUP
                                            : (use_subgroups16 && w == DMMV_WG_SIZE_LARGE)
                                            ? SHADER_REDUCTION_MODE_HYBRID
                                            : SHADER_REDUCTION_MODE_SHMEM;

    for (uint32_t i = 0; i < mul_mat_vec_max_cols; ++i) {
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f32_f32[w][v_TYPE_F32][i],
                              "mul_mat_vec_f32_f32_f32",
                              arr_dmmv_f32_f32_f32_len[reduc],
                              arr_dmmv_f32_f32_f32_data[reduc],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {2, 1, 1},
                              {wg_size_subgroup, 2, i + 1},
                              1,
                              false,
                              use_subgroups,
                              force_subgroup_size);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f32_f32[w][v_TYPE_F16][i],
                              "mul_mat_vec_f16_f32_f32",
                              arr_dmmv_f16_f32_f32_len[reduc],
                              arr_dmmv_f16_f32_f32_data[reduc],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {2, 1, 1},
                              {wg_size_subgroup, 2, i + 1},
                              1,
                              false,
                              use_subgroups,
                              force_subgroup_size);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f32_f32[w][v_TYPE_BF16][i],
                              "mul_mat_vec_bf16_f32_f32",
                              arr_dmmv_bf16_f32_f32_len[reduc],
                              arr_dmmv_bf16_f32_f32_data[reduc],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {2, 1, 1},
                              {wg_size_subgroup, 2, i + 1},
                              1,
                              false,
                              use_subgroups,
                              force_subgroup_size);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f32_f32[w][v_TYPE_Q4_0][i],
                              "mul_mat_vec_q4_0_f32_f32",
                              arr_dmmv_q4_0_f32_f32_len[reduc],
                              arr_dmmv_q4_0_f32_f32_data[reduc],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {2 * rm_stdq, 1, 1},
                              {wg_size_subgroup, 2 * rm_stdq, i + 1},
                              1,
                              true,
                              use_subgroups,
                              force_subgroup_size);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f32_f32[w][v_TYPE_Q4_1][i],
                              "mul_mat_vec_q4_1_f32_f32",
                              arr_dmmv_q4_1_f32_f32_len[reduc],
                              arr_dmmv_q4_1_f32_f32_data[reduc],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {2 * rm_stdq, 1, 1},
                              {wg_size_subgroup, 2 * rm_stdq, i + 1},
                              1,
                              true,
                              use_subgroups,
                              force_subgroup_size);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f32_f32[w][v_TYPE_Q5_0][i],
                              "mul_mat_vec_q5_0_f32_f32",
                              arr_dmmv_q5_0_f32_f32_len[reduc],
                              arr_dmmv_q5_0_f32_f32_data[reduc],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {2 * rm_stdq, 1, 1},
                              {wg_size_subgroup, 2 * rm_stdq, i + 1},
                              1,
                              true,
                              use_subgroups,
                              force_subgroup_size);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f32_f32[w][v_TYPE_Q5_1][i],
                              "mul_mat_vec_q5_1_f32_f32",
                              arr_dmmv_q5_1_f32_f32_len[reduc],
                              arr_dmmv_q5_1_f32_f32_data[reduc],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {2 * rm_stdq, 1, 1},
                              {wg_size_subgroup, 2 * rm_stdq, i + 1},
                              1,
                              true,
                              use_subgroups,
                              force_subgroup_size);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f32_f32[w][v_TYPE_Q8_0][i],
                              "mul_mat_vec_q8_0_f32_f32",
                              arr_dmmv_q8_0_f32_f32_len[reduc],
                              arr_dmmv_q8_0_f32_f32_data[reduc],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {1 * rm_stdq, 1, 1},
                              {wg_size_subgroup, 1 * rm_stdq, i + 1},
                              1,
                              true,
                              use_subgroups,
                              force_subgroup_size);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f32_f32[w][v_TYPE_Q2_K][i],
                              "mul_mat_vec_q2_k_f32_f32",
                              arr_dmmv_q2_k_f32_f32_len[reduc16],
                              arr_dmmv_q2_k_f32_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_kq, 1, 1},
                              {wg_size_subgroup16, rm_kq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f32_f32[w][v_TYPE_Q3_K][i],
                              "mul_mat_vec_q3_k_f32_f32",
                              arr_dmmv_q3_k_f32_f32_len[reduc16],
                              arr_dmmv_q3_k_f32_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_kq, 1, 1},
                              {wg_size_subgroup16, rm_kq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f32_f32[w][v_TYPE_Q4_K][i],
                              "mul_mat_vec_q4_k_f32_f32",
                              arr_dmmv_q4_k_f32_f32_len[reduc16],
                              arr_dmmv_q4_k_f32_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_kq, 1, 1},
                              {wg_size_subgroup16, rm_kq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f32_f32[w][v_TYPE_Q5_K][i],
                              "mul_mat_vec_q5_k_f32_f32",
                              arr_dmmv_q5_k_f32_f32_len[reduc16],
                              arr_dmmv_q5_k_f32_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_kq, 1, 1},
                              {wg_size_subgroup16, rm_kq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f32_f32[w][v_TYPE_Q6_K][i],
                              "mul_mat_vec_q6_k_f32_f32",
                              arr_dmmv_q6_k_f32_f32_len[reduc16],
                              arr_dmmv_q6_k_f32_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_kq, 1, 1},
                              {wg_size_subgroup16, rm_kq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f32_f32[w][v_TYPE_IQ1_S][i],
                              "mul_mat_vec_iq1_s_f32_f32",
                              arr_dmmv_iq1_s_f32_f32_len[reduc16],
                              arr_dmmv_iq1_s_f32_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_iq, 1, 1},
                              {wg_size_subgroup16, rm_iq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f32_f32[w][v_TYPE_IQ1_M][i],
                              "mul_mat_vec_iq1_m_f32_f32",
                              arr_dmmv_iq1_m_f32_f32_len[reduc16],
                              arr_dmmv_iq1_m_f32_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_iq, 1, 1},
                              {wg_size_subgroup16, rm_iq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f32_f32[w][v_TYPE_IQ2_XXS][i],
                              "mul_mat_vec_iq2_xxs_f32_f32",
                              arr_dmmv_iq2_xxs_f32_f32_len[reduc16],
                              arr_dmmv_iq2_xxs_f32_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_iq, 1, 1},
                              {wg_size_subgroup16, rm_iq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f32_f32[w][v_TYPE_IQ2_XS][i],
                              "mul_mat_vec_iq2_xs_f32_f32",
                              arr_dmmv_iq2_xs_f32_f32_len[reduc16],
                              arr_dmmv_iq2_xs_f32_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_iq, 1, 1},
                              {wg_size_subgroup16, rm_iq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f32_f32[w][v_TYPE_IQ2_S][i],
                              "mul_mat_vec_iq2_s_f32_f32",
                              arr_dmmv_iq2_s_f32_f32_len[reduc16],
                              arr_dmmv_iq2_s_f32_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_iq, 1, 1},
                              {wg_size_subgroup16, rm_iq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f32_f32[w][v_TYPE_IQ3_XXS][i],
                              "mul_mat_vec_iq3_xxs_f32_f32",
                              arr_dmmv_iq3_xxs_f32_f32_len[reduc16],
                              arr_dmmv_iq3_xxs_f32_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_iq, 1, 1},
                              {wg_size_subgroup16, rm_iq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f32_f32[w][v_TYPE_IQ3_S][i],
                              "mul_mat_vec_iq3_s_f32_f32",
                              arr_dmmv_iq3_s_f32_f32_len[reduc16],
                              arr_dmmv_iq3_s_f32_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_iq, 1, 1},
                              {wg_size_subgroup16, rm_iq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f32_f32[w][v_TYPE_IQ4_XS][i],
                              "mul_mat_vec_iq4_xs_f32_f32",
                              arr_dmmv_iq4_xs_f32_f32_len[reduc16],
                              arr_dmmv_iq4_xs_f32_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_iq, 1, 1},
                              {wg_size_subgroup16, rm_iq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f32_f32[w][v_TYPE_IQ4_NL][i],
                              "mul_mat_vec_iq4_nl_f32_f32",
                              arr_dmmv_iq4_nl_f32_f32_len[reduc16],
                              arr_dmmv_iq4_nl_f32_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_iq, 1, 1},
                              {wg_size_subgroup16, rm_iq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f32_f32[w][v_TYPE_MXFP4][i],
                              "mul_mat_vec_mxfp4_f32_f32",
                              arr_dmmv_mxfp4_f32_f32_len[reduc16],
                              arr_dmmv_mxfp4_f32_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_iq, 1, 1},
                              {wg_size_subgroup16, rm_iq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);

      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f16_f32[w][v_TYPE_F32][i],
                              "mul_mat_vec_f32_f16_f32",
                              arr_dmmv_f32_f16_f32_len[reduc],
                              arr_dmmv_f32_f16_f32_data[reduc],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {2, 1, 1},
                              {wg_size_subgroup, 2, i + 1},
                              1,
                              false,
                              use_subgroups,
                              force_subgroup_size);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f16_f32[w][v_TYPE_F16][i],
                              "mul_mat_vec_f16_f16_f32",
                              arr_dmmv_f16_f16_f32_len[reduc],
                              arr_dmmv_f16_f16_f32_data[reduc],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {2, 1, 1},
                              {wg_size_subgroup, 2, i + 1},
                              1,
                              false,
                              use_subgroups,
                              force_subgroup_size);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f16_f32[w][v_TYPE_BF16][i],
                              "mul_mat_vec_bf16_f16_f32",
                              arr_dmmv_bf16_f16_f32_len[reduc],
                              arr_dmmv_bf16_f16_f32_data[reduc],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {2, 1, 1},
                              {wg_size_subgroup, 2, i + 1},
                              1,
                              false,
                              use_subgroups,
                              force_subgroup_size);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f16_f32[w][v_TYPE_Q4_0][i],
                              "mul_mat_vec_q4_0_f16_f32",
                              arr_dmmv_q4_0_f16_f32_len[reduc],
                              arr_dmmv_q4_0_f16_f32_data[reduc],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {2 * rm_stdq, 1, 1},
                              {wg_size_subgroup, 2 * rm_stdq, i + 1},
                              1,
                              true,
                              use_subgroups,
                              force_subgroup_size);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f16_f32[w][v_TYPE_Q4_1][i],
                              "mul_mat_vec_q4_1_f16_f32",
                              arr_dmmv_q4_1_f16_f32_len[reduc],
                              arr_dmmv_q4_1_f16_f32_data[reduc],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {2 * rm_stdq, 1, 1},
                              {wg_size_subgroup, 2 * rm_stdq, i + 1},
                              1,
                              true,
                              use_subgroups,
                              force_subgroup_size);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f16_f32[w][v_TYPE_Q5_0][i],
                              "mul_mat_vec_q5_0_f16_f32",
                              arr_dmmv_q5_0_f16_f32_len[reduc],
                              arr_dmmv_q5_0_f16_f32_data[reduc],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {2 * rm_stdq, 1, 1},
                              {wg_size_subgroup, 2 * rm_stdq, i + 1},
                              1,
                              true,
                              use_subgroups,
                              force_subgroup_size);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f16_f32[w][v_TYPE_Q5_1][i],
                              "mul_mat_vec_q5_1_f16_f32",
                              arr_dmmv_q5_1_f16_f32_len[reduc],
                              arr_dmmv_q5_1_f16_f32_data[reduc],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {2 * rm_stdq, 1, 1},
                              {wg_size_subgroup, 2 * rm_stdq, i + 1},
                              1,
                              true,
                              use_subgroups,
                              force_subgroup_size);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f16_f32[w][v_TYPE_Q8_0][i],
                              "mul_mat_vec_q8_0_f16_f32",
                              arr_dmmv_q8_0_f16_f32_len[reduc],
                              arr_dmmv_q8_0_f16_f32_data[reduc],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {1 * rm_stdq, 1, 1},
                              {wg_size_subgroup, 1 * rm_stdq, i + 1},
                              1,
                              true,
                              use_subgroups,
                              force_subgroup_size);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f16_f32[w][v_TYPE_Q2_K][i],
                              "mul_mat_vec_q2_k_f16_f32",
                              arr_dmmv_q2_k_f16_f32_len[reduc16],
                              arr_dmmv_q2_k_f16_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_kq, 1, 1},
                              {wg_size_subgroup16, rm_kq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f16_f32[w][v_TYPE_Q3_K][i],
                              "mul_mat_vec_q3_k_f16_f32",
                              arr_dmmv_q3_k_f16_f32_len[reduc16],
                              arr_dmmv_q3_k_f16_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_kq, 1, 1},
                              {wg_size_subgroup16, rm_kq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f16_f32[w][v_TYPE_Q4_K][i],
                              "mul_mat_vec_q4_k_f16_f32",
                              arr_dmmv_q4_k_f16_f32_len[reduc16],
                              arr_dmmv_q4_k_f16_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_kq, 1, 1},
                              {wg_size_subgroup16, rm_kq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f16_f32[w][v_TYPE_Q5_K][i],
                              "mul_mat_vec_q5_k_f16_f32",
                              arr_dmmv_q5_k_f16_f32_len[reduc16],
                              arr_dmmv_q5_k_f16_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_kq, 1, 1},
                              {wg_size_subgroup16, rm_kq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f16_f32[w][v_TYPE_Q6_K][i],
                              "mul_mat_vec_q6_k_f16_f32",
                              arr_dmmv_q6_k_f16_f32_len[reduc16],
                              arr_dmmv_q6_k_f16_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_kq, 1, 1},
                              {wg_size_subgroup16, rm_kq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f16_f32[w][v_TYPE_IQ1_S][i],
                              "mul_mat_vec_iq1_s_f16_f32",
                              arr_dmmv_iq1_s_f16_f32_len[reduc16],
                              arr_dmmv_iq1_s_f16_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_iq, 1, 1},
                              {wg_size_subgroup16, rm_iq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f16_f32[w][v_TYPE_IQ1_M][i],
                              "mul_mat_vec_iq1_m_f16_f32",
                              arr_dmmv_iq1_m_f16_f32_len[reduc16],
                              arr_dmmv_iq1_m_f16_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_iq, 1, 1},
                              {wg_size_subgroup16, rm_iq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f16_f32[w][v_TYPE_IQ2_XXS][i],
                              "mul_mat_vec_iq2_xxs_f16_f32",
                              arr_dmmv_iq2_xxs_f16_f32_len[reduc16],
                              arr_dmmv_iq2_xxs_f16_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_iq, 1, 1},
                              {wg_size_subgroup16, rm_iq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f16_f32[w][v_TYPE_IQ2_XS][i],
                              "mul_mat_vec_iq2_xs_f16_f32",
                              arr_dmmv_iq2_xs_f16_f32_len[reduc16],
                              arr_dmmv_iq2_xs_f16_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_iq, 1, 1},
                              {wg_size_subgroup16, rm_iq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f16_f32[w][v_TYPE_IQ2_S][i],
                              "mul_mat_vec_iq2_s_f16_f32",
                              arr_dmmv_iq2_s_f16_f32_len[reduc16],
                              arr_dmmv_iq2_s_f16_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_iq, 1, 1},
                              {wg_size_subgroup16, rm_iq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f16_f32[w][v_TYPE_IQ3_XXS][i],
                              "mul_mat_vec_iq3_xxs_f16_f32",
                              arr_dmmv_iq3_xxs_f16_f32_len[reduc16],
                              arr_dmmv_iq3_xxs_f16_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_iq, 1, 1},
                              {wg_size_subgroup16, rm_iq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f16_f32[w][v_TYPE_IQ3_S][i],
                              "mul_mat_vec_iq3_s_f16_f32",
                              arr_dmmv_iq3_s_f16_f32_len[reduc16],
                              arr_dmmv_iq3_s_f16_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_iq, 1, 1},
                              {wg_size_subgroup16, rm_iq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f16_f32[w][v_TYPE_IQ4_XS][i],
                              "mul_mat_vec_iq4_xs_f16_f32",
                              arr_dmmv_iq4_xs_f16_f32_len[reduc16],
                              arr_dmmv_iq4_xs_f16_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_iq, 1, 1},
                              {wg_size_subgroup16, rm_iq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f16_f32[w][v_TYPE_IQ4_NL][i],
                              "mul_mat_vec_iq4_nl_f16_f32",
                              arr_dmmv_iq4_nl_f16_f32_len[reduc16],
                              arr_dmmv_iq4_nl_f16_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_iq, 1, 1},
                              {wg_size_subgroup16, rm_iq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);
      v_vk_create_pipeline(device,
                              device->pipeline_dequant_mul_mat_vec_f16_f32[w][v_TYPE_MXFP4][i],
                              "mul_mat_vec_mxfp4_f16_f32",
                              arr_dmmv_mxfp4_f16_f32_len[reduc16],
                              arr_dmmv_mxfp4_f16_f32_data[reduc16],
                              "main",
                              3,
                              sizeof(vk_mat_vec_push_constants),
                              {rm_iq, 1, 1},
                              {wg_size_subgroup16, rm_iq, i + 1},
                              1,
                              true,
                              use_subgroups16,
                              force_subgroup_size16);

      #if defined(V_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
      if (device->integer_dot_product) {
        const uint32_t subgroup_size_int = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_size_control)
                                             ? device->subgroup_min_size
                                             : device->subgroup_size;
        const uint32_t wg_size_subgroup_int =
          (w == DMMV_WG_SIZE_SUBGROUP) ? subgroup_size_int : (subgroup_size_int * 4);

        v_vk_create_pipeline(device,
                                device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][v_TYPE_Q4_0][i],
                                "mul_mat_vec_q4_0_q8_1_f32",
                                arr_dmmv_q4_0_q8_1_f32_len[reduc],
                                arr_dmmv_q4_0_q8_1_f32_data[reduc],
                                "main",
                                3,
                                sizeof(vk_mat_vec_push_constants),
                                {2 * rm_stdq, 1, 1},
                                {wg_size_subgroup_int, 2 * rm_stdq, i + 1},
                                1,
                                true,
                                use_subgroups,
                                subgroup_size_int);
        v_vk_create_pipeline(device,
                                device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][v_TYPE_Q4_1][i],
                                "mul_mat_vec_q4_1_q8_1_f32",
                                arr_dmmv_q4_1_q8_1_f32_len[reduc],
                                arr_dmmv_q4_1_q8_1_f32_data[reduc],
                                "main",
                                3,
                                sizeof(vk_mat_vec_push_constants),
                                {2 * rm_stdq, 1, 1},
                                {wg_size_subgroup_int, 2 * rm_stdq, i + 1},
                                1,
                                true,
                                use_subgroups,
                                subgroup_size_int);
        v_vk_create_pipeline(device,
                                device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][v_TYPE_Q5_0][i],
                                "mul_mat_vec_q5_0_q8_1_f32",
                                arr_dmmv_q5_0_q8_1_f32_len[reduc],
                                arr_dmmv_q5_0_q8_1_f32_data[reduc],
                                "main",
                                3,
                                sizeof(vk_mat_vec_push_constants),
                                {2 * rm_stdq, 1, 1},
                                {wg_size_subgroup_int, 2 * rm_stdq, i + 1},
                                1,
                                true,
                                use_subgroups,
                                subgroup_size_int);
        v_vk_create_pipeline(device,
                                device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][v_TYPE_Q5_1][i],
                                "mul_mat_vec_q5_1_q8_1_f32",
                                arr_dmmv_q5_1_q8_1_f32_len[reduc],
                                arr_dmmv_q5_1_q8_1_f32_data[reduc],
                                "main",
                                3,
                                sizeof(vk_mat_vec_push_constants),
                                {2 * rm_stdq, 1, 1},
                                {wg_size_subgroup_int, 2 * rm_stdq, i + 1},
                                1,
                                true,
                                use_subgroups,
                                subgroup_size_int);
        v_vk_create_pipeline(device,
                                device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][v_TYPE_Q8_0][i],
                                "mul_mat_vec_q8_0_q8_1_f32",
                                arr_dmmv_q8_0_q8_1_f32_len[reduc],
                                arr_dmmv_q8_0_q8_1_f32_data[reduc],
                                "main",
                                3,
                                sizeof(vk_mat_vec_push_constants),
                                {1 * rm_stdq, 1, 1},
                                {wg_size_subgroup_int, 1 * rm_stdq, i + 1},
                                1,
                                true,
                                use_subgroups,
                                subgroup_size_int);
      }
      #endif // V_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
    }
  }

  v_vk_create_pipeline(device,
                          device->pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_F32],
                          "mul_mat_vec_id_f32_f32",
                          mul_mat_vec_id_f32_f32_len,
                          mul_mat_vec_id_f32_f32_data,
                          "main",
                          4,
                          sizeof(vk_mat_vec_id_push_constants),
                          {2, 1, 1},
                          {device->subgroup_size, 2},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_F16],
                          "mul_mat_vec_id_f16_f32",
                          mul_mat_vec_id_f16_f32_len,
                          mul_mat_vec_id_f16_f32_data,
                          "main",
                          4,
                          sizeof(vk_mat_vec_id_push_constants),
                          {2, 1, 1},
                          {device->subgroup_size, 2},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_BF16],
                          "mul_mat_vec_id_bf16_f32",
                          mul_mat_vec_id_bf16_f32_len,
                          mul_mat_vec_id_bf16_f32_data,
                          "main",
                          4,
                          sizeof(vk_mat_vec_id_push_constants),
                          {2, 1, 1},
                          {device->subgroup_size, 2},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_Q4_0],
                          "mul_mat_vec_id_q4_0_f32",
                          mul_mat_vec_id_q4_0_f32_len,
                          mul_mat_vec_id_q4_0_f32_data,
                          "main",
                          4,
                          sizeof(vk_mat_vec_id_push_constants),
                          {2 * rm_stdq, 1, 1},
                          {device->subgroup_size, 2 * rm_stdq},
                          1,
                          true);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_Q4_1],
                          "mul_mat_vec_id_q4_1_f32",
                          mul_mat_vec_id_q4_1_f32_len,
                          mul_mat_vec_id_q4_1_f32_data,
                          "main",
                          4,
                          sizeof(vk_mat_vec_id_push_constants),
                          {2 * rm_stdq, 1, 1},
                          {device->subgroup_size, 2 * rm_stdq},
                          1,
                          true);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_Q5_0],
                          "mul_mat_vec_id_q5_0_f32",
                          mul_mat_vec_id_q5_0_f32_len,
                          mul_mat_vec_id_q5_0_f32_data,
                          "main",
                          4,
                          sizeof(vk_mat_vec_id_push_constants),
                          {2 * rm_stdq, 1, 1},
                          {device->subgroup_size, 2 * rm_stdq},
                          1,
                          true);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_Q5_1],
                          "mul_mat_vec_id_q5_1_f32",
                          mul_mat_vec_id_q5_1_f32_len,
                          mul_mat_vec_id_q5_1_f32_data,
                          "main",
                          4,
                          sizeof(vk_mat_vec_id_push_constants),
                          {2 * rm_stdq, 1, 1},
                          {device->subgroup_size, 2 * rm_stdq},
                          1,
                          true);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_Q8_0],
                          "mul_mat_vec_id_q8_0_f32",
                          mul_mat_vec_id_q8_0_f32_len,
                          mul_mat_vec_id_q8_0_f32_data,
                          "main",
                          4,
                          sizeof(vk_mat_vec_id_push_constants),
                          {1 * rm_stdq, 1, 1},
                          {device->subgroup_size, 1 * rm_stdq},
                          1,
                          true);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_Q2_K],
                          "mul_mat_vec_id_q2_k_f32",
                          mul_mat_vec_id_q2_k_f32_len,
                          mul_mat_vec_id_q2_k_f32_data,
                          "main",
                          4,
                          sizeof(vk_mat_vec_id_push_constants),
                          {rm_kq, 1, 1},
                          {subgroup_size_16, rm_kq},
                          1,
                          true);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_Q3_K],
                          "mul_mat_vec_id_q3_k_f32",
                          mul_mat_vec_id_q3_k_f32_len,
                          mul_mat_vec_id_q3_k_f32_data,
                          "main",
                          4,
                          sizeof(vk_mat_vec_id_push_constants),
                          {rm_kq, 1, 1},
                          {subgroup_size_16, rm_kq},
                          1,
                          true);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_Q4_K],
                          "mul_mat_vec_id_q4_k_f32",
                          mul_mat_vec_id_q4_k_f32_len,
                          mul_mat_vec_id_q4_k_f32_data,
                          "main",
                          4,
                          sizeof(vk_mat_vec_id_push_constants),
                          {rm_kq, 1, 1},
                          {subgroup_size_16, rm_kq},
                          1,
                          true);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_Q5_K],
                          "mul_mat_vec_id_q5_k_f32",
                          mul_mat_vec_id_q5_k_f32_len,
                          mul_mat_vec_id_q5_k_f32_data,
                          "main",
                          4,
                          sizeof(vk_mat_vec_id_push_constants),
                          {rm_kq, 1, 1},
                          {subgroup_size_16, rm_kq},
                          1,
                          true);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_Q6_K],
                          "mul_mat_vec_id_q6_k_f32",
                          mul_mat_vec_id_q6_k_f32_len,
                          mul_mat_vec_id_q6_k_f32_data,
                          "main",
                          4,
                          sizeof(vk_mat_vec_id_push_constants),
                          {rm_kq, 1, 1},
                          {subgroup_size_16, rm_kq},
                          1,
                          true);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_IQ1_S],
                          "mul_mat_vec_id_iq1_s_f32",
                          mul_mat_vec_id_iq1_s_f32_len,
                          mul_mat_vec_id_iq1_s_f32_data,
                          "main",
                          4,
                          sizeof(vk_mat_vec_id_push_constants),
                          {rm_iq, 1, 1},
                          {subgroup_size_16, rm_iq},
                          1,
                          true);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_IQ1_M],
                          "mul_mat_vec_id_iq1_m_f32",
                          mul_mat_vec_id_iq1_m_f32_len,
                          mul_mat_vec_id_iq1_m_f32_data,
                          "main",
                          4,
                          sizeof(vk_mat_vec_id_push_constants),
                          {rm_iq, 1, 1},
                          {subgroup_size_16, rm_iq},
                          1,
                          true);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_IQ2_XXS],
                          "mul_mat_vec_id_iq2_xxs_f32",
                          mul_mat_vec_id_iq2_xxs_f32_len,
                          mul_mat_vec_id_iq2_xxs_f32_data,
                          "main",
                          4,
                          sizeof(vk_mat_vec_id_push_constants),
                          {rm_iq, 1, 1},
                          {subgroup_size_16, rm_iq},
                          1,
                          true);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_IQ2_XS],
                          "mul_mat_vec_id_iq2_xs_f32",
                          mul_mat_vec_id_iq2_xs_f32_len,
                          mul_mat_vec_id_iq2_xs_f32_data,
                          "main",
                          4,
                          sizeof(vk_mat_vec_id_push_constants),
                          {rm_iq, 1, 1},
                          {subgroup_size_16, rm_iq},
                          1,
                          true);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_IQ2_S],
                          "mul_mat_vec_id_iq2_s_f32",
                          mul_mat_vec_id_iq2_s_f32_len,
                          mul_mat_vec_id_iq2_s_f32_data,
                          "main",
                          4,
                          sizeof(vk_mat_vec_id_push_constants),
                          {rm_iq, 1, 1},
                          {subgroup_size_16, rm_iq},
                          1,
                          true);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_IQ3_XXS],
                          "mul_mat_vec_id_iq3_xxs_f32",
                          mul_mat_vec_id_iq3_xxs_f32_len,
                          mul_mat_vec_id_iq3_xxs_f32_data,
                          "main",
                          4,
                          sizeof(vk_mat_vec_id_push_constants),
                          {rm_iq, 1, 1},
                          {subgroup_size_16, rm_iq},
                          1,
                          true);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_IQ3_S],
                          "mul_mat_vec_id_iq3_s_f32",
                          mul_mat_vec_id_iq3_s_f32_len,
                          mul_mat_vec_id_iq3_s_f32_data,
                          "main",
                          4,
                          sizeof(vk_mat_vec_id_push_constants),
                          {rm_iq, 1, 1},
                          {subgroup_size_16, rm_iq},
                          1,
                          true);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_IQ4_XS],
                          "mul_mat_vec_id_iq4_xs_f32",
                          mul_mat_vec_id_iq4_xs_f32_len,
                          mul_mat_vec_id_iq4_xs_f32_data,
                          "main",
                          4,
                          sizeof(vk_mat_vec_id_push_constants),
                          {rm_iq, 1, 1},
                          {subgroup_size_16, rm_iq},
                          1,
                          true);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_IQ4_NL],
                          "mul_mat_vec_id_iq4_nl_f32",
                          mul_mat_vec_id_iq4_nl_f32_len,
                          mul_mat_vec_id_iq4_nl_f32_data,
                          "main",
                          4,
                          sizeof(vk_mat_vec_id_push_constants),
                          {rm_iq, 1, 1},
                          {subgroup_size_16, rm_iq},
                          1,
                          true);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant_mul_mat_vec_id_f32[v_TYPE_MXFP4],
                          "mul_mat_vec_id_mxfp4_f32",
                          mul_mat_vec_id_mxfp4_f32_len,
                          mul_mat_vec_id_mxfp4_f32_data,
                          "main",
                          4,
                          sizeof(vk_mat_vec_id_push_constants),
                          {rm_iq, 1, 1},
                          {subgroup_size_16, rm_iq},
                          1,
                          true);

  // dequant shaders
  v_vk_create_pipeline(device,
                          device->pipeline_dequant[v_TYPE_F32],
                          "f32_to_f16",
                          dequant_f32_len,
                          dequant_f32_data,
                          "main",
                          2,
                          5 * sizeof(uint32_t),
                          {256 * 16, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant[v_TYPE_Q4_0],
                          "dequant_q4_0",
                          dequant_q4_0_len,
                          dequant_q4_0_data,
                          "main",
                          2,
                          5 * sizeof(uint32_t),
                          {256 * 16, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant[v_TYPE_Q4_1],
                          "dequant_q4_1",
                          dequant_q4_1_len,
                          dequant_q4_1_data,
                          "main",
                          2,
                          5 * sizeof(uint32_t),
                          {256 * 16, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant[v_TYPE_Q5_0],
                          "dequant_q5_0",
                          dequant_q5_0_len,
                          dequant_q5_0_data,
                          "main",
                          2,
                          5 * sizeof(uint32_t),
                          {256 * 16, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant[v_TYPE_Q5_1],
                          "dequant_q5_1",
                          dequant_q5_1_len,
                          dequant_q5_1_data,
                          "main",
                          2,
                          5 * sizeof(uint32_t),
                          {256 * 16, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant[v_TYPE_Q8_0],
                          "dequant_q8_0",
                          dequant_q8_0_len,
                          dequant_q8_0_data,
                          "main",
                          2,
                          5 * sizeof(uint32_t),
                          {256 * 16, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant[v_TYPE_Q2_K],
                          "dequant_q2_k",
                          dequant_q2_k_len,
                          dequant_q2_k_data,
                          "main",
                          2,
                          5 * sizeof(uint32_t),
                          {256 * 64, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant[v_TYPE_Q3_K],
                          "dequant_q3_k",
                          dequant_q3_k_len,
                          dequant_q3_k_data,
                          "main",
                          2,
                          5 * sizeof(uint32_t),
                          {256 * 64, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant[v_TYPE_Q4_K],
                          "dequant_q4_k",
                          dequant_q4_k_len,
                          dequant_q4_k_data,
                          "main",
                          2,
                          5 * sizeof(uint32_t),
                          {256 * 32, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant[v_TYPE_Q5_K],
                          "dequant_q5_k",
                          dequant_q5_k_len,
                          dequant_q5_k_data,
                          "main",
                          2,
                          5 * sizeof(uint32_t),
                          {256 * 64, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant[v_TYPE_Q6_K],
                          "dequant_q6_k",
                          dequant_q6_k_len,
                          dequant_q6_k_data,
                          "main",
                          2,
                          5 * sizeof(uint32_t),
                          {256 * 64, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant[v_TYPE_IQ1_S],
                          "dequant_iq1_s",
                          dequant_iq1_s_len,
                          dequant_iq1_s_data,
                          "main",
                          2,
                          5 * sizeof(uint32_t),
                          {256 * 32, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant[v_TYPE_IQ1_M],
                          "dequant_iq1_m",
                          dequant_iq1_m_len,
                          dequant_iq1_m_data,
                          "main",
                          2,
                          5 * sizeof(uint32_t),
                          {256 * 32, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant[v_TYPE_IQ2_XXS],
                          "dequant_iq2_xxs",
                          dequant_iq2_xxs_len,
                          dequant_iq2_xxs_data,
                          "main",
                          2,
                          5 * sizeof(uint32_t),
                          {256 * 32, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant[v_TYPE_IQ2_XS],
                          "dequant_iq2_xs",
                          dequant_iq2_xs_len,
                          dequant_iq2_xs_data,
                          "main",
                          2,
                          5 * sizeof(uint32_t),
                          {256 * 32, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant[v_TYPE_IQ2_S],
                          "dequant_iq2_s",
                          dequant_iq2_s_len,
                          dequant_iq2_s_data,
                          "main",
                          2,
                          5 * sizeof(uint32_t),
                          {256 * 32, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant[v_TYPE_IQ3_XXS],
                          "dequant_iq3_xxs",
                          dequant_iq3_xxs_len,
                          dequant_iq3_xxs_data,
                          "main",
                          2,
                          5 * sizeof(uint32_t),
                          {256 * 32, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant[v_TYPE_IQ3_S],
                          "dequant_iq3_s",
                          dequant_iq3_s_len,
                          dequant_iq3_s_data,
                          "main",
                          2,
                          5 * sizeof(uint32_t),
                          {256 * 32, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant[v_TYPE_IQ4_XS],
                          "dequant_iq4_xs",
                          dequant_iq4_xs_len,
                          dequant_iq4_xs_data,
                          "main",
                          2,
                          5 * sizeof(uint32_t),
                          {256 * 32, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant[v_TYPE_IQ4_NL],
                          "dequant_iq4_nl",
                          dequant_iq4_nl_len,
                          dequant_iq4_nl_data,
                          "main",
                          2,
                          5 * sizeof(uint32_t),
                          {256 * 16, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_dequant[v_TYPE_MXFP4],
                          "dequant_mxfp4",
                          dequant_mxfp4_len,
                          dequant_mxfp4_data,
                          "main",
                          2,
                          5 * sizeof(uint32_t),
                          {256 * 16, 1, 1},
                          {},
                          1);

  // get_rows
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows[v_TYPE_F32],
                          "get_rows_f32",
                          get_rows_f32_len,
                          get_rows_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows[v_TYPE_F16],
                          "get_rows_f16",
                          get_rows_f16_len,
                          get_rows_f16_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows[v_TYPE_BF16],
                          "get_rows_bf16",
                          get_rows_bf16_len,
                          get_rows_bf16_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows[v_TYPE_Q4_0],
                          "get_rows_q4_0",
                          get_rows_q4_0_len,
                          get_rows_q4_0_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows[v_TYPE_Q4_1],
                          "get_rows_q4_1",
                          get_rows_q4_1_len,
                          get_rows_q4_1_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows[v_TYPE_Q5_0],
                          "get_rows_q5_0",
                          get_rows_q5_0_len,
                          get_rows_q5_0_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows[v_TYPE_Q5_1],
                          "get_rows_q5_1",
                          get_rows_q5_1_len,
                          get_rows_q5_1_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows[v_TYPE_Q8_0],
                          "get_rows_q8_0",
                          get_rows_q8_0_len,
                          get_rows_q8_0_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows[v_TYPE_Q2_K],
                          "get_rows_q2_k",
                          get_rows_q2_k_len,
                          get_rows_q2_k_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows[v_TYPE_Q3_K],
                          "get_rows_q3_k",
                          get_rows_q3_k_len,
                          get_rows_q3_k_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows[v_TYPE_Q4_K],
                          "get_rows_q4_k",
                          get_rows_q4_k_len,
                          get_rows_q4_k_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows[v_TYPE_Q5_K],
                          "get_rows_q5_k",
                          get_rows_q5_k_len,
                          get_rows_q5_k_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows[v_TYPE_Q6_K],
                          "get_rows_q6_k",
                          get_rows_q6_k_len,
                          get_rows_q6_k_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows[v_TYPE_IQ1_S],
                          "get_rows_iq1_s",
                          get_rows_iq1_s_len,
                          get_rows_iq1_s_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows[v_TYPE_IQ1_M],
                          "get_rows_iq1_m",
                          get_rows_iq1_m_len,
                          get_rows_iq1_m_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows[v_TYPE_IQ2_XXS],
                          "get_rows_iq2_xxs",
                          get_rows_iq2_xxs_len,
                          get_rows_iq2_xxs_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows[v_TYPE_IQ2_XS],
                          "get_rows_iq2_xs",
                          get_rows_iq2_xs_len,
                          get_rows_iq2_xs_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows[v_TYPE_IQ2_S],
                          "get_rows_iq2_s",
                          get_rows_iq2_s_len,
                          get_rows_iq2_s_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows[v_TYPE_IQ3_XXS],
                          "get_rows_iq3_xxs",
                          get_rows_iq3_xxs_len,
                          get_rows_iq3_xxs_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows[v_TYPE_IQ3_S],
                          "get_rows_iq3_s",
                          get_rows_iq3_s_len,
                          get_rows_iq3_s_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows[v_TYPE_IQ4_XS],
                          "get_rows_iq4_xs",
                          get_rows_iq4_xs_len,
                          get_rows_iq4_xs_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows[v_TYPE_IQ4_NL],
                          "get_rows_iq4_nl",
                          get_rows_iq4_nl_len,
                          get_rows_iq4_nl_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows[v_TYPE_MXFP4],
                          "get_rows_mxfp4",
                          get_rows_mxfp4_len,
                          get_rows_mxfp4_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_get_rows_f32[v_TYPE_F32],
                          "get_rows_f32_f32",
                          get_rows_f32_f32_len,
                          get_rows_f32_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows_f32[v_TYPE_F16],
                          "get_rows_f16_f32",
                          get_rows_f16_f32_len,
                          get_rows_f16_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows_f32[v_TYPE_BF16],
                          "get_rows_bf16_f32",
                          get_rows_bf16_f32_len,
                          get_rows_bf16_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows_f32[v_TYPE_Q4_0],
                          "get_rows_q4_0_f32",
                          get_rows_q4_0_f32_len,
                          get_rows_q4_0_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows_f32[v_TYPE_Q4_1],
                          "get_rows_q4_1_f32",
                          get_rows_q4_1_f32_len,
                          get_rows_q4_1_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows_f32[v_TYPE_Q5_0],
                          "get_rows_q5_0_f32",
                          get_rows_q5_0_f32_len,
                          get_rows_q5_0_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows_f32[v_TYPE_Q5_1],
                          "get_rows_q5_1_f32",
                          get_rows_q5_1_f32_len,
                          get_rows_q5_1_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows_f32[v_TYPE_Q8_0],
                          "get_rows_q8_0_f32",
                          get_rows_q8_0_f32_len,
                          get_rows_q8_0_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows_f32[v_TYPE_Q2_K],
                          "get_rows_q2_k_f32",
                          get_rows_q2_k_f32_len,
                          get_rows_q2_k_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows_f32[v_TYPE_Q3_K],
                          "get_rows_q3_k_f32",
                          get_rows_q3_k_f32_len,
                          get_rows_q3_k_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows_f32[v_TYPE_Q4_K],
                          "get_rows_q4_k_f32",
                          get_rows_q4_k_f32_len,
                          get_rows_q4_k_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows_f32[v_TYPE_Q5_K],
                          "get_rows_q5_k_f32",
                          get_rows_q5_k_f32_len,
                          get_rows_q5_k_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows_f32[v_TYPE_Q6_K],
                          "get_rows_q6_k_f32",
                          get_rows_q6_k_f32_len,
                          get_rows_q6_k_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows_f32[v_TYPE_IQ1_S],
                          "get_rows_iq1_s_f32",
                          get_rows_iq1_s_f32_len,
                          get_rows_iq1_s_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows_f32[v_TYPE_IQ1_M],
                          "get_rows_iq1_m_f32",
                          get_rows_iq1_m_f32_len,
                          get_rows_iq1_m_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows_f32[v_TYPE_IQ2_XXS],
                          "get_rows_iq2_xxs_f32",
                          get_rows_iq2_xxs_f32_len,
                          get_rows_iq2_xxs_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows_f32[v_TYPE_IQ2_XS],
                          "get_rows_iq2_xs_f32",
                          get_rows_iq2_xs_f32_len,
                          get_rows_iq2_xs_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows_f32[v_TYPE_IQ2_S],
                          "get_rows_iq2_s_f32",
                          get_rows_iq2_s_f32_len,
                          get_rows_iq2_s_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows_f32[v_TYPE_IQ3_XXS],
                          "get_rows_iq3_xxs_f32",
                          get_rows_iq3_xxs_f32_len,
                          get_rows_iq3_xxs_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows_f32[v_TYPE_IQ3_S],
                          "get_rows_iq3_s_f32",
                          get_rows_iq3_s_f32_len,
                          get_rows_iq3_s_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows_f32[v_TYPE_IQ4_XS],
                          "get_rows_iq4_xs_f32",
                          get_rows_iq4_xs_f32_len,
                          get_rows_iq4_xs_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows_f32[v_TYPE_IQ4_NL],
                          "get_rows_iq4_nl_f32",
                          get_rows_iq4_nl_f32_len,
                          get_rows_iq4_nl_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_get_rows_f32[v_TYPE_MXFP4],
                          "get_rows_mxfp4_f32",
                          get_rows_mxfp4_f32_len,
                          get_rows_mxfp4_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {1024, 1, 1},
                          {},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_matmul_split_k_reduce,
                          "split_k_reduce",
                          split_k_reduce_len,
                          split_k_reduce_data,
                          "main",
                          2,
                          2 * sizeof(uint32_t),
                          {256 * 4, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_flash_attn_split_k_reduce,
                          "fa_split_k_reduce",
                          fa_split_k_reduce_len,
                          fa_split_k_reduce_data,
                          "main",
                          3,
                          5 * sizeof(uint32_t),
                          {1, device->subgroup_size, 1},
                          {device->subgroup_size},
                          1,
                          true);

  if (device->subgroup_clustered && device->subgroup_require_full_support) {
    v_vk_create_pipeline(device,
                            device->pipeline_quantize_q8_1,
                            "quantize_q8_1",
                            quantize_q8_1_subgroup_len,
                            quantize_q8_1_subgroup_data,
                            "main",
                            2,
                            1 * sizeof(uint32_t),
                            {32 * device->subgroup_size / 8, 1, 1},
                            {device->subgroup_size},
                            1,
                            true,
                            true);
    v_vk_create_pipeline(device,
                            device->pipeline_quantize_q8_1_x4,
                            "quantize_q8_1_x4",
                            quantize_q8_1_x4_subgroup_len,
                            quantize_q8_1_x4_subgroup_data,
                            "main",
                            2,
                            1 * sizeof(uint32_t),
                            {32 * device->subgroup_size / 8, 1, 1},
                            {device->subgroup_size},
                            1,
                            true,
                            true);
  }
  else {
    v_vk_create_pipeline(device,
                            device->pipeline_quantize_q8_1,
                            "quantize_q8_1",
                            quantize_q8_1_len,
                            quantize_q8_1_data,
                            "main",
                            2,
                            1 * sizeof(uint32_t),
                            {32 * device->subgroup_size / 8, 1, 1},
                            {device->subgroup_size},
                            1);
    v_vk_create_pipeline(device,
                            device->pipeline_quantize_q8_1_x4,
                            "quantize_q8_1_x4",
                            quantize_q8_1_x4_len,
                            quantize_q8_1_x4_data,
                            "main",
                            2,
                            1 * sizeof(uint32_t),
                            {32 * device->subgroup_size / 8, 1, 1},
                            {device->subgroup_size},
                            1);
  }

  for (uint32_t i = 0; i < p021_max_gqa_ratio; ++i) {
    if (device->subgroup_arithmetic && device->subgroup_require_full_support) {
      v_vk_create_pipeline2(device,
                               device->pipeline_mul_mat_vec_p021_f16_f32[i],
                               "mul_mat_vec_p021_f16_f32" + std::to_string(i + 1),
                               mul_mat_vec_p021_f16_f32_subgroup_add_len,
                               mul_mat_vec_p021_f16_f32_subgroup_add_data,
                               "main",
                               3,
                               6 * sizeof(uint32_t),
                               {1, 1, 1},
                               {device->subgroup_size, i + 1},
                               1,
                               true,
                               true);
    }
    else {
      v_vk_create_pipeline2(device,
                               device->pipeline_mul_mat_vec_p021_f16_f32[i],
                               "mul_mat_vec_p021_f16_f32" + std::to_string(i + 1),
                               mul_mat_vec_p021_f16_f32_len,
                               mul_mat_vec_p021_f16_f32_data,
                               "main",
                               3,
                               6 * sizeof(uint32_t),
                               {1, 1, 1},
                               {device->subgroup_size, i + 1},
                               1,
                               true);
    }
  }
  v_vk_create_pipeline(device,
                          device->pipeline_mul_mat_vec_nc_f16_f32,
                          "mul_mat_vec_nc_f16_f32",
                          mul_mat_vec_nc_f16_f32_len,
                          mul_mat_vec_nc_f16_f32_data,
                          "main",
                          3,
                          12 * sizeof(uint32_t),
                          {1, 1, 1},
                          {},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_norm_f32,
                          "norm_f32",
                          norm_f32_len,
                          norm_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_push_constants),
                          {1, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_group_norm_f32,
                          "group_norm_f32",
                          group_norm_f32_len,
                          group_norm_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_push_constants),
                          {1, 1, 1},
                          {},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_rms_norm_f32,
                          "rms_norm_f32",
                          rms_norm_f32_len,
                          rms_norm_f32_data,
                          "main",
                          4,
                          sizeof(vk_op_binary_push_constants),
                          {1, 1, 1},
                          {0, 0},
                          1,
                          true);
  v_vk_create_pipeline(device,
                          device->pipeline_rms_norm_mul_f32,
                          "rms_norm_mul_f32",
                          rms_norm_f32_len,
                          rms_norm_f32_data,
                          "main",
                          4,
                          sizeof(vk_op_binary_push_constants),
                          {1, 1, 1},
                          {0, 1},
                          1,
                          true);
  v_vk_create_pipeline(device,
                          device->pipeline_rms_norm_partials_f32,
                          "rms_norm_partials_f32",
                          rms_norm_partials_f32_len,
                          rms_norm_partials_f32_data,
                          "main",
                          4,
                          sizeof(vk_op_binary_push_constants),
                          {1, 1, 1},
                          {0, 0},
                          1,
                          true);
  v_vk_create_pipeline(device,
                          device->pipeline_rms_norm_mul_partials_f32,
                          "rms_norm_mul_partials_f32",
                          rms_norm_partials_f32_len,
                          rms_norm_partials_f32_data,
                          "main",
                          4,
                          sizeof(vk_op_binary_push_constants),
                          {1, 1, 1},
                          {0, 1},
                          1,
                          true);

  v_vk_create_pipeline(device,
                          device->pipeline_rms_norm_back_f32,
                          "rms_norm_back_f32",
                          rms_norm_back_f32_len,
                          rms_norm_back_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_push_constants),
                          {1, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_l2_norm_f32,
                          "l2_norm_f32",
                          l2_norm_f32_len,
                          l2_norm_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_push_constants),
                          {1, 1, 1},
                          {},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_cpy_f32_f32,
                          "cpy_f32_f32",
                          cpy_f32_f32_len,
                          cpy_f32_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_cpy_f32_f16,
                          "cpy_f32_f16",
                          cpy_f32_f16_len,
                          cpy_f32_f16_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_cpy_f16_f16,
                          "cpy_f16_f16",
                          cpy_f16_f16_len,
                          cpy_f16_f16_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_cpy_f16_f32,
                          "cpy_f16_f32",
                          cpy_f16_f32_len,
                          cpy_f16_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_cpy_f32_bf16,
                          "cpy_f32_bf16",
                          cpy_f32_bf16_len,
                          cpy_f32_bf16_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_cpy_i32_f32,
                          "cpy_i32_f32",
                          cpy_i32_f32_len,
                          cpy_i32_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_cpy_f32_i32,
                          "cpy_f32_i32",
                          cpy_f32_i32_len,
                          cpy_f32_i32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_contig_cpy_f32_f32,
                          "contig_cpy_f32_f32",
                          contig_cpy_f32_f32_len,
                          contig_cpy_f32_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_contig_cpy_f32_f16,
                          "contig_cpy_f32_f16",
                          contig_cpy_f32_f16_len,
                          contig_cpy_f32_f16_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_contig_cpy_f16_f16,
                          "contig_cpy_f16_f16",
                          contig_cpy_f16_f16_len,
                          contig_cpy_f16_f16_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_contig_cpy_f16_f32,
                          "contig_cpy_f16_f32",
                          contig_cpy_f16_f32_len,
                          contig_cpy_f16_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_contig_cpy_f32_bf16,
                          "contig_cpy_f32_bf16",
                          contig_cpy_f32_bf16_len,
                          contig_cpy_f32_bf16_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_contig_cpy_i32_f32,
                          "contig_cpy_i32_f32",
                          contig_cpy_i32_f32_len,
                          contig_cpy_i32_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_contig_cpy_f32_i32,
                          "contig_cpy_f32_i32",
                          contig_cpy_f32_i32_len,
                          contig_cpy_f32_i32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);

  if (device->float_controls_rte_fp16) {
    v_vk_create_pipeline(device,
                            device->pipeline_cpy_f32_quant[v_TYPE_Q4_0],
                            "cpy_f32_q4_0",
                            cpy_f32_q4_0_rte_len,
                            cpy_f32_q4_0_rte_data,
                            "main",
                            2,
                            sizeof(vk_op_unary_push_constants),
                            {32, 1, 1},
                            {},
                            1);
    v_vk_create_pipeline(device,
                            device->pipeline_cpy_f32_quant[v_TYPE_Q4_1],
                            "cpy_f32_q4_1",
                            cpy_f32_q4_1_rte_len,
                            cpy_f32_q4_1_rte_data,
                            "main",
                            2,
                            sizeof(vk_op_unary_push_constants),
                            {32, 1, 1},
                            {},
                            1);
    v_vk_create_pipeline(device,
                            device->pipeline_cpy_f32_quant[v_TYPE_Q5_0],
                            "cpy_f32_q5_0",
                            cpy_f32_q5_0_rte_len,
                            cpy_f32_q5_0_rte_data,
                            "main",
                            2,
                            sizeof(vk_op_unary_push_constants),
                            {32, 1, 1},
                            {},
                            1);
    v_vk_create_pipeline(device,
                            device->pipeline_cpy_f32_quant[v_TYPE_Q5_1],
                            "cpy_f32_q5_1",
                            cpy_f32_q5_1_rte_len,
                            cpy_f32_q5_1_rte_data,
                            "main",
                            2,
                            sizeof(vk_op_unary_push_constants),
                            {32, 1, 1},
                            {},
                            1);
    v_vk_create_pipeline(device,
                            device->pipeline_cpy_f32_quant[v_TYPE_Q8_0],
                            "cpy_f32_q8_0",
                            cpy_f32_q8_0_rte_len,
                            cpy_f32_q8_0_rte_data,
                            "main",
                            2,
                            sizeof(vk_op_unary_push_constants),
                            {32, 1, 1},
                            {},
                            1);
    v_vk_create_pipeline(device,
                            device->pipeline_cpy_f32_quant[v_TYPE_IQ4_NL],
                            "cpy_f32_iq4_nl",
                            cpy_f32_iq4_nl_rte_len,
                            cpy_f32_iq4_nl_rte_data,
                            "main",
                            2,
                            sizeof(vk_op_unary_push_constants),
                            {32, 1, 1},
                            {},
                            1);
  }
  else {
    v_vk_create_pipeline(device,
                            device->pipeline_cpy_f32_quant[v_TYPE_Q4_0],
                            "cpy_f32_q4_0",
                            cpy_f32_q4_0_len,
                            cpy_f32_q4_0_data,
                            "main",
                            2,
                            sizeof(vk_op_unary_push_constants),
                            {32, 1, 1},
                            {},
                            1);
    v_vk_create_pipeline(device,
                            device->pipeline_cpy_f32_quant[v_TYPE_Q4_1],
                            "cpy_f32_q4_1",
                            cpy_f32_q4_1_len,
                            cpy_f32_q4_1_data,
                            "main",
                            2,
                            sizeof(vk_op_unary_push_constants),
                            {32, 1, 1},
                            {},
                            1);
    v_vk_create_pipeline(device,
                            device->pipeline_cpy_f32_quant[v_TYPE_Q5_0],
                            "cpy_f32_q5_0",
                            cpy_f32_q5_0_len,
                            cpy_f32_q5_0_data,
                            "main",
                            2,
                            sizeof(vk_op_unary_push_constants),
                            {32, 1, 1},
                            {},
                            1);
    v_vk_create_pipeline(device,
                            device->pipeline_cpy_f32_quant[v_TYPE_Q5_1],
                            "cpy_f32_q5_1",
                            cpy_f32_q5_1_len,
                            cpy_f32_q5_1_data,
                            "main",
                            2,
                            sizeof(vk_op_unary_push_constants),
                            {32, 1, 1},
                            {},
                            1);
    v_vk_create_pipeline(device,
                            device->pipeline_cpy_f32_quant[v_TYPE_Q8_0],
                            "cpy_f32_q8_0",
                            cpy_f32_q8_0_len,
                            cpy_f32_q8_0_data,
                            "main",
                            2,
                            sizeof(vk_op_unary_push_constants),
                            {32, 1, 1},
                            {},
                            1);
    v_vk_create_pipeline(device,
                            device->pipeline_cpy_f32_quant[v_TYPE_IQ4_NL],
                            "cpy_f32_iq4_nl",
                            cpy_f32_iq4_nl_len,
                            cpy_f32_iq4_nl_data,
                            "main",
                            2,
                            sizeof(vk_op_unary_push_constants),
                            {32, 1, 1},
                            {},
                            1);
  }

  #define SET_ROWS(itype, rte) \
        v_vk_create_pipeline(device, device->pipeline_set_rows ## itype [v_TYPE_F32],  "set_rows_f32" #itype,  set_rows_f32 ## itype ## rte ## _len,  set_rows_f32 ## itype ## rte ## _data,  "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        v_vk_create_pipeline(device, device->pipeline_set_rows ## itype [v_TYPE_F16],  "set_rows_f16" #itype,  set_rows_f16 ## itype ## rte ## _len,  set_rows_f16 ## itype ## rte ## _data,  "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        v_vk_create_pipeline(device, device->pipeline_set_rows ## itype [v_TYPE_BF16], "set_rows_bf16" #itype, set_rows_bf16 ## itype ## rte ## _len, set_rows_bf16 ## itype ## rte ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        v_vk_create_pipeline(device, device->pipeline_set_rows ## itype [v_TYPE_Q4_0], "set_rows_q4_0" #itype, set_rows_q4_0 ## itype ## rte ## _len, set_rows_q4_0 ## itype ## rte ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        v_vk_create_pipeline(device, device->pipeline_set_rows ## itype [v_TYPE_Q4_1], "set_rows_q4_1" #itype, set_rows_q4_1 ## itype ## rte ## _len, set_rows_q4_1 ## itype ## rte ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        v_vk_create_pipeline(device, device->pipeline_set_rows ## itype [v_TYPE_Q5_0], "set_rows_q5_0" #itype, set_rows_q5_0 ## itype ## rte ## _len, set_rows_q5_0 ## itype ## rte ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        v_vk_create_pipeline(device, device->pipeline_set_rows ## itype [v_TYPE_Q5_1], "set_rows_q5_1" #itype, set_rows_q5_1 ## itype ## rte ## _len, set_rows_q5_1 ## itype ## rte ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        v_vk_create_pipeline(device, device->pipeline_set_rows ## itype [v_TYPE_Q8_0], "set_rows_q8_0" #itype, set_rows_q8_0 ## itype ## rte ## _len, set_rows_q8_0 ## itype ## rte ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        v_vk_create_pipeline(device, device->pipeline_set_rows ## itype [v_TYPE_IQ4_NL], "set_rows_iq4_nl" #itype, set_rows_iq4_nl ## itype ## rte ## _len, set_rows_iq4_nl ## itype ## rte ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true);

  if (device->float_controls_rte_fp16) {
    SET_ROWS(_i32, _rte)
    SET_ROWS(_i64, _rte)
  }
  else {
    SET_ROWS(_i32,)
    SET_ROWS(_i64,)
  }
  #undef SET_ROWS


  v_vk_create_pipeline(device,
                          device->pipeline_cpy_quant_f32[v_TYPE_Q4_0],
                          "cpy_q4_0_f32",
                          cpy_q4_0_f32_len,
                          cpy_q4_0_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {
                            (uint32_t)blockSize(v_TYPE_Q4_0), 1, 1
                          },
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_cpy_quant_f32[v_TYPE_Q4_1],
                          "cpy_q4_1_f32",
                          cpy_q4_1_f32_len,
                          cpy_q4_1_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {
                            (uint32_t)blockSize(v_TYPE_Q4_1), 1, 1
                          },
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_cpy_quant_f32[v_TYPE_Q5_0],
                          "cpy_q5_0_f32",
                          cpy_q5_0_f32_len,
                          cpy_q5_0_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {
                            (uint32_t)blockSize(v_TYPE_Q5_0), 1, 1
                          },
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_cpy_quant_f32[v_TYPE_Q5_1],
                          "cpy_q5_1_f32",
                          cpy_q5_1_f32_len,
                          cpy_q5_1_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {
                            (uint32_t)blockSize(v_TYPE_Q5_1), 1, 1
                          },
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_cpy_quant_f32[v_TYPE_Q8_0],
                          "cpy_q8_0_f32",
                          cpy_q8_0_f32_len,
                          cpy_q8_0_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {
                            (uint32_t)blockSize(v_TYPE_Q8_0), 1, 1
                          },
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_cpy_quant_f32[v_TYPE_IQ4_NL],
                          "cpy_iq4_nl_f32",
                          cpy_iq4_nl_f32_len,
                          cpy_iq4_nl_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {(uint32_t)blockSize(v_TYPE_IQ4_NL), 1, 1},
                          {},
                          1);

  auto get_suffix = [](bool src0_f16, bool src1_f16, bool dst_f16) {
    std::string s;
    s += std::string(src0_f16 ? "_f16" : "_f32");
    s += std::string(src1_f16 ? "_f16" : "_f32");
    s += std::string(dst_f16 ? "_f16" : "_f32");
    return s;
  };

  bool rte = device->float_controls_rte_fp16;
  #define CREATE_BINARY(name, namemod, spec, bindings) \
    for (int s0 : {0,1}) for (int s1 : {0,1}) for (int d : {0,1}) \
        v_vk_create_pipeline2(device, device->pipeline_ ## name ## namemod[s0][s1][d], \
                                #name + get_suffix(s0, s1, d) + #namemod, name ## _len[s0][s1][d][rte], name ## _data[s0][s1][d][rte], \
                                "main", (bindings), sizeof(vk_op_binary_push_constants), {512, 1, 1}, spec, 1);

  CREATE_BINARY(add, , {0}, 4)
  CREATE_BINARY(add, _norepeat, {1}, 4)
  CREATE_BINARY(sub, , {0}, 3)
  CREATE_BINARY(sub, _norepeat, {1}, 3)
  CREATE_BINARY(mul, , {0}, 3)
  CREATE_BINARY(mul, _norepeat, {1}, 3)
  CREATE_BINARY(div, , {0}, 3)
  CREATE_BINARY(div, _norepeat, {1}, 3)
  CREATE_BINARY(add_rms, , {0}, 4)
  CREATE_BINARY(add_rms, _norepeat, {1}, 4)
  #undef CREATE_BINARY

  if (device->multi_add) {
    for (uint32_t i = 0; i < MAX_FUSED_ADDS; ++i) {
      v_vk_create_pipeline2(device,
                               device->pipeline_multi_add[i],
                               "multi_add_f32_" + std::to_string(i + 1),
                               multi_add_f32_len,
                               multi_add_f32_data,
                               "main",
                               MAX_PARAMETER_COUNT,
                               sizeof(vk_op_multi_add_push_constants),
                               {512, 1, 1},
                               {i + 2},
                               1);
      v_vk_create_pipeline2(device,
                               device->pipeline_multi_add_rms[i],
                               "multi_add_rms_f32_" + std::to_string(i + 1),
                               multi_add_rms_f32_len,
                               multi_add_rms_f32_data,
                               "main",
                               MAX_PARAMETER_COUNT,
                               sizeof(vk_op_multi_add_push_constants),
                               {512, 1, 1},
                               {i + 2},
                               1);
    }
  }

  v_vk_create_pipeline(device,
                          device->pipeline_add_id_f32,
                          "add_id_f32",
                          add_id_f32_len,
                          add_id_f32_data,
                          "main",
                          4,
                          sizeof(vk_op_add_id_push_constants),
                          {1, 1, 1},
                          {},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_acc_f32,
                          "acc_f32",
                          acc_f32_len,
                          acc_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {512, 1, 1},
                          {},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_concat_f32,
                          "concat_f32",
                          concat_f32_len,
                          concat_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_concat_f16,
                          "concat_f16",
                          concat_f16_len,
                          concat_f16_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_concat_i32,
                          "concat_i32",
                          concat_i32_len,
                          concat_i32_data,
                          "main",
                          3,
                          sizeof(vk_op_binary_push_constants),
                          {512, 1, 1},
                          {},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_upscale_nearest_f32,
                          "upscale_f32",
                          upscale_f32_len,
                          upscale_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_upscale_push_constants),
                          {512, 1, 1},
                          {v_SCALE_MODE_NEAREST},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_upscale_bilinear_f32,
                          "upscale_f32",
                          upscale_f32_len,
                          upscale_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_upscale_push_constants),
                          {512, 1, 1},
                          {v_SCALE_MODE_BILINEAR},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_scale_f32,
                          "scale_f32",
                          scale_f32_len,
                          scale_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_sqr_f32,
                          "sqr_f32",
                          sqr_f32_len,
                          sqr_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_sqrt_f32,
                          "sqrt_f32",
                          sqrt_f32_len,
                          sqrt_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_sin_f32,
                          "sin_f32",
                          sin_f32_len,
                          sin_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_rand_f32,
                          "rand_f32",
                          rand_f32_len,
                          rand_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_cos_f32,
                          "cos_f32",
                          cos_f32_len,
                          cos_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_clamp_f32,
                          "clamp_f32",
                          clamp_f32_len,
                          clamp_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_pad_f32,
                          "pad_f32",
                          pad_f32_len,
                          pad_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_pad_push_constants),
                          {512, 1, 1},
                          {},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_roll_f32,
                          "roll_f32",
                          roll_f32_len,
                          roll_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_repeat_f32,
                          "repeat_f32",
                          repeat_f32_len,
                          repeat_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_repeat_back_f32,
                          "repeat_back_f32",
                          repeat_back_f32_len,
                          repeat_back_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_unary_push_constants),
                          {512, 1, 1},
                          {},
                          1);

  #define CREATE_UNARY(name)  \
    v_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## _f32_data, "main", 2, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);  \
    v_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## _f16_data, "main", 2, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

  CREATE_UNARY(gelu)
  CREATE_UNARY(gelu_erf)

  CREATE_UNARY(gelu_quick)
  CREATE_UNARY(silu)
  CREATE_UNARY(relu)
  CREATE_UNARY(tanh)
  CREATE_UNARY(sigmoid)
  CREATE_UNARY(hardsigmoid)
  CREATE_UNARY(hardswish)
  CREATE_UNARY(log)
  #undef CREATE_UNARY

  #define CREATE_UNARY_RTE(name)  \
    if (device->float_controls_rte_fp16) {  \
        v_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32_rte", name ## _f32_rte_len, name ## _f32_rte_data, "main", 2, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);   \
        v_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16_rte", name ## _f16_rte_len, name ## _f16_rte_data, "main", 2, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);   \
    } else {    \
        v_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## _f32_data, "main", 2, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);   \
        v_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## _f16_data, "main", 2, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);   \
    }
  CREATE_UNARY_RTE(exp)
  #undef CREATE_UNARY_RTE

  #define CREATE_GLU(name)  \
    if (device->float_controls_rte_fp16) {  \
        v_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32_rte", name ## _f32_rte_len, name ## _f32_rte_data, "main", 3, sizeof(vk_op_glu_push_constants), {512, 1, 1}, {}, 1, true);   \
        v_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16_rte", name ## _f16_rte_len, name ## _f16_rte_data, "main", 3, sizeof(vk_op_glu_push_constants), {512, 1, 1}, {}, 1, true);   \
    } else {    \
        v_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## _f32_data, "main", 3, sizeof(vk_op_glu_push_constants), {512, 1, 1}, {}, 1, true);   \
        v_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## _f16_data, "main", 3, sizeof(vk_op_glu_push_constants), {512, 1, 1}, {}, 1, true);   \
    }

  CREATE_GLU(geglu)
  CREATE_GLU(reglu)
  CREATE_GLU(swiglu)
  CREATE_GLU(swiglu_oai)
  CREATE_GLU(geglu_erf)
  CREATE_GLU(geglu_quick)
  #undef CREATE_GLU

  v_vk_create_pipeline(device,
                          device->pipeline_leaky_relu_f32,
                          "leaky_relu_f32",
                          leaky_relu_f32_len,
                          leaky_relu_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_silu_back_f32,
                          "silu_back_f32",
                          silu_back_f32_len,
                          silu_back_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_push_constants),
                          {512, 1, 1},
                          {},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_diag_mask_inf_f32,
                          "diag_mask_inf_f32",
                          diag_mask_inf_f32_len,
                          diag_mask_inf_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_diag_mask_push_constants),
                          {1, 512, 1},
                          {},
                          1,
                          true);

  v_vk_create_pipeline(device,
                          device->pipeline_soft_max_f32,
                          "soft_max_f32",
                          soft_max_f32_len,
                          soft_max_f32_data,
                          "main",
                          4,
                          sizeof(vk_op_soft_max_push_constants),
                          {1, 1, 1},
                          {device->subgroup_size},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_soft_max_f32_wg512,
                          "soft_max_f32_wg512",
                          soft_max_f32_len,
                          soft_max_f32_data,
                          "main",
                          4,
                          sizeof(vk_op_soft_max_push_constants),
                          {1, 1, 1},
                          {512},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_soft_max_f32_f16,
                          "soft_max_f32_f16",
                          soft_max_f32_f16_len,
                          soft_max_f32_f16_data,
                          "main",
                          4,
                          sizeof(vk_op_soft_max_push_constants),
                          {1, 1, 1},
                          {device->subgroup_size},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_soft_max_f32_f16_wg512,
                          "soft_max_f32_f16_wg512",
                          soft_max_f32_f16_len,
                          soft_max_f32_f16_data,
                          "main",
                          4,
                          sizeof(vk_op_soft_max_push_constants),
                          {1, 1, 1},
                          {512},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_soft_max_back_f32,
                          "soft_max_back_f32",
                          soft_max_back_f32_len,
                          soft_max_back_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_push_constants),
                          {1, 1, 1},
                          {device->subgroup_size},
                          1,
                          true);

  v_vk_create_pipeline(device,
                          device->pipeline_rope_norm_f32,
                          "rope_norm_f32",
                          rope_norm_f32_len,
                          rope_norm_f32_data,
                          "main",
                          4,
                          sizeof(vk_op_rope_push_constants),
                          {1, 512, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_rope_neox_f32,
                          "rope_neox_f32",
                          rope_neox_f32_len,
                          rope_neox_f32_data,
                          "main",
                          4,
                          sizeof(vk_op_rope_push_constants),
                          {1, 512, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_rope_multi_f32,
                          "rope_multi_f32",
                          rope_multi_f32_len,
                          rope_multi_f32_data,
                          "main",
                          4,
                          sizeof(vk_op_rope_push_constants),
                          {1, 512, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_rope_vision_f32,
                          "rope_vision_f32",
                          rope_vision_f32_len,
                          rope_vision_f32_data,
                          "main",
                          4,
                          sizeof(vk_op_rope_push_constants),
                          {1, 512, 1},
                          {},
                          1);

  if (device->float_controls_rte_fp16) {
    v_vk_create_pipeline(device,
                            device->pipeline_rope_norm_f16,
                            "rope_norm_f16",
                            rope_norm_f16_rte_len,
                            rope_norm_f16_rte_data,
                            "main",
                            4,
                            sizeof(vk_op_rope_push_constants),
                            {1, 512, 1},
                            {},
                            1);
    v_vk_create_pipeline(device,
                            device->pipeline_rope_neox_f16,
                            "rope_neox_f16",
                            rope_neox_f16_rte_len,
                            rope_neox_f16_rte_data,
                            "main",
                            4,
                            sizeof(vk_op_rope_push_constants),
                            {1, 512, 1},
                            {},
                            1);
    v_vk_create_pipeline(device,
                            device->pipeline_rope_multi_f16,
                            "rope_multi_f16",
                            rope_multi_f16_rte_len,
                            rope_multi_f16_rte_data,
                            "main",
                            4,
                            sizeof(vk_op_rope_push_constants),
                            {1, 512, 1},
                            {},
                            1);
    v_vk_create_pipeline(device,
                            device->pipeline_rope_vision_f16,
                            "rope_vision_f16",
                            rope_vision_f16_rte_len,
                            rope_vision_f16_rte_data,
                            "main",
                            4,
                            sizeof(vk_op_rope_push_constants),
                            {1, 512, 1},
                            {},
                            1);
  }
  else {
    v_vk_create_pipeline(device,
                            device->pipeline_rope_norm_f16,
                            "rope_norm_f16",
                            rope_norm_f16_len,
                            rope_norm_f16_data,
                            "main",
                            4,
                            sizeof(vk_op_rope_push_constants),
                            {1, 512, 1},
                            {},
                            1);
    v_vk_create_pipeline(device,
                            device->pipeline_rope_neox_f16,
                            "rope_neox_f16",
                            rope_neox_f16_len,
                            rope_neox_f16_data,
                            "main",
                            4,
                            sizeof(vk_op_rope_push_constants),
                            {1, 512, 1},
                            {},
                            1);
    v_vk_create_pipeline(device,
                            device->pipeline_rope_multi_f16,
                            "rope_multi_f16",
                            rope_multi_f16_len,
                            rope_multi_f16_data,
                            "main",
                            4,
                            sizeof(vk_op_rope_push_constants),
                            {1, 512, 1},
                            {},
                            1);
    v_vk_create_pipeline(device,
                            device->pipeline_rope_vision_f16,
                            "rope_vision_f16",
                            rope_vision_f16_len,
                            rope_vision_f16_data,
                            "main",
                            4,
                            sizeof(vk_op_rope_push_constants),
                            {1, 512, 1},
                            {},
                            1);
  }

  for (uint32_t i = 0; i < num_argsort_pipelines; ++i) {
    v_vk_create_pipeline2(device,
                             device->pipeline_argsort_f32[i],
                             "argsort_f32_" + std::to_string(i),
                             argsort_f32_len,
                             argsort_f32_data,
                             "main",
                             2,
                             sizeof(vk_op_argsort_push_constants),
                             {1u << i, 1, 1},
                             {1u << i, i},
                             1,
                             true);
  }

  v_vk_create_pipeline(device,
                          device->pipeline_argmax_f32,
                          "argmax_f32",
                          argmax_f32_len,
                          argmax_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_push_constants),
                          {1, 1, 1},
                          {device->subgroup_size},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_sum_rows_f32,
                          "sum_rows_f32",
                          sum_rows_f32_len,
                          sum_rows_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_sum_rows_push_constants),
                          {1, 1, 1},
                          {device->subgroup_size},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_count_equal_i32,
                          "count_equal_i32",
                          count_equal_i32_len,
                          count_equal_i32_data,
                          "main",
                          3,
                          sizeof(vk_op_push_constants),
                          {512, 1, 1},
                          {device->subgroup_size},
                          1);

  #define IM2COL(bda) \
    v_vk_create_pipeline(device, device->pipeline_im2col_f32, "im2col_f32", im2col_f32 ## bda ## _len, im2col_f32 ## bda ## _data, "main", 2, sizeof(vk_op_im2col_push_constants), {512, 1, 1}, { device->subgroup_size }, 1, true);   \
    v_vk_create_pipeline(device, device->pipeline_im2col_3d_f32, "im2col_3d_f32", im2col_3d_f32 ## bda ## _len, im2col_3d_f32 ## bda ## _data, "main", 2, sizeof(vk_op_im2col_3d_push_constants), {512, 1, 1}, { 512 }, 1, true);      \
    if (device->float_controls_rte_fp16) {  \
        v_vk_create_pipeline(device, device->pipeline_im2col_f32_f16, "im2col_f32_f16", im2col_f32_f16_rte ## bda ## _len, im2col_f32_f16_rte ## bda ## _data, "main", 2, sizeof(vk_op_im2col_push_constants), {512, 1, 1}, { device->subgroup_size }, 1, true);   \
        v_vk_create_pipeline(device, device->pipeline_im2col_3d_f32_f16, "im2col_3d_f32_f16", im2col_3d_f32_f16_rte ## bda ## _len, im2col_3d_f32_f16_rte ## bda ## _data, "main", 2, sizeof(vk_op_im2col_3d_push_constants), {512, 1, 1}, { 512 }, 1, true);      \
    } else {    \
        v_vk_create_pipeline(device, device->pipeline_im2col_f32_f16, "im2col_f32_f16", im2col_f32_f16 ## bda ## _len, im2col_f32_f16 ## bda ## _data, "main", 2, sizeof(vk_op_im2col_push_constants), {512, 1, 1}, { device->subgroup_size }, 1, true);   \
        v_vk_create_pipeline(device, device->pipeline_im2col_3d_f32_f16, "im2col_3d_f32_f16", im2col_3d_f32_f16 ## bda ## _len, im2col_3d_f32_f16 ## bda ## _data, "main", 2, sizeof(vk_op_im2col_3d_push_constants), {512, 1, 1}, { 512 }, 1, true);      \
    }
  if (device->shader_int64 && device->buffer_device_address) { IM2COL(_bda) }
  else { IM2COL() }

  v_vk_create_pipeline(device,
                          device->pipeline_timestep_embedding_f32,
                          "timestep_embedding_f32",
                          timestep_embedding_f32_len,
                          timestep_embedding_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_timestep_embedding_push_constants),
                          {256, 1, 1},
                          {},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_conv_transpose_1d_f32,
                          "conv_transpose_1d_f32",
                          conv_transpose_1d_f32_len,
                          conv_transpose_1d_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_conv_transpose_1d_push_constants),
                          {1, 1, 1},
                          {},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_pool2d_f32,
                          "pool2d_f32",
                          pool2d_f32_len,
                          pool2d_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_pool2d_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_pool2d_back_f32,
                          "pool2d_back_f32",
                          pool2d_back_f32_len,
                          pool2d_back_f32_data,
                          "main",
                          2,
                          sizeof(vk_op_pool2d_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_rwkv_wkv6_f32,
                          "rwkv_wkv6_f32",
                          rwkv_wkv6_f32_len,
                          rwkv_wkv6_f32_data,
                          "main",
                          7,
                          sizeof(vk_op_rwkv_wkv6_push_constants),
                          {1, 1, 1},
                          {device->subgroup_size},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_rwkv_wkv7_f32,
                          "rwkv_wkv7_f32",
                          rwkv_wkv7_f32_len,
                          rwkv_wkv7_f32_data,
                          "main",
                          8,
                          sizeof(vk_op_rwkv_wkv7_push_constants),
                          {1, 1, 1},
                          {device->subgroup_size},
                          1);

  if (device->subgroup_arithmetic && device->subgroup_require_full_support) {
    v_vk_create_pipeline(device,
                            device->pipeline_ssm_scan_f32_d128,
                            "ssm_scan_128_f32",
                            ssm_scan_subgroup_f32_len,
                            ssm_scan_subgroup_f32_data,
                            "main",
                            8,
                            sizeof(vk_op_ssm_scan_push_constants),
                            {1, 1, 1},
                            {128, device->subgroup_size, 16},
                            1,
                            true,
                            true);
    v_vk_create_pipeline(device,
                            device->pipeline_ssm_scan_f32_d256,
                            "ssm_scan_256_f32",
                            ssm_scan_subgroup_f32_len,
                            ssm_scan_subgroup_f32_data,
                            "main",
                            8,
                            sizeof(vk_op_ssm_scan_push_constants),
                            {1, 1, 1},
                            {256, device->subgroup_size, 16},
                            1,
                            true,
                            true);
  }
  else {
    v_vk_create_pipeline(device,
                            device->pipeline_ssm_scan_f32_d128,
                            "ssm_scan_128_f32",
                            ssm_scan_f32_len,
                            ssm_scan_f32_data,
                            "main",
                            8,
                            sizeof(vk_op_ssm_scan_push_constants),
                            {1, 1, 1},
                            {128, device->subgroup_size, 16},
                            1,
                            true,
                            true);
    v_vk_create_pipeline(device,
                            device->pipeline_ssm_scan_f32_d256,
                            "ssm_scan_256_f32",
                            ssm_scan_f32_len,
                            ssm_scan_f32_data,
                            "main",
                            8,
                            sizeof(vk_op_ssm_scan_push_constants),
                            {1, 1, 1},
                            {256, device->subgroup_size, 16},
                            1,
                            true,
                            true);
  }

  v_vk_create_pipeline(device,
                          device->pipeline_ssm_conv_f32,
                          "ssm_conv_f32",
                          ssm_conv_f32_len,
                          ssm_conv_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_ssm_conv_push_constants),
                          {32, 1, 1},
                          {32},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_opt_step_adamw_f32,
                          "opt_step_adamw_f32",
                          opt_step_adamw_f32_len,
                          opt_step_adamw_f32_data,
                          "main",
                          5,
                          sizeof(vk_op_push_constants),
                          {512, 1, 1},
                          {},
                          1);

  v_vk_create_pipeline(device,
                          device->pipeline_opt_step_sgd_f32,
                          "opt_step_sgd_f32",
                          opt_step_sgd_f32_len,
                          opt_step_sgd_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_push_constants),
                          {512, 1, 1},
                          {},
                          1);

  // conv2d, conv_transpose_2d
  for (uint32_t s = 0; s < CONV_SHAPE_COUNT; ++s) {
    uint32_t conv2d_WG_SIZE   = 256;
    uint32_t conv2d_BS_K      = 128;
    uint32_t conv2d_BS_CRS    = 16;
    uint32_t use_collectives  = 0; // Enables subgroup ops for preventing the re-calculation of indices.
    uint32_t conv2d_BS_NPQ    = 128;
    uint32_t conv2d_TS_K      = 8;
    uint32_t conv2d_SHMEM_PAD = 4;
    bool conv2d_UNROLL        = true;

    #if defined(V_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    if (device->coopmat2) {
      conv2d_SHMEM_PAD = 8; // 8 float16_t
    }
    #endif

    if (device->vendor_id == VK_VENDOR_ID_INTEL) {
      conv2d_SHMEM_PAD = 0;
      conv2d_UNROLL    = false;
    }
    else if (device->vendor_id == VK_VENDOR_ID_AMD) {
      conv2d_SHMEM_PAD = device->architecture == vk_device_architecture::AMD_GCN ? 1 : 4;
    }

    switch (s) {
    default:
    case CONV_SHAPE_128x128:
      conv2d_BS_K = 128;
      conv2d_BS_NPQ = 128;
      conv2d_BS_CRS = 16;
      if (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != vk_device_architecture::AMD_GCN) {
        conv2d_UNROLL = false;
      }
      break;
    case CONV_SHAPE_64x32:
      conv2d_BS_K = 64;
      conv2d_BS_NPQ = 32;
      conv2d_BS_CRS = 32;
      conv2d_TS_K   = 4;
      break;
    case CONV_SHAPE_32x256:
      conv2d_BS_K = 32;
      conv2d_BS_NPQ = 256;
      conv2d_BS_CRS = 16;
      break;
    }

    // Use collectives on pre-Turing NVIDIA GPUs and GCN AMD cards, which had slower integer math.
    bool allow_collectives_nv = device->vendor_id != VK_VENDOR_ID_NVIDIA ||
      device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
    bool allow_collectives_amd = device->vendor_id != VK_VENDOR_ID_AMD ||
      device->architecture == vk_device_architecture::AMD_GCN;

    if (device->subgroup_shuffle &&
      device->vendor_id != VK_VENDOR_ID_INTEL && // Do not enable collectives on Intel, see PR 14316.
      allow_collectives_nv &&
      allow_collectives_amd) {
      use_collectives = 1;
      conv2d_BS_CRS   = std::min(
        device->subgroup_size,
        conv2d_BS_CRS); // CRS block size should be capped at subgroup size for correctness when shuffle is used.
    }

    uint32_t conv2d_shmem_req =
      (conv2d_BS_K * (conv2d_BS_CRS + conv2d_SHMEM_PAD) + conv2d_BS_CRS * (conv2d_BS_NPQ + conv2d_SHMEM_PAD)) * sizeof(
        float);
    if (device->properties.limits.maxComputeSharedMemorySize < conv2d_shmem_req) {
      conv2d_BS_CRS = 8;
      if (use_collectives) { conv2d_BS_CRS = std::min(device->subgroup_size, conv2d_BS_CRS); }
    }

    std::array<uint32_t, 3> wg_denoms    = {conv2d_BS_K, conv2d_BS_NPQ, 1};
    std::vector<uint32_t> spec_constants = {
      conv2d_WG_SIZE, conv2d_BS_K, conv2d_BS_CRS, conv2d_BS_NPQ, conv2d_TS_K, use_collectives, conv2d_SHMEM_PAD
    };

    #define CREATE_CONV(name, type_suffix, spv_suffix) \
        v_vk_create_pipeline( \
            device, device->pipeline_##name##type_suffix[s], #name #type_suffix, \
            name##type_suffix##spv_suffix##_len, name##type_suffix##spv_suffix##_data, "main", 3, \
            sizeof(vk_op_##name##_push_constants), wg_denoms, spec_constants, 1, true, use_collectives);
    #define CREATE_CONVS(spv_suffix) \
        CREATE_CONV(conv2d, _f32, spv_suffix) \
        CREATE_CONV(conv2d, _f16_f32, spv_suffix) \
        if (device->properties.limits.maxPushConstantsSize >= sizeof(vk_op_conv_transpose_2d_push_constants)) { \
            CREATE_CONV(conv_transpose_2d, _f32, spv_suffix) \
            CREATE_CONV(conv_transpose_2d, _f16_f32, spv_suffix) \
        }
    #if defined(V_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    if (device->coopmat2) { CREATE_CONVS(_cm2) }
    else
    #endif
      if (conv2d_UNROLL) { CREATE_CONVS(_unroll) }
      else { CREATE_CONVS() }
    #undef CREATE_CONV
    #undef CREATE_CONVS
  }

  v_vk_create_pipeline(device,
                          device->pipeline_conv2d_dw_whcn_f32,
                          "conv2d_dw_whcn_f32",
                          conv2d_dw_whcn_f32_len,
                          conv2d_dw_whcn_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_conv2d_dw_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_conv2d_dw_cwhn_f32,
                          "conv2d_dw_cwhn_f32",
                          conv2d_dw_cwhn_f32_len,
                          conv2d_dw_cwhn_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_conv2d_dw_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_conv2d_dw_whcn_f16_f32,
                          "conv2d_dw_whcn_f16_f32",
                          conv2d_dw_whcn_f16_f32_len,
                          conv2d_dw_whcn_f16_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_conv2d_dw_push_constants),
                          {512, 1, 1},
                          {},
                          1);
  v_vk_create_pipeline(device,
                          device->pipeline_conv2d_dw_cwhn_f16_f32,
                          "conv2d_dw_cwhn_f16_f32",
                          conv2d_dw_cwhn_f16_f32_len,
                          conv2d_dw_cwhn_f16_f32_data,
                          "main",
                          3,
                          sizeof(vk_op_conv2d_dw_push_constants),
                          {512, 1, 1},
                          {},
                          1);

  for (uint32_t i = 0; i < num_topk_moe_pipelines; ++i) {
    v_vk_create_pipeline2(device,
                             device->pipeline_topk_moe[i][0],
                             "topk_moe_f32_" + std::to_string(i),
                             topk_moe_f32_len,
                             topk_moe_f32_data,
                             "main",
                             3,
                             sizeof(vk_op_topk_moe_push_constants),
                             {1, 1, 1},
                             {device->subgroup_size, 1u << i, 0},
                             1,
                             true,
                             true);
    v_vk_create_pipeline2(device,
                             device->pipeline_topk_moe[i][1],
                             "topk_moe_f32_" + std::to_string(i),
                             topk_moe_f32_len,
                             topk_moe_f32_data,
                             "main",
                             3,
                             sizeof(vk_op_topk_moe_push_constants),
                             {1, 1, 1},
                             {device->subgroup_size, 1u << i, 1},
                             1,
                             true,
                             true);
  }

  for (auto& c : compiles) { c.wait(); }
  device->need_compiles = false;
}
