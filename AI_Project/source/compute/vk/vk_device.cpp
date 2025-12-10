#include <array>

#include "vk_device.hpp"

#include "v-vulkan-kernels.hpp"
#include "vk_buffer.h"
#include "vk_context.h"
#include "vk_constant.h"
#include "vk_util.hpp"
#include "vk_pipeline.hpp"

// variables to track number of compiles in progress
// The FA coopmat1 shader assumes 16x16x16 matrix multiply support.
// 128 threads split into four subgroups, each subgroup does 1/4
// of the Bc dimension.




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

vk_buffer v_vk_create_buffer_device(vk_device& device, size_t size) {
  vk_buffer buf;
  try {
    if (device->prefer_host_memory) {
      buf = vk_create_buffer(device,
                             size,
                             {
                               vk::MemoryPropertyFlagBits::eHostVisible |
                               vk::MemoryPropertyFlagBits::eHostCoherent,
                               vk::MemoryPropertyFlagBits::eDeviceLocal
                             });
    }
    else if (device->uma) {
      // Fall back to host memory type
      buf = vk_create_buffer(device,
                             size,
                             {
                               vk::MemoryPropertyFlagBits::eDeviceLocal,
                               vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
                             });
    }
    else if (device->disable_host_visible_vidmem) {
      if (device->allow_sysmem_fallback) {
        buf = vk_create_buffer(device,
                               size,
                               {
                                 vk::MemoryPropertyFlagBits::eDeviceLocal,
                                 vk::MemoryPropertyFlagBits::eHostVisible |
                                 vk::MemoryPropertyFlagBits::eHostCoherent
                               });
      }
      else { buf = vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal}); }
    }
    else {
      // use rebar if available, otherwise fallback to device only visible memory
      if (device->allow_sysmem_fallback) {
        buf = vk_create_buffer(device,
                               size,
                               {
                                 vk::MemoryPropertyFlagBits::eDeviceLocal |
                                 vk::MemoryPropertyFlagBits::eHostVisible |
                                 vk::MemoryPropertyFlagBits::eHostCoherent,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal,
                                 vk::MemoryPropertyFlagBits::eHostVisible |
                                 vk::MemoryPropertyFlagBits::eHostCoherent
                               });
      }
      else {
        buf = vk_create_buffer(device,
                               size,
                               {
                                 vk::MemoryPropertyFlagBits::eDeviceLocal |
                                 vk::MemoryPropertyFlagBits::eHostVisible |
                                 vk::MemoryPropertyFlagBits::eHostCoherent,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal
                               });
      }
    }
  }
  catch (const vk::SystemError& e) {
    std::cerr << "v_vulkan: Device memory allocation of size " << size << " failed." << std::endl;
    std::cerr << "v_vulkan: " << e.what() << std::endl;
    throw e;
  }

  return buf;
}

int v_vk_get_device_count() {
  vk_instance_init();
  return vk_instance.device_indices.size();
}
void v_vk_get_device_description(int device, char* description, size_t description_size) {
  vk_instance_init();

  std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

  vk::PhysicalDeviceProperties props;
  devices[device].getProperties(&props);

  snprintf(description, description_size, "%s", props.deviceName.data());
}
