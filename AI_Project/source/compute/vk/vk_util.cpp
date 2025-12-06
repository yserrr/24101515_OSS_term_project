#include "vk_util.h"

uint32_t get_subgroup_size(const std::string& pipeline_name, const vk_device_architecture& arch) {
  for (const auto& config : gpu_pipeline_configs) {
    if (config.arch == arch) {
      auto pipIt = config.pipelines.find(pipeline_name);
      if (pipIt != config.pipelines.end()) {
        return pipIt->second;
      }
      std::vector<std::pair<std::string, uint32_t>> sorted_pipelines(config.pipelines.begin(), config.pipelines.end());
      std::sort(sorted_pipelines.begin(), sorted_pipelines.end(),
                [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
      for (const auto& entry : sorted_pipelines) {
        if (pipeline_name.find(entry.first) != std::string::npos) {
          return entry.second;
        }
      }
      return config.default_subgroup_size;
    }
  }
  return 0; // If no matching configuration is found
}

uint64_t vk_tensor_offset(const v_tensor* tensor) {
  if (tensor->view_src) return static_cast<uint8_t*>(tensor->view_src->data) - static_cast<uint8_t*>(vk_ptr_base);
  return static_cast<uint8_t*>(tensor->data) - static_cast<uint8_t*>(vk_ptr_base);
}

uint32_t get_misalign_bytes(vk_backend_ctx* ctx, const v_tensor* t) {
  return ((vk_tensor_offset(t) + t->view_offs) & (ctx->device->properties.limits.minStorageBufferOffsetAlignment - 1));;
}

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
void init_fastdiv_values(uint32_t d, uint32_t& mp, uint32_t& L) {
  // compute L = ceil(log2(d));
  L = 0;
  while (L < 32 && (uint32_t{1} << L) < d) { L++; }

  mp = (uint32_t)((uint64_t{1} << 32) * ((uint64_t{1} << L) - d) / d + 1);
}

bool v_vk_op_supports_incontiguous(v_operation op) {
  switch (op) {
    case v_OP_CPY:
    case v_OP_GET_ROWS:
    case v_OP_ADD:
    case v_OP_SUB:
    case v_OP_MUL:
    case v_OP_DIV:
    case v_OP_ADD_ID:
    case v_OP_CONCAT:
    case v_OP_UPSCALE:
    case v_OP_SQR:
    case v_OP_SQRT:
    case v_OP_SIN:
    case v_OP_COS:
    case v_OP_CLAMP:
    case v_OP_PAD:
    case v_OP_REPEAT:
    case v_OP_REPEAT_BACK:
    case V_OP_ROPE:
    case v_OP_RMS_NORM:
    case v_OP_CONV_2D_DW:
    case V_OP_IM2COL:
    case v_OP_IM2COL_3D:
    case v_OP_SET_ROWS:
    case v_OP_SUM:
    case v_OP_SUM_ROWS:
    case V_OP_MEAN:
      return true;
    default:
      return false;
  }
}


uint32_t get_fa_num_small_rows(FaCodePath path) {
  if (path == FA_COOPMAT2) {
    return flash_attention_num_small_rows;
  }
  else {
    return scalar_flash_attention_num_small_rows;
  }
}

std::array<uint32_t, 2> fa_rows_cols(FaCodePath path, uint32_t hsk, uint32_t hsv, uint32_t clamp, v_data_type type,
                                     bool small_rows) {
  v_UNUSED(clamp);
  v_UNUSED(hsv);

  if (path == FA_SCALAR) {
    if (small_rows) {
      return {scalar_flash_attention_num_small_rows, 64};
    }
    else {
      if ((hsv | hsk) & 8) {
        // HSV/HSK not being a multiple of 16 makes D_split smaller, which makes cols_per_iter
        // larger, and Bc needs to be >= cols_per_thread. 64 is large enough, 32 is not.
        return {get_fa_scalar_num_large_rows(hsv), 64};
      }
      else {
        return {get_fa_scalar_num_large_rows(hsv), 32};
      }
    }
  }

  if (path == FA_COOPMAT1) {
    if (small_rows) {
      return {scalar_flash_attention_num_small_rows, scalar_flash_attention_Bc};
    }
    else {
      return {coopmat1_flash_attention_num_large_rows, scalar_flash_attention_Bc};
    }
  }

  // small rows, large cols
  if (small_rows) {
    return {get_fa_num_small_rows(FA_COOPMAT2), 32};
  }

  // small cols to reduce register count
  if (v_is_quantized(type) || hsk >= 256 || hsv >= 256) {
    if (hsk >= 512 || hsv >= 512) {
      return {32, 32};
    }
    else {
      return {64, 32};
    }
  }
  return {64, 64};
}

bool v_vk_dim01_contiguous(const v_tensor* tensor) {
  return
    tensor->nb[0] == v_type_size(tensor->type) &&
    tensor->nb[1] == (tensor->nb[0] * tensor->ne[0]) / block_size(tensor->type) &&
    (tensor->ne[3] == 1 || tensor->nb[3] == tensor->nb[2] * tensor->ne[2]);
}

uint32_t fa_align(FaCodePath path, uint32_t hsk, uint32_t hsv, v_data_type type, bool small_rows) {
  return fa_rows_cols(path, hsk, hsv, 0, type, small_rows)[1];
}

void v_vk_print_gpu_info(size_t idx) {
  V_ASSERT(idx < vk_instance.device_indices.size());
  size_t dev_num = vk_instance.device_indices[idx];
  VK_LOG_DEBUG("v_vk_print_gpu_info(" << dev_num << ")");
  V_ASSERT(vk_instance_initialized);

  std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

  if (dev_num >= devices.size()) {
    std::cerr << "v_vulkan: Device with index " << dev_num << " does not exist." << std::endl;
    throw std::runtime_error("Device not found");
  }

  vk::PhysicalDevice physical_device             = devices[dev_num];
  std::vector<vk::ExtensionProperties> ext_props = physical_device.enumerateDeviceExtensionProperties();

  bool fp16_storage        = false;
  bool fp16_compute        = false;
  bool coopmat_support     = false;
  bool coopmat2_support    = false;
  bool integer_dot_product = false;
  bool bfloat16_support    = false;

  for (auto properties : ext_props) {
    if (strcmp("VK_KHR_16bit_storage", properties.extensionName) == 0) {
      fp16_storage = true;
    }
    else if (strcmp("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
      fp16_compute = true;
      #if defined(V_VULKAN_COOPMAT_GLSLC_SUPPORT)
    }
    else if (strcmp("VK_KHR_cooperative_matrix", properties.extensionName) == 0 &&
      !getenv("v_VK_DISABLE_COOPMAT")) {
      coopmat_support = true;
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
      integer_dot_product = true;
      #endif
      #if defined(V_VULKAN_BFLOAT16_GLSLC_SUPPORT)
    }
    else if (strcmp("VK_KHR_shader_bfloat16", properties.extensionName) == 0 &&
      !getenv("v_VK_DISABLE_BFLOAT16")) {
      bfloat16_support = true;
      #endif
    }
  }

  const vk_device_architecture device_architecture = get_device_architecture(physical_device);

  const char* v_VK_DISABLE_F16 = getenv("v_VK_DISABLE_F16");
  bool force_disable_f16       = v_VK_DISABLE_F16 != nullptr;

  bool fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

  vk::PhysicalDeviceProperties2 props2;
  vk::PhysicalDeviceMaintenance3Properties props3;
  vk::PhysicalDeviceSubgroupProperties subgroup_props;
  vk::PhysicalDeviceDriverProperties driver_props;
  vk::PhysicalDeviceShaderIntegerDotProductPropertiesKHR shader_integer_dot_product_props;
  props2.pNext         = &props3;
  props3.pNext         = &subgroup_props;
  subgroup_props.pNext = &driver_props;

  // Pointer to the last chain element
  VkBaseOutStructure* last_struct = (VkBaseOutStructure*)&driver_props;

  if (integer_dot_product) {
    last_struct->pNext = (VkBaseOutStructure*)&shader_integer_dot_product_props;
    last_struct        = (VkBaseOutStructure*)&shader_integer_dot_product_props;
  }

  physical_device.getProperties2(&props2);

  VkPhysicalDeviceFeatures2 device_features2;
  device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  device_features2.pNext = nullptr;

  VkPhysicalDeviceVulkan11Features vk11_features;
  vk11_features.pNext    = nullptr;
  vk11_features.sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
  device_features2.pNext = &vk11_features;

  VkPhysicalDeviceVulkan12Features vk12_features;
  vk12_features.pNext = nullptr;
  vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  vk11_features.pNext = &vk12_features;

  // Pointer to the last chain element
  last_struct = (VkBaseOutStructure*)&vk12_features;

  #if defined(V_VULKAN_COOPMAT_GLSLC_SUPPORT)
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmat_features;
  coopmat_features.pNext             = nullptr;
  coopmat_features.sType             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
  coopmat_features.cooperativeMatrix = VK_FALSE;

  if (coopmat_support) {
    last_struct->pNext = (VkBaseOutStructure*)&coopmat_features;
    last_struct        = (VkBaseOutStructure*)&coopmat_features;
  }
  #endif

  VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR shader_integer_dot_product_features{};
  shader_integer_dot_product_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
  if (integer_dot_product) {
    last_struct->pNext = (VkBaseOutStructure*)&shader_integer_dot_product_features;
    last_struct        = (VkBaseOutStructure*)&shader_integer_dot_product_features;
  }

  #if defined(VK_KHR_shader_bfloat16)
  VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features{};
  bfloat16_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
  if (bfloat16_support) {
    last_struct->pNext = (VkBaseOutStructure*)&bfloat16_features;
    last_struct        = (VkBaseOutStructure*)&bfloat16_features;
  }
  #endif

  vkGetPhysicalDeviceFeatures2(physical_device, &device_features2);

  fp16 = fp16 && vk12_features.shaderFloat16;

  #if defined(VK_KHR_shader_bfloat16)
  bool bf16 = bfloat16_support && bfloat16_features.shaderBFloat16Type;
  #else
  bool bf16 = false;
  #endif

  uint32_t default_subgroup_size = get_subgroup_size("", device_architecture);
  const size_t subgroup_size     = (default_subgroup_size != 0) ? default_subgroup_size : subgroup_props.subgroupSize;
  const bool uma                 = props2.properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;

  integer_dot_product = integer_dot_product
    && shader_integer_dot_product_props.integerDotProduct4x8BitPackedSignedAccelerated
    && shader_integer_dot_product_features.shaderIntegerDotProduct;

  coopmat_support = coopmat_support
    #if defined(V_VULKAN_COOPMAT_GLSLC_SUPPORT)
    && coopmat_features.cooperativeMatrix
    #endif
    && vk_khr_cooperative_matrix_support(props2.properties, driver_props, device_architecture);

  std::string matrix_cores = coopmat2_support ? "NV_coopmat2" : coopmat_support ? "KHR_coopmat" : "none";

  std::string device_name = props2.properties.deviceName.data();
  v_LOG_DEBUG(
    "v_vulkan: %zu = %s (%s) | uma: %d | fp16: %d | bf16: %d | warp size: %zu | shared memory: %d | int dot: %d | matrix cores: %s\n",
    idx, device_name.c_str(), driver_props.driverName.data(), uma, fp16, bf16, subgroup_size,
    props2.properties.limits.maxComputeSharedMemorySize, integer_dot_product, matrix_cores.c_str());

  if (props2.properties.deviceType == vk::PhysicalDeviceType::eCpu) {
    v_LOG_DEBUG("v_vulkan: Warning: Device type is CPU. This is probably not the device you want.\n");
  }
}

bool v_vk_instance_validation_ext_available() {
  #ifdef V_VULKAN_VALIDATE
  // Check if validation layer provides the extension
  const std::string layer_name = "VK_LAYER_KHRONOS_validation";
  for (const auto& layer : vk::enumerateInstanceLayerProperties()) {
    if (layer_name == layer.layerName.data()) {
      for (const auto& ext : vk::enumerateInstanceExtensionProperties(layer_name)) {
        if (strcmp("VK_EXT_validation_features", ext.extensionName.data()) == 0) {
          return true;
        }
      }
    }
  }

  std::cerr << "v_vulkan: WARNING: Validation layer or layer extension VK_EXT_validation_features not found." <<
    std::endl;
  #endif
  return false;
}

bool v_vk_device_is_supported(const vk::PhysicalDevice& vkdev) {
  VkPhysicalDeviceFeatures2 device_features2;
  device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

  VkPhysicalDeviceVulkan11Features vk11_features;
  vk11_features.pNext    = nullptr;
  vk11_features.sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
  device_features2.pNext = &vk11_features;

  vkGetPhysicalDeviceFeatures2(vkdev, &device_features2);

  return vk11_features.storageBuffer16BitAccess;
}

bool v_vk_instance_debug_utils_ext_available(const std::vector<vk::ExtensionProperties>& instance_extensions) {
  // Check for portability enumeration extension for MoltenVK support
  for (const auto& properties : instance_extensions) {
    if (strcmp("VK_EXT_debug_utils", properties.extensionName) == 0) {
      return true;
    }
  }

  std::cerr << "v_vulkan: WARNING: Instance extension VK_EXT_debug_utils not found." << std::endl;
  return false;

  UNUSED(instance_extensions);
}

bool v_vk_instance_portability_enumeration_ext_available(const std::vector<vk::ExtensionProperties>& instance_extensions) {
  #ifdef __APPLE__
  // Check for portability enumeration extension for MoltenVK support
  for (const auto& properties : instance_extensions) {
    if (strcmp("VK_KHR_portability_enumeration", properties.extensionName) == 0) {
      return true;
    }
  }
  std::cerr << "v_vulkan: WARNING: Instance extension VK_KHR_portability_enumeration not found." << std::endl;
  #endif
  return false;

  UNUSED(instance_extensions);
}

size_t vk_align_size(size_t width, size_t align) {
  VK_LOG_DEBUG("v_vk_align_size(" << width << ", " << align << ")");
  return CEIL_DIV(width, align) * align;
}


vk_pipeline v_vk_get_to_fp16(vk_backend_ctx* ctx, v_data_type type) {
  VK_LOG_DEBUG("v_vk_get_to_fp16()");
  switch (type) {
    case v_TYPE_F32:
    case v_TYPE_Q4_0:
    case v_TYPE_Q4_1:
    case v_TYPE_Q5_0:
    case v_TYPE_Q5_1:
    case v_TYPE_Q8_0:
    case v_TYPE_Q2_K:
    case v_TYPE_Q3_K:
    case v_TYPE_Q4_K:
    case v_TYPE_Q5_K:
    case v_TYPE_Q6_K:
    case v_TYPE_IQ1_S:
    case v_TYPE_IQ1_M:
    case v_TYPE_IQ2_XXS:
    case v_TYPE_IQ2_XS:
    case v_TYPE_IQ2_S:
    case v_TYPE_IQ3_XXS:
    case v_TYPE_IQ3_S:
    case v_TYPE_IQ4_XS:
    case v_TYPE_IQ4_NL:
    case v_TYPE_MXFP4:
      break;
    default:
      return nullptr;
  }

  return ctx->device->pipeline_dequant[type];
}

uint32_t get_fa_scalar_num_large_rows(uint32_t hsv) {
  if (hsv >= 192) {
    return 2;
  }
  else {
    return 8;
  }
}

uint32_t find_properties(const vk::PhysicalDeviceMemoryProperties* mem_props, vk::MemoryRequirements* mem_req,
                         vk::MemoryPropertyFlags flags) {
  for (uint32_t i = 0; i < mem_props->memoryTypeCount; ++i) {
    vk::MemoryType memory_type = mem_props->memoryTypes[i];
    if ((mem_req->memoryTypeBits & ((uint64_t)1 << i)) &&
      (flags & memory_type.propertyFlags) == flags &&
      mem_props->memoryHeaps[memory_type.heapIndex].size >= mem_req->size) {
      return static_cast<int32_t>(i);
    }
  }
  return UINT32_MAX;
}

void vk_sync_buffers(vk_backend_ctx* ctx, vk_context& subctx) {
  VK_LOG_DEBUG("v_vk_sync_buffers()");

  const bool transfer_queue = subctx->p->q->transfer_only;

  if (ctx) {
    ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_split_k_need_sync = false;
  }

  subctx->s->buffer.pipelineBarrier(
    subctx->p->q->stage_flags,
    subctx->p->q->stage_flags,
    {},
    {
      {
        {
          !transfer_queue
            ? (vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferRead |
              vk::AccessFlagBits::eTransferWrite)
            : (vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite)
        },
        {
          !transfer_queue
            ? (vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferRead |
              vk::AccessFlagBits::eTransferWrite)
            : (vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite)
        }
      }
    },
    {},
    {}
  );
}
