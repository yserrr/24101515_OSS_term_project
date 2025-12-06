#include <array>

#include "vk_device.h"

#include "v-vulkan-kernels.hpp"
#include "vk_buffer.h"
#include "vk_context.h"
#include "vk_constant.h"
#include "vk_util.h"
#include "vk_pipeline.h"


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
                            (uint32_t)block_size(v_TYPE_Q4_0), 1, 1
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
                            (uint32_t)block_size(v_TYPE_Q4_1), 1, 1
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
                            (uint32_t)block_size(v_TYPE_Q5_0), 1, 1
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
                            (uint32_t)block_size(v_TYPE_Q5_1), 1, 1
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
                            (uint32_t)block_size(v_TYPE_Q8_0), 1, 1
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
                          {(uint32_t)block_size(v_TYPE_IQ4_NL), 1, 1},
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
