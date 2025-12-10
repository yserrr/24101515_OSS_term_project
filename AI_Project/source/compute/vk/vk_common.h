#ifndef MYPROJECT_VK_COMMON_H
#define MYPROJECT_VK_COMMON_H
#include "v_vk.hpp"
#include <vulkan/vulkan_core.h>

#if defined(v_VULKAN_RUN_TESTS) || defined(v_VULKAN_CHECK_RESULTS)
#include <chrono>
#endif
// See https://github.com/KhronosGroup/Vulkan-Hpp?tab=readme-ov-file#extensions--per-device-function-pointers-
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
// We use VULKAN_HPP_DEFAULT_DISPATCHER, but not VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
// to avoid conflicts with applications or other libraries who might use it.
#if VK_HEADER_VERSION >= 301

namespace vk::detail
{
  class DispatchLoaderDynamic;
}

using vk::detail::DispatchLoaderDynamic;
#else
namespace vk
{
  class DispatchLoaderDynamic;
}
using vk::DispatchLoaderDynamic;
#endif
DispatchLoaderDynamic& v_vk_default_dispatcher();
#define VULKAN_HPP_DEFAULT_DISPATCHER v_vk_default_dispatcher()

#include <vulkan/vulkan.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>
#include <sstream>
#include <utility>
#include <memory>
#include <limits>
#include <map>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <future>
#include <thread>

#if defined(_MSC_VER)
# define NOMINMAX 1
# include <windows.h>
# define YIELD() YieldProcessor()
#elif defined(__clang__) || defined(__GNUC__)
# if defined(__x86_64__) ||defined(__i386__)
#  include <immintrin.h>
#  define YIELD() _mm_pause()
# elif defined(__arm__) || defined(__aarch64__)
#  if defined(__clang__)
#   include <arm_acle.h>
#   define YIELD() __yield()
#  else
#   define YIELD() asm volatile("yield")
#  endif
# endif
#endif

#if !defined(YIELD)
#define YIELD()
#endif

#include "ggml-impl.hpp"

// remove this once it's more widely available in the SDK
#if !defined(VK_KHR_shader_bfloat16)

#define VK_KHR_shader_bfloat16 1
#define VK_KHR_SHADER_BFLOAT16_SPEC_VERSION                          1
#define VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME                        "VK_KHR_shader_bfloat16"
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR ((VkStructureType)1000141000)
#define VK_COMPONENT_TYPE_BFLOAT16_KHR                               ((VkComponentTypeKHR)1000141000)

typedef struct VkPhysicalDeviceShaderBfloat16FeaturesKHR
{
  VkStructureType sType;
  void* pNext;
  VkBool32 shaderBFloat16Type;
  VkBool32 shaderBFloat16DotProduct;
  VkBool32 shaderBFloat16CooperativeMatrix;
} VkPhysicalDeviceShaderBfloat16FeaturesKHR;
#endif

#define ROUNDUP_POW2(M, N) (((M) + (N) - 1) & ~((N) - 1))
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define UNUSED V_UNUSED

//#define MML_VK_DEBUG
#ifdef MML_VK_DEBUG
#define VK_LOG_DEBUG(msg) std::cerr << msg << std::endl
#else
#define VK_LOG_DEBUG(msg) ((void) 0)
#endif //MML_VK_DEBUG


inline constexpr uint32_t VK_VENDOR_ID_AMD = 0x1002;
inline constexpr uint32_t VK_VENDOR_ID_APPLE = 0x106b;
inline constexpr uint32_t VK_VENDOR_ID_INTEL = 0x8086;
inline constexpr uint32_t VK_VENDOR_ID_NVIDIA = 0x10de;

inline constexpr uint32_t VK_DEVICE_DESCRIPTOR_POOL_SIZE = 256;
inline constexpr uint32_t v_VK_MAX_NODES = 8192;
inline constexpr uint32_t MAX_PARAMETER_COUNT = 12;
// Max number of adds that can be fused without exceeding MAX_PARAMETER_COUNT.
inline constexpr uint32_t MAX_FUSED_ADDS = (MAX_PARAMETER_COUNT - 3);
inline constexpr uint32_t mul_mat_vec_max_cols = 8;
inline constexpr uint32_t p021_max_gqa_ratio = 8;
inline constexpr uint32_t num_argsort_pipelines = 11;
inline constexpr uint32_t max_argsort_cols = 1 << (num_argsort_pipelines - 1);
inline constexpr uint32_t num_topk_moe_pipelines = 10;
inline constexpr uint32_t coopmat1_flash_attention_num_large_rows = 16;
inline constexpr uint32_t scalar_flash_attention_Bc = 64;
inline constexpr uint32_t scalar_flash_attention_workgroup_size = 128;
// number of rows/cols for flash attention shader
inline constexpr uint32_t flash_attention_num_small_rows = 32;
inline constexpr uint32_t scalar_flash_attention_num_small_rows = 1;

inline bool is_pow2(uint32_t x) { return x > 1 && (x & (x - 1)) == 0; }
#define VK_CHECK(err, msg)                                          \
do {                                                            \
vk::Result err_ = (err);                                    \
if (err_ != vk::Result::eSuccess) {                         \
fprintf(stderr, "v_vulkan: %s error %s at %s:%d\n",  \
#err, to_string(err_).c_str(), __FILE__, __LINE__); \
exit(1);                                                \
}                                                           \
} while (0)


struct vk_device_struct;
struct vk_buffer_struct;
struct vk_pipeline_struct;

struct vk_queue;
struct vk_command_pool;
struct vk_backend_ctx;

struct vk_semaphore;
struct vk_summition;

struct vk_context_struct;
struct v_vk_garbage_collector;

struct vk_matmul_pipeline_struct;
struct vk_matmul_pipeline2_struct;

using vk_pipeline     = std::shared_ptr<vk_pipeline_struct>;
using vk_pipeline_ref = std::weak_ptr<vk_pipeline_struct>;
using vk_matmul_pipeline = std::shared_ptr<vk_matmul_pipeline_struct>;

typedef std::shared_ptr<vk_context_struct> vk_context;
typedef std::weak_ptr<vk_context_struct> vk_context_ref;

typedef std::vector<vk_summition> vk_sequence;
typedef std::shared_ptr<vk_device_struct> vk_device;
typedef std::weak_ptr<vk_device_struct> vk_device_ref;

typedef std::shared_ptr<vk_buffer_struct> vk_buffer;
typedef std::weak_ptr<vk_buffer_struct> vk_buffer_ref;
typedef std::shared_ptr<vk_pipeline_struct> vk_pipeline;
typedef std::weak_ptr<vk_pipeline_struct> vk_pipeline_ref;
typedef std::shared_ptr<vk_matmul_pipeline_struct> vk_matmul_pipeline;
//#ifdef v_VULKAN_DEBUG
//#define VK_LOG_DEBUG(msg) std::cerr << msg << std::endl
//#else
//#define VK_LOG_DEBUG(msg) ((void) 0)
//#endif // v_VULKAN_DEBUG

//constant
struct vk_mat_mat_push_constants;
struct vk_mat_vec_push_constants;
struct vk_mat_mat_id_push_constants;
struct vk_mat_vec_id_push_constants;
struct vk_flash_attn_push_constants;
struct vk_op_push_constants;
struct vk_op_glu_push_constants;
struct vk_op_unary_push_constants;
struct vk_op_pad_push_constants;
struct vk_op_binary_push_constants;
struct vk_op_multi_add_push_constants;
struct vk_op_topk_moe_push_constants;
struct vk_op_add_id_push_constants;
struct vk_op_diag_mask_push_constants;
struct vk_op_rope_push_constants;
struct vk_op_soft_max_push_constants;
struct vk_op_argsort_push_constants;
struct vk_op_im2col_push_constants;
struct vk_op_im2col_3d_push_constants;
struct vk_op_timestep_embedding_push_constants;
struct vk_op_conv_transpose_1d_push_constants;
struct vk_op_pool2d_push_constants;
struct vk_op_rwkv_wkv6_push_constants;
struct vk_op_rwkv_wkv7_push_constants;
struct vk_op_ssm_scan_push_constants;
struct vk_op_ssm_conv_push_constants;
struct vk_op_conv2d_push_constants;
struct vk_op_conv_transpose_2d_push_constants;
struct vk_op_conv2d_dw_push_constants;
struct vk_op_upscale_push_constants;
struct vk_op_sum_rows_push_constants;

VkDeviceSize v_vk_get_max_buffer_range(const vk_backend_ctx* ctx, const vk_buffer& buf, const VkDeviceSize offset);
void vk_buffer_memset_async(vk_context& ctx, vk_buffer& dst, size_t offset, uint32_t c, size_t size);
void init_fastdiv_values(uint32_t d, uint32_t& mp, uint32_t& L);

uint32_t get_misalign_bytes(vk_backend_ctx* ctx, const v_tensor* t);

void* const vk_ptr_base = (void*)(uintptr_t)0x1000; // NOLINT


//// number of rows/cols for flash attention shader
//constexpr uint32_t flash_attention_num_small_rows = 32;
//constexpr uint32_t scalar_flash_attention_num_small_rows = 1;
//
//
//// The FA coopmat1 shader assumes 16x16x16 matrix multiply support.
//// 128 threads split into four subgroups, each subgroup does 1/4
//// of the Bc dimension.
//constexpr uint32_t coopmat1_flash_attention_num_large_rows = 16;
//constexpr uint32_t scalar_flash_attention_Bc = 64;
//constexpr uint32_t scalar_flash_attention_workgroup_size = 128;

#if defined(v_VULKAN_MEMORY_DEBUG) || defined(v_VULKAN_DEBUG)
#define VK_LOG_MEMORY(msg) std::cerr << "v_vulkan memory: " << msg << std::endl

std::string format_size(size_t size)
{
  const size_t kib = 1024;
  const size_t mib = kib * 1024;
  const size_t gib = mib * 1024;

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);

  if (size >= gib)
  {
    oss << static_cast<double>(size) / gib << " GiB";
  }
  else if (size >= mib)
  {
    oss << static_cast<double>(size) / mib << " MiB";
  }
  else if (size >= kib)
  {
    oss << static_cast<double>(size) / kib << " KiB";
  }
  else
  {
    oss << size << " B";
  }

  return oss.str();
}

class vk_memory_logger
{
public:
  vk_memory_logger() : total_device(0), total_host(0)
  {
  }

  void log_allocation(vk_buffer_ref buf_ref, size_t size);
  void log_deallocation(vk_buffer_ref buf_ref);

private:
  std::map<vk::Buffer, size_t> allocations; // Track allocations
  size_t total_device;
  size_t total_host;
};
#else
#define VK_LOG_MEMORY(msg) ((void) 0)
#endif // v_VULKAN_MEMORY_DEBUG

#endif //MYPROJECT_VK_COMMON_H
