#ifndef MYPROJECT_VK_CONTEXT_H
#define MYPROJECT_VK_CONTEXT_H
#include <condition_variable>
#include "vk_common.h"
#include "vk_buffer.h"
#include "vk_config.h"
#include "vk_queue.hpp"
void vk_init(vk_backend_ctx* ctx, size_t idx);
void vk_begin_ctx(vk_device& device, vk_context& subctx);
void vk_ctx_end(vk_context& ctx);
vk_context vk_create_temp_ctx(vk_command_pool& p);
vk_context vk_create_context(vk_backend_ctx* ctx, vk_command_pool& p);
void v_vk_cleanup(vk_backend_ctx* ctx);
void vk_ctx_pre_alloc_buffers(vk_backend_ctx* ctx);
void v_vk_wait_events(vk_context& ctx, std::vector<vk::Event>&& events);

const char* vk_name(v_backend_t backend);

void vk_backend_free(v_backend_t backend);

v_backend_buffer_type_t vk_get_device_buffer(v_backend_t backend);

void v_backend_vk_set_tensor_async(v_backend_t backend, v_tensor* tensor, const void* data,
                                   size_t offset, size_t size);

void v_backend_vk_get_tensor_async(v_backend_t backend, const v_tensor* tensor, void* data,
                                   size_t offset, size_t size);

bool v_backend_buffer_is_vk(v_backend_buffer_t buffer);

void vk_free_buffer(v_backend_buffer_t buffer);

void* vk_device_buffer_get_base(v_backend_buffer_t buffer);

enum v_status vk_buffer_init_tensor(v_backend_buffer_t buffer,
                                    const v_tensor* tensor);


void vk_device_buffer_set_tensor(v_backend_buffer_t buffer, v_tensor* tensor, const void* data,
                                 size_t offset, size_t size);

void vk_device_buffer_get_tensor(v_backend_buffer_t buffer,
                                 const v_tensor* tensor,
                                 void* data,
                                 size_t offset,
                                 size_t size);


void vk_buffer_clear(v_backend_buffer_t buffer, uint8_t value);


bool v_backend_vk_cpy_tensor_async(v_backend_t backend, const v_tensor* src, v_tensor* dst);

void v_backend_vk_synchronize(v_backend_t backend);
v_backend_t backend_vk_init(size_t dev_num);

void vk_get_device_description(int device, char* description, size_t description_size);

void vk_get_device_memory(int device, size_t* free, size_t* total);

vk::PhysicalDeviceType v_backend_vk_get_device_type(int device_idx);

std::string v_backend_vk_get_device_pci_id(int device_idx);


bool vk_device_supports_buft(struct vk_device_ctx* dev, v_backend_buffer_type_t buft);


v_backend_buffer_t vk_device_buffer_alloc(v_backend_buffer_type_t buft, size_t size);


struct vk_device_ctx* v_backend_vk_reg_get_device(size_t device);


struct vk_context_struct {
  vk_summition* s;
  std::vector<vk_sequence> seqs;

  int exit_tensor_idx;

  std::vector<vk_staging_memcpy> in_memcpys;
  std::vector<vk_staging_memcpy> out_memcpys;
  std::vector<vk_staging_memset> memsets;

  vk_command_pool* p{};
};

typedef std::shared_ptr<vk_context_struct> vk_context;
typedef std::weak_ptr<vk_context_struct> vk_context_ref;

struct v_vk_garbage_collector {
  std::vector<vk_semaphore> tl_semaphores;
  std::vector<vk_semaphore> semaphores;
  std::vector<vk::Event> events;
  std::vector<vk_context> contexts;
};


struct vk_backend_ctx {
  std::string name;
  vk_device device;
  size_t semaphore_idx, event_idx;
  v_vk_garbage_collector gc;
  size_t prealloc_size_x, prealloc_size_y, prealloc_size_split_k, prealloc_size_add_rms_partials,
         prealloc_size_add_rms_partials_offset;
  vk_buffer prealloc_x, prealloc_y, prealloc_split_k, prealloc_add_rms_partials;
  vk::Fence fence, almost_ready_fence;
  bool almost_ready_fence_pending{};
  // Set before op_add and unset after op_rms_norm to indicate that the add should
  // write partial sums to accumulate the square of the vector components
  bool do_add_rms_partials;

  // Cache most recent tensor that was converted into prealloc_y, and what pipeline it used_bits__ to convert.
  vk_pipeline_struct* prealloc_y_last_pipeline_used{};
  const v_tensor* prealloc_y_last_tensor_used{};

  // Track which nodes have been used_bits__ since the last sync, and whether they were written to
  std::vector<const v_tensor*> unsynced_nodes_written;
  std::vector<const v_tensor*> unsynced_nodes_read;
  // Track which prealloc buffers have pending reads that need to be synchronized.
  // These are checked before writing to the buffer (and call v_vk_sync_buffers if set),
  // and set to true after the buffer contents are consumed.
  bool prealloc_x_need_sync, prealloc_y_need_sync, prealloc_split_k_need_sync;

  vk_context_ref compute_ctx;
  vk_context_ref transfer_ctx;

  std::vector<vk_context_ref> tensor_ctxs;

  std::vector<vk::DescriptorPool> descriptor_pools;
  std::vector<vk::DescriptorSet> descriptor_sets;
  uint32_t descriptor_set_idx{};
  uint32_t pipeline_descriptor_set_requirements{};

  vk_command_pool compute_cmd_pool;
  vk_command_pool transfer_cmd_pool;
  // number of additional consecutive nodes that are being fused with the
  // node currently being processed
  int num_additional_fused_ops{};
};

struct vk_instance_struct {
  vk::Instance instance;
  bool debug_utils_support                                              = false; // VK_EXT_debug_utils enabled
  PFN_vkSetDebugUtilsObjectNameEXT pfn_vkSetDebugUtilsObjectNameEXT     = {};
  PFN_vkQueueBeginDebugUtilsLabelEXT pfn_vkQueueBeginDebugUtilsLabelEXT = {};
  PFN_vkQueueEndDebugUtilsLabelEXT pfn_vkQueueEndDebugUtilsLabelEXT     = {};
  PFN_vkCmdBeginDebugUtilsLabelEXT pfn_vkCmdBeginDebugUtilsLabelEXT     = {};
  PFN_vkCmdEndDebugUtilsLabelEXT pfn_vkCmdEndDebugUtilsLabelEXT         = {};
  PFN_vkCmdInsertDebugUtilsLabelEXT pfn_vkCmdInsertDebugUtilsLabelEXT   = {};

  std::vector<size_t> device_indices;
  std::vector<bool> device_supports_membudget;
  vk_device devices[v_VK_MAX_DEVICES];
};

extern vk_instance_struct vk_instance;
extern bool vk_instance_initialized;
extern bool vk_perf_logger_enabled;

#endif //MYPROJECT_VK_CONTEXT_H
