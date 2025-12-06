#ifndef VK_QUEUE_H
#define VK_QUEUE_H
#include "vk_common.h"

struct vk_command_pool {
  void init(vk_device& device, vk_queue* q_);
  void destroy(vk::Device& device);
  vk::CommandPool pool;
  uint32_t cmd_buffer_idx;
  std::vector<vk::CommandBuffer> cmd_buffers;
  vk_queue* q;
};


vk::CommandBuffer v_vk_create_cmd_buffer(vk_device& device, vk_command_pool& p);
void v_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p);
vk_summition vk_begin_sub_mission(vk_device& device, vk_command_pool& p, bool one_time = true);
vk::CommandBuffer v_vk_create_cmd_buffer(vk_device& device, vk_command_pool& p) ;
// Stores command pool/buffers. There's an instance of this
// for each (context,queue) pair and for each (device,queue) pair.
// Prevent simultaneous submissions to the same queue.
// This could be per vk_queue if we stopped having two vk_queue structures
// sharing the same vk::Queue.
struct vk_queue {
  uint32_t queue_family_index;
  vk::Queue queue;
  vk_command_pool cmd_pool;
  vk::PipelineStageFlags stage_flags;
  bool transfer_only;
  // copy everything except the cmd_pool
  // cmd pool -> multi-threading
  void copyFrom(vk_queue& other);
};

struct vk_semaphore {
  vk::Semaphore s;
  uint64_t value;
};


struct vk_summition {
  vk::CommandBuffer buffer;
  std::vector<vk_semaphore> wait_semaphores;
  std::vector<vk_semaphore> signal_semaphores;
};

void vk_submit(vk_context& ctx, vk::Fence fence);
void v_vk_end_submission(vk_summition& s, std::vector<vk_semaphore> wait_semaphores, std::vector<vk_semaphore> signal_semaphores);
void vk_queue_command_pools_clean_up(vk_device& device);
uint32_t v_vk_find_queue_family_index(std::vector<vk::QueueFamilyProperties>& queue_family_props, const vk::QueueFlags& required, const vk::QueueFlags& avoid, int32_t compute_index, uint32_t min_num_queues);


vk_semaphore* v_vk_create_binary_semaphore(vk_backend_ctx* ctx);


void vk_submit(vk_context& ctx, vk::Fence fence);


vk_semaphore* v_vk_create_timeline_semaphore(vk_backend_ctx* ctx);

vk::Event v_vk_create_event(vk_backend_ctx* ctx);

void v_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p);

void vk_queue_command_pools_clean_up(vk_device& device);
void v_vk_wait_for_fence(vk_backend_ctx* ctx);


#endif
