#ifndef VK_QUEUE_H
#define VK_QUEUE_H
#include "vk_common.h"
struct MmlCommandPool
{
  void init(vk_device& device, vk_queue* q_);
  void destroy(vk::Device& device);
  vk::CommandPool pool;
  uint32_t cmd_buffer_idx;
  std::vector<vk::CommandBuffer> cmd_buffers;
  vk_queue* q;
};


vk::CommandBuffer v_vk_create_cmd_buffer(vk_device& device, MmlCommandPool& p);
void v_vk_command_pool_cleanup(vk_device& device, MmlCommandPool& p);
MmlVkSummition mmlVKBeginSubMission(vk_device& device, MmlCommandPool& p, bool one_time = true);

// Stores command pool/buffers. There's an instance of this
// for each (context,queue) pair and for each (device,queue) pair.
// Prevent simultaneous submissions to the same queue.
// This could be per vk_queue if we stopped having two vk_queue structures
// sharing the same vk::Queue.
struct vk_queue
{
  uint32_t queue_family_index;
  vk::Queue queue;
  MmlCommandPool cmd_pool;
  vk::PipelineStageFlags stage_flags;
  bool transfer_only;
  // copy everything except the cmd_pool
  // cmd pool -> multi-threading
  void copyFrom(vk_queue& other);

};

struct MmlVkSemaphore
{
  vk::Semaphore s;
  uint64_t value;
};


struct MmlVkSummition
{
  vk::CommandBuffer buffer;
  std::vector<MmlVkSemaphore> wait_semaphores;
  std::vector<MmlVkSemaphore> signal_semaphores;
};

void vk_submit(vk_context& ctx,
                    vk::Fence fence);
void mmlVKEndSubmission(MmlVkSummition& s,
                        std::vector<MmlVkSemaphore> wait_semaphores,
                        std::vector<MmlVkSemaphore> signal_semaphores);
//vk_semaphore* v_vk_create_binary_semaphore(v_backend_vk_context* ctx);
//vk_semaphore* v_vk_create_timeline_semaphore(v_backend_vk_context* ctx);
//vk_submission v_vk_begin_submission(vk_device& device, vk_command_pool& p, bool one_time = true);
//void v_vk_end_submission(vk_submission& s,
//                            std::vector<vk_semaphore> wait_semaphores,
//                            std::vector<vk_semaphore> signal_semaphores);
//
//
//
//
//

void mmlVkQueueCommandPoolsCleanUp(vk_device& device);
uint32_t v_vk_find_queue_family_index(std::vector<vk::QueueFamilyProperties>& queue_family_props,
                                         const vk::QueueFlags& required,
                                         const vk::QueueFlags& avoid,
                                         int32_t compute_index,
                                         uint32_t min_num_queues);

//void v_vk_create_queue(vk_device& device,
//                          vk_queue& q,
//                          uint32_t queue_family_index,
//                          uint32_t queue_index,
//                          vk::PipelineStageFlags&& stage_flags,
//                          bool transfer_only);
//
//void v_vk_queue_command_pools_cleanup(vk_device& device);

MmlVkSemaphore* v_vk_create_binary_semaphore(vk_backend_ctx* ctx);

#endif
