#include <iostream>
#include "vk_queue.h"
#include "vk_constant.h"
#include "vk_context.h"
#include "vk_device.h"

void MmlCommandPool::init(vk_device& device, vk_queue* q_)
{
  cmd_buffer_idx = 0;
  q = q_;

  vk::CommandPoolCreateInfo command_pool_create_info(vk::CommandPoolCreateFlags(VK_COMMAND_POOL_CREATE_TRANSIENT_BIT),
                                                     q->queue_family_index);
  pool = device->device.createCommandPool(command_pool_create_info);
}

MmlVkSemaphore* v_vk_create_binary_semaphore(vk_backend_ctx* ctx)
{
  VK_LOG_DEBUG("v_vk_create_timeline_semaphore()");
  vk::SemaphoreTypeCreateInfo tci{vk::SemaphoreType::eBinary, 0};
  vk::SemaphoreCreateInfo ci{};
  ci.setPNext(&tci);
  vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
  ctx->gc.semaphores.push_back({semaphore, 0});
  return &ctx->gc.semaphores[ctx->gc.semaphores.size() - 1];
}


MmlVkSummition mmlVKBeginSubMission(vk_device& device,
                                    MmlCommandPool& p,
                                    bool one_time)
{
  MmlVkSummition s;
  s.buffer = v_vk_create_cmd_buffer(device, p);
  if (one_time)
  {
    s.buffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  }
  else
  {
    s.buffer.begin({vk::CommandBufferUsageFlags{}});
  }

  return s;
}

void vk_queue::copyFrom(vk_queue& other)
{
  queue_family_index = other.queue_family_index;
  queue = other.queue;
  stage_flags = other.stage_flags;
  transfer_only = other.transfer_only;
}

void mmlVKEndSubmission(MmlVkSummition& s,
                        std::vector<MmlVkSemaphore> wait_semaphores,
                        std::vector<MmlVkSemaphore> signal_semaphores)
{
  s.buffer.end();
  s.wait_semaphores = std::move(wait_semaphores);
  s.signal_semaphores = std::move(signal_semaphores);
}
uint32_t v_vk_find_queue_family_index(std::vector<vk::QueueFamilyProperties>& queue_family_props,
                                         const vk::QueueFlags& required,
                                         const vk::QueueFlags& avoid,
                                         int32_t compute_index,
                                         uint32_t min_num_queues)
{
  VK_LOG_DEBUG("v_vk_find_queue_family_index()");
  const uint32_t qfsize = queue_family_props.size();

  // Try with avoid preferences first
  for (uint32_t i = 0; i < qfsize; i++)
  {
    if (queue_family_props[i].queueCount >= min_num_queues && (compute_index < 0 || i != (uint32_t)compute_index) &&
        queue_family_props[i].queueFlags & required && !(queue_family_props[i].queueFlags & avoid))
    {
      return i;
    }
  }

  // Fall back to only required
  for (size_t i = 0; i < qfsize; i++)
  {
    if (queue_family_props[i].queueCount >= min_num_queues && (compute_index < 0 || i != (uint32_t)compute_index) &&
        queue_family_props[i].queueFlags & required)
    {
      return i;
    }
  }

  // Fall back to reusing compute queue
  for (size_t i = 0; i < qfsize; i++)
  {
    if (queue_family_props[i].queueCount >= min_num_queues && queue_family_props[i].queueFlags & required)
    {
      return i;
    }
  }

  // Fall back to ignoring min_num_queries
  for (size_t i = 0; i < qfsize; i++)
  {
    if (queue_family_props[i].queueFlags & required)
    {
      return i;
    }
  }

  /// All commands that are allowed on a queue that supports transfer operations are also allowed on a queue that supports either graphics or compute operations.
  /// Thus, if the capabilities of a queue family include VK_QUEUE_GRAPHICS_BIT or VK_QUEUE_COMPUTE_BIT, then reporting the VK_QUEUE_TRANSFER_BIT capability separately for that queue family is optional.
  if (compute_index >= 0)
  {
    return compute_index;
  }

  std::cerr << "v_vulkan: No suitable queue family index found." << std::endl;

  for (auto& q_family : queue_family_props)
  {
    std::cerr << "Queue number: " + std::to_string(q_family.queueCount) << " flags: " + to_string(q_family.queueFlags)
      << std::endl;
  }
  abort();
}

void MmlCommandPool::destroy(vk::Device& device)
{
  device.destroyCommandPool(pool);
  pool = nullptr;
  cmd_buffers.clear();
}
