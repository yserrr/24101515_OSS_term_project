#include <iostream>
#include "vk_queue.h"
#include "vk_constant.h"
#include "vk_context.h"
#include "vk_device.h"

std::mutex queue_mutex;

void vk_command_pool::init(vk_device& device, vk_queue* q_) {
  cmd_buffer_idx = 0;
  q              = q_;

  vk::CommandPoolCreateInfo command_pool_create_info(vk::CommandPoolCreateFlags(VK_COMMAND_POOL_CREATE_TRANSIENT_BIT),
                                                     q->queue_family_index);
  pool = device->device.createCommandPool(command_pool_create_info);
}

vk_semaphore* v_vk_create_binary_semaphore(vk_backend_ctx* ctx) {
  VK_LOG_DEBUG("v_vk_create_timeline_semaphore()");
  vk::SemaphoreTypeCreateInfo tci{vk::SemaphoreType::eBinary, 0};
  vk::SemaphoreCreateInfo ci{};
  ci.setPNext(&tci);
  vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
  ctx->gc.semaphores.push_back({semaphore, 0});
  return &ctx->gc.semaphores[ctx->gc.semaphores.size() - 1];
}

void vk_submit(vk_context& ctx, vk::Fence fence) {
  if (ctx->seqs.empty()) {
    if (fence) {
      std::lock_guard<std::mutex> guard(queue_mutex);
      ctx->p->q->queue.submit({}, fence);
    }
    return;
  }
  VK_LOG_DEBUG("v_vk_submit(" << ctx << ", " << fence << ")");

  std::vector<std::vector<uint64_t>> tl_wait_vals;
  std::vector<std::vector<uint64_t>> tl_signal_vals;
  std::vector<std::vector<vk::Semaphore>> tl_wait_semaphores;
  std::vector<std::vector<vk::Semaphore>> tl_signal_semaphores;
  std::vector<vk::TimelineSemaphoreSubmitInfo> tl_submit_infos;
  std::vector<vk::SubmitInfo> submit_infos;
  int idx = -1;
  std::vector<std::vector<vk::PipelineStageFlags>> stage_flags;

  size_t reserve = 0;

  for (const auto& sequence : ctx->seqs) { reserve += sequence.size(); }

  // Pre-reserve vectors to prevent reallocation, which invalidates pointers
  tl_wait_semaphores.reserve(reserve);
  tl_wait_vals.reserve(reserve);
  tl_signal_semaphores.reserve(reserve);
  tl_signal_vals.reserve(reserve);
  tl_submit_infos.reserve(reserve);
  submit_infos.reserve(reserve);
  stage_flags.reserve(reserve);

  for (const auto& sequence : ctx->seqs) {
    for (const auto& submission : sequence) {
      stage_flags.push_back({});
      idx++;
      tl_wait_vals.push_back({});
      tl_wait_semaphores.push_back({});
      tl_signal_vals.push_back({});
      tl_signal_semaphores.push_back({});
      for (size_t i = 0; i < submission.wait_semaphores.size(); i++) {
        stage_flags[idx].push_back(ctx->p->q->stage_flags);
        tl_wait_vals[idx].push_back(submission.wait_semaphores[i].value);
        tl_wait_semaphores[idx].push_back(submission.wait_semaphores[i].s);
      }
      for (size_t i = 0; i < submission.signal_semaphores.size(); i++) {
        tl_signal_vals[idx].push_back(submission.signal_semaphores[i].value);
        tl_signal_semaphores[idx].push_back(submission.signal_semaphores[i].s);
      }
      tl_submit_infos.push_back({
        (uint32_t)submission.wait_semaphores.size(),
        tl_wait_vals[idx].data(),
        (uint32_t)submission.signal_semaphores.size(),
        tl_signal_vals[idx].data(),
      });
      tl_submit_infos[idx].sType = vk::StructureType::eTimelineSemaphoreSubmitInfo;
      tl_submit_infos[idx].pNext = nullptr;
      vk::SubmitInfo si{
        (uint32_t)submission.wait_semaphores.size(),
        tl_wait_semaphores[idx].data(),
        stage_flags[idx].data(),
        1,
        &submission.buffer,
        (uint32_t)submission.signal_semaphores.size(),
        tl_signal_semaphores[idx].data(),
      };
      si.setPNext(&tl_submit_infos[idx]);
      submit_infos.push_back(si);
    }
  }

  std::lock_guard<std::mutex> guard(queue_mutex);
  ctx->p->q->queue.submit(submit_infos, fence);

  ctx->seqs.clear();
}

vk_semaphore* v_vk_create_timeline_semaphore(vk_backend_ctx* ctx) {
  VK_LOG_DEBUG("v_vk_create_timeline_semaphore()");
  if (ctx->semaphore_idx >= ctx->gc.tl_semaphores.size()) {
    vk::SemaphoreTypeCreateInfo tci{vk::SemaphoreType::eTimeline, 0};
    vk::SemaphoreCreateInfo ci{};
    ci.setPNext(&tci);
    vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
    ctx->gc.tl_semaphores.push_back({semaphore, 0});
  }
  return &ctx->gc.tl_semaphores[ctx->semaphore_idx++];
}

vk::Event v_vk_create_event(vk_backend_ctx* ctx) {
  if (ctx->event_idx >= ctx->gc.events.size()) { ctx->gc.events.push_back(ctx->device->device.createEvent({})); }
  return ctx->gc.events[ctx->event_idx++];
}

void v_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p) {
  VK_LOG_DEBUG("v_vk_command_pool_cleanup()");

  // Requires command buffers to be done
  device->device.resetCommandPool(p.pool);
  p.cmd_buffer_idx = 0;
}

void vk_queue_command_pools_clean_up(vk_device& device) {
  VK_LOG_DEBUG("v_vk_queue_command_pools_cleanup()");

  // Arbitrary frequency to cleanup/reuse command buffers
  constexpr uint32_t cleanup_frequency = 10;

  if (device->compute_queue.cmd_pool.cmd_buffer_idx >= cleanup_frequency) { v_vk_command_pool_cleanup(device, device->compute_queue.cmd_pool); }
  if (device->transfer_queue.cmd_pool.cmd_buffer_idx >= cleanup_frequency) { v_vk_command_pool_cleanup(device, device->transfer_queue.cmd_pool); }
}

void v_vk_wait_for_fence(vk_backend_ctx* ctx) {
  // Use waitForFences while most of the graph executes. Hopefully the CPU can sleep
  // during this wait.
  if (ctx->almost_ready_fence_pending) {
    VK_CHECK(ctx->device->device.waitForFences({ ctx->almost_ready_fence }, true, UINT64_MAX), "almost_ready_fence");
    ctx->device->device.resetFences({ctx->almost_ready_fence});
    ctx->almost_ready_fence_pending = false;
  }

  // Spin (w/pause) waiting for the graph to finish executing.
  vk::Result result;
  while ((result = ctx->device->device.getFenceStatus(ctx->fence)) != vk::Result::eSuccess) {
    if (result != vk::Result::eNotReady) {
      fprintf(stderr, "v_vulkan: error %s at %s:%d\n", to_string(result).c_str(), __FILE__, __LINE__);
      exit(1);
    }
    for (uint32_t i = 0; i < 100; ++i) {
      YIELD();
      YIELD();
      YIELD();
      YIELD();
      YIELD();
      YIELD();
      YIELD();
      YIELD();
      YIELD();
      YIELD();
    }
  }
  ctx->device->device.resetFences({ctx->fence});
}


vk_summition vk_begin_sub_mission(vk_device& device,
                                  vk_command_pool& p,
                                  bool one_time) {
  vk_summition s;
  s.buffer = v_vk_create_cmd_buffer(device, p);
  if (one_time) {
    s.buffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  }
  else {
    s.buffer.begin({vk::CommandBufferUsageFlags{}});
  }

  return s;
}

void vk_queue::copyFrom(vk_queue& other) {
  queue_family_index = other.queue_family_index;
  queue              = other.queue;
  stage_flags        = other.stage_flags;
  transfer_only      = other.transfer_only;
}

void v_vk_end_submission(vk_summition& s,
                         std::vector<vk_semaphore> wait_semaphores,
                         std::vector<vk_semaphore> signal_semaphores) {
  s.buffer.end();
  s.wait_semaphores   = std::move(wait_semaphores);
  s.signal_semaphores = std::move(signal_semaphores);
}

uint32_t v_vk_find_queue_family_index(std::vector<vk::QueueFamilyProperties>& queue_family_props,
                                      const vk::QueueFlags& required,
                                      const vk::QueueFlags& avoid,
                                      int32_t compute_index,
                                      uint32_t min_num_queues) {
  VK_LOG_DEBUG("v_vk_find_queue_family_index()");
  const uint32_t qfsize = queue_family_props.size();

  // Try with avoid preferences first
  for (uint32_t i = 0; i < qfsize; i++) {
    if (queue_family_props[i].queueCount >= min_num_queues && (compute_index < 0 || i != (uint32_t)compute_index) &&
      queue_family_props[i].queueFlags & required && !(queue_family_props[i].queueFlags & avoid)) {
      return i;
    }
  }

  // Fall back to only required
  for (size_t i = 0; i < qfsize; i++) {
    if (queue_family_props[i].queueCount >= min_num_queues && (compute_index < 0 || i != (uint32_t)compute_index) &&
      queue_family_props[i].queueFlags & required) {
      return i;
    }
  }

  // Fall back to reusing compute queue
  for (size_t i = 0; i < qfsize; i++) {
    if (queue_family_props[i].queueCount >= min_num_queues && queue_family_props[i].queueFlags & required) {
      return i;
    }
  }

  // Fall back to ignoring min_num_queries
  for (size_t i = 0; i < qfsize; i++) {
    if (queue_family_props[i].queueFlags & required) {
      return i;
    }
  }

  /// All commands that are allowed on a queue that supports transfer operations are also allowed on a queue that supports either graphics or compute operations.
  /// Thus, if the capabilities of a queue family include VK_QUEUE_GRAPHICS_BIT or VK_QUEUE_COMPUTE_BIT, then reporting the VK_QUEUE_TRANSFER_BIT capability separately for that queue family is optional.
  if (compute_index >= 0) {
    return compute_index;
  }

  std::cerr << "v_vulkan: No suitable queue family index found." << std::endl;

  for (auto& q_family : queue_family_props) {
    std::cerr << "Queue number: " + std::to_string(q_family.queueCount) << " flags: " + to_string(q_family.queueFlags)
      << std::endl;
  }
  abort();
}

void vk_command_pool::destroy(vk::Device& device) {
  device.destroyCommandPool(pool);
  pool = nullptr;
  cmd_buffers.clear();
}

vk::CommandBuffer v_vk_create_cmd_buffer(vk_device& device, vk_command_pool& p) {
  VK_LOG_DEBUG("v_vk_create_cmd_buffer()");

  if (p.cmd_buffers.size() > p.cmd_buffer_idx) {
    // Reuse command buffer
    return p.cmd_buffers[p.cmd_buffer_idx++];
  }

  vk::CommandBufferAllocateInfo command_buffer_alloc_info(
    p.pool,
    vk::CommandBufferLevel::ePrimary,
    1);
  const std::vector<vk::CommandBuffer> cmd_buffers = device->device.allocateCommandBuffers(command_buffer_alloc_info);
  auto buf                                         = cmd_buffers.front();

  p.cmd_buffers.push_back(buf);
  p.cmd_buffer_idx++;

  return buf;
}
