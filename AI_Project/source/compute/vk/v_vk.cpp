#include "vk_common.h"
#include "vk_pipeline.h"
#include "vk_queue.h"
#include "vk_device.h"
#include "vk_constant.h"
#include "vk_buffer.h"
#include "vk_context.h"
#include "vk_util.h"
#include "v_util.h"


bool vk_instance_initialized = false;
vk_instance_struct vk_instance;
bool vk_perf_logger_enabled = false;


struct vk_backend_ctx;
void vk_destory_pipeline(vk::Device& device, vk_pipeline& pipeline);

std::mutex queue_mutex;
const char* vk_device_buffer_name(v_backend_buffer_type_t buft);
v_backend_buffer_t vk_device_buffer_alloc(v_backend_buffer_type_t buft, size_t size);
size_t vk_device_buffer_get_align(v_backend_buffer_type_t buft);
size_t vk_device_buffer_get_max_size(v_backend_buffer_type_t buft);
size_t vk_device_buffer_get_alloc_size(v_backend_buffer_type_t buft, const v_tensor* tensor);

//#define v_VULKAN_MEMORY_DEBUG
#ifdef v_VULKAN_MEMORY_DEBUG
class vk_memory_logger;
#endif
class vk_perf_logger;


vk_op_unary_push_constants vk_op_unary_push_constants_init(const v_tensor* src0, const v_tensor* dst,
                                                           int64_t ne = 0) {
  V_ASSERT(ne != 0 || (nelements(src0) == nelements(dst)));
  ne = ne != 0
         ? ne
         : nelements(dst);
  V_ASSERT(ne <= (int64_t)std::numeric_limits<uint32_t>::max());

  vk_op_unary_push_constants p{};
  p.ne = (uint32_t)ne;

  size_t src0_tsize = v_type_size(src0->type);
  p.ne00            = (uint32_t)src0->ne[0];
  p.ne01            = (uint32_t)src0->ne[1];
  p.ne02            = (uint32_t)src0->ne[2];
  p.ne03            = (uint32_t)src0->ne[3];
  p.nb00            = (uint32_t)(src0->nb[0] / src0_tsize);
  p.nb01            = (uint32_t)(src0->nb[1] / src0_tsize);
  p.nb02            = (uint32_t)(src0->nb[2] / src0_tsize);
  p.nb03            = (uint32_t)(src0->nb[3] / src0_tsize);

  size_t dst_tsize = v_type_size(dst->type);
  p.ne10           = (uint32_t)dst->ne[0];
  p.ne11           = (uint32_t)dst->ne[1];
  p.ne12           = (uint32_t)dst->ne[2];
  p.ne13           = (uint32_t)dst->ne[3];
  p.nb10           = (uint32_t)(dst->nb[0] / dst_tsize);
  p.nb11           = (uint32_t)(dst->nb[1] / dst_tsize);
  p.nb12           = (uint32_t)(dst->nb[2] / dst_tsize);
  p.nb13           = (uint32_t)(dst->nb[3] / dst_tsize);

  return p; // offsets are initialized later in v_vk_op
}


vk_op_pad_push_constants vk_op_pad_push_constants_init(const v_tensor* src0, const v_tensor* dst) {
  int64_t ne = nelements(dst);
  V_ASSERT(ne <= (int64_t)std::numeric_limits<uint32_t>::max());

  vk_op_pad_push_constants p{};
  p.ne = (uint32_t)ne;

  size_t src0_tsize = v_type_size(src0->type);
  p.ne00            = (uint32_t)src0->ne[0];
  p.ne01            = (uint32_t)src0->ne[1];
  p.ne02            = (uint32_t)src0->ne[2];
  p.ne03            = (uint32_t)src0->ne[3];
  p.nb00            = (uint32_t)(src0->nb[0] / src0_tsize);
  p.nb01            = (uint32_t)(src0->nb[1] / src0_tsize);
  p.nb02            = (uint32_t)(src0->nb[2] / src0_tsize);
  p.nb03            = (uint32_t)(src0->nb[3] / src0_tsize);

  size_t dst_tsize = v_type_size(dst->type);
  p.ne10           = (uint32_t)dst->ne[0];
  p.ne11           = (uint32_t)dst->ne[1];
  p.ne12           = (uint32_t)dst->ne[2];
  p.ne13           = (uint32_t)dst->ne[3];
  p.nb10           = (uint32_t)(dst->nb[0] / dst_tsize);
  p.nb11           = (uint32_t)(dst->nb[1] / dst_tsize);
  p.nb12           = (uint32_t)(dst->nb[2] / dst_tsize);
  p.nb13           = (uint32_t)(dst->nb[3] / dst_tsize);

  p.lp0 = dst->op_params[0];
  p.rp0 = dst->op_params[1];
  p.lp1 = dst->op_params[2];
  p.rp1 = dst->op_params[3];
  p.lp2 = dst->op_params[4];
  p.rp2 = dst->op_params[5];
  p.lp3 = dst->op_params[6];
  p.rp3 = dst->op_params[7];

  return p; // fastdiv values and offsets are initialized later in v_vk_op
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

template <typename T>
void init_pushconst_fastdiv(T& p) {
  v_UNUSED(p);
  static_assert(!std::is_const<T>::value, "unexpected type");
}

template <>
void init_pushconst_fastdiv(vk_op_unary_push_constants& p) {
  // Compute magic values to divide by these six numbers.
  init_fastdiv_values(p.ne02 * p.ne01 * p.ne00, p.ne0_012mp, p.ne0_012L);
  init_fastdiv_values(p.ne01 * p.ne00, p.ne0_01mp, p.ne0_01L);
  init_fastdiv_values(p.ne00, p.ne0_0mp, p.ne0_0L);
  init_fastdiv_values(p.ne12 * p.ne11 * p.ne10, p.ne1_012mp, p.ne1_012L);
  init_fastdiv_values(p.ne11 * p.ne10, p.ne1_01mp, p.ne1_01L);
  init_fastdiv_values(p.ne10, p.ne1_0mp, p.ne1_0L);
}

template <>
void init_pushconst_fastdiv(vk_op_conv2d_push_constants& p) {
  // Compute magic values to divide by KW, KW*KH, OW, OW*OH
  init_fastdiv_values(p.KW, p.KWmp, p.KWL);
  init_fastdiv_values(p.KW * p.KH, p.KWKHmp, p.KWKHL);
  init_fastdiv_values(p.OW, p.OWmp, p.OWL);
  init_fastdiv_values(p.OW * p.OH, p.OWOHmp, p.OWOHL);
}

template <>
void init_pushconst_fastdiv(vk_op_conv_transpose_2d_push_constants& p) {
  // Compute magic values to divide by KW, KW*KH, OW, OW*OH, s0, s1
  init_fastdiv_values(p.KW, p.KWmp, p.KWL);
  init_fastdiv_values(p.KW * p.KH, p.KWKHmp, p.KWKHL);
  init_fastdiv_values(p.OW, p.OWmp, p.OWL);
  init_fastdiv_values(p.OW * p.OH, p.OWOHmp, p.OWOHL);
  init_fastdiv_values(p.s0, p.s0mp, p.s0L);
  init_fastdiv_values(p.s1, p.s1mp, p.s1L);
}

vk_op_sum_rows_push_constants vk_op_sum_rows_push_constants_init(const v_tensor* src, const v_tensor* dst,
                                                                 int64_t n_cols) {
  uint32_t type_size              = (uint32_t)v_type_size(src->type);
  vk_op_sum_rows_push_constants p = {};
  p.n_cols                        = (uint32_t)n_cols;
  p.ne01                          = (uint32_t)src->ne[1];
  p.ne02                          = (uint32_t)src->ne[2];
  p.nb01                          = (uint32_t)src->nb[1] / type_size;
  p.nb02                          = (uint32_t)src->nb[2] / type_size;
  p.nb03                          = (uint32_t)src->nb[3] / type_size;
  p.nb11                          = (uint32_t)dst->nb[1] / type_size;
  p.nb12                          = (uint32_t)dst->nb[2] / type_size;
  p.nb13                          = (uint32_t)dst->nb[3] / type_size;
  p.weight                        = 1.0f;
  return p;
}

template <>
void init_pushconst_fastdiv(vk_op_sum_rows_push_constants& p) {
  init_fastdiv_values(p.ne01 * p.ne02, p.ne0_12mp, p.ne0_12L);
  init_fastdiv_values(p.ne01, p.ne0_1mp, p.ne0_1L);
}


void* const vk_ptr_base = (void*)(uintptr_t)0x1000; // NOLINT

uint64_t vk_tensor_offset(const v_tensor* tensor) {
  if (tensor->view_src) { return (uint8_t*)tensor->view_src->data - (uint8_t*)vk_ptr_base; }
  return (uint8_t*)tensor->data - (uint8_t*)vk_ptr_base;
}
#ifdef v_VULKAN_MEMORY_DEBUG
std::mutex log_mutex;

void vk_memory_logger::log_allocation(vk_buffer_ref buf_ref, size_t size) {
  std::lock_guard<std::mutex> guard(log_mutex);
  vk_buffer buf          = buf_ref.lock();
  const bool device      = bool(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eDeviceLocal);
  const std::string type = device
                             ? "device"
                             : "host";
  allocations[buf->buffer] = size;
  total_device += device
                    ? size
                    : 0;
  total_host += device
                  ? 0
                  : size;
  VK_LOG_MEMORY(
    buf->device->name << ": +" << format_size(size) << " " << type << " at " << buf->buffer << ". Total device: " <<
    format_size(total_device) << ", total host: " << format_size(total_host));
}

void vk_memory_logger::log_deallocation(vk_buffer_ref buf_ref) {
  if (buf_ref.expired() || buf_ref.lock()->size == 0) { return; }

  std::lock_guard<std::mutex> guard(log_mutex);
  vk_buffer buf     = buf_ref.lock();
  const bool device = bool(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eDeviceLocal);
  std::string type  = device
                        ? "device"
                        : "host";
  auto it = allocations.find(buf->buffer);
  total_device -= device
                    ? it->second
                    : 0;
  total_host -= device
                  ? 0
                  : it->second;
  if (it != allocations.end()) {
    VK_LOG_MEMORY(
      buf->device->name << ": -" << format_size(it->second) << " " << type << " at " << buf->buffer <<
      ". Total device: " << format_size(total_device) << ", total host: " << format_size(total_host));
    allocations.erase(it);
  }
  else {
    VK_LOG_MEMORY(
      "ERROR " << buf->device->name << ": Attempted to deallocate unknown " << type << " memory at " << buf->buffer);
  }
}
#endif // v_VULKAN_MEMORY_DEBUG
//#define v_VULKAN_CHECK_RESULTS
#ifdef v_VULKAN_CHECK_RESULTS
size_t vk_skip_checks;
size_t vk_output_tensor;

void vk_print_tensor(const v_tensor* tensor, const char* name);
void vk_check_results_0(vk_backend_ctx* ctx, v_cgraph* cgraph, int tensor_idx);
void vk_check_results_1(vk_backend_ctx* ctx, v_cgraph* cgraph, int tensor_idx);
#endif

typedef void (*v_vk_func_t)(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                            const v_tensor* src1, v_tensor* dst);

void vk_backend_free(v_backend_t backend);

VkDeviceSize v_vk_get_max_buffer_range(const vk_backend_ctx* ctx, const vk_buffer& buf,
                                       const VkDeviceSize offset) {
  const VkDeviceSize range = std::min(VkDeviceSize{buf->size - offset},
                                      VkDeviceSize{ctx->device->properties.limits.maxStorageBufferRange});
  return range;
}

// Wait for ctx->fence to be signaled.
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

// variables to track number of compiles in progress
uint32_t compile_count = 0;
std::mutex compile_count_mutex;
std::condition_variable compile_count_cond;


void v_pipeline_request_descriptor_sets(vk_backend_ctx* ctx, vk_pipeline& pipeline, uint32_t n) {
  VK_LOG_DEBUG("v_pipeline_request_descriptor_sets(" << pipeline->name << ", " << n << ")");
  ctx->pipeline_descriptor_set_requirements += n;
  if (!pipeline->compiled) {
    pipeline->needed           = true;
    ctx->device->need_compiles = true;
  }
}

void vk_pipeline_allocate_descriptor_sets(vk_backend_ctx* ctx) {
  if (ctx->descriptor_sets.size() >= ctx->pipeline_descriptor_set_requirements) {
    // Enough descriptors are available
    return;
  }

  vk_device& device = ctx->device;

  uint32_t to_alloc       = ctx->pipeline_descriptor_set_requirements - ctx->descriptor_sets.size();
  uint32_t pool_remaining = VK_DEVICE_DESCRIPTOR_POOL_SIZE - ctx->descriptor_sets.size() %
    VK_DEVICE_DESCRIPTOR_POOL_SIZE;
  uint32_t pool_idx = ctx->descriptor_sets.size() / VK_DEVICE_DESCRIPTOR_POOL_SIZE;

  while (to_alloc > 0) {
    const uint32_t alloc_count = std::min(pool_remaining, to_alloc);
    to_alloc -= alloc_count;
    pool_remaining = VK_DEVICE_DESCRIPTOR_POOL_SIZE;

    if (pool_idx >= ctx->descriptor_pools.size()) {
      vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer,
                                                  MAX_PARAMETER_COUNT * VK_DEVICE_DESCRIPTOR_POOL_SIZE);
      vk::DescriptorPoolCreateInfo
        descriptor_pool_create_info({}, VK_DEVICE_DESCRIPTOR_POOL_SIZE, descriptor_pool_size);
      ctx->descriptor_pools.push_back(device->device.createDescriptorPool(descriptor_pool_create_info));
    }

    std::vector<vk::DescriptorSetLayout> layouts(alloc_count);
    for (uint32_t i = 0; i < alloc_count; i++) { layouts[i] = device->dsl; }
    vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(ctx->descriptor_pools[pool_idx],
                                                            alloc_count,
                                                            layouts.data());
    std::vector<vk::DescriptorSet> sets = device->device.allocateDescriptorSets(descriptor_set_alloc_info);
    ctx->descriptor_sets.insert(ctx->descriptor_sets.end(), sets.begin(), sets.end());

    pool_idx++;
  }
}

vk::CommandBuffer v_vk_create_cmd_buffer(vk_device& device, MmlCommandPool& p) {
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


vk_context vk_create_context(vk_backend_ctx* ctx, MmlCommandPool& p) {
  vk_context result = std::make_shared<vk_context_struct>();
  VK_LOG_DEBUG("v_vk_create_context(" << result << ")");
  ctx->gc.contexts.emplace_back(result);
  result->p = &p;
  return result;
}


MmlVkSemaphore* v_vk_create_timeline_semaphore(vk_backend_ctx* ctx) {
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

void v_vk_command_pool_cleanup(vk_device& device, MmlCommandPool& p) {
  VK_LOG_DEBUG("v_vk_command_pool_cleanup()");

  // Requires command buffers to be done
  device->device.resetCommandPool(p.pool);
  p.cmd_buffer_idx = 0;
}

void mmlVkQueueCommandPoolsCleanUp(vk_device& device) {
  VK_LOG_DEBUG("v_vk_queue_command_pools_cleanup()");

  // Arbitrary frequency to cleanup/reuse command buffers
  constexpr uint32_t cleanup_frequency = 10;

  if (device->compute_queue.cmd_pool.cmd_buffer_idx >= cleanup_frequency) { v_vk_command_pool_cleanup(device, device->compute_queue.cmd_pool); }
  if (device->transfer_queue.cmd_pool.cmd_buffer_idx >= cleanup_frequency) { v_vk_command_pool_cleanup(device, device->transfer_queue.cmd_pool); }
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


vk_sub_buffer v_vk_subbuffer(const vk_backend_ctx* ctx,
                             const vk_buffer& buf,
                             size_t offset = 0) { return {buf, offset, v_vk_get_max_buffer_range(ctx, buf, offset)}; }


void v_vk_wait_events(vk_context& ctx, std::vector<vk::Event>&& events) {
  VK_LOG_DEBUG("v_vk_wait_events()");
  if (events.empty()) { return; }

  ctx->s->buffer.waitEvents(
    events,
    ctx->p->q->stage_flags,
    ctx->p->q->stage_flags,
    {},
    {},
    {}
  );
}


bool vk_khr_cooperative_matrix_support(const vk::PhysicalDeviceProperties& props,
                                       const vk::PhysicalDeviceDriverProperties& driver_props,
                                       vk_device_architecture arch);


vk_matmul_pipeline v_vk_get_mul_mat_mat_pipeline(vk_backend_ctx* ctx, v_data_type src0_type,
                                                 v_data_type src1_type, v_prec prec) {
  VK_LOG_DEBUG(
    "v_vk_get_mul_mat_mat_pipeline(" << v_type_name(src0_type) << ", " << v_type_name(src1_type) << ", " <<
    prec << ")");
  if (src0_type == v_TYPE_F32 && src1_type == v_TYPE_F32) { return ctx->device->pipeline_matmul_f32; }
  if (src0_type == v_TYPE_F32 && src1_type == v_TYPE_F16) { return ctx->device->pipeline_matmul_f32_f16; }
  if (src0_type == v_TYPE_BF16 && src1_type == v_TYPE_BF16) { return ctx->device->pipeline_matmul_bf16; }
  if (prec == v_PREC_DEFAULT && ctx->device->fp16 && !(ctx->device->coopmat_support && !ctx->device->
                                                                                             coopmat_acc_f16_support)) {
    if (src0_type == v_TYPE_F16 && src1_type == v_TYPE_F32) { return ctx->device->pipeline_matmul_f16_f32.f16acc; }
    if (src0_type == v_TYPE_F16 && src1_type == v_TYPE_F16) { return ctx->device->pipeline_matmul_f16.f16acc; }
  }
  else {
    if (src0_type == v_TYPE_F16 && src1_type == v_TYPE_F32) { return ctx->device->pipeline_matmul_f16_f32.f32acc; }
    if (src0_type == v_TYPE_F16 && src1_type == v_TYPE_F16) { return ctx->device->pipeline_matmul_f16.f32acc; }
  }

  // MMQ
  if (src1_type == v_TYPE_Q8_1) {
    vk_matmul_pipeline pipelines = (ctx->device->fp16 && prec == v_PREC_DEFAULT)
                                     ? ctx->device->pipeline_dequant_mul_mat_mat_q8_1[src0_type].f16acc
                                     : ctx->device->pipeline_dequant_mul_mat_mat_q8_1[src0_type].f32acc;

    if (pipelines->s == nullptr && pipelines->m == nullptr && pipelines->l == nullptr) { return nullptr; }

    return pipelines;
  }

  if (src1_type != v_TYPE_F32 && !ctx->device->coopmat2) { return nullptr; }

  switch (src0_type) {
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

  if (ctx->device->coopmat2) {
    assert(src1_type == v_TYPE_F16);
    return prec == v_PREC_DEFAULT
             ? ctx->device->pipeline_dequant_mul_mat_mat_f16[src0_type].f16acc
             : ctx->device->pipeline_dequant_mul_mat_mat_f16[src0_type].f32acc;
  }
  if (ctx->device->coopmat_support) {
    return (ctx->device->fp16 && ctx->device->coopmat_acc_f16_support && prec == v_PREC_DEFAULT)
             ? ctx->device->pipeline_dequant_mul_mat_mat[src0_type].f16acc
             : ctx->device->pipeline_dequant_mul_mat_mat[src0_type].f32acc;
  }
  return (ctx->device->fp16 && prec == v_PREC_DEFAULT)
           ? ctx->device->pipeline_dequant_mul_mat_mat[src0_type].f16acc
           : ctx->device->pipeline_dequant_mul_mat_mat[src0_type].f32acc;
}

vk_pipeline v_vk_get_dequantize_mul_mat_vec(vk_backend_ctx* ctx, v_data_type a_type,
                                            v_data_type b_type,
                                            uint32_t num_cols, uint32_t m, uint32_t k) {
  VK_LOG_DEBUG("v_vk_get_dequantize_mul_mat_vec()");
  V_ASSERT(b_type == v_TYPE_F32 || b_type == v_TYPE_F16 || b_type == v_TYPE_Q8_1);
  V_ASSERT(num_cols >= 1 && num_cols <= mul_mat_vec_max_cols);

  if (b_type == v_TYPE_Q8_1) {
    switch (a_type) {
      case v_TYPE_Q4_0:
      case v_TYPE_Q4_1:
      case v_TYPE_Q5_0:
      case v_TYPE_Q5_1:
      case v_TYPE_Q8_0:
        break;
      default:
        return nullptr;
    }
  }

  switch (a_type) {
    case v_TYPE_F32:
    case v_TYPE_F16:
    case v_TYPE_BF16:
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

  // heuristic to choose workgroup size
  uint32_t dmmv_wg = DMMV_WG_SIZE_SUBGROUP;
  if ((ctx->device->vendor_id == VK_VENDOR_ID_NVIDIA && ctx->device->architecture !=
    vk_device_architecture::NVIDIA_PRE_TURING) || ctx->device->vendor_id == VK_VENDOR_ID_INTEL) {
    // Prefer larger workgroups when M is small, to spread the work out more
    // and keep more SMs busy.
    // q6_k seems to prefer small workgroup size even for "medium" values of M.
    if (a_type == v_TYPE_Q6_K) { if (m < 4096 && k >= 1024) { dmmv_wg = DMMV_WG_SIZE_LARGE; } }
    else { if (m <= 8192 && k >= 1024) { dmmv_wg = DMMV_WG_SIZE_LARGE; } }
  }

  if (b_type == v_TYPE_Q8_1) {
    if (ctx->device->vendor_id == VK_VENDOR_ID_INTEL) { dmmv_wg = DMMV_WG_SIZE_SUBGROUP; }
    return ctx->device->pipeline_dequant_mul_mat_vec_q8_1_f32[dmmv_wg][a_type][num_cols - 1];
  }

  return b_type == v_TYPE_F32
           ? ctx->device->pipeline_dequant_mul_mat_vec_f32_f32[dmmv_wg][a_type][num_cols - 1]
           : ctx->device->pipeline_dequant_mul_mat_vec_f16_f32[dmmv_wg][a_type][num_cols - 1];
}

vk_matmul_pipeline v_vk_get_mul_mat_mat_id_pipeline(vk_backend_ctx* ctx, v_data_type src0_type,
                                                    v_data_type src1_type, v_prec prec) {
  VK_LOG_DEBUG("v_vk_get_mul_mat_mat_id_pipeline()");
  if (src0_type == v_TYPE_F32 && src1_type == v_TYPE_F32) { return ctx->device->pipeline_matmul_id_f32; }
  if (src0_type == v_TYPE_BF16 && src1_type == v_TYPE_BF16) { return ctx->device->pipeline_matmul_id_bf16; }
  if (prec == v_PREC_DEFAULT && ctx->device->fp16 && !(ctx->device->coopmat_support && !ctx->device->
                                                                                             coopmat_acc_f16_support)) {
    if (src0_type == v_TYPE_F16 && src1_type == v_TYPE_F32) { return ctx->device->pipeline_matmul_id_f16_f32.f16acc; }
    if (src0_type == v_TYPE_F16 && src1_type == v_TYPE_F16) { return ctx->device->pipeline_matmul_id_f16.f16acc; }
  }
  else {
    if (src0_type == v_TYPE_F16 && src1_type == v_TYPE_F32) { return ctx->device->pipeline_matmul_id_f16_f32.f32acc; }
    if (src0_type == v_TYPE_F16 && src1_type == v_TYPE_F16) { return ctx->device->pipeline_matmul_id_f16.f32acc; }
  }

  V_ASSERT(src1_type == v_TYPE_F32 || (ctx->device->coopmat2 && src1_type == v_TYPE_F16));

  switch (src0_type) {
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

  // XXX TODO 'prec' is not actually allowed in mul_mat_id.
  bool prefer_fp16acc  = ctx->device->fp16 /*&& prec == v_PREC_DEFAULT*/;
  bool support_fp16acc = ctx->device->pipeline_dequant_mul_mat_mat_id[src0_type].f16acc != nullptr;
  bool support_fp32acc = ctx->device->pipeline_dequant_mul_mat_mat_id[src0_type].f32acc != nullptr;

  if (support_fp16acc && (prefer_fp16acc || !support_fp32acc)) { return ctx->device->pipeline_dequant_mul_mat_mat_id[src0_type].f16acc; }
  else {
    V_ASSERT(support_fp32acc);
    return ctx->device->pipeline_dequant_mul_mat_mat_id[src0_type].f32acc;
  }
}

vk_pipeline v_vk_get_dequantize_mul_mat_vec_id(vk_backend_ctx* ctx, v_data_type a_type,
                                               v_data_type b_type) {
  VK_LOG_DEBUG("v_vk_get_dequantize_mul_mat_vec_id()");
  V_ASSERT(b_type == v_TYPE_F32);

  switch (a_type) {
    case v_TYPE_F32:
    case v_TYPE_F16:
    case v_TYPE_BF16:
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

  return ctx->device->pipeline_dequant_mul_mat_vec_id_f32[a_type];
}

void* vkHostMalloc(vk_device& device,
                   size_t size) {
  VK_LOG_MEMORY("v_vk_host_malloc(" << size << ")");
  vk_buffer buf = vk_create_buffer(device,
                                   size,
                                   {
                                     vk::MemoryPropertyFlagBits::eHostVisible |
                                     vk::MemoryPropertyFlagBits::eHostCoherent |
                                     vk::MemoryPropertyFlagBits::eHostCached,
                                     vk::MemoryPropertyFlagBits::eHostVisible |
                                     vk::MemoryPropertyFlagBits::eHostCoherent
                                   });

  if (!(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible)) {
    fprintf(stderr,
            "WARNING: failed to allocate %.2f MB of pinned memory\n",
            size / 1024.0 / 1024.0);
    device->device.freeMemory(buf->device_memory);
    device->device.destroyBuffer(buf->buffer);
    return nullptr;
  }
  std::lock_guard<std::recursive_mutex> guard(device->mutex);
  device->pinned_memory.push_back(std::make_tuple(buf->ptr, size, buf));
  return buf->ptr;
}

void vk_host_free(vk_device& device, void* ptr) {
  if (ptr == nullptr) { return; }
  VK_LOG_MEMORY("v_vk_host_free(" << ptr << ")");
  std::lock_guard<std::recursive_mutex> guard(device->mutex);

  vk_buffer buf;
  size_t index;
  for (size_t i = 0; i < device->pinned_memory.size(); i++) {
    const uint8_t* addr = (const uint8_t*)std::get<0>(device->pinned_memory[i]);
    const uint8_t* endr = addr + std::get<1>(device->pinned_memory[i]);
    if (ptr >= addr && ptr < endr) {
      buf   = std::get<2>(device->pinned_memory[i]);
      index = i;
      break;
    }
  }
  if (buf == nullptr) {
    fprintf(stderr, "WARNING: failed to free pinned memory: memory not in map\n");
    return;
  }

  vk_destroy_buffer(buf);

  device->pinned_memory.erase(device->pinned_memory.begin() + index);
}


template <typename T>
size_t push_constant_size(const T& t) {
  static_assert(std::is_class<T>::value, "T must be a struct/class");
  v_UNUSED(t);
  return sizeof(T);
}

template <typename T>
size_t push_constant_size(const std::vector<T>& t) {
  v_UNUSED(t);
  return sizeof(T) * t.size();
}

template <typename T, uint32_t N>
size_t push_constant_size(const std::array<T, N>& t) {
  v_UNUSED(t);
  return sizeof(T) * N;
}

template <typename T>
const T* push_constant_data(const T& t) {
  static_assert(std::is_class<T>::value, "T must be a struct/class");
  return &t;
}

template <typename T>
const T* push_constant_data(const std::vector<T>& t) { return t.data(); }

template <typename T, uint32_t N>
const T* push_constant_data(const std::array<T, N>& t) { return t.data(); }

template <typename T>
void v_vk_dispatch_pipeline(vk_backend_ctx* ctx, vk_context& subctx, vk_pipeline& pipeline,
                            std::initializer_list<vk::DescriptorBufferInfo> const& descriptor_buffer_infos,
                            const T& push_constants, std::array<uint32_t, 3> elements) {
  const uint32_t wg0 = CEIL_DIV(elements[0], pipeline->wg_denoms[0]);
  const uint32_t wg1 = CEIL_DIV(elements[1], pipeline->wg_denoms[1]);
  const uint32_t wg2 = CEIL_DIV(elements[2], pipeline->wg_denoms[2]);
  VK_LOG_DEBUG("v_vk_dispatch_pipeline(" << pipeline->name << ", {";
    for (auto& buffer : descriptor_buffer_infos) {
    std::cerr << "(" << buffer.buffer << ", " << buffer.offset << ", " << buffer.range << "), ";
    }
    std::cerr << "}, (" << wg0 << "," << wg1 << "," << wg2 << "))");
  V_ASSERT(ctx->descriptor_set_idx < ctx->descriptor_sets.size());
  V_ASSERT(descriptor_buffer_infos.size() <= MAX_PARAMETER_COUNT);
  V_ASSERT(pipeline->parameter_count == descriptor_buffer_infos.size());

  vk::DescriptorSet& descriptor_set = ctx->descriptor_sets[ctx->descriptor_set_idx++];
  vk::WriteDescriptorSet write_descriptor_set{
    descriptor_set, 0, 0, pipeline->parameter_count, vk::DescriptorType::eStorageBuffer, nullptr,
    descriptor_buffer_infos.begin()
  };
  ctx->device->device.updateDescriptorSets({write_descriptor_set}, {});

  subctx->s->buffer.pushConstants(pipeline->layout,
                                  vk::ShaderStageFlagBits::eCompute,
                                  0,
                                  push_constant_size(push_constants),
                                  push_constant_data(push_constants));
  subctx->s->buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
  subctx->s->buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                       pipeline->layout,
                                       0,
                                       {descriptor_set},
                                       {});
  subctx->s->buffer.dispatch(wg0, wg1, wg2);
}


void v_vk_buffer_write_nc_async(vk_backend_ctx* ctx, vk_context& subctx, vk_buffer& dst,
                                size_t offset, const v_tensor* tensor, bool sync_staging = false) {
  VK_LOG_DEBUG("v_vk_buffer_write_nc_async(" << tensor << ")");
  V_ASSERT(!v_is_contiguous(tensor));
  // Buffer is already mapped
  if (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
    std::cerr << "v_vulkan: buffer_write_nc_async dst buffer is host_visible. Use synchronous write." << std::endl;
    v_ABORT("fatal error");
  }
  // Check if src is pinned memory
  vk_buffer buf     = nullptr;
  size_t buf_offset = 0;
  vk_get_host_buffer(ctx->device, tensor->data, buf, buf_offset);

  const uint64_t ne0     = tensor->ne[0];
  const uint64_t ne1     = tensor->ne[1];
  const uint64_t ne2     = tensor->ne[2];
  const uint64_t ne3     = tensor->ne[3];
  const uint64_t nb0     = tensor->nb[0];
  const uint64_t nb1     = tensor->nb[1];
  const uint64_t nb2     = tensor->nb[2];
  const uint64_t nb3     = tensor->nb[3];
  const v_data_type type = tensor->type;
  const uint64_t ts      = v_type_size(type);
  const uint64_t bs      = blockSize(type);

  const uint64_t dstnb0 = ts;
  const uint64_t dstnb1 = dstnb0 * (ne0 / bs);
  const uint64_t dstnb2 = dstnb1 * ne1;
  const uint64_t dstnb3 = dstnb2 * ne2;

  const uint64_t ne = nelements(tensor);

  if (buf != nullptr) {
    // Memory is pinned, use as staging buffer
    std::vector<vk::BufferCopy> slices;

    for (uint64_t i3 = 0; i3 < ne3; i3++) {
      for (uint64_t i2 = 0; i2 < ne2; i2++) {
        // Find longest contiguous slice
        if (ne1 * nb1 == dstnb2) { slices.push_back({buf_offset + i3 * nb3 + i2 * nb2, offset + i3 * dstnb3 + i2 * dstnb2, dstnb2}); }
        else {
          for (uint64_t i1 = 0; i1 < ne1; i1++) {
            if (ne0 * nb0 / bs == dstnb1) {
              slices.push_back({
                buf_offset + i3 * nb3 + i2 * nb2 + i1 * nb1, offset + i3 * dstnb3 + i2 * dstnb2 + i1 * dstnb1, dstnb1
              });
            }
            else {
              const uint64_t s_off = buf_offset + i3 * nb3 + i2 * nb2 + i1 * nb1;
              const uint64_t d_off = offset + i3 * dstnb3 + i2 * dstnb2 + i1 * dstnb1;
              for (uint64_t i0 = 0; i0 < ne0; i0++) { slices.push_back({s_off + i1 * nb0, d_off + i0 * dstnb0, dstnb0}); }
            }
          }
        }
      }
    }

    vk_sync_buffers(ctx, subctx);
    subctx->s->buffer.copyBuffer(buf->buffer, dst->buffer, slices);
    return;
  }

  if (!sync_staging) { v_ABORT("Asynchronous write to non-pinned memory not supported"); }

  // Staging buffer required
  vk_buffer& staging       = ctx->device->sync_staging;
  const uint64_t copy_size = ts * ne / bs;
  vk_ensure_sync_staging_buffer(ctx->device, copy_size);
  VkBufferCopy buf_copy{0, offset, copy_size};

  vk_sync_buffers(ctx, subctx);
  vkCmdCopyBuffer(subctx->s->buffer, (VkBuffer)staging->buffer, (VkBuffer)dst->buffer, 1, &buf_copy);

  for (uint64_t i3 = 0; i3 < ne3; i3++) {
    for (uint64_t i2 = 0; i2 < ne2; i2++) {
      // Find longest contiguous slice
      if (ne1 * nb1 == dstnb2) {
        vk_deffered_memcpy((uint8_t*)staging->ptr + i3 * dstnb3 + i2 * dstnb2,
                           (const uint8_t*)tensor->data + buf_offset + i3 * nb3 + i2 * nb2,
                           dstnb2,
                           &subctx->in_memcpys);
      }
      else {
        for (uint64_t i1 = 0; i1 < ne1; i1++) {
          if (ne0 * nb0 / bs == dstnb1) {
            vk_deffered_memcpy((uint8_t*)staging->ptr + i3 * dstnb3 + i2 * dstnb2 + i1 * dstnb1,
                               (const uint8_t*)tensor->data + buf_offset + i3 * nb3 + i2 * nb2 + i1 * nb1,
                               dstnb1,
                               &subctx->in_memcpys);
          }
          else {
            const uint64_t s_off = buf_offset + i3 * nb3 + i2 * nb2 + i1 * nb1;
            const uint64_t d_off = i3 * dstnb3 + i2 * dstnb2 + i1 * dstnb1;
            for (uint64_t i0 = 0; i0 < ne0; i0++) {
              vk_deffered_memcpy((uint8_t*)staging->ptr + d_off + i0 * dstnb0,
                                 (const uint8_t*)tensor->data + s_off + i0 * nb0,
                                 dstnb0,
                                 &subctx->in_memcpys);
            }
          }
        }
      }
    }
  }
}

void v_vk_buffer_write_2d_async(vk_context subctx, vk_buffer& dst, size_t offset, const void* src,
                                size_t spitch, size_t width, size_t height, bool sync_staging = false) {
  VK_LOG_DEBUG("v_vk_buffer_write_2d_async(" << width << ", " << height << ")");
  // Buffer is already mapped
  if (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
    std::cerr << "v_vulkan: buffer_write_async dst buffer is host_visible. Use synchronous write." << std::endl;
    v_ABORT("fatal error");
  }
  // Check if src is pinned memory
  vk_buffer buf     = nullptr;
  size_t buf_offset = 0;
  vk_get_host_buffer(dst->device, src, buf, buf_offset);

  if (buf != nullptr) {
    // Memory is pinned, use as staging buffer
    std::vector<vk::BufferCopy> slices(1);
    if (width == spitch) {
      // Only do single write if stride is equal
      slices[0].srcOffset = buf_offset;
      slices[0].dstOffset = offset;
      slices[0].size      = width * height;
    }
    else {
      slices.resize(height);
      for (size_t i = 0; i < height; i++) {
        slices[i].srcOffset = buf_offset + i * spitch;
        slices[i].dstOffset = offset + i * width;
        slices[i].size      = width;
      }
    }

    vk_sync_buffers(nullptr, subctx);
    subctx->s->buffer.copyBuffer(buf->buffer, dst->buffer, slices);
    return;
  }
  VK_LOG_DEBUG("STAGING");

  if (!sync_staging) { v_ABORT("Asynchronous write to non-pinned memory not supported"); }
  // Staging buffer required
  const size_t copy_size = width * height;
  vk_ensure_sync_staging_buffer(dst->device, copy_size);
  vk_buffer& staging_buffer = dst->device->sync_staging;

  VkBufferCopy buf_copy = {
    0,
    offset,
    copy_size
  };

  vk_sync_buffers(nullptr, subctx);
  vkCmdCopyBuffer(subctx->s->buffer, (VkBuffer)staging_buffer->buffer, (VkBuffer)dst->buffer, 1, &buf_copy);

  if (width == spitch) { vk_deffered_memcpy((uint8_t*)staging_buffer->ptr, src, width * height, &subctx->in_memcpys); }
  else {
    for (size_t i = 0; i < height; i++) {
      vk_deffered_memcpy((uint8_t*)staging_buffer->ptr + i * width,
                         (const uint8_t*)src + i * spitch,
                         width,
                         &subctx->in_memcpys);
    }
  }
}

void v_vk_buffer_write_async(vk_context subctx, vk_buffer& dst, size_t offset, const void* src, size_t size,
                             bool sync_staging = false) {
  VK_LOG_DEBUG("v_vk_buffer_write_async(" << size << ")");
  return v_vk_buffer_write_2d_async(subctx, dst, offset, src, size, size, 1, sync_staging);
}

void v_vk_buffer_write_2d(vk_buffer& dst, size_t offset, const void* src, size_t spitch, size_t width,
                          size_t height) {
  VK_LOG_DEBUG("v_vk_buffer_write_2d(" << width << ", " << height << ")");
  // Buffer is already mapped
  if (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
    V_ASSERT(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

    for (size_t i = 0; i < height; i++) { memcpy((uint8_t*)dst->ptr + offset + i * width, (const uint8_t*)src + i * spitch, width); }
  }
  else {
    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

    vk_context subctx = vk_create_temp_ctx(dst->device->transfer_queue.cmd_pool);
    vk_begin_ctx(dst->device, subctx);
    v_vk_buffer_write_2d_async(subctx, dst, offset, src, spitch, width, height, true);
    vk_ctx_end(subctx);

    for (auto& cpy : subctx->in_memcpys) { memcpy(cpy.dst, cpy.src, cpy.n); }

    for (auto& mset : subctx->memsets) { memset(mset.dst, mset.val, mset.n); }

    vk_submit(subctx, dst->device->fence);
    VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, true, UINT64_MAX),
             "vk_buffer_write_2d waitForFences");
    dst->device->device.resetFences({dst->device->fence});
    mmlVkQueueCommandPoolsCleanUp(dst->device);
  }
}

void v_vk_buffer_write(vk_buffer& dst, size_t offset, const void* src, size_t size) {
  VK_LOG_DEBUG("v_vk_buffer_write(" << size << ")");
  v_vk_buffer_write_2d(dst, offset, src, 0, size, 1);
}


void v_vk_buffer_copy_async(vk_context& ctx, vk_buffer& dst, size_t dst_offset, vk_buffer& src,
                            size_t src_offset, size_t size) {
  VK_LOG_DEBUG("v_vk_buffer_copy_async(" << size << ")");
  // Make sure both buffers are on same device
  V_ASSERT(src->device == dst->device);

  VkBufferCopy bc{src_offset, dst_offset, size};

  vkCmdCopyBuffer(ctx->s->buffer, (VkBuffer)src->buffer, (VkBuffer)dst->buffer, 1, &bc);
}

void v_vk_buffer_copy(vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
  if (src->device == dst->device) {
    std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
    VK_LOG_DEBUG("v_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
    // Copy within the device
    vk_context subctx = vk_create_temp_ctx(src->device->transfer_queue.cmd_pool);
    vk_begin_ctx(src->device, subctx);
    v_vk_buffer_copy_async(subctx, dst, dst_offset, src, src_offset, size);
    vk_ctx_end(subctx);
    vk_submit(subctx, src->device->fence);
    VK_CHECK(src->device->device.waitForFences({ src->device->fence }, true, UINT64_MAX),
             "vk_buffer_copy waitForFences");
    src->device->device.resetFences({src->device->fence});
    mmlVkQueueCommandPoolsCleanUp(src->device);
  }
  else {
    VK_LOG_DEBUG("v_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
    // Copy device to device
    vk_ensure_sync_staging_buffer(src->device, size);

    // Copy to src staging buffer
    v_vk_buffer_copy(src->device->sync_staging, 0, src, src_offset, size);
    // Copy to dst buffer
    v_vk_buffer_write_2d(dst, dst_offset, src->device->sync_staging->ptr, 0, size, 1);
  }
}

void vk_buffer_memset_async(vk_context& ctx, vk_buffer& dst, size_t offset, uint32_t c, size_t size) {
  VK_LOG_DEBUG("v_vk_buffer_memset_async(" << offset << ", " << c << ", " << size << ")");

  if (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
    dst->device->uma) {
    vk_deffered_memset((uint8_t*)dst->ptr + offset, c, size, &ctx->memsets);
    return;
  }

  // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
  ctx->s->buffer.fillBuffer(dst->buffer, offset, size, c);
}

void vk_buffer_memset(vk_buffer& dst, size_t offset, uint32_t c, size_t size) {
  VK_LOG_DEBUG("v_vk_buffer_memset(" << offset << ", " << c << ", " << size << ")");

  if (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
    dst->device->uma) {
    memset((uint8_t*)dst->ptr + offset, c, size);
    return;
  }

  std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);
  vk_context subctx = vk_create_temp_ctx(dst->device->transfer_queue.cmd_pool);
  vk_begin_ctx(dst->device, subctx);
  subctx->s->buffer.fillBuffer(dst->buffer, offset, size, c);
  vk_ctx_end(subctx);

  vk_submit(subctx, dst->device->fence);
  VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, true, UINT64_MAX), "vk_memset waitForFences");
  dst->device->device.resetFences({dst->device->fence});
  mmlVkQueueCommandPoolsCleanUp(dst->device);
}

uint32_t v_vk_guess_split_k(vk_backend_ctx* ctx, uint32_t m, uint32_t n, uint32_t k,
                            bool disable_split_k, const vk_pipeline& pipeline) {
  VK_LOG_DEBUG("v_vk_guess_split_k(" << m << ", " << n << ", " << k << ", " << disable_split_k << ")");

  if (disable_split_k) { return 1; }

  uint32_t split_k = 1;
  if (ctx->device->shader_core_count != 0 && m >= pipeline->wg_denoms[0] && n >= pipeline->wg_denoms[1]) {
    // If k is 'large' and the SMs will fill less than halfway, use split_k.
    uint32_t m_tiles = CEIL_DIV(m, pipeline->wg_denoms[0]);
    uint32_t n_tiles = CEIL_DIV(n, pipeline->wg_denoms[1]);

    if (k >= 2048) {
      if (m_tiles * n_tiles <= ctx->device->shader_core_count / 2) { split_k = ctx->device->shader_core_count / (m_tiles * n_tiles); }
      else if (m_tiles * n_tiles <= ctx->device->shader_core_count * 2 / 3) { split_k = 3; }
      // Cap the split at 8x. Unless k is huge this is a lot of overhead.
      split_k = std::min(split_k, 8u);

      // v_vk_matmul will align the splits to be a multiple of 256.
      // If this rounded up size would cause the last split to be empty,
      // then reduce the split count.
      while (true) {
        if (split_k == 1) { break; }
        uint32_t k_split = CEIL_DIV(k, split_k);
        k_split          = ROUNDUP_POW2(k_split, 256);
        if (k_split * (split_k - 1) < k) { break; }
        split_k--;
      }
    }
  }

  return split_k;
}

vk_pipeline v_vk_guess_matmul_pipeline(vk_backend_ctx* ctx, vk_matmul_pipeline& mmp, uint32_t m,
                                       uint32_t n, bool aligned, v_data_type src0_type, v_data_type src1_type) {
  VK_LOG_DEBUG(
    "v_vk_guess_matmul_pipeline(" << m << ", " << n << ", " << aligned << ", " << v_type_name(src0_type) << ", "
    << v_type_name(src1_type) << ")");

  if (ctx->device->coopmat2) {
    const uint32_t shader_core_count = ctx->device->shader_core_count;
    const uint32_t tiles_l           = CEIL_DIV(m, mmp->a_l->wg_denoms[0]) * CEIL_DIV(n, mmp->a_l->wg_denoms[1]);
    const uint32_t tiles_m           = CEIL_DIV(m, mmp->a_m->wg_denoms[0]) * CEIL_DIV(n, mmp->a_m->wg_denoms[1]);

    // Use large shader when the N dimension is greater than the medium shader's tile size
    uint32_t crossover_large = mmp->m->wg_denoms[1];

    // Prefer large over medium if either:
    // - medium or large tiles would overfill the GPU
    // - large tiles with a split_k==3 fits in the GPU and medium tiles with split_k==2 does not
    //   (medium with split_k==2 is probably better if it fits - more workgroups running and less split_k overhead)
    bool prefer_large = tiles_m > shader_core_count || tiles_l > shader_core_count ||
      // split_k==3 with large tiles likely better than medium tiles with no split_k.
      (tiles_l <= shader_core_count / 3 && tiles_m > shader_core_count / 2);

    if ((ctx->device->mul_mat_l[src0_type] && (n > crossover_large && prefer_large)) || (!ctx->device->mul_mat_m[
        src0_type] && !ctx->device->
                            mul_mat_s
      [src0_type])) {
      return aligned
               ? mmp->a_l
               : mmp->l;
    }
    // Use medium shader when the N dimension is greater than the small shader's tile size
    uint32_t crossover_medium = mmp->s->wg_denoms[1];
    if ((ctx->device->mul_mat_m[src0_type] && (n > crossover_medium)) || !ctx->device->mul_mat_s[src0_type]) {
      return aligned
               ? mmp->a_m
               : mmp->m;
    }
    return aligned
             ? mmp->a_s
             : mmp->s;
  }

  if ((ctx->device->mul_mat_s[src0_type] && (m <= 32 || n <= 32)) || (!ctx->device->mul_mat_m[src0_type] && !ctx->device
                                                                                                                ->
                                                                                                                mul_mat_l
    [src0_type])) {
    return aligned
             ? mmp->a_s
             : mmp->s;
  }
  if ((ctx->device->mul_mat_m[src0_type] && (m <= 64 || n <= 64)) || !ctx->device->mul_mat_l[src0_type]) {
    return aligned
             ? mmp->a_m
             : mmp->m;
  }
  return aligned
           ? mmp->a_l
           : mmp->l;

  v_UNUSED(src1_type);
}

uint32_t v_vk_guess_matmul_pipeline_align(vk_backend_ctx* ctx, vk_matmul_pipeline& mmp, int m, int n,
                                          v_data_type src0_type, v_data_type src1_type) {
  VK_LOG_DEBUG(
    "v_vk_guess_matmul_pipeline_align(" << m << ", " << n << ", " << v_type_name(src0_type) << ", " <<
    v_type_name(src1_type) << ")");
  return v_vk_guess_matmul_pipeline(ctx, mmp, m, n, true, src0_type, src1_type)->align;
}

void v_vk_matmul(vk_backend_ctx* ctx,
                 vk_context& subctx,
                 vk_pipeline& pipeline,
                 vk_sub_buffer&& a,
                 vk_sub_buffer&& b,
                 vk_sub_buffer&& d,
                 vk_sub_buffer&& split_k_buffer,
                 uint32_t m,
                 uint32_t n,
                 uint32_t k,
                 uint32_t stride_a,
                 uint32_t stride_b,
                 uint32_t stride_d,
                 uint32_t batch_stride_a,
                 uint32_t batch_stride_b, uint32_t batch_stride_d,
                 uint32_t split_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2,
                 uint32_t broadcast3,
                 uint32_t padded_n) {
  VK_LOG_DEBUG(
    "v_vk_matmul(a: (" << a.buffer->buffer << ", " << a.offset << ", " << a.size << "), b: (" << b.buffer->buffer <<
    ", " << b.offset << ", " << b.size << "), d: (" << d.buffer->buffer << ", " << d.offset << ", " << d.size <<
    "), split_k: (" << (split_k_buffer.buffer != nullptr ? split_k_buffer.buffer->buffer : VK_NULL_HANDLE) << ", " <<
    split_k_buffer.offset << ", " << split_k_buffer.size << "), m: " << m << ", n: " << n << ", k: " << k <<
    ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", batch_stride_a: " <<
    batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", split_k: "
    << split_k << ", batch: " << batch << ", ne02: " << ne02 << ", ne12: " << ne12 << ", broadcast2: " << broadcast2 <<
    ", broadcast3: " << broadcast3 << ", padded_n: " << padded_n << ")");
  if (split_k == 1) {
    const vk_mat_mat_push_constants pc = {
      m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, k, ne02, ne12, broadcast2,
      broadcast3, padded_n
    };
    v_vk_dispatch_pipeline(ctx, subctx, pipeline, {a, b, d}, pc, {m, n, batch});
    return;
  }

  if (ctx->prealloc_split_k_need_sync) { vk_sync_buffers(ctx, subctx); }

  V_ASSERT(batch_stride_d == m * n);

  // Round the split size up to a multiple of 256 (k-quant alignment)
  uint32_t k_split = CEIL_DIV(k, split_k);
  k_split          = ROUNDUP_POW2(k_split, 256);

  const vk_mat_mat_push_constants pc1 = {
    m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, k_split, ne02, ne12,
    broadcast2, broadcast3, padded_n
  };
  // Make sure enough workgroups get assigned for split k to work
  v_vk_dispatch_pipeline(ctx,
                         subctx,
                         pipeline,
                         {a, b, split_k_buffer},
                         pc1,
                         {
                           (CEIL_DIV(m, pipeline->wg_denoms[0]) * pipeline->wg_denoms[0]) * split_k, n, batch
                         });
  vk_sync_buffers(ctx, subctx);
  const std::array<uint32_t, 2> pc2 = {(uint32_t)(m * n * batch), split_k};
  v_vk_dispatch_pipeline(ctx,
                         subctx,
                         ctx->device->pipeline_matmul_split_k_reduce,
                         {split_k_buffer, d},
                         pc2,
                         {m * n * batch, 1, 1});
  ctx->prealloc_split_k_need_sync = true;
}

vk_pipeline v_vk_guess_matmul_id_pipeline(vk_backend_ctx* ctx, vk_matmul_pipeline& mmp, uint32_t m,
                                          uint32_t n, bool aligned, v_data_type src0_type) {
  VK_LOG_DEBUG(
    "v_vk_guess_matmul_id_pipeline(" << m << ", " << n << ", " << aligned << ", " << v_type_name(src0_type) <<
    ")");

  if (ctx->device->coopmat2) {
    // Use large shader when the N dimension is greater than the medium shader's tile size
    uint32_t crossover_large = mmp->m->wg_denoms[1];
    if ((ctx->device->mul_mat_id_l[src0_type] && (n > crossover_large)) || (!ctx->device->mul_mat_id_m[src0_type] && !
      ctx->device->mul_mat_id_s[src0_type])) {
      return aligned
               ? mmp->a_l
               : mmp->l;
    }
    // Use medium shader when the N dimension is greater than the small shader's tile size
    uint32_t crossover_medium = mmp->s->wg_denoms[1];
    if ((ctx->device->mul_mat_id_m[src0_type] && (n > crossover_medium)) || !ctx->device->mul_mat_id_s[src0_type]) {
      return aligned
               ? mmp->a_m
               : mmp->m;
    }
    return aligned
             ? mmp->a_s
             : mmp->s;
  }

  if ((ctx->device->mul_mat_id_s[src0_type] && (m <= 32 || n <= 32)) || (!ctx->device->mul_mat_id_m[src0_type] && !ctx->
                                                                                                                   device
                                                                                                                   ->
                                                                                                                   mul_mat_id_l
    [src0_type])) {
    return aligned
             ? mmp->a_s
             : mmp->s;
  }
  if ((ctx->device->mul_mat_id_m[src0_type] && (m <= 64 || n <= 64)) || !ctx->device->mul_mat_id_l[src0_type]) {
    return aligned
             ? mmp->a_m
             : mmp->m;
  }
  return aligned
           ? mmp->a_l
           : mmp->l;
}

uint32_t v_vk_guess_matmul_id_pipeline_align(vk_backend_ctx* ctx, vk_matmul_pipeline& mmp, int m,
                                             int n, v_data_type src0_type) {
  VK_LOG_DEBUG("v_vk_guess_matmul_pipeline_align(" << m << ", " << n << ", " << v_type_name(src0_type) << ")");
  return v_vk_guess_matmul_id_pipeline(ctx, mmp, m, n, true, src0_type)->align;
}

void v_vk_matmul_id(
  vk_backend_ctx* ctx, vk_context& subctx, vk_pipeline& pipeline,
  vk_sub_buffer&& a, vk_sub_buffer&& b, vk_sub_buffer&& d, vk_sub_buffer&& ids,
  uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
  uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
  uint32_t n_as, uint32_t nei0, uint32_t nei1, uint32_t nbi1, uint32_t ne11,
  uint32_t padded_n) {
  VK_LOG_DEBUG(
    "v_vk_matmul_id(a: (" << a.buffer->buffer << ", " << a.offset << ", " << a.size << "), b: (" << b.buffer->buffer
    << ", " << b.offset << ", " << b.size << "), d: (" << d.buffer->buffer << ", " << d.offset << ", " << d.size <<
    "), ids: (" << ids.buffer->buffer << ", " << ids.offset << ", " << ids.size << "), " <<
    "m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b <<
    ", stride_d: " << stride_d << ", " <<
    "batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " <<
    batch_stride_d << ", " <<
    "n_as: " << n_as << ", nei0: " << nei0 << ", nei1: " << nei1 << ", nbi1: " << nbi1 << ", ne11: " << ne11 << ")");
  const vk_mat_mat_id_push_constants pc = {
    m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d,
    nei0, nei1, nbi1, ne11, padded_n
  };
  v_vk_dispatch_pipeline(ctx, subctx, pipeline, {a, b, d, ids}, pc, {m, nei1, n_as});
}

bool v_vk_dim01_contiguous(const v_tensor* tensor) {
  return
    tensor->nb[0] == v_type_size(tensor->type) &&
    tensor->nb[1] == (tensor->nb[0] * tensor->ne[0]) / blockSize(tensor->type) &&
    (tensor->ne[3] == 1 || tensor->nb[3] == tensor->nb[2] * tensor->ne[2]);
}

vk_pipeline v_vk_get_cpy_pipeline(vk_backend_ctx* ctx, const v_tensor* src,
                                  const v_tensor* dst, v_data_type to) {
  // Choose "contiguous copy" shader if src/dst are contiguous
  bool contig = v_is_contiguous(src) && (!dst || v_is_contiguous(dst));

  if (src->type == v_TYPE_F32 && to == v_TYPE_F32) {
    if (contig) { return ctx->device->pipeline_contig_cpy_f32_f32; }
    else { return ctx->device->pipeline_cpy_f32_f32; }
  }
  if (src->type == v_TYPE_F32 && to == v_TYPE_F16) {
    if (contig) { return ctx->device->pipeline_contig_cpy_f32_f16; }
    else { return ctx->device->pipeline_cpy_f32_f16; }
  }
  if (src->type == v_TYPE_F16 && to == v_TYPE_F16) {
    if (contig) { return ctx->device->pipeline_contig_cpy_f16_f16; }
    else { return ctx->device->pipeline_cpy_f16_f16; }
  }
  if (src->type == v_TYPE_F16 && to == v_TYPE_F32) {
    if (contig) { return ctx->device->pipeline_contig_cpy_f16_f32; }
    else { return ctx->device->pipeline_cpy_f16_f32; }
  }
  if (src->type == v_TYPE_F32 && to == v_TYPE_BF16) {
    if (contig) { return ctx->device->pipeline_contig_cpy_f32_bf16; }
    else { return ctx->device->pipeline_cpy_f32_bf16; }
  }
  if (src->type == v_TYPE_F32 && to == v_TYPE_I32) {
    if (contig) { return ctx->device->pipeline_contig_cpy_f32_i32; }
    else { return ctx->device->pipeline_cpy_f32_i32; }
  }
  if (src->type == v_TYPE_I32 && to == v_TYPE_F32) {
    if (contig) { return ctx->device->pipeline_contig_cpy_i32_f32; }
    else { return ctx->device->pipeline_cpy_i32_f32; }
  }
  if (src->type == v_TYPE_F32) {
    switch (to) {
      case v_TYPE_Q4_0:
      case v_TYPE_Q4_1:
      case v_TYPE_Q5_0:
      case v_TYPE_Q5_1:
      case v_TYPE_Q8_0:
      case v_TYPE_IQ4_NL:
        return ctx->device->pipeline_cpy_f32_quant[to];
      default:
        break;
    }
  }

  if (to == v_TYPE_F32) {
    switch (src->type) {
      case v_TYPE_Q4_0:
      case v_TYPE_Q4_1:
      case v_TYPE_Q5_0:
      case v_TYPE_Q5_1:
      case v_TYPE_Q8_0:
      case v_TYPE_IQ4_NL:
        return ctx->device->pipeline_cpy_quant_f32[src->type];
      default:
        break;
    }
  }

  if (src->type == to) {
    // Copy two or four bytes at a time, depending on block size.
    // For quantized types, we scale by block size/type size. But
    // this path is also used for bf16->bf16 for example, where the
    // type size must be exactly 2 or 4.
    V_ASSERT(v_is_quantized(to) || v_type_size(src->type) == 2 || v_type_size(src->type) == 4);
    if ((v_type_size(src->type) % 4) == 0) {
      if (contig) { return ctx->device->pipeline_contig_cpy_f32_f32; }
      else { return ctx->device->pipeline_cpy_f32_f32; }
    }
    else {
      if (contig) { return ctx->device->pipeline_contig_cpy_f16_f16; }
      else { return ctx->device->pipeline_cpy_f16_f16; }
    }
  }

  std::cerr << "Missing CPY op for types: " << v_type_name(src->type) << " " << v_type_name(to) << std::endl;
  v_ABORT("fatal error");
}

void v_vk_cpy_to_contiguous(vk_backend_ctx* ctx, vk_context& subctx, vk_pipeline pipeline,
                            const v_tensor* tensor, vk_sub_buffer&& in, vk_sub_buffer&& out) {
  VK_LOG_DEBUG(
    "v_vk_cpy_to_contiguous((" << tensor << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" <<
    tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" <<
    tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << "), ";
    std::cerr << "buffer in size=" << in.buffer->size << ", buffer out size=" << out.buffer->size << ")");
  const int tensor_type_size = v_type_size(tensor->type);

  const uint32_t ne = nelements(tensor);
  std::array<uint32_t, 3> elements;

  if (ne > 262144) { elements = {512, 512, CEIL_DIV(ne, 262144)}; }
  else if (ne > 512) { elements = {512, CEIL_DIV(ne, 512), 1}; }
  else { elements = {ne, 1, 1}; }

  vk_op_unary_push_constants pc = {
    (uint32_t)ne,
    (uint32_t)tensor->ne[0], (uint32_t)tensor->ne[1], (uint32_t)tensor->ne[2], (uint32_t)tensor->ne[3],
    (uint32_t)tensor->nb[0] / tensor_type_size, (uint32_t)tensor->nb[1] / tensor_type_size,
    (uint32_t)tensor->nb[2] / tensor_type_size, (uint32_t)tensor->nb[3] / tensor_type_size,
    (uint32_t)tensor->ne[0], (uint32_t)tensor->ne[1], (uint32_t)tensor->ne[2], (uint32_t)tensor->ne[3], 1,
    (uint32_t)tensor->ne[0], (uint32_t)(tensor->ne[0] * tensor->ne[1]),
    (uint32_t)(tensor->ne[0] * tensor->ne[1] * tensor->ne[2]),
    0,
    0.0f, 0.0f,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };
  init_pushconst_fastdiv(pc);
  v_vk_dispatch_pipeline(ctx, subctx, pipeline, {in, out}, pc, elements);
  vk_sync_buffers(ctx, subctx);
}

vk_pipeline v_vk_get_quantize_pipeline(vk_backend_ctx* ctx, v_data_type type, bool use_x4_blocks) {
  switch (type) {
    case v_TYPE_Q8_1:
      return use_x4_blocks
               ? ctx->device->pipeline_quantize_q8_1_x4
               : ctx->device->pipeline_quantize_q8_1;
    default:
      std::cerr << "Missing quantize pipeline for type: " << v_type_name(type) << std::endl;
      v_ABORT("fatal error");
  }
}

void v_vk_quantize_q8_1(vk_backend_ctx* ctx, vk_context& subctx, vk_sub_buffer&& in,
                        vk_sub_buffer&& out, uint32_t ne, bool use_x4_blocks = false) {
  VK_LOG_DEBUG(
    "v_vk_quantize_q8_1(" << "buffer in size=" << in.buffer->size << ", buffer out size=" << out.buffer->size << ", "
    << ne << ")");

  vk_pipeline pipeline = use_x4_blocks
                           ? v_vk_get_quantize_pipeline(ctx, v_TYPE_Q8_1, true)
                           : v_vk_get_quantize_pipeline(ctx, v_TYPE_Q8_1, false);

  v_vk_dispatch_pipeline(ctx, subctx, pipeline, {in, out}, std::array<uint32_t, 1>{ne}, {ne, 1, 1});
  vk_sync_buffers(ctx, subctx);
}

void v_vk_mul_mat_q_f16(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                        const v_tensor* src1, v_tensor* dst, bool disable_split_k, bool dryrun = false) {
  VK_LOG_DEBUG(
    "v_vk_mul_mat_q_f16((" << src0 << ", name=" << src0->name << ", type=" << v_type_name(src0->type) << ", ne0="
    << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0
    ->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << v_type_name(src1->type) << ", ne0=" <<
    src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb
    [0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << v_type_name(dst->type) << ", ne0=" << dst->
    ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] <<
    ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << (dryrun ? "dryrun" : "") << ")");
  V_ASSERT(v_vk_dim01_contiguous(src0) || src0->type == v_TYPE_F32 || src0->type == v_TYPE_F16 || src0->type == v_TYPE_BF16); // NOLINT
  V_ASSERT(v_vk_dim01_contiguous(src1) || src1->type == v_TYPE_F32 || src1->type == v_TYPE_F16); // NOLINT

  const uint64_t ne00 = src0->ne[0];
  const uint64_t ne01 = src0->ne[1];
  const uint64_t ne02 = src0->ne[2];
  const uint64_t ne03 = src0->ne[3];

  const uint64_t ne10 = src1->ne[0];
  const uint64_t ne11 = src1->ne[1];
  const uint64_t ne12 = src1->ne[2];
  const uint64_t ne13 = src1->ne[3];

  const uint64_t ne21           = dst->ne[1];
  const uint32_t stride_d       = dst->nb[1] / v_type_size(dst->type);
  const uint32_t stride_batch_d = stride_d * ne21;

  const uint64_t r2 = ne12 / ne02;
  const uint64_t r3 = ne13 / ne03;

  v_backend_vk_buffer_ctx* dst_buf_ctx  = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  v_backend_vk_buffer_ctx* src0_buf_ctx = (v_backend_vk_buffer_ctx*)src0->buffer->context;
  v_backend_vk_buffer_ctx* src1_buf_ctx = (v_backend_vk_buffer_ctx*)src1->buffer->context;

  vk_buffer d_Qx       = nullptr;
  size_t qx_buf_offset = 0;
  vk_buffer d_Qy       = nullptr;
  size_t qy_buf_offset = 0;

  bool src0_uma = false;
  bool src1_uma = false;

  if (ctx->device->uma) {
    vk_get_host_buffer(ctx->device, src0->data, d_Qx, qx_buf_offset);
    vk_get_host_buffer(ctx->device, src1->data, d_Qy, qy_buf_offset);
    src0_uma = d_Qx != nullptr;
    src1_uma = d_Qy != nullptr;
  }

  // Reformat and convert to fp16 if non-contiguous, or for coopmat2 for better perf
  const bool x_non_contig = (ctx->device->coopmat2 && src0->type == v_TYPE_F32) ||
    !v_vk_dim01_contiguous(src0);
  const bool y_non_contig = (ctx->device->coopmat2 && src1->type == v_TYPE_F32) ||
    (src0->type == v_TYPE_BF16 && src1->type != v_TYPE_BF16) ||
    !v_vk_dim01_contiguous(src1);

  // If src0 is BF16, try to use a BF16 x BF16 multiply
  v_data_type f16_type = src0->type == v_TYPE_BF16
                           ? v_TYPE_BF16
                           : v_TYPE_F16;

  const bool y_f32_kernel = src1->type == v_TYPE_F32 && !y_non_contig;

  bool quantize_y = ctx->device->integer_dot_product && src1->type == v_TYPE_F32 && v_is_contiguous(src1) && (ne11
      * ne10)
    % 4 == 0;

  // Check for mmq first
  vk_matmul_pipeline mmp = quantize_y
                             ? v_vk_get_mul_mat_mat_pipeline(ctx,
                                                             src0->type,
                                                             v_TYPE_Q8_1,
                                                             (v_prec)dst->op_params[0])
                             : nullptr;

  if (mmp == nullptr) {
    // Fall back to f16 dequant mul mat
    mmp = v_vk_get_mul_mat_mat_pipeline(ctx,
                                        src0->type,
                                        y_non_contig
                                          ? f16_type
                                          : src1->type,
                                        (v_prec)dst->op_params[0]);
    quantize_y = false;
  }

  const bool qx_needs_dequant = mmp == nullptr || x_non_contig;
  const bool qy_needs_dequant = !quantize_y && ((src1->type != f16_type && !y_f32_kernel) || y_non_contig);

  if (qx_needs_dequant) {
    // Fall back to dequant + f16 mulmat
    mmp = v_vk_get_mul_mat_mat_pipeline(ctx,
                                        f16_type,
                                        y_f32_kernel
                                          ? v_TYPE_F32
                                          : f16_type,
                                        (v_prec)dst->op_params[0]);
  }

  // Not implemented
  V_ASSERT(y_non_contig || !qy_needs_dequant); // NOLINT

  const uint32_t kpad = quantize_y
                          ? 0
                          : mmlVKAlignSize(ne10,
                                           v_vk_guess_matmul_pipeline_align(
                                             ctx,
                                             mmp,
                                             ne01,
                                             ne11,
                                             qx_needs_dequant
                                               ? f16_type
                                               : src0->type,
                                             quantize_y
                                               ? v_TYPE_Q8_1
                                               : (y_f32_kernel
                                                    ? v_TYPE_F32
                                                    : src1->type)));
  const bool aligned = !quantize_y && ne10 == kpad && ne01 > 8 && ne11 > 8;

  vk_pipeline pipeline = v_vk_guess_matmul_pipeline(ctx,
                                                    mmp,
                                                    ne01,
                                                    ne11,
                                                    aligned,
                                                    qx_needs_dequant
                                                      ? f16_type
                                                      : src0->type,
                                                    quantize_y
                                                      ? v_TYPE_Q8_1
                                                      : (y_f32_kernel
                                                           ? v_TYPE_F32
                                                           : src1->type));

  // Reserve extra storage in the N dimension for the Y matrix, so we can avoid bounds-checking
  uint32_t padded_n = qy_needs_dequant
                        ? ROUNDUP_POW2(ne11, pipeline->wg_denoms[1])
                        : ne11;
  const int x_ne = ne01 * ne00;
  const int y_ne = padded_n * ne10;
  const int d_ne = ne11 * ne01;

  const uint32_t split_k = v_vk_guess_split_k(ctx, ne01, ne11, ne10, disable_split_k, pipeline);

  const uint64_t qx_sz = v_type_size(src0->type) * x_ne / blockSize(src0->type);
  const uint64_t qy_sz = v_type_size(src1->type) * y_ne / blockSize(src1->type);
  const uint64_t x_sz  = !qx_needs_dequant
                           ? qx_sz
                           : sizeof(v_fp16_t) * x_ne;
  const uint64_t y_sz = quantize_y
                          ? (y_ne * v_type_size(v_TYPE_Q8_1) / blockSize(v_TYPE_Q8_1))
                          : (y_f32_kernel
                               ? sizeof(float) * y_ne
                               : sizeof(v_fp16_t) * y_ne);
  const uint64_t d_sz = sizeof(float) * d_ne;

  vk_pipeline to_fp16_vk_0 = nullptr;
  vk_pipeline to_fp16_vk_1 = nullptr;
  vk_pipeline to_q8_1      = nullptr;

  if (x_non_contig) { to_fp16_vk_0 = v_vk_get_cpy_pipeline(ctx, src0, nullptr, f16_type); }
  else { to_fp16_vk_0 = v_vk_get_to_fp16(ctx, src0->type); }
  if (y_non_contig) { to_fp16_vk_1 = v_vk_get_cpy_pipeline(ctx, src1, nullptr, f16_type); }
  else { to_fp16_vk_1 = v_vk_get_to_fp16(ctx, src1->type); }
  V_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr); // NOLINT
  V_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr); // NOLINT

  if (quantize_y) { to_q8_1 = v_vk_get_quantize_pipeline(ctx, v_TYPE_Q8_1, true); }

  if (dryrun) {
    const uint64_t x_sz_upd = x_sz * ne02 * ne03;
    uint64_t y_sz_upd       = y_sz * ne12 * ne13;
    if (quantize_y) { y_sz_upd = CEIL_DIV(y_sz_upd, 144) * 144; }
    const uint64_t split_k_size = split_k > 1
                                    ? d_sz * ne12 * ne13 * split_k
                                    : 0;
    if (
      (qx_needs_dequant && x_sz_upd > ctx->device->properties.limits.maxStorageBufferRange) ||
      (qy_needs_dequant && y_sz_upd > ctx->device->properties.limits.maxStorageBufferRange) ||
      (split_k > 1 && split_k_size > ctx->device->properties.limits.maxStorageBufferRange)) { v_ABORT("Requested preallocation size is too large"); }
    if (qx_needs_dequant && ctx->prealloc_size_x < x_sz_upd) { ctx->prealloc_size_x = x_sz_upd; }
    if ((qy_needs_dequant || quantize_y) && ctx->prealloc_size_y < y_sz_upd) { ctx->prealloc_size_y = y_sz_upd; }
    if (split_k > 1 && ctx->prealloc_size_split_k < split_k_size) { ctx->prealloc_size_split_k = split_k_size; }

    // Request descriptor sets
    v_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    if (qx_needs_dequant) { v_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1); }
    if (qy_needs_dequant) { v_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1); }
    if (quantize_y) { v_pipeline_request_descriptor_sets(ctx, to_q8_1, 1); }
    if (split_k > 1) { v_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_split_k_reduce, 1); }
    return;
  }

  vk_buffer d_D               = dst_buf_ctx->dev_buffer;
  const uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
  V_ASSERT(d_D != nullptr);
  V_ASSERT(d_D->size >= d_buf_offset + d_sz * ne02 * ne03);
  vk_buffer d_X;
  uint64_t x_buf_offset = 0;
  vk_buffer d_Y;
  uint64_t y_buf_offset = 0;
  if (!src0_uma) {
    d_Qx          = src0_buf_ctx->dev_buffer;
    qx_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
    V_ASSERT(d_Qx != nullptr);
  }
  if (!src1_uma) {
    d_Qy          = src1_buf_ctx->dev_buffer;
    qy_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
    V_ASSERT(d_Qy != nullptr);
  }
  if (qx_needs_dequant) {
    d_X = ctx->prealloc_x;
    V_ASSERT(d_X->size >= x_sz * ne02 * ne03);
  }
  else {
    d_X          = d_Qx;
    x_buf_offset = qx_buf_offset;
    V_ASSERT(qx_sz == x_sz);
  }
  if (qy_needs_dequant) {
    d_Y = ctx->prealloc_y;
    V_ASSERT(d_Y->size >= y_sz * ne12 * ne13);
  }
  else if (quantize_y) {
    d_Y = ctx->prealloc_y;
    V_ASSERT(d_Y->size >= CEIL_DIV(y_sz * ne12 * ne13, 144) * 144);
  }
  else {
    d_Y          = d_Qy;
    y_buf_offset = qy_buf_offset;
    V_ASSERT(qy_sz == y_sz);
  }

  if (x_non_contig || qx_needs_dequant) { if (ctx->prealloc_x_need_sync) { vk_sync_buffers(ctx, subctx); } }

  if (x_non_contig) {
    v_vk_cpy_to_contiguous(ctx,
                           subctx,
                           to_fp16_vk_0,
                           src0,
                           v_vk_subbuffer(ctx, d_Qx, qx_buf_offset),
                           v_vk_subbuffer(ctx, d_X, 0));
  }
  else if (qx_needs_dequant) {
    const std::vector<uint32_t> pc = {
      (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(nelements(src0))
    };
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           to_fp16_vk_0,
                           {
                             vk_sub_buffer{d_Qx, qx_buf_offset, qx_sz * ne02 * ne03},
                             vk_sub_buffer{d_X, 0, x_sz * ne02 * ne03}
                           },
                           pc,
                           {(uint32_t)(x_ne * ne02 * ne03), 1, 1});
    vk_sync_buffers(ctx, subctx);
  }
  if (y_non_contig) {
    if (ctx->prealloc_y_last_pipeline_used != to_fp16_vk_1.get() ||
      ctx->prealloc_y_last_tensor_used != src1) {
      if (ctx->prealloc_y_need_sync) { vk_sync_buffers(ctx, subctx); }
      v_vk_cpy_to_contiguous(ctx,
                             subctx,
                             to_fp16_vk_1,
                             src1,
                             v_vk_subbuffer(ctx, d_Qy, qy_buf_offset),
                             v_vk_subbuffer(ctx, d_Y, 0));
      ctx->prealloc_y_last_pipeline_used = to_fp16_vk_1.get();
      ctx->prealloc_y_last_tensor_used   = src1;
    }
  }
  if (quantize_y) {
    if (ctx->prealloc_y_last_pipeline_used != to_q8_1.get() ||
      ctx->prealloc_y_last_tensor_used != src1) {
      if (ctx->prealloc_y_need_sync) { vk_sync_buffers(ctx, subctx); }
      v_vk_quantize_q8_1(ctx,
                         subctx,
                         v_vk_subbuffer(ctx, d_Qy, qy_buf_offset),
                         v_vk_subbuffer(ctx, d_Y, 0),
                         y_ne * ne12 * ne13,
                         true);
      ctx->prealloc_y_last_pipeline_used = to_q8_1.get();
      ctx->prealloc_y_last_tensor_used   = src1;
    }
  }

  uint32_t stride_batch_x = ne00 * ne01;
  uint32_t stride_batch_y = ne10 * ne11;

  if (!v_vk_dim01_contiguous(src0) && !qx_needs_dequant) { stride_batch_x = src0->nb[0] / v_type_size(src0->type); }

  if (!v_vk_dim01_contiguous(src1) && !qy_needs_dequant && !quantize_y) { stride_batch_y = src1->nb[0] / v_type_size(src1->type); }

  uint32_t y_sz_total = y_sz * ne12 * ne13;
  if (quantize_y) { y_sz_total = CEIL_DIV(y_sz_total, 144) * 144; }

  // compute
  v_vk_matmul(
    ctx,
    subctx,
    pipeline,
    {d_X, x_buf_offset, x_sz * ne02 * ne03},
    {d_Y, y_buf_offset, y_sz_total},
    v_vk_subbuffer(ctx, d_D, d_buf_offset),
    {ctx->prealloc_split_k, 0, d_sz * ne12 * ne13 * split_k},
    ne01,
    ne11,
    ne10,
    ne10,
    ne10,
    stride_d,
    stride_batch_x,
    stride_batch_y,
    stride_batch_d,
    split_k,
    ne12 * ne13,
    ne02,
    ne12,
    r2,
    r3,
    padded_n
  ); // NOLINT

  if (x_non_contig || qx_needs_dequant) { ctx->prealloc_x_need_sync = true; }
  if (y_non_contig || quantize_y) { ctx->prealloc_y_need_sync = true; }
}

// Device tuning
bool v_vk_should_use_mmvq(const vk_device& device, uint32_t m, uint32_t n, uint32_t k, v_data_type src0_type) {
  if (device->mmvq_mode == 1) { return true; }
  else if (device->mmvq_mode == -1) { return false; }

  // MMVQ is generally good for batches
  if (n > 1) { return true; }

  switch (device->vendor_id) {
    case VK_VENDOR_ID_NVIDIA:
      switch (src0_type) {
        case v_TYPE_Q8_0:
          return device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
        default:
          return true;
      }
    case VK_VENDOR_ID_AMD:
      switch (src0_type) {
        case v_TYPE_Q8_0:
          return device->architecture == vk_device_architecture::AMD_GCN;
        default:
          return true;
      }
    case VK_VENDOR_ID_INTEL:
      switch (src0_type) {
        // From tests on A770 Linux, may need more tuning
        case v_TYPE_Q4_0:
        case v_TYPE_Q5_1:
          return false;
        default:
          return true;
      }
    default:
      return true;
  }

  v_UNUSED(m);
  v_UNUSED(k);
}

void v_vk_mul_mat_vec_q_f16(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                            const v_tensor* src1, v_tensor* dst, bool dryrun = false) {
  VK_LOG_DEBUG(
    "v_vk_mul_mat_vec_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[
      0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] <<
    ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] <<
    ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1="
    << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1="
    << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1
    ] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << (dryrun ? "dryrun" : "") << "),)");
  V_ASSERT(v_vk_dim01_contiguous(src0) || src0->type == v_TYPE_F32 || src0->type == v_TYPE_F16 || src0->type == v_TYPE_BF16); // NOLINT
  V_ASSERT(v_vk_dim01_contiguous(src1) || src1->type == v_TYPE_F32 || src1->type == v_TYPE_F16); // NOLINT

  const uint64_t ne00 = src0->ne[0];
  const uint64_t ne01 = src0->ne[1];
  const uint64_t ne02 = src0->ne[2];
  const uint64_t ne03 = src0->ne[3];

  const uint64_t ne10 = src1->ne[0];
  const uint64_t ne11 = src1->ne[1];
  const uint64_t ne12 = src1->ne[2];
  const uint64_t ne13 = src1->ne[3];

  const uint64_t ne20 = dst->ne[0];
  const uint64_t ne21 = dst->ne[1];
  const uint64_t ne22 = dst->ne[2];
  const uint64_t ne23 = dst->ne[3];

  const uint64_t r2 = ne12 / ne02;
  const uint64_t r3 = ne13 / ne03;

  // batch_n indicates that we need to compute a few vector results, and this assumes
  // ne12 and ne13 are 1. It overloads the batch_strides to hold the row strides.
  V_ASSERT(ne11 == 1 || ne12 * ne13 == 1);
  bool batch_n = ne11 > 1;

  v_backend_vk_buffer_ctx* dst_buf_ctx  = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  v_backend_vk_buffer_ctx* src0_buf_ctx = (v_backend_vk_buffer_ctx*)src0->buffer->context;
  v_backend_vk_buffer_ctx* src1_buf_ctx = (v_backend_vk_buffer_ctx*)src1->buffer->context;

  vk_buffer d_Qx       = nullptr;
  size_t qx_buf_offset = 0;
  vk_buffer d_Qy       = nullptr;
  size_t qy_buf_offset = 0;

  bool src0_uma = false;
  bool src1_uma = false;

  if (ctx->device->uma) {
    vk_get_host_buffer(ctx->device, src0->data, d_Qx, qx_buf_offset);
    vk_get_host_buffer(ctx->device, src1->data, d_Qy, qy_buf_offset);
    src0_uma = d_Qx != nullptr;
    src1_uma = d_Qy != nullptr;
  }

  const bool x_non_contig = !v_vk_dim01_contiguous(src0);
  const bool y_non_contig = !v_vk_dim01_contiguous(src1);

  const bool f16_f32_kernel = src1->type == v_TYPE_F32;
  bool quantize_y           = ctx->device->integer_dot_product && src1->type == v_TYPE_F32 && v_is_contiguous(src1) && (ne11
      * ne10)
    % 4 == 0 && v_vk_should_use_mmvq(ctx->device, ne01, ne11, ne10, src0->type);

  vk_pipeline to_fp16_vk_0 = nullptr;
  vk_pipeline to_fp16_vk_1 = nullptr;
  if (x_non_contig) { to_fp16_vk_0 = v_vk_get_cpy_pipeline(ctx, src0, nullptr, src0->type); }
  if (y_non_contig) { to_fp16_vk_1 = v_vk_get_cpy_pipeline(ctx, src1, nullptr, src1->type); }
  else { to_fp16_vk_1 = v_vk_get_to_fp16(ctx, src1->type); }

  // Check for mmq first
  vk_pipeline dmmv = quantize_y
                       ? v_vk_get_dequantize_mul_mat_vec(ctx, src0->type, v_TYPE_Q8_1, ne11, ne20, ne00)
                       : nullptr;
  vk_pipeline to_q8_1 = nullptr;

  if (dmmv == nullptr) {
    // Fall back to f16 dequant mul mat
    dmmv       = v_vk_get_dequantize_mul_mat_vec(ctx, src0->type, src1->type, ne11, ne20, ne00);
    quantize_y = false;
  }

  if (quantize_y) { to_q8_1 = v_vk_get_quantize_pipeline(ctx, v_TYPE_Q8_1, true); }

  const bool qx_needs_dequant = x_non_contig;
  const bool qy_needs_dequant = !quantize_y && ((src1->type != v_TYPE_F16 && !f16_f32_kernel) || y_non_contig);

  // Not implemented
  V_ASSERT(y_non_contig || !qy_needs_dequant); // NOLINT

  V_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr); // NOLINT
  V_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr); // NOLINT
  V_ASSERT(dmmv != nullptr);

  const uint64_t x_ne = ne01 * ne00;
  const uint64_t y_ne = ne11 * ne10;
  const uint64_t d_ne = ne11 * ne01;

  const uint64_t qx_sz = mmlVKAlignSize(v_type_size(src0->type) * x_ne / blockSize(src0->type),
                                        ctx->device->properties.limits.minStorageBufferOffsetAlignment);
  const uint64_t qy_sz = v_type_size(src1->type) * y_ne / blockSize(src1->type);
  const uint64_t x_sz  = x_non_contig
                           ? mmlVKAlignSize(v_type_size(src0->type) * x_ne,
                                            ctx->device->properties.limits.minStorageBufferOffsetAlignment)
                           : qx_sz;
  const uint64_t y_sz = quantize_y
                          ? (y_ne * v_type_size(v_TYPE_Q8_1) / blockSize(v_TYPE_Q8_1))
                          : (f16_f32_kernel
                               ? sizeof(float) * y_ne
                               : sizeof(v_fp16_t) * y_ne);
  const uint64_t d_sz = sizeof(float) * d_ne;

  if (dryrun) {
    const uint64_t x_sz_upd = x_sz * ne02 * ne03;
    uint64_t y_sz_upd       = y_sz * ne12 * ne13;
    if (quantize_y) { y_sz_upd = CEIL_DIV(y_sz_upd, 144) * 144; }
    if (
      (qx_needs_dequant && x_sz_upd > ctx->device->properties.limits.maxStorageBufferRange) ||
      (qy_needs_dequant && y_sz_upd > ctx->device->properties.limits.maxStorageBufferRange)) { v_ABORT("Requested preallocation size is too large"); }
    if (qx_needs_dequant && ctx->prealloc_size_x < x_sz_upd) { ctx->prealloc_size_x = x_sz_upd; }
    if ((qy_needs_dequant || quantize_y) && ctx->prealloc_size_y < y_sz_upd) { ctx->prealloc_size_y = y_sz_upd; }

    // Request descriptor sets
    if (qx_needs_dequant) { v_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1); }
    if (qy_needs_dequant) { v_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1); }
    if (quantize_y) { v_pipeline_request_descriptor_sets(ctx, to_q8_1, 1); }
    v_pipeline_request_descriptor_sets(ctx, dmmv, 1);
    return;
  }

  vk_buffer d_D               = dst_buf_ctx->dev_buffer;
  const uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
  V_ASSERT(d_D != nullptr);
  vk_buffer d_X;
  uint64_t x_buf_offset = 0;
  vk_buffer d_Y;
  uint64_t y_buf_offset = 0;
  if (!src0_uma) {
    d_Qx          = src0_buf_ctx->dev_buffer;
    qx_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
    V_ASSERT(d_Qx != nullptr);
  }
  if (!src1_uma) {
    d_Qy          = src1_buf_ctx->dev_buffer;
    qy_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
    V_ASSERT(d_Qy != nullptr);
  }
  if (qx_needs_dequant) { d_X = ctx->prealloc_x; }
  else {
    d_X          = d_Qx;
    x_buf_offset = qx_buf_offset;
    V_ASSERT(qx_sz == x_sz);
  }
  if (qy_needs_dequant) { d_Y = ctx->prealloc_y; }
  else if (quantize_y) {
    d_Y = ctx->prealloc_y;
    V_ASSERT(d_Y->size >= CEIL_DIV(y_sz * ne12 * ne13, 144) * 144);
  }
  else {
    d_Y          = d_Qy;
    y_buf_offset = qy_buf_offset;
    V_ASSERT(qy_sz == y_sz);
  }

  if (x_non_contig) {
    if (ctx->prealloc_x_need_sync) { vk_sync_buffers(ctx, subctx); }

    V_ASSERT(
      x_sz == mmlVKAlignSize(v_type_size(src0->type) * x_ne, ctx->device->properties.limits.
        minStorageBufferOffsetAlignment));
    v_vk_cpy_to_contiguous(ctx,
                           subctx,
                           to_fp16_vk_0,
                           src0,
                           v_vk_subbuffer(ctx, d_Qx, qx_buf_offset),
                           v_vk_subbuffer(ctx, d_X, 0));
  }
  if (y_non_contig) {
    V_ASSERT(y_sz == v_type_size(src1->type) * y_ne);
    if (ctx->prealloc_y_last_pipeline_used != to_fp16_vk_1.get() ||
      ctx->prealloc_y_last_tensor_used != src1) {
      if (ctx->prealloc_y_need_sync) { vk_sync_buffers(ctx, subctx); }
      v_vk_cpy_to_contiguous(ctx,
                             subctx,
                             to_fp16_vk_1,
                             src1,
                             v_vk_subbuffer(ctx, d_Qy, qy_buf_offset),
                             v_vk_subbuffer(ctx, d_Y, 0));
      ctx->prealloc_y_last_pipeline_used = to_fp16_vk_1.get();
      ctx->prealloc_y_last_tensor_used   = src1;
    }
  }
  if (quantize_y) {
    if (ctx->prealloc_y_last_pipeline_used != to_q8_1.get() ||
      ctx->prealloc_y_last_tensor_used != src1) {
      if (ctx->prealloc_y_need_sync) { vk_sync_buffers(ctx, subctx); }
      v_vk_quantize_q8_1(ctx,
                         subctx,
                         v_vk_subbuffer(ctx, d_Qy, qy_buf_offset),
                         v_vk_subbuffer(ctx, d_Y, 0),
                         y_ne * ne12 * ne13,
                         true);
      ctx->prealloc_y_last_pipeline_used = to_q8_1.get();
      ctx->prealloc_y_last_tensor_used   = src1;
    }
  }

  // For batch_n, the A matrix is the same for each batch, and B/D use the row stride as the batch stride
  uint32_t stride_batch_x = batch_n
                              ? 0
                              : ne00 * ne01;
  uint32_t stride_batch_y = batch_n
                              ? ne10
                              : (ne10 * ne11);
  uint32_t stride_batch_d = batch_n
                              ? ne20
                              : (ne20 * ne21);

  if (!v_vk_dim01_contiguous(src0) && !qx_needs_dequant) { stride_batch_x = src0->nb[0] / v_type_size(src0->type); }

  if (!v_vk_dim01_contiguous(src1) && !qy_needs_dequant) { stride_batch_y = src1->nb[0] / v_type_size(src1->type); }

  const uint32_t max_groups_x = ctx->device->properties.limits.maxComputeWorkGroupCount[0];

  uint32_t groups_x = ne01;
  uint32_t groups_z = 1;

  if (ne01 > max_groups_x) {
    groups_z = 64;
    groups_x = CEIL_DIV(groups_x, groups_z);
  }

  // TODO: Clean up this whole sz * ne_2 * ne_3 thing, it hasn't been necessary for a long time
  uint32_t y_sz_total = y_sz * ne12 * ne13;
  if (quantize_y) { y_sz_total = CEIL_DIV(y_sz_total, 144) * 144; }

  // compute
  const vk_mat_vec_push_constants pc = {
    (uint32_t)ne00, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne01,
    stride_batch_x, stride_batch_y, stride_batch_d,
    (uint32_t)ne02, (uint32_t)ne12, (uint32_t)r2, (uint32_t)r3,
  };
  v_vk_dispatch_pipeline(ctx,
                         subctx,
                         dmmv,
                         {
                           vk_sub_buffer{d_X, x_buf_offset, x_sz * ne02 * ne03},
                           vk_sub_buffer{d_Y, y_buf_offset, y_sz_total},
                           vk_sub_buffer{d_D, d_buf_offset, d_sz * ne22 * ne23}
                         },
                         pc,
                         {groups_x, (uint32_t)(ne12 * ne13), groups_z});

  if (x_non_contig) { ctx->prealloc_x_need_sync = true; }
  if (y_non_contig || quantize_y) { ctx->prealloc_y_need_sync = true; }
}

void v_vk_mul_mat_vec_p021_f16_f32(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                                   const v_tensor* src1, v_tensor* dst, bool dryrun = false) {
  VK_LOG_DEBUG(
    "v_vk_mul_mat_p021_f16_f32(" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->
    ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] <<
    ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] <<
    ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1="
    << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1="
    << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1
    ] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << (dryrun ? "dryrun" : "") << ")");
  V_ASSERT(v_is_permuted(src0) && v_is_permuted(src1));
  V_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // NOLINT
  V_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // NOLINT
  V_ASSERT(src0->type == v_TYPE_F16);
  V_ASSERT(src1->type == v_TYPE_F32);

  const uint64_t ne00 = src0->ne[0];
  const uint64_t ne01 = src0->ne[1];
  const uint64_t ne02 = src0->ne[2];
  // const uint64_t ne03 = src0->ne[3];

  const uint64_t ne10 = src1->ne[0];
  const uint64_t ne11 = src1->ne[1];
  const uint64_t ne12 = src1->ne[2];
  // const uint64_t ne13 = src1->ne[3];

  V_ASSERT(ne11 == 1);

  v_backend_vk_buffer_ctx* dst_buf_ctx  = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  v_backend_vk_buffer_ctx* src0_buf_ctx = (v_backend_vk_buffer_ctx*)src0->buffer->context;
  v_backend_vk_buffer_ctx* src1_buf_ctx = (v_backend_vk_buffer_ctx*)src1->buffer->context;

  vk_buffer d_Qy       = nullptr;
  size_t qy_buf_offset = 0;

  bool src1_uma = false;

  if (ctx->device->uma) {
    vk_get_host_buffer(ctx->device, src1->data, d_Qy, qy_buf_offset);
    src1_uma = d_Qy != nullptr;
  }

  const uint64_t x_ne = ne00 * ne01 * ne02;
  const uint64_t y_ne = ne10 * ne11 * ne12;
  const uint64_t d_ne = ne01 * ne11 * ne12;

  const uint64_t qx_sz = mmlVKAlignSize(v_type_size(src0->type) * x_ne / blockSize(src0->type),
                                        ctx->device->properties.limits.minStorageBufferOffsetAlignment);
  const uint64_t qy_sz = v_type_size(src1->type) * y_ne / blockSize(src1->type);
  const uint64_t d_sz  = sizeof(float) * d_ne;

  // With grouped query attention there are > 1 Q matrices per K, V matrix.
  uint32_t gqa_ratio = (uint32_t)ne12 / (uint32_t)ne02;
  if (gqa_ratio > 8 || gqa_ratio == 0 || ne12 != ne02 * gqa_ratio) { gqa_ratio = 1; }

  if (dryrun) {
    // Request descriptor sets
    v_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_mul_mat_vec_p021_f16_f32[gqa_ratio - 1], 1);
    return;
  }

  vk_buffer d_D               = dst_buf_ctx->dev_buffer;
  const uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
  V_ASSERT(d_D != nullptr);
  vk_buffer d_Qx               = src0_buf_ctx->dev_buffer;
  const uint64_t qx_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
  V_ASSERT(d_Qx != nullptr);
  if (!src1_uma) {
    d_Qy          = src1_buf_ctx->dev_buffer;
    qy_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
    V_ASSERT(d_Qx != nullptr);
  }

  const uint64_t qy_buffer_offset = (qy_buf_offset / ctx->device->properties.limits.minStorageBufferOffsetAlignment) *
    ctx->device->properties.limits.minStorageBufferOffsetAlignment;
  const uint64_t qy_shader_offset = qy_buf_offset - qy_buffer_offset;

  const uint64_t d_buffer_offset = (d_buf_offset / ctx->device->properties.limits.minStorageBufferOffsetAlignment) * ctx
                                                                                                                     ->
                                                                                                                     device
                                                                                                                     ->
                                                                                                                     properties
                                                                                                                     .limits
                                                                                                                     .minStorageBufferOffsetAlignment;
  const uint64_t d_shader_offset = d_buf_offset - d_buffer_offset;

  // compute
  const std::array<uint32_t, 6> pc = {
    (uint32_t)ne00, (uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne12,
    (uint32_t)(qy_shader_offset / v_type_size(src1->type)), (uint32_t)(d_shader_offset / v_type_size(dst->type))
  };

  uint32_t workgroups_z = (uint32_t)ne12;
  // When gqa_ratio > 1, each invocation does multiple rows and we can launch fewer workgroups
  if (gqa_ratio > 1) { workgroups_z /= gqa_ratio; }

  v_vk_dispatch_pipeline(ctx,
                         subctx,
                         ctx->device->pipeline_mul_mat_vec_p021_f16_f32[gqa_ratio - 1],
                         {
                           vk_sub_buffer{d_Qx, qx_buf_offset, qx_sz},
                           vk_sub_buffer{d_Qy, qy_buffer_offset, qy_sz + qy_shader_offset},
                           vk_sub_buffer{d_D, d_buffer_offset, d_sz + d_shader_offset}
                         },
                         pc,
                         {1, (uint32_t)ne01, workgroups_z});
}

void v_vk_mul_mat_vec_nc_f16_f32(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                                 const v_tensor* src1, v_tensor* dst, bool dryrun = false) {
  VK_LOG_DEBUG(
    "v_vk_mul_mat_nc_f16_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne
    [0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] <<
    ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] <<
    ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1="
    << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1="
    << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1
    ] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << (dryrun ? "dryrun" : "") << ")");
  V_ASSERT(!v_is_transposed(src0));
  V_ASSERT(!v_is_transposed(src1));
  V_ASSERT(!v_is_permuted(src0));
  V_ASSERT(src0->type == v_TYPE_F16);
  V_ASSERT(src1->type == v_TYPE_F32);

  const uint64_t ne00 = src0->ne[0];
  const uint64_t ne01 = src0->ne[1];
  const uint64_t ne02 = src0->ne[2];
  const uint64_t ne03 = src0->ne[3];

  const uint64_t nb01 = src0->nb[1];
  const uint64_t nb02 = src0->nb[2];

  const uint64_t nb12 = src1->nb[2];

  // const uint64_t ne10 = src1->ne[0];
  const uint64_t ne11 = src1->ne[1];
  const uint64_t ne12 = src1->ne[2];
  // const uint64_t ne13 = src1->ne[3];

  const uint32_t nb03 = (uint32_t)(src0->nb[3] / sizeof(v_fp16_t));
  const uint32_t nb13 = (uint32_t)(src1->nb[3] / sizeof(float));
  const uint32_t nb23 = (uint32_t)(dst->nb[3] / sizeof(float));

  V_ASSERT(ne11 == 1);
  V_ASSERT(src0->ne[3] == src1->ne[3]); // checked in supports_op

  v_backend_vk_buffer_ctx* dst_buf_ctx  = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  v_backend_vk_buffer_ctx* src0_buf_ctx = (v_backend_vk_buffer_ctx*)src0->buffer->context;
  v_backend_vk_buffer_ctx* src1_buf_ctx = (v_backend_vk_buffer_ctx*)src1->buffer->context;

  vk_buffer d_Qy       = nullptr;
  size_t qy_buf_offset = 0;

  bool src1_uma = false;

  if (ctx->device->uma) {
    vk_get_host_buffer(ctx->device, src1->data, d_Qy, qy_buf_offset);
    src1_uma = d_Qy != nullptr;
  }

  const uint64_t d_ne = ne01 * ne11 * ne12 * ne03;

  const uint32_t row_stride_x     = nb01 / sizeof(v_fp16_t);
  const uint32_t channel_stride_x = nb02 / sizeof(v_fp16_t);
  const uint32_t channel_stride_y = nb12 / sizeof(float);

  const uint64_t qx_sz = num_bytes(src0);
  const uint64_t qy_sz = num_bytes(src1);
  const uint64_t d_sz  = sizeof(float) * d_ne;

  if (dryrun) {
    // Request descriptor sets
    v_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_mul_mat_vec_nc_f16_f32, 1);
    return;
  }

  vk_buffer d_D               = dst_buf_ctx->dev_buffer;
  const uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
  V_ASSERT(d_D != nullptr);
  vk_buffer d_Qx               = src0_buf_ctx->dev_buffer;
  const uint64_t qx_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
  V_ASSERT(d_Qx != nullptr);
  if (!src1_uma) {
    d_Qy          = src1_buf_ctx->dev_buffer;
    qy_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
    V_ASSERT(d_Qx != nullptr);
  }

  const uint64_t qy_buffer_offset = (qy_buf_offset / ctx->device->properties.limits.minStorageBufferOffsetAlignment) *
    ctx->device->properties.limits.minStorageBufferOffsetAlignment;
  const uint64_t qy_shader_offset = qy_buf_offset - qy_buffer_offset;

  const uint64_t d_buffer_offset = (d_buf_offset / ctx->device->properties.limits.minStorageBufferOffsetAlignment) * ctx
                                                                                                                     ->
                                                                                                                     device
                                                                                                                     ->
                                                                                                                     properties
                                                                                                                     .limits
                                                                                                                     .minStorageBufferOffsetAlignment;
  const uint64_t d_shader_offset = d_buf_offset - d_buffer_offset;

  // compute
  const std::array<uint32_t, 12> pc = {
    (uint32_t)ne00, (uint32_t)ne01, row_stride_x, channel_stride_x, channel_stride_y, (uint32_t)(ne12 / ne02),
    (uint32_t)ne12, (uint32_t)(qy_shader_offset / v_type_size(src1->type)),
    (uint32_t)(d_shader_offset / v_type_size(dst->type)), nb03, nb13, nb23
  };
  v_vk_dispatch_pipeline(ctx,
                         subctx,
                         ctx->device->pipeline_mul_mat_vec_nc_f16_f32,
                         {
                           vk_sub_buffer{d_Qx, qx_buf_offset, qx_sz},
                           vk_sub_buffer{d_Qy, qy_buffer_offset, qy_sz + qy_shader_offset},
                           vk_sub_buffer{d_D, d_buffer_offset, d_sz + d_shader_offset}
                         },
                         pc,
                         {(uint32_t)ne03, (uint32_t)ne01, (uint32_t)ne12});
}

void v_vk_mul_mat(vk_backend_ctx* ctx, vk_context& subctx, v_tensor* src0, v_tensor* src1,
                  v_tensor* dst, bool dryrun = false) {
  VK_LOG_DEBUG("v_vk_mul_mat(" << src0 << ", " << src1 << ", " << dst << ")");

  // Handle huge A matrix by splitting the M dimensions. This works well for convolution use cases
  // where the M dimension is very large.
  // Split_k doesn't work with M splitting.
  const size_t nbytes    = num_bytes(src0);
  const bool needs_split = nbytes > ctx->device->properties.limits.maxStorageBufferRange;
  if (needs_split) {
    // Choose the number of rows that can fit (and divide by two, to allow for any additional offsets)
    const uint32_t M_split = ctx->device->properties.limits.maxStorageBufferRange / (2 * src0->nb[1]);
    uint32_t m_offset      = 0;
    while (m_offset < dst->ne[0]) {
      const uint32_t cur_M_size = std::min(M_split, (uint32_t)(dst->ne[0] - m_offset));
      v_tensor dst2             = *dst;
      v_tensor src02            = *src0;

      dst2.view_src = dst->view_src
                        ? dst->view_src
                        : dst;
      src02.view_src = src0->view_src
                         ? src0->view_src
                         : src0;

      dst2.view_offs += m_offset * dst->nb[0];
      src02.view_offs += m_offset * src0->nb[1];
      dst2.ne[0]  = cur_M_size;
      src02.ne[1] = cur_M_size;

      v_vk_mul_mat_q_f16(ctx, subctx, &src02, src1, &dst2, true, dryrun);

      m_offset += cur_M_size;
    }
  }
  else if (src0->type == v_TYPE_F16 && v_is_permuted(src0) && v_is_permuted(src1) && dst->ne[1] == 1 &&
    // detect 0213 permutation, and batch size of 1
    src0->nb[0] <= src0->nb[2] &&
    src0->nb[2] <= src0->nb[1] &&
    src0->nb[1] <= src0->nb[3] &&
    src1->nb[0] <= src1->nb[2] &&
    src1->nb[2] <= src1->nb[1] &&
    src1->nb[1] <= src1->nb[3] &&
    src0->ne[3] == 1 &&
    src1->ne[3] == 1) { v_vk_mul_mat_vec_p021_f16_f32(ctx, subctx, src0, src1, dst, dryrun); }
  else if (src0->type == v_TYPE_F16 && !v_is_contiguous(src0) && !v_is_transposed(src1) && dst->ne[1] == 1 &&
    !v_is_permuted(src0) && !v_is_permuted(src1)) {
    v_vk_mul_mat_vec_nc_f16_f32(ctx, subctx, src0, src1, dst, dryrun);
    // mul_mat_vec supports batching ne12*ne13 when ne11==1, or treating ne11 as the batch size (up to four)
    // when ne12 and ne13 are one.
  }
  else if ((dst->ne[1] == 1 || (dst->ne[1] <= mul_mat_vec_max_cols && src1->ne[2] * src1->ne[3] == 1)) &&
    (src0->type == v_TYPE_F32 || src0->type == v_TYPE_F16 || src0->type == v_TYPE_BF16 ||
      v_is_quantized(src0->type))) { v_vk_mul_mat_vec_q_f16(ctx, subctx, src0, src1, dst, dryrun); }
  else { v_vk_mul_mat_q_f16(ctx, subctx, src0, src1, dst, false, dryrun); }
}

void v_vk_mul_mat_id_q_f16(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                           const v_tensor* src1, const v_tensor* ids, v_tensor* dst,
                           bool dryrun = false) {
  VK_LOG_DEBUG(
    "v_vk_mul_mat_id_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0
    ] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] <<
    ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] <<
    ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1="
    << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << ids << ", name=" << ids->name << ", type=" << ids->type << ", ne0=" << ids->ne[0] << ", ne1="
    << ids->ne[1] << ", ne2=" << ids->ne[2] << ", ne3=" << ids->ne[3] << ", nb0=" << ids->nb[0] << ", nb1=" << ids->nb[1
    ] << ", nb2=" << ids->nb[2] << ", nb3=" << ids->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1="
    << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1
    ] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3] << "),)");
  V_ASSERT(v_vk_dim01_contiguous(src1) || src1->type == v_TYPE_F32 || src1->type == v_TYPE_F16); // NOLINT
  V_ASSERT(ids->type == v_TYPE_I32);

  const uint64_t ne00 = src0->ne[0];
  const uint64_t ne01 = src0->ne[1];
  const uint64_t ne02 = src0->ne[2];
  const uint64_t ne03 = src0->ne[3];

  const uint64_t ne10 = src1->ne[0];
  const uint64_t ne11 = src1->ne[1];
  const uint64_t ne12 = src1->ne[2];
  const uint64_t ne13 = src1->ne[3];

  const uint64_t nei0 = ids->ne[0];
  const uint64_t nei1 = ids->ne[1];

  const uint32_t nbi1 = ids->nb[1];
  const uint32_t nbi2 = ids->nb[2];

  const uint64_t ne20 = dst->ne[0];
  const uint64_t ne21 = dst->ne[1];
  const uint64_t ne22 = dst->ne[2];
  const uint64_t ne23 = dst->ne[3];

  const uint64_t n_as = ne02;

  v_backend_vk_buffer_ctx* dst_buf_ctx  = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  v_backend_vk_buffer_ctx* src0_buf_ctx = (v_backend_vk_buffer_ctx*)src0->buffer->context;
  v_backend_vk_buffer_ctx* src1_buf_ctx = (v_backend_vk_buffer_ctx*)src1->buffer->context;
  v_backend_vk_buffer_ctx* ids_buf_ctx  = (v_backend_vk_buffer_ctx*)ids->buffer->context;

  vk_buffer d_Qx        = nullptr;
  size_t qx_buf_offset  = 0;
  vk_buffer d_Qy        = nullptr;
  size_t qy_buf_offset  = 0;
  vk_buffer d_ids       = nullptr;
  size_t ids_buf_offset = 0;

  bool src0_uma = false;
  bool src1_uma = false;
  bool ids_uma  = false;

  if (ctx->device->uma) {
    vk_get_host_buffer(ctx->device, src0->data, d_Qx, qx_buf_offset);
    vk_get_host_buffer(ctx->device, src1->data, d_Qy, qy_buf_offset);
    vk_get_host_buffer(ctx->device, ids->data, d_ids, ids_buf_offset);
    src0_uma = d_Qx != nullptr;
    src1_uma = d_Qy != nullptr;
    ids_uma  = d_ids != nullptr;
  }

  // Reformat and convert to fp16 if non-contiguous, or for coopmat2 for better perf
  const bool x_non_contig = (ctx->device->coopmat2 && src0->type == v_TYPE_F32) ||
    !v_vk_dim01_contiguous(src0);
  const bool y_non_contig = (ctx->device->coopmat2 && src1->type == v_TYPE_F32) ||
    (src0->type == v_TYPE_BF16 && src1->type != v_TYPE_BF16) ||
    !v_vk_dim01_contiguous(src1);

  // If src0 is BF16, try to use a BF16 x BF16 multiply
  v_data_type f16_type = src0->type == v_TYPE_BF16
                           ? v_TYPE_BF16
                           : v_TYPE_F16;

  const bool y_f32_kernel = src1->type == v_TYPE_F32 && !y_non_contig;

  vk_matmul_pipeline mmp = v_vk_get_mul_mat_mat_id_pipeline(ctx,
                                                            src0->type,
                                                            y_non_contig
                                                              ? f16_type
                                                              : src1->type,
                                                            (v_prec)dst->op_params[0]);

  const bool qx_needs_dequant = mmp == nullptr || x_non_contig;
  const bool qy_needs_dequant = (src1->type != f16_type && !y_f32_kernel) || y_non_contig;

  if (qx_needs_dequant) {
    // Fall back to dequant + f16 mulmat
    mmp = v_vk_get_mul_mat_mat_id_pipeline(ctx,
                                           f16_type,
                                           y_f32_kernel
                                             ? v_TYPE_F32
                                             : f16_type,
                                           (v_prec)dst->op_params[0]);
  }

  // Not implemented
  V_ASSERT(y_non_contig || !qy_needs_dequant); // NOLINT

  const uint32_t kpad = mmlVKAlignSize(
    ne10,
    v_vk_guess_matmul_id_pipeline_align(ctx, mmp, ne01, nei1, qx_needs_dequant
                                                                ? f16_type
                                                                : src0->type));
  const bool aligned = ne10 == kpad && ne01 > 8 && nei1 > 8;

  vk_pipeline pipeline = v_vk_guess_matmul_id_pipeline(ctx,
                                                       mmp,
                                                       ne01,
                                                       nei1,
                                                       aligned,
                                                       qx_needs_dequant
                                                         ? f16_type
                                                         : src0->type);

  // Reserve extra storage in the N dimension for the Y matrix, so we can avoid bounds-checking
  uint32_t padded_n = qy_needs_dequant
                        ? ROUNDUP_POW2(ne11, pipeline->wg_denoms[1])
                        : ne11;
  const uint64_t x_ne = ne01 * ne00;
  const uint64_t y_ne = padded_n * ne10;
  const uint64_t d_ne = ne21 * ne20;

  const uint64_t qx_sz = v_type_size(src0->type) * x_ne / blockSize(src0->type);
  const uint64_t qy_sz = v_type_size(src1->type) * y_ne / blockSize(src1->type);
  const uint64_t x_sz  = !qx_needs_dequant
                           ? qx_sz
                           : sizeof(v_fp16_t) * x_ne;
  const uint64_t y_sz = y_f32_kernel
                          ? sizeof(float) * y_ne
                          : sizeof(v_fp16_t) * y_ne;
  const uint64_t ids_sz = nbi2;
  const uint64_t d_sz   = sizeof(float) * d_ne;

  vk_pipeline to_fp16_vk_0 = nullptr;
  vk_pipeline to_fp16_vk_1 = nullptr;

  if (x_non_contig) { to_fp16_vk_0 = v_vk_get_cpy_pipeline(ctx, src0, nullptr, f16_type); }
  else { to_fp16_vk_0 = v_vk_get_to_fp16(ctx, src0->type); }
  if (y_non_contig) { to_fp16_vk_1 = v_vk_get_cpy_pipeline(ctx, src1, nullptr, f16_type); }
  else { to_fp16_vk_1 = v_vk_get_to_fp16(ctx, src1->type); }
  V_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr); // NOLINT
  V_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr); // NOLINT

  if (dryrun) {
    const uint64_t x_sz_upd = x_sz * ne02 * ne03;
    const uint64_t y_sz_upd = y_sz * ne12 * ne13;
    if (
      (qx_needs_dequant && x_sz_upd > ctx->device->properties.limits.maxStorageBufferRange) ||
      (qy_needs_dequant && y_sz_upd > ctx->device->properties.limits.maxStorageBufferRange)) { v_ABORT("Requested preallocation size is too large"); }
    if (qx_needs_dequant && ctx->prealloc_size_x < x_sz_upd) { ctx->prealloc_size_x = x_sz_upd; }
    if (qy_needs_dequant && ctx->prealloc_size_y < y_sz_upd) { ctx->prealloc_size_y = y_sz_upd; }

    // Request descriptor sets
    v_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    if (qx_needs_dequant) { v_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1); }
    if (qy_needs_dequant) { v_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1); }
    return;
  }

  vk_buffer d_D               = dst_buf_ctx->dev_buffer;
  const uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
  V_ASSERT(d_D != nullptr);
  vk_buffer d_X;
  uint64_t x_buf_offset = 0;
  vk_buffer d_Y;
  uint64_t y_buf_offset = 0;
  if (!src0_uma) {
    d_Qx          = src0_buf_ctx->dev_buffer;
    qx_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
    V_ASSERT(d_Qx != nullptr);
  }
  if (!src1_uma) {
    d_Qy          = src1_buf_ctx->dev_buffer;
    qy_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
    V_ASSERT(d_Qy != nullptr);
  }
  if (!ids_uma) {
    d_ids          = ids_buf_ctx->dev_buffer;
    ids_buf_offset = vk_tensor_offset(ids) + ids->view_offs;
    V_ASSERT(d_ids != nullptr);
  }
  if (qx_needs_dequant) {
    d_X = ctx->prealloc_x;
    V_ASSERT(d_X->size >= x_sz * ne02 * ne03);
  }
  else {
    d_X          = d_Qx;
    x_buf_offset = qx_buf_offset;
    V_ASSERT(qx_sz == x_sz);
  }
  if (qy_needs_dequant) {
    d_Y = ctx->prealloc_y;
    V_ASSERT(d_Y->size >= y_sz * ne12 * ne13);
  }
  else {
    d_Y          = d_Qy;
    y_buf_offset = qy_buf_offset;
    V_ASSERT(qy_sz == y_sz);
  }

  if (x_non_contig || qx_needs_dequant) { if (ctx->prealloc_x_need_sync) { vk_sync_buffers(ctx, subctx); } }

  if (x_non_contig) {
    v_vk_cpy_to_contiguous(ctx,
                           subctx,
                           to_fp16_vk_0,
                           src0,
                           v_vk_subbuffer(ctx, d_Qx, qx_buf_offset),
                           v_vk_subbuffer(ctx, d_X, 0));
  }
  else if (qx_needs_dequant) {
    const std::vector<uint32_t> pc = {
      (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(nelements(src0))
    };
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           to_fp16_vk_0,
                           {
                             vk_sub_buffer{d_Qx, qx_buf_offset, qx_sz * ne02 * ne03},
                             vk_sub_buffer{d_X, 0, x_sz * ne02 * ne03}
                           },
                           pc,
                           {(uint32_t)(x_ne * ne02 * ne03), 1, 1});
    vk_sync_buffers(ctx, subctx);
  }
  if (y_non_contig) {
    if (ctx->prealloc_y_last_pipeline_used != to_fp16_vk_1.get() ||
      ctx->prealloc_y_last_tensor_used != src1) {
      if (ctx->prealloc_y_need_sync) { vk_sync_buffers(ctx, subctx); }
      v_vk_cpy_to_contiguous(ctx,
                             subctx,
                             to_fp16_vk_1,
                             src1,
                             v_vk_subbuffer(ctx, d_Qy, qy_buf_offset),
                             v_vk_subbuffer(ctx, d_Y, 0));
      ctx->prealloc_y_last_pipeline_used = to_fp16_vk_1.get();
      ctx->prealloc_y_last_tensor_used   = src1;
    }
  }

  uint32_t stride_batch_x = ne00 * ne01;
  uint32_t stride_batch_y = ne10 * ne11;

  if (!v_vk_dim01_contiguous(src0) && !qx_needs_dequant) { stride_batch_x = src0->nb[0] / v_type_size(src0->type); }

  if (!v_vk_dim01_contiguous(src1) && !qy_needs_dequant) { stride_batch_y = src1->nb[0] / v_type_size(src1->type); }

  // compute
  v_vk_matmul_id(
    ctx,
    subctx,
    pipeline,
    {d_X, x_buf_offset, x_sz * ne02 * ne03},
    {d_Y, y_buf_offset, y_sz * ne12 * ne13},
    {d_D, d_buf_offset, d_sz * ne22 * ne23},
    {d_ids, ids_buf_offset, ids_sz},
    ne01,
    ne21,
    ne10,
    ne10,
    ne10,
    ne01,
    stride_batch_x,
    stride_batch_y,
    ne20 * ne21,
    n_as,
    nei0,
    nei1,
    nbi1 / v_type_size(ids->type),
    ne11,
    padded_n
  ); // NOLINT

  if (x_non_contig || qx_needs_dequant) { ctx->prealloc_x_need_sync = true; }
  if (y_non_contig) { ctx->prealloc_y_need_sync = true; }
}

void v_vk_mul_mat_vec_id_q_f16(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                               const v_tensor* src1, const v_tensor* ids, v_tensor* dst,
                               bool dryrun = false) {
  VK_LOG_DEBUG(
    "v_vk_mul_mat_vec_id_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->
    ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] <<
    ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] <<
    ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1="
    << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << ids << ", name=" << ids->name << ", type=" << ids->type << ", ne0=" << ids->ne[0] << ", ne1="
    << ids->ne[1] << ", ne2=" << ids->ne[2] << ", ne3=" << ids->ne[3] << ", nb0=" << ids->nb[0] << ", nb1=" << ids->nb[1
    ] << ", nb2=" << ids->nb[2] << ", nb3=" << ids->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1="
    << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1
    ] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << (dryrun ? "dryrun" : "") << ")");
  V_ASSERT(v_vk_dim01_contiguous(src0) || src0->type == v_TYPE_F32 || src0->type == v_TYPE_F16 || src0->type == v_TYPE_BF16); // NOLINT
  V_ASSERT(v_vk_dim01_contiguous(src1) || src1->type == v_TYPE_F32 || src1->type == v_TYPE_F16); // NOLINT
  V_ASSERT(ids->type == v_TYPE_I32);

  const uint64_t ne00 = src0->ne[0];
  const uint64_t ne01 = src0->ne[1];
  const uint64_t ne02 = src0->ne[2];
  const uint64_t ne03 = src0->ne[3];

  const uint64_t ne10 = src1->ne[0];
  const uint64_t ne11 = src1->ne[1];
  const uint64_t ne12 = src1->ne[2];
  const uint64_t ne13 = src1->ne[3];

  const uint64_t nei0 = ids->ne[0];
  const uint64_t nei1 = ids->ne[1];

  const uint64_t nbi2 = ids->nb[2];

  V_ASSERT(nei1 == 1);

  const uint64_t ne20 = dst->ne[0];
  const uint64_t ne21 = dst->ne[1];
  const uint64_t ne22 = dst->ne[2];
  const uint64_t ne23 = dst->ne[3];

  v_backend_vk_buffer_ctx* dst_buf_ctx  = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  v_backend_vk_buffer_ctx* src0_buf_ctx = (v_backend_vk_buffer_ctx*)src0->buffer->context;
  v_backend_vk_buffer_ctx* src1_buf_ctx = (v_backend_vk_buffer_ctx*)src1->buffer->context;
  v_backend_vk_buffer_ctx* ids_buf_ctx  = (v_backend_vk_buffer_ctx*)ids->buffer->context;

  vk_buffer d_Qx        = nullptr;
  size_t qx_buf_offset  = 0;
  vk_buffer d_Qy        = nullptr;
  size_t qy_buf_offset  = 0;
  vk_buffer d_ids       = nullptr;
  size_t ids_buf_offset = 0;

  bool src0_uma = false;
  bool src1_uma = false;
  bool ids_uma  = false;

  if (ctx->device->uma) {
    vk_get_host_buffer(ctx->device, src0->data, d_Qx, qx_buf_offset);
    vk_get_host_buffer(ctx->device, src1->data, d_Qy, qy_buf_offset);
    vk_get_host_buffer(ctx->device, ids->data, d_ids, ids_buf_offset);
    src0_uma = d_Qx != nullptr;
    src1_uma = d_Qy != nullptr;
    ids_uma  = d_ids != nullptr;
  }

  const bool x_non_contig = !v_vk_dim01_contiguous(src0);
  const bool y_non_contig = !v_vk_dim01_contiguous(src1);

  const bool f16_f32_kernel = src1->type == v_TYPE_F32;

  const bool qx_needs_dequant = x_non_contig;
  const bool qy_needs_dequant = (src1->type != v_TYPE_F16 && !f16_f32_kernel) || y_non_contig;

  // Not implemented
  V_ASSERT(y_non_contig || !qy_needs_dequant); // NOLINT

  const uint64_t x_ne = ne01 * ne00;
  const uint64_t y_ne = ne11 * ne10;
  const uint64_t d_ne = ne21 * ne20;

  const uint64_t qx_sz = mmlVKAlignSize(v_type_size(src0->type) * x_ne / blockSize(src0->type),
                                        ctx->device->properties.limits.minStorageBufferOffsetAlignment);
  const uint64_t qy_sz = v_type_size(src1->type) * y_ne / blockSize(src1->type);
  const uint64_t x_sz  = x_non_contig
                           ? mmlVKAlignSize(v_type_size(src0->type) * x_ne,
                                            ctx->device->properties.limits.minStorageBufferOffsetAlignment)
                           : qx_sz;
  const uint64_t y_sz = f16_f32_kernel
                          ? sizeof(float) * y_ne
                          : sizeof(v_fp16_t) * y_ne;
  const uint64_t ids_sz = nbi2;
  const uint64_t d_sz   = sizeof(float) * d_ne;

  vk_pipeline to_fp16_vk_0 = nullptr;
  vk_pipeline to_fp16_vk_1 = nullptr;
  if (x_non_contig) { to_fp16_vk_0 = v_vk_get_cpy_pipeline(ctx, src0, nullptr, src0->type); }
  if (y_non_contig) { to_fp16_vk_1 = v_vk_get_cpy_pipeline(ctx, src1, nullptr, src1->type); }
  else { to_fp16_vk_1 = v_vk_get_to_fp16(ctx, src1->type); }
  vk_pipeline dmmv = v_vk_get_dequantize_mul_mat_vec_id(ctx, src0->type, src1->type);
  V_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr); // NOLINT
  V_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr); // NOLINT
  V_ASSERT(dmmv != nullptr);

  if (dryrun) {
    const uint64_t x_sz_upd = x_sz * ne02 * ne03;
    const uint64_t y_sz_upd = y_sz * ne12 * ne13;
    if (
      (qx_needs_dequant && x_sz_upd > ctx->device->properties.limits.maxStorageBufferRange) ||
      (qy_needs_dequant && y_sz_upd > ctx->device->properties.limits.maxStorageBufferRange)) { v_ABORT("Requested preallocation size is too large"); }
    if (qx_needs_dequant && ctx->prealloc_size_x < x_sz_upd) { ctx->prealloc_size_x = x_sz_upd; }
    if (qy_needs_dequant && ctx->prealloc_size_y < y_sz_upd) { ctx->prealloc_size_y = y_sz_upd; }

    // Request descriptor sets
    if (qx_needs_dequant) { v_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1); }
    if (qy_needs_dequant) { v_pipeline_request_descriptor_sets(ctx, to_fp16_vk_1, 1); }
    v_pipeline_request_descriptor_sets(ctx, dmmv, 1);
    return;
  }

  vk_buffer d_D               = dst_buf_ctx->dev_buffer;
  const uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
  V_ASSERT(d_D != nullptr);
  vk_buffer d_X;
  uint64_t x_buf_offset = 0;
  vk_buffer d_Y;
  uint64_t y_buf_offset = 0;
  if (!src0_uma) {
    d_Qx          = src0_buf_ctx->dev_buffer;
    qx_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
    V_ASSERT(d_Qx != nullptr);
  }
  if (!src1_uma) {
    d_Qy          = src1_buf_ctx->dev_buffer;
    qy_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
    V_ASSERT(d_Qy != nullptr);
  }
  if (!ids_uma) {
    d_ids          = ids_buf_ctx->dev_buffer;
    ids_buf_offset = vk_tensor_offset(ids) + ids->view_offs;
    V_ASSERT(d_ids != nullptr);
  }
  if (qx_needs_dequant) { d_X = ctx->prealloc_x; }
  else {
    d_X          = d_Qx;
    x_buf_offset = qx_buf_offset;
    V_ASSERT(qx_sz == x_sz);
  }
  if (qy_needs_dequant) { d_Y = ctx->prealloc_y; }
  else {
    d_Y          = d_Qy;
    y_buf_offset = qy_buf_offset;
    V_ASSERT(qy_sz == y_sz);
  }

  if (x_non_contig) { if (ctx->prealloc_x_need_sync) { vk_sync_buffers(ctx, subctx); } }

  if (x_non_contig) {
    V_ASSERT(
      x_sz == mmlVKAlignSize(v_type_size(src0->type) * x_ne, ctx->device->properties.limits.
        minStorageBufferOffsetAlignment));
    v_vk_cpy_to_contiguous(ctx,
                           subctx,
                           to_fp16_vk_0,
                           src0,
                           v_vk_subbuffer(ctx, d_Qx, qx_buf_offset),
                           v_vk_subbuffer(ctx, d_X, 0));
  }
  if (y_non_contig) {
    V_ASSERT(y_sz == v_type_size(src1->type) * y_ne);
    if (ctx->prealloc_y_last_pipeline_used != to_fp16_vk_1.get() ||
      ctx->prealloc_y_last_tensor_used != src1) {
      if (ctx->prealloc_y_need_sync) { vk_sync_buffers(ctx, subctx); }
      v_vk_cpy_to_contiguous(ctx,
                             subctx,
                             to_fp16_vk_1,
                             src1,
                             v_vk_subbuffer(ctx, d_Qy, qy_buf_offset),
                             v_vk_subbuffer(ctx, d_Y, 0));
      ctx->prealloc_y_last_pipeline_used = to_fp16_vk_1.get();
      ctx->prealloc_y_last_tensor_used   = src1;
    }
  }

  uint32_t stride_batch_y = ne10 * ne11;

  if (!v_vk_dim01_contiguous(src1) && !qy_needs_dequant) { stride_batch_y = src1->nb[0] / v_type_size(src1->type); }

  const uint32_t max_groups_x = ctx->device->properties.limits.maxComputeWorkGroupCount[0];

  uint32_t groups_x = ne01;
  uint32_t groups_z = 1;

  if (ne01 > max_groups_x) {
    groups_z = 64;
    groups_x = CEIL_DIV(groups_x, groups_z);
  }

  // compute
  const vk_mat_vec_id_push_constants pc = {
    (uint32_t)ne00, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne01,
    (uint32_t)x_ne, stride_batch_y, (uint32_t)(ne20 * ne21),
    (uint32_t)nei0, (uint32_t)ne11,
  };
  v_vk_dispatch_pipeline(ctx,
                         subctx,
                         dmmv,
                         {
                           vk_sub_buffer{d_X, x_buf_offset, x_sz * ne02 * ne03},
                           vk_sub_buffer{d_Y, y_buf_offset, y_sz * ne12 * ne13},
                           vk_sub_buffer{d_D, d_buf_offset, d_sz * ne22 * ne23},
                           vk_sub_buffer{d_ids, ids_buf_offset, ids_sz}
                         },
                         pc,
                         {groups_x, (uint32_t)nei0, groups_z});

  if (x_non_contig) { ctx->prealloc_x_need_sync = true; }
  if (y_non_contig) { ctx->prealloc_y_need_sync = true; }
}

void v_vk_mul_mat_id(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                     const v_tensor* src1, const v_tensor* src2, v_tensor* dst, bool dryrun = false) {
  VK_LOG_DEBUG("v_vk_mul_mat_id(" << src0 << ", " << src1 << ", " << src2 << ", " << dst << ")");
  if (src2->ne[1] == 1 && (src0->type == v_TYPE_F32 || src0->type == v_TYPE_F16 ||
    v_is_quantized(src0->type))) { v_vk_mul_mat_vec_id_q_f16(ctx, subctx, src0, src1, src2, dst, dryrun); }
  else { v_vk_mul_mat_id_q_f16(ctx, subctx, src0, src1, src2, dst, dryrun); }
}

bool v_vk_flash_attn_scalar_shmem_support(const vk_device& device, const uint32_t hsk, uint32_t hsv) {
  // Needs to be kept up to date on shader changes
  v_UNUSED(hsv);
  const uint32_t wg_size = scalar_flash_attention_workgroup_size;
  const uint32_t Br      = get_fa_scalar_num_large_rows(hsv);
  const uint32_t Bc      = scalar_flash_attention_Bc;

  const uint32_t tmpsh   = wg_size * sizeof(float);
  const uint32_t tmpshv4 = wg_size * 4 * sizeof(float);

  const uint32_t masksh = Bc * Br * sizeof(float);

  const uint32_t Qf = Br * (hsk / 4 + 2) * 4 * sizeof(float);

  const uint32_t total_size = tmpsh + tmpshv4 + masksh + Qf;
  const bool supported      = total_size <= device->properties.limits.maxComputeSharedMemorySize;

  VK_LOG_DEBUG(
    "v_vk_flash_attn_coopmat_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", total_size=" << total_size <<
    ", supported=" << supported);

  return supported;
}

bool v_vk_flash_attn_coopmat_shmem_support(const vk_device& device, const uint32_t hsk, uint32_t hsv,
                                           bool f32acc) {
  // Needs to be kept up to date on shader changes
  v_UNUSED(hsv);
  const uint32_t wg_size = scalar_flash_attention_workgroup_size;
  const uint32_t Br      = coopmat1_flash_attention_num_large_rows;
  const uint32_t Bc      = scalar_flash_attention_Bc;

  const uint32_t hsk_pad = ROUNDUP_POW2(hsk, 16);

  const uint32_t acctype = f32acc
                             ? 4
                             : 2;
  const uint32_t f16vec4 = 8;

  const uint32_t tmpsh   = wg_size * sizeof(float);
  const uint32_t tmpshv4 = wg_size * 4 * acctype;

  const uint32_t qstride = hsk_pad / 4 + 2;
  const uint32_t Qf      = Br * qstride * f16vec4;

  const uint32_t sfshstride = (hsk <= 128)
                                ? (Br + 8)
                                : Br;
  const uint32_t sfsh = Bc * sfshstride * acctype;

  const uint32_t kshstride = hsk_pad / 4 + 2;
  const uint32_t ksh       = Bc * kshstride * f16vec4;

  const uint32_t slope = Br * sizeof(float);

  const uint32_t total_size = tmpsh + tmpshv4 + Qf + sfsh + ksh + slope;
  const bool supported      = total_size <= device->properties.limits.maxComputeSharedMemorySize;

  VK_LOG_DEBUG(
    "v_vk_flash_attn_coopmat_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", f32acc=" << f32acc <<
    ", total_size=" << total_size << ", supported=" << supported);

  return supported;
}

void v_vk_flash_attn(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* q,
                     const v_tensor* k, const v_tensor* v, const v_tensor* mask,
                     const v_tensor* sinks, v_tensor* dst, bool dryrun = false) {
  VK_LOG_DEBUG(
    "v_vk_flash_attn((" << q << ", name=" << q->name << ", type=" << q->type << ", ne0=" << q->ne[0] << ", ne1=" << q
    ->ne[1] << ", ne2=" << q->ne[2] << ", ne3=" << q->ne[3] << ", nb0=" << q->nb[0] << ", nb1=" << q->nb[1] << ", nb2="
    << q->nb[2] << ", nb3=" << q->nb[3];
    std::cerr << "), (" << k << ", name=" << k->name << ", type=" << k->type << ", ne0=" << k->ne[0] << ", ne1=" << k->
    ne[1] << ", ne2=" << k->ne[2] << ", ne3=" << k->ne[3] << ", nb0=" << k->nb[0] << ", nb1=" << k->nb[1] << ", nb2=" <<
    k->nb[2] << ", nb3=" << k->nb[3];
    std::cerr << "), (" << v << ", name=" << v->name << ", type=" << v->type << ", ne0=" << v->ne[0] << ", ne1=" << v->
    ne[1] << ", ne2=" << v->ne[2] << ", ne3=" << v->ne[3] << ", nb0=" << v->nb[0] << ", nb1=" << v->nb[1] << ", nb2=" <<
    v->nb[2] << ", nb3=" << v->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1="
    << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1
    ] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    if (sinks) {
    std::cerr << "), (" << sinks << ", name=" << sinks->name << ", type=" << sinks->type << ", ne0=" << sinks->ne[0] <<
    ", ne1=" << sinks->ne[1] << ", ne2=" << sinks->ne[2] << ", ne3=" << sinks->ne[3] << ", nb0=" << sinks->nb[0] <<
    ", nb1=" << sinks->nb[1] << ", nb2=" << sinks->nb[2] << ", nb3=" << sinks->nb[3];
    }
    std::cerr << "), " << (dryrun ? "dryrun" : "") << ")");

  v_TENSOR_LOCALS(int64_t, neq, q, ne)
  v_TENSOR_LOCALS(size_t, nbq, q, nb)
  v_TENSOR_LOCALS(int64_t, nek, k, ne)
  v_TENSOR_LOCALS(size_t, nbk, k, nb)
  v_TENSOR_LOCALS(int64_t, nev, v, ne)
  v_TENSOR_LOCALS(size_t, nbv, v, nb)
  v_TENSOR_LOCALS(int64_t, ne, dst, ne)
  v_TENSOR_LOCALS(size_t, nb, dst, nb)

  const uint32_t nem1 = mask
                          ? mask->ne[1]
                          : 0;
  const uint32_t nem2 = mask
                          ? mask->ne[2]
                          : 0;
  const uint32_t nem3 = mask
                          ? mask->ne[3]
                          : 0;

  const uint32_t HSK = nek0;
  const uint32_t HSV = nev0;
  uint32_t N         = neq1;
  const uint32_t KV  = nek1;

  V_ASSERT(ne0 == HSV);
  V_ASSERT(ne2 == N);

  // input tensor rows must be contiguous
  V_ASSERT(nbq0 == v_type_size(q->type));
  V_ASSERT(nbk0 == v_type_size(k->type));
  V_ASSERT(nbv0 == v_type_size(v->type));

  V_ASSERT(neq0 == HSK);

  V_ASSERT(neq1 == N);

  V_ASSERT(nev1 == nek1);

  // dst cannot be transposed or permuted
  V_ASSERT(nb0 == sizeof(float));
  V_ASSERT(nb0 <= nb1);
  V_ASSERT(nb1 <= nb2);
  V_ASSERT(nb2 <= nb3);

  assert(dst->type == v_TYPE_F32);
  assert(q->type == v_TYPE_F32);
  assert(k->type == v->type);

  FaCodePath path = ctx->device->coopmat2
                      ? FA_COOPMAT2
                      : ctx->device->coopmat1_fa_support
                      ? FA_COOPMAT1
                      : FA_SCALAR;

  if (path == FA_COOPMAT1) {
    const bool coopmat_shape_supported = (dst->op_params[3] == v_PREC_F32 && ctx->device->
                                                                                  coopmat_support_16x16x16_f32acc) ||
    (dst->op_params[3] != v_PREC_F32 && ctx->device->
                                             coopmat_support_16x16x16_f16acc);

    const bool coopmat_shmem_supported = v_vk_flash_attn_coopmat_shmem_support(
      ctx->device,
      HSK,
      HSV,
      dst->op_params[3] == v_PREC_F32);

    if (!coopmat_shape_supported || !coopmat_shmem_supported) { path = FA_SCALAR; }
  }

  uint32_t gqa_ratio    = 1;
  uint32_t qk_ratio     = neq2 / nek2;
  uint32_t workgroups_x = (uint32_t)neq1;
  uint32_t workgroups_y = (uint32_t)neq2;
  uint32_t workgroups_z = (uint32_t)neq3;

  // For scalar/coopmat1 FA, we can use the "large" size to accommodate qga.
  // For coopmat2 FA, we always use the small size (which is still pretty large for gqa).
  uint32_t max_gqa;
  switch (path) {
    case FA_SCALAR:
    case FA_COOPMAT1:
      // We may switch from coopmat1 to scalar, so use the scalar limit for both
      max_gqa = get_fa_scalar_num_large_rows(HSV);
      break;
    case FA_COOPMAT2:
      max_gqa = get_fa_num_small_rows(FA_COOPMAT2);
      break;
    default:
      V_ASSERT(0);
  }

  if (N == 1 && qk_ratio > 1 && qk_ratio <= max_gqa &&
    qk_ratio * nek2 == neq2 && nek2 == nev2 && nem2 <= 1) {
    // grouped query attention - make the N dimension equal to gqa_ratio, reduce
    // workgroups proportionally in y dimension. The shader will detect gqa_ratio > 1
    // and change addressing calculations to index Q's dimension 2.
    gqa_ratio = qk_ratio;
    N         = gqa_ratio;
    workgroups_y /= N;
  }

  bool small_rows = N <= get_fa_num_small_rows(path);

  // coopmat1 does not actually support "small rows" (it needs 16 rows).
  // So use scalar instead.
  if (small_rows && path == FA_COOPMAT1) { path = FA_SCALAR; }

  // scalar is faster than coopmat2 when N==1
  if (N == 1 && path == FA_COOPMAT2) { path = FA_SCALAR; }

  // with large hsk/hsv, scalar path may need to use small_rows to fit in shared memory
  if (path == FA_SCALAR &&
    !v_vk_flash_attn_scalar_shmem_support(ctx->device, HSK, HSV)) { small_rows = true; }

  const uint32_t q_stride = (uint32_t)(nbq1 / v_type_size(q->type));
  uint32_t k_stride       = (uint32_t)(nbk1 / v_type_size(k->type));
  uint32_t v_stride       = (uint32_t)(nbv1 / v_type_size(v->type));

  // For F32, the shader treats it as a block of size 4 (for vec4 loads)
  if (k->type == v_TYPE_F32) { k_stride /= 4; }
  if (v->type == v_TYPE_F32) { v_stride /= 4; }

  uint32_t alignment = fa_align(path, HSK, HSV, k->type, small_rows);
  bool aligned       = (KV % alignment) == 0 &&
    // the "aligned" shader variant will forcibly align strides, for performance
    (q_stride & 7) == 0 && (k_stride & 7) == 0 && (v_stride & 7) == 0;

  // Need to use the coopmat2 variant that clamps loads when HSK/HSV aren't sufficiently aligned.
  if (((HSK | HSV) % 16) != 0 && path == FA_COOPMAT2) { aligned = false; }

  bool f32acc = path == FA_SCALAR || dst->op_params[3] == v_PREC_F32;

  vk_fa_pipeline_state fa_pipeline_state(HSK, HSV, small_rows, path, aligned, f32acc);

  vk_pipeline pipeline = nullptr;

  auto& pipelines = ctx->device->pipeline_flash_attn_f32_f16[k->type];
  auto it         = pipelines.find(fa_pipeline_state);
  if (it != pipelines.end()) { pipeline = it->second; }
  else { pipelines[fa_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>(); }

  assert(pipeline);

  uint32_t split_kv = KV;
  uint32_t split_k  = 1;

  // Use a placeholder core count if one isn't available. split_k is a big help for perf.
  const uint32_t shader_core_count = ctx->device->shader_core_count
                                       ? ctx->device->shader_core_count
                                       : 16;

  // Try to use split_k when KV is large enough to be worth the overhead
  if (workgroups_x == 1 && shader_core_count > 0) {
    // Try to run two workgroups per SM.
    split_k = shader_core_count * 2 / (workgroups_y * workgroups_z);
    if (split_k > 1) {
      // Try to evenly split KV into split_k chunks, but it needs to be a multiple
      // of "align", so recompute split_k based on that.
      split_kv     = ROUNDUP_POW2(std::max(1u, KV / split_k), alignment);
      split_k      = CEIL_DIV(KV, split_kv);
      workgroups_x = split_k;
    }
  }

  // Reserve space for split_k temporaries. For each split x batch, we need to store the O matrix (D x ne1)
  // and the per-row m and L values (ne1 rows). We store all the matrices first, followed by the rows.
  const uint64_t split_k_size = split_k > 1
                                  ? (HSV * ne1 * sizeof(float) + ne1 * sizeof(float) * 2) * split_k * ne3
                                  : 0;
  if (split_k_size > ctx->device->properties.limits.maxStorageBufferRange) { v_ABORT("Requested preallocation size is too large"); }
  if (ctx->prealloc_size_split_k < split_k_size) { ctx->prealloc_size_split_k = split_k_size; }

  if (dryrun) {
    // Request descriptor sets
    v_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    if (split_k > 1) { v_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_flash_attn_split_k_reduce, 1); }
    return;
  }

  float scale         = 1.0f;
  float max_bias      = 0.0f;
  float logit_softcap = 0.0f;

  memcpy(&scale, (const float*)dst->op_params + 0, sizeof(float));
  memcpy(&max_bias, (const float*)dst->op_params + 1, sizeof(float));
  memcpy(&logit_softcap, (const float*)dst->op_params + 2, sizeof(float));

  if (logit_softcap != 0) { scale /= logit_softcap; }

  const uint32_t n_head_kv   = neq2;
  const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head_kv));
  const float m0             = powf(2.0f, -(max_bias) / n_head_log2);
  const float m1             = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

  vk_buffer d_Q       = nullptr, d_K    = nullptr, d_V    = nullptr, d_D    = nullptr, d_M    = nullptr, d_S    = nullptr;
  size_t q_buf_offset = 0, k_buf_offset = 0, v_buf_offset = 0, d_buf_offset = 0, m_buf_offset = 0, s_buf_offset = 0;

  bool Q_uma = false, K_uma = false, V_uma = false, D_uma = false, M_uma = false, S_uma = false;

  if (ctx->device->uma) {
    vk_get_host_buffer(ctx->device, q->data, d_Q, q_buf_offset);
    vk_get_host_buffer(ctx->device, k->data, d_K, k_buf_offset);
    vk_get_host_buffer(ctx->device, v->data, d_V, v_buf_offset);
    vk_get_host_buffer(ctx->device, dst->data, d_D, d_buf_offset);
    Q_uma = d_Q != nullptr;
    K_uma = d_K != nullptr;
    V_uma = d_V != nullptr;
    D_uma = d_D != nullptr;
    if (mask) {
      vk_get_host_buffer(ctx->device, mask->data, d_M, m_buf_offset);
      M_uma = d_M != nullptr;
    }
    if (sinks) {
      vk_get_host_buffer(ctx->device, sinks->data, d_S, s_buf_offset);
      S_uma = d_S != nullptr;
    }
  }


  v_backend_vk_buffer_ctx* d_buf_ctx = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  v_backend_vk_buffer_ctx* q_buf_ctx = (v_backend_vk_buffer_ctx*)q->buffer->context;
  v_backend_vk_buffer_ctx* k_buf_ctx = (v_backend_vk_buffer_ctx*)k->buffer->context;
  v_backend_vk_buffer_ctx* v_buf_ctx = (v_backend_vk_buffer_ctx*)v->buffer->context;

  if (!Q_uma) {
    d_Q          = q_buf_ctx->dev_buffer;
    q_buf_offset = vk_tensor_offset(q) + q->view_offs;
  }
  if (!K_uma) {
    d_K          = k_buf_ctx->dev_buffer;
    k_buf_offset = vk_tensor_offset(k) + k->view_offs;
  }
  if (!V_uma) {
    d_V          = v_buf_ctx->dev_buffer;
    v_buf_offset = vk_tensor_offset(v) + v->view_offs;
  }
  if (!D_uma) {
    d_D          = d_buf_ctx->dev_buffer;
    d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
  }

  if (!M_uma) {
    d_M          = d_Q;
    m_buf_offset = q_buf_offset;
    if (mask) {
      v_backend_vk_buffer_ctx* m_buf_ctx = (v_backend_vk_buffer_ctx*)mask->buffer->context;
      d_M                                = m_buf_ctx->dev_buffer;
      m_buf_offset                       = vk_tensor_offset(mask) + mask->view_offs;
    }
  }

  if (!S_uma) {
    d_S          = d_Q;
    s_buf_offset = q_buf_offset;
    if (sinks) {
      v_backend_vk_buffer_ctx* s_buf_ctx = (v_backend_vk_buffer_ctx*)sinks->buffer->context;
      d_S                                = s_buf_ctx->dev_buffer;
      s_buf_offset                       = vk_tensor_offset(sinks) + sinks->view_offs;
    }
  }

  uint32_t mask_n_head_log2 = ((sinks != nullptr) << 24) | ((mask != nullptr) << 16) | n_head_log2;

  const vk_flash_attn_push_constants pc = {
    N, KV,
    (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
    (uint32_t)neq2, (uint32_t)neq3,
    (uint32_t)nek2, (uint32_t)nek3,
    (uint32_t)nev2, (uint32_t)nev3,
    nem1, nem2, nem3,
    q_stride, (uint32_t)nbq2, (uint32_t)nbq3,
    k_stride, (uint32_t)nbk2, (uint32_t)nbk3,
    v_stride, (uint32_t)nbv2, (uint32_t)nbv3,
    scale, max_bias, logit_softcap,
    mask_n_head_log2, m0, m1,
    gqa_ratio, split_kv, split_k
  };

  if (split_k > 1) {
    if (ctx->prealloc_split_k_need_sync) { vk_sync_buffers(ctx, subctx); }

    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             v_vk_subbuffer(ctx, d_Q, q_buf_offset),
                             v_vk_subbuffer(ctx, d_K, k_buf_offset),
                             v_vk_subbuffer(ctx, d_V, v_buf_offset),
                             v_vk_subbuffer(ctx, d_M, m_buf_offset),
                             v_vk_subbuffer(ctx, d_S, s_buf_offset),
                             v_vk_subbuffer(ctx, ctx->prealloc_split_k, 0),
                           },
                           // We only use split_k when group query attention is enabled, which means
                           // there's no more than one tile of rows (i.e. workgroups_x would have been
                           // one). We reuse workgroups_x to mean the number of splits, so we need to
                           // cancel out the divide by wg_denoms[0].
                           pc,
                           {workgroups_x * pipeline->wg_denoms[0], workgroups_y, workgroups_z});

    vk_sync_buffers(ctx, subctx);
    const std::array<uint32_t, 5> pc2 = {HSV, (uint32_t)ne1, (uint32_t)ne3, split_k, (sinks != nullptr)};
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           ctx->device->pipeline_flash_attn_split_k_reduce,
                           {
                             v_vk_subbuffer(ctx, ctx->prealloc_split_k, 0),
                             v_vk_subbuffer(ctx, d_S, s_buf_offset),
                             v_vk_subbuffer(ctx, d_D, d_buf_offset),
                           },
                           pc2,
                           {(uint32_t)ne1, HSV, (uint32_t)ne3});
    ctx->prealloc_split_k_need_sync = true;
  }
  else {
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             v_vk_subbuffer(ctx, d_Q, q_buf_offset),
                             v_vk_subbuffer(ctx, d_K, k_buf_offset),
                             v_vk_subbuffer(ctx, d_V, v_buf_offset),
                             v_vk_subbuffer(ctx, d_M, m_buf_offset),
                             v_vk_subbuffer(ctx, d_S, s_buf_offset),
                             v_vk_subbuffer(ctx, d_D, d_buf_offset),
                           },
                           pc,
                           {workgroups_x, workgroups_y, workgroups_z});
  }
}

std::array<uint32_t, 3> v_vk_get_conv_elements(const v_tensor* dst) {
  const v_tensor* src0 = dst->src[0];
  const v_tensor* src1 = dst->src[1];

  // src0 - kernel:   [KW, KH, Cin, Cout]
  // src1 - input:    [W, H, Cin, N]
  // dst - result:    [OW, OH, Cout, N]

  // Copied from ggml.c: int64_t v_calc_conv_output_size(int64_t ins, int64_t ks, int s, int p, int d)
  auto calc_conv_output_size = [](int64_t ins, int64_t ks, int s, int p, int d) -> int64_t { return (ins + 2 * p - d * (ks - 1) - 1) / s + 1; };
  // parallelize in {OW/BS_K, OH/BS_NPQ, 1}
  int64_t W    = src1->ne[0];
  int64_t H    = src1->ne[1];
  int64_t KW   = src0->ne[0];
  int64_t KH   = src0->ne[1];
  int64_t Cout = src0->ne[3];
  int64_t N    = src1->ne[3];
  int64_t OH   = calc_conv_output_size(H, KH, dst->op_params[1], dst->op_params[3], dst->op_params[5]);
  int64_t OW   = calc_conv_output_size(W, KW, dst->op_params[0], dst->op_params[2], dst->op_params[4]);
  int64_t NPQ  = N * OW * OH;

  // Tile output matrix to (K/NB_K, NPQ/NB_NPQ, 1) workgroups
  std::array<uint32_t, 3> elements = {static_cast<uint32_t>(Cout), static_cast<uint32_t>(NPQ), 1};
  return elements;
}

std::array<uint32_t, 3> v_vk_get_conv_transpose_2d_elements(const v_tensor* dst) {
  const v_tensor* src0 = dst->src[0];
  const v_tensor* src1 = dst->src[1];

  // src0 - kernel:   [KW, KH, Cout, Cin]
  // src1 - input:    [W, H, Cin, N]
  // dst - result:    [OW, OH, Cout, N]

  auto calc_conv_output_size = [](int64_t ins, int64_t ks, int s, int p, int d) -> int64_t { return (ins - 1) * s - 2 * p + (ks - 1) * d + 1; };
  // parallelize in {OW/BS_K, OH/BS_NPQ, 1}
  int64_t W    = src1->ne[0];
  int64_t H    = src1->ne[1];
  int64_t KW   = src0->ne[0];
  int64_t KH   = src0->ne[1];
  int64_t Cout = src0->ne[2];
  int64_t N    = src1->ne[3];
  int64_t OH   = calc_conv_output_size(H, KH, dst->op_params[0], 0, 1);
  int64_t OW   = calc_conv_output_size(W, KW, dst->op_params[0], 0, 1);
  int64_t NPQ  = N * OW * OH;

  // Tile output matrix to (K/NB_K, NPQ/NB_NPQ, 1) workgroups
  std::array<uint32_t, 3> elements = {static_cast<uint32_t>(Cout), static_cast<uint32_t>(NPQ), 1};
  return elements;
}

vk_pipeline v_vk_op_get_pipeline(vk_backend_ctx* ctx, const v_tensor* src0,
                                 const v_tensor* src1, const v_tensor* src2, const v_tensor* dst,
                                 v_operation op) {
  switch (op) {
    case v_OP_GET_ROWS:
      V_ASSERT(src1->type == v_TYPE_I32);
      if (dst->type == v_TYPE_F16) { return ctx->device->pipeline_get_rows[src0->type]; }
      if (dst->type == v_TYPE_F32) { return ctx->device->pipeline_get_rows_f32[src0->type]; }
      return nullptr;
    case v_OP_ACC:
      if (src0->type == v_TYPE_F32 && src1->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_acc_f32; }
      return nullptr;
    case v_OP_ADD:
    case v_OP_SUB:
    case v_OP_MUL:
    case v_OP_DIV:
      if ((src0->type != v_TYPE_F32 && src0->type != v_TYPE_F16) ||
        (src1->type != v_TYPE_F32 && src1->type != v_TYPE_F16) ||
        (dst->type != v_TYPE_F32 && dst->type != v_TYPE_F16)) { return nullptr; }
      switch (op) {
        case v_OP_ADD: {
          if (ctx->num_additional_fused_ops > 0) {
            if (ctx->do_add_rms_partials) { return ctx->device->pipeline_multi_add_rms[ctx->num_additional_fused_ops]; }
            else { return ctx->device->pipeline_multi_add[ctx->num_additional_fused_ops]; }
          }
          if (ctx->do_add_rms_partials) {
            auto pipelines = v_are_same_shape(src0, src1)
                               ? ctx->device->pipeline_add_rms_norepeat
                               : ctx->device->pipeline_add_rms;
            return pipelines[src0->type == v_TYPE_F16][src1->type == v_TYPE_F16][dst->type == v_TYPE_F16];
          }
          else {
            auto pipelines = v_are_same_shape(src0, src1)
                               ? ctx->device->pipeline_add_norepeat
                               : ctx->device->pipeline_add;
            return pipelines[src0->type == v_TYPE_F16][src1->type == v_TYPE_F16][dst->type == v_TYPE_F16];
          }
        }
        case v_OP_SUB: {
          auto pipelines = v_are_same_shape(src0, src1)
                             ? ctx->device->pipeline_sub_norepeat
                             : ctx->device->pipeline_sub;
          return pipelines[src0->type == v_TYPE_F16][src1->type == v_TYPE_F16][dst->type == v_TYPE_F16];
        }
        case v_OP_MUL: {
          auto pipelines = v_are_same_shape(src0, src1)
                             ? ctx->device->pipeline_mul_norepeat
                             : ctx->device->pipeline_mul;
          return pipelines[src0->type == v_TYPE_F16][src1->type == v_TYPE_F16][dst->type == v_TYPE_F16];
        }
        case v_OP_DIV: {
          auto pipelines = v_are_same_shape(src0, src1)
                             ? ctx->device->pipeline_div_norepeat
                             : ctx->device->pipeline_div;
          return pipelines[src0->type == v_TYPE_F16][src1->type == v_TYPE_F16][dst->type == v_TYPE_F16];
        }
        default:
          break;
      }
      return nullptr;
    case v_OP_ADD_ID:
      if (src0->type == v_TYPE_F32 && src1->type == v_TYPE_F32 && src2->type == v_TYPE_I32 && dst->type ==
        v_TYPE_F32) { return ctx->device->pipeline_add_id_f32; }
      return nullptr;
    case v_OP_CONCAT:
      if (src0->type == v_TYPE_F32 && src1->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_concat_f32; }
      if (src0->type == v_TYPE_F16 && src1->type == v_TYPE_F16 && dst->type == v_TYPE_F16) { return ctx->device->pipeline_concat_f16; }
      if (src0->type == v_TYPE_I32 && src1->type == v_TYPE_I32 && dst->type == v_TYPE_I32) { return ctx->device->pipeline_concat_i32; }
      return nullptr;
    case v_OP_UPSCALE:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) {
        v_scale_mode mode = (v_scale_mode)(v_get_op_params_i32(dst, 0) & 0xFF);
        switch (mode) {
          case v_SCALE_MODE_NEAREST:
            return ctx->device->pipeline_upscale_nearest_f32;
          case v_SCALE_MODE_BILINEAR:
            return ctx->device->pipeline_upscale_bilinear_f32;
          default:
            return nullptr;
        }
      }
      return nullptr;
    case V_OP_SCALE:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_scale_f32; }
      return nullptr;
    case v_OP_SQR:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_sqr_f32; }
      return nullptr;
    case v_OP_SQRT:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_sqrt_f32; }
      return nullptr;
    case v_OP_SIN:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_sin_f32; }
      return nullptr;
    case v_OP_COS:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_cos_f32; }
      return nullptr;
    case v_OP_CLAMP:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_clamp_f32; }
      return nullptr;
    case v_OP_PAD:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_pad_f32; }
      return nullptr;
    case v_OP_ROLL:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_roll_f32; }
      return nullptr;
    case v_OP_REPEAT:
      if (v_type_size(src0->type) == sizeof(float) && v_type_size(dst->type) == sizeof(float)) { return ctx->device->pipeline_repeat_f32; }
      return nullptr;
    case v_OP_REPEAT_BACK:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_repeat_back_f32; }
      return nullptr;
    case v_OP_CPY:
    case v_OP_CONT:
    case v_OP_DUP:
      return v_vk_get_cpy_pipeline(ctx, src0, dst, dst->type);
    case v_OP_SET_ROWS:
      if (src1->type == v_TYPE_I64) { return ctx->device->pipeline_set_rows_i64[dst->type]; }
      else { return ctx->device->pipeline_set_rows_i32[dst->type]; }
    case v_OP_SILU_BACK:
      if (src0->type == v_TYPE_F32 && src1->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_silu_back_f32; }
      return nullptr;
    case v_OP_NORM:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_norm_f32; }
      return nullptr;
    case v_OP_GROUP_NORM:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_group_norm_f32; }
      return nullptr;
    case v_OP_RMS_NORM:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) {
        if (ctx->do_add_rms_partials) {
          return ctx->num_additional_fused_ops > 0
                   ? ctx->device->pipeline_rms_norm_mul_partials_f32
                   : ctx->device->pipeline_rms_norm_partials_f32;
        }
        else {
          return ctx->num_additional_fused_ops > 0
                   ? ctx->device->pipeline_rms_norm_mul_f32
                   : ctx->device->pipeline_rms_norm_f32;
        }
      }
      return nullptr;
    case v_OP_RMS_NORM_BACK:
      if (src0->type == v_TYPE_F32 && src1->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_rms_norm_back_f32; }
      return nullptr;
    case v_OP_L2_NORM:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_l2_norm_f32; }
      return nullptr;
    case v_OP_UNARY:
      if ((src0->type != v_TYPE_F32 && src0->type != v_TYPE_F16) ||
        (dst->type != v_TYPE_F32 && dst->type != v_TYPE_F16) ||
        (src0->type != dst->type)) { return nullptr; }

      switch (v_get_unary_op(dst)) {
        case v_UNARY_OP_EXP:
          return ctx->device->pipeline_exp[dst->type == v_TYPE_F16];

        case v_UNARY_OP_LOG:
          return ctx->device->pipeline_log[dst->type == v_TYPE_F16];

        case v_UNARY_OP_SILU:
          return ctx->device->pipeline_silu[dst->type == v_TYPE_F16];
        case v_UNARY_OP_GELU:
          return ctx->device->pipeline_gelu[dst->type == v_TYPE_F16];
        case v_UNARY_OP_GELU_ERF:
          return ctx->device->pipeline_gelu_erf[dst->type == v_TYPE_F16];
        case v_UNARY_OP_GELU_QUICK:
          return ctx->device->pipeline_gelu_quick[dst->type == v_TYPE_F16];
        case v_UNARY_OP_RELU:
          return ctx->device->pipeline_relu[dst->type == v_TYPE_F16];
        case v_UNARY_OP_TANH:
          return ctx->device->pipeline_tanh[dst->type == v_TYPE_F16];
        case v_UNARY_OP_SIGMOID:
          return ctx->device->pipeline_sigmoid[dst->type == v_TYPE_F16];
        case v_UNARY_OP_HARDSIGMOID:
          return ctx->device->pipeline_hardsigmoid[dst->type == v_TYPE_F16];
        case v_UNARY_OP_HARDSWISH:
          return ctx->device->pipeline_hardswish[dst->type == v_TYPE_F16];
        default:
          break;
      }
      return nullptr;
    case v_OP_GLU:
      if ((src0->type != v_TYPE_F32 && src0->type != v_TYPE_F16) ||
        (dst->type != v_TYPE_F32 && dst->type != v_TYPE_F16) ||
        (src0->type != dst->type)) { return nullptr; }

      switch (v_get_glu_op(dst)) {
        case v_GLU_OP_GEGLU:
          return ctx->device->pipeline_geglu[dst->type == v_TYPE_F16];
        case v_GLU_OP_REGLU:
          return ctx->device->pipeline_reglu[dst->type == v_TYPE_F16];
        case v_GLU_OP_SWIGLU:
          return ctx->device->pipeline_swiglu[dst->type == v_TYPE_F16];
        case v_GLU_OP_SWIGLU_OAI:
          return ctx->device->pipeline_swiglu_oai[dst->type == v_TYPE_F16];
        case v_GLU_OP_GEGLU_ERF:
          return ctx->device->pipeline_geglu_erf[dst->type == v_TYPE_F16];
        case v_GLU_OP_GEGLU_QUICK:
          return ctx->device->pipeline_geglu_quick[dst->type == v_TYPE_F16];
        default:
          break;
      }
      return nullptr;
    case V_OP_DIAG_MASK_INF:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_diag_mask_inf_f32; }
      return nullptr;
    case V_OP_SOFT_MAX:
      V_ASSERT(!src1 || src1->type == v_TYPE_F32 || src1->type == v_TYPE_F16);
      V_ASSERT(!src2 || src2->type == v_TYPE_F32);

      if (ctx->num_additional_fused_ops) {
        uint32_t idx = (uint32_t)ceilf(log2f(float(dst->ne[0])));
        V_ASSERT(idx < num_topk_moe_pipelines);
        bool with_norm = ctx->num_additional_fused_ops == topk_moe_norm.size() - 1;
        return ctx->device->pipeline_topk_moe[idx][with_norm];
      }

      if (src0->type == v_TYPE_F32 && (src1 == nullptr || src1->type == v_TYPE_F32) && dst->type == v_TYPE_F32) {
        return src0->ne[0] > 1024
                 ? ctx->device->pipeline_soft_max_f32_wg512
                 : ctx->device->pipeline_soft_max_f32;
      }
      if (src0->type == v_TYPE_F32 && src1->type == v_TYPE_F16 && dst->type == v_TYPE_F32) {
        return src0->ne[0] > 1024
                 ? ctx->device->pipeline_soft_max_f32_f16_wg512
                 : ctx->device->pipeline_soft_max_f32_f16;
      }
      return nullptr;
    case v_OP_SOFT_MAX_BACK:
      if (src0->type == v_TYPE_F32 && src1->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_soft_max_back_f32; }
      return nullptr;
    case V_OP_ROPE:
    case v_OP_ROPE_BACK: {
      const int mode       = ((const int32_t*)dst->op_params)[2];
      const bool is_neox   = mode & v_ROPE_TYPE_NEOX;
      const bool is_mrope  = mode & v_ROPE_TYPE_MROPE;
      const bool is_vision = mode == v_ROPE_TYPE_VISION;

      if (is_neox) {
        if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_rope_neox_f32; }
        if (src0->type == v_TYPE_F16 && dst->type == v_TYPE_F16) { return ctx->device->pipeline_rope_neox_f16; }
      }
      else if (is_mrope && !is_vision) {
        if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_rope_multi_f32; }
        if (src0->type == v_TYPE_F16 && dst->type == v_TYPE_F16) { return ctx->device->pipeline_rope_multi_f16; }
      }
      else if (is_vision) {
        if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_rope_vision_f32; }
        if (src0->type == v_TYPE_F16 && dst->type == v_TYPE_F16) { return ctx->device->pipeline_rope_vision_f16; }
      }
      else {
        if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_rope_norm_f32; }
        if (src0->type == v_TYPE_F16 && dst->type == v_TYPE_F16) { return ctx->device->pipeline_rope_norm_f16; }
      }
      return nullptr;
    }
    case v_OP_ARGSORT:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_I32) {
        uint32_t idx = (uint32_t)ceilf(log2f(float(dst->ne[0])));
        return ctx->device->pipeline_argsort_f32[idx];
      }
      return nullptr;
    case v_OP_SUM:
    case v_OP_SUM_ROWS:
    case V_OP_MEAN:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_sum_rows_f32; }
      return nullptr;
    case V_OP_ARGMAX:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_I32) { return ctx->device->pipeline_argmax_f32; }
      return nullptr;
    case v_OP_COUNT_EQUAL:
      if (src0->type == v_TYPE_I32 && src1->type == v_TYPE_I32 && dst->type == v_TYPE_I64) { return ctx->device->pipeline_count_equal_i32; }
      return nullptr;
    case V_OP_IM2COL:
      if (src1->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_im2col_f32; }
      if (src1->type == v_TYPE_F32 && dst->type == v_TYPE_F16) { return ctx->device->pipeline_im2col_f32_f16; }
      return nullptr;
    case v_OP_IM2COL_3D:
      if (src1->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_im2col_3d_f32; }
      if (src1->type == v_TYPE_F32 && dst->type == v_TYPE_F16) { return ctx->device->pipeline_im2col_3d_f32_f16; }
      return nullptr;
    case v_OP_TIMESTEP_EMBEDDING:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_timestep_embedding_f32; }
      return nullptr;
    case v_OP_CONV_TRANSPOSE_1D:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_conv_transpose_1d_f32; }
      return nullptr;
    case V_OP_POOL_2D:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_pool2d_f32; }
      return nullptr;
    case V_OP_POOL_2D_BACK:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_pool2d_back_f32; }
      return nullptr;

    case v_OP_RWKV_WKV6:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_rwkv_wkv6_f32; }
      return nullptr;
    case v_OP_RWKV_WKV7:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_rwkv_wkv7_f32; }
      return nullptr;
    case v_OP_SSM_SCAN:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) {
        const uint32_t d_state = src0->ne[0];
        if (d_state == 128) { return ctx->device->pipeline_ssm_scan_f32_d128; }
        else if (d_state == 256) { return ctx->device->pipeline_ssm_scan_f32_d256; }
      }
      return nullptr;
    case v_OP_SSM_CONV:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_ssm_conv_f32; }
      return nullptr;
    case v_OP_OPT_STEP_ADAMW:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_opt_step_adamw_f32; }
      return nullptr;
    case v_OP_OPT_STEP_SGD:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_opt_step_sgd_f32; }
      return nullptr;
    case v_OP_LEAKY_RELU:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) { return ctx->device->pipeline_leaky_relu_f32; }
      return nullptr;
    case v_OP_CONV_2D:
    case v_OP_CONV_TRANSPOSE_2D:
      if (src1->type == v_TYPE_F32 && dst->type == v_TYPE_F32 &&
        v_is_contiguous(src0) && v_is_contiguous(src1) && v_is_contiguous(dst)) {
        std::array<uint32_t, 3> elements;
        if (op == v_OP_CONV_2D) elements = v_vk_get_conv_elements(dst);
        else if (op == v_OP_CONV_TRANSPOSE_2D) elements = v_vk_get_conv_transpose_2d_elements(dst);
        vk_conv_shapes shape;

        uint32_t tiles[CONV_SHAPE_COUNT];
        for (uint32_t i = 0; i < CONV_SHAPE_COUNT; ++i) {
          tiles[i] = CEIL_DIV(elements[0], ctx->device->pipeline_conv2d_f32[i]->wg_denoms[0]) * CEIL_DIV(
            elements[1],
            ctx->device->pipeline_conv2d_f32[i]->wg_denoms[1]);
        }

        // We can't query number of shader cores on Intel, use 32 as a placeholder
        // so small convolutions will still choose a smaller tile.
        const uint32_t shader_core_count = ctx->device->shader_core_count > 0
                                             ? ctx->device->shader_core_count
                                             : 32;

        if (elements[0] > 64 && tiles[CONV_SHAPE_128x128] >= shader_core_count * 2) { shape = CONV_SHAPE_128x128; }
        else if (elements[0] <= 32 && tiles[CONV_SHAPE_32x256] >= shader_core_count * 2) { shape = CONV_SHAPE_32x256; }
        else { shape = CONV_SHAPE_64x32; }

        if (op == v_OP_CONV_2D) {
          if (src0->type == v_TYPE_F32) { return ctx->device->pipeline_conv2d_f32[shape]; }
          else if (src0->type == v_TYPE_F16) { return ctx->device->pipeline_conv2d_f16_f32[shape]; }
        }
        else if (op == v_OP_CONV_TRANSPOSE_2D) {
          if (src0->type == v_TYPE_F32) { return ctx->device->pipeline_conv_transpose_2d_f32[shape]; }
          else if (src0->type == v_TYPE_F16) { return ctx->device->pipeline_conv_transpose_2d_f16_f32[shape]; }
        }
      }
      return nullptr;
    case v_OP_CONV_2D_DW:
      if (src0->type == v_TYPE_F32 && dst->type == v_TYPE_F32) {
        if (v_is_contiguous(src1)) { return ctx->device->pipeline_conv2d_dw_whcn_f32; }
        else if (v_is_contiguous_channels(src1)) { return ctx->device->pipeline_conv2d_dw_cwhn_f32; }
      }
      else if (src0->type == v_TYPE_F16 && dst->type == v_TYPE_F32) {
        if (v_is_contiguous(src1)) { return ctx->device->pipeline_conv2d_dw_whcn_f16_f32; }
        else if (v_is_contiguous_channels(src1)) { return ctx->device->pipeline_conv2d_dw_cwhn_f16_f32; }
      }
      return nullptr;
    default:
      return nullptr;
  }

  v_UNUSED(src2);
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

uint32_t get_misalign_bytes(vk_backend_ctx* ctx, const v_tensor* t) {
  return ((vk_tensor_offset(t) + t->view_offs) & (ctx->device->properties.limits.minStorageBufferOffsetAlignment - 1));;
}

template <typename T>
void init_pushconst_tensor_offsets(vk_backend_ctx* ctx, T& p, const v_tensor* src0, const v_tensor* src1,
                                   const v_tensor* src2, v_tensor* dst) {
  v_UNUSED(p);
  v_UNUSED(src0);
  v_UNUSED(src1);
  v_UNUSED(src2);
  v_UNUSED(dst);
  static_assert(!std::is_const<T>::value, "unexpected type");
  V_ASSERT(!src0 || get_misalign_bytes(ctx, src0) == 0);
  V_ASSERT(!src1 || get_misalign_bytes(ctx, src1) == 0);
  V_ASSERT(!src2 || get_misalign_bytes(ctx, src2) == 0);
  V_ASSERT(!dst || get_misalign_bytes(ctx, dst) == 0);
}

template <>
void init_pushconst_tensor_offsets(vk_backend_ctx* ctx, vk_op_unary_push_constants& p, const v_tensor* src0,
                                   const v_tensor* src1, const v_tensor* src2, v_tensor* dst) {
  const uint32_t a_offset = get_misalign_bytes(ctx, src0) / v_type_size(src0->type);
  const uint32_t d_offset = get_misalign_bytes(ctx, dst) / v_type_size(dst->type);

  p.misalign_offsets = (a_offset << 16) | d_offset;

  v_UNUSED(src1);
  v_UNUSED(src2);
}

template <>
void init_pushconst_tensor_offsets(vk_backend_ctx* ctx, vk_op_sum_rows_push_constants& p,
                                   const v_tensor* src0, const v_tensor* src1, const v_tensor* src2,
                                   v_tensor* dst) {
  const uint32_t a_offset = get_misalign_bytes(ctx, src0) / v_type_size(src0->type);
  const uint32_t d_offset = get_misalign_bytes(ctx, dst) / v_type_size(dst->type);

  p.misalign_offsets = (a_offset << 16) | d_offset;

  v_UNUSED(src1);
  v_UNUSED(src2);
}

template <>
void init_pushconst_tensor_offsets(vk_backend_ctx* ctx, vk_op_pad_push_constants& p, const v_tensor* src0,
                                   const v_tensor* src1, const v_tensor* src2, v_tensor* dst) {
  const uint32_t a_offset = get_misalign_bytes(ctx, src0) / v_type_size(src0->type);
  const uint32_t d_offset = get_misalign_bytes(ctx, dst) / v_type_size(dst->type);

  p.misalign_offsets = (a_offset << 16) | d_offset;

  v_UNUSED(src1);
  v_UNUSED(src2);
}

template <>
void init_pushconst_tensor_offsets(vk_backend_ctx* ctx, vk_op_im2col_3d_push_constants& p,
                                   const v_tensor* src0, const v_tensor* src1, const v_tensor* src2,
                                   v_tensor* dst) {
  const uint32_t a_offset = get_misalign_bytes(ctx, src1) / v_type_size(src1->type);
  const uint32_t d_offset = get_misalign_bytes(ctx, dst) / v_type_size(dst->type);

  p.misalign_offsets = (a_offset << 16) | d_offset;

  v_UNUSED(src0);
  v_UNUSED(src2);
}

template <>
void init_pushconst_tensor_offsets(vk_backend_ctx* ctx, vk_op_binary_push_constants& p,
                                   const v_tensor* src0, const v_tensor* src1, const v_tensor* src2,
                                   v_tensor* dst) {
  const uint32_t a_offset = get_misalign_bytes(ctx, src0) / v_type_size(src0->type);
  const uint32_t b_offset = get_misalign_bytes(ctx, src1) / v_type_size(src1->type);
  const uint32_t d_offset = get_misalign_bytes(ctx, dst) / v_type_size(dst->type);

  V_ASSERT(dst->op != v_OP_GET_ROWS || (a_offset == 0 && b_offset == 0 && d_offset == 0));

  p.misalign_offsets = (a_offset << 16) | (b_offset << 8) | d_offset;

  v_UNUSED(src2);
}

template <>
void init_pushconst_tensor_offsets(vk_backend_ctx* ctx, vk_op_upscale_push_constants& p,
                                   const v_tensor* src0, const v_tensor* src1, const v_tensor* src2,
                                   v_tensor* dst) {
  const uint32_t a_offset = get_misalign_bytes(ctx, src0) / v_type_size(src0->type);
  const uint32_t d_offset = get_misalign_bytes(ctx, dst) / v_type_size(dst->type);

  p.a_offset = a_offset;
  p.d_offset = d_offset;

  v_UNUSED(src1);
  v_UNUSED(src2);
}

template <typename PC>
void v_vk_op_f32(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                 const v_tensor* src1, const v_tensor* src2, v_tensor* dst, v_operation op, PC&& pc,
                 bool dryrun = false) {
  VK_LOG_DEBUG(
    "v_vk_op_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] <<
    ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1="
    << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    if (src1 != nullptr) {
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] <<
    ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1="
    << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    }
    if (src2 != nullptr) {
    std::cerr << "), (" << src2 << ", name=" << src2->name << ", type=" << src2->type << ", ne0=" << src2->ne[0] <<
    ", ne1=" << src2->ne[1] << ", ne2=" << src2->ne[2] << ", ne3=" << src2->ne[3] << ", nb0=" << src2->nb[0] << ", nb1="
    << src2->nb[1] << ", nb2=" << src2->nb[2] << ", nb3=" << src2->nb[3];
    }
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1="
    << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1
    ] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << op_name(op) << ", " << (dryrun ? "dryrun" : "") << ")");
  V_ASSERT(op == v_OP_GET_ROWS || op == v_OP_CPY || (!v_is_quantized(src0->type) && (src1 == nullptr || !v_is_quantized(src1->type)))); // NOLINT
  V_ASSERT(v_vk_op_supports_incontiguous(op) || v_vk_dim01_contiguous(src0)); // NOLINT
  V_ASSERT(dst->buffer != nullptr);
  const uint64_t ne00 = src0->ne[0];
  const uint64_t ne01 = src0->ne[1];
  const uint64_t ne02 = src0->ne[2];
  const uint64_t ne03 = src0->ne[3];
  const uint64_t ne0  = ne00 * ne01;

  const bool use_src1 = src1 != nullptr;
  const uint64_t ne10 = use_src1
                          ? src1->ne[0]
                          : 0;
  const uint64_t ne11 = use_src1
                          ? src1->ne[1]
                          : 0;
  const uint64_t ne12 = use_src1
                          ? src1->ne[2]
                          : 0;
  const uint64_t ne13 = use_src1
                          ? src1->ne[3]
                          : 0;
  const uint64_t ne1 = ne10 * ne11;
  // const uint64_t nb10 = use_src1 ? src1->nb[0] : 0;

  const bool use_src2 = src2 != nullptr;
  const uint64_t ne20 = use_src2
                          ? src2->ne[0]
                          : 0;
  const uint64_t ne21 = use_src2
                          ? src2->ne[1]
                          : 0;
  const uint64_t ne22 = use_src2
                          ? src2->ne[2]
                          : 0;
  const uint64_t ne23 = use_src2
                          ? src2->ne[3]
                          : 0;
  const uint64_t ne2 = ne20 * ne21;

  const uint64_t ned0 = dst->ne[0];
  const uint64_t ned1 = dst->ne[1];
  const uint64_t ned2 = dst->ne[2];
  const uint64_t ned3 = dst->ne[3];
  const uint64_t ned  = ned0 * ned1;

  init_pushconst_fastdiv(pc);

  vk_pipeline pipeline = v_vk_op_get_pipeline(ctx, src0, src1, src2, dst, op);

  if (pipeline == nullptr) {
    std::cerr << "v_vulkan: Error: Missing op: " << v_op_name(op) << " for " << v_type_name(src0->type);
    if (src1 != nullptr) { std::cerr << " and " << v_type_name(src1->type); }
    std::cerr << " to " << v_type_name(dst->type) << std::endl;
    v_ABORT("fatal error");
  }

  if (dryrun) {
    v_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    return;
  }

  const bool op_supports_incontiguous = v_vk_op_supports_incontiguous(op);

  v_backend_vk_buffer_ctx* dst_buf_ctx  = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  v_backend_vk_buffer_ctx* src0_buf_ctx = (v_backend_vk_buffer_ctx*)src0->buffer->context;
  v_backend_vk_buffer_ctx* src1_buf_ctx = use_src1
                                            ? (v_backend_vk_buffer_ctx*)src1->buffer->context
                                            : nullptr;
  v_backend_vk_buffer_ctx* src2_buf_ctx = use_src2
                                            ? (v_backend_vk_buffer_ctx*)src2->buffer->context
                                            : nullptr;

  vk_buffer d_X       = nullptr;
  size_t x_buf_offset = 0;
  vk_buffer d_Y       = nullptr;
  size_t y_buf_offset = 0;
  vk_buffer d_Z       = nullptr;
  size_t z_buf_offset = 0;

  bool src0_uma = false;
  bool src1_uma = false;
  bool src2_uma = false;

  if (ctx->device->uma) {
    vk_get_host_buffer(ctx->device, src0->data, d_X, x_buf_offset);
    src0_uma = d_X != nullptr;
    if (use_src1) {
      vk_get_host_buffer(ctx->device, src1->data, d_Y, y_buf_offset);
      src1_uma = d_Y != nullptr;
    }
    if (use_src2) {
      vk_get_host_buffer(ctx->device, src2->data, d_Z, z_buf_offset);
      src2_uma = d_Z != nullptr;
    }
  }

  vk_buffer d_D = dst_buf_ctx->dev_buffer;

  V_ASSERT(d_D != nullptr);
  uint64_t d_buf_offset = vk_tensor_offset(dst) + dst->view_offs;
  if (!src0_uma) {
    d_X          = src0_buf_ctx->dev_buffer;
    x_buf_offset = vk_tensor_offset(src0) + src0->view_offs;
    V_ASSERT(d_X != nullptr);
  }
  if (use_src1 && !src1_uma) {
    d_Y          = src1_buf_ctx->dev_buffer;
    y_buf_offset = vk_tensor_offset(src1) + src1->view_offs;
    V_ASSERT(d_Y != nullptr);
  }
  if (use_src2 && !src2_uma) {
    d_Z          = src2_buf_ctx->dev_buffer;
    z_buf_offset = vk_tensor_offset(src2) + src2->view_offs;
    V_ASSERT(d_Z != nullptr);
  }
  // Compute misalignment offset for descriptors and store it in in push constants, then align the descriptor offsets.
  init_pushconst_tensor_offsets(ctx, pc, src0, src1, src2, dst);
  x_buf_offset &= ~(ctx->device->properties.limits.minStorageBufferOffsetAlignment - 1);
  y_buf_offset &= ~(ctx->device->properties.limits.minStorageBufferOffsetAlignment - 1);
  z_buf_offset &= ~(ctx->device->properties.limits.minStorageBufferOffsetAlignment - 1);
  d_buf_offset &= ~(ctx->device->properties.limits.minStorageBufferOffsetAlignment - 1);

  std::array<uint32_t, 3> elements;

  // Single call if dimension 2 is contiguous
  V_ASSERT(op_supports_incontiguous || (v_is_contiguous(src0) && (src1 == nullptr || v_is_contiguous(src1))));

  switch (op) {
    case v_OP_NORM:
    case v_OP_RMS_NORM_BACK:
    case v_OP_L2_NORM:
    case V_OP_SOFT_MAX:
    case v_OP_SOFT_MAX_BACK:
    case v_OP_SUM_ROWS:
    case V_OP_MEAN:
    case V_OP_ARGMAX: {
      const uint32_t nr = v_nrows(src0);
      if (nr > 262144) { elements = {512, 512, CEIL_DIV(nr, 262144)}; }
      else if (nr > 512) { elements = {512, CEIL_DIV(nr, 512), 1}; }
      else { elements = {nr, 1, 1}; }
    }
    break;
    case v_OP_RMS_NORM:
      if (ctx->do_add_rms_partials) {
        // Run one element per thread, 128 threads per workgroup
        elements = {(uint32_t)CEIL_DIV(ne00, 128), 1, 1};
      }
      else { elements = {(uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne03}; }
      break;

    case v_OP_SUM:
      // We use v_OP_SUM_ROWS with 1 row.
      elements = {1, 1, 1};
      break;
    case v_OP_GROUP_NORM: {
      const uint32_t num_groups = dst->op_params[0];
      elements                  = {num_groups * (uint32_t)src0->ne[3], 1, 1};
    }
    break;
    case V_OP_DIAG_MASK_INF:
    case V_OP_ROPE:
    case v_OP_ROPE_BACK:
      elements = {(uint32_t)v_nrows(src0), (uint32_t)ne00, 1};
      break;
    case v_OP_GET_ROWS:
      elements = {(uint32_t)ne00, (uint32_t)ne10, (uint32_t)(ne11 * ne12)};
      elements[1] = std::min(elements[1], ctx->device->properties.limits.maxComputeWorkGroupCount[1]);
      elements[2] = std::min(elements[2], ctx->device->properties.limits.maxComputeWorkGroupCount[2]);
      break;
    case v_OP_ARGSORT:
      elements = {(uint32_t)ne00, (uint32_t)v_nrows(src0), 1};
      break;
    case V_OP_IM2COL: {
      const bool is_2D = dst->op_params[6] == 1;

      const uint32_t IC = src1->ne[is_2D
                                     ? 2
                                     : 1];

      const uint32_t KH = is_2D
                            ? src0->ne[1]
                            : 1;
      const uint32_t KW = src0->ne[0];

      const uint32_t OH = is_2D
                            ? dst->ne[2]
                            : 1;
      const uint32_t OW = dst->ne[1];

      const uint32_t batch = src1->ne[is_2D
                                        ? 3
                                        : 2];

      elements = {OW * KW * KH, OH, batch * IC};
    }
    break;
    case v_OP_IM2COL_3D: {
      const uint32_t IC = ((const uint32_t*)(dst->op_params))[9];

      const uint32_t N = ne13 / IC;

      const uint32_t KD = ne02;
      const uint32_t KH = ne01;
      const uint32_t KW = ne00;

      const uint32_t OD = ned3 / N;
      const uint32_t OH = ned2;
      const uint32_t OW = ned1;

      const uint32_t IC_KD_KH_KW = IC * KD * KH * KW;
      const uint32_t N_OD_OH     = N * OD * OH;

      elements    = {IC_KD_KH_KW, OW, N_OD_OH};
      elements[2] = std::min(elements[2], ctx->device->properties.limits.maxComputeWorkGroupCount[2]);
    }
    break;
    case v_OP_TIMESTEP_EMBEDDING: {
      const uint32_t dim = dst->op_params[0];
      uint32_t half_ceil = (dim + 1) / 2;
      elements           = {half_ceil, (uint32_t)src0->ne[0], 1};
    }
    break;
    case v_OP_CONV_TRANSPOSE_1D: {
      elements = {uint32_t(src0->ne[1]), 1, 1}; // parallelize in {Cout, 1, 1}
    }
    break;
    case V_OP_POOL_2D: {
      const uint32_t N  = dst->ne[3];
      const uint32_t OC = dst->ne[2];
      const uint32_t OH = dst->ne[1];
      const uint32_t OW = dst->ne[0];
      elements          = {N * OC * OH * OW, 1, 1};
    }
    case V_OP_POOL_2D_BACK: {
      const uint32_t N  = dst->ne[3];
      const uint32_t OC = dst->ne[2];
      const uint32_t OH = dst->ne[1];
      const uint32_t OW = dst->ne[0];
      elements          = {N * OC * OH * OW, 1, 1};
    }
    break;
    case v_OP_CONV_2D: { elements = v_vk_get_conv_elements(dst); }
    break;
    case v_OP_CONV_TRANSPOSE_2D: { elements = v_vk_get_conv_transpose_2d_elements(dst); }
    break;
    case v_OP_ADD:
    case v_OP_SUB:
    case v_OP_DIV:
    case v_OP_MUL:
    case V_OP_SCALE:
    case v_OP_SQR:
    case v_OP_SQRT:
    case v_OP_SIN:
    case v_OP_COS:
    case v_OP_CLAMP:
    case v_OP_PAD:
    case v_OP_ROLL:
    case v_OP_REPEAT:
    case v_OP_REPEAT_BACK:
    case v_OP_CPY:
    case v_OP_CONCAT:
    case v_OP_UPSCALE:
    case v_OP_UNARY:
    case v_OP_GLU:
    case v_OP_CONV_2D_DW: {
      uint32_t ne = nelements(dst);
      if (op == v_OP_CPY && v_is_quantized(src0->type) && v_is_quantized(dst->type)) {
        // Convert from number of logical elements to 2- or 4-byte units.
        ne /= blockSize(src0->type);
        if ((v_type_size(src0->type) % 4) == 0) { ne *= v_type_size(src0->type) / 4; }
        else { ne *= v_type_size(src0->type) / 2; }
      }
      // copy_to_quant has block size of 32, and each thread does QUANT_K elements.
      // Splitting into 512x512xZ wouldn't work well since each workgroup does 1024 elements.
      // So divide by block size here before splitting into 512x512 groups.
      if (op == v_OP_CPY && !v_is_quantized(src0->type) && v_is_quantized(dst->type)) { ne = CEIL_DIV(ne, blockSize(dst->type)); }
      if (ne > 262144) { elements = {512, 512, CEIL_DIV(ne, 262144)}; }
      else if (ne > 512) { elements = {512, CEIL_DIV(ne, 512), 1}; }
      else { elements = {ne, 1, 1}; }
    }
    break;
    case v_OP_ADD_ID: { elements = {(uint32_t)ne01, (uint32_t)ne02, 1}; }
    break;
    case v_OP_SET_ROWS: {
      uint32_t ne = nelements(src0);
      if (v_is_quantized(dst->type)) {
        // quants run 32 threads each doing QUANT_K elements
        ne = CEIL_DIV(ne, 32 * blockSize(dst->type));
      }
      else {
        // scalar types do one element per thread, running 512 threads
        ne = CEIL_DIV(ne, 512);
      }
      if (ne > 262144) { elements = {512, 512, CEIL_DIV(ne, 262144)}; }
      else if (ne > 512) { elements = {512, CEIL_DIV(ne, 512), 1}; }
      else { elements = {ne, 1, 1}; }
    }
    break;
    case v_OP_SSM_CONV: {
      const uint32_t nr  = src0->ne[1];
      const uint32_t n_t = dst->ne[1];
      const uint32_t n_s = dst->ne[2];
      elements           = {nr, n_t, n_s};
    }
    break;
    default:
      elements = {(uint32_t)nelements(src0), 1, 1};
      break;
  }

  uint64_t x_sz, y_sz, z_sz, d_sz;

  if (op_supports_incontiguous) {
    x_sz = num_bytes(src0) + get_misalign_bytes(ctx, src0);
    y_sz = use_src1
             ? num_bytes(src1) + get_misalign_bytes(ctx, src1)
             : 0;
    z_sz = use_src2
             ? num_bytes(src2) + get_misalign_bytes(ctx, src2)
             : 0;
    d_sz = num_bytes(dst) + get_misalign_bytes(ctx, dst);

    if (x_buf_offset + x_sz >= d_X->size) { x_sz = v_vk_get_max_buffer_range(ctx, d_X, x_buf_offset); }
    if (use_src1 && y_buf_offset + y_sz >= d_Y->size) { y_sz = v_vk_get_max_buffer_range(ctx, d_Y, y_buf_offset); }
    if (use_src2 && z_buf_offset + z_sz >= d_Z->size) { z_sz = v_vk_get_max_buffer_range(ctx, d_Z, z_buf_offset); }
    if (d_buf_offset + d_sz >= d_D->size) { d_sz = v_vk_get_max_buffer_range(ctx, d_D, d_buf_offset); }
  }
  else {
    x_sz = v_type_size(src0->type) / blockSize(src0->type) * ne0 * ne02 * ne03;
    y_sz = use_src1
             ? v_type_size(src1->type) * ne1 * ne12 * ne13
             : 0;
    z_sz = use_src2
             ? v_type_size(src2->type) * ne2 * ne22 * ne23
             : 0;
    d_sz = v_type_size(dst->type) * ned * ned2 * ned3;
  }

  if (op == v_OP_ADD || op == v_OP_RMS_NORM) {
    vk_buffer d_A = ctx->do_add_rms_partials
                      ? ctx->prealloc_add_rms_partials
                      : d_X;
    size_t a_buf_offset = ctx->do_add_rms_partials
                            ? ctx->prealloc_size_add_rms_partials_offset
                            : 0;
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             vk_sub_buffer{d_X, x_buf_offset, x_sz},
                             vk_sub_buffer{d_Y, y_buf_offset, y_sz},
                             vk_sub_buffer{d_D, d_buf_offset, d_sz},
                             v_vk_subbuffer(ctx, d_A, a_buf_offset),
                           },
                           pc,
                           elements);
  }
  else if (op == v_OP_GLU) {
    // Empty src1 is possible in glu, but the shader needs a buffer
    vk_sub_buffer subbuf_y;
    if (use_src1) { subbuf_y = {d_Y, y_buf_offset, y_sz}; }
    else { subbuf_y = {d_X, 0, x_sz}; }

    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             vk_sub_buffer{d_X, x_buf_offset, x_sz}, subbuf_y, vk_sub_buffer{d_D, d_buf_offset, d_sz}
                           },
                           pc,
                           elements);
  }
  else if (op == V_OP_SOFT_MAX) {
    // Empty src1 and src2 is possible in soft_max, but the shader needs a buffer
    vk_sub_buffer subbuf_y;
    if (use_src1) { subbuf_y = {d_Y, y_buf_offset, y_sz}; }
    else { subbuf_y = {d_X, 0, x_sz}; }

    vk_sub_buffer subbuf_z;
    if (use_src2) { subbuf_z = {d_Z, z_buf_offset, z_sz}; }
    else { subbuf_z = {d_X, 0, x_sz}; }

    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             vk_sub_buffer{d_X, x_buf_offset, x_sz}, subbuf_y, subbuf_z,
                             vk_sub_buffer{d_D, d_buf_offset, d_sz}
                           },
                           pc,
                           elements);
  }
  else if (op == V_OP_ROPE || op == v_OP_ROPE_BACK) {
    // Empty src2 is possible in rope, but the shader needs a buffer
    vk_sub_buffer subbuf_z;
    if (use_src2) { subbuf_z = {d_Z, z_buf_offset, z_sz}; }
    else { subbuf_z = {d_X, 0, x_sz}; }

    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             vk_sub_buffer{d_X, x_buf_offset, x_sz}, vk_sub_buffer{d_Y, y_buf_offset, y_sz}, subbuf_z,
                             vk_sub_buffer{d_D, d_buf_offset, d_sz}
                           },
                           pc,
                           elements);
  }
  else if (op == V_OP_IM2COL || op == v_OP_IM2COL_3D) {
    if (ctx->device->shader_int64 && ctx->device->buffer_device_address) {
      // buffer device address path doesn't use dst buffer
      d_sz = 1;
    }
    // im2col uses only src1 and dst buffers
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {vk_sub_buffer{d_Y, y_buf_offset, y_sz}, vk_sub_buffer{d_D, d_buf_offset, d_sz}},
                           pc,
                           elements);
  }
  else if (op == v_OP_COUNT_EQUAL) {
    // count_equal assumes that destination buffer is initialized with zeroes
    vk_buffer_memset_async(subctx, d_D, d_buf_offset, 0, d_sz);
    vk_sync_buffers(ctx, subctx);
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             vk_sub_buffer{d_X, x_buf_offset, x_sz}, vk_sub_buffer{d_Y, y_buf_offset, y_sz},
                             vk_sub_buffer{d_D, d_buf_offset, d_sz}
                           },
                           pc,
                           elements);
  }
  else if (op == v_OP_OPT_STEP_SGD) {
    // OPT_STEP_SGD works on src0, it does not need dst
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             vk_sub_buffer{d_X, x_buf_offset, x_sz}, vk_sub_buffer{d_Y, y_buf_offset, y_sz},
                             vk_sub_buffer{d_Z, z_buf_offset, z_sz}
                           },
                           pc,
                           elements);
  }
  else if (use_src2) {
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             vk_sub_buffer{d_X, x_buf_offset, x_sz}, vk_sub_buffer{d_Y, y_buf_offset, y_sz},
                             vk_sub_buffer{d_Z, z_buf_offset, z_sz}, vk_sub_buffer{d_D, d_buf_offset, d_sz}
                           },
                           pc,
                           elements);
  }
  else if (use_src1) {
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             vk_sub_buffer{d_X, x_buf_offset, x_sz}, vk_sub_buffer{d_Y, y_buf_offset, y_sz},
                             vk_sub_buffer{d_D, d_buf_offset, d_sz}
                           },
                           pc,
                           elements);
  }
  else {
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {vk_sub_buffer{d_X, x_buf_offset, x_sz}, vk_sub_buffer{d_D, d_buf_offset, d_sz}},
                           pc,
                           elements);
  }
}

void v_vk_get_rows(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                   const v_tensor* src1, v_tensor* dst, bool dryrun = false) {
  const uint32_t src0_type_size = v_type_size(src0->type);
  const uint32_t src1_type_size = v_type_size(src1->type);
  const uint32_t dst_type_size  = v_type_size(dst->type);

  v_vk_op_f32<vk_op_binary_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           src1,
                                           nullptr,
                                           dst,
                                           v_OP_GET_ROWS,
                                           {
                                             (uint32_t)nelements(src0),
                                             (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
                                             (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size,
                                             (uint32_t)src0->nb[1] / src0_type_size,
                                             (uint32_t)src0->nb[2] / src0_type_size,
                                             (uint32_t)src0->nb[3] / src0_type_size,
                                             (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],
                                             (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size,
                                             (uint32_t)src1->nb[1] / src1_type_size,
                                             (uint32_t)src1->nb[2] / src1_type_size,
                                             (uint32_t)src1->nb[3] / src1_type_size,
                                             (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], (uint32_t)dst->ne[2],
                                             (uint32_t)dst->ne[3], (uint32_t)dst->nb[0] / dst_type_size,
                                             (uint32_t)dst->nb[1] / dst_type_size,
                                             (uint32_t)dst->nb[2] / dst_type_size,
                                             (uint32_t)dst->nb[3] / dst_type_size,
                                             0,
                                             0.0f, 0.0f, 0,
                                           },
                                           dryrun);
}

void v_vk_acc(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
              const v_tensor* src1, v_tensor* dst, bool dryrun = false) {
  const uint32_t src0_type_size = v_type_size(src0->type);
  const uint32_t src1_type_size = v_type_size(src1->type);
  const uint32_t dst_type_size  = v_type_size(dst->type);

  int nb1 = dst->op_params[0] / 4; // 4 bytes of float32
  int nb2 = dst->op_params[1] / 4; // 4 bytes of float32
  // int nb3 = dst->op_params[2] / 4; // 4 bytes of float32 - unused
  int offset = dst->op_params[3] / 4; // offset in bytes

  v_vk_op_f32<vk_op_binary_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           src1,
                                           nullptr,
                                           dst,
                                           v_OP_ACC,
                                           {
                                             (uint32_t)nelements(src0),
                                             (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
                                             (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size,
                                             (uint32_t)nb1, (uint32_t)nb2, (uint32_t)src0->nb[3] / src0_type_size,
                                             (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],
                                             (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size,
                                             (uint32_t)src1->nb[1] / src1_type_size,
                                             (uint32_t)src1->nb[2] / src1_type_size,
                                             (uint32_t)src1->nb[3] / src1_type_size,
                                             (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], (uint32_t)dst->ne[2],
                                             (uint32_t)dst->ne[3], (uint32_t)dst->nb[0] / dst_type_size,
                                             (uint32_t)nb1, (uint32_t)nb2, (uint32_t)dst->nb[3] / dst_type_size,
                                             0,
                                             0.0f, 0.0f, offset,
                                           },
                                           dryrun);
}

void v_vk_multi_add(vk_backend_ctx* ctx, vk_context& subctx, v_cgraph* cgraph, int node_idx,
                    bool dryrun = false) {
  const v_tensor* first_node = cgraph->nodes[node_idx];
  const v_tensor* dst        = cgraph->nodes[node_idx + ctx->num_additional_fused_ops];

  // Make a list of all the tensors used by the op.
  // Last element of the list is the dest tensor.
  const v_tensor* tensors[MAX_PARAMETER_COUNT];
  uint32_t num_srcs    = ctx->num_additional_fused_ops + 2;
  uint32_t num_tensors = num_srcs + 1;
  V_ASSERT(num_tensors + ctx->do_add_rms_partials <= MAX_PARAMETER_COUNT);

  tensors[0] = first_node->src[0];
  tensors[1] = first_node->src[1];
  for (int32_t i = 0; i < ctx->num_additional_fused_ops; ++i) {
    // check whether the previous result is src[0] or src[1]
    if (cgraph->nodes[node_idx + i] == cgraph->nodes[node_idx + i + 1]->src[0]) { tensors[i + 2] = cgraph->nodes[node_idx + i + 1]->src[1]; }
    else { tensors[i + 2] = cgraph->nodes[node_idx + i + 1]->src[0]; }
  }
  tensors[num_srcs] = dst;

  vk_op_multi_add_push_constants pc;
  pc.ne20 = (uint32_t)dst->ne[0];
  pc.ne21 = (uint32_t)dst->ne[1];
  pc.ne22 = (uint32_t)dst->ne[2];
  pc.ne23 = (uint32_t)dst->ne[3];

  for (uint32_t i = 0; i < num_tensors; ++i) {
    const v_tensor* t = tensors[i];
    pc.nb[i][0]       = (uint32_t)t->nb[0] / sizeof(float);
    pc.nb[i][1]       = (uint32_t)t->nb[1] / sizeof(float);
    pc.nb[i][2]       = (uint32_t)t->nb[2] / sizeof(float);
    pc.nb[i][3]       = (uint32_t)t->nb[3] / sizeof(float);
  }
  pc.rms_partials = ctx->do_add_rms_partials;

  vk_pipeline pipeline = v_vk_op_get_pipeline(ctx, tensors[0], tensors[1], nullptr, dst, dst->op);

  if (pipeline == nullptr) {
    std::cerr << "v_vulkan: Error: Missing multi_add";
    v_ABORT("fatal error");
  }

  if (dryrun) {
    v_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    return;
  }

  v_backend_vk_buffer_ctx* buf_ctx[MAX_PARAMETER_COUNT];
  vk_buffer buf[MAX_PARAMETER_COUNT];
  size_t offset[MAX_PARAMETER_COUNT];
  bool uma[MAX_PARAMETER_COUNT];

  for (uint32_t i = 0; i < num_tensors; ++i) {
    buf_ctx[i] = (v_backend_vk_buffer_ctx*)tensors[i]->buffer->context;
    buf[i]     = nullptr;
    offset[i]  = 0;
    uma[i]     = false;

    if (ctx->device->uma) {
      vk_get_host_buffer(ctx->device, tensors[i]->data, buf[i], offset[i]);
      uma[i] = buf[i] != nullptr;
    }
    if (!uma[i]) {
      buf[i]    = buf_ctx[i]->dev_buffer;
      offset[i] = vk_tensor_offset(tensors[i]) + tensors[i]->view_offs;
    }
    V_ASSERT(buf[i] != nullptr);
  }
  // If any remaining descriptors are unused, just point them at src[0]
  for (uint32_t i = num_tensors; i < MAX_PARAMETER_COUNT; ++i) {
    buf[i]    = buf[0];
    offset[i] = 0;
  }
  if (ctx->do_add_rms_partials) {
    buf[num_tensors]    = ctx->prealloc_add_rms_partials;
    offset[num_tensors] = ctx->prealloc_size_add_rms_partials_offset;
  }

  std::array<uint32_t, 3> elements;

  uint32_t ne = nelements(dst);
  if (ne > 262144) { elements = {512, 512, CEIL_DIV(ne, 262144)}; }
  else if (ne > 512) { elements = {512, CEIL_DIV(ne, 512), 1}; }
  else { elements = {ne, 1, 1}; }

  static_assert(MAX_PARAMETER_COUNT == 12);
  v_vk_dispatch_pipeline(ctx,
                         subctx,
                         pipeline,
                         {
                           v_vk_subbuffer(ctx, buf[0], offset[0]),
                           v_vk_subbuffer(ctx, buf[1], offset[1]),
                           v_vk_subbuffer(ctx, buf[2], offset[2]),
                           v_vk_subbuffer(ctx, buf[3], offset[3]),
                           v_vk_subbuffer(ctx, buf[4], offset[4]),
                           v_vk_subbuffer(ctx, buf[5], offset[5]),
                           v_vk_subbuffer(ctx, buf[6], offset[6]),
                           v_vk_subbuffer(ctx, buf[7], offset[7]),
                           v_vk_subbuffer(ctx, buf[8], offset[8]),
                           v_vk_subbuffer(ctx, buf[9], offset[9]),
                           v_vk_subbuffer(ctx, buf[10], offset[10]),
                           v_vk_subbuffer(ctx, buf[11], offset[11]),
                         },
                         pc,
                         elements);
}

void v_vk_add(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
              const v_tensor* src1, v_tensor* dst, bool dryrun = false) {
  const uint32_t src0_type_size = v_type_size(src0->type);
  const uint32_t src1_type_size = v_type_size(src1->type);
  const uint32_t dst_type_size  = v_type_size(dst->type);

  v_vk_op_f32<vk_op_binary_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           src1,
                                           nullptr,
                                           dst,
                                           v_OP_ADD,
                                           {
                                             (uint32_t)nelements(src0),
                                             (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
                                             (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size,
                                             (uint32_t)src0->nb[1] / src0_type_size,
                                             (uint32_t)src0->nb[2] / src0_type_size,
                                             (uint32_t)src0->nb[3] / src0_type_size,
                                             (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],
                                             (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size,
                                             (uint32_t)src1->nb[1] / src1_type_size,
                                             (uint32_t)src1->nb[2] / src1_type_size,
                                             (uint32_t)src1->nb[3] / src1_type_size,
                                             (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], (uint32_t)dst->ne[2],
                                             (uint32_t)dst->ne[3], (uint32_t)dst->nb[0] / dst_type_size,
                                             (uint32_t)dst->nb[1] / dst_type_size,
                                             (uint32_t)dst->nb[2] / dst_type_size,
                                             (uint32_t)dst->nb[3] / dst_type_size,
                                             0,
                                             0.0f, 0.0f, ctx->do_add_rms_partials,
                                           },
                                           dryrun);
}

void v_vk_sub(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
              const v_tensor* src1, v_tensor* dst, bool dryrun = false) {
  const uint32_t src0_type_size = v_type_size(src0->type);
  const uint32_t src1_type_size = v_type_size(src1->type);
  const uint32_t dst_type_size  = v_type_size(dst->type);

  v_vk_op_f32<vk_op_binary_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           src1,
                                           nullptr,
                                           dst,
                                           v_OP_SUB,
                                           {
                                             (uint32_t)nelements(src0),
                                             (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
                                             (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size,
                                             (uint32_t)src0->nb[1] / src0_type_size,
                                             (uint32_t)src0->nb[2] / src0_type_size,
                                             (uint32_t)src0->nb[3] / src0_type_size,
                                             (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],
                                             (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size,
                                             (uint32_t)src1->nb[1] / src1_type_size,
                                             (uint32_t)src1->nb[2] / src1_type_size,
                                             (uint32_t)src1->nb[3] / src1_type_size,
                                             (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], (uint32_t)dst->ne[2],
                                             (uint32_t)dst->ne[3], (uint32_t)dst->nb[0] / dst_type_size,
                                             (uint32_t)dst->nb[1] / dst_type_size,
                                             (uint32_t)dst->nb[2] / dst_type_size,
                                             (uint32_t)dst->nb[3] / dst_type_size,
                                             0,
                                             0.0f, 0.0f, 0,
                                           },
                                           dryrun);
}

void v_vk_mul(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
              const v_tensor* src1, v_tensor* dst, bool dryrun = false) {
  const uint32_t src0_type_size = v_type_size(src0->type);
  const uint32_t src1_type_size = v_type_size(src1->type);
  const uint32_t dst_type_size  = v_type_size(dst->type);

  v_vk_op_f32<vk_op_binary_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           src1,
                                           nullptr,
                                           dst,
                                           v_OP_MUL,
                                           {
                                             (uint32_t)nelements(src0),
                                             (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
                                             (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size,
                                             (uint32_t)src0->nb[1] / src0_type_size,
                                             (uint32_t)src0->nb[2] / src0_type_size,
                                             (uint32_t)src0->nb[3] / src0_type_size,
                                             (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],
                                             (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size,
                                             (uint32_t)src1->nb[1] / src1_type_size,
                                             (uint32_t)src1->nb[2] / src1_type_size,
                                             (uint32_t)src1->nb[3] / src1_type_size,
                                             (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], (uint32_t)dst->ne[2],
                                             (uint32_t)dst->ne[3], (uint32_t)dst->nb[0] / dst_type_size,
                                             (uint32_t)dst->nb[1] / dst_type_size,
                                             (uint32_t)dst->nb[2] / dst_type_size,
                                             (uint32_t)dst->nb[3] / dst_type_size,
                                             0,
                                             0.0f, 0.0f, 0,
                                           },
                                           dryrun);
}

void v_vk_div(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
              const v_tensor* src1, v_tensor* dst, bool dryrun = false) {
  const uint32_t src0_type_size = v_type_size(src0->type);
  const uint32_t src1_type_size = v_type_size(src1->type);
  const uint32_t dst_type_size  = v_type_size(dst->type);

  v_vk_op_f32<vk_op_binary_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           src1,
                                           nullptr,
                                           dst,
                                           v_OP_DIV,
                                           {
                                             (uint32_t)nelements(src0),
                                             (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
                                             (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size,
                                             (uint32_t)src0->nb[1] / src0_type_size,
                                             (uint32_t)src0->nb[2] / src0_type_size,
                                             (uint32_t)src0->nb[3] / src0_type_size,
                                             (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],
                                             (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size,
                                             (uint32_t)src1->nb[1] / src1_type_size,
                                             (uint32_t)src1->nb[2] / src1_type_size,
                                             (uint32_t)src1->nb[3] / src1_type_size,
                                             (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], (uint32_t)dst->ne[2],
                                             (uint32_t)dst->ne[3], (uint32_t)dst->nb[0] / dst_type_size,
                                             (uint32_t)dst->nb[1] / dst_type_size,
                                             (uint32_t)dst->nb[2] / dst_type_size,
                                             (uint32_t)dst->nb[3] / dst_type_size,
                                             0,
                                             0.0f, 0.0f, 0,
                                           },
                                           dryrun);
}

void v_vk_add_id(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                 const v_tensor* src1, const v_tensor* src2, v_tensor* dst, bool dryrun = false) {
  const uint32_t src0_type_size = v_type_size(src0->type);
  const uint32_t src1_type_size = v_type_size(src1->type);
  const uint32_t src2_type_size = v_type_size(src2->type);

  v_vk_op_f32<vk_op_add_id_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           src1,
                                           src2,
                                           dst,
                                           v_OP_ADD_ID,
                                           {
                                             (uint32_t)dst->ne[0],
                                             (uint32_t)dst->ne[1],
                                             (uint32_t)src0->nb[1] / src0_type_size,
                                             (uint32_t)src0->nb[2] / src0_type_size,
                                             (uint32_t)src1->nb[1] / src1_type_size,
                                             (uint32_t)src2->nb[1] / src2_type_size,
                                           },
                                           dryrun);
}

void v_vk_op_f32_wkv(vk_backend_ctx* ctx, vk_context& subctx, v_tensor* dst,
                     const vk_op_rwkv_wkv6_push_constants&& pc, int version, bool dryrun = false) {
  V_ASSERT(version == 6 || version == 7);
  int num_srcs = version == 6
                   ? 6
                   : 7;

  for (int i = 0; i < num_srcs; i++) { V_ASSERT(!v_is_quantized(dst->src[i]->type)); }

  V_ASSERT(dst->buffer != nullptr);

  vk_pipeline pipeline = v_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
  V_ASSERT(pipeline != nullptr);

  if (dryrun) {
    v_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    return;
  }

  v_backend_vk_buffer_ctx* dst_buf_ctx     = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  v_backend_vk_buffer_ctx* src_buf_ctxs[7] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  for (int i = 0; i < num_srcs; i++) { src_buf_ctxs[i] = (v_backend_vk_buffer_ctx*)dst->src[i]->buffer->context; }

  vk_buffer d_D     = nullptr, d_srcs[7] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  size_t dst_offset = 0, src_offsets[7]  = {0, 0, 0, 0, 0, 0, 0};
  bool dst_uma      = false, srcs_uma[7] = {false, false, false, false, false, false, false};

  if (ctx->device->uma) {
    for (int i = 0; i < num_srcs; i++) {
      vk_get_host_buffer(ctx->device, dst->src[i]->data, d_srcs[i], src_offsets[i]);
      srcs_uma[i] = d_srcs[i] != nullptr;
    }

    vk_get_host_buffer(ctx->device, dst->data, d_D, dst_offset);
    dst_uma = d_D != nullptr;
  }

  uint64_t src_sizes[7] = {0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < num_srcs; i++) {
    src_sizes[i] = num_bytes(dst->src[i]);
    if (!srcs_uma[i]) {
      d_srcs[i]      = src_buf_ctxs[i]->dev_buffer;
      src_offsets[i] = vk_tensor_offset(dst->src[i]) + dst->src[i]->view_offs;
    }
  }

  const uint64_t dst_size = num_bytes(dst);
  if (!dst_uma) {
    d_D        = dst_buf_ctx->dev_buffer;
    dst_offset = vk_tensor_offset(dst) + dst->view_offs;
  }

  std::array<uint32_t, 3> elements = {
    (uint32_t)(pc.B * pc.H),
    1,
    1
  };

  if (version == 6) {
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             vk_sub_buffer{d_srcs[0], src_offsets[0], src_sizes[0]},
                             vk_sub_buffer{d_srcs[1], src_offsets[1], src_sizes[1]},
                             vk_sub_buffer{d_srcs[2], src_offsets[2], src_sizes[2]},
                             vk_sub_buffer{d_srcs[3], src_offsets[3], src_sizes[3]},
                             vk_sub_buffer{d_srcs[4], src_offsets[4], src_sizes[4]},
                             vk_sub_buffer{d_srcs[5], src_offsets[5], src_sizes[5]},
                             vk_sub_buffer{d_D, dst_offset, dst_size}
                           },
                           pc,
                           elements);
  }
  else if (version == 7) {
    v_vk_dispatch_pipeline(ctx,
                           subctx,
                           pipeline,
                           {
                             vk_sub_buffer{d_srcs[0], src_offsets[0], src_sizes[0]},
                             vk_sub_buffer{d_srcs[1], src_offsets[1], src_sizes[1]},
                             vk_sub_buffer{d_srcs[2], src_offsets[2], src_sizes[2]},
                             vk_sub_buffer{d_srcs[3], src_offsets[3], src_sizes[3]},
                             vk_sub_buffer{d_srcs[4], src_offsets[4], src_sizes[4]},
                             vk_sub_buffer{d_srcs[5], src_offsets[5], src_sizes[5]},
                             vk_sub_buffer{d_srcs[6], src_offsets[6], src_sizes[6]},
                             vk_sub_buffer{d_D, dst_offset, dst_size}
                           },
                           pc,
                           elements);
  }
  else {
    // shouldn't happen
    V_ASSERT(false);
  }
}

void v_vk_rwkv_wkv6(vk_backend_ctx* ctx, vk_context& subctx, v_tensor* dst, bool dryrun = false) {
  const size_t seq_length = dst->src[0]->ne[2];
  const size_t n_embed    = dst->ne[0];
  const size_t n_heads    = dst->src[0]->ne[1];
  const size_t n_seqs     = dst->src[5]->ne[1];

  v_vk_op_f32_wkv(
    ctx,
    subctx,
    dst,
    {
      (uint32_t)n_seqs,
      (uint32_t)seq_length,
      (uint32_t)n_embed,
      (uint32_t)n_heads,
    },
    6,
    dryrun
  );
}

void v_vk_rwkv_wkv7(vk_backend_ctx* ctx, vk_context& subctx, v_tensor* dst, bool dryrun = false) {
  const size_t seq_length = dst->src[0]->ne[2];
  const size_t n_embed    = dst->ne[0];
  const size_t n_heads    = dst->src[0]->ne[1];
  const size_t n_seqs     = dst->src[6]->ne[1];

  v_vk_op_f32_wkv(
    ctx,
    subctx,
    dst,
    {
      (uint32_t)n_seqs,
      (uint32_t)seq_length,
      (uint32_t)n_embed,
      (uint32_t)n_heads,
    },
    7,
    dryrun
  );
}

void v_vk_ssm_scan(vk_backend_ctx* ctx, vk_context& subctx, v_tensor* dst, bool dryrun = false) {
  const v_tensor* src0 = dst->src[0];
  const v_tensor* src1 = dst->src[1];
  const v_tensor* src2 = dst->src[2];
  const v_tensor* src3 = dst->src[3];
  const v_tensor* src4 = dst->src[4];
  const v_tensor* src5 = dst->src[5];

  V_ASSERT(dst->buffer != nullptr);

  const uint32_t head_dim = src0->ne[1];
  const uint32_t n_head   = src1->ne[1];
  const uint32_t n_group  = src4->ne[1];
  const uint32_t n_tok    = src1->ne[2];
  const uint32_t n_seq    = src1->ne[3];

  bool is_mamba2 = (src3->nb[1] == sizeof(float));
  V_ASSERT(is_mamba2);

  vk_pipeline pipeline = v_vk_op_get_pipeline(ctx, src0, src1, src2, dst, dst->op);
  V_ASSERT(pipeline != nullptr);

  if (dryrun) {
    v_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    return;
  }

  const int64_t s_off = nelements(src1) * sizeof(float);

  const vk_op_ssm_scan_push_constants pc = {
    (uint32_t)src0->nb[2], (uint32_t)src0->nb[3],
    (uint32_t)src1->nb[2], (uint32_t)src1->nb[3],
    (uint32_t)src2->nb[1], (uint32_t)src2->nb[2],
    (uint32_t)src3->nb[1],
    (uint32_t)src4->nb[2], (uint32_t)src4->nb[3],
    (uint32_t)src5->nb[2], (uint32_t)src5->nb[3],
    (uint32_t)s_off,
    n_head, head_dim, n_group, n_tok
  };

  v_backend_vk_buffer_ctx* dst_buf_ctx = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  v_backend_vk_buffer_ctx* src_buf_ctxs[v_MAX_SRC];
  for (int i = 0; i < v_MAX_SRC && dst->src[i] != nullptr; i++) { src_buf_ctxs[i] = (v_backend_vk_buffer_ctx*)dst->src[i]->buffer->context; }

  vk_buffer d_D     = nullptr, d_srcs[v_MAX_SRC] = {nullptr};
  size_t dst_offset = 0, src_offsets[v_MAX_SRC]  = {0};
  bool dst_uma      = false, srcs_uma[v_MAX_SRC] = {false};

  if (ctx->device->uma) {
    for (int i = 0; i < v_MAX_SRC && dst->src[i] != nullptr; i++) {
      vk_get_host_buffer(ctx->device, dst->src[i]->data, d_srcs[i], src_offsets[i]);
      srcs_uma[i] = d_srcs[i] != nullptr;
    }
    vk_get_host_buffer(ctx->device, dst->data, d_D, dst_offset);
    dst_uma = d_D != nullptr;
  }

  if (!dst_uma) {
    d_D        = dst_buf_ctx->dev_buffer;
    dst_offset = vk_tensor_offset(dst) + dst->view_offs;
  }
  for (int i = 0; i < v_MAX_SRC && dst->src[i] != nullptr; i++) {
    if (!srcs_uma[i]) {
      d_srcs[i]      = src_buf_ctxs[i]->dev_buffer;
      src_offsets[i] = vk_tensor_offset(dst->src[i]) + dst->src[i]->view_offs;
    }
  }

  size_t dst_size = num_bytes(dst);
  size_t src_sizes[v_MAX_SRC];
  for (int i = 0; i < v_MAX_SRC && dst->src[i] != nullptr; i++) { src_sizes[i] = num_bytes(dst->src[i]); }

  std::array<uint32_t, 3> elements;

  const int splitH                = 16;
  const uint32_t num_workgroups_x = CEIL_DIV(n_head * head_dim, splitH);
  const uint32_t num_workgroups_y = n_seq;
  elements                        = {num_workgroups_x, num_workgroups_y, 1};

  v_vk_dispatch_pipeline(ctx,
                         subctx,
                         pipeline,
                         {
                           vk_sub_buffer{d_srcs[0], src_offsets[0], src_sizes[0]},
                           vk_sub_buffer{d_srcs[1], src_offsets[1], src_sizes[1]},
                           vk_sub_buffer{d_srcs[2], src_offsets[2], src_sizes[2]},
                           vk_sub_buffer{d_srcs[3], src_offsets[3], src_sizes[3]},
                           vk_sub_buffer{d_srcs[4], src_offsets[4], src_sizes[4]},
                           vk_sub_buffer{d_srcs[5], src_offsets[5], src_sizes[5]},
                           vk_sub_buffer{d_srcs[6], src_offsets[6], src_sizes[6]},
                           vk_sub_buffer{d_D, dst_offset, dst_size}
                         },
                         pc,
                         elements);
}

void v_vk_ssm_conv(vk_backend_ctx* ctx, vk_context& subctx, v_tensor* dst, bool dryrun = false) {
  const v_tensor* src0 = dst->src[0];
  const v_tensor* src1 = dst->src[1];

  v_vk_op_f32<vk_op_ssm_conv_push_constants>(ctx,
                                             subctx,
                                             src0,
                                             src1,
                                             nullptr,
                                             dst,
                                             v_OP_SSM_CONV,
                                             {
                                               (uint32_t)src0->nb[1], (uint32_t)src0->nb[2],
                                               (uint32_t)src1->nb[1],
                                               (uint32_t)dst->nb[0], (uint32_t)dst->nb[1], (uint32_t)dst->nb[2],
                                               (uint32_t)src1->ne[0],
                                               (uint32_t)src0->ne[0],
                                               (uint32_t)src0->ne[1],
                                               (uint32_t)dst->ne[1],
                                               (uint32_t)dst->ne[2],
                                             },
                                             dryrun);
}

void v_vk_op_f32_opt_step_adamw(vk_backend_ctx* ctx, vk_context& subctx, v_tensor* dst,
                                const vk_op_push_constants&& pc, bool dryrun = false) {
  const v_tensor* x  = dst->src[0];
  const v_tensor* g  = dst->src[1];
  const v_tensor* gm = dst->src[2];
  const v_tensor* gv = dst->src[3];
  const v_tensor* p  = dst->src[4];

  V_ASSERT(x->type == v_TYPE_F32);
  V_ASSERT(g->type == v_TYPE_F32);
  V_ASSERT(gm->type == v_TYPE_F32);
  V_ASSERT(gv->type == v_TYPE_F32);
  V_ASSERT(p->type == v_TYPE_F32);
  V_ASSERT(dst->buffer != nullptr);
  V_ASSERT(v_is_contiguous(x));
  V_ASSERT(v_is_contiguous(g));
  V_ASSERT(v_is_contiguous(gm));
  V_ASSERT(v_is_contiguous(gv));
  V_ASSERT(v_is_contiguous(p));
  V_ASSERT(v_are_same_shape(x, g));
  V_ASSERT(v_are_same_shape(x, gm));
  V_ASSERT(v_are_same_shape(x, gv));
  V_ASSERT(nelements(p) == 7);

  vk_pipeline pipeline = v_vk_op_get_pipeline(ctx, g, gm, gv, dst, v_OP_OPT_STEP_ADAMW);
  V_ASSERT(pipeline != nullptr);

  if (dryrun) {
    v_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    return;
  }

  v_backend_vk_buffer_ctx* x_buf_ctx  = (v_backend_vk_buffer_ctx*)x->buffer->context;
  v_backend_vk_buffer_ctx* g_buf_ctx  = (v_backend_vk_buffer_ctx*)g->buffer->context;
  v_backend_vk_buffer_ctx* gm_buf_ctx = (v_backend_vk_buffer_ctx*)gm->buffer->context;
  v_backend_vk_buffer_ctx* gv_buf_ctx = (v_backend_vk_buffer_ctx*)gv->buffer->context;
  v_backend_vk_buffer_ctx* p_buf_ctx  = (v_backend_vk_buffer_ctx*)p->buffer->context;

  vk_buffer d_X   = nullptr, d_G = nullptr, d_GM = nullptr, d_GV = nullptr, d_P = nullptr;
  size_t x_offset = 0, g_offset  = 0, gm_offset  = 0, gv_offset  = 0, p_offset  = 0;
  bool X_uma      = false, G_uma = false, GM_uma = false, GV_uma = false, P_uma = false;

  if (ctx->device->uma) {
    vk_get_host_buffer(ctx->device, x->data, d_X, x_offset);
    vk_get_host_buffer(ctx->device, g->data, d_G, g_offset);
    vk_get_host_buffer(ctx->device, gm->data, d_GM, gm_offset);
    vk_get_host_buffer(ctx->device, gv->data, d_GV, gv_offset);
    vk_get_host_buffer(ctx->device, p->data, d_P, p_offset);

    X_uma  = d_X != nullptr;
    G_uma  = d_G != nullptr;
    GM_uma = d_GM != nullptr;
    GV_uma = d_GV != nullptr;
    P_uma  = d_P != nullptr;
  }

  if (!X_uma) {
    d_X      = x_buf_ctx->dev_buffer;
    x_offset = vk_tensor_offset(x) + x->view_offs;
  }
  if (!G_uma) {
    d_G      = g_buf_ctx->dev_buffer;
    g_offset = vk_tensor_offset(g) + g->view_offs;
  }
  if (!GM_uma) {
    d_GM      = gm_buf_ctx->dev_buffer;
    gm_offset = vk_tensor_offset(gm) + gm->view_offs;
  }
  if (!GV_uma) {
    d_GV      = gv_buf_ctx->dev_buffer;
    gv_offset = vk_tensor_offset(gv) + gv->view_offs;
  }
  if (!P_uma) {
    d_P      = p_buf_ctx->dev_buffer;
    p_offset = vk_tensor_offset(p) + p->view_offs;
  }

  const uint64_t x_size  = num_bytes(x);
  const uint64_t g_size  = num_bytes(g);
  const uint64_t gm_size = num_bytes(gm);
  const uint64_t gv_size = num_bytes(gv);
  const uint64_t p_size  = num_bytes(p);

  std::array<uint32_t, 3> elements = {(uint32_t)nelements(x), 1, 1};

  v_vk_dispatch_pipeline(ctx,
                         subctx,
                         pipeline,
                         {
                           vk_sub_buffer{d_X, x_offset, x_size},
                           vk_sub_buffer{d_G, g_offset, g_size},
                           vk_sub_buffer{d_GM, gm_offset, gm_size},
                           vk_sub_buffer{d_GV, gv_offset, gv_size},
                           vk_sub_buffer{d_P, p_offset, p_size},
                         },
                         pc,
                         elements);
}

void v_vk_opt_step_adamw(vk_backend_ctx* ctx, vk_context& subctx, v_tensor* dst,
                         bool dryrun = false) {
  const size_t n = nelements(dst->src[0]);

  v_vk_op_f32_opt_step_adamw(
    ctx,
    subctx,
    dst,
    {(uint32_t)n, 0, 0.0f, 0.0f},
    dryrun
  );
}

void v_vk_opt_step_sgd(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                       const v_tensor* src1, const v_tensor* src2, v_tensor* dst,
                       bool dryrun = false) {
  const size_t n = nelements(dst->src[0]);

  v_vk_op_f32<vk_op_push_constants>(ctx,
                                    subctx,
                                    src0,
                                    src1,
                                    src2,
                                    dst,
                                    v_OP_OPT_STEP_SGD,
                                    {(uint32_t)n, 0, 0.0f, 0.0f},
                                    dryrun);
}

void v_vk_concat(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                 const v_tensor* src1, v_tensor* dst, bool dryrun = false) {
  int* op_params = (int*)dst->op_params;

  const uint32_t src0_type_size = v_type_size(src0->type);
  const uint32_t src1_type_size = v_type_size(src1->type);
  const uint32_t dst_type_size  = v_type_size(dst->type);

  v_vk_op_f32<vk_op_binary_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           src1,
                                           nullptr,
                                           dst,
                                           v_OP_CONCAT,
                                           {
                                             (uint32_t)nelements(dst),
                                             (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
                                             (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size,
                                             (uint32_t)src0->nb[1] / src0_type_size,
                                             (uint32_t)src0->nb[2] / src0_type_size,
                                             (uint32_t)src0->nb[3] / src0_type_size,
                                             (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],
                                             (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size,
                                             (uint32_t)src1->nb[1] / src1_type_size,
                                             (uint32_t)src1->nb[2] / src1_type_size,
                                             (uint32_t)src1->nb[3] / src1_type_size,
                                             (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], (uint32_t)dst->ne[2],
                                             (uint32_t)dst->ne[3], (uint32_t)dst->nb[0] / dst_type_size,
                                             (uint32_t)dst->nb[1] / dst_type_size,
                                             (uint32_t)dst->nb[2] / dst_type_size,
                                             (uint32_t)dst->nb[3] / dst_type_size,
                                             0,
                                             0.0f, 0.0f, op_params[0],
                                           },
                                           dryrun);
}

void v_vk_upscale(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
                  bool dryrun = false) {
  const uint32_t src0_type_size = v_type_size(src0->type);
  const uint32_t mode           = (uint32_t)v_get_op_params_i32(dst, 0);

  v_TENSOR_UNARY_OP_LOCALS

  float sf0          = (float)ne0 / ne00;
  float sf1          = (float)ne1 / ne01;
  float sf2          = (float)ne2 / ne02;
  float sf3          = (float)ne3 / ne03;
  float pixel_offset = 0.5f;

  if (mode & v_SCALE_FLAG_ALIGN_CORNERS) {
    sf0 = ne0 > 1 && ne00 > 1
            ? (float)(ne0 - 1) / (ne00 - 1)
            : sf0;
    sf1 = ne1 > 1 && ne01 > 1
            ? (float)(ne1 - 1) / (ne01 - 1)
            : sf1;
    pixel_offset = 0.0f;
  }

  v_vk_op_f32<vk_op_upscale_push_constants>(ctx,
                                            subctx,
                                            src0,
                                            nullptr,
                                            nullptr,
                                            dst,
                                            v_OP_UPSCALE,
                                            {
                                              (uint32_t)nelements(dst), 0, 0,
                                              (uint32_t)ne00, (uint32_t)ne01,
                                              (uint32_t)nb00 / src0_type_size, (uint32_t)nb01 / src0_type_size,
                                              (uint32_t)nb02 / src0_type_size, (uint32_t)nb03 / src0_type_size,
                                              (uint32_t)ne0, (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
                                              sf0, sf1, sf2, sf3, pixel_offset
                                            },
                                            dryrun);
}

void v_vk_scale(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
                bool dryrun = false) {
  vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst);
  p.param1                     = v_get_op_params_f32(dst, 0);
  p.param2                     = v_get_op_params_f32(dst, 1);

  v_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, V_OP_SCALE, std::move(p), dryrun);
}

void v_vk_sqr(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
              bool dryrun = false) {
  v_vk_op_f32(ctx,
              subctx,
              src0,
              nullptr,
              nullptr,
              dst,
              v_OP_SQR,
              vk_op_unary_push_constants_init(src0, dst),
              dryrun);
}

void v_vk_sqrt(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
               bool dryrun = false) {
  v_vk_op_f32(ctx,
              subctx,
              src0,
              nullptr,
              nullptr,
              dst,
              v_OP_SQRT,
              vk_op_unary_push_constants_init(src0, dst),
              dryrun);
}

void v_vk_sin(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
              bool dryrun = false) {
  v_vk_op_f32(ctx,
              subctx,
              src0,
              nullptr,
              nullptr,
              dst,
              v_OP_SIN,
              vk_op_unary_push_constants_init(src0, dst),
              dryrun);
}

void v_vk_cos(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
              bool dryrun = false) {
  v_vk_op_f32(ctx,
              subctx,
              src0,
              nullptr,
              nullptr,
              dst,
              v_OP_COS,
              vk_op_unary_push_constants_init(src0, dst),
              dryrun);
}

void v_vk_clamp(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
                bool dryrun = false) {
  vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst);
  p.param1                     = v_get_op_params_f32(dst, 0);
  p.param2                     = v_get_op_params_f32(dst, 1);

  v_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, v_OP_CLAMP, std::move(p), dryrun);
}

void v_vk_pad(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
              bool dryrun = false) {
  vk_op_pad_push_constants p = vk_op_pad_push_constants_init(src0, dst);
  v_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, v_OP_PAD, std::move(p), dryrun);
}

void v_vk_roll(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
               bool dryrun = false) {
  const int32_t s0          = v_get_op_params_i32(dst, 0);
  const int32_t s1          = v_get_op_params_i32(dst, 1);
  const int32_t s2          = v_get_op_params_i32(dst, 2);
  const int32_t s3          = v_get_op_params_i32(dst, 3);
  const uint32_t s01_packed = ((s0 + 0x8000) << 16) | (s1 + 0x8000);
  const uint32_t s23_packed = ((s2 + 0x8000) << 16) | (s3 + 0x8000);

  vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst);
  memcpy(&p.param1, &s01_packed, sizeof(float));
  memcpy(&p.param2, &s23_packed, sizeof(float));

  v_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, v_OP_ROLL, std::move(p), dryrun);
}

void v_vk_repeat(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
                 bool dryrun = false) {
  vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst, nelements(dst));
  v_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, v_OP_REPEAT, std::move(p), dryrun);
}

void v_vk_repeat_back(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                      v_tensor* dst, bool dryrun = false) {
  vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst, nelements(dst));
  v_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, v_OP_REPEAT_BACK, std::move(p), dryrun);
}

void v_vk_cpy(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
              bool dryrun = false) {
  uint32_t ne = (uint32_t)nelements(src0);
  if (v_is_quantized(src0->type) && v_is_quantized(dst->type)) {
    // Convert from number of logical elements to 2- or 4-byte units.
    ne /= blockSize(src0->type);
    if ((v_type_size(src0->type) % 4) == 0) { ne *= v_type_size(src0->type) / 4; }
    else { ne *= v_type_size(src0->type) / 2; }
  }

  vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst, ne);
  v_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, v_OP_CPY, std::move(p), dryrun);
}

void v_vk_set_rows(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                   const v_tensor* src1, v_tensor* dst, bool dryrun = false) {
  const uint32_t src0_type_size = v_type_size(src0->type);
  const uint32_t src1_type_size = v_type_size(src1->type);
  const uint32_t dst_type_size  = v_type_size(dst->type);

  // Skip empty skip_rows operations. For most ops the empty check at the start
  // of v_vk_build_graph is sufficient, but set_rows can have a nonempty dst
  // with empty srcs.
  if (is_empty(src0) || is_empty(src1)) { return; }

  v_vk_op_f32<vk_op_binary_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           src1,
                                           nullptr,
                                           dst,
                                           v_OP_SET_ROWS,
                                           {
                                             (uint32_t)nelements(src0),
                                             (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
                                             (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size,
                                             (uint32_t)src0->nb[1] / src0_type_size,
                                             (uint32_t)src0->nb[2] / src0_type_size,
                                             (uint32_t)src0->nb[3] / src0_type_size,
                                             (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],
                                             (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size,
                                             (uint32_t)src1->nb[1] / src1_type_size,
                                             (uint32_t)src1->nb[2] / src1_type_size,
                                             (uint32_t)src1->nb[3] / src1_type_size,
                                             (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], (uint32_t)dst->ne[2],
                                             (uint32_t)dst->ne[3], (uint32_t)dst->nb[0] / dst_type_size,
                                             (uint32_t)dst->nb[1] / dst_type_size,
                                             (uint32_t)dst->nb[2] / dst_type_size,
                                             (uint32_t)dst->nb[3] / dst_type_size,
                                             0,
                                             0.0f, 0.0f, 0,
                                           },
                                           dryrun);
}

void v_vk_silu_back(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                    const v_tensor* src1, v_tensor* dst, bool dryrun = false) {
  v_vk_op_f32<vk_op_push_constants>(ctx,
                                    subctx,
                                    src0,
                                    src1,
                                    nullptr,
                                    dst,
                                    v_OP_SILU_BACK,
                                    {(uint32_t)nelements(src0), 0, 0.0f, 0.0f},
                                    dryrun);
}

void v_vk_norm(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
               bool dryrun = false) {
  float* op_params = (float*)dst->op_params;

  v_vk_op_f32<vk_op_push_constants>(ctx,
                                    subctx,
                                    src0,
                                    nullptr,
                                    nullptr,
                                    dst,
                                    v_OP_NORM,
                                    {(uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0], 0.0f},
                                    dryrun);
}

void v_vk_group_norm(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                     v_tensor* dst, bool dryrun = false) {
  const int* int_op_params     = (const int*)dst->op_params;
  const float* float_op_params = (const float*)dst->op_params;

  const uint32_t num_groups = int_op_params[0];
  const float eps           = float_op_params[1];
  const uint32_t group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);

  v_vk_op_f32<vk_op_push_constants>(ctx,
                                    subctx,
                                    src0,
                                    nullptr,
                                    nullptr,
                                    dst,
                                    v_OP_GROUP_NORM,
                                    {group_size, 0, eps, 0.0f},
                                    dryrun);
}

uint32_t v_vk_rms_num_partials(vk_backend_ctx* ctx, const v_tensor* node) {
  const uint32_t ne           = (uint32_t)node->ne[0];
  const uint32_t denom        = ctx->device->pipeline_add_rms[0][0][0]->wg_denoms[0];
  const uint32_t num_partials = CEIL_DIV(ne, denom);
  return num_partials;
}

uint32_t v_vk_rms_partials_size(vk_backend_ctx* ctx, const v_tensor* node) {
  const uint32_t num_partials = v_vk_rms_num_partials(ctx, node);
  const uint32_t num_bytes    = ROUNDUP_POW2(num_partials * sizeof(uint32_t), ctx->device->partials_binding_alignment);
  return num_bytes;
}

void v_vk_rms_norm(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                   const v_tensor* src1, v_tensor* dst, float* op_params, bool dryrun = false) {
  const uint32_t src0_type_size = v_type_size(src0->type);
  const uint32_t src1_type_size = v_type_size(src1->type);
  const uint32_t dst_type_size  = v_type_size(dst->type);

  uint32_t param3 = ctx->do_add_rms_partials
                      ? v_vk_rms_num_partials(ctx, dst)
                      : 0;

  v_vk_op_f32<vk_op_binary_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           src1,
                                           nullptr,
                                           dst,
                                           v_OP_RMS_NORM,
                                           {
                                             (uint32_t)nelements(src0),
                                             (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
                                             (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size,
                                             (uint32_t)src0->nb[1] / src0_type_size,
                                             (uint32_t)src0->nb[2] / src0_type_size,
                                             (uint32_t)src0->nb[3] / src0_type_size,
                                             (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],
                                             (uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size,
                                             (uint32_t)src1->nb[1] / src1_type_size,
                                             (uint32_t)src1->nb[2] / src1_type_size,
                                             (uint32_t)src1->nb[3] / src1_type_size,
                                             (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], (uint32_t)dst->ne[2],
                                             (uint32_t)dst->ne[3], (uint32_t)dst->nb[0] / dst_type_size,
                                             (uint32_t)dst->nb[1] / dst_type_size,
                                             (uint32_t)dst->nb[2] / dst_type_size,
                                             (uint32_t)dst->nb[3] / dst_type_size,
                                             0,
                                             op_params[0], 0.0f, (int32_t)param3,
                                           },
                                           dryrun);

  if (ctx->do_add_rms_partials) {
    ctx->prealloc_size_add_rms_partials_offset += v_vk_rms_partials_size(ctx, src0);
    ctx->do_add_rms_partials = false;
  }
}

void v_vk_rms_norm_back(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                        const v_tensor* src1, v_tensor* dst, bool dryrun = false) {
  float* op_params = (float*)dst->op_params;
  v_vk_op_f32<vk_op_push_constants>(ctx,
                                    subctx,
                                    src0,
                                    src1,
                                    nullptr,
                                    dst,
                                    v_OP_RMS_NORM_BACK,
                                    {(uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0], 0.0f},
                                    dryrun);
}

void v_vk_l2_norm(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
                  bool dryrun = false) {
  float* op_params = (float*)dst->op_params;
  v_vk_op_f32<vk_op_push_constants>(ctx,
                                    subctx,
                                    src0,
                                    nullptr,
                                    nullptr,
                                    dst,
                                    v_OP_L2_NORM,
                                    {(uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0], 0.0f},
                                    dryrun);
}

void v_vk_unary(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
                bool dryrun = false) {
  v_vk_op_f32<vk_op_push_constants>(ctx,
                                    subctx,
                                    src0,
                                    nullptr,
                                    nullptr,
                                    dst,
                                    v_OP_UNARY,
                                    {(uint32_t)nelements(src0), 0, 0.0f, 0.0f},
                                    dryrun);
}

void v_vk_glu(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
              const v_tensor* src1, v_tensor* dst, bool dryrun = false) {
  const float* op_params_f = (const float*)dst->op_params;

  const bool swapped = (bool)dst->op_params[1];
  const bool split   = src1 != nullptr;
  const float alpha  = op_params_f[2];
  const float limit  = op_params_f[3];

  V_ASSERT(v_is_contiguous(src0));

  if (!split) { V_ASSERT(src0->ne[0] / 2 == dst->ne[0]); }
  else {
    V_ASSERT(src0->ne[0] == src1->ne[0]);
    V_ASSERT(src0->ne[0] == dst->ne[0]);
    V_ASSERT(src0->type == src1->type);
  }

  const uint32_t mode = split
                          ? 2
                          : (swapped
                               ? 1
                               : 0);

  v_vk_op_f32<vk_op_glu_push_constants>(ctx,
                                        subctx,
                                        src0,
                                        src1,
                                        nullptr,
                                        dst,
                                        v_OP_GLU,
                                        {
                                          (uint32_t)nelements(dst),
                                          (uint32_t)src0->ne[0],
                                          (uint32_t)dst->ne[0],
                                          mode,
                                          alpha,
                                          limit
                                        },
                                        dryrun);
}

void v_vk_diag_mask_inf(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                        v_tensor* dst, bool dryrun = false) {
  int32_t* op_params = (int32_t*)dst->op_params;
  v_vk_op_f32<vk_op_diag_mask_push_constants>(ctx,
                                              subctx,
                                              src0,
                                              nullptr,
                                              nullptr,
                                              dst,
                                              V_OP_DIAG_MASK_INF,
                                              {(uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0]},
                                              dryrun);
}

void v_vk_soft_max(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                   const v_tensor* src1, const v_tensor* src2, v_tensor* dst, bool dryrun = false) {
  float* op_params = (float*)dst->op_params;

  float scale    = op_params[0];
  float max_bias = op_params[1];

  const uint32_t ncols   = (uint32_t)src0->ne[0];
  const uint32_t nrows_x = (uint32_t)v_nrows(src0);
  const uint32_t nrows_y = (uint32_t)src0->ne[1];

  const uint32_t ne12 = src1
                          ? (uint32_t)(src1->ne[2])
                          : 0u;
  const uint32_t ne13 = src1
                          ? (uint32_t)(src1->ne[3])
                          : 0u;
  const uint32_t nb11 = src1
                          ? (uint32_t)(src1->nb[1] / src1->nb[0])
                          : 0u;
  const uint32_t nb12 = src1
                          ? (uint32_t)(src1->nb[2] / src1->nb[0])
                          : 0u;
  const uint32_t nb13 = src1
                          ? (uint32_t)(src1->nb[3] / src1->nb[0])
                          : 0u;

  const uint32_t n_head_kv   = src0->ne[2];
  const uint32_t n_head_log2 = 1u << (uint32_t)floorf(log2f((float)n_head_kv));

  const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
  const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

  v_vk_op_f32<vk_op_soft_max_push_constants>(ctx,
                                             subctx,
                                             src0,
                                             src1,
                                             src2,
                                             dst,
                                             V_OP_SOFT_MAX,
                                             {
                                               ncols,
                                               src1 != nullptr
                                                 ? nrows_y
                                                 : (uint32_t)0,
                                               (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
                                               ne12, ne13,
                                               nb11, nb12, nb13,
                                               scale, max_bias,
                                               m0, m1,
                                               n_head_log2,
                                               nrows_x,
                                               src2 != nullptr
                                             },
                                             dryrun);
}

void v_vk_soft_max_back(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                        const v_tensor* src1, v_tensor* dst, bool dryrun = false) {
  float* op_params = (float*)dst->op_params;
  v_vk_op_f32<vk_op_push_constants>(ctx,
                                    subctx,
                                    src0,
                                    src1,
                                    nullptr,
                                    dst,
                                    v_OP_SOFT_MAX_BACK,
                                    {
                                      (uint32_t)src0->ne[0], (uint32_t)v_nrows(src0), op_params[0], op_params[1]
                                    },
                                    dryrun);
}

void v_vk_topk_moe(vk_backend_ctx* ctx, vk_context& subctx, v_cgraph* cgraph, int node_idx,
                   bool dryrun = false) {
  bool with_norm    = ctx->num_additional_fused_ops == topk_moe_norm.size() - 1;
  v_tensor* logits  = cgraph->nodes[node_idx + 0]->src[0];
  v_tensor* weights = with_norm
                        ? cgraph->nodes[node_idx + 8]
                        : cgraph->nodes[node_idx + 4];
  v_tensor* ids = cgraph->nodes[node_idx + 3];

  V_ASSERT(logits->type == v_TYPE_F32);
  V_ASSERT(weights->type == v_TYPE_F32);
  V_ASSERT(ids->type == v_TYPE_I32);

  const int n_experts     = logits->ne[0];
  const int n_rows        = logits->ne[1];
  const int n_expert_used = weights->ne[1];

  V_ASSERT(ids->nb[1] / v_type_size(ids->type) == (size_t) n_experts);

  vk_pipeline pipeline = v_vk_op_get_pipeline(ctx,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              cgraph->nodes[node_idx],
                                              V_OP_SOFT_MAX);

  if (dryrun) {
    v_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    return;
  }

  v_backend_vk_buffer_ctx* logits_buf_ctx  = (v_backend_vk_buffer_ctx*)logits->buffer->context;
  v_backend_vk_buffer_ctx* weights_buf_ctx = (v_backend_vk_buffer_ctx*)weights->buffer->context;
  v_backend_vk_buffer_ctx* ids_buf_ctx     = (v_backend_vk_buffer_ctx*)ids->buffer->context;

  vk_buffer d_logits        = nullptr;
  size_t logits_buf_offset  = 0;
  vk_buffer d_weights       = nullptr;
  size_t weights_buf_offset = 0;
  vk_buffer d_ids           = nullptr;
  size_t ids_buf_offset     = 0;

  bool logits_uma  = false;
  bool weights_uma = false;
  bool ids_uma     = false;

  if (ctx->device->uma) {
    vk_get_host_buffer(ctx->device, logits->data, d_logits, logits_buf_offset);
    vk_get_host_buffer(ctx->device, weights->data, d_weights, weights_buf_offset);
    vk_get_host_buffer(ctx->device, ids->data, d_ids, ids_buf_offset);
    logits_uma  = d_logits != nullptr;
    weights_uma = d_weights != nullptr;
    ids_uma     = d_ids != nullptr;
  }

  if (!logits_uma) {
    d_logits          = logits_buf_ctx->dev_buffer;
    logits_buf_offset = vk_tensor_offset(logits) + logits->view_offs;
    V_ASSERT(d_logits != nullptr);
  }
  if (!weights_uma) {
    d_weights          = weights_buf_ctx->dev_buffer;
    weights_buf_offset = vk_tensor_offset(weights) + weights->view_offs;
    V_ASSERT(d_weights != nullptr);
  }
  if (!ids_uma) {
    d_ids          = ids_buf_ctx->dev_buffer;
    ids_buf_offset = vk_tensor_offset(ids) + ids->view_offs;
    V_ASSERT(d_ids != nullptr);
  }

  vk_op_topk_moe_push_constants pc;
  pc.n_rows        = n_rows;
  pc.n_expert_used = n_expert_used;

  V_ASSERT(n_expert_used <= n_experts);

  const uint32_t rows_per_block    = 4;
  std::array<uint32_t, 3> elements = {CEIL_DIV(n_rows, rows_per_block), 1, 1};

  v_vk_dispatch_pipeline(ctx,
                         subctx,
                         pipeline,
                         {
                           v_vk_subbuffer(ctx, d_logits, logits_buf_offset),
                           v_vk_subbuffer(ctx, d_weights, weights_buf_offset),
                           v_vk_subbuffer(ctx, d_ids, ids_buf_offset),
                         },
                         pc,
                         elements);
}

void v_vk_rope(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
               const v_tensor* src1, const v_tensor* src2, v_tensor* dst, bool backprop,
               bool dryrun = false) {
  const int n_dims = ((int32_t*)dst->op_params)[1];
  const int mode   = ((int32_t*)dst->op_params)[2];
  // const int n_ctx         = ((int32_t *) dst->op_params)[3];
  const int n_ctx_orig    = ((int32_t*)dst->op_params)[4];
  const float freq_base   = ((float*)dst->op_params)[5];
  const float freq_scale  = ((float*)dst->op_params)[6];
  const float ext_factor  = ((float*)dst->op_params)[7];
  const float attn_factor = ((float*)dst->op_params)[8];
  const float beta_fast   = ((float*)dst->op_params)[9];
  const float beta_slow   = ((float*)dst->op_params)[10];
  int sections[4]{};
  if (mode & v_ROPE_TYPE_MROPE) { memcpy(sections, (int32_t*)dst->op_params + 11, sizeof(int) * 4); }

  float corr_dims[2];
  v_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

  const float theta_scale = powf(freq_base, -2.0f / n_dims);

  uint32_t s1 = src0->nb[1] / v_type_size(src0->type);
  uint32_t s2 = src0->nb[2] / v_type_size(src0->type);

  v_vk_op_f32<vk_op_rope_push_constants>(ctx,
                                         subctx,
                                         src0,
                                         src1,
                                         src2,
                                         dst,
                                         V_OP_ROPE,
                                         {
                                           (uint32_t)src0->ne[0], (uint32_t)n_dims, freq_scale,
                                           (uint32_t)src0->ne[1],
                                           freq_base, ext_factor, attn_factor, {corr_dims[0], corr_dims[1]},
                                           theta_scale,
                                           src2 != nullptr, (uint32_t)src0->ne[2], s1, s2,
                                           {sections[0], sections[1], sections[2], sections[3]}, backprop
                                         },
                                         dryrun);
}

void v_vk_argsort(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
                  bool dryrun = false) {
  int32_t* op_params = (int32_t*)dst->op_params;

  uint32_t ncols = src0->ne[0];

  v_vk_op_f32<vk_op_argsort_push_constants>(ctx,
                                            subctx,
                                            src0,
                                            nullptr,
                                            nullptr,
                                            dst,
                                            v_OP_ARGSORT,
                                            {
                                              ncols,
                                              op_params[0],
                                            },
                                            dryrun);
}

void v_vk_sum(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
              bool dryrun = false) {
  vk_op_sum_rows_push_constants p = vk_op_sum_rows_push_constants_init(src0, dst, nelements(src0));
  v_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, v_OP_SUM, p, dryrun);
}

void v_vk_sum_rows(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                   v_tensor* dst, bool dryrun = false) {
  vk_op_sum_rows_push_constants p = vk_op_sum_rows_push_constants_init(src0, dst, src0->ne[0]);
  v_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, v_OP_SUM_ROWS, p, dryrun);
}

void v_vk_mean(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
               bool dryrun = false) {
  vk_op_sum_rows_push_constants p = vk_op_sum_rows_push_constants_init(src0, dst, src0->ne[0]);
  p.weight                        = 1.0f / (float)src0->ne[0];
  v_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, dst, V_OP_MEAN, p, dryrun);
}

void v_vk_argmax(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
                 bool dryrun = false) {
  v_vk_op_f32<vk_op_push_constants>(ctx,
                                    subctx,
                                    src0,
                                    nullptr,
                                    nullptr,
                                    dst,
                                    V_OP_ARGMAX,
                                    {(uint32_t)src0->ne[0], (uint32_t)src0->ne[1], 0.0f, 0.0f},
                                    dryrun);
}

void v_vk_count_equal(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                      const v_tensor* src1, v_tensor* dst, bool dryrun = false) {
  v_vk_op_f32<vk_op_push_constants>(ctx,
                                    subctx,
                                    src0,
                                    src1,
                                    nullptr,
                                    dst,
                                    v_OP_COUNT_EQUAL,
                                    {(uint32_t)nelements(src0), 0, 0.0f, 0.0f},
                                    dryrun);
}

void v_vk_im2col(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                 const v_tensor* src1, v_tensor* dst, bool dryrun = false) {
  const int32_t s0 = dst->op_params[0];
  const int32_t s1 = dst->op_params[1];
  const int32_t p0 = dst->op_params[2];
  const int32_t p1 = dst->op_params[3];
  const int32_t d0 = dst->op_params[4];
  const int32_t d1 = dst->op_params[5];

  const bool is_2D = dst->op_params[6] == 1;

  const uint32_t IC = src1->ne[is_2D
                                 ? 2
                                 : 1];
  const uint32_t IH = is_2D
                        ? src1->ne[1]
                        : 1;
  const uint32_t IW = src1->ne[0];

  const uint32_t KH = is_2D
                        ? src0->ne[1]
                        : 1;
  const uint32_t KW = src0->ne[0];

  const uint32_t OH = is_2D
                        ? dst->ne[2]
                        : 1;
  const uint32_t OW = dst->ne[1];

  const uint32_t offset_delta = src1->nb[is_2D
                                           ? 2
                                           : 1] / 4; // nb is byte offset, src is type float32
  const uint32_t batch_offset = src1->nb[is_2D
                                           ? 3
                                           : 2] / 4; // nb is byte offset, src is type float32

  const uint32_t pelements = OW * KW * KH;

  const v_backend_vk_buffer_ctx* d_buf_ctx = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  const vk_buffer d_buf                    = d_buf_ctx->dev_buffer;

  const vk::DeviceAddress dst_addr = d_buf->bda_addr + vk_tensor_offset(dst) + dst->view_offs;

  v_vk_op_f32<vk_op_im2col_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           src1,
                                           nullptr,
                                           dst,
                                           V_OP_IM2COL,
                                           {
                                             dst_addr,
                                             batch_offset, offset_delta,
                                             IC, IW, IH, OW, OH, KW, KH,
                                             pelements,
                                             IC * KH * KW,
                                             s0, s1, p0, p1, d0, d1,
                                           },
                                           dryrun);
}

void v_vk_im2col_3d(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                    const v_tensor* src1, v_tensor* dst, bool dryrun = false) {
  v_TENSOR_BINARY_OP_LOCALS

  const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
  const int32_t s1 = ((const int32_t*)(dst->op_params))[1];
  const int32_t s2 = ((const int32_t*)(dst->op_params))[2];
  const int32_t p0 = ((const int32_t*)(dst->op_params))[3];
  const int32_t p1 = ((const int32_t*)(dst->op_params))[4];
  const int32_t p2 = ((const int32_t*)(dst->op_params))[5];
  const int32_t d0 = ((const int32_t*)(dst->op_params))[6];
  const int32_t d1 = ((const int32_t*)(dst->op_params))[7];
  const int32_t d2 = ((const int32_t*)(dst->op_params))[8];
  const int32_t IC = ((const int32_t*)(dst->op_params))[9];

  const int64_t N  = ne13 / IC;
  const int64_t ID = ne12;
  const int64_t IH = ne11;
  const int64_t IW = ne10;

  const int64_t KD = ne02;
  const int64_t KH = ne01;
  const int64_t KW = ne00;

  const int64_t OD = ne3 / N;
  const int64_t OH = ne2;
  const int64_t OW = ne1;

  const v_backend_vk_buffer_ctx* d_buf_ctx = (v_backend_vk_buffer_ctx*)dst->buffer->context;
  const vk_buffer d_buf                    = d_buf_ctx->dev_buffer;

  const vk::DeviceAddress dst_addr = d_buf->bda_addr + vk_tensor_offset(dst) + dst->view_offs;

  vk_op_im2col_3d_push_constants pc{};

  pc.dst_addr             = dst_addr;
  pc.nb10                 = nb10 / v_type_size(src1->type);
  pc.nb11                 = nb11 / v_type_size(src1->type);
  pc.nb12                 = nb12 / v_type_size(src1->type);
  pc.nb13                 = nb13 / v_type_size(src1->type);
  pc.s0                   = s0;
  pc.s1                   = s1;
  pc.s2                   = s2;
  pc.p0                   = p0;
  pc.p1                   = p1;
  pc.p2                   = p2;
  pc.d0                   = d0;
  pc.d1                   = d1;
  pc.d2                   = d2;
  pc.IW                   = IW;
  pc.IH                   = IH;
  pc.ID                   = ID;
  pc.IC                   = IC;
  pc.KW                   = KW;
  pc.OH                   = OH;
  pc.KD_KH_KW             = KD * KH * KW;
  pc.KH_KW                = KH * KW;
  pc.IC_KD_KH_KW          = IC * KD * KH * KW;
  pc.N_OD_OH              = N * OD * OH;
  pc.OD_OH                = OD * OH;
  pc.OD_OH_OW_IC_KD_KH_KW = OD * OH * OW * IC * KD * KH * KW;
  pc.OH_OW_IC_KD_KH_KW    = OH * OW * IC * KD * KH * KW;
  pc.OW_IC_KD_KH_KW       = OW * IC * KD * KH * KW;

  v_vk_op_f32<vk_op_im2col_3d_push_constants>(ctx,
                                              subctx,
                                              src0,
                                              src1,
                                              nullptr,
                                              dst,
                                              v_OP_IM2COL_3D,
                                              std::move(pc),
                                              dryrun);
}

void v_vk_timestep_embedding(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                             v_tensor* dst, bool dryrun = false) {
  const uint32_t dim        = dst->op_params[0];
  const uint32_t max_period = dst->op_params[1];
  const uint32_t nb1        = dst->nb[1] / v_type_size(dst->type);

  v_vk_op_f32<vk_op_timestep_embedding_push_constants>(ctx,
                                                       subctx,
                                                       src0,
                                                       nullptr,
                                                       nullptr,
                                                       dst,
                                                       v_OP_TIMESTEP_EMBEDDING,
                                                       {
                                                         nb1, dim, max_period,
                                                       },
                                                       dryrun);
}

void v_vk_conv_transpose_1d(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                            const v_tensor* src1, v_tensor* dst, bool dryrun = false) {
  // src0: (K, Cout, Cin, 1) -- kernel
  // src1: (L, Cin, 1, 1) -- input
  // dst: (*, Cout, 1, 1)

  V_ASSERT(src0->type == v_TYPE_F32);
  V_ASSERT(src1->type == v_TYPE_F32);
  V_ASSERT(dst->type == v_TYPE_F32);

  v_TENSOR_BINARY_OP_LOCALS

  V_ASSERT(nb00 == sizeof(float));
  V_ASSERT(nb10 == sizeof(float));

  const int32_t s0 = dst->op_params[0];

  vk_op_conv_transpose_1d_push_constants p{};
  p.Cout = static_cast<uint32_t>(ne01);
  p.Cin  = static_cast<uint32_t>(ne02);
  p.K    = static_cast<uint32_t>(ne00);
  p.L    = static_cast<uint32_t>(ne10);
  p.KL   = static_cast<uint32_t>(ne0);
  p.nb01 = static_cast<uint32_t>(nb01 / nb00);
  p.nb02 = static_cast<uint32_t>(nb02 / nb00);
  p.nb11 = static_cast<uint32_t>(nb11 / nb10);
  p.nb1  = static_cast<uint32_t>(nb1 / nb0);
  p.s0   = static_cast<uint32_t>(s0);

  v_vk_op_f32(ctx, subctx, src0, src1, nullptr, dst, v_OP_CONV_TRANSPOSE_1D, std::move(p), dryrun);
}

void v_vk_pool_2d(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0, v_tensor* dst,
                  bool dryrun = false) {
  uint32_t op      = static_cast<uint32_t>(dst->op_params[0]);
  const int32_t k1 = dst->op_params[1];
  const int32_t k0 = dst->op_params[2];
  const int32_t s1 = dst->op_params[3];
  const int32_t s0 = dst->op_params[4];
  const int32_t p1 = dst->op_params[5];
  const int32_t p0 = dst->op_params[6];

  const uint32_t IH = src0->ne[1];
  const uint32_t IW = src0->ne[0];

  const uint32_t N = dst->ne[3];

  const uint32_t OC = dst->ne[2];
  const uint32_t OH = dst->ne[1];
  const uint32_t OW = dst->ne[0];

  const uint32_t parallel_elements = N * OC * OH * OW;

  v_vk_op_f32<vk_op_pool2d_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           nullptr,
                                           nullptr,
                                           dst,
                                           V_OP_POOL_2D,
                                           {
                                             IW, IH, OW, OH, OC,
                                             parallel_elements,
                                             op,
                                             k0, k1, s0, s1, p0, p1,
                                           },
                                           dryrun);
}

void v_vk_pool_2d_back(vk_backend_ctx* ctx,
                       vk_context& subctx,
                       const v_tensor* src0, v_tensor* dst,
                       bool dryrun = false) {
  uint32_t op      = static_cast<uint32_t>(dst->op_params[0]);
  const int32_t k1 = dst->op_params[1];
  const int32_t k0 = dst->op_params[2];
  const int32_t s1 = dst->op_params[3];
  const int32_t s0 = dst->op_params[4];
  const int32_t p1 = dst->op_params[5];
  const int32_t p0 = dst->op_params[6];

  const uint32_t IH = src0->ne[1];
  const uint32_t IW = src0->ne[0];

  const uint32_t N = dst->ne[3];

  const uint32_t OC = dst->ne[2];
  const uint32_t OH = dst->ne[1];
  const uint32_t OW = dst->ne[0];

  const uint32_t parallel_elements = N * OC * OH * OW;

  v_vk_op_f32<vk_op_pool2d_push_constants>(ctx,
                                           subctx,
                                           src0,
                                           nullptr,
                                           nullptr,
                                           dst,
                                           V_OP_POOL_2D_BACK,
                                           {
                                             IW, IH, OW, OH, OC,
                                             parallel_elements,
                                             op,
                                             k0, k1, s0, s1, p0, p1,
                                           },
                                           dryrun);
}

void v_vk_conv_2d(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                  const v_tensor* src1, v_tensor* dst, bool dryrun = false) {
  V_ASSERT(src0->type == v_TYPE_F32 || src0->type == v_TYPE_F16);
  V_ASSERT(src1->type == v_TYPE_F32);
  V_ASSERT(dst->type == v_TYPE_F32);

  v_TENSOR_BINARY_OP_LOCALS

  V_ASSERT(nb00 == sizeof(float) || nb00 == sizeof(v_fp16_t));
  V_ASSERT(nb10 == sizeof(float));
  V_ASSERT(nb0 == sizeof(float));

  vk_op_conv2d_push_constants p{};
  p.Cout = static_cast<uint32_t>(ne03);
  p.Cin  = static_cast<uint32_t>(ne02);
  p.N    = static_cast<uint32_t>(ne13);

  p.KW = static_cast<uint32_t>(ne00);
  p.KH = static_cast<uint32_t>(ne01);
  p.W  = static_cast<uint32_t>(ne10);
  p.H  = static_cast<uint32_t>(ne11);
  p.OW = static_cast<uint32_t>(ne0);
  p.OH = static_cast<uint32_t>(ne1);

  p.s0 = static_cast<uint32_t>(dst->op_params[0]);
  p.s1 = static_cast<uint32_t>(dst->op_params[1]);
  p.p0 = static_cast<uint32_t>(dst->op_params[2]);
  p.p1 = static_cast<uint32_t>(dst->op_params[3]);
  p.d0 = static_cast<uint32_t>(dst->op_params[4]);
  p.d1 = static_cast<uint32_t>(dst->op_params[5]);

  p.nb01 = static_cast<uint32_t>(nb01 / nb00);
  p.nb02 = static_cast<uint32_t>(nb02 / nb00);
  p.nb03 = static_cast<uint32_t>(nb03 / nb00);

  p.nb11 = static_cast<uint32_t>(nb11 / nb10);
  p.nb12 = static_cast<uint32_t>(nb12 / nb10);
  p.nb13 = static_cast<uint32_t>(nb13 / nb10);

  p.nb1 = static_cast<uint32_t>(nb1 / nb0);
  p.nb2 = static_cast<uint32_t>(nb2 / nb0);
  p.nb3 = static_cast<uint32_t>(nb3 / nb0);

  V_ASSERT(ne03 == ne2);
  V_ASSERT(ne02 == ne12);

  v_vk_op_f32(ctx, subctx, src0, src1, nullptr, dst, v_OP_CONV_2D, std::move(p), dryrun);
}

void v_vk_conv_transpose_2d(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                            const v_tensor* src1, v_tensor* dst, bool dryrun = false) {
  V_ASSERT(src0->type == v_TYPE_F32 || src0->type == v_TYPE_F16);
  V_ASSERT(src1->type == v_TYPE_F32);
  V_ASSERT(dst->type == v_TYPE_F32);

  v_TENSOR_BINARY_OP_LOCALS

  V_ASSERT(nb00 == sizeof(float) || nb00 == sizeof(v_fp16_t));
  V_ASSERT(nb10 == sizeof(float));
  V_ASSERT(nb0 == sizeof(float));

  vk_op_conv_transpose_2d_push_constants p{};
  p.Cout = static_cast<uint32_t>(ne02);
  p.Cin  = static_cast<uint32_t>(ne03);
  p.N    = static_cast<uint32_t>(ne13);

  p.KW = static_cast<uint32_t>(ne00);
  p.KH = static_cast<uint32_t>(ne01);
  p.W  = static_cast<uint32_t>(ne10);
  p.H  = static_cast<uint32_t>(ne11);
  p.OW = static_cast<uint32_t>(ne0);
  p.OH = static_cast<uint32_t>(ne1);

  p.s0 = static_cast<uint32_t>(dst->op_params[0]);
  p.s1 = static_cast<uint32_t>(dst->op_params[0]);
  p.p0 = 0;
  p.p1 = 0;
  p.d0 = 1;
  p.d1 = 1;

  p.nb01 = static_cast<uint32_t>(nb01 / nb00);
  p.nb02 = static_cast<uint32_t>(nb02 / nb00);
  p.nb03 = static_cast<uint32_t>(nb03 / nb00);

  p.nb11 = static_cast<uint32_t>(nb11 / nb10);
  p.nb12 = static_cast<uint32_t>(nb12 / nb10);
  p.nb13 = static_cast<uint32_t>(nb13 / nb10);

  p.nb1 = static_cast<uint32_t>(nb1 / nb0);
  p.nb2 = static_cast<uint32_t>(nb2 / nb0);
  p.nb3 = static_cast<uint32_t>(nb3 / nb0);

  V_ASSERT(ne02 == ne2);
  V_ASSERT(ne03 == ne12);

  v_vk_op_f32(ctx, subctx, src0, src1, nullptr, dst, v_OP_CONV_TRANSPOSE_2D, std::move(p), dryrun);
}

void v_vk_conv_2d_dw(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                     const v_tensor* src1, v_tensor* dst, bool dryrun = false) {
  vk_op_conv2d_dw_push_constants p{};
  p.ne         = nelements(dst);
  p.channels   = dst->ne[2];
  p.batches    = dst->ne[3];
  p.dst_w      = dst->ne[0];
  p.dst_h      = dst->ne[1];
  p.src_w      = src1->ne[0];
  p.src_h      = src1->ne[1];
  p.knl_w      = src0->ne[0];
  p.knl_h      = src0->ne[1];
  p.stride_x   = dst->op_params[0];
  p.stride_y   = dst->op_params[1];
  p.pad_x      = dst->op_params[2];
  p.pad_y      = dst->op_params[3];
  p.dilation_x = dst->op_params[4];
  p.dilation_y = dst->op_params[5];

  V_ASSERT(src0->ne[3] == p.channels);
  V_ASSERT(src1->ne[3] == p.batches);

  v_vk_op_f32(ctx, subctx, src0, src1, nullptr, dst, v_OP_CONV_2D_DW, std::move(p), dryrun);
}

void v_vk_leaky_relu(vk_backend_ctx* ctx, vk_context& subctx, const v_tensor* src0,
                     v_tensor* dst, bool dryrun = false) {
  const float* op_params = (const float*)dst->op_params;
  v_vk_op_f32<vk_op_push_constants>(ctx,
                                    subctx,
                                    src0,
                                    nullptr,
                                    nullptr,
                                    dst,
                                    v_OP_LEAKY_RELU,
                                    {(uint32_t)nelements(src0), 0, op_params[0], 0.0f},
                                    dryrun);
}

void vk_preallocate_buffers(vk_backend_ctx* ctx) {
  #if defined(v_VULKAN_RUN_TESTS)
  const std::vector<size_t> vals{
    512, 512, 128,
    128, 512, 512,
    4096, 512, 4096,
    11008, 512, 4096,
    4096, 512, 11008,
    32000, 512, 4096,
    8, 8, 8,
    100, 46, 576,
    623, 111, 128,
    100, 46, 558,
    512, 1, 256,
    128, 110, 622,
    511, 511, 127,
    511, 511, 7,
    511, 511, 17,
    49, 49, 128,
    128, 49, 49,
    4096, 49, 4096,
  };
  const size_t num_it = 100;

  v_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 0, v_TYPE_Q4_0);
  v_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 1, v_TYPE_Q4_0);
  v_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 2, v_TYPE_Q4_0);

  v_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 0, v_TYPE_Q4_0, true);
  v_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 1, v_TYPE_Q4_0, true);
  v_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 2, v_TYPE_Q4_0, true);

  v_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 0, v_TYPE_Q8_0);
  v_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 1, v_TYPE_Q8_0);
  v_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 2, v_TYPE_Q8_0);

  v_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 0, v_TYPE_Q8_0, true);
  v_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 1, v_TYPE_Q8_0, true);
  v_vk_test_dequant_matmul(ctx, 4096, 512, 4096, 2, num_it, 1, 2, v_TYPE_Q8_0, true);

  abort();

  for (size_t i = 0; i < vals.size(); i += 3) {
    v_vk_test_matmul<v_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 0);
    v_vk_test_matmul<v_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 1);
    v_vk_test_matmul<v_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 2);
    std::cerr << '\n';
    v_vk_test_matmul<v_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 0);
    v_vk_test_matmul<v_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 1);
    v_vk_test_matmul<v_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 2);
    std::cerr << '\n';
    v_vk_test_matmul<v_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 0);
    v_vk_test_matmul<v_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 1);
    v_vk_test_matmul<v_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 2);
    std::cerr << '\n' << std::endl;

    if (vals[i + 2] % 32 == 0) {
      v_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 0, v_TYPE_Q4_0);
      v_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 1, v_TYPE_Q4_0);
      v_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 2, v_TYPE_Q4_0);
      std::cerr << '\n';
      v_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 0, v_TYPE_Q4_0);
      v_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 1, v_TYPE_Q4_0);
      v_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 2, v_TYPE_Q4_0);
      std::cerr << '\n';
      v_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 0, v_TYPE_Q4_0);
      v_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 1, v_TYPE_Q4_0);
      v_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 2, v_TYPE_Q4_0);
      std::cerr << '\n' << std::endl;
    }

    if (vals[i + 2] % 256 == 0) {
      v_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 0, v_TYPE_Q4_K);
      v_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 1, v_TYPE_Q4_K);
      v_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 2, v_TYPE_Q4_K);
      std::cerr << '\n';
      v_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 0, v_TYPE_Q4_K);
      v_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 1, v_TYPE_Q4_K);
      v_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 2, 2, v_TYPE_Q4_K);
      std::cerr << '\n';
      v_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 0, v_TYPE_Q4_K);
      v_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 1, v_TYPE_Q4_K);
      v_vk_test_dequant_matmul(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 2, v_TYPE_Q4_K);
      std::cerr << '\n' << std::endl;
    }
  }

  v_ABORT("fatal error");
  #endif

  if (ctx->prealloc_x == nullptr || (ctx->prealloc_size_x > 0 && ctx->prealloc_x->size < ctx->prealloc_size_x)) {
    VK_LOG_MEMORY("v_vk_preallocate_buffers(x_size: " << ctx->prealloc_size_x << ")");
    // Resize buffer
    if (ctx->prealloc_x != nullptr) { vk_destroy_buffer(ctx->prealloc_x); }
    ctx->prealloc_x = v_vk_create_buffer_device(ctx->device, ctx->prealloc_size_x);
  }
  if (ctx->prealloc_y == nullptr || (ctx->prealloc_size_y > 0 && ctx->prealloc_y->size < ctx->prealloc_size_y)) {
    VK_LOG_MEMORY("v_vk_preallocate_buffers(y_size: " << ctx->prealloc_size_y << ")");
    // Resize buffer
    if (ctx->prealloc_y != nullptr) { vk_destroy_buffer(ctx->prealloc_y); }
    ctx->prealloc_y = v_vk_create_buffer_device(ctx->device, ctx->prealloc_size_y);
  }
  if (ctx->prealloc_split_k == nullptr || (ctx->prealloc_size_split_k > 0 && ctx->prealloc_split_k->size < ctx->
    prealloc_size_split_k)) {
    VK_LOG_MEMORY("v_vk_preallocate_buffers(split_k_size: " << ctx->prealloc_size_split_k << ")");
    // Resize buffer
    if (ctx->prealloc_split_k != nullptr) { vk_destroy_buffer(ctx->prealloc_split_k); }
    ctx->prealloc_split_k = v_vk_create_buffer_device(ctx->device, ctx->prealloc_size_split_k);
  }
  if (ctx->prealloc_add_rms_partials == nullptr || (ctx->prealloc_size_add_rms_partials > 0 && ctx->
                                                                                               prealloc_add_rms_partials
                                                                                               ->size < ctx->
    prealloc_size_add_rms_partials)) {
    VK_LOG_MEMORY("v_vk_preallocate_buffers(add_partials_size: " << ctx->prealloc_add_rms_partials << ")");
    // Resize buffer
    if (ctx->prealloc_add_rms_partials != nullptr) { vk_destroy_buffer(ctx->prealloc_add_rms_partials); }
    ctx->prealloc_add_rms_partials = v_vk_create_buffer_device(ctx->device, ctx->prealloc_size_add_rms_partials);
  }
}

bool v_vk_compute_forward(vk_backend_ctx* ctx, v_cgraph* cgraph, v_tensor* tensor,
                          int tensor_idx, bool use_fence, bool almost_ready);

// Returns true if node has enqueued work into the queue, false otherwise
// If submit is true the current all operations queued so far are being submitted to Vulkan to overlap cmdlist creation and GPU execution.
bool vk_build_graph(vk_backend_ctx* ctx, v_cgraph* cgraph, int node_idx,
                    v_tensor* node_begin, int node_idx_begin, bool dryrun, bool last_node,
                    bool almost_ready, bool submit) {
  v_tensor* node = cgraph->nodes[node_idx];
  if (is_empty(node) || !node->buffer) { return false; }

  VK_LOG_DEBUG("v_vk_build_graph(" << node << ", " << op_name(node->op) << ")");
  ctx->semaphore_idx = 0;

  v_tensor* src0 = node->src[0];
  v_tensor* src1 = node->src[1];
  v_tensor* src2 = node->src[2];
  v_tensor* src3 = node->src[3];

  switch (node->op) {
    // Return on empty ops to avoid generating a compute_ctx and setting exit_tensor
    case v_OP_RESHAPE:
    case V_OP_VIEW:
    case V_OP_PERMUTE:
    case v_OP_TRANSPOSE:
    case v_OP_NONE:
      return false;
    case v_OP_UNARY:
      switch (v_get_unary_op(node)) {
        case v_UNARY_OP_EXP:
        case v_UNARY_OP_SILU:
        case v_UNARY_OP_GELU:
        case v_UNARY_OP_LOG:
        case v_UNARY_OP_GELU_ERF:
        case v_UNARY_OP_GELU_QUICK:
        case v_UNARY_OP_RELU:
        case v_UNARY_OP_TANH:
        case v_UNARY_OP_SIGMOID:
        case v_UNARY_OP_HARDSIGMOID:
        case v_UNARY_OP_HARDSWISH:
          break;
        default:
          std::cout << " false case operation :: " << v_get_unary_op(node) << std::endl;
          throw std::runtime_error("fail to operate ");
          return false;
      }
      break;
    case v_OP_GLU:
      switch (v_get_glu_op(node)) {
        case v_GLU_OP_GEGLU:
        case v_GLU_OP_REGLU:
        case v_GLU_OP_SWIGLU:
        case v_GLU_OP_SWIGLU_OAI:
        case v_GLU_OP_GEGLU_ERF:
        case v_GLU_OP_GEGLU_QUICK:
          break;
        default:
          return false;
      }
      break;
    case v_OP_ADD: {
      int next_node_idx = node_idx + 1 + ctx->num_additional_fused_ops;
      if (next_node_idx < cgraph->n_nodes &&
        cgraph->nodes[next_node_idx]->op == v_OP_RMS_NORM &&
        cgraph->nodes[next_node_idx]->src[0] == cgraph->nodes[next_node_idx - 1] &&
        v_nrows(cgraph->nodes[next_node_idx]) == 1 &&
        ctx->device->add_rms_fusion) {
        if (dryrun) { ctx->prealloc_size_add_rms_partials += v_vk_rms_partials_size(ctx, cgraph->nodes[node_idx]); }
        ctx->do_add_rms_partials = true;
      }
    }
    break;
    case v_OP_REPEAT:
    case v_OP_REPEAT_BACK:
    case v_OP_GET_ROWS:
    case v_OP_ADD_ID:
    case v_OP_ACC:
    case v_OP_SUB:
    case v_OP_MUL:
    case v_OP_DIV:
    case v_OP_CONCAT:
    case v_OP_UPSCALE:
    case V_OP_SCALE:
    case v_OP_SQR:
    case v_OP_SQRT:
    case v_OP_SIN:
    case v_OP_COS:
    case v_OP_CLAMP:
    case v_OP_PAD:
    case v_OP_ROLL:
    case v_OP_CPY:
    case v_OP_SET_ROWS:
    case v_OP_CONT:
    case v_OP_DUP:
    case v_OP_SILU_BACK:
    case v_OP_NORM:
    case v_OP_GROUP_NORM:
    case v_OP_RMS_NORM:
    case v_OP_RMS_NORM_BACK:
    case v_OP_L2_NORM:
    case V_OP_DIAG_MASK_INF:
    case V_OP_SOFT_MAX:
    case v_OP_SOFT_MAX_BACK:
    case V_OP_ROPE:
    case v_OP_ROPE_BACK:
    case v_OP_MUL_MAT:
    case v_OP_MUL_MAT_ID:
    case v_OP_ARGSORT:
    case v_OP_SUM:
    case v_OP_SUM_ROWS:
    case V_OP_MEAN:
    case V_OP_ARGMAX:
    case v_OP_COUNT_EQUAL:
    case V_OP_IM2COL:
    case v_OP_IM2COL_3D:
    case v_OP_TIMESTEP_EMBEDDING:
    case v_OP_CONV_TRANSPOSE_1D:
    case V_OP_POOL_2D:
    case V_OP_POOL_2D_BACK:
    case v_OP_CONV_2D:
    case v_OP_CONV_TRANSPOSE_2D:
    case v_OP_CONV_2D_DW:
    case v_OP_RWKV_WKV6:
    case v_OP_RWKV_WKV7:
    case v_OP_SSM_SCAN:
    case v_OP_SSM_CONV:
    case v_OP_LEAKY_RELU:
    case v_OP_FLASH_ATTN_EXT:
    case v_OP_OPT_STEP_ADAMW:
    case v_OP_OPT_STEP_SGD:
      break;
    default:
      std::cerr << "v_vulkan: Error: Missing op: " << v_op_name(node->op) << std::endl;
      //v_ABORT("fatal error");
  }

  vk_context compute_ctx;

  if (!dryrun) {
    if (ctx->compute_ctx.expired()) {
      compute_ctx      = vk_create_context(ctx, ctx->compute_cmd_pool);
      ctx->compute_ctx = compute_ctx;
      vk_begin_ctx(ctx->device, compute_ctx);
    }
    else { compute_ctx = ctx->compute_ctx.lock(); }
  }
  else {
    switch (node->op) {
      case v_OP_REPEAT:
      case v_OP_REPEAT_BACK:
      case v_OP_ACC:
      case v_OP_GET_ROWS:
      case v_OP_ADD:
      case v_OP_SUB:
      case v_OP_MUL:
      case v_OP_DIV:
      case v_OP_CONCAT:
      case v_OP_UPSCALE:
      case V_OP_SCALE:
      case v_OP_SQR:
      case v_OP_SQRT:
      case v_OP_SIN:
      case v_OP_COS:
      case v_OP_CLAMP:
      case v_OP_PAD:
      case v_OP_CPY:
      case v_OP_SET_ROWS:
      case v_OP_CONT:
      case v_OP_DUP:
      case v_OP_SILU_BACK:
      case v_OP_NORM:
      case v_OP_GROUP_NORM:
      case v_OP_RMS_NORM:
      case v_OP_RMS_NORM_BACK:
      case v_OP_L2_NORM:
      case v_OP_UNARY:
      case v_OP_GLU:
      case V_OP_DIAG_MASK_INF:
      case V_OP_SOFT_MAX:
      case v_OP_SOFT_MAX_BACK:
      case V_OP_ROPE:
      case v_OP_ROPE_BACK:
      case v_OP_ARGSORT:
      case v_OP_SUM:
      case v_OP_SUM_ROWS:
      case V_OP_MEAN:
      case V_OP_ARGMAX:
      case v_OP_COUNT_EQUAL:
      case V_OP_IM2COL:
      case v_OP_IM2COL_3D:
      case v_OP_TIMESTEP_EMBEDDING:
      case v_OP_CONV_TRANSPOSE_1D:
      case V_OP_POOL_2D:
      case V_OP_POOL_2D_BACK:
      case v_OP_CONV_2D:
      case v_OP_CONV_TRANSPOSE_2D:
      case v_OP_CONV_2D_DW:
      case v_OP_LEAKY_RELU:
      case v_OP_OPT_STEP_SGD: {
        // These operations all go through v_vk_op_f32, so short-circuit and
        // do the only thing needed for the dryrun.
        vk_pipeline pipeline = v_vk_op_get_pipeline(ctx, src0, src1, src2, node, node->op);
        v_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        if (node->op == v_OP_RMS_NORM) { ctx->do_add_rms_partials = false; }
        return false;
      }
      default:
        break;
    }
  }

  if (!dryrun) {
    // This logic detects dependencies between modes in the graph and calls v_vk_sync_buffers
    // to synchronize them. This handles most "normal" synchronization when computing the graph, and when
    // there is no auxiliary memory use, it shouldn't be necessary to call v_vk_sync_buffers
    // outside of this logic. When a node uses one of the prealloc buffers for something like
    // dequantization or split_k, additional synchronization is needed between those passes.
    bool need_sync = false;
    // Check whether "node" requires synchronization. The node requires synchronization if it
    // overlaps in memory with another unsynchronized node and at least one of them is a write.
    // Destination nodes are checked against both the written/read lists. Source nodes are only
    // checked against the written list. Two nodes overlap in memory if they come from the same
    // buffer and the tensor or view ranges overlap.
    auto const& overlaps_unsynced = [&](const v_tensor* node,
                                        const std::vector<const v_tensor*>& unsynced_nodes) -> bool {
      if (unsynced_nodes.size() == 0) { return false; }
      auto n_base                        = vk_tensor_offset(node) + node->view_offs;
      auto n_size                        = num_bytes(node);
      v_backend_vk_buffer_ctx* a_buf_ctx = (v_backend_vk_buffer_ctx*)node->buffer->context;
      vk_buffer a_buf                    = a_buf_ctx->dev_buffer;

      for (auto& other : unsynced_nodes) {
        v_backend_vk_buffer_ctx* o_buf_ctx = (v_backend_vk_buffer_ctx*)other->buffer->context;
        vk_buffer o_buf                    = o_buf_ctx->dev_buffer;
        if (a_buf == o_buf) {
          auto o_base = vk_tensor_offset(other) + other->view_offs;
          auto o_size = num_bytes(other);

          if ((o_base <= n_base && n_base < o_base + o_size) ||
            (n_base <= o_base && o_base < n_base + n_size)) { return true; }
        }
      }
      return false;
    };

    // For all fused ops, check if the destination node or any of the source
    // nodes require synchronization.
    for (int32_t i = 0; i < ctx->num_additional_fused_ops + 1 && !need_sync; ++i) {
      const v_tensor* cur_node = cgraph->nodes[node_idx + i];
      if (overlaps_unsynced(cur_node, ctx->unsynced_nodes_read) || overlaps_unsynced(
        cur_node,
        ctx->unsynced_nodes_written)) {
        need_sync = true;
        break;
      }
      for (uint32_t j = 0; j < v_MAX_SRC; ++j) {
        if (!cur_node->src[j]) { continue; }
        if (overlaps_unsynced(cur_node->src[j], ctx->unsynced_nodes_written)) {
          need_sync = true;
          break;
        }
      }
    }
    if (need_sync) {
      ctx->unsynced_nodes_written.clear();
      ctx->unsynced_nodes_read.clear();
      vk_sync_buffers(ctx, compute_ctx);
    }
    // Add all fused nodes to the unsynchronized lists.
    for (int32_t i = 0; i < ctx->num_additional_fused_ops + 1; ++i) {
      const v_tensor* cur_node = cgraph->nodes[node_idx + i];
      // Multiple outputs could be written, e.g. in topk_moe. Add them all to the list.
      ctx->unsynced_nodes_written.push_back(cur_node);
      for (uint32_t j = 0; j < v_MAX_SRC; ++j) {
        if (!cur_node->src[j]) { continue; }
        ctx->unsynced_nodes_read.push_back(cur_node->src[j]);
      }
    }
  }

  switch (node->op) {
    case v_OP_REPEAT:
      v_vk_repeat(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_REPEAT_BACK:
      v_vk_repeat_back(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_ACC:
      v_vk_acc(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_GET_ROWS:
      v_vk_get_rows(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_ADD:
      if (ctx->num_additional_fused_ops) { v_vk_multi_add(ctx, compute_ctx, cgraph, node_idx, dryrun); }
      else { v_vk_add(ctx, compute_ctx, src0, src1, node, dryrun); }
      break;
    case v_OP_SUB:
      v_vk_sub(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_MUL:
      v_vk_mul(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_DIV:
      v_vk_div(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_ADD_ID:
      v_vk_add_id(ctx, compute_ctx, src0, src1, src2, node, dryrun);

      break;
    case v_OP_CONCAT:
      v_vk_concat(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_UPSCALE:
      v_vk_upscale(ctx, compute_ctx, src0, node, dryrun);

      break;
    case V_OP_SCALE:
      v_vk_scale(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_SQR:
      v_vk_sqr(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_SQRT:
      v_vk_sqrt(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_SIN:
      v_vk_sin(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_COS:
      v_vk_cos(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_CLAMP:
      v_vk_clamp(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_PAD:
      v_vk_pad(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_ROLL:
      v_vk_roll(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_CPY:
    case v_OP_CONT:
    case v_OP_DUP:
      v_vk_cpy(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_SET_ROWS:
      v_vk_set_rows(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_SILU_BACK: v_vk_silu_back(ctx, compute_ctx, src0, src1, node, dryrun);
      break;
    case v_OP_NORM: v_vk_norm(ctx, compute_ctx, src0, node, dryrun);
      break;
    case v_OP_GROUP_NORM:
      v_vk_group_norm(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_RMS_NORM:
      if (ctx->num_additional_fused_ops > 0) {
        // fused rms_norm + mul
        v_tensor* mul       = cgraph->nodes[node_idx + 1];
        v_tensor* other_src = mul->src[0] == node
                                ? mul->src[1]
                                : mul->src[0];
        v_vk_rms_norm(ctx, compute_ctx, src0, other_src, mul, (float*)node->op_params, dryrun);
      }
      else { v_vk_rms_norm(ctx, compute_ctx, src0, src0, node, (float*)node->op_params, dryrun); }
      break;
    case v_OP_RMS_NORM_BACK:
      v_vk_rms_norm_back(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_L2_NORM:
      v_vk_l2_norm(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_UNARY:
      switch (v_get_unary_op(node)) {
        case v_UNARY_OP_EXP:
        case v_UNARY_OP_SILU:
        case v_UNARY_OP_GELU:
        case v_UNARY_OP_GELU_ERF:
        case v_UNARY_OP_GELU_QUICK:
        case v_UNARY_OP_RELU:
        case v_UNARY_OP_TANH:
        case v_UNARY_OP_SIGMOID:
        case v_UNARY_OP_LOG:
        case v_UNARY_OP_HARDSIGMOID:
        case v_UNARY_OP_HARDSWISH:
          v_vk_unary(ctx, compute_ctx, src0, node, dryrun);
          break;
        default:
          std::cout << " false case operation :: " << v_get_unary_op(node) << std::endl;
          throw std::runtime_error("fail to operate ");

          return false;
      }
      break;
    case v_OP_GLU:
      switch (v_get_glu_op(node)) {
        case v_GLU_OP_GEGLU:
        case v_GLU_OP_REGLU:
        case v_GLU_OP_SWIGLU:
        case v_GLU_OP_SWIGLU_OAI:
        case v_GLU_OP_GEGLU_ERF:
        case v_GLU_OP_GEGLU_QUICK:
          v_vk_glu(ctx, compute_ctx, src0, src1, node, dryrun);
          break;
        default:
          return false;
      }
      break;
    case V_OP_DIAG_MASK_INF:
      v_vk_diag_mask_inf(ctx, compute_ctx, src0, node, dryrun);

      break;
    case V_OP_SOFT_MAX:
      if (ctx->num_additional_fused_ops) { v_vk_topk_moe(ctx, compute_ctx, cgraph, node_idx, dryrun); }
      else { v_vk_soft_max(ctx, compute_ctx, src0, src1, src2, node, dryrun); }

      break;
    case v_OP_SOFT_MAX_BACK:
      v_vk_soft_max_back(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case V_OP_ROPE:
      v_vk_rope(ctx, compute_ctx, src0, src1, src2, node, false, dryrun);

      break;
    case v_OP_ROPE_BACK:
      v_vk_rope(ctx, compute_ctx, src0, src1, src2, node, true, dryrun);

      break;
    case v_OP_ARGSORT:
      v_vk_argsort(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_SUM:
      v_vk_sum(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_SUM_ROWS:
      v_vk_sum_rows(ctx, compute_ctx, src0, node, dryrun);

      break;
    case V_OP_MEAN:
      v_vk_mean(ctx, compute_ctx, src0, node, dryrun);

      break;
    case V_OP_ARGMAX:
      v_vk_argmax(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_COUNT_EQUAL:
      v_vk_count_equal(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case V_OP_IM2COL:
      v_vk_im2col(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_IM2COL_3D:
      v_vk_im2col_3d(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_TIMESTEP_EMBEDDING:
      v_vk_timestep_embedding(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_CONV_TRANSPOSE_1D:
      v_vk_conv_transpose_1d(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case V_OP_POOL_2D:
      v_vk_pool_2d(ctx, compute_ctx, src0, node, dryrun);
      break;

    case V_OP_POOL_2D_BACK :
      v_vk_pool_2d_back(ctx, compute_ctx, src0, node, dryrun);
      break;
    case v_OP_CONV_2D:
      v_vk_conv_2d(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_CONV_TRANSPOSE_2D:
      v_vk_conv_transpose_2d(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_CONV_2D_DW:
      v_vk_conv_2d_dw(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_LEAKY_RELU:
      v_vk_leaky_relu(ctx, compute_ctx, src0, node, dryrun);

      break;
    case v_OP_MUL_MAT:
      v_vk_mul_mat(ctx, compute_ctx, src0, src1, node, dryrun);

      break;
    case v_OP_MUL_MAT_ID:
      v_vk_mul_mat_id(ctx, compute_ctx, src0, src1, src2, node, dryrun);

      break;

    case v_OP_FLASH_ATTN_EXT:
      v_vk_flash_attn(ctx, compute_ctx, src0, src1, src2, src3, node->src[4], node, dryrun);

      break;

    case v_OP_RWKV_WKV6:
      v_vk_rwkv_wkv6(ctx, compute_ctx, node, dryrun);

      break;

    case v_OP_RWKV_WKV7:
      v_vk_rwkv_wkv7(ctx, compute_ctx, node, dryrun);

      break;

    case v_OP_SSM_SCAN:
      v_vk_ssm_scan(ctx, compute_ctx, node, dryrun);

      break;

    case v_OP_SSM_CONV:
      v_vk_ssm_conv(ctx, compute_ctx, node, dryrun);

      break;

    case v_OP_OPT_STEP_ADAMW:
      v_vk_opt_step_adamw(ctx, compute_ctx, node, dryrun);

      break;

    case v_OP_OPT_STEP_SGD:
      v_vk_opt_step_sgd(ctx, compute_ctx, src0, src1, src2, node, dryrun);

      break;
    default:
      return false;
  }

  if (dryrun) { return false; }

  ctx->tensor_ctxs[node_idx] = compute_ctx;

  #if defined(v_VULKAN_CHECK_RESULTS)
  // Force context reset on each node so that each tensor ends up in its own context
  // and can be run and compared to its CPU equivalent separately
  last_node = true;
  #endif

  if (submit || last_node) {
    vk_ctx_end(compute_ctx);

    // TODO probably it'd be better to pass a exit_node flag to v_vk_compute_forward
    if (last_node) { compute_ctx->exit_tensor_idx = node_idx_begin; }
    else { compute_ctx->exit_tensor_idx = -1; }

    ctx->compute_ctx.reset();

    bool ok = v_vk_compute_forward(ctx, cgraph, node_begin, node_idx_begin, false, almost_ready);
    if (!ok) {
      if (node->op == v_OP_UNARY) {
        std::cerr << __func__ << ": error: op not supported UNARY " << node->name << " (" << v_unary_op_name(
          static_cast<v_unary_op>(node->op_params[0])) << ")" << std::endl;
      }
      else if (node->op == v_OP_GLU) {
        std::cerr << __func__ << ": error: op not supported GLU " << node->name << " (" << v_glu_op_name(
          static_cast<v_glu_op>(node->op_params[0])) << ")" << std::endl;
      }
      else {
        std::cerr << __func__ << ": error: op not supported " << node->name << " (" << v_op_name(node->op) << ")" <<
          std::endl;
      }
    }
  }
  return true;
}

bool v_vk_compute_forward(vk_backend_ctx* ctx, v_cgraph* cgraph, v_tensor* tensor,
                          int tensor_idx, bool use_fence = true, bool almost_ready = false) {
  v_UNUSED(cgraph);
  v_backend_buffer* buf = nullptr;

  switch (tensor->op) {
    case v_OP_ADD:
    case v_OP_ACC:
    case v_OP_GET_ROWS:
    case v_OP_SUB:
    case v_OP_MUL:
    case v_OP_DIV:
    case v_OP_ADD_ID:
    case v_OP_CONCAT:
    case v_OP_UPSCALE:
    case V_OP_SCALE:
    case v_OP_SQR:
    case v_OP_SQRT:
    case v_OP_SIN:
    case v_OP_COS:
    case v_OP_CLAMP:
    case v_OP_PAD:
    case v_OP_ROLL:
    case v_OP_CPY:
    case v_OP_SET_ROWS:
    case v_OP_CONT:
    case v_OP_DUP:
    case v_OP_SILU_BACK:
    case v_OP_NORM:
    case v_OP_GROUP_NORM:
    case v_OP_RMS_NORM:
    case v_OP_RMS_NORM_BACK:
    case v_OP_L2_NORM:
    case V_OP_DIAG_MASK_INF:
    case V_OP_SOFT_MAX:
    case v_OP_SOFT_MAX_BACK:
    case V_OP_ROPE:
    case v_OP_ROPE_BACK:
    case v_OP_RESHAPE:
    case V_OP_VIEW:
    case V_OP_PERMUTE:
    case v_OP_TRANSPOSE:
    case v_OP_NONE:
    case v_OP_ARGSORT:
    case v_OP_SUM:
    case v_OP_SUM_ROWS:
    case V_OP_MEAN:
    case V_OP_ARGMAX:
    case v_OP_COUNT_EQUAL:
    case V_OP_IM2COL:
    case v_OP_IM2COL_3D:
    case v_OP_TIMESTEP_EMBEDDING:
    case v_OP_CONV_TRANSPOSE_1D:
    case V_OP_POOL_2D:
    case V_OP_POOL_2D_BACK:
    case v_OP_CONV_2D:
    case v_OP_CONV_TRANSPOSE_2D:
    case v_OP_CONV_2D_DW:
    case v_OP_RWKV_WKV6:
    case v_OP_RWKV_WKV7:
    case v_OP_SSM_SCAN:
    case v_OP_SSM_CONV:
    case v_OP_LEAKY_RELU:
    case v_OP_REPEAT:
    case v_OP_REPEAT_BACK:
    case v_OP_OPT_STEP_ADAMW:
    case v_OP_OPT_STEP_SGD:
      buf = tensor->buffer;
      break;
    case v_OP_UNARY:
      switch (v_get_unary_op(tensor)) {
        case v_UNARY_OP_EXP:
        case v_UNARY_OP_LOG:
        case v_UNARY_OP_SILU:
        case v_UNARY_OP_GELU:
        case v_UNARY_OP_GELU_ERF:
        case v_UNARY_OP_GELU_QUICK:
        case v_UNARY_OP_RELU:
        case v_UNARY_OP_TANH:
        case v_UNARY_OP_SIGMOID:
        case v_UNARY_OP_HARDSIGMOID:
        case v_UNARY_OP_HARDSWISH:
          buf = tensor->buffer;
          break;
        default:
          return false;
      }
      break;
    case v_OP_GLU:
      switch (v_get_glu_op(tensor)) {
        case v_GLU_OP_GEGLU:
        case v_GLU_OP_REGLU:
        case v_GLU_OP_SWIGLU:
        case v_GLU_OP_SWIGLU_OAI:
        case v_GLU_OP_GEGLU_ERF:
        case v_GLU_OP_GEGLU_QUICK:
          buf = tensor->buffer;
          break;
        default:
          return false;
      }
      break;
    case v_OP_MUL_MAT:
    case v_OP_MUL_MAT_ID:
    case v_OP_FLASH_ATTN_EXT:
      buf = tensor->buffer;

      break;
    default:
      return false;
  }

  if (buf == nullptr) { return false; }

  VK_LOG_DEBUG(
    "v_vk_compute_forward(" << tensor << ", name=" << tensor->name << ", op=" << op_name(tensor->op) <<
    ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] <<
    ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] <<
    ", nb3=" << tensor->nb[3] << ", view_src=" << tensor->view_src << ", view_offs=" << tensor->view_offs << ")");

  vk_context subctx = ctx->tensor_ctxs[tensor_idx].lock();
  // always wait for the GPU work to be done for the last submit
  if (tensor_idx == subctx->exit_tensor_idx) { use_fence = true; }

  // Only run if ctx hasn't been submitted yet
  if (!subctx->seqs.empty()) {
    #ifdef v_VULKAN_CHECK_RESULTS
    vk_check_results_0(ctx, cgraph, tensor_idx);
    use_fence = true;
    #endif

    // Do staging buffer copies
    for (auto& cpy : subctx->in_memcpys) { memcpy(cpy.dst, cpy.src, cpy.n); }

    for (auto& mset : subctx->memsets) { memset(mset.dst, mset.val, mset.n); }

    if (almost_ready && !ctx->almost_ready_fence_pending && !use_fence) {
      vk_submit(subctx, ctx->almost_ready_fence);
      ctx->almost_ready_fence_pending = true;
    }
    else {
      vk_submit(subctx, use_fence
                            ? ctx->fence
                            : vk::Fence{});
    }

    if (use_fence) { v_vk_wait_for_fence(ctx); }
    #ifdef v_VULKAN_CHECK_RESULTS
    vk_check_results_1(ctx, cgraph, tensor_idx);
    #endif
  }

  if (tensor_idx == subctx->exit_tensor_idx) {
    // Do staging buffer copies
    for (auto& cpy : subctx->out_memcpys) { memcpy(cpy.dst, cpy.src, cpy.n); }
    subctx->in_memcpys.clear();
    subctx->out_memcpys.clear();
    subctx->memsets.clear();
  }

  return true;
}

// Clean up after graph processing is done
void vk_graph_cleanup(vk_backend_ctx* ctx) {
  VK_LOG_DEBUG("v_vk_graph_cleanup()");
  ctx->prealloc_y_last_pipeline_used = {};

  ctx->unsynced_nodes_written.clear();
  ctx->unsynced_nodes_read.clear();
  ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_split_k_need_sync = false;

  v_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);
  v_vk_command_pool_cleanup(ctx->device, ctx->transfer_cmd_pool);

  for (size_t i = 0; i < ctx->gc.semaphores.size(); i++) { ctx->device->device.destroySemaphore({ctx->gc.semaphores[i].s}); }
  ctx->gc.semaphores.clear();

  for (size_t i = 0; i < ctx->gc.tl_semaphores.size(); i++) { ctx->device->device.destroySemaphore({ctx->gc.tl_semaphores[i].s}); }
  ctx->gc.tl_semaphores.clear();
  ctx->semaphore_idx = 0;

  ctx->event_idx = 0;

  for (auto& event : ctx->gc.events) { ctx->device->device.resetEvent(event); }

  ctx->tensor_ctxs.clear();
  ctx->gc.contexts.clear();
  ctx->pipeline_descriptor_set_requirements = 0;
  ctx->descriptor_set_idx                   = 0;
}

// Clean up on backend free
void v_vk_cleanup(vk_backend_ctx* ctx) {
  VK_LOG_DEBUG("v_vk_cleanup(" << ctx->name << ")");
  vk_graph_cleanup(ctx);

  vk_destroy_buffer(ctx->prealloc_x);
  vk_destroy_buffer(ctx->prealloc_y);
  vk_destroy_buffer(ctx->prealloc_split_k);
  ctx->prealloc_y_last_pipeline_used = nullptr;

  ctx->prealloc_size_x       = 0;
  ctx->prealloc_size_y       = 0;
  ctx->prealloc_size_split_k = 0;

  for (auto& event : ctx->gc.events) { ctx->device->device.destroyEvent(event); }
  ctx->gc.events.clear();

  ctx->device->device.destroyFence(ctx->fence);
  ctx->device->device.destroyFence(ctx->almost_ready_fence);

  for (auto& pool : ctx->descriptor_pools) { ctx->device->device.destroyDescriptorPool(pool); }
  ctx->descriptor_pools.clear();
  ctx->descriptor_sets.clear();

  ctx->compute_cmd_pool.destroy(ctx->device->device);
  ctx->transfer_cmd_pool.destroy(ctx->device->device);
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

// backend interface

#define UNUSED v_UNUSED

const char* vk_device_buffer_name(v_backend_buffer_type_t buft) {
  vk_buffer_type_context* ctx = (vk_buffer_type_context*)buft->context;
  return ctx->name.c_str();
}

size_t vk_device_buffer_get_align(v_backend_buffer_type_t buft) {
  vk_buffer_type_context* ctx = (vk_buffer_type_context*)buft->context;
  return ctx->device->properties.limits.minStorageBufferOffsetAlignment;
}

size_t vk_device_buffer_get_max_size(v_backend_buffer_type_t buft) {
  vk_buffer_type_context* ctx = (vk_buffer_type_context*)buft->context;
  return ctx->device->suballocation_block_size;
}

size_t vk_device_buffer_get_alloc_size(v_backend_buffer_type_t buft, const v_tensor* tensor) {
  return num_bytes(tensor);
  UNUSED(buft);
}

v_backend_buffer_type_t vk_device_buffer_type(size_t dev_num) {
  vk_instance_init();
  VK_LOG_DEBUG("v_backend_vk_buffer_type(" << dev_num << ")");
  vk_device dev = v_vk_get_device(dev_num);
  return &dev->buffer_type;
}

const char* vk_host_buffer_name(v_backend_buffer_type_t buft) {
  return v_VK_NAME "_Host";
  UNUSED(buft);
}

void vk_host_buffer_free(v_backend_buffer_t buffer) {
  VK_LOG_MEMORY("v_backend_vk_host_buffer_free_buffer()");
  vk_host_free(vk_instance.devices[0], buffer->context);
}

v_backend_buffer_t vk_host_buffer_alloc(v_backend_buffer_type_t buft,
                                        size_t size) {
  VK_LOG_MEMORY("v_backend_vk_host_buffer_type_alloc_buffer(" << size << ")");
  size += 32; // Behave like the CPU buffer type
  void* ptr = nullptr;
  try { ptr = vkHostMalloc(vk_instance.devices[0], size); }
  catch (vk::SystemError& e) {
    v_LOG_WARN("v_vulkan: Failed to allocate pinned memory (%s)\n", e.what());
    throw std::runtime_error("fail to alloc host mem");
  }
  struct v_backend_buffer* buffer = v_backend_cpu_buffer_from_ptr(ptr, size);
  buffer->buft                    = buft;
  buffer->buft->host              = true;
  return buffer;
  UNUSED(buft);
}

size_t vk_host_buffer_get_align(v_backend_buffer_type_t buft) {
  return vk_instance.devices[0]->properties.limits.minMemoryMapAlignment;
  UNUSED(buft);
}

size_t vk_host_buffer_get_align() { return vk_instance.devices[0]->properties.limits.minMemoryMapAlignment; }

size_t vk_host_buffer_get_max_size(v_backend_buffer_type_t buft) {
  return vk_instance.devices[0]->suballocation_block_size;

  UNUSED(buft);
}


struct v_backend_buffer_type host{
  0,
  nullptr,
  true
};

v_backend_buffer_type_t vk_host_buffer_type() {
  // Make sure device 0 is initialized
  vk_instance_init();
  v_vk_get_device(0);
  return &host;
}

const char* vk_name(v_backend_t backend) {
  vk_backend_ctx* ctx = (vk_backend_ctx*)backend->context;

  return ctx->name.c_str();
}

void vk_backend_free(v_backend_t backend) {
  vk_backend_ctx* ctx = (vk_backend_ctx*)backend->context;
  VK_LOG_DEBUG("v_backend_vk_free(" << ctx->name << ")");
  v_vk_cleanup(ctx);
  delete ctx;
  delete backend;
}

v_backend_buffer_type_t vk_get_device_buffer(v_backend_t backend) {
  vk_backend_ctx* ctx = (vk_backend_ctx*)backend->context;

  return &ctx->device->buffer_type;
}

void v_backend_vk_set_tensor_async(v_backend_t backend, v_tensor* tensor, const void* data,
                                   size_t offset, size_t size) {
  VK_LOG_DEBUG("v_backend_vk_set_tensor_async(" << size << ")");
  vk_backend_ctx* ctx = (vk_backend_ctx*)backend->context;
  V_ASSERT(
    (tensor->buffer->buft == vk_get_device_buffer(backend) || tensor->buffer->buft ==
      vk_host_buffer_type()) && "unsupported buffer type");

  v_backend_vk_buffer_ctx* buf_ctx = (v_backend_vk_buffer_ctx*)tensor->buffer->context;

  vk_context transfer_ctx;

  if (ctx->transfer_ctx.expired()) {
    // Initialize new transfer context
    transfer_ctx      = vk_create_context(ctx, ctx->transfer_cmd_pool);
    ctx->transfer_ctx = transfer_ctx;
    vk_begin_ctx(ctx->device, transfer_ctx);
  }
  else { transfer_ctx = ctx->transfer_ctx.lock(); }

  vk_buffer buf = buf_ctx->dev_buffer;

  v_vk_buffer_write_async(transfer_ctx, buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, data, size);
}

void v_backend_vk_get_tensor_async(v_backend_t backend, const v_tensor* tensor, void* data,
                                   size_t offset, size_t size) {
  VK_LOG_DEBUG("v_backend_vk_get_tensor_async(" << size << ")");
  vk_backend_ctx* ctx = (vk_backend_ctx*)backend->context;
  V_ASSERT(
    (tensor->buffer->buft == vk_get_device_buffer(backend) || tensor->buffer->buft ==
      vk_host_buffer_type()) && "unsupported buffer type");

  v_backend_vk_buffer_ctx* buf_ctx = (v_backend_vk_buffer_ctx*)tensor->buffer->context;

  vk_context transfer_ctx;

  if (ctx->transfer_ctx.expired()) {
    // Initialize new transfer context
    transfer_ctx      = vk_create_context(ctx, ctx->transfer_cmd_pool);
    ctx->transfer_ctx = transfer_ctx;
    vk_begin_ctx(ctx->device, transfer_ctx);
  }
  else { transfer_ctx = ctx->transfer_ctx.lock(); }

  vk_buffer buf = buf_ctx->dev_buffer;

  vk_buffer_read_async(transfer_ctx, buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, data, size);
}

bool v_backend_buffer_is_vk(v_backend_buffer_t buffer) { return true; }

void vk_free_buffer(v_backend_buffer_t buffer) {
  VK_LOG_MEMORY("v_backend_vk_buffer_free_buffer()");
  v_backend_vk_buffer_ctx* ctx = (v_backend_vk_buffer_ctx*)buffer->context;
  vk_destroy_buffer(ctx->dev_buffer);
  delete ctx;
}

void* vk_device_buffer_get_base(v_backend_buffer_t buffer) {
  return vk_ptr_base;
  UNUSED(buffer);
}

enum v_status vk_buffer_init_tensor(v_backend_buffer_t buffer,
                                    const v_tensor* tensor) {
  V_ASSERT(buffer);
  VK_LOG_DEBUG("v_backend_vk_buffer_init_tensor(" << buffer << " (" << buffer->context << "), " << tensor << ")");
  if (tensor->view_src != nullptr) { V_ASSERT(tensor->view_src->buffer->buft == buffer->buft); }
  return v_STATUS_SUCCESS;
}

void vk_device_buffer_memset_tensor(v_backend_buffer_t buffer, v_tensor* tensor, uint8_t value,
                                    size_t offset, size_t size) {
  VK_LOG_DEBUG(
    "v_backend_vk_buffer_memset_tensor(" << buffer << ", " << tensor << ", " << value << ", " << offset << ", " <<
    size << ")");
  v_backend_vk_buffer_ctx* buf_ctx = (v_backend_vk_buffer_ctx*)buffer->context;
  vk_buffer buf                    = buf_ctx->dev_buffer;
  uint32_t val32                   = (uint32_t)value * 0x01010101;
  vk_buffer_memset(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, val32, size);
}

void vk_device_buffer_set_tensor(v_backend_buffer_t buffer, v_tensor* tensor, const void* data,
                                 size_t offset, size_t size) {
  VK_LOG_DEBUG(
    "v_backend_vk_buffer_set_tensor(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size
    << ")");
  v_backend_vk_buffer_ctx* buf_ctx = (v_backend_vk_buffer_ctx*)buffer->context;
  vk_buffer buf                    = buf_ctx->dev_buffer;
  v_vk_buffer_write(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, data, size);
}

void vk_device_buffer_get_tensor(v_backend_buffer_t buffer,
                                 const v_tensor* tensor,
                                 void* data,
                                 size_t offset,
                                 size_t size) {
  VK_LOG_DEBUG(
    "vk_buffer_get_tensor(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size
    << ")");
  v_backend_vk_buffer_ctx* buf_ctx = (v_backend_vk_buffer_ctx*)buffer->context;
  vk_buffer buf                    = buf_ctx->dev_buffer;
  mmlVKBufferRead(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, data, size);
  VK_LOG_DEBUG(
    "F32 " << ((float*)(buf->ptr))[0] << ", " << ((float*)(buf->ptr))[1] << ", " << data << ", " << offset << ", " <<
    size
    << ")");
}

bool vk_buffer_cpy_tensor(v_backend_buffer_t buffer, const v_tensor* src, v_tensor* dst) {
  if (v_backend_buffer_is_vk(src->buffer)) {
    v_backend_vk_buffer_ctx* src_buf_ctx = (v_backend_vk_buffer_ctx*)src->buffer->context;
    v_backend_vk_buffer_ctx* dst_buf_ctx = (v_backend_vk_buffer_ctx*)dst->buffer->context;

    vk_buffer src_buf = src_buf_ctx->dev_buffer;
    vk_buffer dst_buf = dst_buf_ctx->dev_buffer;

    v_vk_buffer_copy(dst_buf,
                     vk_tensor_offset(dst) + dst->view_offs,
                     src_buf,
                     vk_tensor_offset(src) + src->view_offs,
                     num_bytes(src));

    return true;
  }
  return false;
  UNUSED(buffer);
}

void vk_buffer_clear(v_backend_buffer_t buffer, uint8_t value) {
  v_backend_vk_buffer_ctx* ctx = (v_backend_vk_buffer_ctx*)buffer->context;
  vk_buffer_memset(ctx->dev_buffer, 0, value, buffer->size);
}


// The FA coopmat1 shader assumes 16x16x16 matrix multiply support.
// 128 threads split into four subgroups, each subgroup does 1/4
// of the Bc dimension.

bool v_backend_vk_cpy_tensor_async(v_backend_t backend, const v_tensor* src, v_tensor* dst) {
  VK_LOG_DEBUG("v_backend_vk_cpy_tensor_async()");
  vk_backend_ctx* ctx = (vk_backend_ctx*)backend->context;
  if ((dst->buffer->buft == vk_get_device_buffer(backend) || dst->buffer->buft ==
    vk_host_buffer_type()) && v_backend_buffer_is_vk(src->buffer)) {
    v_backend_vk_buffer_ctx* src_buf_ctx = (v_backend_vk_buffer_ctx*)src->buffer->context;
    v_backend_vk_buffer_ctx* dst_buf_ctx = (v_backend_vk_buffer_ctx*)dst->buffer->context;

    vk_context transfer_ctx;

    if (ctx->transfer_ctx.expired()) {
      // Initialize new transfer context
      transfer_ctx      = vk_create_context(ctx, ctx->transfer_cmd_pool);
      ctx->transfer_ctx = transfer_ctx;
      vk_begin_ctx(ctx->device, transfer_ctx);
    }
    else { transfer_ctx = ctx->transfer_ctx.lock(); }

    vk_buffer src_buf = src_buf_ctx->dev_buffer;
    vk_buffer dst_buf = dst_buf_ctx->dev_buffer;

    v_vk_buffer_copy_async(transfer_ctx,
                           dst_buf,
                           vk_tensor_offset(dst) + dst->view_offs,
                           src_buf,
                           vk_tensor_offset(src) + src->view_offs,
                           num_bytes(src));
    return true;
  }

  return false;
}

void v_backend_vk_synchronize(v_backend_t backend) {
  VK_LOG_DEBUG("v_backend_vk_synchronize()");
  vk_backend_ctx* ctx = (vk_backend_ctx*)backend->context;
  if (ctx->transfer_ctx.expired()) { return; }

  vk_context transfer_ctx = ctx->transfer_ctx.lock();

  vk_ctx_end(transfer_ctx);

  for (auto& cpy : transfer_ctx->in_memcpys) { memcpy(cpy.dst, cpy.src, cpy.n); }

  vk_submit(transfer_ctx, ctx->fence);
  v_vk_wait_for_fence(ctx);

  for (auto& cpy : transfer_ctx->out_memcpys) { memcpy(cpy.dst, cpy.src, cpy.n); }

  ctx->transfer_ctx.reset();
}

bool vk_is_empty(v_tensor* node) {
  return is_empty(node) || node->op == v_OP_NONE || node->op == v_OP_RESHAPE || node->op == v_OP_TRANSPOSE
    || node->op == V_OP_VIEW || node->op == V_OP_PERMUTE;
}

bool v_vk_can_fuse(const struct v_cgraph* cgraph, int node_idx,
                   std::initializer_list<enum v_operation> ops) {
  if (!v_can_fuse(cgraph, node_idx, ops)) { return false; }
  if (ops.size() == 2 && ops.begin()[0] == v_OP_RMS_NORM && ops.begin()[1] == v_OP_MUL) {
    // additional constraints specific to this fusion
    const v_tensor* rms_norm = cgraph->nodes[node_idx];
    const v_tensor* mul      = cgraph->nodes[node_idx + 1];

    V_ASSERT(rms_norm->src[0]->type == v_TYPE_F32);
    V_ASSERT(rms_norm->type == v_TYPE_F32);
    // rms_norm only supports f32
    if (mul->src[0]->type != v_TYPE_F32 ||
      mul->src[1]->type != v_TYPE_F32 ||
      mul->type != v_TYPE_F32) { return false; }
    // if rms_norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] &&
      !v_are_same_shape(mul->src[0], rms_norm)) { return false; }
    // rms_norm shader assumes contiguous rows
    if (!v_is_contiguous_rows(mul->src[0]) || !v_is_contiguous_rows(mul->src[1])) { return false; }
  }
  return true;
}

bool v_vk_can_fuse_topk_moe(vk_backend_ctx* ctx, const struct v_cgraph* cgraph,
                            int node_idx, bool with_norm) {
  if (with_norm) {
    if (node_idx + (int)topk_moe_norm.size() > cgraph->n_nodes) { return false; }
    for (size_t i = 0; i < topk_moe_norm.size(); ++i) { if (cgraph->nodes[node_idx + i]->op != topk_moe_norm[i]) { return false; } }
  }
  else {
    if (node_idx + (int)topk_moe.size() > cgraph->n_nodes) { return false; }
    for (size_t i = 0; i < topk_moe.size(); ++i) { if (cgraph->nodes[node_idx + i]->op != topk_moe[i]) { return false; } }
  }

  const v_tensor* softmax = cgraph->nodes[node_idx + 0];
  const v_tensor* weights = with_norm
                              ? cgraph->nodes[node_idx + 8]
                              : cgraph->nodes[node_idx + 4];

  const float* op_params = (const float*)softmax->op_params;

  float scale    = op_params[0];
  float max_bias = op_params[1];

  if (!v_is_contiguous(softmax->src[0]) || !v_is_contiguous(weights)) { return false; }

  if (scale != 1.0f || max_bias != 0.0f) { return false; }

  // don't fuse when masks or sinks are present
  if (softmax->src[1] || softmax->src[2]) { return false; }

  const int n_expert = softmax->ne[0];
  // n_expert must be a power of 2
  if (!is_pow2(n_expert) || n_expert > (1 << (num_topk_moe_pipelines - 1))) { return false; }

  // Check that the nodes don't have any unexpected uses
  const v_tensor* reshape1 = cgraph->nodes[node_idx + 1];
  const v_tensor* argsort  = cgraph->nodes[node_idx + 2];
  const v_tensor* view     = cgraph->nodes[node_idx + 3];
  const v_tensor* get_rows = cgraph->nodes[node_idx + 4];
  const v_tensor* reshape5 = with_norm
                               ? cgraph->nodes[node_idx + 5]
                               : nullptr;
  const v_tensor* sum_rows = with_norm
                               ? cgraph->nodes[node_idx + 6]
                               : nullptr;
  const v_tensor* div = with_norm
                          ? cgraph->nodes[node_idx + 7]
                          : nullptr;
  const v_tensor* reshape8 = with_norm
                               ? cgraph->nodes[node_idx + 8]
                               : nullptr;

  // softmax is used by reshape and argsort
  if (v_node_get_use_count(cgraph, node_idx) != 2 ||
    reshape1->src[0] != softmax ||
    argsort->src[0] != softmax) { return false; }
  // reshape is used by get_rows
  if (v_node_get_use_count(cgraph, node_idx + 1) != 1 ||
    get_rows->src[0] != reshape1) { return false; }
  // argsort is used by view
  if (v_node_get_use_count(cgraph, node_idx + 2) != 1 ||
    view->src[0] != argsort) { return false; }
  // view is written (via argsort), we can skip checking it

  if (with_norm) {
    // get_rows is used by reshape
    if (v_node_get_use_count(cgraph, node_idx + 4) != 1 ||
      reshape5->src[0] != get_rows) { return false; }

    // reshape is used by sum_rows and div
    if (v_node_get_use_count(cgraph, node_idx + 5) != 2 ||
      sum_rows->src[0] != reshape5 ||
      div->src[0] != reshape5) { return false; }

    // sum_rows is used by div
    if (v_node_get_use_count(cgraph, node_idx + 6) != 1 ||
      div->src[1] != sum_rows) { return false; }

    // div/reshape are written
    if (reshape8->src[0] != div) { return false; }
  }

  if (!ctx->device->subgroup_arithmetic ||
    !ctx->device->subgroup_shuffle ||
    !ctx->device->subgroup_require_full_support ||
    ctx->device->disable_fusion) { return false; }

  return true;
}

uint32_t v_vk_fuse_multi_add(vk_backend_ctx* ctx, const struct v_cgraph* cgraph,
                             int node_idx) {
  const v_tensor* first_node = cgraph->nodes[node_idx];
  if (first_node->op != v_OP_ADD) { return 0; }

  if (!ctx->device->multi_add) { return 0; }

  int32_t num_adds = 1;
  while (node_idx + num_adds < cgraph->n_nodes &&
    cgraph->nodes[node_idx + num_adds]->op == v_OP_ADD &&
    num_adds < MAX_FUSED_ADDS) { num_adds++; }

  // The shader currently requires same shapes (but different strides are allowed),
  // everything f32, and no misalignment
  for (int32_t i = 0; i < num_adds; ++i) {
    const v_tensor* next_node = cgraph->nodes[node_idx + i];
    if (!v_are_same_shape(first_node, next_node->src[0]) ||
      !v_are_same_shape(first_node, next_node->src[1]) ||
      next_node->type != v_TYPE_F32 ||
      next_node->src[0]->type != v_TYPE_F32 ||
      next_node->src[1]->type != v_TYPE_F32 ||
      get_misalign_bytes(ctx, next_node) ||
      get_misalign_bytes(ctx, next_node->src[0]) ||
      get_misalign_bytes(ctx, next_node->src[1])) { num_adds = i; }
  }

  // Verify we can fuse these
  v_operation adds[MAX_FUSED_ADDS];
  for (int32_t i = 0; i < num_adds; ++i) { adds[i] = v_OP_ADD; }

  // decrease num_adds if they can't all be fused
  while (num_adds > 1 && !v_can_fuse(cgraph, node_idx, adds, num_adds)) { num_adds--; }

  // a single add is not "fused", so just return zero
  if (num_adds == 1) { return 0; }
  return num_adds;
}

v_status vk_graph_compute(v_backend_t backend, v_cgraph* cgraph) {
  VK_LOG_DEBUG("v_backend_vk_graph_compute(" << cgraph->n_nodes << " nodes)");
  vk_backend_ctx* ctx = (vk_backend_ctx*)backend->context;

  if (vk_instance.debug_utils_support) {
    vk::DebugUtilsLabelEXT dul = {};
    dul.pLabelName             = "v_backend_vk_graph_compute";
    dul.color                  = std::array<float, 4>{1.0f, 1.0f, 1.0f, 1.0f};
    vk_instance.pfn_vkQueueBeginDebugUtilsLabelEXT(ctx->device->compute_queue.queue,
                                                   reinterpret_cast<VkDebugUtilsLabelEXT*>(&dul));
  }

  ctx->prealloc_size_add_rms_partials        = 0;
  ctx->prealloc_size_add_rms_partials_offset = 0;
  ctx->do_add_rms_partials                   = false;

  uint64_t total_mat_mul_bytes = 0;
  for (int i = 0; i < cgraph->n_nodes; i++) {
    if (!ctx->device->disable_fusion) {
      uint32_t num_adds = v_vk_fuse_multi_add(ctx, cgraph, i);
      if (num_adds) { ctx->num_additional_fused_ops = num_adds - 1; }
      else if (v_vk_can_fuse(cgraph, i, {v_OP_RMS_NORM, v_OP_MUL})) { ctx->num_additional_fused_ops = 1; }
      else if (v_vk_can_fuse_topk_moe(ctx, cgraph, i, true)) { ctx->num_additional_fused_ops = topk_moe_norm.size() - 1; }
      else if (v_vk_can_fuse_topk_moe(ctx, cgraph, i, false)) { ctx->num_additional_fused_ops = topk_moe.size() - 1; }
    }
    vk_build_graph(ctx, cgraph, i, nullptr, 0, true, false, false, false);
    if (cgraph->nodes[i]->op == v_OP_MUL_MAT || cgraph->nodes[i]->op == v_OP_MUL_MAT_ID) { total_mat_mul_bytes += num_bytes(cgraph->nodes[i]->src[0]); }
    else if (cgraph->nodes[i]->op == v_OP_CONV_2D || cgraph->nodes[i]->op == v_OP_CONV_TRANSPOSE_2D) {
      // Return CRSxNPQxsizeof(*) to account as many bytes as mul_mat has in im2col->mul_mat mode.
      auto CRS_size =
        cgraph->nodes[i]->src[0]->ne[0] * cgraph->nodes[i]->src[0]->ne[1] * cgraph->nodes[i]->src[1]->ne[2];
      auto NPQ_size = cgraph->nodes[i]->ne[0] * cgraph->nodes[i]->ne[1] * cgraph->nodes[i]->ne[3];
      total_mat_mul_bytes += NPQ_size * CRS_size * v_type_size(cgraph->nodes[i]->type);
    }
    i += ctx->num_additional_fused_ops;
    ctx->num_additional_fused_ops = 0;
  }
  if (ctx->device->need_compiles) { vk_load_shaders(ctx->device); }
  vk_preallocate_buffers(ctx);
  vk_pipeline_allocate_descriptor_sets(ctx);

  int last_node = cgraph->n_nodes - 1;

  // If the last op in the cgraph isn't backend GPU, the command buffer doesn't get closed properly
  while (last_node > 0 && vk_is_empty(cgraph->nodes[last_node])) { last_node -= 1; }

  // Reserve tensor context space for all nodes
  ctx->tensor_ctxs.resize(cgraph->n_nodes);

  bool first_node_in_batch = true; // true if next node will be first node in a batch
  int submit_node_idx      = 0; // index to first node in a batch

  vk_context compute_ctx;
  if (vk_perf_logger_enabled) {
    // allocate/resize the query pool
    if (ctx->device->num_queries < cgraph->n_nodes + 1) {
      if (ctx->device->query_pool) { ctx->device->device.destroyQueryPool(ctx->device->query_pool); }
      vk::QueryPoolCreateInfo query_create_info;
      query_create_info.queryType  = vk::QueryType::eTimestamp;
      query_create_info.queryCount = cgraph->n_nodes + 100;
      ctx->device->query_pool      = ctx->device->device.createQueryPool(query_create_info);
      ctx->device->num_queries     = query_create_info.queryCount;
    }

    ctx->device->device.resetQueryPool(ctx->device->query_pool, 0, cgraph->n_nodes + 1);

    V_ASSERT(ctx->compute_ctx.expired());
    compute_ctx      = vk_create_context(ctx, ctx->compute_cmd_pool);
    ctx->compute_ctx = compute_ctx;
    vk_begin_ctx(ctx->device, compute_ctx);
    compute_ctx->s->buffer.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->device->query_pool, 0);
  }

  ctx->prealloc_y_last_pipeline_used = nullptr;
  ctx->prealloc_y_last_tensor_used   = nullptr;

  if (ctx->prealloc_size_add_rms_partials) {
    if (ctx->compute_ctx.expired()) {
      compute_ctx      = vk_create_context(ctx, ctx->compute_cmd_pool);
      ctx->compute_ctx = compute_ctx;
      vk_begin_ctx(ctx->device, compute_ctx);
    }
    else { compute_ctx = ctx->compute_ctx.lock(); }
    // initialize partial sums to zero.
    vk_buffer_memset_async(compute_ctx, ctx->prealloc_add_rms_partials, 0, 0, ctx->prealloc_size_add_rms_partials);
    vk_sync_buffers(ctx, compute_ctx);
  }

  // Submit after enough work has accumulated, to overlap CPU cmdbuffer generation with GPU execution.
  // Estimate the amount of matmul work by looking at the weight matrix size, and submit every 100MB
  // (and scaled down based on model size, so smaller models submit earlier).
  // Also submit at least every 100 nodes, in case there are workloads without as much matmul.
  int nodes_per_submit              = 100;
  int submitted_nodes               = 0;
  int submit_count                  = 0;
  uint64_t mul_mat_bytes            = 0;
  uint64_t mul_mat_bytes_per_submit = std::min(uint64_t(100 * 1000 * 1000), total_mat_mul_bytes / 40u);
  for (int i = 0; i < cgraph->n_nodes; i++) {
    if (first_node_in_batch) { submit_node_idx = i; }

    if (cgraph->nodes[i]->op == v_OP_MUL_MAT || cgraph->nodes[i]->op == v_OP_MUL_MAT_ID) { mul_mat_bytes += num_bytes(cgraph->nodes[i]->src[0]); }

    if (!ctx->device->disable_fusion) {
      uint32_t num_adds = v_vk_fuse_multi_add(ctx, cgraph, i);
      if (num_adds) { ctx->num_additional_fused_ops = num_adds - 1; }
      else if (v_vk_can_fuse(cgraph, i, {v_OP_RMS_NORM, v_OP_MUL})) { ctx->num_additional_fused_ops = 1; }
      else if (v_vk_can_fuse_topk_moe(ctx, cgraph, i, true)) { ctx->num_additional_fused_ops = topk_moe_norm.size() - 1; }
      else if (v_vk_can_fuse_topk_moe(ctx, cgraph, i, false)) { ctx->num_additional_fused_ops = topk_moe.size() - 1; }
    }

    // Signal the almost_ready fence when the graph is mostly complete (< 20% remaining)
    bool almost_ready = (cgraph->n_nodes - i) < cgraph->n_nodes / 5;
    bool submit       = (submitted_nodes >= nodes_per_submit) ||
      (mul_mat_bytes >= mul_mat_bytes_per_submit) ||
      (i + ctx->num_additional_fused_ops >= last_node) ||
      (almost_ready && !ctx->almost_ready_fence_pending);

    bool enqueued = vk_build_graph(ctx,
                                   cgraph,
                                   i,
                                   cgraph->nodes[submit_node_idx],
                                   submit_node_idx,
                                   false,
                                   i + ctx->num_additional_fused_ops >= last_node,
                                   almost_ready,
                                   submit);

    if (vk_perf_logger_enabled) {
      if (ctx->compute_ctx.expired()) {
        compute_ctx      = vk_create_context(ctx, ctx->compute_cmd_pool);
        ctx->compute_ctx = compute_ctx;
        vk_begin_ctx(ctx->device, compute_ctx);
      }
      else { compute_ctx = ctx->compute_ctx.lock(); }
      // If there are fused ops, just write out timestamps for all nodes to keep the accounting simple
      for (int j = 0; j < ctx->num_additional_fused_ops + 1; ++j) {
        compute_ctx->s->buffer.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands,
                                              ctx->device->query_pool,
                                              i + j + 1);
      }
    }

    if (enqueued) {
      ++submitted_nodes;

      #ifndef v_VULKAN_CHECK_RESULTS
      if (first_node_in_batch) { first_node_in_batch = false; }
      #endif
    }

    if (submit && enqueued) {
      first_node_in_batch = true;
      submitted_nodes     = 0;
      mul_mat_bytes       = 0;
      if (submit_count < 3) { mul_mat_bytes_per_submit *= 2; }
      submit_count++;
    }
    i += ctx->num_additional_fused_ops;
    ctx->num_additional_fused_ops = 0;
  }

  if (vk_perf_logger_enabled) {
    // End the command buffer and submit/wait
    V_ASSERT(!ctx->compute_ctx.expired());
    compute_ctx = ctx->compute_ctx.lock();
    vk_ctx_end(compute_ctx);

    vk_submit(compute_ctx, ctx->device->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->device->fence }, true, UINT64_MAX),
             "v_VULKAN_PERF waitForFences");
    ctx->device->device.resetFences({ctx->device->fence});

    // Get the results and pass them to the logger
    std::vector<uint64_t> timestamps(cgraph->n_nodes + 1);
    VK_CHECK(
      ctx->device->device.getQueryPoolResults(ctx->device->query_pool, 0, cgraph->n_nodes + 1, (cgraph->n_nodes + 1)*
        sizeof(uint64_t), timestamps.data(), sizeof(uint64_t), vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::
        eWait),
      "get timestamp results");
    for (int i = 0; i < cgraph->n_nodes; i++) {
      if (!vk_is_empty(cgraph->nodes[i])) {
        ctx->device->perf_logger->log_timing(cgraph->nodes[i],
                                             uint64_t(
                                               (timestamps[i + 1] - timestamps[i]) * ctx->device->properties.limits.
                                                                                          timestampPeriod));
      }
    }

    ctx->device->perf_logger->print_timings();
  }

  vk_graph_cleanup(ctx);

  return v_STATUS_SUCCESS;

  UNUSED(backend);
}

// Sort the graph for improved parallelism.
void vk_graph_optimize(v_backend_t backend, struct v_cgraph* graph) {
  VK_LOG_DEBUG("v_vk_graph_optimize(" << graph->n_nodes << " nodes)");
  vk_backend_ctx* ctx = (vk_backend_ctx*)backend->context;
  if (ctx->device->disable_graph_optimize) { return; }
  auto const& is_empty = [](v_tensor* node) -> bool {
    return node->op == v_OP_NONE || node->op == v_OP_RESHAPE || node->op == v_OP_TRANSPOSE || node->op ==
      V_OP_VIEW || node->op == V_OP_PERMUTE;
  };

  auto const& is_src_of = [](const v_tensor* dst, const v_tensor* src) -> bool {
    for (uint32_t s = 0; s < v_MAX_SRC; ++s) { if (dst->src[s] == src) { return true; } }
    // implicit dependency if they view the same tensor
    const v_tensor* dst2 = dst->view_src
                             ? dst->view_src
                             : dst;
    const v_tensor* src2 = src->view_src
                             ? src->view_src
                             : src;
    if (dst2 == src2) { return true; }
    return false;
  };

  // This function tries to reorder the graph to allow nodes to run in parallel.
  // This helps with small batches, but for large batches its a slowdown, probably
  // due to cache contention. So only reorder if the majority of nodes have few rows.
  int num_small_nodes   = 0;
  int num_counted_nodes = 0;
  for (int i = 0; i < graph->n_nodes; ++i) {
    if (!is_empty(graph->nodes[i]) &&
      graph->nodes[i]->op != v_OP_SET_ROWS) {
      if (v_nrows(graph->nodes[i]) <= 8) { num_small_nodes++; }
      num_counted_nodes++;
    }
  }
  if (num_small_nodes < num_counted_nodes / 2) { return; }

  std::vector<v_tensor*> new_order;
  std::vector<bool> used(graph->n_nodes, false);
  int first_unused = 0;
  while (first_unused < graph->n_nodes) {
    std::vector<int> current_set;

    // Avoid reordering topk_moe_norm
    if (first_unused + (int)topk_moe_norm.size() <= graph->n_nodes) {
      bool is_topk_moe_norm = true;
      for (size_t j = 0; j < topk_moe_norm.size(); ++j) { if (graph->nodes[first_unused + j]->op != topk_moe_norm[j] || used[first_unused + j]) { is_topk_moe_norm = false; } }
      if (is_topk_moe_norm) {
        for (size_t j = 0; j < topk_moe_norm.size(); ++j) {
          new_order.push_back(graph->nodes[first_unused + j]);
          used[first_unused + j] = true;
        }
        while (first_unused < graph->n_nodes && used[first_unused]) { first_unused++; }
        continue;
      }
    }
    // First, grab the next unused node.
    current_set.push_back(first_unused);

    // Loop through the next N nodes. Grab any that don't depend on other nodes that
    // haven't already been run. Nodes that have already been run have used[i] set
    // to true. Allow nodes that depend on the previous node if it's a fusion pattern
    // that we support (e.g. RMS_NORM + MUL).
    // This first pass only grabs "real" (non-view nodes). Second pass grabs view nodes.
    // The goal is to not interleave real and view nodes in a way that breaks fusion.
    const int NUM_TO_CHECK = 20;
    for (int j = first_unused + 1; j < std::min(first_unused + NUM_TO_CHECK, graph->n_nodes); ++j) {
      if (used[j]) { continue; }
      if (is_empty(graph->nodes[j])) { continue; }
      bool ok = true;
      for (int c = first_unused; c < j; ++c) {
        if (!used[c] &&
          is_src_of(graph->nodes[j], graph->nodes[c]) &&
          !(j == c + 1 && c == current_set.back() && graph->nodes[c]->op == v_OP_RMS_NORM && graph->nodes[j]->op ==
            v_OP_MUL)) {
          ok = false;
          break;
        }
      }
      if (ok) { current_set.push_back(j); }
    }
    // Second pass grabs view nodes.
    // Skip this if it would break a fusion optimization (don't split up add->rms_norm or add->add).
    if (graph->nodes[current_set.back()]->op != v_OP_ADD) {
      for (int j = first_unused + 1; j < std::min(first_unused + NUM_TO_CHECK, graph->n_nodes); ++j) {
        if (used[j]) { continue; }
        if (!is_empty(graph->nodes[j])) { continue; }
        bool ok = true;
        for (int c = first_unused; c < j; ++c) {
          bool c_in_current_set = std::find(current_set.begin(), current_set.end(), c) != current_set.end();
          // skip views whose srcs haven't been processed.
          if (!used[c] &&
            is_src_of(graph->nodes[j], graph->nodes[c]) &&
            !c_in_current_set) {
            ok = false;
            break;
          }
        }
        if (ok) { current_set.push_back(j); }
      }
    }

    // Push the current set into new_order
    for (auto c : current_set) {
      new_order.push_back(graph->nodes[c]);
      used[c] = true;
    }
    while (first_unused < graph->n_nodes && used[first_unused]) { first_unused++; }
  }
  // Replace the graph with the new order.
  for (int i = 0; i < graph->n_nodes; ++i) { graph->nodes[i] = new_order[i]; }
}


v_backend_t backend_vk_init(size_t dev_num) {
  VK_LOG_DEBUG("v_backend_vk_init(" << dev_num << ")");
  vk_backend_ctx* ctx = new vk_backend_ctx;
  vk_init(ctx, dev_num);
  v_backend_t vk_backend = new v_backend;
  vk_backend->device     = v_backend_vk_reg_get_device(dev_num),
    vk_backend->context  = ctx;
  return vk_backend;
}

int vk_get_device_count() { return v_vk_get_device_count(); }

void vk_get_device_description(int device, char* description, size_t description_size) {
  V_ASSERT(device < (int) vk_instance.device_indices.size());
  int dev_idx = vk_instance.device_indices[device];
  v_vk_get_device_description(dev_idx, description, description_size);
}

void vk_get_device_memory(int device, size_t* free, size_t* total) {
  V_ASSERT(device < (int) vk_instance.device_indices.size());
  V_ASSERT(device < (int) vk_instance.device_supports_membudget.size());

  vk::PhysicalDevice vkdev = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device]];
  vk::PhysicalDeviceMemoryBudgetPropertiesEXT budgetprops;
  vk::PhysicalDeviceMemoryProperties2 memprops = {};
  bool membudget_supported                     = vk_instance.device_supports_membudget[device];

  if (membudget_supported) { memprops.pNext = &budgetprops; }
  vkdev.getMemoryProperties2(&memprops);

  for (uint32_t i = 0; i < memprops.memoryProperties.memoryHeapCount; ++i) {
    const vk::MemoryHeap& heap = memprops.memoryProperties.memoryHeaps[i];

    if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
      *total = heap.size;

      if (membudget_supported && i < budgetprops.heapUsage.size()) { *free = budgetprops.heapBudget[i] - budgetprops.heapUsage[i]; }
      else { *free = heap.size; }
      break;
    }
  }
}

vk::PhysicalDeviceType v_backend_vk_get_device_type(int device_idx) {
  V_ASSERT(device_idx >= 0 && device_idx < (int) vk_instance.device_indices.size());

  vk::PhysicalDevice device = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device_idx]];

  vk::PhysicalDeviceProperties2 props = {};
  device.getProperties2(&props);

  return props.properties.deviceType;
}

std::string v_backend_vk_get_device_pci_id(int device_idx) {
  V_ASSERT(device_idx >= 0 && device_idx < (int) vk_instance.device_indices.size());

  vk::PhysicalDevice device = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device_idx]];

  const std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

  bool ext_support = false;

  for (const auto& properties : ext_props) {
    if (strcmp("VK_EXT_pci_bus_info", properties.extensionName) == 0) {
      ext_support = true;
      break;
    }
  }

  if (!ext_support) { return ""; }

  vk::PhysicalDeviceProperties2 props                    = {};
  vk::PhysicalDevicePCIBusInfoPropertiesEXT pci_bus_info = {};

  props.pNext = &pci_bus_info;

  device.getProperties2(&props);

  const uint32_t pci_domain  = pci_bus_info.pciDomain;
  const uint32_t pci_bus     = pci_bus_info.pciBus;
  const uint32_t pci_device  = pci_bus_info.pciDevice;
  const uint8_t pci_function = (uint8_t)pci_bus_info.pciFunction;
  // pci function is between 0 and 7, prevent printf overflow warning

  char pci_bus_id[16] = {};
  snprintf(pci_bus_id, sizeof(pci_bus_id), "%04x:%02x:%02x.%x", pci_domain, pci_bus, pci_device, pci_function);
  return std::string(pci_bus_id);
}


bool vk_device_supports_buft(struct vk_device_ctx* dev, v_backend_buffer_type_t buft) {
  vk_device_ctx* ctx               = (vk_device_ctx*)dev;
  vk_buffer_type_context* buft_ctx = (vk_buffer_type_context*)buft->context;
  return buft_ctx->device->idx == ctx->device;
}


v_backend_buffer_t vk_device_buffer_alloc(v_backend_buffer_type_t buft, size_t size) {
  VK_LOG_MEMORY("v_backend_vk_buffer_type_alloc_buffer(" << size << ")");
  vk_buffer_type_context* ctx = (vk_buffer_type_context*)buft->context;
  vk_buffer dev_buffer        = nullptr;
  try { dev_buffer = v_vk_create_buffer_device(ctx->device, size); }
  catch (const vk::SystemError& e) { return nullptr; }
  v_backend_vk_buffer_ctx* bufctx = new v_backend_vk_buffer_ctx(
    ctx->device,
    std::move(dev_buffer),
    ctx->name);

  return v_backend_buffer_init(buft, bufctx, size);
}


struct vk_device_ctx* v_backend_vk_reg_get_device(size_t device) {
  std::vector<struct vk_device_ctx*> devices;
  bool initialized = false;
  {
    std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (!initialized) {
      for (int i = 0; i < v_vk_get_device_count(); i++) {
        vk_device_ctx* ctx = new vk_device_ctx;
        char desc[256];
        vk_get_device_description(i, desc, sizeof(desc));
        ctx->device            = i;
        ctx->name              = v_VK_NAME + std::to_string(i);
        ctx->description       = desc;
        ctx->is_integrated_gpu = v_backend_vk_get_device_type(i) == vk::PhysicalDeviceType::eIntegratedGpu;
        ctx->pci_bus_id        = v_backend_vk_get_device_pci_id(i);
        devices.push_back(new vk_device_ctx{
        });
      }
      initialized = true;
    }
  }

  V_ASSERT(device < devices.size());
  return devices[device];
}


#ifndef MYPROJECT_MML_VK_TEST_H
#define MYPROJECT_MML_VK_TEST_H
#define v_VULKAN_CHECK_RESULT
#include "v_vk.h"
#include "vk_device.h"
#include "vk_buffer.h"
#ifdef v_VULKAN_RUN_TESTS
static void v_vk_print_matrix_area(const void* data, v_data_type type, int ne0, int ne1, int i0, int i1, int i2) {
  if (type != v_TYPE_F32 && type != v_TYPE_F16) { return; }
  i0 = std::max(i0, 5);
  i1 = std::max(i1, 5);
  i2 = std::max(i2, 0);
  fprintf(stderr, "         ");
  for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) { fprintf(stderr, "%7d ", idx1); }
  fprintf(stderr, "\n");
  for (int idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
    fprintf(stderr, "%7d: ", idx0);
    for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
      if (idx0 >= 0 && idx0 < ne0 && idx1 >= 0 && idx1 < ne1) {
        float val;
        if (type == v_TYPE_F32) { val = *((const float*)data + i2 * ne1 * ne0 + idx1 * ne0 + idx0); }
        else if (type == v_TYPE_F16) { val = v_fp16_to_fp32(*((const v_fp16_t*)data + i2 * ne1 * ne0 + idx1 * ne0 + idx0)); }
        else { v_ABORT("fatal error"); }
        fprintf(stderr, "% 7.2f ", val);
      }
      else { fprintf(stderr, "        "); }
    }
    fprintf(stderr, "\n");
  }
}

template <typename X_TYPE, typename Y_TYPE>
static void v_vk_test_matmul(vk_backend_ctx* ctx, size_t m, size_t n, size_t k, size_t batch, size_t num_it,
                             int split_k, int shader_size) {
  VK_LOG_DEBUG(
    "v_vk_test_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << num_it << ", " << split_k << ", "
    << shader_size << ")");
  const size_t x_ne = m * k * batch;
  const size_t y_ne = k * n * batch;
  const size_t d_ne = m * n * batch;

  vk_pipeline p;
  std::string shname;
  if (shader_size == 0) {
    if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f32->a_s;
      shname = "F32_ALIGNED_S";
    }
    else if (std::is_same<float, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f32_f16->a_s;
      shname = "F32_F16_ALIGNED_S";
    }
    else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f16_f32.f32acc->a_s;
      shname = "F16_F32_ALIGNED_S";
    }
    else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f16.f32acc->a_s;
      shname = "F16_ALIGNED_S";
    }
    else { v_ABORT("fatal error"); }
  }
  else if (shader_size == 1) {
    if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f32->a_m;
      shname = "F32_ALIGNED_M";
    }
    else if (std::is_same<float, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f32_f16->a_m;
      shname = "F32_F16_ALIGNED_M";
    }
    else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f16_f32.f32acc->a_m;
      shname = "F16_F32_ALIGNED_M";
    }
    else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f16.f32acc->a_m;
      shname = "F16_ALIGNED_M";
    }
    else { v_ABORT("fatal error"); }
  }
  else if (shader_size == 2) {
    if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f32->a_l;
      shname = "F32_ALIGNED_L";
    }
    else if (std::is_same<float, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f32_f16->a_l;
      shname = "F32_F16_ALIGNED_L";
    }
    else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f16_f32.f32acc->a_l;
      shname = "F16_F32_ALIGNED_L";
    }
    else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f16.f32acc->a_l;
      shname = "F16_ALIGNED_L";
    }
    else { v_ABORT("fatal error"); }
  }
  else { V_ASSERT(0); }

  const size_t kpad = mmlVKAlignSize(k, p->align);

  if (k != kpad) {
    if (shader_size == 0) {
      if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f32->s;
        shname = "F32_S";
      }
      else if (std::is_same<float, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f32_f16->s;
        shname = "F32_F16_S";
      }
      else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f16_f32.f32acc->s;
        shname = "F16_F32_S";
      }
      else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f16.f32acc->s;
        shname = "F16_S";
      }
    }
    else if (shader_size == 1) {
      if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f32->m;
        shname = "F32_M";
      }
      else if (std::is_same<float, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f32_f16->m;
        shname = "F32_F16_M";
      }
      else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f16_f32.f32acc->m;
        shname = "F16_F32_M";
      }
      else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f16.f32acc->m;
        shname = "F16_M";
      }
    }
    else if (shader_size == 2) {
      if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f32->l;
        shname = "F32_L";
      }
      else if (std::is_same<float, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f32_f16->l;
        shname = "F32_F16_L";
      }
      else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f16_f32.f32acc->l;
        shname = "F16_F32_L";
      }
      else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f16.f32acc->l;
        shname = "F16_L";
      }
    }
  }

  v_pipeline_request_descriptor_sets(ctx, p, num_it);
  if (split_k > 1) {
    v_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_split_k_reduce, num_it);

    if (ctx->prealloc_split_k == nullptr || ctx->prealloc_split_k->size < sizeof(float) * d_ne * split_k) {
      // Resize buffer
      if (ctx->prealloc_split_k != nullptr) { vk_destroy_buffer(ctx->prealloc_split_k); }
      ctx->prealloc_split_k = vk_create_buffer_check(ctx->device,
                                                     sizeof(float) * d_ne * split_k,
                                                     {vk::MemoryPropertyFlagBits::eDeviceLocal});
    }
  }

  if (ctx->device->need_compiles) { vk_load_shaders(ctx->device); }

  vk_pipeline_allocate_descriptor_sets(ctx);

  vk_buffer d_X = vk_create_buffer_check(ctx->device,
                                         sizeof(X_TYPE) * x_ne,
                                         {vk::MemoryPropertyFlagBits::eDeviceLocal});
  vk_buffer d_Y = vk_create_buffer_check(ctx->device,
                                         sizeof(Y_TYPE) * y_ne,
                                         {vk::MemoryPropertyFlagBits::eDeviceLocal});
  vk_buffer d_D = vk_create_buffer_check(ctx->device,
                                         sizeof(float) * d_ne,
                                         {vk::MemoryPropertyFlagBits::eDeviceLocal});

  X_TYPE* x = (X_TYPE*)malloc(sizeof(X_TYPE) * x_ne);
  Y_TYPE* y = (Y_TYPE*)malloc(sizeof(Y_TYPE) * y_ne);
  float* d  = (float*)malloc(sizeof(float) * d_ne);

  for (size_t i = 0; i < x_ne; i++) {
    if (std::is_same<float, X_TYPE>()) {
      x[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
      // x[i] = 1.0f;
      // x[i] = i + 1;
      // x[i] = (i % k == i / k) ? 1.0f : 0.0f;
    }
    else if (std::is_same<v_fp16_t, X_TYPE>()) {
      x[i] = v_fp32_to_fp16((rand() / (float)RAND_MAX) * 2.0f - 1.0f);
      // x[i] = v_fp32_to_fp16(1.0f);
      // x[i] = v_fp32_to_fp16(i + 1);
      // x[i] = v_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
    }
    else { v_ABORT("fatal error"); }
  }
  for (size_t i = 0; i < y_ne; i++) {
    if (std::is_same<float, Y_TYPE>()) {
      y[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
      // y[i] = (i % k == i / k) ? 1.0f : 0.0f;
      // y[i] = i + 1;
    }
    else if (std::is_same<v_fp16_t, Y_TYPE>()) {
      y[i] = v_fp32_to_fp16((rand() / (float)RAND_MAX) * 2.0f - 1.0f);
      // y[i] = v_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
      // y[i] = v_fp32_to_fp16(i + 1);
    }
    else { v_ABORT("fatal error"); }
  }

  v_vk_buffer_write(d_X, 0, x, sizeof(X_TYPE) * k * m * batch);
  v_vk_buffer_write(d_Y, 0, y, sizeof(Y_TYPE) * k * n * batch);

  vk_context subctx = vk_create_context(ctx, ctx->compute_cmd_pool);
  vk_begin_ctx(ctx->device, subctx);
  for (size_t i = 0; i < num_it; i++) {
    v_vk_matmul(
      ctx,
      subctx,
      p,
      v_vk_subbuffer(ctx, d_X),
      v_vk_subbuffer(ctx, d_Y),
      v_vk_subbuffer(ctx, d_D),
      v_vk_subbuffer(ctx, ctx->prealloc_split_k),
      m,
      n,
      k,
      k,
      k,
      m,
      k * m,
      k * n,
      m * n,
      split_k,
      batch,
      batch,
      batch,
      1,
      1,
      n
    );
  }
  vk_ctx_end(subctx);

  //auto begin = std::chrono::high_resolution_clock::now();
  vk_submit(subctx, ctx->fence);
  VK_CHECK(ctx->device->device.waitForFences({ctx->fence}, true, UINT64_MAX), "v_vk_test_matmul waitForFences");
  ctx->device->device.resetFences({ctx->fence});
  mmlVkQueueCommandPoolsCleanUp(ctx->device);

  //auto end = std::chrono::high_resolution_clock::now();
  //double time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0;

  // copy dst to host
  mmlVKBufferRead(d_D, 0, d, sizeof(float) * d_ne);

  float* d_chk = (float*)malloc(sizeof(float) * d_ne);

  //InitParmaeters iparams = {
  //  /*.mem_size   =*/ 1024 * 1024 * 1024,
  //  /*.mem_buffer =*/ NULL,
  //  /*.no_alloc   =*/ true,
  //};
  //
  //Context* v_ctx = mmlCtxInit(iparams);
  //
  //MmlDataType src0_type;
  //MmlDataType src1_type;
  //
  //if (std::is_same<float, X_TYPE>())
  //{
  //  src0_type = v_TYPE_F32;
  //}
  //else if (std::is_same<v_fp16_t, X_TYPE>())
  //{
  //  src0_type = v_TYPE_F16;
  //}
  //else
  //{
  //  v_ABORT("fatal error");
  //}
  //if (std::is_same<float, Y_TYPE>())
  //{
  //  src1_type = v_TYPE_F32;
  //}
  //else if (std::is_same<v_fp16_t, Y_TYPE>())
  //{
  //  src1_type = v_TYPE_F16;
  //}
  //else
  //{
  //  v_ABORT("fatal error");
  //}
  //
  //MmlTensor* src0_ggml = mmlNewTensor3D(v_ctx, src0_type, k, m, batch);
  //MmlTensor* src1_ggml = mmlNewTensor3D(v_ctx, src1_type, k, n, batch);
  //MmlTensor* tensor_ggml = mmlMatrixMul(v_ctx, src0_ggml, src1_ggml);
  //
  //src0_ggml->data = x;
  //src1_ggml->data = y;
  //tensor_ggml->data = d_chk;
  //
  //MmlComputeGraph* cgraph = mmlNewGraph(v_ctx);
  //mmlBuildFowardExpand(cgraph, tensor_ggml);
  //
  ////v_backend_vk_graph_compute(&, cgraph);
  //
  //mmlFree(v_ctx);
  //
  //double avg_err = 0.0;
  //int first_err_n = -1;
  //int first_err_m = -1;
  //int first_err_b = -1;
  //
  //for (size_t i = 0; i < m * n * batch; i++)
  //{
  //  double err = std::fabs(d[i] - d_chk[i]);
  //  avg_err += err;
  //
  //  if ((err > 0.05f || std::isnan(err)) && first_err_n == -1)
  //  {
  //    first_err_b = i / (m * n);
  //    first_err_n = (i % (m * n)) / m;
  //    first_err_m = (i % (m * n)) % m;
  //  }
  //}
  //
  //avg_err /= m * n;
  //
  ////double tflops = 2.0 * m * n * k * batch * num_it / (time / 1000.0) / (1000.0 * 1000.0 * 1000.0 * 1000.0);
  //
  ////std::cerr << "TEST " << shname << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" <<
  ////  split_k << " matmul " << time / num_it << "ms " << tflops << " TFLOPS avg_err=" << avg_err << std::endl;

  //if (avg_err > 0.1 || std::isnan(avg_err))
  //{
  //  std::cerr << "m = " << first_err_m << " n = " << first_err_n << " b = " << first_err_b << std::endl;
  //  std::cerr << "Actual result: " << std::endl << std::endl;
  //  v_vk_print_matrix_area(d, v_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
  //  std::cerr << "Expected result: " << std::endl << std::endl;
  //  v_vk_print_matrix_area(d_chk, v_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
  //
  //  if (split_k > 1)
  //  {
  //    float* split_k_buf = (float*)malloc(sizeof(float) * d_ne * split_k);
  //    mmlVKBufferRead(ctx->prealloc_split_k, 0, split_k_buf, sizeof(float) * d_ne * split_k);
  //
  //    std::cerr << "d_buf0: " << std::endl << std::endl;
  //    v_vk_print_matrix_area(split_k_buf, v_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
  //
  //    std::cerr << "d_buf1: " << std::endl << std::endl;
  //    v_vk_print_matrix_area(split_k_buf + d_ne, v_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
  //
  //    std::cerr << "d_buf2: " << std::endl << std::endl;
  //    v_vk_print_matrix_area(split_k_buf + 2 * d_ne, v_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
  //
  //    std::cerr << "d_buf3: " << std::endl << std::endl;
  //    v_vk_print_matrix_area(split_k_buf + 3 * d_ne, v_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
  //
  //    free(split_k_buf);
  //  }
  //}
  //
  //free(d_chk);

  v_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);
  v_vk_command_pool_cleanup(ctx->device, ctx->transfer_cmd_pool);
  vk_destroy_buffer(d_X);
  vk_destroy_buffer(d_Y);
  vk_destroy_buffer(d_D);

  free(x);
  free(y);
  free(d);
}

static void v_vk_print_tensor_area(const v_tensor* tensor, int i0, int i1, int i2, int i3) {
  if (tensor->type != v_TYPE_F32 && tensor->type != v_TYPE_F16) { return; }
  i0 = std::max(i0, 5);
  i1 = std::max(i1, 5);
  i2 = std::max(i2, 0);
  i3 = std::max(i3, 0);
  fprintf(stderr, "         ");
  for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) { fprintf(stderr, "%7d ", idx1); }
  fprintf(stderr, "\n");
  for (int idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
    fprintf(stderr, "%7d: ", idx0);
    for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
      if (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3
        >= 0 && i3 < tensor->ne[3]) {
        float val;
        if (tensor->type == v_TYPE_F32) {
          val = *(float*)((char*)tensor->data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + idx1 * tensor->nb[1] + idx0 *
            tensor->nb[0]);
        }
        else if (tensor->type == v_TYPE_F16) {
          val = v_fp16_to_fp32(
            *(v_fp16_t*)((char*)tensor->data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + idx1 * tensor->nb[1] + idx0
              * tensor->nb[0]));
        }
        else { v_ABORT("fatal error"); }
        fprintf(stderr, "% 7.2f ", val);
      }
      else { fprintf(stderr, "        "); }
    }
    fprintf(stderr, "\n");
  }
}

static void v_vk_quantize_data(const float* from, void* to, size_t ne, v_data_type quant) { v_quantize_chunk(quant, from, to, 0, 1, ne, nullptr); }

static void v_vk_dequantize_data(const void* from, float* to, size_t ne, v_data_type quant) {
  if (quant == v_TYPE_F32) {
    memcpy(to, from, sizeof(float) * ne);
    return;
  }

  const auto* tt = v_get_type_traits(quant);

  v_to_float_t dequant_fn = tt->to_float;

  dequant_fn(from, to, ne);
}

static void v_vk_test_dequant(vk_backend_ctx* ctx, size_t ne, v_data_type quant) {
  VK_LOG_DEBUG("v_vk_test_dequant(" << ne << ")");
  const size_t x_sz     = sizeof(float) * ne;
  const size_t x_sz_f16 = sizeof(v_fp16_t) * ne;
  const size_t qx_sz    = ne * v_type_size(quant) / blockSize(quant);
  float* x              = (float*)malloc(x_sz);
  void* qx              = malloc(qx_sz);
  vk_buffer qx_buf      = vk_create_buffer_check(ctx->device, qx_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
  vk_buffer x_buf       = vk_create_buffer_check(ctx->device, x_sz_f16, {vk::MemoryPropertyFlagBits::eDeviceLocal});
  float* x_ref          = (float*)malloc(x_sz);
  v_fp16_t* x_chk       = (v_fp16_t*)malloc(x_sz_f16);

  for (size_t i = 0; i < ne; i++) { x[i] = rand() / (float)RAND_MAX; }

  vk_pipeline p = v_vk_get_to_fp16(ctx, quant);

  v_vk_quantize_data(x, qx, ne, quant);
  v_vk_dequantize_data(qx, x_ref, ne, quant);

  v_pipeline_request_descriptor_sets(ctx, p, 1);

  if (ctx->device->need_compiles) { vk_load_shaders(ctx->device); }

  vk_pipeline_allocate_descriptor_sets(ctx);

  v_vk_buffer_write(qx_buf, 0, qx, qx_sz);

  vk_context subctx = vk_create_context(ctx, ctx->compute_cmd_pool);
  //v_vk_ctx_begin(ctx->device, subctx);
  const std::vector<uint32_t> pc = {1, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne};
  v_vk_dispatch_pipeline(ctx,
                         subctx,
                         p,
                         {vk_sub_buffer{qx_buf, 0, qx_sz}, vk_sub_buffer{x_buf, 0, x_sz_f16}},
                         pc,
                         {(uint32_t)ne, 1, 1});
  //v_vk_ctx_end(subctx);
  //
  //auto begin = std::chrono::high_resolution_clock::now();
  //
  //v_vk_submit(subctx, ctx->fence);
  //VK_CHECK(ctx->device->device.waitForFences({ctx->fence}, true, UINT64_MAX), "v_vk_test_dequant waitForFences");
  //ctx->device->device.resetFences({ctx->fence});
  //v_vk_queue_command_pools_cleanup(ctx->device);
  //
  //auto end = std::chrono::high_resolution_clock::now();
  //
  //double ms_dequant = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0;
  //mmlVKBufferRead(x_buf, 0, x_chk, x_sz_f16);
  //
  //int first_err = -1;
  //
  //double avg_err = 0.0;
  //for (size_t i = 0; i < ne; i++)
  //{
  //  double error = std::fabs(x_ref[i] - v_fp16_to_fp32(x_chk[i]));
  //  avg_err += error;
  //
  //  if (first_err < 0 && error > 0.05)
  //  {
  //    first_err = i;
  //  }
  //}
  //
  //avg_err /= ne;
  //
  //std::cerr << "TEST DEQUANT " << v_type_name(quant) << " time=" << ms_dequant << "ms avg_err=" << avg_err <<
  //  std::endl;
  //
  //if (avg_err > 0.1)
  //{
  //  std::cerr << "first_error = " << first_err << std::endl;
  //  std::cerr << "Actual result: " << std::endl << std::endl;
  //  for (int i = std::max(0, first_err - 5); i < std::min((int)ne, first_err + 5); i++)
  //  {
  //    std::cerr << v_fp16_to_fp32(x_chk[i]) << ", ";
  //  }
  //  std::cerr << std::endl << "Expected result: " << std::endl << std::endl;
  //  for (int i = std::max(0, first_err - 5); i < std::min((int)ne, first_err + 5); i++)
  //  {
  //    std::cerr << x_ref[i] << ", ";
  //  }
  //  std::cerr << std::endl;
  //}
  //
  //mmlVKDestroyBuffer(x_buf);
  //mmlVKDestroyBuffer(qx_buf);
  //
  //free(x);
  //free(qx);
  //free(x_ref);
  //free(x_chk);
}

// This does not work without ggml q8_1 quantization support
//
// typedef uint16_t v_half;
// typedef uint32_t v_half2;
//
// #define QK8_1 32
// typedef struct {
//     union {
//         struct {
//             v_half d; // delta
//             v_half s; // d * sum(qs[i])
//         } v_COMMON_AGGR_S;
//         v_half2 ds;
//     } v_COMMON_AGGR_U;
//     int8_t qs[QK8_1]; // quants
// } block_q8_1;
//
// static void v_vk_test_quantize(v_backend_vk_context * ctx, size_t ne, v_type quant) {
//     VK_LOG_DEBUG("v_vk_test_quantize(" << ne << ")");
//     MML_ASSERT(quant == v_TYPE_Q8_1);
//
//     const size_t x_sz = sizeof(float) * ne;
//     const size_t qx_sz = ne * v_type_size(quant)/v_blck_size(quant);
//     float * x = (float *) malloc(x_sz);
//     block_q8_1 * qx     = (block_q8_1 *)malloc(qx_sz);
//     block_q8_1 * qx_res = (block_q8_1 *)malloc(qx_sz);
//     VKBuffer x_buf = mmlVKCreateBufferCheck(ctx->device, x_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
//     VKBuffer qx_buf = mmlVKCreateBufferCheck(ctx->device, qx_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
//
//     for (size_t i = 0; i < ne; i++) {
//         x[i] = rand() / (float)RAND_MAX;
//     }
//
//     vk_pipeline p = v_vk_get_quantize_pipeline(ctx, quant);
//
//     v_pipeline_request_descriptor_sets(ctx, p, 1);
//
//     if (ctx->device->need_compiles) {
//         v_vk_load_shaders(ctx->device);
//     }
//
//     v_pipeline_allocate_descriptor_sets(ctx);
//
//     v_vk_buffer_write(x_buf, 0, x, x_sz);
//
//     vk_context subctx = v_vk_create_context(ctx, ctx->compute_cmd_pool);
//     v_vk_ctx_begin(ctx->device, subctx);
//     v_vk_quantize_q8_1(ctx, subctx, v_vk_subbuffer(ctx, x_buf), v_vk_subbuffer(ctx, qx_buf), ne);
//     v_vk_ctx_end(subctx);
//
//     auto begin = std::chrono::high_resolution_clock::now();
//
//     v_vk_submit(subctx, ctx->fence);
//     VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, true, UINT64_MAX), "v_vk_test_quantize waitForFences");
//     ctx->device->device.resetFences({ ctx->fence });
//     v_vk_queue_command_pools_cleanup(ctx->device);
//
//     auto end = std::chrono::high_resolution_clock::now();
//
//     double ms_quant = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
//     mmlVKBufferRead(qx_buf, 0, qx, qx_sz);
//
//     v_vk_quantize_data(x, qx_res, ne, quant);
//
//     int first_err = -1;
//
//     for (size_t i = 0; i < ne / 32; i++) {
//         double error = std::fabs(v_fp16_to_fp32(qx_res[i].v_COMMON_AGGR_U.v_COMMON_AGGR_S.d) - v_fp16_to_fp32(qx[i].v_COMMON_AGGR_U.v_COMMON_AGGR_S.d));
//
//         if (first_err < 0 && error > 0.1) {
//             first_err = i;
//         }
//
//         error = std::fabs(v_fp16_to_fp32(qx_res[i].v_COMMON_AGGR_U.v_COMMON_AGGR_S.s) - v_fp16_to_fp32(qx[i].v_COMMON_AGGR_U.v_COMMON_AGGR_S.s));
//
//         if (first_err < 0 && error > 0.1) {
//             first_err = i;
//         }
//
//         for (size_t j = 0; j < 32; j++) {
//             uint64_t error = std::abs(qx_res[i].qs[j] - qx[i].qs[j]);
//
//             if (first_err < 0 && error > 1) {
//                 first_err = i;
//             }
//         }
//     }
//
//     std::cerr << "TEST QUANTIZE " << v_type_name(quant) << " time=" << ms_quant << "ms " << (first_err == -1 ? "CORRECT" : "INCORRECT") << std::endl;
//
//     if (first_err != -1) {
//         std::cerr << "first_error = " << first_err << std::endl;
//         std::cerr << "Actual result: " << std::endl << std::endl;
//         std::cout << "d=" << v_fp16_to_fp32(qx[first_err].v_COMMON_AGGR_U.v_COMMON_AGGR_S.d) << " s=" << v_fp16_to_fp32(qx[first_err].v_COMMON_AGGR_U.v_COMMON_AGGR_S.s) << " ";
//         for (size_t j = 0; j < 32; j++) {
//             std::cout << " qs" << j << "=" << (uint32_t)qx[first_err].qs[j] << " ";
//         }
//         std::cerr << std::endl << std::endl << "Expected result: " << std::endl << std::endl;
//         std::cout << "d=" << v_fp16_to_fp32(qx_res[first_err].v_COMMON_AGGR_U.v_COMMON_AGGR_S.d) << " s=" << v_fp16_to_fp32(qx_res[first_err].v_COMMON_AGGR_U.v_COMMON_AGGR_S.s) << " ";
//         for (size_t j = 0; j < 32; j++) {
//             std::cout << " qs" << j << "=" << (uint32_t)qx_res[first_err].qs[j] << " ";
//         }
//         std::cerr << std::endl;
//     }
//
//     mmlVKDestroyBuffer(x_buf);
//     mmlVKDestroyBuffer(qx_buf);
//
//     free(x);
//     free(qx);
//     free(qx_res);
// }

//static void v_vk_test_dequant_matmul(v_backend_vk_context* ctx, size_t m, size_t n, size_t k, size_t batch,
//                                        size_t num_it, size_t split_k, size_t shader_size, MmlDataType quant,
//                                        bool mmq = false)
//{
//  VK_LOG_DEBUG(
//    "v_vk_test_dequant_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << num_it << ", " << split_k
//    << ", " << v_type_name(quant) << ")");
//  const size_t x_ne = m * k * batch;
//  const size_t y_ne = k * n * batch;
//  const size_t d_ne = m * n * batch;
//
//  vk_matmul_pipeline2* pipelines;
//
//  if (mmq)
//  {
//    pipelines = ctx->device->pipeline_dequant_mul_mat_mat_q8_1;
//  }
//  else
//  {
//    pipelines = ctx->device->pipeline_dequant_mul_mat_mat;
//  }
//
//  const bool fp16acc = ctx->device->fp16;
//
//  vk_pipeline p;
//  std::string shname;
//  if (shader_size == 0)
//  {
//    p = fp16acc ? pipelines[quant].f16acc->a_s : pipelines[quant].f32acc->a_s;
//    shname = std::string(v_type_name(quant)) + "_ALIGNED_S";
//  }
//  else if (shader_size == 1)
//  {
//    p = fp16acc ? pipelines[quant].f16acc->a_m : pipelines[quant].f32acc->a_m;
//    shname = std::string(v_type_name(quant)) + "_ALIGNED_M";
//  }
//  else if (shader_size == 2)
//  {
//    p = fp16acc ? pipelines[quant].f16acc->a_l : pipelines[quant].f32acc->a_l;
//    shname = std::string(v_type_name(quant)) + "_ALIGNED_L";
//  }
//  else
//  {
//    MML_ASSERT(0);
//  }
//
//  const size_t kpad = mmq ? 0 : v_vk_align_size(k, p->align);
//
//  if (mmq || k != kpad)
//  {
//    if (shader_size == 0)
//    {
//      p = fp16acc ? pipelines[quant].f16acc->s : pipelines[quant].f32acc->s;
//      shname = std::string(v_type_name(quant)) + "_S";
//    }
//    else if (shader_size == 1)
//    {
//      p = fp16acc ? pipelines[quant].f16acc->m : pipelines[quant].f32acc->m;
//      shname = std::string(v_type_name(quant)) + "_M";
//    }
//    else if (shader_size == 2)
//    {
//      p = fp16acc ? pipelines[quant].f16acc->l : pipelines[quant].f32acc->l;
//      shname = std::string(v_type_name(quant)) + "_L";
//    }
//    else
//    {
//      MML_ASSERT(0);
//    }
//  }
//
//  if (p == nullptr)
//  {
//    std::cerr << "error: no pipeline for v_vk_test_dequant_matmul " << v_type_name(quant) << std::endl;
//    return;
//  }
//
//  const size_t x_sz = sizeof(float) * x_ne;
//  const size_t y_sz = sizeof(float) * y_ne;
//  const size_t qx_sz = x_ne * v_type_size(quant) / blockSize(quant);
//  const size_t qy_sz = mmq ? y_ne * v_type_size(v_TYPE_Q8_1) / blockSize(v_TYPE_Q8_1) : y_sz;
//  const size_t d_sz = sizeof(float) * d_ne;
//  float* x = (float*)malloc(x_sz);
//  float* y = (float*)malloc(y_sz);
//  void* qx = malloc(qx_sz);
//  VKBuffer qx_buf = mmlVKCreateBufferCheck(ctx->device, qx_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
//  VKBuffer y_buf = mmlVKCreateBufferCheck(ctx->device, y_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
//  VKBuffer qy_buf = mmlVKCreateBufferCheck(ctx->device, qy_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
//  VKBuffer d_buf = mmlVKCreateBufferCheck(ctx->device, d_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
//  float* d = (float*)malloc(d_sz);
//  float* d_chk = (float*)malloc(d_sz);
//  for (size_t i = 0; i < x_ne; i++)
//  {
//    x[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
//    // x[i] = (i % k == i / k) ? 1.0f : 0.0f;
//    // x[i] = i % k;
//  }
//
//  v_vk_quantize_data(x, qx, x_ne, quant);
//
//  for (size_t i = 0; i < y_ne; i++)
//  {
//    y[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
//    // y[i] = (i % k == i / k) ? 1.0f : 0.0f;
//    // y[i] = i % k;
//  }
//
//  v_pipeline_request_descriptor_sets(ctx, p, num_it);
//  if (split_k > 1)
//  {
//    v_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_split_k_reduce, num_it);
//
//    if (ctx->prealloc_split_k == nullptr || ctx->prealloc_split_k->size < sizeof(float) * d_ne * split_k)
//    {
//      // Resize buffer
//      if (ctx->prealloc_split_k != nullptr)
//      {
//        mmlVKDestroyBuffer(ctx->prealloc_split_k);
//      }
//      ctx->prealloc_split_k = mmlVKCreateBufferCheck(ctx->device, sizeof(float) * d_ne * split_k,
//                                                          {vk::MemoryPropertyFlagBits::eDeviceLocal});
//    }
//  }
//  if (mmq)
//  {
//    v_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_quantize_q8_1, num_it);
//  }
//
//  if (ctx->device->need_compiles)
//  {
//    mmlVkLoadShaders(ctx->device);
//  }
//
//  v_pipeline_allocate_descriptor_sets(ctx);
//
//  v_vk_buffer_write(qx_buf, 0, qx, qx_sz);
//  v_vk_buffer_write(y_buf, 0, y, y_sz);
//
//  vk_context subctx = v_vk_create_context(ctx, ctx->compute_cmd_pool);
//  v_vk_ctx_begin(ctx->device, subctx);
//  if (mmq)
//  {
//    for (size_t i = 0; i < num_it; i++)
//    {
//      v_vk_quantize_q8_1(ctx, subctx, {y_buf, 0, y_sz}, {qy_buf, 0, qy_sz}, y_ne);
//      v_vk_matmul(
//        ctx, subctx, p, {qx_buf, 0, qx_sz}, {qy_buf, 0, qy_sz}, {d_buf, 0, d_sz}, {
//          ctx->prealloc_split_k, 0, ctx->prealloc_size_split_k
//        },
//        m, n, k,
//        k, k, m, k * m, k * n, m * n,
//        split_k, batch, batch, batch, 1, 1, n
//      );
//    }
//  }
//  else
//  {
//    for (size_t i = 0; i < num_it; i++)
//    {
//      v_vk_matmul(
//        ctx, subctx, p, {qx_buf, 0, qx_sz}, {y_buf, 0, y_sz}, {d_buf, 0, d_sz}, {
//          ctx->prealloc_split_k, 0, ctx->prealloc_size_split_k
//        },
//        m, n, k,
//        k, k, m, k * m, k * n, m * n,
//        split_k, batch, batch, batch, 1, 1, n
//      );
//    }
//  }
//  v_vk_ctx_end(subctx);
//
//  auto begin = std::chrono::high_resolution_clock::now();
//
//  v_vk_submit(subctx, ctx->fence);
//  VK_CHECK(ctx->device->device.waitForFences({ctx->fence}, true, UINT64_MAX), "v_vk_test_dequant waitForFences");
//  ctx->device->device.resetFences({ctx->fence});
//  v_vk_queue_command_pools_cleanup(ctx->device);
//
//  auto end = std::chrono::high_resolution_clock::now();
//
//  double time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0;
//  mmlVKBufferRead(d_buf, 0, d, d_sz);
//
//  InitParmaeters iparams = {
//    /*.mem_size   =*/ 1024 * 1024 * 1024,
//    /*.mem_buffer =*/ NULL,
//    /*.no_alloc   =*/ true,
//  };
//
//  Context* v_ctx = mmlCtxInit(iparams);
//
//  MmlTensor* src0_ggml = mmlNewTensor3D(v_ctx, quant, k, m, batch);
//  MmlTensor* src1_ggml = mmlNewTensor3D(v_ctx, v_TYPE_F32, k, n, batch);
//  MmlTensor* tensor_ggml = mmlMatrixMul(v_ctx, src0_ggml, src1_ggml);
//
//  src0_ggml->data = qx;
//  src1_ggml->data = y;
//  tensor_ggml->data = d_chk;
//
//  MmlComputeGraph* cgraph = mmlNewGraph(v_ctx);
//  mmlBuildFowardExpand(cgraph, tensor_ggml);
//
//  v_graph_compute_with_ctx(v_ctx, cgraph, 1);
//
//  mmlFree(v_ctx);
//
//  double avg_err = 0.0;
//  int first_err_n = -1;
//  int first_err_m = -1;
//  int first_err_b = -1;
//
//  for (size_t i = 0; i < m * n * batch; i++)
//  {
//    double err = std::fabs(d[i] - d_chk[i]);
//    avg_err += err;
//
//    if ((err > 0.05f || std::isnan(err)) && first_err_n == -1)
//    {
//      first_err_b = i / (m * n);
//      first_err_n = (i % (m * n)) / m;
//      first_err_m = (i % (m * n)) % m;
//    }
//  }
//
//  avg_err /= m * n;
//
//  double tflops = 2.0 * m * n * k * batch * num_it / (time_ms / 1000.0) / (1000.0 * 1000.0 * 1000.0 * 1000.0);
//
//  std::cerr << "TEST dequant matmul " << shname;
//  if (mmq)
//  {
//    std::cerr << " mmq";
//  }
//  std::cerr << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" << split_k << " matmul " <<
//    time_ms / num_it << "ms " << tflops << " TFLOPS avg_err=" << avg_err << std::endl;
//
//  if (avg_err > 0.01 || std::isnan(avg_err))
//  {
//    std::cerr << "m = " << first_err_m << " n = " << first_err_n << " b = " << first_err_b << std::endl;
//    std::cerr << "Actual result: " << std::endl << std::endl;
//    v_vk_print_matrix_area(d, v_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
//    std::cerr << std::endl;
//    std::cerr << "Expected result: " << std::endl << std::endl;
//    v_vk_print_matrix_area(d_chk, v_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
//
//    std::cerr << "src0: " << std::endl << std::endl;
//    v_vk_print_matrix_area(x, v_TYPE_F32, k, m, first_err_m, first_err_n, first_err_b);
//    std::cerr << std::endl;
//    std::cerr << "src1: " << std::endl << std::endl;
//    v_vk_print_matrix_area(y, v_TYPE_F32, k, n, first_err_m, first_err_n, first_err_b);
//
//    if (split_k > 1)
//    {
//      float* split_k_buf = (float*)malloc(sizeof(float) * d_ne * split_k);
//      mmlVKBufferRead(ctx->prealloc_split_k, 0, split_k_buf, sizeof(float) * d_ne * split_k);
//
//      std::cerr << "d_buf0: " << std::endl << std::endl;
//      v_vk_print_matrix_area(split_k_buf, v_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
//
//      std::cerr << "d_buf1: " << std::endl << std::endl;
//      v_vk_print_matrix_area(split_k_buf + d_ne, v_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
//
//      std::cerr << "d_buf2: " << std::endl << std::endl;
//      v_vk_print_matrix_area(split_k_buf + 2 * d_ne, v_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
//
//      std::cerr << "d_buf3: " << std::endl << std::endl;
//      v_vk_print_matrix_area(split_k_buf + 3 * d_ne, v_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
//
//      free(split_k_buf);
//    }
//  }
//
//  mmlVKDestroyBuffer(qx_buf);
//  mmlVKDestroyBuffer(y_buf);
//  mmlVKDestroyBuffer(qy_buf);
//  mmlVKDestroyBuffer(d_buf);
//
//  free(x);
//  free(qx);
//  free(y);
//  free(d);
//  free(d_chk);
//}
#endif
// checks
//#define v_VULKAN_CHECK_RESULTS
//#include <iostream>
#ifdef v_VULKAN_CHECK_RESULTS
void v_vk_print_graph_origin(const v_tensor* tensor, std::vector<const v_tensor*>& done, int level = 0) {
  if (std::find(done.begin(), done.end(), tensor) != done.end() || level > 10) { return; }
  for (int j = 0; j < level; j++) { std::cerr << " "; }
  std::cerr << v_op_name(tensor->op) << " gpu=" << (tensor->extra != nullptr) << std::endl;

  done.push_back(tensor);

  for (int i = 0; i < v_MAX_SRC; i++) { if (tensor->src[i] != nullptr) { v_vk_print_graph_origin(tensor->src[i], done, level + 1); } }
}

void v_vk_print_tensor_area(const v_tensor* tensor, const void* data, int i0, int i1, int i2, int i3) {
  if (tensor->type != v_TYPE_F32 && tensor->type != v_TYPE_F16 && tensor->type != v_TYPE_I32) { return; }
  i0 = std::max(i0, 5);
  i1 = std::max(i1, 5);
  i2 = std::max(i2, 0);
  i3 = std::max(i3, 0);
  fprintf(stderr, "         ");
  for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) { fprintf(stderr, "%7d ", idx1); }
  fprintf(stderr, "\n");
  for (int idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
    fprintf(stderr, "%7d: ", idx0);
    for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
      if (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3
        >= 0 && i3 < tensor->ne[3]) {
        float val;
        if (tensor->type == v_TYPE_F32) {
          val = *(const float*)((const char*)data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + idx1 * tensor->nb[1] +
            idx0 * tensor->nb[0]);
        }
        else if (tensor->type == v_TYPE_F16) {
          val = v_fp16_to_fp32(
            *(const v_fp16_t*)((const char*)data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + idx1 * tensor->nb[1] +
              idx0 * tensor->nb[0]));
        }
        else if (tensor->type == v_TYPE_I32) {
          val = *(const int32_t*)((const char*)data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + idx1 * tensor->nb[1] +
            idx0 * tensor->nb[0]);
        }
        else { v_ABORT("fatal error"); }
        fprintf(stderr, "% 7.7f ", val);
      }
      else { fprintf(stderr, "        "); }
    }
    fprintf(stderr, "\n");
  }
}

void vk_print_tensor(const v_tensor* tensor, const char* name) {
  void* tensor_data = tensor->data;
  const bool is_gpu = tensor->buffer != nullptr && v_backend_buffer_is_vk(tensor->buffer);
  if (is_gpu) {
    const size_t tensor_size         = num_bytes(tensor);
    tensor_data                      = malloc(tensor_size);
    v_backend_vk_buffer_ctx* buf_ctx = (v_backend_vk_buffer_ctx*)tensor->buffer->context;
    vk_buffer buffer_gpu             = buf_ctx->dev_buffer;
    mmlVKBufferRead(buffer_gpu, vk_tensor_offset(tensor) + tensor->view_offs, tensor_data, tensor_size);
  }

  std::cerr << "TENSOR CHECK " << name << " (" << tensor->name << "): " << v_op_name(tensor->op) << std::endl;
  std::cerr << "tensor=" << tensor << " tensor->type: " << v_type_name(tensor->type) << " ne0=" << tensor->ne[0] <<
    " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] <<
    " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << std::endl;
  if (tensor->src[0] != nullptr) {
    std::cerr << "tensor->src[0]=" << tensor->src[0] << " name=" << tensor->src[0]->name << " op=" <<
      v_op_name(tensor->src[0]->op) << " type=" << v_type_name(tensor->src[0]->type) << " ne0=" << tensor->src[0]
      ->ne[0] << " nb0=" << tensor->src[0]->nb[0] << " ne1=" << tensor->src[0]->ne[1] << " nb1=" << tensor->src[0]->nb[
        1] << " ne2=" << tensor->src[0]->ne[2] << " nb2=" << tensor->src[0]->nb[2] << " ne3=" << tensor->src[0]->ne[3]
      << " nb3=" << tensor->src[0]->nb[3] << std::endl;
  }
  if (tensor->src[1] != nullptr) {
    std::cerr << "tensor->src[1]=" << tensor->src[1] << " name=" << tensor->src[1]->name << " op=" <<
      v_op_name(tensor->src[1]->op) << " type=" << v_type_name(tensor->src[1]->type) << " ne0=" << tensor->src[1]
      ->ne[0] << " nb0=" << tensor->src[1]->nb[0] << " ne1=" << tensor->src[1]->ne[1] << " nb1=" << tensor->src[1]->nb[
        1] << " ne2=" << tensor->src[1]->ne[2] << " nb2=" << tensor->src[1]->nb[2] << " ne3=" << tensor->src[1]->ne[3]
      << " nb3=" << tensor->src[1]->nb[3] << std::endl;
  }
  std::cerr << std::endl << "Result:" << std::endl;
  v_vk_print_tensor_area(tensor, tensor_data, 5, 5, 0, 0);
  std::cerr << std::endl;
  std::vector<const v_tensor*> done;
  v_vk_print_graph_origin(tensor, done);
  if (is_gpu) { free(tensor_data); }
}

void* comp_result;
size_t comp_size;
size_t comp_nb[4];
size_t check_counter = 0;

void vk_check_results_0(vk_backend_ctx* ctx, v_cgraph* cgraph, int tensor_idx) {
  v_tensor* tensor = cgraph->nodes[tensor_idx];
  if (tensor->op == v_OP_TRANSPOSE || tensor->op == v_OP_SET_ROWS) { return; }

  bool fused_rms_norm_mul = false;
  int rms_norm_idx        = -1;
  if (ctx->num_additional_fused_ops == 1 &&
    tensor->op == v_OP_RMS_NORM &&
    cgraph->nodes[tensor_idx + 1]->op == v_OP_MUL) {
    fused_rms_norm_mul = true;
    tensor             = cgraph->nodes[tensor_idx + 1];
  }

  check_counter++;
  if (!(vk_output_tensor > 0 && vk_output_tensor == check_counter) && check_counter <= vk_skip_checks) { return; }

  VK_LOG_DEBUG("v_vk_check_results_0(" << tensor->name << ")");

  v_tensor* src0 = tensor->src[0];
  v_tensor* src1 = tensor->src[1];

  struct v_init_param iparams = {
    /*.mem_size   =*/ 2ul * 1024ul * 1024ul * 1024ul,
    /*.mem_buffer =*/ NULL,
    /*.no_alloc   =*/ false,
  };

  struct v_ctx* v_ctx = v_ctx_init(iparams);

  std::array<struct v_tensor*, v_MAX_SRC> src_clone = {
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr
  };
  std::array<size_t, v_MAX_SRC> src_size  = {};
  std::array<void*, v_MAX_SRC> src_buffer = {};
  const char* srci_name[v_MAX_SRC]        = {
    "src0", "src1", "src2", "src3", "src4", "src5", "src6", "src7", "src8", "src9"
  };

  struct v_tensor* tensor_clone = nullptr;

  for (int i = 0; i < v_MAX_SRC; i++) {
    v_tensor* srci = tensor->src[i];
    if (fused_rms_norm_mul) {
      rms_norm_idx = tensor->src[0]->op == v_OP_RMS_NORM
                       ? 0
                       : 1;
      v_tensor* rms_norm = tensor->src[rms_norm_idx];
      switch (i) {
        case 0: srci = rms_norm->src[0];
          break;
        case 1: srci = tensor->src[1 - rms_norm_idx];
          break;
        default: continue;
      }
    }
    if (srci == nullptr) { continue; }
    v_tensor* srci_clone = v_dup_tensor(v_ctx, srci);
    size_t srci_size     = num_bytes(srci);

    src_clone[i]  = srci_clone;
    src_size[i]   = num_bytes(srci);
    src_buffer[i] = malloc(srci_size);

    srci_clone->data = src_buffer[i];
    if (v_backend_buffer_is_host(srci->buffer)) {
      memcpy(srci_clone->data, srci->data, srci_size);
      memcpy(srci_clone->nb, srci->nb, sizeof(size_t) * 4);
    }
    else if (v_backend_buffer_is_vk(srci->buffer)) {
      v_backend_vk_buffer_ctx* buf_ctx = (v_backend_vk_buffer_ctx*)srci->buffer->context;
      vk_buffer& buffer_gpu            = buf_ctx->dev_buffer;
      uint64_t offset                  = vk_tensor_offset(srci) + srci->view_offs;
      if (!v_is_contiguous(srci) && v_vk_dim01_contiguous(srci)) {
        for (int i3 = 0; i3 < srci->ne[3]; i3++) {
          for (int i2 = 0; i2 < srci->ne[2]; i2++) {
            const int idx = i3 * srci->ne[2] + i2;
            mmlVKBufferRead(buffer_gpu,
                            offset + idx * srci->nb[2],
                            ((char*)srci_clone->data + idx * srci_clone->nb[2]),
                            srci->ne[1] * srci->nb[1]);
          }
        }

        srci_clone->nb[0] = srci->nb[0];
        srci_clone->nb[1] = srci->nb[1];
        for (int i = 2; i < 4; i++) { srci_clone->nb[i] = srci_clone->nb[i - 1] * srci_clone->ne[i - 1]; }
      }
      else {
        if (offset + srci_size >= buffer_gpu->size) { srci_size = buffer_gpu->size - offset; }
        mmlVKBufferRead(buffer_gpu, offset, srci_clone->data, srci_size);
        memcpy(srci_clone->nb, srci->nb, sizeof(size_t) * 4);
      }
    }
    else { v_ABORT("fatal error"); }

    if (vk_output_tensor > 0 && vk_output_tensor == check_counter) { vk_print_tensor(srci, srci_name[i]); }
  }

  if (tensor->op == v_OP_FLASH_ATTN_EXT) {
    const float* params = (const float*)tensor->op_params;
    tensor_clone        = v_flash_attn_ext(v_ctx,
                                           src_clone[0],
                                           src_clone[1],
                                           src_clone[2],
                                           src_clone[3],
                                           params[0],
                                           params[1],
                                           params[2]);
    if (src_clone[4]) { v_flash_attn_ext_add_sinks(tensor_clone, src_clone[4]); }
  }
  else if (tensor->op == v_OP_MUL_MAT) { tensor_clone = v_matmul(v_ctx, src_clone[0], src_clone[1]); }
  else if (tensor->op == v_OP_MUL_MAT_ID) { tensor_clone = mmlMatrixMulId(v_ctx, src_clone[0], src_clone[1], src_clone[2]); }
  else if (tensor->op == v_OP_SUB) { tensor_clone = v_sub(v_ctx, src_clone[0], src_clone[1]); }
  else if (tensor->op == v_OP_MUL) {
    if (fused_rms_norm_mul) {
      tensor_clone = v_rms_norm(v_ctx, src_clone[0], *(float*)tensor->src[rms_norm_idx]->op_params);
      tensor_clone = v_mul(v_ctx, tensor_clone, src_clone[1 - rms_norm_idx]);
    }
    else { tensor_clone = v_mul(v_ctx, src_clone[0], src_clone[1]); }
  }
  else if (tensor->op == v_OP_DIV) { tensor_clone = v_div(v_ctx, src_clone[0], src_clone[1]); }
  else if (tensor->op == v_OP_CONCAT) { tensor_clone = v_concat(v_ctx, src_clone[0], src_clone[1], *(int*)tensor->op_params); }
  else if (tensor->op == v_OP_UPSCALE) {
    tensor_clone = v_interpolate(v_ctx,
                                 src_clone[0],
                                 tensor->ne[0],
                                 tensor->ne[1],
                                 tensor->ne[2],
                                 tensor->ne[3],
                                 (v_scale_mode)tensor->op_params[0]);
  }
  else if (tensor->op == V_OP_SCALE) {
    const float* params = (const float*)tensor->op_params;
    tensor_clone        = v_scale_bias(v_ctx, src_clone[0], params[0], params[1]);
  }
  else if (tensor->op == v_OP_SQR) { tensor_clone = v_sqr(v_ctx, src_clone[0]); }
  else if (tensor->op == v_OP_SQRT) { tensor_clone = v_sqrt(v_ctx, src_clone[0]); }
  else if (tensor->op == v_OP_SIN) { tensor_clone = v_sin(v_ctx, src_clone[0]); }
  else if (tensor->op == v_OP_COS) { tensor_clone = v_cos(v_ctx, src_clone[0]); }
  else if (tensor->op == v_OP_CLAMP) {
    const float* params = (const float*)tensor->op_params;
    tensor_clone        = v_clamp(v_ctx, src_clone[0], params[0], params[1]);
  }
  else if (tensor->op == v_OP_PAD) {
    tensor_clone = v_pad_ext(v_ctx,
                             src_clone[0],
                             tensor->op_params[0],
                             tensor->op_params[1],
                             tensor->op_params[2],
                             tensor->op_params[3],
                             tensor->op_params[4],
                             tensor->op_params[5],
                             tensor->op_params[6],
                             tensor->op_params[7]);
  }
  else if (tensor->op == v_OP_REPEAT) { tensor_clone = v_repeat(v_ctx, src_clone[0], tensor); }
  else if (tensor->op == v_OP_REPEAT_BACK) { tensor_clone = v_repeat_back(v_ctx, src_clone[0], tensor); }
  else if (tensor->op == v_OP_ADD) { tensor_clone = v_add(v_ctx, src_clone[0], src_clone[1]); }
  else if (tensor->op == v_OP_ACC) {
    tensor_clone = v_acc(v_ctx,
                         src_clone[0],
                         src_clone[1],
                         tensor->op_params[0],
                         tensor->op_params[1],
                         tensor->op_params[2],
                         tensor->op_params[3]);
  }
  else if (tensor->op == v_OP_NORM) { tensor_clone = v_norm(v_ctx, src_clone[0], *(float*)tensor->op_params); }
  else if (tensor->op == v_OP_GROUP_NORM) {
    const float* float_params = (const float*)tensor->op_params;
    tensor_clone              = v_group_norm(v_ctx, src_clone[0], tensor->op_params[0], float_params[1]);
  }
  else if (tensor->op == v_OP_RMS_NORM) { tensor_clone = v_rms_norm(v_ctx, src_clone[0], *(float*)tensor->op_params); }
  else if (tensor->op == v_OP_RMS_NORM_BACK) {
    const float eps = ((float*)tensor->op_params)[0];
    tensor_clone    = v_rms_norm_back(v_ctx, src_clone[0], src_clone[1], eps);
  }
  else if (tensor->op == v_OP_SILU_BACK) { tensor_clone = v_silu_back(v_ctx, src_clone[0], src_clone[1]); }
  else if (tensor->op == v_OP_L2_NORM) {
    const float eps = ((float*)tensor->op_params)[0];
    tensor_clone    = v_norm_l2(v_ctx, src_clone[0], eps);
  }
  else if (tensor->op == V_OP_SOFT_MAX) {
    if (src1 != nullptr) {
      const float* params = (const float*)tensor->op_params;
      tensor_clone        = v_soft_max_ext(v_ctx, src_clone[0], src_clone[1], params[0], params[1]);
    }
    else { tensor_clone = v_soft_max(v_ctx, src_clone[0]); }
  }
  else if (tensor->op == v_OP_SOFT_MAX_BACK) {
    tensor_clone = v_soft_max_ext_back(v_ctx,
                                       src_clone[0],
                                       src_clone[1],
                                       ((float*)tensor->op_params)[0],
                                       ((float*)tensor->op_params)[1]);
  }
  else if (tensor->op == V_OP_DIAG_MASK_INF) { tensor_clone = v_diag_mask_inf(v_ctx, src_clone[0], tensor->op_params[0]); }
  else if (tensor->op == V_OP_ROPE || tensor->op == v_OP_ROPE_BACK) {
    const int n_dims = ((int32_t*)tensor->op_params)[1];
    const int mode   = ((int32_t*)tensor->op_params)[2];
    //const int n_ctx_ggml       = ((int32_t *) tensor->op_params)[3];
    const int n_ctx_orig_ggml = ((int32_t*)tensor->op_params)[4];
    const float freq_base     = ((float*)tensor->op_params)[5];
    const float freq_scale    = ((float*)tensor->op_params)[6];
    const float ext_factor    = ((float*)tensor->op_params)[7];
    const float attn_factor   = ((float*)tensor->op_params)[8];
    const float beta_fast     = ((float*)tensor->op_params)[9];
    const float beta_slow     = ((float*)tensor->op_params)[10];
    if (mode & v_ROPE_TYPE_MROPE) {
      int32_t* sections = ((int32_t*)tensor->op_params) + 11;
      if (tensor->op == V_OP_ROPE) {
        tensor_clone = v_rope_multi(v_ctx,
                                    src_clone[0],
                                    src_clone[1],
                                    src_clone[2],
                                    n_dims,
                                    sections,
                                    mode,
                                    n_ctx_orig_ggml,
                                    freq_base,
                                    freq_scale,
                                    ext_factor,
                                    attn_factor,
                                    beta_fast,
                                    beta_slow);
      }
      else {
        tensor_clone = v_rope_multi_back(v_ctx,
                                         src_clone[0],
                                         src_clone[1],
                                         src_clone[2],
                                         n_dims,
                                         sections,
                                         mode,
                                         n_ctx_orig_ggml,
                                         freq_base,
                                         freq_scale,
                                         ext_factor,
                                         attn_factor,
                                         beta_fast,
                                         beta_slow);
      }
    }
    else {
      if (tensor->op == V_OP_ROPE) {
        tensor_clone = v_rope_ext(v_ctx,
                                  src_clone[0],
                                  src_clone[1],
                                  src_clone[2],
                                  n_dims,
                                  mode,
                                  n_ctx_orig_ggml,
                                  freq_base,
                                  freq_scale,
                                  ext_factor,
                                  attn_factor,
                                  beta_fast,
                                  beta_slow);
      }
      else {
        tensor_clone = v_rope_ext_back(v_ctx,
                                       src_clone[0],
                                       src_clone[1],
                                       src_clone[2],
                                       n_dims,
                                       mode,
                                       n_ctx_orig_ggml,
                                       freq_base,
                                       freq_scale,
                                       ext_factor,
                                       attn_factor,
                                       beta_fast,
                                       beta_slow);
      }
    }
  }
  else if (tensor->op == v_OP_UNARY) {
    switch (v_get_unary_op(tensor)) {
      case v_UNARY_OP_EXP:
        tensor_clone = v_exp(v_ctx, src_clone[0]);
        break;
      case v_UNARY_OP_SILU:
        tensor_clone = v_silu(v_ctx, src_clone[0]);
        break;
      case v_UNARY_OP_GELU:
        tensor_clone = v_gelu(v_ctx, src_clone[0]);
        break;
      case v_UNARY_OP_GELU_ERF:
        tensor_clone = v_gelu_erf(v_ctx, src_clone[0]);
        break;
      case v_UNARY_OP_GELU_QUICK:
        tensor_clone = v_gelu_quick(v_ctx, src_clone[0]);
        break;
      case v_UNARY_OP_RELU:
        tensor_clone = v_relu(v_ctx, src_clone[0]);
        break;
      case v_UNARY_OP_TANH:
        tensor_clone = v_tanh(v_ctx, src_clone[0]);
        break;
      case v_UNARY_OP_SIGMOID:
        tensor_clone = v_sigmoid(v_ctx, src_clone[0]);
        break;
      case v_UNARY_OP_LOG:
        tensor_clone = v_log(v_ctx, src_clone[0]);
        break;
      case v_UNARY_OP_HARDSIGMOID:
        tensor_clone = v_hardsigmoid(v_ctx, src_clone[0]);
        break;
      case v_UNARY_OP_HARDSWISH:
        tensor_clone = v_hardswish(v_ctx, src_clone[0]);
        break;
      default:
        std::cerr << "Missing vk_check_results OP: " << v_op_name(tensor->op) << std::endl;
        v_ABORT("fatal error");
    }
  }
  else if (tensor->op == v_OP_GLU) {
    if (src_clone[1] == nullptr) { tensor_clone = v_glu(v_ctx, src_clone[0], (v_glu_op)tensor->op_params[0], tensor->op_params[1]); }
    else { tensor_clone = v_glu_split(v_ctx, src_clone[0], src_clone[1], (v_glu_op)tensor->op_params[0]); }
    v_set_op_params_i32(tensor_clone, 2, v_get_op_params_i32(tensor, 2));
    v_set_op_params_i32(tensor_clone, 3, v_get_op_params_i32(tensor, 3));
  }
  else if (tensor->op == v_OP_CPY || tensor->op == v_OP_DUP) {
    if (src1 == nullptr) {
      tensor_clone       = v_dup(v_ctx, src_clone[0]);
      tensor_clone->type = tensor->type;
    }
    else { tensor_clone = v_cpy(v_ctx, src_clone[0], src_clone[1]); }
  }
  else if (tensor->op == v_OP_CONT) { tensor_clone = v_cont_4d(v_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]); }
  else if (tensor->op == v_OP_RESHAPE) { tensor_clone = v_reshape_4d(v_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]); }
  else if (tensor->op == V_OP_VIEW) {
    tensor_clone = v_view_4d(v_ctx,
                             src_clone[0],
                             tensor->ne[0],
                             tensor->ne[1],
                             tensor->ne[2],
                             tensor->ne[3],
                             tensor->nb[1],
                             tensor->nb[2],
                             tensor->nb[3],
                             ((int32_t*)tensor->op_params)[0]);
  }
  else if (tensor->op == V_OP_PERMUTE) {
    int32_t* params = (int32_t*)tensor->op_params;
    tensor_clone    = v_permute(v_ctx, src_clone[0], params[0], params[1], params[2], params[3]);
  }
  else if (tensor->op == v_OP_TRANSPOSE) { tensor_clone = v_transpose(v_ctx, src_clone[0]); }
  else if (tensor->op == v_OP_GET_ROWS) { tensor_clone = v_get_rows(v_ctx, src_clone[0], src_clone[1]); }
  else if (tensor->op == v_OP_ARGSORT) { tensor_clone = v_argsort(v_ctx, src_clone[0], (v_sort_order)*(int*)tensor->op_params); }
  else if (tensor->op == v_OP_SUM) { tensor_clone = v_sum(v_ctx, src_clone[0]); }
  else if (tensor->op == v_OP_SUM_ROWS) { tensor_clone = v_sum_rows(v_ctx, src_clone[0]); }
  else if (tensor->op == V_OP_MEAN) { tensor_clone = v_mean(v_ctx, src_clone[0]); }
  else if (tensor->op == V_OP_ARGMAX) { tensor_clone = v_argmax(v_ctx, src_clone[0]); }
  else if (tensor->op == v_OP_COUNT_EQUAL) { tensor_clone = v_count_equal(v_ctx, src_clone[0], src_clone[1]); }
  else if (tensor->op == V_OP_IM2COL) {
    const int32_t s0 = tensor->op_params[0];
    const int32_t s1 = tensor->op_params[1];
    const int32_t p0 = tensor->op_params[2];
    const int32_t p1 = tensor->op_params[3];
    const int32_t d0 = tensor->op_params[4];
    const int32_t d1 = tensor->op_params[5];

    const bool is_2D = tensor->op_params[6] == 1;
    tensor_clone     = v_im2col(v_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1, is_2D, tensor->type);
  }
  else if (tensor->op == v_OP_IM2COL_3D) {
    const int32_t s0 = tensor->op_params[0];
    const int32_t s1 = tensor->op_params[1];
    const int32_t s2 = tensor->op_params[2];
    const int32_t p0 = tensor->op_params[3];
    const int32_t p1 = tensor->op_params[4];
    const int32_t p2 = tensor->op_params[5];
    const int32_t d0 = tensor->op_params[6];
    const int32_t d1 = tensor->op_params[7];
    const int32_t d2 = tensor->op_params[8];
    const int32_t IC = tensor->op_params[9];

    tensor_clone = v_im2col_3d(v_ctx,
                               src_clone[0],
                               src_clone[1],
                               IC,
                               s0,
                               s1,
                               s2,
                               p0,
                               p1,
                               p2,
                               d0,
                               d1,
                               d2,
                               tensor->type);
  }
  else if (tensor->op == v_OP_TIMESTEP_EMBEDDING) {
    const int32_t dim        = tensor->op_params[0];
    const int32_t max_period = tensor->op_params[1];
    tensor_clone             = v_timestep_embedding(v_ctx, src_clone[0], dim, max_period);
  }
  else if (tensor->op == v_OP_CONV_TRANSPOSE_1D) {
    const int32_t s0 = tensor->op_params[0];
    const int32_t p0 = tensor->op_params[1];
    const int32_t d0 = tensor->op_params[2];
    tensor_clone     = v_conv_transpose_1d(v_ctx, src_clone[0], src_clone[1], s0, p0, d0);
  }
  else if (tensor->op == V_OP_POOL_2D) {
    enum v_op_pool op = static_cast<v_op_pool>(tensor->op_params[0]);
    const int32_t k0  = tensor->op_params[1];
    const int32_t k1  = tensor->op_params[2];
    const int32_t s0  = tensor->op_params[3];
    const int32_t s1  = tensor->op_params[4];
    const int32_t p0  = tensor->op_params[5];
    const int32_t p1  = tensor->op_params[6];

    tensor_clone = v_pool_2d(v_ctx, src_clone[0], op, k0, k1, s0, s1, p0, p1);
  }
  else if (tensor->op == v_OP_CONV_2D) {
    const int32_t s0 = tensor->op_params[0];
    const int32_t s1 = tensor->op_params[1];
    const int32_t p0 = tensor->op_params[2];
    const int32_t p1 = tensor->op_params[3];
    const int32_t d0 = tensor->op_params[4];
    const int32_t d1 = tensor->op_params[5];
    tensor_clone     = v_conv_2d(v_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1);
  }
  else if (tensor->op == v_OP_CONV_TRANSPOSE_2D) {
    const int32_t s = tensor->op_params[0];
    tensor_clone    = v_conv_transpose_2d_p0(v_ctx, src_clone[0], src_clone[1], s);
  }
  else if (tensor->op == v_OP_LEAKY_RELU) {
    const float* op_params = (const float*)tensor->op_params;
    tensor_clone           = v_leaky_relu(v_ctx, src_clone[0], op_params[0], false);
  }
  else if (tensor->op == v_OP_RWKV_WKV6) {
    tensor_clone = v_rwkv_wkv6(v_ctx,
                               src_clone[0],
                               src_clone[1],
                               src_clone[2],
                               src_clone[3],
                               src_clone[4],
                               src_clone[5]);
  }
  else if (tensor->op == v_OP_RWKV_WKV7) {
    tensor_clone = v_rwkv_wkv7(v_ctx,
                               src_clone[0],
                               src_clone[1],
                               src_clone[2],
                               src_clone[3],
                               src_clone[4],
                               src_clone[5],
                               src_clone[6]);
  }
  else if (tensor->op == v_OP_OPT_STEP_ADAMW) {
    src_clone[0]->flags = src0->flags;
    tensor_clone        = v_opt_step_adamw(v_ctx,
                                           src_clone[0],
                                           src_clone[1],
                                           src_clone[2],
                                           src_clone[3],
                                           src_clone[4]);
  }
  else if (tensor->op == v_OP_OPT_STEP_SGD) {
    src_clone[0]->flags = src0->flags;
    tensor_clone        = v_opt_step_sgd(v_ctx,
                                         src_clone[0],
                                         src_clone[1],
                                         src_clone[2]);
  }
  else if (tensor->op == v_OP_ADD_ID) { tensor_clone = add_id(v_ctx, src_clone[0], src_clone[1], src_clone[2]); }
  else if (tensor->op == v_OP_SSM_SCAN) {
    tensor_clone = v_ssm_scan(v_ctx,
                              src_clone[0],
                              src_clone[1],
                              src_clone[2],
                              src_clone[3],
                              src_clone[4],
                              src_clone[5],
                              src_clone[6]);
  }
  else if (tensor->op == v_OP_SSM_CONV) { tensor_clone = v_ssm_conv(v_ctx, src_clone[0], src_clone[1]); }
  else {
    std::cerr << "Missing vk_check_results OP: " << v_op_name(tensor->op) << std::endl;
    v_ABORT("fatal error");
  }

  //v_cgraph* cgraph_cpu = new_graph(v_ctx);
  //v_build_foward_expand(cgraph_cpu, tensor_clone);
  //
  //v_graph_compute_with_ctx(v_ctx, cgraph_cpu, 8);

  if (vk_output_tensor > 0 && vk_output_tensor == check_counter) { vk_print_tensor(tensor_clone, "tensor_clone"); }

  comp_size = num_bytes(tensor_clone);

  comp_result = malloc(comp_size);
  memcpy(comp_result, tensor_clone->data, comp_size);
  memcpy(comp_nb, tensor_clone->nb, sizeof(size_t) * 4);

  for (int i = 0; i < v_MAX_SRC; i++) { if (src_buffer[i] != nullptr) { free(src_buffer[i]); } }

  free_ctx(v_ctx);

  VK_LOG_DEBUG("END v_vk_check_results_0(" << tensor->name << ")");
}

void vk_check_results_1(vk_backend_ctx* ctx, v_cgraph* cgraph, int tensor_idx) {
  v_tensor* tensor = cgraph->nodes[tensor_idx];
  if (tensor->op == v_OP_TRANSPOSE || tensor->op == v_OP_SET_ROWS) { return; }
  if (ctx->num_additional_fused_ops == 1 &&
    tensor->op == v_OP_RMS_NORM &&
    cgraph->nodes[tensor_idx + 1]->op == v_OP_MUL) { tensor = cgraph->nodes[tensor_idx + 1]; }

  if (!(vk_output_tensor > 0 && vk_output_tensor == check_counter) && check_counter <= vk_skip_checks) { return; }

  VK_LOG_DEBUG("v_vk_check_results_1(" << tensor->name << ")");

  v_tensor* src0 = tensor->src[0];
  v_tensor* src1 = tensor->src[1];
  v_tensor* src2 = tensor->src[2];
  v_tensor* src3 = tensor->src[3];

  void* tensor_data = tensor->data;

  if (v_backend_buffer_is_vk(tensor->buffer)) {
    size_t tensor_size = num_bytes(tensor);
    tensor_data        = malloc(tensor_size);

    v_backend_vk_buffer_ctx* buf_ctx = (v_backend_vk_buffer_ctx*)tensor->buffer->context;

    vk_buffer& buffer_gpu = buf_ctx->dev_buffer;
    uint64_t offset       = vk_tensor_offset(tensor) + tensor->view_offs;
    if (offset + tensor_size >= buffer_gpu->size) { tensor_size = buffer_gpu->size - offset; }

    mmlVKBufferRead(buffer_gpu, offset, tensor_data, tensor_size);
  }

  float first_error_result       = -1.0f;
  float first_error_correct      = -1.0f;
  std::array<int, 4> first_error = {-1, -1, -1, -1};
  double avg_err                 = 0.0;
  size_t counter                 = 0;

  for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
    for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
      for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
        for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
          const bool buffer_size_fit = i3 * comp_nb[3] + i2 * comp_nb[2] + i1 * comp_nb[1] + i0 * comp_nb[0] <
            comp_size;
          float correct = 0.0f;
          float result  = 0.0f;

          if (buffer_size_fit) {
            if (tensor->type == v_TYPE_F32) {
              correct = *(float*)((char*)comp_result + i3 * comp_nb[3] + i2 * comp_nb[2] + i1 * comp_nb[1] + i0 *
                comp_nb[0]);
              result = *(float*)((char*)tensor_data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0
                * tensor->nb[0]);
            }
            else if (tensor->type == v_TYPE_F16) {
              correct = v_fp16_to_fp32(
                *(v_fp16_t*)((char*)comp_result + i3 * comp_nb[3] + i2 * comp_nb[2] + i1 * comp_nb[1] + i0 * comp_nb[
                  0]));
              result = v_fp16_to_fp32(
                *(v_fp16_t*)((char*)tensor_data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 *
                  tensor->nb[0]));
            }
            else if (tensor->type == v_TYPE_BF16) {
              correct = v_bf16_to_fp32(
                *(v_bf16_t*)((char*)comp_result + i3 * comp_nb[3] + i2 * comp_nb[2] + i1 * comp_nb[1] + i0 * comp_nb[
                  0]));
              result = v_bf16_to_fp32(
                *(v_bf16_t*)((char*)tensor_data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 *
                  tensor->nb[0]));
            }
            else if (tensor->type == v_TYPE_I32) {
              correct = *(int32_t*)((char*)comp_result + i3 * comp_nb[3] + i2 * comp_nb[2] + i1 * comp_nb[1] + i0 *
                comp_nb[0]);
              result = *(int32_t*)((char*)tensor_data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] +
                i0 * tensor->nb[0]);
            }
            else if (tensor->type == v_TYPE_I64) {
              correct = *(int64_t*)((char*)comp_result + i3 * comp_nb[3] + i2 * comp_nb[2] + i1 * comp_nb[1] + i0 *
                comp_nb[0]);
              result = *(int64_t*)((char*)tensor_data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] +
                i0 * tensor->nb[0]);
            }
            else { std::cerr << "Results check not implemented for type " << v_type_name(tensor->type) << std::endl; }
          }
          else {
            std::cerr << "Missing debug code for type " << v_type_name(tensor->type) << std::endl;
            v_ABORT("fatal error");
          }

          if ((std::isnan(correct) != std::isnan(result)) || (std::isinf(correct) != std::isinf(result)) || !
            buffer_size_fit) {
            std::cerr << "ERROR: Invalid value in " << v_op_name(tensor->op) << " i3=" << i3 << " i2=" << i2 <<
              " i1=" << i1 << " i0=" << i0 << " result=" << result << " correct=" << correct << " avg_err=" << (avg_err
                / counter)
              << std::endl;
            std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " <<
              v_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor
              ->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" <<
              tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
            if (src0 != nullptr) {
              std::cerr << "src0=" << src0 << " src0->name=" << src0->name << " op=" << v_op_name(src0->op) <<
                " type=" << v_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" <<
                src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" <<
                src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::endl;
            }
            if (src1 != nullptr) {
              std::cerr << "src1=" << src1 << " src1->name=" << src1->name << " op=" << v_op_name(src1->op) <<
                " type=" << v_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" <<
                src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" <<
                src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::endl;
            }
            if (src2 != nullptr) {
              std::cerr << "src2=" << src2 << " src2->name=" << src2->name << " op=" << v_op_name(src2->op) <<
                " type=" << v_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" <<
                src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" <<
                src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::endl;
            }
            if (src3 != nullptr) {
              std::cerr << "src3=" << src3 << " src3->name=" << src3->name << " op=" << v_op_name(src3->op) <<
                " type=" << v_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" <<
                src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" <<
                src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::endl;
            }
            //std::cerr << "First error: result=" << first_error_result << " correct=" << first_error_correct << " i3=" <<
            //  first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] <<
            //  std::endl;
            std::cerr << std::endl << "Result:" << std::endl;
            v_vk_print_tensor_area(tensor, tensor_data, i0, i1, i2, i3);
            //std::cerr << std::endl << "Correct:" << std::endl//;
            //v_vk_print_tensor_area(tensor, comp_result, i0, i1, i2, i3);
            //std::cerr << std::endl;
            std::vector<const v_tensor*> done;
            v_vk_print_graph_origin(tensor, done);
            v_ABORT("fatal error");
          }
          const double denom = std::fabs(correct) > 1.0f
                                 ? (std::fabs(correct) > 1e-8
                                      ? std::fabs(correct)
                                      : 1e-8)
                                 : 1.0f;
          if (first_error[0] == -1 && std::fabs(correct - result) / denom > 0.5) {
            first_error[0]      = i0;
            first_error[1]      = i1;
            first_error[2]      = i2;
            first_error[3]      = i3;
            first_error_result  = result;
            first_error_correct = correct;
          }

          // Special case, value is infinite, avoid NaN result in avg_err
          // NaN also appears in results, if both are nan error is 0
          if (!std::isinf(correct) && !std::isinf(result) && !std::isnan(correct) && !std::isnan(result)) { avg_err += std::fabs(correct - result) / denom; }
          counter++;
        }
      }
    }
  }

  avg_err /= counter;

  if (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
    ///std::cerr << "TENSOR CHECK: avg_err=" << avg_err << " in " << v_op_name(tensor->op) << " (check " <<
    ///  check_counter << ")" << std::endl;
    ///std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " <<
    ///  v_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1]
    ///  << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] <<
    ///  " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
    if (src0 != nullptr) {
      std::cerr << "src0=" << src0 << " op=" << v_op_name(src0->op) << " type=" << v_type_name(src0->type) <<
        " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2="
        << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" <<
        src0->view_offs << std::endl;
    }
    if (src1 != nullptr) {
      std::cerr << "src1=" << src1 << " op=" << v_op_name(src1->op) << " type=" << v_type_name(src1->type) <<
        " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2="
        << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" <<
        src1->view_offs << std::endl;
    }
    if (src2 != nullptr) {
      std::cerr << "src2=" << src2 << " op=" << v_op_name(src2->op) << " type=" << v_type_name(src2->type) <<
        " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2="
        << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" <<
        src2->view_offs << std::endl;
    }
    if (src3 != nullptr) {
      std::cerr << "src3=" << src3 << " op=" << v_op_name(src3->op) << " type=" << v_type_name(src3->type) <<
        " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2="
        << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" <<
        src3->view_offs << std::endl;
    }
    //std::cerr << "First error: result=" << first_error_result << " correct=" << first_error_correct << " i3=" <<
    //  first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
    //std::cerr << std::endl << "Result:" << std::endl;
    v_vk_print_tensor_area(tensor, tensor_data, 5, 5, 0, 0);
    //std::cerr << std::endl << "Correct:" << std::endl;
    //v_vk_print_tensor_area(tensor, comp_result, 5, 5, 0, 0);
    std::cerr << std::endl;
    std::vector<const v_tensor*> done;
    v_vk_print_graph_origin(tensor, done);
  }

  if (avg_err > 0.5 || std::isnan(avg_err)) {
    ///std::cerr << "ERROR: avg_err=" << avg_err << " in " << v_op_name(tensor->op) << " (check " << check_counter <<
    ///  ")" << std::endl;
    ///std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " <<
    ///  v_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1]
    ///  << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] <<
    ///  " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
    if (src0 != nullptr) {
      std::cerr << "src0=" << src0 << " op=" << v_op_name(src0->op) << " type=" << v_type_name(src0->type) <<
        " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2="
        << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" <<
        src0->view_offs << std::endl;
    }
    if (src1 != nullptr) {
      std::cerr << "src1=" << src1 << " op=" << v_op_name(src1->op) << " type=" << v_type_name(src1->type) <<
        " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2="
        << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" <<
        src1->view_offs << std::endl;
    }
    if (src2 != nullptr) {
      std::cerr << "src2=" << src2 << " op=" << v_op_name(src2->op) << " type=" << v_type_name(src2->type) <<
        " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2="
        << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" <<
        src2->view_offs << std::endl;
    }
    if (src3 != nullptr) {
      std::cerr << "src3=" << src3 << " op=" << v_op_name(src3->op) << " type=" << v_type_name(src3->type) <<
        " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2="
        << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" <<
        src3->view_offs << std::endl;
    }
    //std::cerr << "First error: result=" << first_error_result << " correct=" << first_error_correct << " i3=" <<
    //  first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
    //std::cerr << std::endl << "Result:" << std::endl;
    v_vk_print_tensor_area(tensor, tensor_data, first_error[0], first_error[1], first_error[2], first_error[3]);
    //std::cerr << std::endl << "Correct:" << std::endl;
    //v_vk_print_tensor_area(tensor, comp_result, first_error[0], first_error[1], first_error[2], first_error[3]);
    std::cerr << std::endl;
    std::vector<const v_tensor*> done;
    //v_vk_print_graph_origin(tensor, done);
    //v_ABORT("fatal error");
  }
  else {
    std::cerr << check_counter << " " << tensor->name << " op=" << v_op_name(tensor->op) << " avg_err=" << avg_err <<
      std::endl;
  }

  free(comp_result);
  comp_result = nullptr;
  comp_size   = 0;

  if (v_backend_buffer_is_vk(tensor->buffer)) { free(tensor_data); }

  VK_LOG_DEBUG("END v_vk_check_results_1(" << tensor->name << ")");
}
#endif


#endif //MYPROJECT_MML_VK_TEST_H
