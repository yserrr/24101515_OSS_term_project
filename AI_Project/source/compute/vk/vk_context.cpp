#include <memory>
#include "vk_context.h"
#include <vk_util.hpp>
#include "vk_device.hpp"
#include "vk_queue.hpp"
bool vk_instance_initialized = false;
vk_instance_struct vk_instance;
bool vk_perf_logger_enabled = false;

void vk_begin_ctx(vk_device& device, vk_context& subctx) {
  VK_LOG_DEBUG("v_vk_ctx_begin(" << device->name << ")");
  if (subctx->s != nullptr) {
    vk_ctx_end(subctx);
  }

  subctx->seqs.push_back({vk_begin_sub_mission(device, *subctx->p)});
  subctx->s = subctx->seqs[subctx->seqs.size() - 1].data();
}

void vk_ctx_end(vk_context& ctx) {
  VK_LOG_DEBUG("v_vk_ctx_end(" << ctx << ", " << ctx->seqs.size() << ")");
  if (ctx->s == nullptr) {
    return;
  }

  ctx->s->buffer.end();
  ctx->s = nullptr;
}

vk_context vk_create_temp_ctx(vk_command_pool& p) {
  vk_context result = std::make_shared<vk_context_struct>();
  VK_LOG_DEBUG("v_vk_create_temporary_context(" << result << ")");
  result->p = &p;
  return result;
}


vk_context vk_create_context(vk_backend_ctx* ctx, vk_command_pool& p) {
  vk_context result = std::make_shared<vk_context_struct>();
  VK_LOG_DEBUG("v_vk_create_context(" << result << ")");
  ctx->gc.contexts.emplace_back(result);
  result->p = &p;
  return result;
}

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

void vk_ctx_pre_alloc_buffers(vk_backend_ctx* ctx) {
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
  vk_backend_ctx* ctx = backend->context;

  return &ctx->device->buffer_type;
}

void v_backend_vk_set_tensor_async(v_backend_t backend, v_tensor* tensor, const void* data, size_t offset, size_t size) {
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

void v_backend_vk_get_tensor_async(v_backend_t backend, v_tensor* const tensor, void* data, size_t offset, size_t size) {
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

v_status vk_buffer_init_tensor(v_backend_buffer_t buffer, const v_tensor* tensor) {
  V_ASSERT(buffer);
  VK_LOG_DEBUG("v_backend_vk_buffer_init_tensor(" << buffer << " (" << buffer->context << "), " << tensor << ")");
  if (tensor->view_src != nullptr) { V_ASSERT(tensor->view_src->buffer->buft == buffer->buft); }
  return V_STATUS_SUCCESS;
}

void vk_device_buffer_set_tensor(v_backend_buffer_t buffer, v_tensor* tensor, const void* data, size_t offset, size_t size) {
  VK_LOG_DEBUG(
    "v_backend_vk_buffer_set_tensor(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size
    << ")");
  v_backend_vk_buffer_ctx* buf_ctx = (v_backend_vk_buffer_ctx*)buffer->context;
  vk_buffer buf                    = buf_ctx->dev_buffer;
  v_vk_buffer_write(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, data, size);
}

void vk_device_buffer_get_tensor(v_backend_buffer_t buffer, const v_tensor* tensor, void* data, size_t offset, size_t size) {
  VK_LOG_DEBUG(
    "vk_buffer_get_tensor(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size
    << ")");
  v_backend_vk_buffer_ctx* buf_ctx = (v_backend_vk_buffer_ctx*)buffer->context;
  vk_buffer buf                    = buf_ctx->dev_buffer;
  vk_read_buffer(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, data, size);
  VK_LOG_DEBUG(
    "F32 " << ((float*)(buf->ptr))[0] << ", " << ((float*)(buf->ptr))[1] << ", " << data << ", " << offset << ", " <<
    size
    << ")");
}

void vk_buffer_clear(v_backend_buffer_t buffer, uint8_t value) {
  v_backend_vk_buffer_ctx* ctx = (v_backend_vk_buffer_ctx*)buffer->context;
  vk_buffer_memset(ctx->dev_buffer, 0, value, buffer->size);
}

bool v_backend_vk_cpy_tensor_async(v_backend_t backend, v_tensor* const src, v_tensor* dst) {
  VK_LOG_DEBUG("v_backend_vk_cpy_tensor_async()");
  vk_backend_ctx* ctx = backend->context;
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
  vk_backend_ctx* ctx = backend->context;
  if (ctx->transfer_ctx.expired()) { return; }
  vk_context transfer_ctx = ctx->transfer_ctx.lock();
  vk_ctx_end(transfer_ctx);
  for (auto& cpy : transfer_ctx->in_memcpys) memcpy(cpy.dst, cpy.src, cpy.n);
  vk_submit(transfer_ctx, ctx->fence);
  v_vk_wait_for_fence(ctx);
  for (auto& cpy : transfer_ctx->out_memcpys) memcpy(cpy.dst, cpy.src, cpy.n);
  ctx->transfer_ctx.reset();
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

struct vk_device_ctx* v_backend_vk_reg_get_device(size_t index) {
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

  V_ASSERT(index < devices.size());
  return devices[index];
}
