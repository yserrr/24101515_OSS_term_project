#include <iostream>
#include "vk_buffer.h"

#include "vk_device.hpp"
#include "vk_context.h"
#include "vk_util.hpp"
#include "vk_op_f32.hpp"

v_backend_buffer_type host{
  0,
  nullptr,
  true
};


vk_buffer_struct::~vk_buffer_struct() {
  if (size == 0) {
    return;
  }
  VK_LOG_DEBUG("~vk_buffer_struct(" << buffer << ", " << size << ")");

  device->device.freeMemory(device_memory);
  device->device.destroyBuffer(buffer);
}

vk_sub_buffer v_vk_subbuffer(const vk_backend_ctx* ctx,
                             const vk_buffer& buf,
                             size_t offset) {
  return {buf, offset, v_vk_get_max_buffer_range(ctx, buf, offset)};
}

void vk_device_buffer_memset_tensor(v_backend_buffer_t buffer, v_tensor* tensor, uint8_t value, size_t offset, size_t size) {
  VK_LOG_DEBUG(
    "v_backend_vk_buffer_memset_tensor(" << buffer << ", " << tensor << ", " << value << ", " << offset << ", " <<
    size << ")");
  v_backend_vk_buffer_ctx* buf_ctx = (v_backend_vk_buffer_ctx*)buffer->context;
  vk_buffer buf                    = buf_ctx->dev_buffer;
  uint32_t val32                   = (uint32_t)value * 0x01010101;
  vk_buffer_memset(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, val32, size);
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
  vk_queue_command_pools_clean_up(dst->device);
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
    vk_queue_command_pools_clean_up(src->device);
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

void v_vk_buffer_write(vk_buffer& dst, size_t offset, const void* src, size_t size) {
  VK_LOG_DEBUG("v_vk_buffer_write(" << size << ")");
  v_vk_buffer_write_2d(dst, offset, src, 0, size, 1);
}

void v_vk_buffer_copy_async(vk_context& ctx, vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
  VK_LOG_DEBUG("v_vk_buffer_copy_async(" << size << ")");
  // Make sure both buffers are on same device
  V_ASSERT(src->device == dst->device);

  VkBufferCopy bc{src_offset, dst_offset, size};

  vkCmdCopyBuffer(ctx->s->buffer, (VkBuffer)src->buffer, (VkBuffer)dst->buffer, 1, &bc);
}

void v_vk_buffer_write_async(vk_context subctx, vk_buffer& dst, size_t offset, const void* src, size_t size, bool sync_staging) {
  VK_LOG_DEBUG("v_vk_buffer_write_async(" << size << ")");
  return v_vk_buffer_write_2d_async(subctx, dst, offset, src, size, size, 1, sync_staging);
}

void v_vk_buffer_write_2d(vk_buffer& dst, size_t offset, const void* src, size_t spitch, size_t width, size_t height) {
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
    vk_queue_command_pools_clean_up(dst->device);
  }
}

void v_vk_buffer_write_2d_async(vk_context subctx, vk_buffer& dst, size_t offset, const void* src, size_t spitch, size_t width, size_t height, bool sync_staging) {
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

void* vk_host_malloc(vk_device& device, size_t size) {
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
    auto addr = static_cast<const uint8_t*>(std::get<0>(device->pinned_memory[i]));
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

void v_vk_buffer_write_nc_async(vk_backend_ctx* ctx, vk_context& subctx, vk_buffer& dst, size_t offset, v_tensor* const tensor, bool sync_staging) {
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
  const uint64_t bs      = block_size(type);

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

const char* vk_host_buffer_name(v_backend_buffer_type_t buft) {
  return v_VK_NAME "_Host";
  UNUSED(buft);
}

VkDeviceSize v_vk_get_max_buffer_range(const vk_backend_ctx* ctx, const vk_buffer& buf, const VkDeviceSize offset) {
  const VkDeviceSize range = std::min(VkDeviceSize{buf->size - offset},
                                      VkDeviceSize{ctx->device->properties.limits.maxStorageBufferRange});
  return range;
}

void vk_host_buffer_free(v_backend_buffer_t buffer) {
  VK_LOG_MEMORY("v_backend_vk_host_buffer_free_buffer()");
  vk_host_free(vk_instance.devices[0], buffer->context);
}

bool vk_buffer_cpy_tensor(v_backend_buffer_t buffer, v_tensor* const src, v_tensor* dst) {
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

  UNUSED(buffer);
}

void vk_get_host_buffer(vk_device& device,
                        const void* ptr,
                        vk_buffer& buf,
                        size_t& buf_offset) {
  std::lock_guard<std::recursive_mutex> guard(device->mutex);
  buf        = nullptr;
  buf_offset = 0;
  for (size_t i = 0; i < device->pinned_memory.size(); i++) {
    const uint8_t* addr = (const uint8_t*)std::get<0>(device->pinned_memory[i]);
    const uint8_t* endr = addr + std::get<1>(device->pinned_memory[i]);
    if (ptr >= addr && ptr < endr) {
      buf        = std::get<2>(device->pinned_memory[i]);
      buf_offset = ((const uint8_t*)ptr) - addr;
      break;
    }
  }
}

vk_buffer vk_create_buffer(vk_device& device, size_t size,
                           const std::initializer_list<vk::MemoryPropertyFlags>& req_flags_list) {
  VK_LOG_DEBUG(
    "v_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags_list.begin()[0]) << ", " <<
    to_string(req_flags_list.begin()[req_flags_list.size()-1]) << ")");
  if (size > device->max_buffer_size) {
    throw vk::OutOfDeviceMemoryError("Requested buffer size exceeds device buffer size limit");
  }

  vk_buffer buf = std::make_shared<vk_buffer_struct>();

  if (size == 0) {
    buf->size = 0;
    return buf;
  }

  vk::BufferUsageFlags usage_flags = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
    vk::BufferUsageFlagBits::eTransferDst;
  vk::MemoryAllocateFlags mem_flags{};
  if (device->buffer_device_address) {
    usage_flags |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
    mem_flags |= vk::MemoryAllocateFlagBits::eDeviceAddress;
  }
  vk::BufferCreateInfo buffer_create_info{
    vk::BufferCreateFlags(),
    size,
    usage_flags,
    vk::SharingMode::eExclusive,
    0,
    nullptr,
  };
  buf->buffer                                  = device->device.createBuffer(buffer_create_info);
  vk::MemoryRequirements mem_req               = device->device.getBufferMemoryRequirements(buf->buffer);
  vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();
  const vk::MemoryAllocateFlagsInfo mem_flags_info{mem_flags};
  for (auto it = req_flags_list.begin(); it != req_flags_list.end(); it++) {
    const auto& req_flags      = *it;
    uint32_t memory_type_index = find_properties(&mem_props, &mem_req, req_flags);
    if (memory_type_index == UINT32_MAX) {
      continue;
    }
    buf->memory_property_flags = req_flags;
    try {
      buf->device_memory = device->device.allocateMemory({mem_req.size, memory_type_index, &mem_flags_info});
      break;
    }
    catch (const vk::SystemError& e) {
      // loop and retry
      // during last attempt throw the exception
      if (it + 1 == req_flags_list.end()) {
        device->device.destroyBuffer(buf->buffer);
        throw e;
      }
    }
  }

  if (!buf->device_memory) {
    device->device.destroyBuffer(buf->buffer);
    throw vk::OutOfDeviceMemoryError("No suitable memory type found");
  }

  buf->ptr = nullptr;

  if (buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
    buf->ptr = device->device.mapMemory(buf->device_memory, 0, VK_WHOLE_SIZE);
  }
  device->device.bindBufferMemory(buf->buffer, buf->device_memory, 0);
  buf->device = device;
  buf->size   = size;
  if (device->buffer_device_address) {
    const vk::BufferDeviceAddressInfo addressInfo(buf->buffer);
    buf->bda_addr = device->device.getBufferAddress(addressInfo);
  }

  #ifdef v_VULKAN_MEMORY_DEBUG
  device->memory_logger->log_allocation(buf, size);
  #endif

  return buf;
}

vk_buffer vk_create_buffer_check(vk_device& device,
                                 size_t size,
                                 vk::MemoryPropertyFlags req_flags,
                                 vk::MemoryPropertyFlags fallback_flags) {
  try {
    return vk_create_buffer(device, size, {req_flags, fallback_flags});
  }
  catch (const vk::SystemError& e) {
    std::cerr << "mml_vulkan: Memory allocation of size " << size << " failed." << std::endl;
    std::cerr << "mml_vulkan: " << e.what() << std::endl;
    throw e;
  }
}

void vk_ensure_sync_staging_buffer(vk_device& device, size_t size) {
  if (device->sync_staging == nullptr ||
    size > device->sync_staging->size) {
    VK_LOG_MEMORY("v_vk_ensure_sync_staging_buffer(" << size << ")");
    vk_destroy_buffer(device->sync_staging);
    device->sync_staging = vk_create_buffer_check(device, size,
                                                  vk::MemoryPropertyFlagBits::eHostVisible |
                                                  vk::MemoryPropertyFlagBits::eHostCoherent |
                                                  vk::MemoryPropertyFlagBits::eHostCached,
                                                  vk::MemoryPropertyFlagBits::eHostVisible |
                                                  vk::MemoryPropertyFlagBits::eHostCoherent);
  }
}

void vk_buffer_read2d_async(vk_context subctx,
                            vk_buffer& src,
                            size_t offset,
                            void* dst,
                            size_t spitch,
                            size_t dpitch,
                            size_t width,
                            size_t height,
                            bool sync_staging) {
  VK_LOG_DEBUG("v_vk_buffer_read_2d_async(offset=" << offset << ", width=" << width << ", height=" << height << ")");
  V_ASSERT(width > 0);
  V_ASSERT(height > 0);
  V_ASSERT(src != nullptr);
  // TODO: staging_offset is not used_bits__
  // Check if dst is pinned memory
  vk_buffer buf     = nullptr;
  size_t buf_offset = 0;
  vk_get_host_buffer(src->device, dst, buf, buf_offset);
  std::vector<vk::BufferCopy> slices(1);
  if (width == spitch && width == dpitch) {
    // Only do single write if stride is equal
    slices[0].srcOffset = offset;
    slices[0].dstOffset = buf_offset;
    slices[0].size      = width * height;
  }
  else {
    slices.resize(height);
    for (size_t i = 0; i < height; i++) {
      slices[i].srcOffset = offset + i * spitch;
      slices[i].dstOffset = buf_offset + i * dpitch;
      slices[i].size      = width;
    }
  }
  if (buf != nullptr) {
    // Memory is pinned, use as staging buffer
    vk_sync_buffers(nullptr, subctx);
    subctx->s->buffer.copyBuffer(src->buffer, buf->buffer, slices);
    return;
  }
  VK_LOG_DEBUG("STAGING");
  if (!sync_staging) {
    v_ABORT("Asynchronous read from non-pinned memory not supported");
  }
  // Fall back to staging buffer
  const size_t copy_size = dpitch * height;
  vk_ensure_sync_staging_buffer(src->device, copy_size);
  vk_buffer& staging_buffer = src->device->sync_staging;
  vk_sync_buffers(nullptr, subctx);
  subctx->s->buffer.copyBuffer(src->buffer, staging_buffer->buffer, slices);
  vk_deffered_memcpy(dst, staging_buffer->ptr, copy_size, &subctx->out_memcpys);
}

void vk_deffered_memcpy(void* dst, const void* src, size_t size, std::vector<vk_staging_memcpy>* memcpys) {
  if (memcpys == nullptr) {
    memcpy(dst, src, size);
  }
  else {
    memcpys->emplace_back(dst, src, size);
  }
}

void vk_deffered_memset(void* dst, uint32_t val, size_t size, std::vector<vk_staging_memset>* memsets) {
  if (memsets == nullptr) {
    memset(dst, val, size);
  }
  else {
    memsets->emplace_back(dst, val, size);
  }
}

void vk_buffer_read_async(vk_context subctx, vk_buffer& src, size_t offset, void* dst, size_t size, bool sync_staging) { return vk_buffer_read2d_async(subctx, src, offset, dst, size, size, size, 1, sync_staging); }


void vk_destroy_buffer(vk_buffer& buf) {
  if (buf == nullptr) { return; }

  #ifdef v_VULKAN_MEMORY_DEBUG
  if (buf->device != nullptr) { buf->device->memory_logger->log_deallocation(buf); }
  #endif

  buf.reset();
}

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

v_backend_buffer_type_t vk_device_buffer_type(size_t dev_num) {
  vk_instance_init();
  VK_LOG_DEBUG("v_backend_vk_buffer_type(" << dev_num << ")");
  vk_device dev = v_vk_get_device(dev_num);
  return &dev->buffer_type;
}

v_backend_buffer_t vk_host_buffer_alloc(v_backend_buffer_type_t buft, size_t size) {
  VK_LOG_MEMORY("v_backend_vk_host_buffer_type_alloc_buffer(" << size << ")");
  size += 32; // Behave like the CPU buffer type
  void* ptr = nullptr;
  try { ptr = vk_host_malloc(vk_instance.devices[0], size); }
  catch (vk::SystemError& e) {
    v_LOG_WARN("v_vulkan: Failed to allocate pinned memory (%s)\n", e.what());
    throw std::runtime_error("fail to alloc host mem");
  }
  v_backend_buffer* buffer = v_backend_cpu_buffer_from_ptr(ptr, size);
  buffer->buft             = buft;
  buffer->buft->host       = true;
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

v_backend_buffer_type_t vk_host_buffer_type() {
  // Make sure device 0 is initialized
  vk_instance_init();
  v_vk_get_device(0);
  return &host;
}

void vk_read_buffer(vk_buffer& src,
                    size_t offset,
                    void* dst,
                    size_t size) {
  VK_LOG_DEBUG("mml vk buffer Read  buffer H:  " << src->buffer << " offset: , " << offset << ", size:" << size << ")");
  if (src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible && src->device->uma) {
    V_ASSERT(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);
    memcpy(dst, (uint8_t*)src->ptr + offset, size);
  }
  else {
    std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
    vk_context subctx = vk_create_temp_ctx(src->device->transfer_queue.cmd_pool);
    vk_begin_ctx(src->device, subctx);
    vk_buffer_read_async(subctx, src, offset, dst, size, true);
    vk_ctx_end(subctx);
    vk_submit(subctx, src->device->fence);
    VK_CHECK(src->device->device.waitForFences({ src->device->fence },
               true,
               UINT64_MAX),
             "vk_buffer_read waitForFences");
    src->device->device.resetFences({src->device->fence});
    vk_queue_command_pools_clean_up(src->device);
    for (auto& cpy : subctx->out_memcpys) {
      memcpy(cpy.dst, cpy.src, cpy.n);
    }
  }
}
