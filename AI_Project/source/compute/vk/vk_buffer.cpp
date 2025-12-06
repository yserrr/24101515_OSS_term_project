#include <iostream>
#include "vk_buffer.h"
#include "vk_device.h"
#include "vk_context.h"
#include "vk_util.h"


vk_buffer_struct::~vk_buffer_struct()
{
  if (size == 0)
  {
    return;
  }
  VK_LOG_DEBUG("~vk_buffer_struct(" << buffer << ", " << size << ")");

  device->device.freeMemory(device_memory);
  device->device.destroyBuffer(buffer);
}

void vk_get_host_buffer(vk_device& device,
                        const void* ptr,
                        vk_buffer& buf,
                        size_t& buf_offset)
{
  std::lock_guard<std::recursive_mutex> guard(device->mutex);
  buf = nullptr;
  buf_offset = 0;
  for (size_t i = 0; i < device->pinned_memory.size(); i++)
  {
    const uint8_t* addr = (const uint8_t*)std::get<0>(device->pinned_memory[i]);
    const uint8_t* endr = addr + std::get<1>(device->pinned_memory[i]);
    if (ptr >= addr && ptr < endr)
    {
      buf = std::get<2>(device->pinned_memory[i]);
      buf_offset = ((const uint8_t*)ptr) - addr;
      break;
    }
  }
}

vk_buffer vk_create_buffer(vk_device& device, size_t size,
                           const std::initializer_list<vk::MemoryPropertyFlags>& req_flags_list)
{
  VK_LOG_DEBUG(
    "v_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags_list.begin()[0]) << ", " <<
    to_string(req_flags_list.begin()[req_flags_list.size()-1]) << ")");
  if (size > device->max_buffer_size)
  {
    throw vk::OutOfDeviceMemoryError("Requested buffer size exceeds device buffer size limit");
  }

  vk_buffer buf = std::make_shared<vk_buffer_struct>();

  if (size == 0)
  {
    buf->size = 0;
    return buf;
  }

  vk::BufferUsageFlags usage_flags = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                                     vk::BufferUsageFlagBits::eTransferDst;
  vk::MemoryAllocateFlags mem_flags{};
  if (device->buffer_device_address)
  {
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
  buf->buffer = device->device.createBuffer(buffer_create_info);
  vk::MemoryRequirements mem_req = device->device.getBufferMemoryRequirements(buf->buffer);
  vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();
  const vk::MemoryAllocateFlagsInfo mem_flags_info{mem_flags};
  for (auto it = req_flags_list.begin(); it != req_flags_list.end(); it++)
  {
    const auto& req_flags = *it;
    uint32_t memory_type_index = find_properties(&mem_props, &mem_req, req_flags);
    if (memory_type_index == UINT32_MAX)
    {
      continue;
    }
    buf->memory_property_flags = req_flags;
    try
    {
      buf->device_memory = device->device.allocateMemory({mem_req.size, memory_type_index, &mem_flags_info});
      break;
    }
    catch (const vk::SystemError& e)
    {
      // loop and retry
      // during last attempt throw the exception
      if (it + 1 == req_flags_list.end())
      {
        device->device.destroyBuffer(buf->buffer);
        throw e;
      }
    }
  }

  if (!buf->device_memory)
  {
    device->device.destroyBuffer(buf->buffer);
    throw vk::OutOfDeviceMemoryError("No suitable memory type found");
  }

  buf->ptr = nullptr;

  if (buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible)
  {
    buf->ptr = device->device.mapMemory(buf->device_memory, 0, VK_WHOLE_SIZE);
  }
  device->device.bindBufferMemory(buf->buffer, buf->device_memory, 0);
  buf->device = device;
  buf->size = size;
  if (device->buffer_device_address)
  {
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
                                vk::MemoryPropertyFlags fallback_flags)
{
  try
  {
    return vk_create_buffer(device, size, {req_flags, fallback_flags});
  }
  catch (const vk::SystemError& e)
  {
    std::cerr << "mml_vulkan: Memory allocation of size " << size << " failed." << std::endl;
    std::cerr << "mml_vulkan: " << e.what() << std::endl;
    throw e;
  }
}

void vk_ensure_sync_staging_buffer(vk_device& device, size_t size)
{
  if (device->sync_staging == nullptr ||
      size > device->sync_staging->size)
  {
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
                                  bool sync_staging)
{
  VK_LOG_DEBUG("v_vk_buffer_read_2d_async(offset=" << offset << ", width=" << width << ", height=" << height << ")");
  V_ASSERT(width > 0);
  V_ASSERT(height > 0);
  V_ASSERT(src != nullptr);
  // TODO: staging_offset is not used
  // Check if dst is pinned memory
  vk_buffer buf = nullptr;
  size_t buf_offset = 0;
  vk_get_host_buffer(src->device, dst, buf, buf_offset);
  std::vector<vk::BufferCopy> slices(1);
  if (width == spitch && width == dpitch)
  {
    // Only do single write if stride is equal
    slices[0].srcOffset = offset;
    slices[0].dstOffset = buf_offset;
    slices[0].size = width * height;
  }
  else
  {
    slices.resize(height);
    for (size_t i = 0; i < height; i++)
    {
      slices[i].srcOffset = offset + i * spitch;
      slices[i].dstOffset = buf_offset + i * dpitch;
      slices[i].size = width;
    }
  }
  if (buf != nullptr)
  {
    // Memory is pinned, use as staging buffer
    vk_sync_buffers(nullptr, subctx);
    subctx->s->buffer.copyBuffer(src->buffer, buf->buffer, slices);
    return;
  }
  VK_LOG_DEBUG("STAGING");
  if (!sync_staging)
  {
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

void vk_deffered_memcpy(void* dst, const void* src, size_t size, std::vector<vk_staging_memcpy>* memcpys)
{
  if (memcpys == nullptr)
  {
    memcpy(dst, src, size);
  }
  else
  {
    memcpys->emplace_back(dst, src, size);
  }
}

void vk_deffered_memset(void* dst, uint32_t val, size_t size, std::vector<vk_staging_memset>* memsets)
{
  if (memsets == nullptr)
  {
    memset(dst, val, size);
  }
  else
  {
    memsets->emplace_back(dst, val, size);
  }
}


void mmlVKBufferRead(vk_buffer& src,
                     size_t offset,
                     void* dst,
                     size_t size)
{
  VK_LOG_DEBUG("mml vk buffer Read  buffer H:  " << src->buffer << " offset: , " << offset << ", size:" << size << ")");
  if (src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible && src->device->uma)
  {
    V_ASSERT(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);
    memcpy(dst, (uint8_t*)src->ptr + offset, size);
  }
  else
  {
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
    mmlVkQueueCommandPoolsCleanUp(src->device);
    for (auto& cpy : subctx->out_memcpys)
    {
      memcpy(cpy.dst, cpy.src, cpy.n);
    }
  }
}
