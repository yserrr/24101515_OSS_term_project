//
// Created by ljh on 25. 9. 16..
//

#include "buffer_builder.hpp"
#include "common.hpp"
#include <assert.h>

BufferBuilder::BufferBuilder(
    MemoryAllocator &allocator,
    BufferType type,
    AccessPolicy policy
  )
  : device_(allocator.getDevice()),
    allocator_(allocator),
    type_(type),
    policy_(policy) {}

BufferBuilder::~BufferBuilder()
{
  for (auto &buffers: allocatedBuffers)
  {
    buffers.second.reset();
  }
}

BufferContext BufferBuilder::buildBuffer(VkDeviceSize size, const char *debugName)
{
  decidePolicy();
  VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  ci.size        = size;
  ci.usage       = usageFromType(type_);
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  std::unique_ptr<StaticBuffer> buffer = std::make_unique<StaticBuffer>(allocator_,size ,BufferType::UNIFORM);
  buffer->createUniformBuffer();
  BufferContext context{};
  context.buffer =  buffer.get();
  allocatedBuffers[*buffer->getBuffer()] = std::move(buffer);
  return context;
}

void BufferBuilder::destroy(VkBuffer Buffer)
{
  //todo: find hash tabel -> reset
}

void BufferBuilder::decidePolicy()
{
  if (policy_ == AccessPolicy::Auto)
  {
    policy_ = (type_ == BufferType::UNIFORM) ? AccessPolicy::HostPreferred : AccessPolicy::DeviceLocal;
  }
}

VkBufferUsageFlags BufferBuilder::usageFromType(BufferType t) const
{
  switch (t)
  {
    case BufferType::VERTEX: return VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    case BufferType::INDEX: return VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    case BufferType::STORAGE: return VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    case BufferType::UNIFORM: return VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    default: return VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  }
}

BufferType BufferBuilder::type() const
{
  return type_;
}