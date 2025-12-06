//
// Created by ljh on 25. 9. 16..
//

#include "memory_pool.hpp"
#include "spdlog/spdlog.h"
#include "common.hpp"
MemoryPool::MemoryPool(VkDevice     device,
                       uint32_t     memoryTypeIndex,
                       VkDeviceSize size)
  :
  device(device),
  memoryTypeIndex(memoryTypeIndex),
  totalSize(size)
{
  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize  = size;
  allocInfo.memoryTypeIndex = memoryTypeIndex;

  if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to allocate memory pool");
  }
  freeBlocks.push_back({0, size});
  spdlog::info("create memory Pool");
}

MemoryPool::~MemoryPool()
{
  if (memory != VK_NULL_HANDLE)
  {
    vkFreeMemory(device, memory, nullptr);
    spdlog::info("terminate memory pool");
  }
}

bool MemoryPool::allocate(VkDeviceSize size, VkDeviceSize alignment, Allocation &out)
{
  {
    for (size_t i = 0; i < freeBlocks.size(); ++i)
    {
      VkDeviceSize alignedOffset = (freeBlocks[i].offset + alignment - 1) & ~(alignment - 1);
      VkDeviceSize padding       = alignedOffset - freeBlocks[i].offset;
      if (freeBlocks[i].size >= size + padding)
      {
        out.memory             = memory;
        out.offset             = alignedOffset;
        VkDeviceSize newOffset = alignedOffset + size;
        VkDeviceSize remaining = (freeBlocks[i].offset + freeBlocks[i].size) - newOffset;
        if (remaining > 0)
        {
          freeBlocks[i] = {newOffset, remaining};
        } else
        {
          freeBlocks.erase(freeBlocks.begin() + i);
        }
        return true;
      }
    }
    return false;
  }
}

void MemoryPool::free(VkDeviceSize offset, VkDeviceSize size)
{
  freeBlocks.push_back({offset, size});
  mergeFreeBlocks();
}

uint32_t MemoryPool::getMemoryTypeIndex() const
{
  return memoryTypeIndex;
}

VkDeviceMemory MemoryPool::getMemory() const
{
  return memory;
}

void MemoryPool::mergeFreeBlocks()
{
  std::sort(freeBlocks.begin(),
            freeBlocks.end(),
            [](const auto &a, const auto &b)
            {
              return a.offset < b.offset;
            });
  for (size_t i = 0; i + 1 < freeBlocks.size();)
  {
    if (freeBlocks[i].offset + freeBlocks[i].size == freeBlocks[i + 1].offset)
    {
      freeBlocks[i].size += freeBlocks[i + 1].size;
      freeBlocks.erase(freeBlocks.begin() + i + 1);
    } else
    {
      ++i;
    }
  }
}

void *MemoryPool::map()
{
  assert(!mapped);
  void* data;
  VK_ASSERT(vkMapMemory(device, memory, 0,VK_WHOLE_SIZE, 0, &data));
  mapped = true;
}