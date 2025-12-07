//
// Created by ljh on 25. 9. 16..
//

#include "vk_memory_pool.hpp"
#include "spdlog/spdlog.h"
#include "vk_common.hpp"

namespace gpu
{

  VkMemoryPool::VkMemoryPool(VkDevice device,
                             uint32_t memoryTypeIndex,
                             VkDeviceSize size)
    :
    device(device),
    memoryTypeIndex(memoryTypeIndex),
    totalSize(size)
  {
    VkMemoryAllocateInfo allocInfo{
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = size,
      .memoryTypeIndex = memoryTypeIndex,
    };
    if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
    {
      std::cerr << "vkAllocateMemory failed: " << vkAllocateMemory(device, &allocInfo, nullptr, &memory) << std::endl;
      throw std::runtime_error("Failed to allocate memory pool");
    }
    freeBlocks.push_back({0, size});
    spdlog::info("create memory Pool");
  }

  VkMemoryPool::~VkMemoryPool()
  {
    if (memory != VK_NULL_HANDLE)
    {
      vkFreeMemory(device, memory, nullptr);
      spdlog::info("terminate memory pool");
    }
  }

  bool VkMemoryPool::allocate(VkDeviceSize size, VkDeviceSize alignment, VkAllocation& out)
  {
    {
      for (size_t i = 0; i < freeBlocks.size(); ++i)
      {
        VkDeviceSize alignedOffset = (freeBlocks[i].offset + alignment - 1) & ~(alignment - 1);
        VkDeviceSize padding = alignedOffset - freeBlocks[i].offset;
        if (freeBlocks[i].size >= size + padding)
        {
          out.memory__ = memory;
          out.offset__ = alignedOffset;
          out.size = size;
          VkDeviceSize newOffset = alignedOffset + size;
          VkDeviceSize remaining = (freeBlocks[i].offset + freeBlocks[i].size) - newOffset;
          if (remaining > 0)
          {
            freeBlocks[i] = {newOffset, remaining};
          }
          else
          {
            freeBlocks.erase(freeBlocks.begin() + i);
          }
          return true;
        }
      }
      return false;
    }
  }

  void VkMemoryPool::free(VkDeviceSize offset, VkDeviceSize size)
  {
    freeBlocks.push_back({offset, size});
    mergeFreeBlocks();
  }

  uint32_t VkMemoryPool::getMemoryTypeIndex() const
  {
    return memoryTypeIndex;
  }

  VkDeviceMemory VkMemoryPool::getMemory() const
  {
    return memory;
  }

  void VkMemoryPool::mergeFreeBlocks()
  {
    std::sort(freeBlocks.begin(),
              freeBlocks.end(),
              [](const auto& a, const auto& b)
              {
                return a.offset < b.offset;
              });
    for (size_t i = 0; i + 1 < freeBlocks.size();)
    {
      if (freeBlocks[i].offset + freeBlocks[i].size == freeBlocks[i + 1].offset)
      {
        freeBlocks[i].size += freeBlocks[i + 1].size;
        freeBlocks.erase(freeBlocks.begin() + i + 1);
      }
      else
      {
        ++i;
      }
    }
  }
}
