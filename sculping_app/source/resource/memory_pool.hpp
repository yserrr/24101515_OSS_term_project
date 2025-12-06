#ifndef MEMORYPOOL_HPP
#define MEMORYPOOL_HPP
#include<vulkan/vulkan.h>
#include <algorithm>
#include <vector>
#include <map>
#include <stdexcept>

enum class AllocationType{
  GENERIC,
  BUFFER,
  IMAGE
};

struct Allocation{
  VkDeviceMemory memory    = VK_NULL_HANDLE;
  VkDeviceSize offset      = 0;
  VkDeviceSize size        = 0;
  uint32_t memoryTypeIndex = 0;
  AllocationType type   = AllocationType::GENERIC;
  std::string debugName = "not allocated";
};

class MemoryPool{
  friend class MemoryAllocator;

public:
  MemoryPool(VkDevice device,
             uint32_t memoryTypeIndex,
             VkDeviceSize size);
  ~MemoryPool();

  bool allocate(VkDeviceSize size,
                VkDeviceSize alignment,
                Allocation &out);

  void free(VkDeviceSize offset, VkDeviceSize size);
  uint32_t getMemoryTypeIndex() const;
  VkDeviceMemory getMemory() const;
  void *map();
private:
  struct FreeBlock{
    VkDeviceSize offset;
    VkDeviceSize size;
  };
  bool mapped = false;
  VkDevice device;
  VkDeviceMemory memory;
  uint32_t memoryTypeIndex;
  VkDeviceSize totalSize;
  std::vector<FreeBlock> freeBlocks;

  void mergeFreeBlocks();
};

#endif //MEMORYPOOL_HPP