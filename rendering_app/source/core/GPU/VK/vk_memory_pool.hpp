#ifndef VK_MEMORYPOOL_HPP
#define VK_MEMORYPOOL_HPP
#include <algorithm>
#include <vector>
#include <map>
#include <stdexcept>
#include <vulkan/vulkan.h>

namespace gpu{
  enum class VkAllocationType{
    GENERIC,
    BUFFER,
    IMAGE,
    SHADER,
  };

  struct VkAllocation{
    VkDeviceMemory memory__ = VK_NULL_HANDLE;
    VkDeviceSize ptr__;
    VkDeviceSize offset__    = 0;
    VkDeviceSize size        = 0;
    uint32_t memoryTypeIndex = 0;
    VkAllocationType type    = VkAllocationType::GENERIC;
    std::string debugName    = "not allocated";
  };

  class VkMemoryPool{
    friend class VkMemoryAllocator;

  public:
    VkMemoryPool(VkDevice device,
                 uint32_t memoryTypeIndex,
                 VkDeviceSize size);
    ~VkMemoryPool();

    bool allocate(VkDeviceSize size,
                  VkDeviceSize alignment,
                  VkAllocation &out);

    void free(VkDeviceSize offset, VkDeviceSize size);
    uint32_t getMemoryTypeIndex() const;
    VkDeviceMemory getMemory() const;

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
}

#endif //MEMORYPOOL_HPP