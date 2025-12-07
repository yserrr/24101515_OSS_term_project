#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <vector>
#include "../vk_context.hpp"

class VkMemoryAllocator;
struct TextureMeta;
struct Allocation;

struct StreamingBlock{
  VkBuffer buffer     = VK_NULL_HANDLE;
  VkDeviceSize offset = 0;       // start place
  VkDeviceSize size   = 0;       // aligned size
  void* ptr           = nullptr; // persistent mapped ptr
};

struct streamingRegion{
  VkDeviceSize offset = 0;
  VkDeviceSize size   = 0;
  uint64_t fenceValue = 0;
};

enum class MinAlignType:uint32_t{
  UNIFORM,
  STORAGE,
  TEXEL,
  IMAGE,
};

namespace GPU
{
  class VkStreamingBuffer{
  public:
    VkStreamingBuffer(VkContext* pCtxt);
    ~VkStreamingBuffer();
    bool create(const char* debugName = "streaming buffer");
    void destroy();
    StreamingBlock acquire(VkDeviceSize size, VkDeviceSize alignment = 256);
    void map(const void* data, VkDeviceSize dstOffset, VkDeviceSize size);
    void flush(const StreamingBlock& block, VkDeviceSize relOffset, VkDeviceSize size) const;
    void flush(const StreamingBlock& block) const;
    void markInflight(const StreamingBlock& block, uint64_t fenceValue);
    void releaseCompleted(uint64_t completedFenceValue);
    void setAlign(VkPhysicalDeviceProperties properties);
  private:
    VkDeviceSize alignUp(VkDeviceSize v, VkDeviceSize a) const;
    void* ptrAt(VkDeviceSize absOffset) const;
    bool hasFreeSpace(VkDeviceSize need, VkDeviceSize alignment) const;
    VkDevice device_;
    VkMemoryAllocator* pAllocator_;
    VkBuffer streamingBuffer_ = VK_NULL_HANDLE;
    Allocation* allocation_   = nullptr;
    VkDeviceSize capacity_    = 0;

    std::vector<streamingRegion> regions;
    VkDeviceSize head_ = 0;
    VkDeviceSize tail_ = 0;
    void* mapped_      = nullptr;
  };
}
