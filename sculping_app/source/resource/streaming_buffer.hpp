#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <vector>
#include <common.hpp>

///todo: alignment setting
/// minUniformBufferOffsetAlignment
/// minStorageBufferOffsetAlignment
/// minTexelBufferOffsetAlignment
/// minMemoryMapAlignment
/// optimalBufferCopyOffsetAlignment
/// optimalBufferCopyRowPitchAlignment
/// nonCoherentAtomSize

class MemoryAllocator;
struct TextureMeta;
struct Allocation;

struct StreamingBlock{
  VkBuffer buffer     = VK_NULL_HANDLE;
  VkDeviceSize offset = 0;       // start place
  VkDeviceSize size   = 0;       // aligned size
  void *ptr           = nullptr; // persistent mapped ptr
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

class StreamingBuffer{
public:
  StreamingBuffer(MemoryAllocator* pAllocator, VkDeviceSize capacity);
  ~StreamingBuffer();
  bool create(const char *debugName = "streaming buffer");
  void destroy();

  StreamingBlock acquire(VkDeviceSize size, VkDeviceSize alignment = 256);
  StreamingBlock acquire(VkDeviceSize size, MinAlignType align);
  void map(const void *data, VkDeviceSize dstOffset, VkDeviceSize size);
  void flush(const StreamingBlock &block, VkDeviceSize relOffset, VkDeviceSize size) const;
  void flush(const StreamingBlock &block) const;

  void recordCopyToBuffer(
      VkCommandBuffer command,
      const StreamingBlock &src,
      VkBuffer destination,
      VkDeviceSize dstOffset
    ) const;

  void recordCopyToImage(
      VkCommandBuffer command,
      const StreamingBlock &src,
      const VkBufferImageCopy &region,
      VkImage dstImage,
      VkImageLayout imageLayout
    ) const;

  void recordCopyToImage(
      VkCommandBuffer command,
      const StreamingBlock &src,
      const VkBufferImageCopy *region,
      uint32_t regionCount,
      VkImage dstImage,
      VkImageLayout imageLayout
    ) const;

  // if need to use multi queue  , write src queue index explicit
  void recordBufferBarrier(
      VkCommandBuffer commandBuffer,
      VkAccessFlags srcAccessMask,
      VkAccessFlags dstAccessMask,
      VkPipelineStageFlagBits srcStage,
      VkPipelineStageFlagBits dstStage,
      VkBuffer dstBuffer,
      VkDeviceSize dstOffset,
      VkDeviceSize dstSize   = VK_WHOLE_SIZE,
      uint32_t srcQueueIndex = VK_QUEUE_FAMILY_IGNORED,
      uint32_t dstQueueIndex = VK_QUEUE_FAMILY_IGNORED
    );

  void recordImageBarrier(
      VkCommandBuffer commandBuffer,
      VkImage dstImage,
      VkImageSubresourceRange subresourceRange,
      VkAccessFlags srcAccessMask,
      VkAccessFlags dstAccessMask,
      VkImageLayout oldLayout,
      VkImageLayout newLayout,
      VkPipelineStageFlags srcStageMask,
      VkPipelineStageFlags dstStageMask,
      uint32_t srcQueueIndex = VK_QUEUE_FAMILY_IGNORED,
      uint32_t dstQueueIndex = VK_QUEUE_FAMILY_IGNORED
    );

  void markInflight(const StreamingBlock &block, uint64_t fenceValue);
  void releaseCompleted(uint64_t completedFenceValue); // garbage correct and free stage
  void setCoherentFlag(bool coherent);

  void setAlign(VkPhysicalDeviceProperties properties);
  VkBuffer getBuffer() const;
  VkDeviceSize getCapacity() const;

private:
  VkDevice device() const;
  VkDeviceSize alignUp(VkDeviceSize v, VkDeviceSize a) const;
  void *ptrAt(VkDeviceSize absOffset) const;
  bool hasFreeSpace(VkDeviceSize need, VkDeviceSize alignment) const;

private: //ALIGNMENT
  VkDeviceSize minUniformAlign_;
  VkDeviceSize minStorageAlign_;
  VkDeviceSize minTexelAlign_;

private:
  VkDevice device_;
  MemoryAllocator* pAllocator_;
  VkBuffer streaming_     = VK_NULL_HANDLE;
  Allocation *allocation_ = nullptr;
  VkDeviceSize capacity_  = 0;

private:
  std::vector<streamingRegion> regions;
  VkDeviceSize head_ = 0;
  VkDeviceSize tail_ = 0;
  void *mapped_      = nullptr;
  bool coherent_flag;
};