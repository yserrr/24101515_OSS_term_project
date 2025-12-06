#include <streaming_buffer.hpp>
#include <memory_pool.hpp>
#include <memory_allocator.hpp>
#include <cstring>

constexpr VkDeviceSize baseCapacity = 1024 * 1024 * 1024;

StreamingBuffer::StreamingBuffer(MemoryAllocator *pAllocator, VkDeviceSize capacity)
  : pAllocator_(pAllocator), capacity_(capacity), coherent_flag(false)
{
  device_ = pAllocator->getDevice();
  if (baseCapacity > capacity_)
  {
    //base capacity => 1G
    capacity_ = baseCapacity;
  }
}

StreamingBuffer::~StreamingBuffer()
{
  destroy();
}

bool StreamingBuffer::create(const char *debugName)
{
  if (streaming_) return true; //already create
  VkBufferCreateInfo info{};
  info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  info.size        = capacity_;
  info.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT; //staing buffer
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(device_, &info, nullptr, &streaming_) != VK_SUCCESS)
  {
    throw std::runtime_error("failed to create buffer!");
  }
  allocation_ = new Allocation(); // 파괴 시 destroyBuffer에 필요
  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device_, streaming_, &memRequirements);
  VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  *allocation_ = pAllocator_->allocate(memRequirements, properties);
  if (vkBindBufferMemory(device_, streaming_, allocation_->memory, allocation_->offset) != VK_SUCCESS)
  {
    spdlog::info("falil to bind buffer to memory");
  }
  VkMemoryPropertyFlags desired = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  mapped_ = allocation_->maped;
  head_   = tail_ = 0;
  regions.clear();
  return true;
}

void StreamingBuffer::destroy()
{
  if (mapped_)
  {
    mapped_ = nullptr;
  }
  if (streaming_)
  {
    assert(allocation_!=nullptr);
    pAllocator_->destroyBuffer(streaming_, *allocation_);
    streaming_ = VK_NULL_HANDLE;
  }
  if (allocation_)
  {
    delete allocation_;
    allocation_ = nullptr;
  }
  regions.clear();
  head_ = tail_ = 0;
}

StreamingBlock StreamingBuffer::acquire(VkDeviceSize size, VkDeviceSize alignment)
{
  assert(streaming_ && mapped_ && "create() 먼저 호출");
  VkDeviceSize aligned = alignUp(head_, alignment);
  if (aligned >= capacity_) aligned -= capacity_;
  if (alignment == 0)
  {
    aligned = size;
  }
  if (!hasFreeSpace(size, alignment))
  {
    assert(false && "StagingPool 공간 부족: releaseCompleted 호출 필요");
  }

  StreamingBlock block{};
  if (aligned + size <= capacity_)
  {
    //not fulled
    spdlog::info("staging not fulled");
    block.buffer = streaming_;
    block.offset = aligned;
    block.size   = size;
    block.ptr    = ptrAt(block.offset);
    head_        = aligned + size;
  } else
  {
    // 래핑: 끝으로는 못 넣으니 0부터
    aligned      = 0;
    block.buffer = streaming_;
    block.offset = aligned;
    block.size   = size;
    block.ptr    = ptrAt(block.offset);
    head_        = size;
  }
  if (head_ == tail_)
  {
    //ring overflow case::
    //need to aseert
  }
  return block;
}

StreamingBlock StreamingBuffer::acquire(VkDeviceSize size, MinAlignType align)
{
  switch (align)
  {
    case(MinAlignType::UNIFORM):
    {
      return acquire(size, minUniformAlign_);
    }
    case (MinAlignType::IMAGE):
    {
      return acquire(size, 0);
    }
    default:
      return acquire(size, 0);
  }
}

void StreamingBuffer::map(const void *data, VkDeviceSize dstOffset, VkDeviceSize size)
{
  assert(mapped_!= nullptr);
  assert(streaming_&& "create 먼저");
  spdlog::info("maped:{} , alloc offset: {} , size:{}", mapped_, allocation_->offset, allocation_->size);
  void *offset = ptrAt(dstOffset);
  spdlog::info("offset ptr = {:p}", offset);
  std::memcpy(offset, data, (size_t) size);
}

void StreamingBuffer::flush(const StreamingBlock &block, VkDeviceSize relativeOffset, VkDeviceSize size) const
{
  //coherent no need to flush
  // todo: coherent_flag runtime check
  if (coherent_flag) return;
  VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};

  //start + subset start + relative(subset)
  range.memory = allocation_->memory;
  range.offset = allocation_->offset + block.offset + relativeOffset;
  range.size   = size;
  vkFlushMappedMemoryRanges(device(), 1, &range);
}

auto StreamingBuffer::flush(
    const StreamingBlock &block
  ) const -> void
// todo: coherent_flag runtime check
{
  //  coherent no need to flush
  //  ReSharper disable once CppDFAConstantConditions
  if (coherent_flag) return;
  VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
  //start + subset start + relative(subset)
  range.memory = allocation_->memory;
  range.offset = allocation_->offset + block.offset;
  range.size   = block.size;
  vkFlushMappedMemoryRanges(device(), 1, &range);
}

void StreamingBuffer::recordCopyToBuffer(
    VkCommandBuffer command,
    const StreamingBlock &src,
    VkBuffer destination,
    VkDeviceSize dstOffset
  ) const
{
  VkBufferCopy region{};
  region.srcOffset = src.offset;
  region.dstOffset = dstOffset;
  region.size      = src.size;
  vkCmdCopyBuffer(command, streaming_, destination, 1, &region);
}

void StreamingBuffer::recordCopyToImage(
    VkCommandBuffer command,
    const StreamingBlock &src,
    const VkBufferImageCopy &region,
    VkImage dstImage,
    VkImageLayout imageLayout
  ) const
{
  vkCmdCopyBufferToImage(command, src.buffer, dstImage, imageLayout, 1, &region);
}

void StreamingBuffer::recordCopyToImage(VkCommandBuffer command,
                                        const StreamingBlock &src,
                                        const VkBufferImageCopy *region,
                                        uint32_t regionCount,
                                        VkImage dstImage,
                                        VkImageLayout imageLayout) const
{
  vkCmdCopyBufferToImage(command, src.buffer, dstImage, imageLayout, regionCount, region);
}

/// todo :
///  if need -> dependency flag setting
void StreamingBuffer::recordBufferBarrier(
    VkCommandBuffer commandBuffer,
    VkAccessFlags srcAccessMask,
    VkAccessFlags dstAccessMask,
    VkPipelineStageFlagBits srcStage,
    VkPipelineStageFlagBits dstStage,
    VkBuffer dstBuffer,
    VkDeviceSize dstOffset,
    VkDeviceSize dstSize,
    uint32_t srcQueueIndex,
    uint32_t dstQueueIndex
  )
{
  VkBufferMemoryBarrier barrier{};
  barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.srcAccessMask       = srcAccessMask;
  barrier.dstAccessMask       = dstAccessMask;
  barrier.srcQueueFamilyIndex = srcQueueIndex;
  barrier.dstQueueFamilyIndex = dstQueueIndex;
  barrier.buffer              = dstBuffer;
  barrier.offset              = dstOffset;
  barrier.size                = dstSize;

  vkCmdPipelineBarrier(
                       commandBuffer,
                       srcStage,
                       dstStage,
                       0,
                       0,
                       nullptr,
                       1,
                       &barrier,
                       0,
                       nullptr
                      );
}

void StreamingBuffer::recordImageBarrier(
    VkCommandBuffer commandBuffer,
    VkImage dstImage,
    VkImageSubresourceRange subresourceRange,
    VkAccessFlags srcAccessMask,
    VkAccessFlags dstAccessMask,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    VkPipelineStageFlags srcStageMask,
    VkPipelineStageFlags dstStageMask,
    uint32_t srcQueueIndex,
    uint32_t dstQueueIndex
  )
{
  VkImageMemoryBarrier imageMemoryBarrier{};
  imageMemoryBarrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  imageMemoryBarrier.srcAccessMask       = srcAccessMask;
  imageMemoryBarrier.dstAccessMask       = dstAccessMask;
  imageMemoryBarrier.oldLayout           = oldLayout;
  imageMemoryBarrier.newLayout           = newLayout;
  imageMemoryBarrier.image               = dstImage;
  imageMemoryBarrier.subresourceRange    = subresourceRange;
  imageMemoryBarrier.srcQueueFamilyIndex = srcQueueIndex;
  imageMemoryBarrier.dstQueueFamilyIndex = dstQueueIndex;

  vkCmdPipelineBarrier(
                       commandBuffer,
                       srcStageMask,
                       dstStageMask,
                       0,
                       0,
                       nullptr,
                       0,
                       nullptr,
                       1,
                       &imageMemoryBarrier);
}

void StreamingBuffer::markInflight(const StreamingBlock &block, uint64_t fenceValue)
{
  streamingRegion pr;
  pr.offset     = block.offset;
  pr.size       = block.size;
  pr.fenceValue = fenceValue;
  regions.push_back(pr);
}

void StreamingBuffer::releaseCompleted(uint64_t completedFenceValue)
{
  VkDeviceSize newTail = tail_;
  size_t eraseCount    = 0;

  for (const auto &pr: regions)
  {
    if (pr.fenceValue <= completedFenceValue)
    {
      newTail = pr.offset + pr.size;
      if (newTail >= capacity_) newTail -= capacity_;
      eraseCount++;
    } else
    {
      break;
    }
  }
  if (eraseCount > 0)
  {
    regions.erase(regions.begin(), regions.begin() + eraseCount);
    tail_ = newTail;
  }
}

void StreamingBuffer::setCoherentFlag(bool coherent)
{
  coherent_flag = coherent;
}

void StreamingBuffer::setAlign(VkPhysicalDeviceProperties properties)
{
  minUniformAlign_ = properties.limits.minUniformBufferOffsetAlignment;
  minStorageAlign_ = properties.limits.minStorageBufferOffsetAlignment;
  minTexelAlign_   = properties.limits.minTexelBufferOffsetAlignment;
}

VkBuffer StreamingBuffer::getBuffer() const
{
  return streaming_;
}

VkDeviceSize StreamingBuffer::getCapacity() const
{
  return capacity_;
}

VkDevice StreamingBuffer::device() const
{
  return device_;
}

VkDeviceSize StreamingBuffer::alignUp(VkDeviceSize v, VkDeviceSize a) const
{
  return (v + (a - 1)) & ~(a - 1);
}

void *StreamingBuffer::ptrAt(VkDeviceSize absOffset) const
{
  return static_cast<std::byte *>(mapped_) + absOffset;
}

bool StreamingBuffer::hasFreeSpace(VkDeviceSize need, VkDeviceSize alignment) const
{
  // free space::(tail + capacity - head - 1) mod capacity
  VkDeviceSize alignedHead = alignUp(head_, alignment);
  if (alignedHead >= capacity_) alignedHead -= capacity_;
  if (alignedHead >= tail_)
  {
    VkDeviceSize right = capacity_ - alignedHead;
    VkDeviceSize left  = tail_;
    return (need <= right) || (need <= left);
  } else
  {
    VkDeviceSize mid = tail_ - alignedHead;
    return need <= mid;
  }
}