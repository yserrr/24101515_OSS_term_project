#include "../vk_streaming_buffer.hpp"
#include "../vk_memory_pool.hpp"
#include "../vk_memory_allocator.hpp"
#include <cstring>

namespace GPU
{
  VkStreamingBuffer::VkStreamingBuffer(MemoryAllocator* pAllocator, VkDeviceSize capacity)
    : pAllocator_(pAllocator), capacity_(capacity), coherent_flag(false)
  {
    device_ = pAllocator->getDevice();
    if (baseCapacity > capacity_)
    {
      capacity_ = baseCapacity;
    }
  }

  VkStreamingBuffer::~VkStreamingBuffer()
  {
    destroy();
  }

  bool VkStreamingBuffer::create(const char* debugName)
  {
    if (streaming_) return true; //already create
    VkMemoryPropertyFlags desired = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    mapped_                       = allocation_->maped;
    head_                         = tail_ = 0;
    regions.clear();
    return true;
  }

  void VkStreamingBuffer::destroy()
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

  StreamingBlock VkStreamingBuffer::acquire(VkDeviceSize size, VkDeviceSize alignment)
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
    }
    else
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

  void VkStreamingBuffer::map(const void* data, VkDeviceSize dstOffset, VkDeviceSize size)
  {
    assert(mapped_!= nullptr);
    assert(streaming_&& "create 먼저");
    spdlog::info("maped:{} , alloc offset: {} , size:{}", mapped_, allocation_->offset, allocation_->size);
    void* offset = ptrAt(dstOffset);
    spdlog::info("offset ptr = {:p}", offset);
    std::memcpy(offset, data, (size_t)size);
  }

  void VkStreamingBuffer::flush(const StreamingBlock& block, VkDeviceSize relativeOffset, VkDeviceSize size) const
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

  auto VkStreamingBuffer::flush(
    const StreamingBlock& block
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

  void VkStreamingBuffer::markInflight(const StreamingBlock& block, uint64_t fenceValue)
  {
    streamingRegion pr;
    pr.offset     = block.offset;
    pr.size       = block.size;
    pr.fenceValue = fenceValue;
    regions.push_back(pr);
  }

  void VkStreamingBuffer::releaseCompleted(uint64_t completedFenceValue)
  {
    VkDeviceSize newTail = tail_;
    size_t eraseCount    = 0;

    for (const auto& pr : regions)
    {
      if (pr.fenceValue <= completedFenceValue)
      {
        newTail = pr.offset + pr.size;
        if (newTail >= capacity_) newTail -= capacity_;
        eraseCount++;
      }
      else
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
  VkDeviceSize VkStreamingBuffer::alignUp(VkDeviceSize v, VkDeviceSize a) const
  {
    return (v + (a - 1)) & ~(a - 1);
  }

  void* VkStreamingBuffer::ptrAt(VkDeviceSize absOffset) const
  {
    return static_cast<std::byte*>(mapped_) + absOffset;
  }

  bool VkStreamingBuffer::hasFreeSpace(VkDeviceSize need, VkDeviceSize alignment) const
  {
    // free space::(tail + capacity - head - 1) mod capacity
    VkDeviceSize alignedHead = alignUp(head_, alignment);
    if (alignedHead >= capacity_) alignedHead -= capacity_;
    if (alignedHead >= tail_)
    {
      VkDeviceSize right = capacity_ - alignedHead;
      VkDeviceSize left  = tail_;
      return (need <= right) || (need <= left);
    }
    else
    {
      VkDeviceSize mid = tail_ - alignedHead;
      return need <= mid;
    }
  }
}
