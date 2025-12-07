#include "vk_discard_pool.hpp"
#include "vk_context.hpp"

gpu::VkDiscardPool::VkDiscardPool(VkContext* pCtxt) :
  pCtxt_(pCtxt),
  frameDiscardPool_(pCtxt->renderingContext.maxInflight__)
{
}


void gpu::VkDiscardPool::clean()
{
  auto& discardPool = frameDiscardPool_[pCtxt_->renderingContext.currentFrame__];
  for (auto& handle: discardPool)
  {
    handle.deleteFunction();
  }
  discardPool.clear();
}

void gpu::VkDiscardPool::registerResource(std::vector<VkResource*>& frameResource)
{
  auto& discardPool = frameDiscardPool_[pCtxt_->renderingContext.currentFrame__];
  for (auto rec : frameResource)
  {
    DiscardHandle discardHandle;
    discardHandle.handler = rec;
    switch (rec->type_)
    {
      case(ResourceType::IMAGE):
      {
        discardHandle.deleteFunction = [this, rec ]
        {
          VkFrameAttachment* node = reinterpret_cast<VkFrameAttachment*>(rec);
          vkDestroyImageView(pCtxt_->deviceh__,
                             node->imageView__,
                             nullptr);
          vkDestroyImage(pCtxt_->deviceh__,
                         node->imageh__,
                         nullptr);
          VkAllocation mAlloc = node->allocation__;
          pCtxt_->pMemoryAllocator->free(mAlloc, mAlloc.size);
        };
      }
      case(ResourceType::BUFFER):
      {
        discardHandle.deleteFunction = [this, rec ]
        {
          VkHostBuffer* node = reinterpret_cast<VkHostBuffer*>(rec);
          vkDestroyBuffer(pCtxt_->deviceh__,
                          node->bufferh_,
                          nullptr);
          VkAllocation mAlloc = node->allocation__;
          pCtxt_->pMemoryAllocator->free(mAlloc, mAlloc.size);
        };
      }
      default:
      {
        if (discardHandle.deleteFunction == nullptr)
        {
          spdlog::info("need to check handle {}", rec->nodeId_);
        }
        discardPool.push_back(discardHandle);
        break;
      }
    }
  }
}
