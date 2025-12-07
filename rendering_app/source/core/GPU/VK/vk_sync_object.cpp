#include "vk_sync_object.hpp"

#include "vk_context.hpp"


gpu::VkSemaphorePool::VkSemaphorePool(VkContext* pCtxt) :
  pCtxt(pCtxt),
  MAX_FRAMES_IN_FLIGHT__(pCtxt->renderingContext.maxInflight__),
  semaphores__(pCtxt->renderingContext.maxInflight__ )
{
  VkSemaphoreCreateInfo createInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT__; ++i)
  {
    if (vkCreateSemaphore(pCtxt->deviceh__
                          , &createInfo,
                          nullptr,
                          &semaphores__[i]) != VK_SUCCESS)
    {
      throw std::runtime_error("failed to create semaphore for frame " + std::to_string(i));
    }
  }
}


gpu::VkSemaphorePool::~VkSemaphorePool()
{
  for (VkSemaphore semaphore : semaphores__)
    if (semaphore != VK_NULL_HANDLE)
      vkDestroySemaphore(pCtxt->deviceh__,
                         semaphore,
                         nullptr);
}


void gpu::VkSemaphorePool::recreate()
{
  for (VkSemaphore& semaphore : semaphores__)
  {
    if (semaphore != VK_NULL_HANDLE)
    {
      vkDestroySemaphore(pCtxt->deviceh__,
                         semaphore,
                         nullptr);
    }
  }
  VkSemaphoreCreateInfo createInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  for (VkSemaphore& semaphore : semaphores__)
  {
    if (vkCreateSemaphore(pCtxt->deviceh__,
                          &createInfo,
                          nullptr,
                          &semaphore) != VK_SUCCESS)
    {
      throw std::runtime_error("failed to create semaphore for frame ");
    }
  }
}


gpu::VkFencePool::VkFencePool(VkContext* pCtxt, bool signaled) :
  pCtxt(pCtxt),
  MAX_FRAMES_IN_FLIGHT__(pCtxt->renderingContext.maxInflight__ ),
  fences(pCtxt->renderingContext.maxInflight__ )
{
  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  if (signaled) fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  for (VkFence& fence : fences)
  {
    if (vkCreateFence(pCtxt->deviceh__,
                      &fenceInfo,
                      nullptr,
                      &fence) != VK_SUCCESS)
    {
      throw std::runtime_error("failed to create fence");
    }
  }
  //spdlog::info("create fence");
}

gpu::VkFencePool::~VkFencePool()
{
  for (VkFence& fence : fences)
    if (fence != VK_NULL_HANDLE)
    {
      vkDestroyFence(pCtxt->deviceh__,
                     fence,
                     nullptr);
    }
}


void gpu::VkFencePool::recreate()
{
  for (VkFence& fence : fences)
  {
    if (VK_VALID(fence)) vkDestroyFence(pCtxt->deviceh__, fence, nullptr);
  }
  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  for (VkFence& fence : fences)
  {
    if (vkCreateFence(pCtxt->deviceh__,
                      &fenceInfo,
                      nullptr,
                      &fence) != VK_SUCCESS)
    {
      throw std::runtime_error("failed to create fence");
    }
  }
  spdlog::info("recreate fence");
}
