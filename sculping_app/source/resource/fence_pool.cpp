#include <fence_pool.hpp>

FencePool::FencePool(SignalCreateInfo info, bool signaled)
  : device(info.device),
    MAX_FRAMES_IN_FLIGHT(info.MAX_FRAMES_IN_FLIGHT),
    fences(info.MAX_FRAMES_IN_FLIGHT)
{
  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  if (signaled) fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  for (VkFence &fence: fences)
  {
    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS)
    {
      throw std::runtime_error("failed to create fence");
    }
  }
  spdlog::info("create fence");
}

FencePool::~FencePool()
{
  for (VkFence &fence: fences)
    if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, nullptr);
}

VkFence FencePool::get(uint32_t currentFrame) const
{
  return fences[currentFrame];
}

void FencePool::wait(uint32_t currentFrame, uint64_t timeout)
{
  vkWaitForFences(device, 1, &fences[currentFrame], VK_TRUE, timeout);
}

void FencePool::reset(uint32_t currentFrame)
{
  vkResetFences(device, 1, &fences[currentFrame]);
}

void FencePool::recreate() {
  for (VkFence& fence : fences)
  {
    if (VK_VALID(fence)) vkDestroyFence(device, fence, nullptr);
  }
  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  for (VkFence& fence : fences)
  {
    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS)
    {
      throw std::runtime_error("failed to create fence");
    }
  }
  spdlog::info("recreate fence");
}
