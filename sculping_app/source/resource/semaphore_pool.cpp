#include "semaphore_pool.hpp"

SemaphorePool::SemaphorePool(SignalCreateInfo &info) :
  device(info.device),
  MAX_FRAMES_IN_FLIGHT(info.MAX_FRAMES_IN_FLIGHT),
  semaphores(info.MAX_FRAMES_IN_FLIGHT)
{
  VkSemaphoreCreateInfo createInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
  {
    if (vkCreateSemaphore(device, &createInfo, nullptr, &semaphores[i]) != VK_SUCCESS)
    {
      throw std::runtime_error("failed to create semaphore for frame " + std::to_string(i));
    }
  }
}

SemaphorePool::~SemaphorePool()
{
  for (VkSemaphore semaphore: semaphores)
    if (semaphore != VK_NULL_HANDLE) vkDestroySemaphore(device, semaphore, nullptr);
}

VkSemaphore SemaphorePool::get(uint32_t currentFrame) const
{
  return semaphores[currentFrame];
}

void SemaphorePool::recreate()
{
  for (VkSemaphore &semaphore: semaphores)
  {
    if (semaphore != VK_NULL_HANDLE) vkDestroySemaphore(device, semaphore, nullptr);
  }
  VkSemaphoreCreateInfo createInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  for (VkSemaphore &semaphore: semaphores)
  {
    if (vkCreateSemaphore(device, &createInfo, nullptr, &semaphore) != VK_SUCCESS)
    {
      throw std::runtime_error("failed to create semaphore for frame ");
    }
  }
}