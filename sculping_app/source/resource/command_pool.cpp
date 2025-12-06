//
// Created by ljh on 25. 9. 15..
//

#include"command_pool.hpp"

// CommandPool
CommandPool::CommandPool(CommandPoolCreateInfo info)
  : device(info.device),
    commandPool(VK_NULL_HANDLE)
{
  VkCommandPoolCreateInfo ci{};
  ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  ci.queueFamilyIndex = info.queueFamilyIndex;
  ci.flags            = info.flags;

  VK_ASSERT(vkCreateCommandPool(device, &ci, nullptr, &commandPool));
}

CommandPool::~CommandPool()
{
  if (commandPool) vkDestroyCommandPool(device, commandPool, nullptr);
}

VkCommandPool CommandPool::get() const {
  return commandPool;
}


void CommandPool::reset()
{
  vkResetCommandPool(device, commandPool, 0);
}