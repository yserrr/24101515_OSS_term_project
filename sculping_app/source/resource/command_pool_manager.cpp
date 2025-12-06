//
// Created by ljh on 25. 9. 22..
//

#include "command_pool_manager.hpp"

CommandPoolManager::CommandPoolManager(const CommandPoolManagerCreateInfo &info): device(info.device),
                                                                                  commandPool(info.commandPool) {
  commandBuffers.resize(info.frameCount);
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool        = commandPool;
  allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = info.frameCount;
  if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
  {
    throw std::runtime_error("failed to allocate command buffers!");
  }

}

CommandPoolManager::~CommandPoolManager() {
  if (commandPool != VK_NULL_HANDLE)
    vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
}
VkCommandBuffer CommandPoolManager::record(uint32_t imageIndex) {
  if (vkResetCommandBuffer(commandBuffers[imageIndex], 0) != VK_SUCCESS)
    throw std::runtime_error("fail to reset command buffer");
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags            = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  beginInfo.pInheritanceInfo = nullptr;

  if (vkBeginCommandBuffer(commandBuffers[imageIndex], &beginInfo) != VK_SUCCESS)
  {
    throw std::runtime_error("failed to begin recording command buffer!");
  }
  return commandBuffers[imageIndex];
}
void CommandPoolManager::endRecord(uint32_t imageIndex) {
  if (vkEndCommandBuffer(commandBuffers[imageIndex]) != VK_SUCCESS)
  {
    spdlog::error("Command buffer recording failed");
  }
}