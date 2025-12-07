#include "vk_context.hpp"
#include "vk_common.hpp"
#include "vk_command_buffer.hpp"


gpu::VkCommandBufferPool::VkCommandBufferPool(VkContext* context)
  : pCtxt__(context)
{
  VkCommandPoolCreateInfo ci{
    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    .queueFamilyIndex = context->graphicsFamailyIdx__,
  };

  VK_ASSERT(vkCreateCommandPool(pCtxt__->deviceh__,
    &ci,
    nullptr,
    &commandPool__));

  commandBuffers__.resize(context->renderingContext.maxInflight__);
  VkCommandBufferAllocateInfo allocInfo{
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .commandPool = commandPool__,
    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandBufferCount = context->renderingContext.maxInflight__,
  };

  if (vkAllocateCommandBuffers(pCtxt__->deviceh__,
                               &allocInfo,
                               commandBuffers__.data()) != VK_SUCCESS)
  {
    throw std::runtime_error("failed to allocate command buffers!");
  }
}

gpu::VkCommandBufferPool::~VkCommandBufferPool()
{
  if (commandPool__ != VK_NULL_HANDLE)
  {
    vkFreeCommandBuffers(pCtxt__->deviceh__,
                         commandPool__,
                         static_cast<uint32_t>(commandBuffers__.size()),
                         commandBuffers__.data());
  }

  if (commandPool__)
  {
    vkDestroyCommandPool(pCtxt__->deviceh__,
                         commandPool__,
                         nullptr);
  }
}
