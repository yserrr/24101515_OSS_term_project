//
// Created by ljh on 25. 9. 16..
//
#include "frame_pool.hpp"

FramebufferPool::FramebufferPool(FramePoolCreateInfo info) : device(info.device)
{
  framebuffers.resize(info.imageViews->size());
  //color + depth를 바탕으로 frameBuffer 할당
  for (size_t i = 0; i < info.imageViews->size(); i++)
  {
    std::array<VkImageView, 2> attachments = {info.imageViews->at(i), info.depthViews->at(i)};
    VkFramebufferCreateInfo    framebufferInfo{};
    framebufferInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass      = info.renderPass;
    framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    framebufferInfo.pAttachments    = attachments.data();
    framebufferInfo.width           = info.extent.width;
    framebufferInfo.height          = info.extent.height;
    framebufferInfo.layers          = 1;
    if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers[i]) != VK_SUCCESS)
    {
      throw std::runtime_error("failed to create framebuffer!");
    }
  }
}

FramebufferPool::~FramebufferPool()
{
  for (auto framebuffer: framebuffers)
  {
    if (framebuffer != VK_NULL_HANDLE)
      vkDestroyFramebuffer(device, framebuffer, nullptr);
  }
}

VkFramebuffer FramebufferPool::get(uint32_t imageIndex)
{
  return framebuffers[imageIndex];
}