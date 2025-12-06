#include "common.hpp"
#ifndef FRAMEBUFFER_HPP
#define FRAMEBUFFER_HPP

struct FramePoolCreateInfo{
  VkDevice                        device;
  VkRenderPass                    renderPass;
  const std::vector<VkImageView> *imageViews;
  const std::vector<VkImageView> *depthViews;
  VkExtent2D                      extent;
};

class FramebufferPool{
public:
  FramebufferPool(FramePoolCreateInfo info);
  ~FramebufferPool();
  VkFramebuffer get(uint32_t imageIndex);

private:
  VkDevice                   device;
  std::vector<VkFramebuffer> framebuffers;
};

#endif