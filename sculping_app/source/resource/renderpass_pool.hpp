#ifndef RENDERPASS_HPP
#define RENDERPASS_HPP
#include "common.hpp"

struct RenderPassPoolCreateInfo{
  VkDevice device;
  VkFormat colorFormat;
};

struct SubpassInfo{
  std::vector<VkAttachmentReference> colorRefs;
  VkAttachmentReference depthRef;
  VkPipelineBindPoint bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
};

struct RenderPassInfo{
  std::vector<VkAttachmentDescription> attachments;
  std::vector<SubpassInfo> subpasses;
  std::vector<VkSubpassDependency> dependencies;
};

//legacy -> simple pass only ;
class RenderPassPool{
public:
  RenderPassPool(RenderPassPoolCreateInfo &info);
  ~RenderPassPool();
  VkRenderPass buildForwardPass();
  VkRenderPass buildImGuiOnlyPass();

private:
  VkDevice device;
  VkRenderPass sysForwardPass_;
  VkFormat colorFormat;
};

#endif