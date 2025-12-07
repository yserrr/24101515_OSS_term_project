

#ifndef MYPROJECT_RENDER_GRAPH_LEGACY_HPP
#define MYPROJECT_RENDER_GRAPH_LEGACY_HPP
#include <vulkan/vulkan.h>
#include <vector>

//by using dynamic rendering, all most don't use
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
  VkRenderPass simpleForwardPass_;
  VkFormat colorFormat;
};



#endif //MYPROJECT_RENDER_GRAPH_LEGACY_HPP