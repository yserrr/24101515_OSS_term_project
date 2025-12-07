//
// Created by dlwog on 25. 10. 23..
//

#ifndef MYPROJECT_VK_PASS_HPP
#define MYPROJECT_VK_PASS_HPP
#include "vk_resource.hpp"
#include "vk_context.hpp"
#include "../flag.hpp"

namespace gpu
{
  class VkPass
  {
    public:
    std::string name;
    VkPass();
    void clear();
    std::unordered_set<VkPass*> dependency__ = {};
    std::unordered_set<VkPass*> dependent__ = {};
    RenderPassType passType;
    std::vector<VkResource*> read__;
    std::vector<VkResource*> write__;
    std::function<void(VkCommandBuffer cmd)> execute = nullptr;
    std::optional<VkViewport> setViewPort_;
    std::optional<VkClearValue> clearValue_;
    std::optional<VkPolygonMode> polygonMode_;
    std::vector<VkImageMemoryBarrier> imageMemoryBarriers__;
    std::vector<VkBufferMemoryBarrier> bufferBarriers__;
    uint32_t dependencyLevel;

    struct
    {
      std::vector<VkRenderingAttachmentInfo> colorAttachment__;
      std::optional<VkRenderingAttachmentInfo> depthAttachment__;
      std::optional<VkRenderingAttachmentInfo> stencilAttachment__;
      std::optional<VkViewport> viewport__;
      VkBool32 writen__;
      VkExtent2D renderingArea{};
      VkRect2D scissor;
      VkBool32 useDepthTest;
      VkPolygonMode polygonMode = VK_POLYGON_MODE_FILL;
      PFN_vkCmdSetPolygonModeEXT vkCmdSetPolygonModeEXT;
      VkClearColorValue clearColor__;
    } passParameter__;

    void link();
    friend class VkGraphBuilder;
    friend class VkGraphBuilder;
    friend class VkResourceAllocator;
    friend class VkGraph;
    uint32_t linkCount = 0;

    uint32_t passId__ = 0;
    bool culled = false;
    bool transitionPass = false;
  };
}

#endif //MYPROJECT_VK_PASS_HPP
