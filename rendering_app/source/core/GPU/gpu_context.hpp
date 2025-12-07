#ifndef MYPROJECT_CONTEXT_HPP
#define MYPROJECT_CONTEXT_HPP
#include "unique.hpp"
#include "IShader.hpp"

namespace gpu
{
  class IDescriptorAllocator;
  class ICommandBuffer;
  class IPipeline;

  class Context
  {
    public:
    virtual ~Context() = default;
    virtual void loadContext() =0;

    private:
    IDescriptorAllocator* descriptor_;
    ICommandBuffer* commandBuffer_;
    IPipeline* pipeline_;
  };

  extern std::unique_ptr<IShader> iShd__;
}


#ifdef DX12
// todo: imple #include
#endif

#define IMPLE_VULKAN
#ifdef IMPLE_VULKAN
#include "imgui_impl_glfw.h"
#include "flag.hpp"
#include "vk_scheduler.hpp"
#include "vk_vertex_attr.hpp"
#include "back/imgui_impl_vulkan.h"
#include "vk_pass.hpp"
#include "vk_texture.hpp"

namespace gpu
{
  extern VkContext* ctx__;
  using FrameAttachment = VkFrameAttachment;
  using SwapchainHandle = uint32_t;
  using CommandBuffer = VkCommandBuffer;
  using DescriptorSet = VkDescriptorSet;
  using Extent = VkExtent2D;
  using PolygonMode = VkPolygonMode;
  using MeshBuffer = VkMeshBuffer;
  using Scheduler = VkScheduler;
  using RenderPass = VkPass;
  using RenderNode = VkResource;
  using ImageHanle = VkFrameAttachment;
  using GPUBuffer = VkHostBuffer;
  using RenderingAttachment = VkRenderingAttachmentInfo;
  using PipelineLayout = VkPipelineLayout;
  using DescriptorSetLayout = VkDescriptorSetLayout;
  using Pipeline = VkPipeline;
  using PipelineCache = VkPipelineCache;
  using PipelineLayout = VkPipelineLayout;
  using RenderingInfo = VkRenderingInfo;
  using viewport = VkViewport;
  using ShaderObject = VkShaderModule;
  using RenderTarget = uint32_t;
  using Texture = VkTexture;
  using PipelineProgram = VkPipelineProgram;
  struct GraphicsPipelineObject
  {
    Pipeline pipeline;
    PipelineLayout pipelineLayout;
    DescriptorSetLayout descriptorLayout;
    ShaderObject vertexShader;
    ShaderObject fragmentShader;
  };

  using VertexBindingDescriptor = VkVertexInputBindingDescription;
  using VertexAttribute = VkVertexInputAttributeDescription;
  constexpr uint32_t FORMAT_R8_UNORM = VK_FORMAT_R8_UNORM;
  constexpr uint32_t FORMAT_R16_UNORM = VK_FORMAT_R16_UNORM;

  constexpr uint32_t FORMAT_R8G8_UNORM = 16;
  constexpr uint32_t FORMAT_R8G8_SNORM = 17;
  constexpr uint32_t FORMAT_R8G8_USCALED = 18;
  constexpr uint32_t FORMAT_R8G8_SSCALED = 19;
  constexpr uint32_t FORMAT_R8G8_UINT = 20;
  constexpr uint32_t FORMAT_R8G8_SINT = 21;
  constexpr uint32_t FORMAT_R8G8_SRGB = 22;
  constexpr uint32_t FORMAT_R8G8B8_UNORM = 23;
  constexpr uint32_t FORMAT_R8G8B8_SNORM = 24;
  constexpr uint32_t FORMAT_R8G8B8_USCALED = 25;
  constexpr uint32_t FORMAT_R8G8B8_SSCALED = 26;
  constexpr uint32_t FORMAT_R8G8B8_UINT = 27;
  constexpr uint32_t FORMAT_R8G8B8_SINT = 28;
  constexpr uint32_t FORMAT_R8G8B8_SRGB = 29;
  constexpr uint32_t FORMAT_B8G8R8_UNORM = 30;
  constexpr uint32_t FORMAT_B8G8R8_SNORM = 31;
  constexpr uint32_t FORMAT_B8G8R8_USCALED = 32;
  constexpr uint32_t FORMAT_B8G8R8_SSCALED = 33;
  constexpr uint32_t FORMAT_B8G8R8_UINT = 34;
  constexpr uint32_t FORMAT_B8G8R8_SINT = 35;
  constexpr uint32_t FORMAT_B8G8R8_SRGB = 36;
  constexpr uint32_t FORMAT_R8G8B8A8_UNORM = 37;
  constexpr uint32_t FORMAT_R8G8B8A8_SNORM = 38;
  constexpr uint32_t FORMAT_R8G8B8A8_USCALED = 39;
  constexpr uint32_t FORMAT_R8G8B8A8_SSCALED = 40;
  constexpr uint32_t FORMAT_R8G8B8A8_UINT = 41;
  constexpr uint32_t FORMAT_R8G8B8A8_SINT = 42;
  constexpr uint32_t FORMAT_R8G8B8A8_SRGB = 43;
  constexpr uint32_t FORMAT_B8G8R8A8_UNORM = 44;
  constexpr uint32_t FORMAT_B8G8R8A8_SNORM = 45;
  constexpr uint32_t FORMAT_B8G8R8A8_USCALED = 46;
  constexpr uint32_t FORMAT_B8G8R8A8_SSCALED = 47;
  constexpr uint32_t FORMAT_B8G8R8A8_UINT = 48;
  constexpr uint32_t FORMAT_B8G8R8A8_SINT = 49;
  constexpr uint32_t FORMAT_B8G8R8A8_SRGB = 50;
  constexpr uint32_t FORMAT_D32_SFLOAT = 126;

  constexpr uint32_t FORMAT_R16_SNORM = 71;
  constexpr uint32_t FORMAT_R16_USCALED = 72;
  constexpr uint32_t FORMAT_R16_SSCALED = 73;
  constexpr uint32_t FORMAT_R16_UINT = 74;
  constexpr uint32_t FORMAT_R16_SINT = 75;
  constexpr uint32_t FORMAT_R16_SFLOAT = 76;
  constexpr uint32_t FORMAT_R16G16_UNORM = 77;
  constexpr uint32_t FORMAT_R16G16_SNORM = 78;
  constexpr uint32_t FORMAT_R16G16_USCALED = 79;
  constexpr uint32_t FORMAT_R16G16_SSCALED = 80;
  constexpr uint32_t FORMAT_R16G16_UINT = 81;
  constexpr uint32_t FORMAT_R16G16_SINT = 82;
  constexpr uint32_t FORMAT_R16G16_SFLOAT = 83;
  constexpr uint32_t FORMAT_R16G16B16_UNORM = 84;
  constexpr uint32_t FORMAT_R16G16B16_SNORM = 85;
  constexpr uint32_t FORMAT_R16G16B16_USCALED = 86;
  constexpr uint32_t FORMAT_R16G16B16_SSCALED = 87;
  constexpr uint32_t FORMAT_R16G16B16_UINT = 88;
  constexpr uint32_t FORMAT_R16G16B16_SINT = 89;
  constexpr uint32_t FORMAT_R16G16B16_SFLOAT = 90;
  constexpr uint32_t FORMAT_R16G16B16A16_UNORM = 91;
  constexpr uint32_t FORMAT_R16G16B16A16_SNORM = 92;
  constexpr uint32_t FORMAT_R16G16B16A16_USCALED = 93;
  constexpr uint32_t FORMAT_R16G16B16A16_SSCALED = 94;
  constexpr uint32_t FORMAT_R16G16B16A16_UINT = 95;
  constexpr uint32_t FORMAT_R16G16B16A16_SINT = 96;
  constexpr uint32_t FORMAT_R16G16B16A16_SFLOAT = 97;
  constexpr uint32_t FORMAT_R32_UINT = 98;
  constexpr uint32_t FORMAT_R32_SINT = 99;
  constexpr uint32_t FORMAT_R32_SFLOAT = 100;
  constexpr uint32_t FORMAT_R32G32_UINT = 101;
  constexpr uint32_t FORMAT_R32G32_SINT = 102;
  constexpr uint32_t FORMAT_R32G32_SFLOAT = 103;
  constexpr uint32_t FORMAT_R32G32B32_UINT = 104;
  constexpr uint32_t FORMAT_R32G32B32_SINT = 105;
  constexpr uint32_t FORMAT_R32G32B32_SFLOAT = 106;
  constexpr uint32_t FORMAT_R32G32B32A32_UINT = 107;
  constexpr uint32_t FORMAT_R32G32B32A32_SINT = 108;
  constexpr uint32_t FORMAT_R32G32B32A32_SFLOAT = 109;

  constexpr uint32_t ATTACHMENT_STORE_OP_STORE = 0;
  constexpr uint32_t ATTACHMENT_STORE_OP_DONT_CARE = 1;
  constexpr uint32_t ATTACHMENT_STORE_OP_NONE = 1000301000;
  constexpr uint32_t ATTACHMENT_STORE_OP_NONE_KHR = VK_ATTACHMENT_STORE_OP_NONE;
  constexpr uint32_t ATTACHMENT_STORE_OP_NONE_QCOM = VK_ATTACHMENT_STORE_OP_NONE;
  constexpr uint32_t ATTACHMENT_STORE_OP_NONE_EXT = VK_ATTACHMENT_STORE_OP_NONE;
  constexpr uint32_t ATTACHMENT_STORE_OP_MAX_ENUM = 0x7FFFFFFF;
  void cmdSetViewports(CommandBuffer cmd,
                       float x,
                       float y,
                       float width,
                       float height);
  void cmdBeginRendering(CommandBuffer cmd, RenderPass* pass);
  void cmdDraw(CommandBuffer cmd, uint32_t handle);
  void cmdDrawQuad(CommandBuffer cmd);

  inline auto cmdBindDescriptorSets = vkCmdBindDescriptorSets;
  inline auto cmdBindPipeline = vkCmdBindPipeline;
  inline auto cmdPushConstant = vkCmdPushConstants;

  inline auto cmdDisPatch = vkCmdDispatch;
  inline auto cmdEndRendering = vkCmdEndRendering;
  inline auto cmdSetViewPort = vkCmdSetViewport;
  inline auto cmdCopyBuffer = vkCmdCopyBuffer;
  inline auto cmdCopyImage = vkCmdCopyImage;
  inline auto cmdBlitImage = vkCmdBlitImage;
  inline auto cmdResolveImage = vkCmdResolveImage;
  inline auto cmdCopyBufferToImage = vkCmdCopyBufferToImage;
  inline auto cmdCopyImageToBuffer = vkCmdCopyImageToBuffer;
  inline auto newFrameApiCall = ImGui_ImplVulkan_NewFrame;
  inline auto newFrameSurfaceCall = ImGui_ImplGlfw_NewFrame;
  inline auto cmdDrawUiData = ImGui_ImplVulkan_RenderDrawData;
}
#endif


#endif //MYPROJECT_CONTEXT_HPP
