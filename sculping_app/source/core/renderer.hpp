#ifndef RENDERER_HPP
#define RENDERER_HPP
#include <common.hpp>
#include <../../extern/examples/pipeline.hpp>
#include <../resource/renderpass_pool.hpp>
#include <../resource/descriptor_manager.hpp>
#include  <command_pool_manager.hpp>
#include  <command_pool.hpp>
#include  <frame_pool.hpp>
#include <../resource/shader_pool.hpp>
#include  <swapchain_view.hpp>
#include  <swapchain.hpp>
#include <../model/mesh.hpp>
#include <../resource/sampler_builder.hpp>
#include <../resource/texture.hpp>
#include  <resource_pool.hpp>
#include  <semaphore_pool.hpp>
#include  <fence_pool.hpp>
#include "../../extern/examples/renderer_resource.hpp"

struct RenderInitInfo{
  VkDevice device_h;
  ResourceManager *resourceManager;
  MemoryAllocator *allocator;
  VkExtent2D extent;
  Swapchain *swapchain;
  SwapchainViewManager *imageManager;
  VkDescriptorSetLayout *pDescriptorSetLayouts;
  FramebufferPool *frameManager;
  VkRenderPass renderPass;
  uint32_t descriptorSetLayoutCount;
};

enum class ViewMode{
  SINGLE,
  MULTI,
  FPS,
  TPS,
  VR
};

class RenderingSystem{
  friend class App;
  friend class EventManager;

public:
  RenderingSystem(RenderInitInfo info);
  ~RenderingSystem() {}
  void pushConstant(VkCommandBuffer cmdBuffer);
  void setUp(VkCommandBuffer cmd);
  void draw(VkCommandBuffer cmd, uint32_t currentFrame);

  void setCamera(Camera *cameraP)
  {
    camera = cameraP;
  }

private:
  VkPhysicalDevice physical_device_h;
  VkDevice device_h;
  VkQueue graphics_q;
  VkQueue present_q;
  VkSurfaceKHR surface_h;
  VkFormat format;
  VkPipeline backgroundPipeline_;
  VkPipeline pipeline_h;
  VkPipelineLayout pipelineLayout_h;
  VkDescriptorSetLayout *pDescriptorSetLayouts;
  uint32_t descriptorLayoutCount;
  VkRenderPass renderpass_h;
  uint32_t present_family;
  uint32_t graphics_family;

  std::vector<BatchContext> batches_;
  ViewMode viewMode = ViewMode::SINGLE;
  PFN_vkCmdSetPolygonModeEXT vkCmdSetPolygonModeEXT;
  VkPolygonMode polygonMode = VK_POLYGON_MODE_FILL;
  VkBool32 depthTest        = VK_TRUE;
  VkBool32 drawBackground   = VK_TRUE;

  std::string fragPath     = "/home/ljh/CLionProjects/VkMain/source/shader/fragment.frag";
  std::string vertPath     = "/home/ljh/CLionProjects/VkMain/source/shader/vertex.vert";
  std::string fragBackPath = "/home/ljh/CLionProjects/VkMain/source/shader/sculptor_background.frag";
  std::string VertBackPath = "/home/ljh/CLionProjects/VkMain/source/shader/sculptor_background.vert";
  VkViewport viewport{};
  VkRect2D scissor{};
  FramebufferPool *framebufferManager;
  SwapchainViewManager *imageManager;
  Swapchain *swapchain;

  Camera *camera;
  VulkanTexture *texture;
  ResourceManager &pResourceManager;
  MemoryAllocator &allocator;
  VkDeviceSize offsets = 0;
};
#endif