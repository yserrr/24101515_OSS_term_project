#include "../gpu_context.hpp"
#include "../io/io.hpp"
#include "vk_context.hpp"

extern mns::IoSystem io__;
#define IMPLE_VULKAN
#ifdef IMPLE_VULKAN
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include "vk_shader_pool.hpp"
namespace gpu
{
  VkContext* ctx__ = new VkContext();
  std::unique_ptr<IShader> iShd__ = std::make_unique<VkShaderPool>();
  constexpr uint32_t POLYGON_MODE_FILL = 0;
  constexpr uint32_t POLYGON_MODE_LINE = 1;
  constexpr uint32_t POLYGON_MODE_POINT = 2;
  void cmdSetViewports(CommandBuffer cmd,
                       float x,
                       float y,
                       float width,
                       float height)
  {
    viewport viewport{};
    viewport.x = x;
    viewport.y = y;
    viewport.width = width;
    viewport.height = height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor;
    scissor.offset = {
      static_cast<int>(x),
      static_cast<int>(y)
    };

    scissor.extent = {
      static_cast<uint32_t>(width),
      static_cast<uint32_t>(height)
    };

    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
  }

  void cmdSetViewports(
    CommandBuffer cmd,
    float x,
    float y,
    float width,
    float height,
    float rectXs,
    float rectYs,
    float rectWidths,
    float rectHeights)
  {
    viewport viewport{};
    viewport.x = x;
    viewport.y = y;
    viewport.width = width;
    viewport.height = height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor;
    scissor.offset = {
      static_cast<int>(rectXs),
      static_cast<int>(rectYs)
    };

    scissor.extent = {
      static_cast<uint32_t>(rectWidths),
      static_cast<uint32_t>(rectHeights)
    };

    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
  }

  void cmdBeginRendering(CommandBuffer cmd, RenderPass* pass)
  {
    gpu::RenderingInfo renderingInfo{
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
      .layerCount = 1,
    };
    renderingInfo.colorAttachmentCount = pass->passParameter__.colorAttachment__.size();
    renderingInfo.pColorAttachments = pass->passParameter__.colorAttachment__.data();
    if (pass->passParameter__.depthAttachment__.has_value())
    {
      renderingInfo.pDepthAttachment = &pass->passParameter__.depthAttachment__.value();
    }
    if (pass->passParameter__.stencilAttachment__.has_value())
    {
      renderingInfo.pStencilAttachment = &pass->passParameter__.stencilAttachment__.value();
    }
    renderingInfo.renderArea = {
      0,
      0,
      gpu::ctx__->pSwapChainContext->extent__.width,
      gpu::ctx__->pSwapChainContext->extent__.height
    };
    vkCmdBeginRendering(cmd, &renderingInfo);
  }


  void cmdDrawQuad(CommandBuffer cmd)
  {
    vkCmdDraw(cmd, 6, 1, 0, 0);
  }

  void cmdRect(CommandBuffer cmd)
  {
  }

  void cmdView(CommandBuffer cmd, float x, float y, int width, int height)
  {
  }
}
#endif


//
//   GPU::PassId test                            = ctx__->graphBuilder->addPass(pass);
//   GPU::NodeId swapchain                       = ctx__.graphBuilder->buildSwapchainImage();
//   std::unique_ptr<GPU::VkGraphicsImage> image = std::unique_ptr<GPU::VkGraphicsImage>(new GPU::VkGraphicsImage());
//   image->aspectMask__                         = VK_IMAGE_ASPECT_COLOR_BIT;
//   image->format__                             = VK_FORMAT_R8G8B8A8_UNORM;
//   image->aspectMask__                         = VK_IMAGE_ASPECT_COLOR_BIT;
//   image->format__                             = VK_FORMAT_R8G8B8A8_UNORM;
//   image->nodeName_                            = "color image";
//   image->usage_                               = GPU::ResourceUsage::G_BUFFER;
//   image->type_                                = GPU::ResourceType::IMAGE;
//   image->mSpace_                              = GPU::MemorySpace::DEVICE_LOCAL;
//   image->height_                              = ctx__.pSwapChainContext->extent__.height;
//   image->width__                              = ctx__.pSwapChainContext->extent__.width;
//   GPU::NodeId color                           = ctx__.graphBuilder->registerImage(image);
//   ctx__.graphBuilder->addWriteResource(test, swapchain);
//   ctx__.graphBuilder->addReadResource(test, color);
//   GPU::Scheduler scheduler(&ctx__);
//   while (!glfwWindowShouldClose(ctx__.windowh__))
//   {
//     //glfwPollEvents();
//     scheduler.graphicsRun();
//   }
// }
//    vkCmdEndRenderPass(command);
//summitQueue(command, imageIndex_);
//currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
//    VkCommandBuffer command = rec(imageIndex_);
//for (uint32_t i = 0; i < uIRenderer->callStack_.size(); i++)
//{
//  UICall call = uIRenderer->callStack_.back();
//  spdlog::info("call stack {}", call.path.c_str());
//  if (call.type == CallType::Mesh)
//  {
//    resourceManager_->uploadMesh(command, call.path);
//    uIRenderer->callStack_.pop_back();
//    if (EventManager_->singleModel == true)
//    {
//      std::vector<std::string> trashes;
//      for (auto &mesh: resourceManager_->meshes_)
//      {
//        if (!mesh.second->selected)
//        {
//          mesh.second.reset();
//          trashes.push_back(mesh.first);
//        }
//      }
//      for (auto key: trashes)
//      {
//        resourceManager_->meshes_.erase(key);
//      }
//    }
//  }
//  if (call.type == CallType::Texture)
//  {
//    resourceManager_->uploadTexture(command, call.path);
//    uIRenderer->callStack_.pop_back();
//    uIRenderer->uploadImageToUI();
//  }
//}
//if (EventManager_->actor_->shoudAct)
//{
//  EventManager_->actor_->act(command);
//}
//
//inFlightFences->wait(currentFrame);
//inFlightFences->reset(currentFrame);
//VkSemaphore semaphore = imageAvailableSemaphores->get(currentFrame);
//VkResult result       = vkAcquireNextImageKHR(GPU::ctx__.deviceh__,
//                                        GPU::ctx__.pSwapChainContext->swapchain__,
//                                        UINT64_MAX,
//                                        semaphore,
//                                        VK_NULL_HANDLE,
//                                        &(GPU::ctx__.renderingContext.currentSwapchainIndex__));
//imageIndex_             = GPU::ctx__.renderingContext.currentSwapchainIndex__;
//VkCommandBuffer command = rec(imageIndex_);
//
//
//
//
//    VkSemaphore semaphore = imageAvailableSemaphores->get(currentFrame);
//VkResult result       = vkAcquireNextImageKHR(GPU::ctx__.deviceh__,
//                                        GPU::ctx__.pSwapChainContext->swapchain__,
//                                        UINT64_MAX,
//                                        semaphore,
//                                        VK_NULL_HANDLE,
//                                        &(GPU::ctx__.renderingContext.currentSwapchainIndex__));
//imageIndex_             = GPU::ctx__.renderingContext.currentSwapchainIndex__;
//
//
//
//vkCmdBindDescriptorSets(command,
//                        VK_PIPELINE_BIND_POINT_GRAPHICS,
//                        pipeline_layout_h,
//                        1,
//                        1,
//                        &resourceManager_->bindlessDescirptor_,
//
//                        0,
//                        nullptr);
//vkCmdBindDescriptorSets(command,
//                        VK_PIPELINE_BIND_POINT_GRAPHICS,
//                        pipeline_layout_h,
//                        0,
//                        1,
//                        &resourceManager_->currentCamBuffer[currentFrame].descriptorSet,
//                        0,
//                        nullptr);
//
//vkCmdBeginRenderPass(command, &renderPassInfos[imageIndex_], VK_SUBPASS_CONTENTS_INLINE); // RenderPass 시작
//
//Renderer->draw(command, currentFrame);
//vkCmdNextSubpass(command, VK_SUBPASS_CONTENTS_INLINE);
//
//
//
//
//
//
//
//
//
//
