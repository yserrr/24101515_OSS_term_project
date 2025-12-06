#define   STB_IMAGE_IMPLEMENTATION 
#include <app.hpp>
#include <spdlog/spdlog.h>
#include "ui_renderer.hpp"
#include "log_sink.hpp"

void App::run()
{
  UIsink = std::make_shared<UILogSink>();
  //spdlog::set_default_logger(std::make_shared<spdlog::logger>("default", UIsink));
  spdlog::set_level(spdlog::level::trace);
  vkDeviceload();
  init();
  uiRenderer->sink_ = UIsink;
  setUp();
  Camera *cam = resourceManager_->getCamera();
  Renderer->setCamera(cam);
  eventManager_ = std::make_unique<EventManager>(window_h,
                                                 cam,
                                                 resourceManager_.get(),
                                                 swapchain->getExtent());
  eventManager_->resourcesManager_ = resourceManager_.get();
  eventManager_->currentExtent     = swapchain->getExtent();
  eventManager_->setRenderer(Renderer.get());
  uiRenderer->sculpting = dynamic_cast<SculptorMode *>(eventManager_->actor_.get());
  vkDeviceWaitIdle(device_h);
  uiRenderer->uploadImageToUI();
  spdlog::info("render set up");
  while (!glfwWindowShouldClose(window_h))
  {
    glfwPollEvents();
    resourceManager_->updateDescriptorSet(currentFrame);
    eventManager_->getKey();
    eventManager_->wheelUpdate();
    eventManager_->getMouseEvent();
    uiRenderer->uploadImageToUI();
    eventManager_->moved = false;
    inFlightFences->wait(currentFrame);
    inFlightFences->reset(currentFrame);
    VkSemaphore semaphore   = imageAvailableSemaphores->get(currentFrame);
    imageIndex_             = swapchain->getImageIndex(semaphore);
    VkCommandBuffer command = rec(imageIndex_);
    for (uint32_t i = 0; i < uiRenderer->callStack_.size(); i++)
    {
      UICall call = uiRenderer->callStack_.back();
      spdlog::info("call stack {}", call.path.c_str());
      if (call.type == CallType::Mesh)
      {
        resourceManager_->uploadMesh(command, call.path);
        uiRenderer->callStack_.pop_back();
        if (eventManager_->singleModel == true)
        {
          std::vector<std::string> trashes;
          for (auto &mesh: resourceManager_->meshes_)
          {
            if (!mesh.second->selected)
            {
              mesh.second.reset();
              trashes.push_back(mesh.first);
            }
          }
          for (auto key: trashes)
          {
            resourceManager_->meshes_.erase(key);
          }
        }
      }
      if (call.type == CallType::Texture)
      {
        resourceManager_->uploadTexture(command, call.path);
        uiRenderer->callStack_.pop_back();
        uiRenderer->uploadImageToUI();
      }
    }
    if (eventManager_->actor_->shoudAct)
    {
      eventManager_->actor_->act(command);
    }
    vkCmdBindDescriptorSets(command,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline_layout_h,
                            1,
                            1,
                            &resourceManager_->bindlessDescirptor_,

                            0,
                            nullptr);
    vkCmdBindDescriptorSets(command,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipeline_layout_h,
                            0,
                            1,
                            &resourceManager_->mainCamBuffers_[currentFrame].descriptorSet,
                            0,
                            nullptr);

    vkCmdBeginRenderPass(command, &renderPassInfos[imageIndex_], VK_SUBPASS_CONTENTS_INLINE); // RenderPass 시작

    Renderer->draw(command, currentFrame);
    vkCmdNextSubpass(command, VK_SUBPASS_CONTENTS_INLINE);

    uiRenderer->rec(command);
    uiRenderer->drawcall(command);
    uiRenderer->drawTransition(command);
    uiRenderer->render(command);
    vkCmdEndRenderPass(command);
    summitQueue(command, imageIndex_);
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
  }
  vkDeviceWaitIdle(device_h); // GPU가 다 끝났다는 보장
  inFlightFences.reset();
  renderFinishedSemaphores.reset();
  imageAvailableSemaphores.reset();
  Renderer.reset();
  resourceManager_.reset();
  allocator.reset();
  eventManager_.reset();
  uiRenderer.reset();
  swapchain.reset();
  imageManager.reset();
  renderPass.reset();
  frameBufferManager.reset();
  commandBufferManager.reset();
  commandPool.reset();
}

void App::setUp()
{
  inFlightFences->wait(currentFrame);
  inFlightFences->reset(currentFrame);
  VkSemaphore semaphore   = imageAvailableSemaphores->get(currentFrame);
  imageIndex_             = swapchain->getImageIndex(semaphore);
  VkCommandBuffer command = rec(imageIndex_);
  resourceManager_->uploadMesh(command, "sphere.gltf");
  resourceManager_->uploadTexture(command, "VkVideo.png");
  resourceManager_->updateDescriptorSet(currentFrame);
  vkCmdBeginRenderPass(command, &renderPassInfos[imageIndex_], VK_SUBPASS_CONTENTS_INLINE); // RenderPass 시작
  Renderer->setUp(command);
  vkCmdNextSubpass(command, VK_SUBPASS_CONTENTS_INLINE);
  uiRenderer->draw(command);
  vkCmdEndRenderPass(command);
  summitQueue(command, imageIndex_);
  currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

VkCommandBuffer App::rec(uint32_t imageIndex)
{
  clearValues[0].color                          = {0.1f, 0.1f, 0.1f, 1.0f};
  clearValues[1].depthStencil                   = {1.0f, 0}; // 깊이 초기화 값
  renderPassInfos[imageIndex].sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfos[imageIndex].renderPass        = renderpass_h;
  renderPassInfos[imageIndex].framebuffer       = frameBufferManager->get(imageIndex);
  renderPassInfos[imageIndex].renderArea.offset = {0, 0};
  renderPassInfos[imageIndex].renderArea.extent = swapchain->getExtent();
  renderPassInfos[imageIndex].clearValueCount   = 2;
  renderPassInfos[imageIndex].pClearValues      = clearValues.data();
  VkCommandBuffer command                       = commandBufferManager->record(imageIndex);
  return command;
}

void App::summitQueue(VkCommandBuffer command, uint32_t imageIndex)
{
  commandBufferManager->endRecord(imageIndex);
  VkSubmitInfo submitInfo{};
  submitInfo.sType                  = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  VkSemaphore waitSemaphores[]      = {imageAvailableSemaphores->get(currentFrame)};
  VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  submitInfo.waitSemaphoreCount     = 1;
  submitInfo.pWaitSemaphores        = waitSemaphores;
  submitInfo.pWaitDstStageMask      = waitStages;
  submitInfo.commandBufferCount     = 1;
  submitInfo.pCommandBuffers        = &command;

  VkSemaphore signalSemaphores[]  = {renderFinishedSemaphores->get(currentFrame)};
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores    = signalSemaphores;
  if (vkQueueSubmit(graphics_q, 1, &submitInfo, inFlightFences->get(currentFrame)) != VK_SUCCESS)
  {
    throw std::runtime_error("failed to submit draw command buffer!");
  }
  VkSwapchainKHR swapchains[] = {swapchain->get()};
  VkPresentInfoKHR presentInfo{};
  presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores    = signalSemaphores;
  presentInfo.swapchainCount     = 1;
  presentInfo.pSwapchains        = swapchains;
  presentInfo.pImageIndices      = &(imageIndex);
  if (vkQueuePresentKHR(present_q, &presentInfo) != VK_SUCCESS)
  {
    throw std::runtime_error("present Queue error");
  }
}

App::~App()
{
  if (device_h != VK_NULL_HANDLE) vkDestroyDevice(device_h, nullptr);
  if (surface_h != VK_NULL_HANDLE) vkDestroySurfaceKHR(instance_h, surface_h, nullptr);
  if (instance_h != VK_NULL_HANDLE) vkDestroyInstance(instance_h, nullptr);
  if (window_h != nullptr)glfwDestroyWindow(window_h);
}