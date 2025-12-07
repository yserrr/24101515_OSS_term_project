//
// Created by ljh on 25. 10. 7..
//
#include "vk_context.hpp"
#include "vk_scheduler.hpp"
#include "vk_graph_builder.hpp"

gpu::VkScheduler::VkScheduler(gpu::VkContext* context) :
  pCtxt_(context),
  imageAvailiableSemaphorePool_(context),
  renderFinishSemaphorePool_(context),
  maxInflightFence_(context),
  commandBufferPool_(context)
{
  vkReleaseSwapchainImagesEXT = (PFN_vkReleaseSwapchainImagesEXT)vkGetDeviceProcAddr(context->deviceh__,
       "vkReleaseSwapchainImagesEXT");
  if (!vkReleaseSwapchainImagesEXT)
  {
    throw std::runtime_error("Failed to load vkReleaseSwapchainImagesEXT");
  }
  pCtxt_->renderingContext.currentFrame__ = 0;
  pCtxt_->renderingContext.inflightIndex__.resize(pCtxt_->pSwapChainContext->img__.size() + 1);
}

gpu::VkScheduler::~VkScheduler() = default;

VkBool32 gpu::VkScheduler::nextFrame()
{
  VkFence vkFence = maxInflightFence_.fences[pCtxt_->
                                             renderingContext.currentFrame__];

  vkWaitForFences(pCtxt_->deviceh__,
                  1,
                  &vkFence,
                  VK_TRUE,
                  UINT64_MAX);


  VkResult result = vkAcquireNextImageKHR(pCtxt_->deviceh__,
                                          pCtxt_->pSwapChainContext->swapchain__,
                                          UINT64_MAX,
                                          (imageAvailiableSemaphorePool_.semaphores__[pCtxt_->renderingContext.
                                            currentFrame__]),
                                          VK_NULL_HANDLE,
                                          &(pCtxt_->renderingContext.inflightIndex__[pCtxt_->renderingContext.
                                            currentFrame__]));
  vkResetFences(pCtxt_->deviceh__,
                1,
                &vkFence);


  if (result == VK_ERROR_OUT_OF_DATE_KHR)
  {
    pCtxt_->pSwapChainContext->broked__ = true;
    //this->swapchain -> recreate()
    //this->recreateSwapchain__ = VK_TRUE;
    //this->passes_.clear();
  }
  else if ((result != VK_SUCCESS) && (result != VK_SUBOPTIMAL_KHR))
  {
    throw std::runtime_error("Could not acquire the next swap chain image!");
  }
  //physical reosource mapping on table

  pCtxt_->dirty_ = false;
  return result;
}

void gpu::VkScheduler::run(std::vector<VkPass>& passes)
{
  VkCommandBuffer cmd = commandBufferPool_.
    commandBuffers__[pCtxt_->renderingContext.currentFrame__];

  if (vkResetCommandBuffer(cmd, 0) != VK_SUCCESS)
  {
    throw std::runtime_error("fail to reset command buffer");
  }
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  beginInfo.pInheritanceInfo = nullptr;

  if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
  {
    throw std::runtime_error("failed to begin recording command buffer!");
  }
  for (auto pass : gpu::ctx__->transitionPass)
  {
    pass.execute(cmd);
  }
  gpu::ctx__->transitionPass.clear();
  for (auto pass : passes)
  {
    pass.execute(cmd);
  }
  vkEndCommandBuffer(cmd);
  ///for (auto pass: graphCompiler_.waveFrontPasses)
  ///{
  /// setting level and queue sync
  /// timeline semaphore sync with queue
  /// barrier insert and use pass semaphore
  ///   if(passType == compute)
  ///      ->copute excute
  /// slice with timeline semaphore, async pass level
  /// time line level
  /// async pass
  /// main graphics CB
  /// main
  ///
  ///
  ///}
  VkSubmitInfo submitInfo{};
  VkSemaphore waitSemaphores[] = {
    (imageAvailiableSemaphorePool_.semaphores__
      [pCtxt_->renderingContext.currentFrame__])
  };

  VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = waitSemaphores;
  submitInfo.pWaitDstStageMask = waitStages;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmd;

  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = &renderFinishSemaphorePool_.semaphores__
    [pCtxt_->renderingContext.inflightIndex__[pCtxt_->renderingContext.currentFrame__]];
  uint64_t timelineValue = 1;

  //for (auto& pass : passes) {
  //  VkTimelineSemaphoreSubmitInfo timelineInfo = {};
  //  timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  //  timelineInfo.waitSemaphoreValueCount = 1;
  //  timelineInfo.pWaitSemaphoreValues = &pass->waitValue; // 이전 pass signal value
  //  timelineInfo.signalSemaphoreValueCount = 1;
  //  timelineInfo.pSignalSemaphoreValues = &timelineValue;
  //
  //  VkSubmitInfo submitInfo = {};
  //  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  //  submitInfo.pNext = &timelineInfo;
  //  submitInfo.commandBufferCount = 1;
  //  submitInfo.pCommandBuffers = &pass->cmd;
  //  submitInfo.waitSemaphoreCount = 1;
  //  submitInfo.pWaitSemaphores = &timelineSemaphore;
  //  submitInfo.signalSemaphoreCount = 1;
  //  submitInfo.pSignalSemaphores = &timelineSemaphore;
  //
  //  vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
  //
  //  pass->signalValue = timelineValue; // 다음 pass가 wait할 값
  //  timelineValue++;
  //}
  //
  if (vkQueueSubmit(this->pCtxt_->graphicsQh__,
                    1,
                    &submitInfo,
                    maxInflightFence_.fences
                    [pCtxt_->renderingContext.currentFrame__]) != VK_SUCCESS)
  {
    throw std::runtime_error("failed to submit draw command buffer!");
  }

  VkSwapchainKHR swapchains[] = {pCtxt_->pSwapChainContext->swapchain__};
  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &renderFinishSemaphorePool_.semaphores__
    [pCtxt_->renderingContext.inflightIndex__[pCtxt_->renderingContext.currentFrame__]];
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = swapchains;
  presentInfo.pImageIndices = &(pCtxt_->renderingContext.inflightIndex__[pCtxt_->renderingContext.currentFrame__]);
  if (vkQueuePresentKHR(this->pCtxt_->presentQh__, &presentInfo) != VK_SUCCESS)
  {
    throw std::runtime_error("present Queue error");
  }
  pCtxt_->renderingContext.currentFrame__ = (pCtxt_->renderingContext.currentFrame__ + 1) %
    (pCtxt_->renderingContext.maxInflight__);
  ////spdlog::debug(" present");
}

///
// VkReleaseSwapchainImagesInfoEXT releaseInfo{};
// releaseInfo.sType = VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_EXT;
// releaseInfo.imageIndexCount = 1;
// releaseInfo.swapchain = pCtxt_->pSwapChainContext->swapchain__;
// releaseInfo.pImageIndices = &pCtxt_->renderingContext.inflightIndex__[pCtxt_->renderingContext.currentFrame__];
// vkReleaseSwapchainImagesEXT(pCtxt_->deviceh__, &releaseInfo);
///
///
///
///
///

//if (pCtxt_->renderingContext.inflightIndex__[pCtxt_->renderingContext.
//              currentFrame__] != pCtxt_->renderingContext.currentFrame__)
//{
//  //vkWaitForFences(pCtxt_->deviceh__,
//  //              maxInflightFence_.fences.size(),
//  //              maxInflightFence_.fences.data(),
//  //              VK_TRUE,
//  //              10000);
//}
