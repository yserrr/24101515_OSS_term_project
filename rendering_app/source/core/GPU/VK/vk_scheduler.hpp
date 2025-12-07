//
// Created by ljh on 25. 10. 7..
//

#ifndef _VK_RENDERER_HPP_
#define _VK_RENDERER_HPP_
#include <vector>
#include "vk_command_buffer.hpp"
#include "vk_sync_object.hpp"
#include "vk_swapchain.hpp"
#include "vk_graph_builder.hpp"
//renderer :
namespace gpu
{
  ///this class -> per frame resource tracking and schejule,
  ///fence, semaphore control class from current context
  ///frame, resource reference-> delete handler and binding control class
  class VkContext ;
  class VkScheduler{
  public:
    VkScheduler(gpu::VkContext* context);
    ~VkScheduler();
    VkBool32 nextFrame();
    void run(std::vector<VkPass>& passes);
    VkBool32 dirty_;
  private:
    uint32_t lastFrame_ = 0;
    gpu::VkCommandBufferPool commandBufferPool_;
    gpu::VkSemaphorePool imageAvailiableSemaphorePool_;
    gpu::VkSemaphorePool renderFinishSemaphorePool_;
    gpu::VkFencePool maxInflightFence_;

    gpu::VkContext* pCtxt_;
    VkTimelineSemaphoreSubmitInfoKHR submitInfo_;
    PFN_vkReleaseSwapchainImagesEXT vkReleaseSwapchainImagesEXT = nullptr;
  };
}
#endif //_VK_RENDERER_HPP_
