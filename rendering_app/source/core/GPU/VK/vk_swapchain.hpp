#ifndef VK_SWAPCHAIN_HPP
#define VK_SWAPCHAIN_HPP

#include "vk_common.hpp"
#include "vk_memory_allocator.hpp"
#include "vk_texture.hpp"

namespace gpu
{
  class VkContext;
  class VkSwapchainContext
  {
    public:
    VkSwapchainContext(VkContext* context);
    ~VkSwapchainContext();
    VkDevice device;
    VkFormat imgFormat__;
    VkSwapchainKHR swapchain__;
    std::vector<VkImage> img__;
    std::vector<VkImageView> imgView__;
    std::vector<VkFrameAttachment* > swapchainAttachment__;
    VkBool32 broked__ = true;
    VkExtent2D extent__;
    private:
    VkPresentModeKHR choosePresentMode(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);
    VkSurfaceFormatKHR chooseSurfaceFormat(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);
    VkExtent2D chooseExtent(const VkSurfaceCapabilitiesKHR& capabilities, VkExtent2D desireExtent__);
  };
}


#endif
