#include "common.hpp"
#ifndef SWAPCHAIN_HPP
#define SWAPCHAIN_HPP
struct SwapchainCreateInfo{
  VkDevice         device;
  VkPhysicalDevice physicalDevice;
  VkSurfaceKHR     surface;
  uint32_t         graphicsFamily;
  uint32_t         presentFamily;
  VkExtent2D       windowExtent;
};

class Swapchain{
public:
  Swapchain(const SwapchainCreateInfo &info);
  ~Swapchain();
  uint32_t getImageIndex(VkSemaphore semaphore);
  VkSwapchainKHR get() const;
  const VkSwapchainKHR *getptr() const;
  VkFormat getFormat() ;
  VkExtent2D getExtent() const;
  const std::vector<VkImage> &getImages() const;

private:
  uint32_t             imageIndex;
  VkDevice             device;
  VkExtent2D           windowExtent_;
  VkSwapchainKHR       swapchain;
  std::vector<VkImage> images;
  VkFormat             imageFormat;

  VkSurfaceFormatKHR chooseSurfaceFormat(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);
  VkPresentModeKHR   choosePresentMode(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);
  VkExtent2D         chooseExtent(const VkSurfaceCapabilitiesKHR &capabilities, VkExtent2D windowExtent);
};
#endif