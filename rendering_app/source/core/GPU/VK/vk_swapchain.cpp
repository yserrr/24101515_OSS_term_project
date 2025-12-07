#include "vk_swapchain.hpp"
#include "vk_context.hpp"

gpu::VkSwapchainContext::VkSwapchainContext(gpu::VkContext* pCtxt) :
  device(pCtxt->deviceh__),
  swapchain__(VK_NULL_HANDLE),
  extent__(pCtxt->initExtent__)
{
  VkSurfaceCapabilitiesKHR capabilities;
  if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pCtxt->physicalDeviceh__,
                                                pCtxt->surfaceh__,
                                                &capabilities)
      != VK_SUCCESS)
  {
    throw std::runtime_error("fail to find surface capability");
  }

  VkSurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(pCtxt->physicalDeviceh__,
                                                         pCtxt->surfaceh__);
  VkPresentModeKHR presentMode = choosePresentMode(pCtxt->physicalDeviceh__,
                                                   pCtxt->surfaceh__);

  extent__ = chooseExtent(capabilities, extent__);


  uint32_t imageCount = capabilities.minImageCount + 1;
  if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount)
  {
    imageCount = capabilities.maxImageCount;
  }
  VkSwapchainCreateInfoKHR createInfo{};
  createInfo.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  createInfo.surface          = pCtxt->surfaceh__;
  createInfo.minImageCount    = 3;
  createInfo.imageFormat      = surfaceFormat.format;
  createInfo.imageColorSpace  = surfaceFormat.colorSpace;
  createInfo.imageExtent      = extent__;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

  uint32_t queueFamilyIndices[] = {
    pCtxt->graphicsFamailyIdx__,
    pCtxt->presentFamilyIdx__
  };
  if (pCtxt->graphicsFamailyIdx__ !=
      pCtxt->presentFamilyIdx__
  )
  {
    createInfo.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
    createInfo.queueFamilyIndexCount = 2;
    createInfo.pQueueFamilyIndices   = queueFamilyIndices;
  }
  else
  {
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }
  createInfo.preTransform   = capabilities.currentTransform;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.presentMode    = presentMode;
  createInfo.clipped        = VK_TRUE;
  createInfo.oldSwapchain   = VK_NULL_HANDLE;

  if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain__) != VK_SUCCESS)
  {
    throw std::runtime_error("failed to create swapchain!");
  }

  uint32_t swapImageCount = 0;
  if (vkGetSwapchainImagesKHR(device, swapchain__, &swapImageCount, nullptr) != VK_SUCCESS)
  {
    throw std::runtime_error("fail to get image in swapchain");
  }
  img__.resize(swapImageCount);
  imgView__.resize(swapImageCount);
  swapchainAttachment__.resize(swapImageCount);
  vkGetSwapchainImagesKHR(device, swapchain__, &swapImageCount, img__.data());
  imgFormat__ = surfaceFormat.format;
  extent__    = createInfo.imageExtent;
  spdlog::info("create swapchain");

  for (size_t i = 0; i < swapImageCount; i++)
  {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image                           = img__[i];
    viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format                          = imgFormat__;
    viewInfo.components.r                    = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.components.g                    = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.components.b                    = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.components.a                    = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 1;
    if (vkCreateImageView(device, &viewInfo, nullptr, &imgView__[i]) != VK_SUCCESS)
    {
      throw std::runtime_error("failed to create image view!");
    }
  }
  pCtxt->renderingContext.maxInflight__ = swapImageCount;
  spdlog::info("make swap chain ");
  spdlog::info("swaphchain image : {}", this->img__.size());
}

gpu::VkSwapchainContext::~VkSwapchainContext()
{
  if (swapchain__ != VK_NULL_HANDLE)
  {
    vkDestroySwapchainKHR(device, swapchain__, nullptr);
  }

  for (auto view : imgView__)
  {
    vkDestroyImageView(device, view, nullptr);
  }
}


VkSurfaceFormatKHR gpu::VkSwapchainContext::chooseSurfaceFormat(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface)
{
  uint32_t count;
  vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &count, nullptr);
  std::vector<VkSurfaceFormatKHR> formats(count);
  vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &count, formats.data());
  for (const auto& available : formats)
  {
    if (available.format == VK_FORMAT_B8G8R8A8_SRGB &&
        available.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
    {
      return available;
    }
  }
  return formats[0]; // fallback
}

VkPresentModeKHR gpu::VkSwapchainContext::choosePresentMode(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface)
{
  uint32_t count;
  vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &count, nullptr);
  std::vector<VkPresentModeKHR> modes(count);
  vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &count, modes.data());
  for (const auto& mode : modes)
  {
    if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
    {
      return mode;
    }
  }
  return VK_PRESENT_MODE_FIFO_KHR; // 보장된 기본 모드
}

VkExtent2D gpu::VkSwapchainContext::chooseExtent(const VkSurfaceCapabilitiesKHR& capabilities, VkExtent2D windowExtent)
{
  if (capabilities.currentExtent.width != UINT32_MAX)
  {
    return capabilities.currentExtent;
  }
  else
  {
    VkExtent2D actualExtent = windowExtent;
    actualExtent.width      = std::max(capabilities.minImageExtent.width,
                                  std::min(capabilities.maxImageExtent.width, actualExtent.width));
    actualExtent.height = std::max(capabilities.minImageExtent.height,
                                   std::min(capabilities.maxImageExtent.height, actualExtent.height));
    return actualExtent;
  }
}
