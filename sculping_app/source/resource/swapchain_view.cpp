//
// Created by ljh on 25. 9. 17..
//\

#include"swapchain_view.hpp"

SwapchainViewManager::SwapchainViewManager(const ViewManagerCreateInfo &info) : device(info.device),
                                                                                allocator(*info.allocator),
                                                                                extent(info.extent)
{
  imageViews.resize(info.images->size());
  for (size_t i = 0; i < info.images->size(); i++)
  {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image                           = info.images->at(i);
    viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format                          = info.format;
    viewInfo.components.r                    = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.components.g                    = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.components.b                    = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.components.a                    = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 1;
    if (vkCreateImageView(device, &viewInfo, nullptr, &imageViews[i]) != VK_SUCCESS)
    {
      throw std::runtime_error("failed to create image view!");
    }
    spdlog::info("create imageManager");
  }
  setDepthView();
}

SwapchainViewManager::~SwapchainViewManager()
{
  for (auto view: imageViews)
  {
    vkDestroyImageView(device, view, nullptr);
  }
  for (auto view: depthViews)
  {
    vkDestroyImageView(device, view, nullptr);
  }

// depthImage는 vkDestroyImage 호출해야 하는 대상 아님. VkImage는 vkDestroyImage로 해제해야 함
  for (auto img: depthImage)
  {
    if (img != VK_NULL_HANDLE)
    {
      vkDestroyImage(device, img, nullptr);
    }
  }
  spdlog::info("terminate imageView");
  return;
}

const std::vector<VkImageView> &SwapchainViewManager::getImageViews() const
{
  return imageViews;
}

const std::vector<VkImageView> &SwapchainViewManager::getDepthViews() const
{
  return depthViews;
}

void SwapchainViewManager::setDepthView()
{
  depthImage.resize(imageViews.size());
  allocations.resize(imageViews.size());
  for (int i = 0; i < depthImage.size(); i++)
  {
    VkImageCreateInfo depthInfo{};
    depthInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    depthInfo.imageType     = VK_IMAGE_TYPE_2D;
    depthInfo.extent.width  = extent.width;
    depthInfo.extent.height = extent.height;
    depthInfo.extent.depth  = 1;
    depthInfo.mipLevels     = 1;
    depthInfo.arrayLayers   = 1;
    depthInfo.format        = VK_FORMAT_D32_SFLOAT; // 깊이 포맷
    depthInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    depthInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthInfo.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT; // 깊이 스텐실 용도
    depthInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    depthInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateImage(device, &depthInfo, nullptr, &depthImage[i]) != VK_SUCCESS)
    {
      throw std::runtime_error("failed to create depth image!");
    }
    allocations[i] = memoryBind(i);
    if (vkBindImageMemory(device, depthImage[i], allocations[i].memory, allocations[i].offset) != VK_SUCCESS)
    {
      throw std::runtime_error("fail to allocate depth memroy");
    }
  }
  depthViews.resize(imageViews.size());
  for (int i = 0; i < depthViews.size(); i++)
  {
    VkImageViewCreateInfo depthInfo{};
    depthInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    depthInfo.image                           = depthImage[i];
    depthInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    depthInfo.format                          = VK_FORMAT_D32_SFLOAT;
    depthInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthInfo.subresourceRange.baseMipLevel   = 0;
    depthInfo.subresourceRange.levelCount     = 1;
    depthInfo.subresourceRange.baseArrayLayer = 0;
    depthInfo.subresourceRange.layerCount     = 1;
    if (vkCreateImageView(device, &depthInfo, nullptr, &depthViews[i]) != VK_SUCCESS)
    {
      throw std::runtime_error("fail to make depth View");
    }
  }
}

Allocation SwapchainViewManager::memoryBind(uint32_t index)
{
  VkMemoryRequirements memRequirements;
  VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  vkGetImageMemoryRequirements(device, depthImage[index], &memRequirements);
  return allocator.allocate(memRequirements, properties);
}