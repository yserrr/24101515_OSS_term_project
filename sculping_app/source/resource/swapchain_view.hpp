#include "common.hpp"
#ifndef IMAGEVIEW_HPP
#define IMAGEVIEW_HPP 

struct ViewManagerCreateInfo{
  VkDevice                    device;
  const std::vector<VkImage> *images;
  VkFormat                    format;
  VkExtent2D                  extent;
  MemoryAllocator *           allocator;
};

class SwapchainViewManager{
public:
  SwapchainViewManager(const ViewManagerCreateInfo &info);
  ~SwapchainViewManager();
  const std::vector<VkImageView> &getImageViews() const;
  const std::vector<VkImageView> &getDepthViews() const;

private:
  VkDevice                 device;
  VkExtent2D               extent;
  std::vector<VkImageView> imageViews;
  std::vector<VkImageView> depthViews;
  std::vector<VkImage>     depthImage;
  std::vector<Allocation>  allocations;
  MemoryAllocator &        allocator;
  void setDepthView();

  Allocation memoryBind(uint32_t index);
};

#endif