//
// Created by dlwog on 25. 10. 23..
//

#ifndef MYPROJECT_VK_TEXTURE_HPP
#define MYPROJECT_VK_TEXTURE_HPP
#include "vk_resource.hpp"
#include "stb_image/stb_image.h"

namespace gpu
{
  class VkFrameAttachment : public VkResource
  {
    public:
    VkFrameAttachment();

    public:
    std::string filepath__;
    stbi_uc* pixels__;
    VkBool32 ready = VK_FALSE;
    VkDeviceSize imageSize__ = 0;

    inline void loadImage()
    {
      int32_t texWidth, texHeight, texChannels;
      stbi_uc* pixels = stbi_load(filepath__.c_str(),
                                  &texWidth,
                                  &texHeight,
                                  &texChannels,
                                  STBI_rgb_alpha);
      if (!pixels)
      {
        spdlog::info("failed to load texture image!");
        return;
      }
      pixels__ = (stbi_uc*)pixels;
      this->width__ = texWidth;
      this->height__ = texHeight;
      this->aspectMask__ = VK_SAMPLE_COUNT_1_BIT;
      ready = VK_TRUE;
    }

    VkFormat format__ = VK_FORMAT_R8G8B8A8_UNORM;
    VkImageType imageType__ = VK_IMAGE_TYPE_2D;
    uint32_t height__ = 0;
    uint32_t width__ = 0;
    uint32_t mipLevels__ = 1;
    uint32_t levelCount__ = 1;
    VkImageAspectFlags aspectMask__;
    std::vector<VkDescriptorSet> descriptorSet__;
    uint32_t descriptorArrayIndex__ = -1;
    VkImage imageh__ = VK_NULL_HANDLE;
    VkImageView imageView__;
    VkAllocation allocation__;
    VkSampler sampler = VK_NULL_HANDLE;
    VkImageViewType imgViewType = VK_IMAGE_VIEW_TYPE_2D;
    VkFormat imgFormat = VK_FORMAT_R8G8B8A8_SRGB;
    VkImageLayout imgLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    uint32_t depth = 0;
    uint32_t mipLevels = 0;
    uint32_t arrayLevels = 0;
    VkDeviceSize imageOffset = 0;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkBool32 writen__ = false;
    bool ktx = false;

    private:
    friend class VkGraph;
    friend class VkGraphBuilder;
    friend class VkGraphBuilder;
    friend class VkResourceAllocator;
    friend class VkScheduler;
    friend class VkDiscardPool;
    void* data__;
    VkBool32 needChangeSampler__ = false;
    std::string name;
    VkImageLayout currentLayout__ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout writeLayout__ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageUsageFlags usage__;
  };
  using VkTexture = VkFrameAttachment;
}

#endif //MYPROJECT_VK_TEXTURE_HPP
