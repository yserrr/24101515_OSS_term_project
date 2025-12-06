#ifndef TEXTURE_HPP
#define TEXTURE_HPP
#include <common.hpp>
#include <../resource/sampler_builder.hpp>

#include <filesystem>
#include <stb_image.h>
#include <static_buffer.hpp>
#include <ktx/include/ktx.h>
#include <ktx/include/ktxvulkan.h>

struct TextureCreateInfo{
  VkDevice device;
  VkSampler sampler;
  const char *filename;
  MemoryAllocator *allocator;
};

struct VulkanTexture{
  VulkanTexture(TextureCreateInfo info);
  ~VulkanTexture();
  void loadKtx(VkCommandBuffer commandBuffer);
  void loadImage(VkCommandBuffer commandBuffer);
  void uploadDescriptor(VkDescriptorSet set, uint32_t arrayIndex);
  void copyBufferToImage(VkCommandBuffer command , uint32_t level = 0);
  MemoryAllocator &allocator;
  Allocation textureMemory;
  VkDevice device               = VK_NULL_HANDLE;
  VkImage textureImage          = VK_NULL_HANDLE;
  VkImageView textureImageView  = VK_NULL_HANDLE;
  VkSampler sampler             = VK_NULL_HANDLE;
  VkDescriptorSet descriptor    = VK_NULL_HANDLE;
  VkImageViewType imgViewType   = VK_IMAGE_VIEW_TYPE_2D;
  VkFormat imgFormat            = VK_FORMAT_R8G8B8A8_SRGB;
  VkImageLayout imgLayout       = VK_IMAGE_LAYOUT_UNDEFINED;
  uint32_t depth                = 0;
  uint32_t mipLevels            = 0;
  uint32_t arrayLevels          = 0;
  VkDeviceSize imageOffset      = 0;
  uint32_t descriptorArrayIndex = 0;
  uint32_t waitFrame            = -1;
  uint32_t width                = 0;
  uint32_t height               = 0;
  uint32_t bindigIndex          = -1;
  std::string name;
  std::unique_ptr<StaticBuffer> stagingBuffer;
  bool ktx           = false;
  bool uploadedReady = false;
};
#endif //