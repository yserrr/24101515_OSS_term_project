#include <texture.hpp>

VulkanTexture::VulkanTexture(TextureCreateInfo info)
  : device(info.device),
    sampler(info.sampler),
    allocator(*info.allocator),
    name(info.filename) {}

VulkanTexture::~VulkanTexture()
{
  if (textureImageView != VK_NULL_HANDLE)
  {
    vkDestroyImageView(device, textureImageView, nullptr);
  }
  if (textureImage != VK_NULL_HANDLE)
  {
    vkDestroyImage(device, textureImage, nullptr);
  }
}

//void VulkanTexture::loadKtx(VkCommandBuffer commandBuffer)
//{
//  ktxTexture *ktxTexture;
//
//  ktxResult result = ktxTexture_CreateFromNamedFile(name.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktxTexture);
//  if (result != KTX_SUCCESS)
//  {
//    spdlog::info("failed to load texture with error ");
//    return;
//  }
//  VkFormat vkFormat = static_cast<VkFormat>(ktxTexture->glFormat);
//  VkExtent3D extent = {ktxTexture->baseWidth, ktxTexture->baseHeight, 1};
//
//  VkImageCreateInfo imageInfo{};
//  imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
//  imageInfo.imageType     = VK_IMAGE_TYPE_2D;
//  imageInfo.format        = vkFormat;
//  imageInfo.extent        = extent;
//  imageInfo.mipLevels     = ktxTexture->numLevels;
//  imageInfo.arrayLayers   = 1;
//  imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
//  imageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
//  imageInfo.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
//  imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
//  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
//  vkCreateImage(device, &imageInfo, nullptr, &textureImage);
//
//  VkMemoryRequirements memoryReq;
//  vkGetImageMemoryRequirements(device, textureImage, &memoryReq);
//  textureMemory = allocator.allocate(memoryReq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
//  vkBindImageMemory(device, textureImage, textureMemory.memory, textureMemory.offset);
//
//  VkImageViewCreateInfo viewInfo{};
//  viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
//  viewInfo.image                           = textureImage;
//  viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
//  viewInfo.format                          = VK_FORMAT_R8G8B8A8_UNORM;
//  viewInfo.components.r                    = VK_COMPONENT_SWIZZLE_IDENTITY;
//  viewInfo.components.g                    = VK_COMPONENT_SWIZZLE_IDENTITY;
//  viewInfo.components.b                    = VK_COMPONENT_SWIZZLE_IDENTITY;
//  viewInfo.components.a                    = VK_COMPONENT_SWIZZLE_IDENTITY;
//  viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
//  viewInfo.subresourceRange.baseMipLevel   = 0;
//  viewInfo.subresourceRange.levelCount     = ktxTexture->numLevels;
//  viewInfo.subresourceRange.baseArrayLayer = 0;
//  viewInfo.subresourceRange.layerCount     = 1;
//  if (vkCreateImageView(device, &viewInfo, nullptr, &textureImageView) != VK_SUCCESS)
//  {
//    throw std::runtime_error("error to make texture view");
//  }
//  ktx_size_t offset;
//  stagingBuffer = std::make_unique<StaticBuffer>(allocator, ktxTexture->dataSize, BufferType::STAGE);
//  for (uint32_t level = 0; level < ktxTexture->numLevels; level++)
//  {
//    ktxTexture_GetImageOffset(ktxTexture, level, 0, 0, &offset);
//    uint32_t levelWidth  = std::max(1u, ktxTexture->baseWidth >> level);
//    uint32_t levelHeight = std::max(1u, ktxTexture->baseHeight >> level);
//
//    VkBufferImageCopy region{};
//    region.bufferOffset                    = offset;
//    region.bufferRowLength                 = 0;
//    region.bufferImageHeight               = 0;
//    region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
//    region.imageSubresource.mipLevel       = level;
//    region.imageSubresource.baseArrayLayer = 0;
//    region.imageSubresource.layerCount     = 1;
//    region.imageOffset                     = {0, 0, 0};
//    region.imageExtent                     = {levelWidth, levelHeight, 1};
//    vkCmdCopyBufferToImage(commandBuffer,
//                           stagingBuffer->getStagingBuffer(),
//                           textureImage,
//                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
//                           1,
//                           &region);
//  }
//  spdlog::info("create staging buffer for texture");
//  ktxTexture_Destroy(ktxTexture);
//}
//
void VulkanTexture::loadImage(VkCommandBuffer commandBuffer)
{
  uploadedReady = false;

  int32_t texWidth, texHeight, texChannels;
  stbi_uc *pixels = stbi_load(name.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
  if (!pixels)
  {
    printf("Failed to load image: %s\n", stbi_failure_reason());
    spdlog::info("failed to load texture image!");
    return;
  }
  uploadedReady          = true;
  VkDeviceSize imageSize = texWidth * texHeight * 4;
  width                  = texWidth;
  height                 = texHeight;

  VkImageCreateInfo imageInfo{};
  imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType     = VK_IMAGE_TYPE_2D;
  imageInfo.flags         = 0;
  imageInfo.extent.width  = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth  = 1;
  imageInfo.mipLevels     = 1;
  imageInfo.arrayLayers   = 1;
  imageInfo.format        = VK_FORMAT_R8G8B8A8_UNORM;
  imageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL; //gpu optional
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.pNext         = nullptr;
  if (vkCreateImage(device, &imageInfo, nullptr, &textureImage) != VK_SUCCESS)
  {
    std::runtime_error("fail to make texture Image buffer");
  }
  VkMemoryRequirements memoryReq;
  vkGetImageMemoryRequirements(device, textureImage, &memoryReq);
  textureMemory = allocator.allocate(memoryReq, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vkBindImageMemory(device, textureImage, textureMemory.memory, textureMemory.offset);

  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image                           = textureImage;
  viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format                          = VK_FORMAT_R8G8B8A8_UNORM;
  viewInfo.components.r                    = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewInfo.components.g                    = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewInfo.components.b                    = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewInfo.components.a                    = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.baseMipLevel   = 0;
  viewInfo.subresourceRange.levelCount     = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount     = 1;
  if (vkCreateImageView(device, &viewInfo, nullptr, &textureImageView) != VK_SUCCESS)
  {
    throw std::runtime_error("error to make texture view");
  }
  stagingBuffer = std::make_unique<StaticBuffer>(allocator, imageSize, BufferType::STAGE);
  stagingBuffer->getStagingBuffer(pixels);
  copyBufferToImage(commandBuffer);
  spdlog::info("create staging buffer for texture");
  stbi_image_free(pixels);
}

void VulkanTexture::uploadDescriptor(VkDescriptorSet set, uint32_t arrayIndex)
{
  if (sampler == VK_NULL_HANDLE)
  {
    spdlog::info("fuck");
    return;
  }
  VkDescriptorImageInfo imageInfo{};
  imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  imageInfo.imageView   = textureImageView;
  imageInfo.sampler     = sampler;

  VkWriteDescriptorSet descriptorWrite{};
  descriptorWrite.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptorWrite.dstSet          = set;
  descriptorWrite.dstBinding      = 0;
  descriptorWrite.dstArrayElement = arrayIndex;
  descriptorWrite.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  descriptorWrite.descriptorCount = 1;
  descriptorWrite.pImageInfo      = &imageInfo;
  vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
}

void VulkanTexture::copyBufferToImage(VkCommandBuffer command, uint32_t level)
{
  VkImageMemoryBarrier barrier{};
  barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout                       = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
  barrier.image                           = textureImage;
  barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel   = 0;
  barrier.subresourceRange.levelCount     = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount     = 1;
  barrier.srcAccessMask                   = 0;
  barrier.dstAccessMask                   = VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(
                       command,
                       VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT,
                       0,
                       0,
                       nullptr,
                       0,
                       nullptr,
                       1,
                       &barrier
                      );
  VkBufferImageCopy region{};
  region.bufferOffset                    = 0;
  region.bufferRowLength                 = 0;
  region.bufferImageHeight               = 0;
  region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel       = level;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount     = 1;
  region.imageOffset                     = {0, 0, 0};
  region.imageExtent                     = {width, height, 1};
  spdlog::info("call to command buffer to trenslate to texture");
  vkCmdCopyBufferToImage(
                         command,
                         stagingBuffer->getStagingBuffer(),
                         textureImage,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         1,
                         &region
                        );
  barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(
                       command,
                       VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                       0,
                       0,
                       nullptr,
                       0,
                       nullptr,
                       1,
                       &barrier
                      );
  waitFrame = 1;
}