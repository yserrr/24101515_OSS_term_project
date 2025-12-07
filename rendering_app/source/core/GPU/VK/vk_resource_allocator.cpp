#include <memory>
#include "spdlog/spdlog.h"
#include "vk_resource_allocator.hpp"
#include "stb_image/stb_image.h"
#include "vk_context.hpp"
#include "vk_swapchain.hpp"
#include "ktx.h"

gpu::VkResourceAllocator::VkResourceAllocator(VkContext* pCtxt) :
  pCtxt(pCtxt),
  samplerBuilder_(pCtxt)

{
}

gpu::VkResourceAllocator::~VkResourceAllocator() = default;

void gpu::VkResourceAllocator::buildMeshNode(VkMeshBuffer* buffer)
{
  buffer->vSize__ = sizeof(buffer->vertex[0]) * buffer->vertex.size();
  buffer->iSize__ = sizeof(buffer->indices[0]) * buffer->indices.size();
  buffer->vData__ = buffer->vertex.data();
  buffer->iData__ = buffer->indices.data();
  buffer->vertexBuffer__ = buildBufferHandle(buffer->vSize__,
                                             VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  buffer->vAllocation__ = mBindBuffer(buffer->vertexBuffer__,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VkHostBuffer vStaging = getStagingBuffer(buffer->vData__,
                                           buffer->vSize__);
  uploadCopyPass(vStaging.bufferh_,
                 buffer->vertexBuffer__,
                 0,
                 0,
                 buffer->vSize__);
  buffer->indexBuffer__ = buildBufferHandle(buffer->iSize__,
                                            VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  buffer->iAllocation__ = mBindBuffer(buffer->indexBuffer__,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VkHostBuffer iStaging = getStagingBuffer(buffer->iData__,
                                           buffer->iSize__);
  uploadCopyPass(iStaging.bufferh_,
                 buffer->indexBuffer__,
                 0,
                 0,
                 buffer->iSize__);
  buffer->hostUpdate__ = true;
  buffer->allocated__ = true;
}

gpu::VkHostBuffer gpu::VkResourceAllocator::getStagingBuffer(void* data,
                                                             VkDeviceSize size)
{
  VkHostBuffer stagingBuffer{};
  stagingBuffer.data_ = data;
  stagingBuffer.bufferh_ = buildBufferHandle(size,
                                             VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

  stagingBuffer.allocation__ = mBindBuffer(stagingBuffer.bufferh_,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  hostUpdate(&stagingBuffer);
  return stagingBuffer;
}


void gpu::VkResourceAllocator::buildFrameAttachment(VkFrameAttachment* image)
{
  image->allocated__ = VK_TRUE;

  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent.width = image->width__;
  imageInfo.extent.height = image->height__;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = image->format__;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = image->usage__;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateImage(pCtxt->deviceh__,
                    &imageInfo,
                    nullptr,
                    &image->imageh__) != VK_SUCCESS)
  {
    throw std::runtime_error("failed to create depth image!");
  }
  image->allocated__ = true;
  image->hostUpdate__ = true;
  switch (image->mSpace_)
  {
    case gpu::MemorySpace::DEVICE_LOCAL:
    {
      image->allocation__ = mBindImage(image->imageh__, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      break;
    }
    case gpu::MemorySpace::HOST_VISIBLE:
    {
      image->allocation__ = mBindImage(image->imageh__,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      break;
    }
    default:
      image->allocation__ = mBindImage(image->imageh__, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      break;
  }

  VkImageViewCreateInfo view{};
  view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view.image = image->imageh__;
  view.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view.format = image->format__;
  view.subresourceRange.aspectMask = image->aspectMask__;
  view.subresourceRange.baseMipLevel = 0;
  view.subresourceRange.levelCount = 1;
  view.subresourceRange.baseArrayLayer = 0;
  view.subresourceRange.layerCount = 1;

  if (vkCreateImageView(pCtxt->deviceh__,
                        &view,
                        nullptr,
                        &image->imageView__) != VK_SUCCESS)
  {
    throw std::runtime_error("fail to make depth View");
  }
}

void gpu::VkResourceAllocator::uploadBufferTransferPass(VkBuffer src,
                                                        VkBuffer dst,
                                                        VkDeviceSize srcOffset,
                                                        VkDeviceSize dstOffset,
                                                        VkDeviceSize size)
{
  gpu::VkPass copyPass;
  copyPass.passType = RenderPassType::COPY_PASS,
    copyPass.read__ = {},
    copyPass.write__ = {},
    copyPass.execute =
    [&src,
      &dst,
      &srcOffset,
      &dstOffset,
      &size]
  (VkCommandBuffer cmd)
    {
      VkBufferCopy region{};
      region.srcOffset = srcOffset;
      region.dstOffset = dstOffset;
      region.size = size;
      vkCmdCopyBuffer(cmd, src, dst, 1, &region);
    };
  copyPass.transitionPass = true;
  pCtxt->transitionPass.push_back(copyPass);
}

void gpu::VkResourceAllocator::uploadCopyPass(VkBuffer src,
                                              VkBuffer dst,
                                              VkDeviceSize srcOffset,
                                              VkDeviceSize dstOffset,
                                              VkDeviceSize size)
{
  VkPass copyPass;

  copyPass.passType = RenderPassType::COPY_PASS,
    copyPass.read__ = {};
  copyPass.write__ = {};
  copyPass.execute =
    [this,
      src,
      dst,
      srcOffset,
      dstOffset,
      size]
  (VkCommandBuffer cmd)
    {
      VkBufferCopy region{};
      region.srcOffset = 0;
      region.dstOffset = 0;
      region.size = size;
      vkCmdCopyBuffer(cmd, src, dst, 1, &region);

      VkBufferMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_NONE_KHR,
        .dstAccessMask = VK_ACCESS_NONE_KHR,
        .srcQueueFamilyIndex = pCtxt->graphicsFamailyIdx__,
        .dstQueueFamilyIndex = pCtxt->graphicsFamailyIdx__,
        .buffer = dst,
        .size = size,
      };
      barrier.offset = dstOffset;
      vkCmdPipelineBarrier(
                           cmd,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                           0,
                           0,
                           nullptr,
                           1,
                           &barrier,
                           0,
                           nullptr
                          );
    };
  copyPass.transitionPass = true;
  pCtxt->transitionPass.push_back(copyPass);
}

void gpu::VkResourceAllocator::buildImageCopyPass(VkBuffer buffer,
                                                  VkTexture* texture)
{
  VkPass copyPass;
  copyPass.passType = RenderPassType::COPY_PASS,
    copyPass.read__ = {},
    copyPass.write__ = {},
    copyPass.execute =
    [buffer, texture](VkCommandBuffer cmd)
    {
      VkBufferImageCopy region{};
      region.bufferOffset = 0;
      region.bufferRowLength = 0;
      region.bufferImageHeight = 0;
      region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      region.imageSubresource.mipLevel = 0;
      region.imageSubresource.baseArrayLayer = 0;
      region.imageSubresource.layerCount = 1;
      region.imageOffset = {0, 0, 0};
      region.imageExtent = {
        texture->width__,
        texture->height__,
        1
      };
      spdlog::info("call to command buffer to trenslate to texture");
      vkCmdCopyBufferToImage(
                             cmd,
                             buffer,
                             texture->imageh__,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             1,
                             &region
                            );
    };
  copyPass.transitionPass = true;
  pCtxt->transitionPass.push_back(copyPass);
}

//immediate upload
void gpu::VkResourceAllocator::buildImageBarrierPass(VkImage img,
                                                     VkPipelineStageFlagBits src,
                                                     VkPipelineStageFlags dst,
                                                     VkAccessFlags srcAccess,
                                                     VkAccessFlags dstAccess,
                                                     VkImageLayout oldLayout,
                                                     VkImageLayout newLayout)
{
  VkPass BarrierPass;
  BarrierPass.passType = RenderPassType::BARRIER_PASS,
  BarrierPass.read__ = {},
  BarrierPass.write__ = {},
  BarrierPass.execute =
    [img,
      src,
      dst,
      srcAccess,
      dstAccess,
      oldLayout,
      newLayout]
  (VkCommandBuffer cmd)
    {
      VkImageMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barrier.oldLayout = oldLayout;
      barrier.newLayout = newLayout;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image = img;
      barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      barrier.subresourceRange.baseMipLevel = 0;
      barrier.subresourceRange.levelCount = 1;
      barrier.subresourceRange.baseArrayLayer = 0;
      barrier.subresourceRange.layerCount = 1;
      barrier.srcAccessMask = srcAccess;
      barrier.dstAccessMask = dstAccess;
      vkCmdPipelineBarrier(cmd,
                           src,
                           dst,
                           0,
                           0,
                           nullptr,
                           0,
                           nullptr,
                           1,
                           &barrier
                          );
    };
  BarrierPass.transitionPass = true;
  pCtxt->transitionPass.push_back(BarrierPass);
}

gpu::VkAllocation gpu::VkResourceAllocator::mBindImage(VkImage image,
                                                       VkMemoryPropertyFlags desiredFlag)
{
  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(pCtxt->deviceh__,
                               image,
                               &memRequirements);
  VkMemoryPropertyFlags properties = desiredFlag;

  gpu::VkAllocation allocation = pCtxt->pMemoryAllocator->allocate(memRequirements,
                                                                   properties);
  if (vkBindImageMemory(pCtxt->deviceh__,
                        image,
                        allocation.memory__,
                        allocation.offset__) != VK_SUCCESS)

  {
    throw std::runtime_error("fail to allocate depth memroy");
  }
  return allocation;
}


void gpu::VkResourceAllocator::hostUpdate(VkHostBuffer* buffer__)
{
  void* data = nullptr;
  VkAllocation alloc = buffer__->allocation__;
  vkMapMemory(pCtxt->deviceh__,
              alloc.memory__,
              alloc.offset__,
              alloc.size,
              0,
              &data);
  std::memcpy(data,
              buffer__->data_,
              (size_t)alloc.size);

  vkUnmapMemory(pCtxt->deviceh__,
                alloc.memory__);
}


VkBuffer gpu::VkResourceAllocator::buildBufferHandle(VkDeviceSize size,
                                                     VkBufferUsageFlags usage)
{
  VkBufferCreateInfo bufferInfo{
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    .size = size,
    .usage = usage,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  VkBuffer buffer;
  if (vkCreateBuffer(pCtxt->deviceh__,
                     &bufferInfo,
                     nullptr,
                     &buffer) != VK_SUCCESS)
  {
    throw std::runtime_error("failed to create buffer!");
  }
  return buffer;
}

//void gpu::VkResourceAllocator::buildKtxTexture(gpu::VkTexture* texture)
//{
//  std::vector<VkBufferImageCopy> bufferCopyRegions;
//  for (uint32_t i = 0; i < texture.mipLevels; i++)
//  {
//    // Calculate offset into staging buffer for the current mip level
//    ktx_size_t offset;
//    KTX_error_code ret = ktxTexture_GetImageOffset(ktxTexture, i, 0, 0, &offset);
//    assert(ret == KTX_SUCCESS);
//    // Setup a buffer image copy structure for the current mip level
//    VkBufferImageCopy bufferCopyRegion = {};
//    bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
//    bufferCopyRegion.imageSubresource.mipLevel = i;
//    bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
//    bufferCopyRegion.imageSubresource.layerCount = 1;
//    bufferCopyRegion.imageExtent.width = ktxTexture->baseWidth >> i;
//    bufferCopyRegion.imageExtent.height = ktxTexture->baseHeight >> i;
//    bufferCopyRegion.imageExtent.depth = 1;
//    bufferCopyRegion.bufferOffset = offset;
//    bufferCopyRegions.push_back(bufferCopyRegion);
//  }
//}
//
void gpu::VkResourceAllocator::buildTexture(gpu::VkTexture* texture)
{
  texture->loadImage();
  texture->allocated__ = VK_TRUE;

  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.flags = 0;
  imageInfo.extent.width = texture->width__;
  imageInfo.extent.height = texture->height__;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.pNext = nullptr;

  if (vkCreateImage(pCtxt->deviceh__, &imageInfo, nullptr, &texture->imageh__) != VK_SUCCESS)
  {
    spdlog::info("error ");
    throw std::runtime_error("fail to make texture Image buffer");
  }

  mBindImage(texture->imageh__,
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = texture->imageh__;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
  viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;
  if (vkCreateImageView(pCtxt->deviceh__,
                        &viewInfo,
                        nullptr,
                        &texture->imageView__) != VK_SUCCESS)
  {
    throw std::runtime_error("error to make texture view");
  }
  texture->imageSize__ = texture->width__ * texture->height__ * 4;
  VkHostBuffer staging = getStagingBuffer(texture->pixels__,
                                          texture->imageSize__);
  buildImageBarrierPass(texture->imageh__,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_ACCESS_NONE_KHR,
                        VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

  buildImageCopyPass(staging.bufferh_, texture);

  buildImageBarrierPass(texture->imageh__,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_ACCESS_SHADER_READ_BIT,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  texture->currentLayout__ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  texture->currentPipeline__ = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  texture->currentAccessMask__ = VK_ACCESS_SHADER_READ_BIT;
  stbi_image_free(texture->pixels__);
}


gpu::VkAllocation gpu::VkResourceAllocator::mBindBuffer(VkBuffer buffer,
                                                        VkMemoryPropertyFlags desiredFlag)
{
  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(pCtxt->deviceh__,
                                buffer,
                                &memRequirements);
  VkAllocation allocation = pCtxt->pMemoryAllocator->allocate(memRequirements, desiredFlag);
  if (vkBindBufferMemory(pCtxt->deviceh__,
                         buffer,
                         allocation.memory__,
                         allocation.offset__) != VK_SUCCESS)
  {
    spdlog::info("falil to bind buffer to memory");
  }
  return allocation;
}
