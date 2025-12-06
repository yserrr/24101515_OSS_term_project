#include "descriptor_uploader.hpp"
#include "memory_pool.hpp"

DescriptorUploader::DescriptorUploader(VkDevice device) :
  device_(device) {}

inline uintptr_t vkToUptr(VkDescriptorSet descriptor_set)
{
  return reinterpret_cast<uintptr_t>(descriptor_set);
}

void DescriptorUploader::UploadUboSet(VkBuffer srcBuffer,
                                      VkDeviceSize srcSize,
                                      VkDescriptorSet dstSet,
                                      uint32_t dstBindingIndex,
                                      uint32_t dstBindingArrayIndex,
                                      uint32_t dstCount)
{
  VkDescriptorBufferInfo bufferInfo{};
  bufferInfo.buffer = srcBuffer;
  bufferInfo.range  = VK_WHOLE_SIZE;
  bufferInfo.offset = 0;

  VkWriteDescriptorSet writeDescriptorSet{};
  writeDescriptorSet.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writeDescriptorSet.dstSet          = dstSet;
  writeDescriptorSet.dstBinding      = dstBindingIndex;
  writeDescriptorSet.pBufferInfo     = &bufferInfo;
  writeDescriptorSet.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  writeDescriptorSet.descriptorCount = dstCount;
  writeDescriptorSet.dstArrayElement = dstBindingArrayIndex;
  vkUpdateDescriptorSets(device_, 1, &writeDescriptorSet, 0, nullptr);
}

//void DescriptorUploader::uploadBindlessTextureSet(VulkanTexture *texture)
//{
//  VkDescriptorImageInfo imageInfo{};
//  imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
//  imageInfo.imageView   = texture->view;
//  imageInfo.sampler     = texture->sampler;
//  imageInfos_.push_back(imageInfo);
//
//  VkWriteDescriptorSet descriptorWrite{};
//  descriptorWrite.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
//  descriptorWrite.dstSet          = texture->descriptor;
//  descriptorWrite.dstBinding      = texture->bindigIndex;
//  descriptorWrite.dstArrayElement = texture->descriptorArrayIndex;
//  descriptorWrite.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
//  descriptorWrite.descriptorCount = 1;
//  descriptorWrite.pImageInfo      = &imageInfos_[imageInfos_.size() - 1];
//  writedSets_.push_back(descriptorWrite);
//}

void DescriptorUploader::update()
{
  vkUpdateDescriptorSets(device_, writedSets_.size(), writedSets_.data(), 0, nullptr);
  bufferInfos_.clear();
  imageInfos_.clear();
  writedSets_.clear();
}

void DescriptorUploader::free()
{
  // todo :
  //  not default allocate setting
  //  changa able structure, more large pool control
  //  dynamic offset control
  //  before large, just emtpy function
}