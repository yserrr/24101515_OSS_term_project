//
// Created by ljh on 25. 10. 10..
//

#include "vk_descriptor_allocator.hpp"
#include "vk_context.hpp"
#include "vk_resource.hpp"


gpu::VkDescriptorAllocator::VkDescriptorAllocator(VkContext* pCtxt) :
  pCtxt(pCtxt),
  samplerBuilder_(pCtxt),
  baseSet(VK_NULL_HANDLE)
{
  /// todo: vector resized, all writed set reseted
  ///  so, if resize, find or use static array
  VkDescriptorPoolSize PoolSize[] = {
    {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
    {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
    {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
    {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
    {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
    {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
    {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
    {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}
  };
  VkDescriptorPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT |
    VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
  poolInfo.maxSets = 1024;
  poolInfo.poolSizeCount = sizeof(PoolSize) / sizeof(PoolSize[0]);
  poolInfo.pPoolSizes = PoolSize;
  writedSets_.reserve(300);
  imageInfos_.reserve(300);
  bufferInfos_.reserve(300);

  if (vkCreateDescriptorPool(pCtxt->deviceh__,
                             &poolInfo,
                             nullptr,
                             &pool) != VK_SUCCESS)
  {
    throw std::runtime_error("fail to create Imgui Pool");
  }
  buildDefaultLayout();
  descriptorSets.reserve(pCtxt->renderingContext.maxInflight__);
  for (uint32_t i = 0; i < pCtxt->renderingContext.maxInflight__; i++)
  {
    descriptorSets.push_back(allocateSet(VK_NULL_HANDLE));
  }
}


void gpu::VkDescriptorAllocator::buildDefaultLayout()
{
  std::vector<gpu::VkDescriptorLayoutBindingInfo> infos(3);
  infos[0].bindingIndex = 0;
  infos[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  infos[0].usage = gpu::DescriptorFlag::UBO;

  infos[1].bindingIndex = 1;
  infos[1].stage = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  infos[1].usage = gpu::DescriptorFlag::UBO;

  infos[2].usage = gpu::DescriptorFlag::TEXTURE_BINDLESS;
  infos[2].bindingIndex = 2;
  infos[2].stage = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT;
  defaultLayout = pCtxt->pLayoutBuilder_->createDescriptorSetLayout(infos);
}

VkDescriptorSet gpu::VkDescriptorAllocator::allocateSet(VkDescriptorSetLayout layout, uint32_t count)
{
  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = pool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &layout;
  if (layout == VK_NULL_HANDLE) allocInfo.pSetLayouts = &defaultLayout;
  VkDescriptorSet set;
  if (vkAllocateDescriptorSets(pCtxt->deviceh__,
                               &allocInfo,
                               &set) != VK_SUCCESS)
  {
    throw std::runtime_error("failed to allocate descriptor set");
  }
  return set;
}

void gpu::VkDescriptorAllocator::writeUbo(VkBuffer srcBuffer,
                                          VkDeviceSize srcSize,
                                          VkDescriptorSet dstSet,
                                          uint32_t dstBindingIndex,
                                          uint32_t dstBindingArrayIndex,
                                          uint32_t dstCount)
{
  VkDescriptorBufferInfo bufferInfo{};
  bufferInfo.buffer = srcBuffer;
  bufferInfo.range = srcSize;
  bufferInfo.offset = 0;
  bufferInfos_.push_back(bufferInfo);

  VkWriteDescriptorSet writeDescriptorSet{};
  writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writeDescriptorSet.dstSet = dstSet;
  writeDescriptorSet.dstBinding = dstBindingIndex;
  writeDescriptorSet.pBufferInfo = &bufferInfos_.back();
  writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  writeDescriptorSet.descriptorCount = dstCount;
  writeDescriptorSet.dstArrayElement = dstBindingArrayIndex;
  writedSets_.push_back(writeDescriptorSet);
}

void gpu::VkDescriptorAllocator::uploadBindlessTextureSet(gpu::VkFrameAttachment* texture)
{
  if (texture->sampler == VK_NULL_HANDLE)
  {
    texture->sampler = this->samplerBuilder_.dftSampler;
  }
  texture->descriptorArrayIndex__ = currentBindlessIndex_;
  for (uint32_t i = 0; i < pCtxt->renderingContext.maxInflight__; i++)
  {
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView = texture->imageView__;
    imageInfo.sampler = texture->sampler;
    imageInfos_.push_back(imageInfo);
    VkDescriptorImageInfo* ptr = &this->imageInfos_[imageInfos_.size() - 1];

    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = texture->descriptorSet__[i];
    descriptorWrite.dstBinding = gpu::BINDLESS_TEXTURE;
    descriptorWrite.dstArrayElement = texture->descriptorArrayIndex__;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pImageInfo = ptr;
    this->writedSets_.push_back(descriptorWrite);
  }
  currentBindlessIndex_++;
}


void gpu::VkDescriptorAllocator::update()
{
  if (writedSets_.size() > 0)
  {
    vkUpdateDescriptorSets(pCtxt->deviceh__,
                           writedSets_.size(),
                           writedSets_.data(),
                           0,
                           nullptr);
    bufferInfos_.clear();
    imageInfos_.clear();
    writedSets_.clear();
  }
}

//
//void gpu::VkDescriptorAllocator::free()
//{
//
//}
