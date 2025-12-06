//
// Created by ljh on 25. 9. 17..
//
#include "descriptor_manager.hpp"

DescriptorManager::DescriptorManager(VkDevice device) : device(device)
{
  std::vector<VkDescriptorSetLayoutBinding> descriptorLayoutBindings(2);
  std::vector<VkDescriptorBindingFlags> bindingFlags(2);
  descriptorLayoutBindings[0].binding            = 0;
  descriptorLayoutBindings[0].descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  descriptorLayoutBindings[0].descriptorCount    = 1;
  descriptorLayoutBindings[0].stageFlags         = VK_SHADER_STAGE_VERTEX_BIT;
  descriptorLayoutBindings[0].pImmutableSamplers = nullptr;

  descriptorLayoutBindings[1].binding            = 1;
  descriptorLayoutBindings[1].descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  descriptorLayoutBindings[1].descriptorCount    = 1;
  descriptorLayoutBindings[1].stageFlags         = VK_SHADER_STAGE_FRAGMENT_BIT;
  descriptorLayoutBindings[1].pImmutableSamplers = nullptr;

  bindingFlags[0] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
  bindingFlags[1] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;

  VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
  bindingFlagsInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
  bindingFlagsInfo.bindingCount  = static_cast<uint32_t>(bindingFlags.size());
  bindingFlagsInfo.pBindingFlags = bindingFlags.data();

  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = descriptorLayoutBindings.size();
  layoutInfo.pBindings    = descriptorLayoutBindings.data();

  if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &vertexLayout) != VK_SUCCESS)
    throw std::runtime_error("failed to create descriptor set layout!");

  std::vector<VkDescriptorSetLayoutBinding> fragmentBindings(1);
  std::vector<VkDescriptorBindingFlags> bindingsFlags(1);
  fragmentBindings[0].binding            = 0;
  fragmentBindings[0].descriptorType     = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  fragmentBindings[0].descriptorCount    = 1024;
  fragmentBindings[0].stageFlags         = VK_SHADER_STAGE_FRAGMENT_BIT;
  fragmentBindings[0].pImmutableSamplers = nullptr;

  bindingFlags[0] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;

  VkDescriptorSetLayoutBindingFlagsCreateInfo layoutFlags = {};
  layoutFlags.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
  layoutFlags.bindingCount = 1;
  layoutFlags.pBindingFlags = bindingFlags.data();

  VkDescriptorSetLayoutCreateInfo fragInfo{};
  fragInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  fragInfo.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
  fragInfo.bindingCount = fragmentBindings.size();
  fragInfo.pBindings    = fragmentBindings.data();
  fragInfo.pNext        = &layoutFlags;

  if (vkCreateDescriptorSetLayout(device, &fragInfo, nullptr, &textureLayout) != VK_SUCCESS)
    throw std::runtime_error("failed to create descriptor set layout!");
//light material : set 0 , biniding 1
  VkDescriptorSetLayoutBinding lightBinding{};
  lightBinding.binding            = 0;
  lightBinding.descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  lightBinding.descriptorCount    = 1;
  lightBinding.stageFlags         = VK_SHADER_STAGE_FRAGMENT_BIT;
  lightBinding.pImmutableSamplers = nullptr;
  VkDescriptorSetLayoutCreateInfo lightInfo{};
  lightInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  lightInfo.bindingCount = 1;
  lightInfo.pBindings    = &lightBinding;

  if (vkCreateDescriptorSetLayout(device, &lightInfo, nullptr, &lightLayout) != VK_SUCCESS)
    throw std::runtime_error("failed to create descriptor set layout!");
  layouts = {vertexLayout, textureLayout, lightLayout};

  cameraPool  = std::make_unique<CameraDescriptorPool>(device, vertexLayout);
  texturePool = std::make_unique<TextureDescriptorPool>(device, textureLayout);
  lightPool   = std::make_unique<LightDescriptorPool>(device, lightLayout);
  cameraPool->createPool(UBO_MAX); //max camera sets
  texturePool->createPool(TEXTURE_MAX);
}

DescriptorManager::~DescriptorManager()
{
  if (vertexLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, vertexLayout, nullptr);
  if (textureLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, textureLayout, nullptr);
  if (lightLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, lightLayout, nullptr);
}

VkDescriptorSet DescriptorManager::bindlessSet()
{
  return texturePool->allocateSet();
}

std::vector<VkDescriptorSetLayout> *DescriptorManager::getLayouts()
{
  return &layouts;
}

std::vector<VkDescriptorSet> DescriptorManager::getSets()
{
  return descriptors;
}

VkDescriptorSet DescriptorManager::allocateSet()
{
  return cameraPool->allocateSet();
}

VkDescriptorSetLayout DescriptorManager::getVert()
{
  return vertexLayout;
}

VkDescriptorSetLayout DescriptorManager::getFrag()
{
  return textureLayout;
}

VkDescriptorSetLayout DescriptorManager::getLight()
{
  return lightLayout;
}