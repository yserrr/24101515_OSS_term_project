#include <descriptor_pool.hpp>

VkDescriptorSet DescriptorPool::allocateSet()
{
  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool              = descriptorPool;
  allocInfo.descriptorSetCount          = 1;
  allocInfo.pSetLayouts                 = &layout;
  VkDescriptorSet set;
  if (vkAllocateDescriptorSets(device, &allocInfo, &set) != VK_SUCCESS)
    throw std::runtime_error("failed to allocate descriptor set");

  allocatedSets.push_back(set);
  return set;
};

void CameraDescriptorPool::createPool(uint32_t count)
{
  VkDescriptorPoolSize poolSize{};
  poolSize.type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // 필요 타입
  poolSize.descriptorCount = count;                             // 이 타입 몇 개 필요한지
  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes    = &poolSize;
  poolInfo.maxSets       = count; // 총 몇 개의 descriptor set을 만들건지
  poolInfo.flags         = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;     // VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT 등 가능
  if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
  {
    throw std::runtime_error("failed to create descriptor pool!");
  }
}

void TextureDescriptorPool::createPool(uint32_t count)
{
  VkDescriptorPoolSize poolSize{};
  poolSize.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; // 필요 타입
  poolSize.descriptorCount = count;                                     // 이 타입 몇 개 필요한지
  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes    = &poolSize;
  poolInfo.maxSets       = count; // 총 몇 개의 descriptor set을 만들건지
  poolInfo.flags         = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;     // VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT 등 가능
  if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
  {
    throw std::runtime_error("failed to create descriptor pool!");
  }
}

void LightDescriptorPool::createPool(uint32_t count)
{
  VkDescriptorPoolSize poolSize{};
  poolSize.type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // 필요 타입
  poolSize.descriptorCount = count;                             // 이 타입 몇 개 필요한지
  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes    = &poolSize;
  poolInfo.maxSets       = count; // 총 몇 개의 descriptor set을 만들건지
  poolInfo.flags         = 0;     // VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT 등 가능
  if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
  {
    throw std::runtime_error("failed to create descriptor pool!");
  }
}