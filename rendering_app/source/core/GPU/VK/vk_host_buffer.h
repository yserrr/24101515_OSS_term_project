//
// Created by dlwog on 25. 10. 22..
//

#ifndef MYPROJECT_VK_HOST_BUFFER_H
#define MYPROJECT_VK_HOST_BUFFER_H
#include "vk_resource.hpp"
namespace gpu
{
  class VkHostBuffer : public VkResource
  {
    friend class VkGraphBuilder;
    friend class VkGraphBuilder;
    friend class VkResourceAllocator;
    friend class VkGraph;
    friend class VkDiscardPool;
    public:
    void* data_;
    VkDeviceSize size_;
    VkBuffer bufferh_;
    VkAllocation allocation__;
    std::vector<VkDescriptorSet> sets;
    void uploadData();
    private:
    VkBool32 descriptorAllocated__ = false;
    VkPipelineStageFlagBits writePipelineStage__;
    VkPipelineStageFlagBits currentPipelineStage__;
    VkBufferUsageFlags usage__;
    VkAccessFlagBits currentAccess__ = VK_ACCESS_NONE;
    VkAccessFlagBits writeAccess__ = VK_ACCESS_NONE;
  };
}

#endif //MYPROJECT_VK_HOST_BUFFER_H
