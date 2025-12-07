//
// Created by ljh on 25. 10. 13..
//

#ifndef MYPROJECT_VK_SHADER_MEMORY_HPP
#define MYPROJECT_VK_SHADER_MEMORY_HPP

#include <vulkan/vulkan.h>

namespace gpu
{
  class VkShaderMemory{
    //VkBufferDeviceAddressInfo info{};
    //info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    //info.buffer = myUBO;
    //VkDeviceAddress addr = vkGetBufferDeviceAddress(device, &info);
    //vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(addr), &addr);
    //VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT

  public:
    VkShaderMemory();
    ~VkShaderMemory()
    {
      VkDeviceSize size = 100;
      VkBuffer buffer;
      VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      };
      vkCreateBuffer(context.deviceh__, &bufferInfo, nullptr, &buffer);
      VkMemoryRequirements memRequirements;
      vkGetBufferMemoryRequirements(context.deviceh__, buffer, &memRequirements);
      VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

      VkAllocation allocation = context.pMemoryAllocator->allocate(memRequirements, properties);

      if (vkBindBufferMemory(context.deviceh__,
                             buffer,
                             allocation.memory__,
                             allocation.offset__) != VK_SUCCESS)
      {
        spdlog::info("falil to bind buffer to memory");
      }

      VkBufferDeviceAddressInfo addrInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = buffer,
      };
      VkDeviceAddress gpuAddress = vkGetBufferDeviceAddress(context.deviceh__, &addrInfo);

      VkMemoryAllocateFlagsInfo allocFlags{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, // GPU 주소로 접근 가능하게
      };
      VkMemoryAllocateInfo allocInfo{};
      allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      allocInfo.allocationSize  = size;
      allocInfo.memoryTypeIndex = memoryTypeIndex;
      allocInfo.pNext           = &allocFlags;
      if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
      {
        throw std::runtime_error ();
      }
    }
  };
}


#endif //MYPROJECT_VK_SHADER_MEMORY_HPP
