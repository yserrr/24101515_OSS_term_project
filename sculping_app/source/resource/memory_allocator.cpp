//
// Created by ljh on 25. 9. 16..
//
#include "memory_allocator.hpp"
#include "spdlog/spdlog.h"

extern PFN_vkSetDebugUtilsObjectNameEXT g_pfnSetDebugUtilsObjectNameEXT;
extern PFN_vkSetDebugUtilsObjectTagEXT g_pfnSetDebugUtilsObjectTagEXT;

namespace VulkanObjectHelpers{
  template<typename HandleType>
  VkObjectType GetVulkanObjectType(HandleType handle);
  template<> VkObjectType GetVulkanObjectType(VkBuffer)
  {
    return VK_OBJECT_TYPE_BUFFER;
  }

  template<> VkObjectType GetVulkanObjectType(VkImage)
  {
    return VK_OBJECT_TYPE_IMAGE;
  }

  template<> VkObjectType GetVulkanObjectType(VkDeviceMemory)
  {
    return VK_OBJECT_TYPE_DEVICE_MEMORY;
  }

  template<> VkObjectType GetVulkanObjectType(VkSampler)
  {
    return VK_OBJECT_TYPE_SAMPLER;
  }

  template<> VkObjectType GetVulkanObjectType(VkImageView)
  {
    return VK_OBJECT_TYPE_IMAGE_VIEW;
  }

  template<> VkObjectType GetVulkanObjectType(VkBufferView)
  {
    return VK_OBJECT_TYPE_BUFFER_VIEW;
  }

  template<> VkObjectType GetVulkanObjectType(VkCommandPool)
  {
    return VK_OBJECT_TYPE_COMMAND_POOL;
  }

  template<> VkObjectType GetVulkanObjectType(VkCommandBuffer)
  {
    return VK_OBJECT_TYPE_COMMAND_BUFFER;
  }

  template<> VkObjectType GetVulkanObjectType(VkDescriptorPool)
  {
    return VK_OBJECT_TYPE_DESCRIPTOR_POOL;
  }

  template<> VkObjectType GetVulkanObjectType(VkDescriptorSet)
  {
    return VK_OBJECT_TYPE_DESCRIPTOR_SET;
  }

  template<> VkObjectType GetVulkanObjectType(VkDescriptorSetLayout)
  {
    return VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT;
  }

  template<> VkObjectType GetVulkanObjectType(VkFence)
  {
    return VK_OBJECT_TYPE_FENCE;
  }

  template<> VkObjectType GetVulkanObjectType(VkSemaphore)
  {
    return VK_OBJECT_TYPE_SEMAPHORE;
  }

  template<> VkObjectType GetVulkanObjectType(VkEvent)
  {
    return VK_OBJECT_TYPE_EVENT;
  }

  template<> VkObjectType GetVulkanObjectType(VkQueryPool)
  {
    return VK_OBJECT_TYPE_QUERY_POOL;
  }

  template<> VkObjectType GetVulkanObjectType(VkFramebuffer)
  {
    return VK_OBJECT_TYPE_FRAMEBUFFER;
  }

  template<> VkObjectType GetVulkanObjectType(VkRenderPass)
  {
    return VK_OBJECT_TYPE_RENDER_PASS;
  }

  template<> VkObjectType GetVulkanObjectType(VkPipeline)
  {
    return VK_OBJECT_TYPE_PIPELINE;
  }

  template<> VkObjectType GetVulkanObjectType(VkPipelineLayout)
  {
    return VK_OBJECT_TYPE_PIPELINE_LAYOUT;
  }

  template<> VkObjectType GetVulkanObjectType(VkShaderModule)
  {
    return VK_OBJECT_TYPE_SHADER_MODULE;
  }

  //template <> VkObjectType GetVulkanObjectType(VkImageView)         { return VK_OBJECT_TYPE_IMAGE_VIEW;             }
  template<> VkObjectType GetVulkanObjectType(VkQueue)
  {
    return VK_OBJECT_TYPE_QUEUE;
  }

  template<typename HandleType>
  void SetVulkanObjectName(VkDevice device, HandleType handle, const std::string &name)
  {
    if (g_pfnSetDebugUtilsObjectNameEXT != nullptr)
    {
      VkDebugUtilsObjectNameInfoEXT nameInfo{};
      nameInfo.sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
      nameInfo.objectType   = GetVulkanObjectType(handle);
      nameInfo.objectHandle = reinterpret_cast<uint64_t>(handle);
      nameInfo.pObjectName  = name.c_str();
      //g_pfnSetDebugUtilsObjectNameEXT(device, &nameInfo);
    }
  }
};

MemoryAllocator::MemoryAllocator(VkPhysicalDevice physicalDevice, VkDevice device) : physicalDevice(physicalDevice),
                                                                                     device(device) {}

MemoryAllocator::~MemoryAllocator()
{
  for (auto *pool: pools)
  {
    if (pool != VK_NULL_HANDLE) delete pool;
  }
}

Allocation MemoryAllocator::allocate(VkMemoryRequirements requirements,
                                     VkMemoryPropertyFlags desiredFlags,
                                     const std::string &debugName)
{
  uint32_t memoryType = findMemoryType(requirements.memoryTypeBits, desiredFlags);
  for (auto *pool: pools)
  {
    if (pool->getMemoryTypeIndex() == memoryType)
    {
      Allocation result;
      if (pool->allocate(requirements.size, requirements.alignment, result))
      {
        return result;
      }
    }
  }
  VkDeviceSize poolSize = std::max(requirements.size * 8, (VkDeviceSize) 256 * 1024 * 1024); // 256MB 기본
  MemoryPool *newPool   = new MemoryPool(device, memoryType, poolSize);
  pools.push_back(newPool);
  Allocation result;
  if (!newPool->allocate(requirements.size, requirements.alignment, result))
  {
    throw std::runtime_error("Failed to allocate memory from new pool");
  }
  return result;
}

void MemoryAllocator::free(Allocation allocation, VkDeviceSize size)
{
  for (auto *pool: pools)
  {
    if (pool->getMemory() == allocation.memory)
    {
      pool->free(allocation.offset, size);
      return;
    }
  }
  //throw std::runtime_error("Tried to free memory from unknown pool");
}

VkDevice MemoryAllocator::getDevice()
{
  return device;
}

VkPhysicalDevice MemoryAllocator::getPhysicalDevice()
{
  return physicalDevice;
}

uint32_t MemoryAllocator::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
  {
    if ((typeFilter & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
    {
      return i;
    }
  }
  throw std::runtime_error("Failed to find suitable memory type");
}