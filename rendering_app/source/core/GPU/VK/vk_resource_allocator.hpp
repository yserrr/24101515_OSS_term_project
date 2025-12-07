//
// Created by ljh on 25. 10. 10..
//

#ifndef MYPROJECT_VK_RESOURCE_ALLOCATOR_HPP
#define MYPROJECT_VK_RESOURCE_ALLOCATOR_HPP
#include <cstring>
#include <memory>
#include <unordered_map>
#include "vk_sampler_builder.hpp"
#include "vk_memory_allocator.hpp"
#include "vk_host_buffer.h"
#include "vk_texture.hpp"
namespace gpu
{
  class VkContext;

  class VkResourceAllocator
  {
    friend class VkDiscardPool;

    public:
    VkResourceAllocator(VkContext* pCtxt);
    ~VkResourceAllocator();
    void buildMeshNode(VkMeshBuffer* buffer);
    void buildFrameAttachment(VkFrameAttachment* image);
    void uploadBufferTransferPass(VkBuffer src,
                                  VkBuffer dst,
                                  VkDeviceSize srcOffset,
                                  VkDeviceSize dstOffset,
                                  VkDeviceSize size);

    void uploadCopyPass(VkBuffer src,
                             VkBuffer dst,
                             VkDeviceSize srcOffset,
                             VkDeviceSize dstOffset,
                             VkDeviceSize size);

    void buildImageCopyPass(VkBuffer buffer,
                            VkTexture* texture);

    VkBuffer buildBufferHandle(VkDeviceSize size,
                           VkBufferUsageFlags usage);

    void buildKtxTexture(gpu::VkTexture* texture);
    void buildTexture(gpu::VkTexture* texture);

    VkHostBuffer getStagingBuffer(void* data,
                                  VkDeviceSize size);
    VkAllocation mBindBuffer(VkBuffer buffer,
                             VkMemoryPropertyFlags desiredFlag);
    VkAllocation mBindImage(VkImage image,
                            VkMemoryPropertyFlags desiredFlag);
    void buildImageBarrierPass(VkImage img,
                               VkPipelineStageFlagBits src,
                               VkPipelineStageFlags dst, VkAccessFlags srcAccess, VkAccessFlags dstAccess, VkImageLayout oldLayout, VkImageLayout
                               newLayout);

    void allocateSampler();
    void allocateDescriptorSet();
    void allocateDescriptorSetLayout();
    void allocatePipeline();
    void hostUpdate(VkHostBuffer* buffer__);
    void hostUpdate(VkTexture* texture);

    private:
    VkContext* pCtxt;
    VkPhysicalDeviceProperties physicalDeviceProperties;
    uint32_t textureBindingSlot_;
    gpu::VkSamplerBuilder samplerBuilder_;
    //std::unique_ptr<GPU::VkShaderPool> fragShaderPool_;
    //std::unique_ptr<GPU::VkShaderPool> vertexShaderPool_;
    //std::unique_ptr<GPU::VkPipelinePool> pipelinePool_;
    //ptr binding std::unique_ptr<VkDescriptoraAlocator> pDescriptoraAllocator_;
    //std::unique
  };
}

#endif //MYPROJECT_VK_RESOURCE_ALLOCATOR_HPP
