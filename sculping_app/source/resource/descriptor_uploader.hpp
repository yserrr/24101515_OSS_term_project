#ifndef MYPROJECT_DESCRIPTOR_SET_HPP
#define MYPROJECT_DESCRIPTOR_SET_HPP
#include "descriptor_layout_builder.hpp"
#include "descriptor_pool.hpp"

struct VulkanTexture;

using VkBindlessDescriptor = VkDescriptorSet;

class DescriptorUploader{
public:
  DescriptorUploader(VkDevice device);
  ~DescriptorUploader() = default;
  void UploadUboSet(VkBuffer srcBuffer,
                    VkDeviceSize srcSize,
                    VkDescriptorSet dstSet,
                    uint32_t dstBindingIndex,
                    uint32_t dstBindingArrayIndex=0,
                    uint32_t dstCount = 1);

  void uploadBindlessTextureSet(VulkanTexture *texture);
  void update();
  void writeSsboSet();
  void writeDynamicUboSet();

public:
  VkDevice device_;
  std::vector<VkWriteDescriptorSet> writedSets_;
  std::vector<VkDescriptorBufferInfo> bufferInfos_;
  std::vector<VkDescriptorImageInfo> imageInfos_;
  void free();
  //todo: implete
};

#endif //MYPROJECT_DESCRIPTOR_SET_HPP