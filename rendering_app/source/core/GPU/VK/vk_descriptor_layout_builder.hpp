#ifndef MYPROJECT_DESCRIPTOR_LAYOUT_BUILDER_HPP
#define MYPROJECT_DESCRIPTOR_LAYOUT_BUILDER_HPP

#include<vulkan/vulkan.h>
#include<unordered_map>
#include<vector>
#include<string>
#include "vk_resource.hpp"

///todo:
/// shader stage setting
/// layout: descriptor set and binding Size tracker

namespace gpu
{
  struct VkDescriptorLayoutBindingInfo
  {
    gpu::VkDescriptorFlag usage;
    VkShaderStageFlags stage;
    uint32_t bindingIndex;
    ///check descriptor set hash
  };
  class VkContext;

  class VkDescriptorLayoutBuilder
  {
    public:
    VkDescriptorLayoutBuilder(VkContext* pCtxt);

    ~VkDescriptorLayoutBuilder();

    VkDescriptorSetLayout createDescriptorSetLayout(std::vector<VkDescriptorLayoutBindingInfo>& infos);

    private:
    void buildUboLayout(std::vector<VkDescriptorSetLayoutBinding>& bindings,
                        uint32_t bindingIndex,
                        VkShaderStageFlags stage,
                        std::vector<VkDescriptorBindingFlags>& bindingFlags,
                        uint32_t arrayCnt = 1);

    void buildBindlessTextureLayout(std::vector<VkDescriptorSetLayoutBinding>& bindings,
                                    uint32_t bindingIndex,
                                    VkShaderStageFlags stage,
                                    std::vector<VkDescriptorBindingFlags>& bindingFlags,
                                    uint32_t arrayCnt = gpu::VK_BINDELSS_TEXTURE_ARRAY_COUNT);

    void buildDynamicUboLayout(std::vector<VkDescriptorSetLayoutBinding>& bindings,
                               uint32_t bindingIndex,
                               VkShaderStageFlags stage,
                               std::vector<VkDescriptorBindingFlags>& bindingFlags,
                               uint32_t arrayCnt = gpu::VK_DYNAMIC_UBO_ARRAY_COUNT);

    void buildSSBOLayout(std::vector<VkDescriptorSetLayoutBinding>& bindings,
                         uint32_t bindingIndex,
                         VkShaderStageFlags stage,
                         std::vector<VkDescriptorBindingFlags>& bindingFlags,
                         uint32_t arrayCnt = 1);

    void buildTextureLayout(std::vector<VkDescriptorSetLayoutBinding>& bindings,
                            uint32_t bindingIndex,
                            VkShaderStageFlags stage,
                            std::vector<VkDescriptorBindingFlags>& bindingFlags,
                            uint32_t arrayCnt);
    VkContext* pCtxt_;
    std::unordered_map<uint32_t, VkDescriptorSetLayout> layoutHash;
  };
}

#endif
