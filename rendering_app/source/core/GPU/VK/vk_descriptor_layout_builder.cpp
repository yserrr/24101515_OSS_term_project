#include <utility>
#include "vk_common.hpp"
#include "vk_descriptor_layout_builder.hpp"
#include "vk_context.hpp"

namespace gpu
{
  VkDescriptorLayoutBuilder::VkDescriptorLayoutBuilder(VkContext* pCtxt) : pCtxt_(pCtxt)
  {
  }

  VkDescriptorLayoutBuilder::~VkDescriptorLayoutBuilder() = default;

  VkDescriptorSetLayout VkDescriptorLayoutBuilder::createDescriptorSetLayout(
    std::vector<VkDescriptorLayoutBindingInfo>& infos)
  {
    std::vector<VkDescriptorSetLayoutBinding> bindings{};
    std::vector<VkDescriptorBindingFlags> bindingFlags{};
    bindings.reserve(infos.size());
    bindingFlags.reserve(infos.size());
    for (VkDescriptorLayoutBindingInfo& info : infos)
    {
      switch (info.usage)
      {
        case(gpu::DescriptorFlag::DYNAMIC_UBO):
        {
          buildDynamicUboLayout(bindings, info.bindingIndex, info.stage, bindingFlags);
          break;
        }
        case(gpu::DescriptorFlag::UBO):
        {
          buildUboLayout(bindings, info.bindingIndex, info.stage, bindingFlags);
          break;
        }
        case(gpu::DescriptorFlag::TEXTURE_BINDLESS):
        {
          buildBindlessTextureLayout(bindings, info.bindingIndex, info.stage, bindingFlags);
          break;
        }
        case(gpu::DescriptorFlag::SSBO):
        {
          buildSSBOLayout(bindings, info.bindingIndex, info.stage, bindingFlags);
          break;
        }
        case(gpu::DescriptorFlag::TEXTURE):
        {
          buildTextureLayout(bindings, info.bindingIndex, info.stage, bindingFlags, 1);
          break;
        }
        default:
          break;
      }
    }

    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
    bindingFlagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    bindingFlagsInfo.bindingCount = static_cast<uint32_t>(bindingFlags.size());
    bindingFlagsInfo.pBindingFlags = bindingFlags.data();

    VkDescriptorSetLayoutCreateInfo layoutCreateInfo{};
    layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCreateInfo.bindingCount = bindings.size();
    layoutCreateInfo.pBindings = bindings.data();
    layoutCreateInfo.flags |= (VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT );
    layoutCreateInfo.pNext = &bindingFlagsInfo;

    VkDescriptorSetLayout layout{};
    VK_ASSERT(vkCreateDescriptorSetLayout(pCtxt_->deviceh__ ,
                &layoutCreateInfo,
                nullptr,
                &layout));
    return layout;
  }

  void VkDescriptorLayoutBuilder::buildUboLayout(std::vector<VkDescriptorSetLayoutBinding>& bindings,
                                                 uint32_t bindingIndex,
                                                 VkShaderStageFlags stage,
                                                 std::vector<VkDescriptorBindingFlags>& bindingFlags,
                                                 uint32_t arrayCnt)
  {
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = bindingIndex;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount = arrayCnt;
    binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    binding.pImmutableSamplers = nullptr;
    bindingFlags.push_back(0);
    bindings.push_back(binding);
  }

  void VkDescriptorLayoutBuilder::buildBindlessTextureLayout(std::vector<VkDescriptorSetLayoutBinding>& bindings,
                                                             uint32_t bindingIndex,
                                                             VkShaderStageFlags stage,
                                                             std::vector<VkDescriptorBindingFlags>& bindingFlags,
                                                             uint32_t arrayCnt)
  {
    VkDescriptorSetLayoutBinding bindlessTextureBinding{};
    bindlessTextureBinding.binding = bindingIndex;
    bindlessTextureBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindlessTextureBinding.descriptorCount = arrayCnt; //texture count
    bindlessTextureBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindlessTextureBinding.pImmutableSamplers = nullptr;

    bindingFlags.push_back(
                           VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                           VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT );
    bindings.push_back(bindlessTextureBinding);
  }

  void VkDescriptorLayoutBuilder::buildDynamicUboLayout(std::vector<VkDescriptorSetLayoutBinding>& bindings,
                                                        uint32_t bindingIndex,
                                                        VkShaderStageFlags stage,
                                                        std::vector<VkDescriptorBindingFlags>& bindingFlags,
                                                        uint32_t arrayCnt)
  {
    VkDescriptorSetLayoutBinding dynamicBinding{};
    dynamicBinding.binding = bindingIndex;
    dynamicBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    dynamicBinding.descriptorCount = arrayCnt;
    dynamicBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    dynamicBinding.pImmutableSamplers = nullptr;
    bindingFlags.push_back(0);
    bindings.push_back(dynamicBinding);
  }

  void VkDescriptorLayoutBuilder::buildSSBOLayout(std::vector<VkDescriptorSetLayoutBinding>& bindings,
                                                  uint32_t bindingIndex,
                                                  VkShaderStageFlags stage,
                                                  std::vector<VkDescriptorBindingFlags>& bindingFlags,
                                                  uint32_t arrayCnt)
  {
    VkDescriptorSetLayoutBinding ssboBinding{};
    ssboBinding.binding = bindingIndex;
    ssboBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ssboBinding.descriptorCount = arrayCnt;
    ssboBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
    ssboBinding.pImmutableSamplers = nullptr;
    bindingFlags.push_back(0);
    bindings.push_back(ssboBinding);
  }

  void VkDescriptorLayoutBuilder::buildTextureLayout(std::vector<VkDescriptorSetLayoutBinding>& bindings,
                                                     uint32_t bindingIndex,
                                                     VkShaderStageFlags stage,
                                                     std::vector<VkDescriptorBindingFlags>& bindingFlags,
                                                     uint32_t arrayCnt)
  {
    VkDescriptorSetLayoutBinding textureBinding{};
    textureBinding.binding = bindingIndex;
    textureBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    textureBinding.descriptorCount = arrayCnt;
    textureBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    textureBinding.pImmutableSamplers = nullptr;
    bindingFlags.push_back(0);
    bindings.push_back(textureBinding);
  }
}
