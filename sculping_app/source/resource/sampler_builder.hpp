#ifndef SAMPLER_HPP
#define SAMPLER_HPP
#include <vulkan/vulkan.h>
#include <stdexcept>
#include <vector>

struct SamplerDesc{
  VkFilter mag                   = VK_FILTER_LINEAR;
  VkFilter min                   = VK_FILTER_LINEAR;
  VkSamplerMipmapMode mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  VkSamplerAddressMode addressU  = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  VkSamplerAddressMode addressV  = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  VkSamplerAddressMode addressW  = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  float mipLodBias               = 0.0f;
  float maxAnisotropy            = 0.0f;
  float minLod                   = 0.0f;
  float maxLod                   = VK_LOD_CLAMP_NONE;
  bool enableCompare             = false;
  VkBorderColor borderColor      = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  VkCompareOp compareOp          = VK_COMPARE_OP_LESS_OR_EQUAL;
};

class SamplerBuilder{
public:
  SamplerBuilder(VkDevice device);
  ~SamplerBuilder();
  VkSampler get() const;
  VkSampler builderSampler(const SamplerDesc &desc);

private:
  VkDevice device;
  VkSampler dftSampler;
  std::vector<VkSampler> samplerPool_;
  void baseSampler();
};

#endif