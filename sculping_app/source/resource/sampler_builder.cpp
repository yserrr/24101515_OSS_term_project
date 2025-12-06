#include "sampler_builder.hpp"
#include <spdlog/spdlog.h>
#include "common.hpp"

SamplerBuilder::SamplerBuilder(VkDevice device) : device(device), dftSampler(VK_NULL_HANDLE)
{
  baseSampler();
}

SamplerBuilder::~SamplerBuilder()
{
  if (dftSampler != VK_NULL_HANDLE)
  {
    vkDestroySampler(device, dftSampler, nullptr);
  }
  for (VkSampler sampler: samplerPool_)
  {
    if (sampler != VK_NULL_HANDLE)
    {
      vkDestroySampler(device, sampler, nullptr);
    }
  }
}

VkSampler SamplerBuilder::get() const
{
  return dftSampler;
}

VkSampler SamplerBuilder::builderSampler(const SamplerDesc &desc)
{
  VkSamplerCreateInfo samplerInfo     = {};
  samplerInfo.sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter               = desc.mag;
  samplerInfo.minFilter               = desc.min;
  samplerInfo.mipmapMode              = desc.mipmapMode;
  samplerInfo.addressModeU            = desc.addressU;
  samplerInfo.addressModeV            = desc.addressV;
  samplerInfo.addressModeW            = desc.addressW;
  samplerInfo.mipLodBias              = desc.mipLodBias;
  samplerInfo.anisotropyEnable        = VK_TRUE;
  samplerInfo.maxAnisotropy           = 16.0f;
  samplerInfo.compareEnable           = VK_FALSE;
  samplerInfo.compareOp               = VK_COMPARE_OP_ALWAYS;
  samplerInfo.minLod                  = 0.0f;
  samplerInfo.maxLod                  = static_cast<float>(VK_LOD_CLAMP_NONE);
  samplerInfo.borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  VkSampler sampler                   = VK_NULL_HANDLE;
  VK_ASSERT(vkCreateSampler(device, &samplerInfo, nullptr, &sampler));
  samplerPool_.push_back(sampler);
  return sampler;
}

void SamplerBuilder::baseSampler()
{
  spdlog::info("create sampler");
  VkSamplerCreateInfo samplerInfo{};
  samplerInfo.sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter               = VK_FILTER_LINEAR;
  samplerInfo.minFilter               = VK_FILTER_LINEAR;
  samplerInfo.addressModeU            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeV            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.anisotropyEnable        = VK_FALSE;
  samplerInfo.maxAnisotropy           = 16;
  samplerInfo.borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  samplerInfo.compareEnable           = VK_FALSE;
  samplerInfo.compareOp               = VK_COMPARE_OP_ALWAYS;
  samplerInfo.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerInfo.mipLodBias              = 0.0f;
  samplerInfo.minLod                  = 0.0f;
  samplerInfo.maxLod                  = VK_LOD_CLAMP_NONE;
  if (vkCreateSampler(device, &samplerInfo, nullptr, &dftSampler) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to create sampler!");
  }
}