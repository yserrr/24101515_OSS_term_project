#include "render_target_film.hpp"

void flm::RenderTragetFilm::buildFrameResources(uint32_t index)
{
  swapchainAttachment_ = gpu::VkGraphBuilder::buildSwapchainAttachment(index);
  depthAttachmentHandle_ = (gpu::VkGraphBuilder::buildDepthAttachment());
  gBufferAlbedoHandle_ = (gpu::VkGraphBuilder::buildFrameAttachment(gpu::FORMAT_B8G8R8A8_SRGB));
  gBufferPositionHandle_ = (gpu::VkGraphBuilder::buildFrameAttachment(gpu::FORMAT_R16G16B16A16_SFLOAT));
  gBufferNormalHandle_ = (gpu::VkGraphBuilder::buildFrameAttachment(gpu::FORMAT_R16G16B16A16_SNORM));
  gBufferRoughnessHandle_ = (gpu::VkGraphBuilder::buildFrameAttachment(gpu::FORMAT_R16_UNORM));
  gBufferMetalicHandle_ = (gpu::VkGraphBuilder::buildFrameAttachment(gpu::FORMAT_R8_UNORM));
  lightningAttachmentHandle_ = (gpu::VkGraphBuilder::buildFrameAttachment(gpu::FORMAT_R16G16B16A16_SFLOAT));
  bloomingExtractAttachment_ = (gpu::VkGraphBuilder::buildFrameAttachment(gpu::FORMAT_R32G32B32A32_SFLOAT));
  bloomingBlurAttachment_ = (gpu::VkGraphBuilder::buildFrameAttachment(gpu::FORMAT_R32G32B32A32_SFLOAT));
  toneMappingAttachment_ = (gpu::VkGraphBuilder::buildFrameAttachment(gpu::FORMAT_R16G16B16A16_SFLOAT));
  gammaAttachment_ = (gpu::VkGraphBuilder::buildFrameAttachment(gpu::FORMAT_B8G8R8A8_SRGB));
  shadowAttachmentHandleXp_ = (gpu::VkGraphBuilder::buildDepthAttachment());
  shadowAttachmentHandleXm_ = (gpu::VkGraphBuilder::buildDepthAttachment());
  shadowAttachmentHandleYp_ = (gpu::VkGraphBuilder::buildDepthAttachment());
  shadowAttachmentHandleYm_ = (gpu::VkGraphBuilder::buildDepthAttachment());
  shadowAttachmentHandleZp_ = (gpu::VkGraphBuilder::buildDepthAttachment());
  shadowAttachmentHandleZm_ = (gpu::VkGraphBuilder::buildDepthAttachment());
}

void flm::RenderTragetFilm::updateFrameConstant()
{
  this->renderAttachment.DepthBuffer =
    depthAttachmentHandle_->descriptorArrayIndex__;
  this->renderAttachment.gBufferAlbedo =
    gBufferAlbedoHandle_->descriptorArrayIndex__;
  this->renderAttachment.gBufferNormal =
    gBufferNormalHandle_->descriptorArrayIndex__;
  this->renderAttachment.gBufferPositon =
    gBufferPositionHandle_->descriptorArrayIndex__;
  this->renderAttachment.gBufferRoughness =
    gBufferRoughnessHandle_->descriptorArrayIndex__;
  this->renderAttachment.lightningBuffer =
    lightningAttachmentHandle_->descriptorArrayIndex__;
  this->renderAttachment.tonemapping = toneMappingAttachment_->descriptorArrayIndex__;
  this->renderAttachment.gamma = gammaAttachment_->descriptorArrayIndex__;
  this->renderAttachment.bloomingExtract = bloomingExtractAttachment_->descriptorArrayIndex__;
  this->renderAttachment.bloomingBlur = bloomingBlurAttachment_->descriptorArrayIndex__;
}

void flm::RenderTragetFilm::clearGbuffer()
{
  gBufferAlbedoHandle_->lastWriter__ = nullptr;
  gBufferNormalHandle_->lastWriter__ = nullptr;
  gBufferPositionHandle_->lastWriter__ = nullptr;
  gBufferRoughnessHandle_->lastWriter__ = nullptr;
  gBufferAlbedoHandle_->writen__ = false;
  gBufferNormalHandle_->writen__ = false;
  gBufferPositionHandle_->writen__ = false;
  gBufferRoughnessHandle_->writen__ = false;
}

void flm::RenderTragetFilm::clearDepthBuffer()
{
  depthAttachmentHandle_->writen__ = false;
  depthAttachmentHandle_->lastWriter__ = nullptr;
}

void flm::RenderTragetFilm::clearShadowBuffer()
{
  shadowAttachmentHandleXm_->lastWriter__ = nullptr;
  shadowAttachmentHandleXm_->writen__ = false;
  shadowAttachmentHandleYm_->lastWriter__ = nullptr;
  shadowAttachmentHandleYm_->writen__ = false;
  shadowAttachmentHandleZm_->lastWriter__ = nullptr;
  shadowAttachmentHandleZm_->writen__ = false;
  shadowAttachmentHandleXp_->lastWriter__ = nullptr;
  shadowAttachmentHandleXp_->writen__ = false;
  shadowAttachmentHandleYp_->lastWriter__ = nullptr;
  shadowAttachmentHandleYp_->writen__ = false;
  shadowAttachmentHandleZp_->lastWriter__ = nullptr;
  shadowAttachmentHandleZp_->writen__ = false;
}

void flm::RenderTragetFilm::clearLightBuffer()
{
  lightningAttachmentHandle_->writen__ = false;
  lightningAttachmentHandle_->lastWriter__ = nullptr;
}

void flm::RenderTragetFilm::clearBloomingBuffer()
{
  bloomingBlurAttachment_->writen__ = false;
  bloomingBlurAttachment_->lastWriter__ = nullptr;

  bloomingExtractAttachment_->writen__ = false;
  bloomingExtractAttachment_->lastWriter__ = nullptr;
}
