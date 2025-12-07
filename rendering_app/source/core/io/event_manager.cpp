#include <event_manager.hpp>
#include "io.hpp"
//#include "../sculptor/sculptor_act.hpp"

//EventManager_->getKey();
//EventManager_->wheelUpdate();
//EventManager_->getMouseEvent();
//uIRenderer->uploadImageToUI();
//EventManager_->moved  = false;

void EventManager::init()
{
  offscreenTarget.resize(gpu::ctx__->renderingContext.maxInflight__);
}

void EventManager::updateUITexture()
{
  if (!offscreenTarget[gpu::ctx__->renderingContext.currentFrame__])
  {
    //{
    //  ui->offscreenTagets.updated[gpu::ctx__->renderingContext.currentFrame__] = true;
//
    //  offscreenTarget[gpu::ctx__->renderingContext.currentFrame__] = true;
    //  auto view = pRenderpassBuilder_->gBufferAlbedoHandle_[gpu::ctx__->renderingContext.currentFrame__]->imageView__;
    //  gpu::DescriptorSet textureDesc = ImGui_ImplVulkan_AddTexture(gpu::ctx__->pDescriptorAllocator->samplerBuilder_.
    //                                                                           dftSampler,
    //                                                               view,
    //                                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
//
    //  ui->offscreenTagets.albedoTargets[gpu::ctx__->renderingContext.currentFrame__].descriptorSet = textureDesc;
    //  ui->offscreenTagets.albedoTargets[gpu::ctx__->renderingContext.currentFrame__].index = pRenderpassBuilder_->
    //    gBufferAlbedoHandle_[gpu::ctx__->renderingContext.currentFrame__]->descriptorArrayIndex__;
    //}
    //{
    //  auto view = pRenderpassBuilder_->gBufferPositionHandle_[gpu::ctx__->renderingContext.currentFrame__]->imageView__;
    //  gpu::DescriptorSet textureDesc = ImGui_ImplVulkan_AddTexture(gpu::ctx__->pDescriptorAllocator->samplerBuilder_.
    //                                                                           dftSampler,
    //                                                               view,
    //                                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
//
    //  ui->offscreenTagets.positionTargets[gpu::ctx__->renderingContext.currentFrame__].descriptorSet = textureDesc;
    //  ui->offscreenTagets.positionTargets[gpu::ctx__->renderingContext.currentFrame__].index = pRenderpassBuilder_->
    //    gBufferPositionHandle_[gpu::ctx__->renderingContext.currentFrame__]->descriptorArrayIndex__;
    //}
//
//
    //{
    //  auto view = pRenderpassBuilder_->gBufferNormalHandle_[gpu::ctx__->renderingContext.currentFrame__]->imageView__;
    //  gpu::DescriptorSet textureDesc = ImGui_ImplVulkan_AddTexture(gpu::ctx__->pDescriptorAllocator->samplerBuilder_.
    //                                                                           dftSampler,
    //                                                               view,
    //                                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
//
    //  ui->offscreenTagets.normalTargets[gpu::ctx__->renderingContext.currentFrame__].descriptorSet = textureDesc;
    //  ui->offscreenTagets.normalTargets[gpu::ctx__->renderingContext.currentFrame__].index = pRenderpassBuilder_->
    //    gBufferNormalHandle_[gpu::ctx__->renderingContext.currentFrame__]->descriptorArrayIndex__;
    //}
//
    //{
    //  auto view = pRenderpassBuilder_->depthAttachmentHandle_[gpu::ctx__->renderingContext.currentFrame__]->imageView__;
    //  gpu::DescriptorSet textureDesc = ImGui_ImplVulkan_AddTexture(gpu::ctx__->pDescriptorAllocator->samplerBuilder_.
    //                                                                           dftSampler,
    //                                                               view,
    //                                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
//
    //  ui->offscreenTagets.depthTargets[gpu::ctx__->renderingContext.currentFrame__].descriptorSet = textureDesc;
    //  ui->offscreenTagets.depthTargets[gpu::ctx__->renderingContext.currentFrame__].index = pRenderpassBuilder_->
    //    depthAttachmentHandle_[gpu::ctx__->renderingContext.currentFrame__]->descriptorArrayIndex__;
    //}
  }
}

void EventManager::moveProcessEvent()
{
  if (mns::io__.dirty_)
  {
    auto& state = mns::io__.keyState__;
    //if (state.keySpace) pRenderpassBuilder_->pipeline.polygonMode = VK_POLYGON_MODE_FILL;
    //if (state.keyCtrl) pRenderpassBuilder_->pipeline.polygonMode = VK_POLYGON_MODE_LINE;
    //if (state.keyAlt) pRenderpassBuilder_->pipeline.depthTest = !pRenderpassBuilder_->pipeline.depthTest ;
    ////if (state.key0) renderer_->viewMode = ViewMode::VR;
    ////if (state.key1) renderer_->viewMode = ViewMode::SINGLE;
    if (state.keyQ) glfwSetWindowShouldClose(gpu::ctx__->windowh__, GLFW_TRUE);
    if (state.keyT) pResourcesManager_->camera.noUpdate = !pResourcesManager_->camera.noUpdate;
  }
}

void EventManager::getMouseEvent()
{
}
