#include "frame_graph.hpp"
#include "context.hpp"

namespace dag
{
  FrameGraph::FrameGraph(uint32_t frameIndex,
                         flm::RenderTragetFilm* film) :
    frameIndex_(frameIndex),
    renderTargetFilm_(film)
  {
  }

  void FrameGraph::init()
  {
    addGBufferWritePass();
    addShadowPass();
    addLightningPass();
    addTonemapPass();
    addGammaCorrectionPass();
    offscreenRenderPass();
  //  addUiDraw();
    renderPasses_ = gpu::ctx__->pGraphBuilder->build(uploadPasses_, frameIndex_);
  }

  void FrameGraph::addDepthOnlyPass()
  {
    auto* pass = passPool_->depthOnlyPass[frameIndex_].get();
    pass->clear();
    renderTargetFilm_->clearDepthBuffer();
    pass->write__.push_back(renderTargetFilm_->depthAttachmentHandle_.get());
    pass->execute = [this,pass ](gpu::CommandBuffer cmd)
    {
      gpu::cmdBindDescriptorSets(cmd,
                                 VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 pipeline_->pipelineLayout_h,
                                 0,
                                 1,
                                 &gpu::ctx__->pDescriptorAllocator->descriptorSets
                                 [frameIndex_],
                                 0,
                                 nullptr);
      gpu::cmdBeginRendering(cmd, pass);
      pushFrameConstant(cmd);
      gpu::cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->depthWritePipeline__);
      pipeline_->cmdSetPolygonMode(cmd, pipeline_->polygonMode);
      vkCmdSetDepthTestEnable(cmd, VK_TRUE);
      gpu::cmdSetViewports(cmd,
                           0.0,
                           0.0,
                           (float)gpu::ctx__->pSwapChainContext->extent__.width,
                           (float)gpu::ctx__->pSwapChainContext->extent__.height
                          );
      for (auto handle : cpu::ctx__->drawHandle_)
      {
        pushModelConstant(cmd, &handle->constant);
        handle->mesh->draw(cmd);
      }
      gpu::cmdEndRendering(cmd);
    };
    uploadPasses_.push_back(pass);
  }

  void FrameGraph::addShadowPass()
  {
    auto* pass = passPool_->shadowPass[frameIndex_].get();
    pass->clear();
    renderTargetFilm_->clearShadowBuffer();
    pass->write__.push_back(renderTargetFilm_->shadowAttachmentHandleXm_.get());
    pass->write__.push_back(renderTargetFilm_->shadowAttachmentHandleXp_.get());
    pass->write__.push_back(renderTargetFilm_->shadowAttachmentHandleYm_.get());
    pass->write__.push_back(renderTargetFilm_->shadowAttachmentHandleYp_.get());
    pass->write__.push_back(renderTargetFilm_->shadowAttachmentHandleZm_.get());
    pass->write__.push_back(renderTargetFilm_->shadowAttachmentHandleZp_.get());
    pass->execute = [this,pass ](gpu::CommandBuffer cmd)
    {
      gpu::cmdBindDescriptorSets(cmd,
                                 VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 pipeline_->pipelineLayout_h,
                                 0,
                                 1,
                                 &gpu::ctx__->pDescriptorAllocator->descriptorSets
                                 [frameIndex_],
                                 0,
                                 nullptr);
      gpu::cmdBeginRendering(cmd, pass);
      pushFrameConstant(cmd);
      gpu::cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->shadowWritePipeline__);
      pipeline_->cmdSetPolygonMode(cmd, pipeline_->polygonMode);
      vkCmdSetDepthTestEnable(cmd, VK_TRUE);
      gpu::cmdSetViewports(cmd,
                           0.0,
                           0.0,
                           (float)gpu::ctx__->pSwapChainContext->extent__.width,
                           (float)gpu::ctx__->pSwapChainContext->extent__.height
                          );
      for (auto handle : cpu::ctx__->drawHandle_)
      {
        pushModelConstant(cmd, &handle->constant);
        handle->mesh->draw(cmd);
      }
      gpu::cmdEndRendering(cmd);
    };
    uploadPasses_.push_back(pass);
  }

  void FrameGraph::addGBufferWritePass()
  {
    auto* pass = passPool_->gBufferPass[frameIndex_].get();
    pass->clear();
    renderTargetFilm_->clearGbuffer();
    renderTargetFilm_->clearDepthBuffer();
    pass->write__.push_back(renderTargetFilm_->gBufferPositionHandle_.get());
    pass->write__.push_back(renderTargetFilm_->gBufferAlbedoHandle_.get());
    pass->write__.push_back(renderTargetFilm_->gBufferNormalHandle_.get());
    pass->write__.push_back(renderTargetFilm_->gBufferRoughnessHandle_.get());
    pass->write__.push_back(renderTargetFilm_->depthAttachmentHandle_.get());
    pass->execute = [this, pass](gpu::CommandBuffer cmd)
    {
      gpu::cmdBindDescriptorSets(cmd,
                                 VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 pipeline_->pipelineLayout_h,
                                 0,
                                 1,
                                 &gpu::ctx__->pDescriptorAllocator->descriptorSets
                                 [frameIndex_],
                                 0,
                                 nullptr);

      gpu::cmdBeginRendering(cmd, pass);
      renderTargetFilm_->updateFrameConstant();
      gpu::cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->gBufferWritePipeline__);
      pipeline_->cmdSetPolygonMode(cmd, pipeline_->polygonMode);
      vkCmdSetDepthTestEnable(cmd, pipeline_->depthTest);
      gpu::cmdSetViewports(cmd,
                           0.0,
                           0.0,
                           (float)gpu::ctx__->pSwapChainContext->extent__.width,
                           (float)gpu::ctx__->pSwapChainContext->extent__.height
                          );
      for (auto handle : cpu::ctx__->drawHandle_)
      {
        pushModelConstant(cmd, &handle->constant);
        handle->mesh->draw(cmd);
      }
      gpu::cmdEndRendering(cmd);
    };
    uploadPasses_.push_back(pass);
  }

  void FrameGraph::addLightningPass()
  {
    auto* pass = passPool_->lightningPass[frameIndex_].get();
    pass->clear();
    renderTargetFilm_->clearLightBuffer();
    pass->write__.push_back(renderTargetFilm_->lightningAttachmentHandle_.get());
    pass->read__.push_back(renderTargetFilm_->shadowAttachmentHandleXm_.get());
    pass->read__.push_back(renderTargetFilm_->shadowAttachmentHandleXp_.get());
    pass->read__.push_back(renderTargetFilm_->shadowAttachmentHandleYm_.get());
    pass->read__.push_back(renderTargetFilm_->shadowAttachmentHandleYp_.get());
    pass->read__.push_back(renderTargetFilm_->shadowAttachmentHandleZm_.get());
    pass->read__.push_back(renderTargetFilm_->shadowAttachmentHandleZp_.get());
    pass->read__.push_back(renderTargetFilm_->gBufferAlbedoHandle_.get());
    pass->read__.push_back(renderTargetFilm_->gBufferPositionHandle_.get());
    pass->read__.push_back(renderTargetFilm_->gBufferNormalHandle_.get());
    pass->read__.push_back(renderTargetFilm_->gBufferRoughnessHandle_.get());
    pass->execute = [this, pass](gpu::CommandBuffer cmd)
    {
      gpu::cmdBindDescriptorSets(cmd,
                                 VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 pipeline_->pipelineLayout_h,
                                 0,
                                 1,
                                 &gpu::ctx__->pDescriptorAllocator->descriptorSets
                                 [frameIndex_],
                                 0,
                                 nullptr);
      gpu::cmdBeginRendering(cmd, pass);
      renderTargetFilm_->updateFrameConstant();
      pushFrameConstant(cmd);
      gpu::cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->lightningWritePipeline__);
      pipeline_->cmdSetPolygonMode(cmd, pipeline_->polygonMode);
      vkCmdSetDepthTestEnable(cmd, pipeline_->depthTest);
      gpu::cmdSetViewports(cmd,
                           0.0,
                           0.0,
                           (float)gpu::ctx__->pSwapChainContext->extent__.width,
                           (float)gpu::ctx__->pSwapChainContext->extent__.height
                          );
      gpu::cmdDrawQuad(cmd);
      gpu::cmdEndRendering(cmd);
    };
    uploadPasses_.push_back(pass);
  }

  void FrameGraph::addBloomingExtractPass()
  {
    auto* pass = passPool_->bloomingExtractPass[frameIndex_].get();
    pass->clear();
    renderTargetFilm_->bloomingExtractAttachment_->writen__ = false;
    renderTargetFilm_->bloomingExtractAttachment_->lastWriter__ = nullptr;
    pass->read__.push_back(renderTargetFilm_->lightningAttachmentHandle_.get());
    pass->write__.push_back(renderTargetFilm_->bloomingExtractAttachment_.get());
    pass->execute = [this, pass](gpu::CommandBuffer cmd)
    {
      gpu::cmdBindDescriptorSets(cmd,
                                 VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 pipeline_->pipelineLayout_h,
                                 0,
                                 1,
                                 &gpu::ctx__->pDescriptorAllocator->descriptorSets
                                 [frameIndex_],
                                 0,
                                 nullptr);
      gpu::cmdBeginRendering(cmd, pass);
      renderTargetFilm_->updateFrameConstant();
      pushFrameConstant(cmd);
      gpu::cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->bloomingExtractWritePipeline__);
      pipeline_->cmdSetPolygonMode(cmd, pipeline_->polygonMode);
      vkCmdSetDepthTestEnable(cmd, pipeline_->depthTest);
      gpu::cmdSetViewports(cmd,
                           0.0,
                           0.0,
                           (float)gpu::ctx__->pSwapChainContext->extent__.width,
                           (float)gpu::ctx__->pSwapChainContext->extent__.height
                          );
      gpu::cmdDrawQuad(cmd);
      gpu::cmdEndRendering(cmd);
    };
    uploadPasses_.push_back(pass);
  }

  void FrameGraph::addBloomingBlurPass()
  {
    auto* pass = passPool_->bloomingBlurPass[frameIndex_].get();
    pass->clear();
    renderTargetFilm_->bloomingBlurAttachment_->writen__ = false;
    renderTargetFilm_->bloomingBlurAttachment_->lastWriter__ = nullptr;
    pass->read__.push_back(renderTargetFilm_->bloomingExtractAttachment_.get());
    pass->write__.push_back(renderTargetFilm_->bloomingBlurAttachment_.get());
    pass->execute = [this, pass](gpu::CommandBuffer cmd)
    {
      gpu::cmdBindDescriptorSets(cmd,
                                 VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 pipeline_->pipelineLayout_h,
                                 0,
                                 1,
                                 &gpu::ctx__->pDescriptorAllocator->descriptorSets
                                 [frameIndex_],
                                 0,
                                 nullptr);
      gpu::cmdBeginRendering(cmd, pass);
      renderTargetFilm_->updateFrameConstant();
      pushFrameConstant(cmd);
      gpu::cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->bloomingBlurWritePipeline__);
      pipeline_->cmdSetPolygonMode(cmd, pipeline_->polygonMode);
      vkCmdSetDepthTestEnable(cmd, pipeline_->depthTest);
      gpu::cmdSetViewports(cmd,
                           0.0,
                           0.0,
                           (float)gpu::ctx__->pSwapChainContext->extent__.width,
                           (float)gpu::ctx__->pSwapChainContext->extent__.height
                          );
      gpu::cmdDrawQuad(cmd);
      gpu::cmdEndRendering(cmd);
    };
    uploadPasses_.push_back(pass);
  }


  void FrameGraph::addTonemapPass()
  {
    auto* pass = passPool_->tonemappingPass[frameIndex_].get();
    pass->clear();
    renderTargetFilm_->toneMappingAttachment_->writen__ = false;
    renderTargetFilm_->toneMappingAttachment_->lastWriter__ = nullptr;
    pass->read__.push_back(renderTargetFilm_->lightningAttachmentHandle_.get());
    pass->write__.push_back(renderTargetFilm_->toneMappingAttachment_.get());
    pass->execute = [this, pass](gpu::CommandBuffer cmd)
    {
      gpu::cmdBindDescriptorSets(cmd,
                                 VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 pipeline_->pipelineLayout_h,
                                 0,
                                 1,
                                 &gpu::ctx__->pDescriptorAllocator->descriptorSets
                                 [frameIndex_],
                                 0,
                                 nullptr);
      gpu::cmdBeginRendering(cmd, pass);
      renderTargetFilm_->updateFrameConstant();
      pushFrameConstant(cmd);
      gpu::cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->tonemappingWritePipeline__);
      pipeline_->cmdSetPolygonMode(cmd, pipeline_->polygonMode);
      vkCmdSetDepthTestEnable(cmd, pipeline_->depthTest);
      gpu::cmdSetViewports(cmd,
                           0.0,
                           0.0,
                           (float)gpu::ctx__->pSwapChainContext->extent__.width,
                           (float)gpu::ctx__->pSwapChainContext->extent__.height
                          );
      gpu::cmdDrawQuad(cmd);
      gpu::cmdEndRendering(cmd);
    };
    uploadPasses_.push_back(pass);
  }

  void FrameGraph::addGammaCorrectionPass()
  {
    auto* pass = passPool_->gammaPass[frameIndex_].get();
    pass->clear();
    renderTargetFilm_->gammaAttachment_->writen__ = false;
    renderTargetFilm_->gammaAttachment_->lastWriter__ = nullptr;
    pass->read__.push_back(renderTargetFilm_->toneMappingAttachment_.get());
    pass->write__.push_back(renderTargetFilm_->gammaAttachment_.get());
    pass->execute = [this, pass](gpu::CommandBuffer cmd)
    {
      gpu::cmdBindDescriptorSets(cmd,
                                 VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 pipeline_->pipelineLayout_h,
                                 0,
                                 1,
                                 &gpu::ctx__->pDescriptorAllocator->descriptorSets
                                 [frameIndex_],
                                 0,
                                 nullptr);
      gpu::cmdBeginRendering(cmd, pass);
      renderTargetFilm_->updateFrameConstant();
      pushFrameConstant(cmd);
      gpu::cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->gammaWritePipeline__);
      pipeline_->cmdSetPolygonMode(cmd, pipeline_->polygonMode);
      vkCmdSetDepthTestEnable(cmd, pipeline_->depthTest);
      gpu::cmdSetViewports(cmd,
                           0.0,
                           0.0,
                           (float)gpu::ctx__->pSwapChainContext->extent__.width,
                           (float)gpu::ctx__->pSwapChainContext->extent__.height
                          );
      gpu::cmdDrawQuad(cmd);
      gpu::cmdEndRendering(cmd);
    };
    uploadPasses_.push_back(pass);
  }

  void FrameGraph::addUiDraw()
  {
    auto* pass = passPool_->uiDrawPass[frameIndex_].get();
    pass->clear();
    pass->dependency__ = {};
    pass->dependent__ = {};
    pass->linkCount = 0;

    pass->execute = [pass](gpu::CommandBuffer cmd)
    {
      gpu::cmdBeginRendering(cmd, pass);
      ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
      gpu::cmdEndRendering(cmd);
    };
    renderTargetFilm_->swapchainAttachment_.get()->lastWriter__ = nullptr;
    renderTargetFilm_->swapchainAttachment_.get()->writen__ = false;

    pass->write__.push_back(renderTargetFilm_->swapchainAttachment_.get());
    uploadPasses_.push_back(pass);
  }

  void FrameGraph::offscreenRenderPass()
  {
    auto* pass = passPool_->swapchainRenderPass[frameIndex_].get();
    pass->clear();
    pass->execute = [this, pass](gpu::CommandBuffer cmd)
    {
      gpu::cmdBeginRendering(cmd, pass);
      renderTargetFilm_->updateFrameConstant();
      pushFrameConstant(cmd);
      gpu::cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->offScreenTexture);
      vkCmdSetDepthTestEnable(cmd, VK_FALSE);
      pipeline_->cmdSetPolygonMode(cmd, pipeline_->polygonMode);
      gpu::cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->offScreenTexture);

      if (cpu::ctx__->debug)
      {
        gpu::cmdSetViewports(cmd,
                             0.0,
                             0.0,
                             gpu::ctx__->pSwapChainContext->extent__.width / 4,
                             gpu::ctx__->pSwapChainContext->extent__.height / 4
                            );
        drawTexture(cmd, renderTargetFilm_->gBufferPositionHandle_->descriptorArrayIndex__);

        gpu::cmdSetViewports(cmd,
                             gpu::ctx__->pSwapChainContext->extent__.width / 4,
                             0.0,
                             gpu::ctx__->pSwapChainContext->extent__.width / 4,
                             gpu::ctx__->pSwapChainContext->extent__.height / 4
                            );
        drawTexture(cmd, renderTargetFilm_->gBufferNormalHandle_->descriptorArrayIndex__);

        gpu::cmdSetViewports(cmd,
                             0.0,
                             gpu::ctx__->pSwapChainContext->extent__.height / 4,
                             gpu::ctx__->pSwapChainContext->extent__.width / 4,
                             gpu::ctx__->pSwapChainContext->extent__.height / 4
                            );
        drawTexture(cmd, this->renderTargetFilm_->gBufferAlbedoHandle_->descriptorArrayIndex__);
        gpu::cmdSetViewports(cmd,
                             gpu::ctx__->pSwapChainContext->extent__.width / 4,
                             gpu::ctx__->pSwapChainContext->extent__.height / 4,
                             gpu::ctx__->pSwapChainContext->extent__.width / 4,
                             gpu::ctx__->pSwapChainContext->extent__.height / 4
                            );
        drawTexture(cmd, this->renderTargetFilm_->gBufferRoughnessHandle_->descriptorArrayIndex__);
        gpu::cmdSetViewports(cmd,
                             gpu::ctx__->pSwapChainContext->extent__.width / 2,
                             0,
                             gpu::ctx__->pSwapChainContext->extent__.width / 2,
                             gpu::ctx__->pSwapChainContext->extent__.height / 2
                            );
        drawTexture(cmd, this->renderTargetFilm_->lightningAttachmentHandle_->descriptorArrayIndex__);
        gpu::cmdSetViewports(cmd,
                             0,
                             gpu::ctx__->pSwapChainContext->extent__.height / 2,
                             gpu::ctx__->pSwapChainContext->extent__.width / 2,
                             gpu::ctx__->pSwapChainContext->extent__.height / 2
                            );
        drawTexture(cmd, this->renderTargetFilm_->toneMappingAttachment_->descriptorArrayIndex__);
        gpu::cmdSetViewports(cmd,
                             gpu::ctx__->pSwapChainContext->extent__.width / 2,
                             gpu::ctx__->pSwapChainContext->extent__.height / 2,
                             gpu::ctx__->pSwapChainContext->extent__.width / 2,
                             gpu::ctx__->pSwapChainContext->extent__.height / 2
                            );
        drawTexture(cmd, this->renderTargetFilm_->gammaAttachment_->descriptorArrayIndex__);
      }
      else
      {
        int offRender = this->renderTargetFilm_->gammaAttachment_->descriptorArrayIndex__;
        gpu::cmdPushConstant(
                             cmd,
                             pipeline_->pipelineLayout_h,
                             VK_SHADER_STAGE_VERTEX_BIT |
                             VK_SHADER_STAGE_FRAGMENT_BIT,
                             0,
                             4,
                             &offRender
                            );
        gpu::cmdSetViewports(cmd,
                             0.0,
                             0.0,
                             gpu::ctx__->pSwapChainContext->extent__.width,
                             gpu::ctx__->pSwapChainContext->extent__.height
                            );
        gpu::cmdDrawQuad(cmd);
        gpu::cmdEndRendering(cmd);
      }
    };
    pass->write__.push_back(renderTargetFilm_->swapchainAttachment_.get());
    pass->read__.push_back(renderTargetFilm_->gBufferAlbedoHandle_.get());
    pass->read__.push_back(renderTargetFilm_->gBufferPositionHandle_.get());
    pass->read__.push_back(renderTargetFilm_->gBufferNormalHandle_.get());
    pass->read__.push_back(renderTargetFilm_->gBufferRoughnessHandle_.get());
    pass->read__.push_back(renderTargetFilm_->lightningAttachmentHandle_.get());
    pass->read__.push_back(renderTargetFilm_->depthAttachmentHandle_.get());
    pass->read__.push_back(renderTargetFilm_->bloomingBlurAttachment_.get());
    pass->read__.push_back(renderTargetFilm_->bloomingExtractAttachment_.get());
    pass->read__.push_back(renderTargetFilm_->toneMappingAttachment_.get());
    pass->read__.push_back(renderTargetFilm_->gammaAttachment_.get());
    renderTargetFilm_->swapchainAttachment_.get()->lastWriter__ = nullptr;
    renderTargetFilm_->swapchainAttachment_.get()->writen__ = false;
    uploadPasses_.push_back(pass);
  }


  void FrameGraph::drawTexture(gpu::CommandBuffer cmd, uint32_t descriptorArrayIndex)
  {
    gpu::cmdPushConstant(
                         cmd,
                         pipeline_->pipelineLayout_h,
                         VK_SHADER_STAGE_VERTEX_BIT |
                         VK_SHADER_STAGE_FRAGMENT_BIT,
                         0,
                         4,
                         &descriptorArrayIndex
                        );
    gpu::cmdDrawQuad(cmd);
  }

  void FrameGraph::pushModelConstant(VkCommandBuffer command, ModelConstant* modelConstant)
  {
    gpu::cmdPushConstant(
                         command,
                         pipeline_->pipelineLayout_h,
                         VK_SHADER_STAGE_VERTEX_BIT |
                         VK_SHADER_STAGE_FRAGMENT_BIT,
                         0,
                         sizeof(ModelConstant),
                         modelConstant
                        );
  }

  void FrameGraph::pushFrameConstant(VkCommandBuffer command)
  {
    gpu::cmdPushConstant(
                         command,
                         pipeline_->pipelineLayout_h,
                         VK_SHADER_STAGE_VERTEX_BIT |
                         VK_SHADER_STAGE_FRAGMENT_BIT,
                         0,
                         sizeof(this->renderTargetFilm_->renderAttachment),
                         &renderTargetFilm_->renderAttachment
                        );
  }
}
