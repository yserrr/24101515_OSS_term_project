#include "../GPU/gpu_context.hpp"
#include "RenderPassPool.hpp"
#include "cast.hpp"

RenderPassPool::RenderPassPool() = default;

// void RenderPass::init()
// {
//   pipeline.buildPipeline();
//   offscreenFilm_.buildFrameResources(TODO);
//   buildPass();
// }

void RenderPassPool::buildPass()
{
  for (uint32_t i = 0; i < gpu::ctx__->renderingContext.maxInflight__; i++)
  {
    {
      std::unique_ptr<gpu::RenderPass> pass = std::make_unique<gpu::RenderPass>();
      pass->passType = gpu::RenderPassType::DEPTH_PASS;
      this->depthOnlyPass.push_back(std::move(pass));
    }
    {
      std::unique_ptr<gpu::RenderPass> pass = std::make_unique<gpu::RenderPass>();
      pass->passType = gpu::RenderPassType::G_BUFFER_PASS;
      pass->name = "g buffer write";
      this->gBufferPass.push_back(std::move(pass));
    }
    {
      std::unique_ptr<gpu::RenderPass> pass = std::make_unique<gpu::RenderPass>();
      pass->passType = gpu::RenderPassType::UI;
      pass->name = "ui render";
      this->uiDrawPass.push_back(std::move(pass));
    }

    {
      std::unique_ptr<gpu::RenderPass> pass = std::make_unique<gpu::RenderPass>();
      pass->passType = gpu::RenderPassType::SHADOW_PASS;
      pass->name = "shadowPass";
      this->shadowPass.push_back(std::move(pass));
    }

    {
      std::unique_ptr<gpu::RenderPass> pass = std::make_unique<gpu::RenderPass>();
      pass->passType = gpu::RenderPassType::LIGHTING_PASS;
      pass->name = "lightningPass";
      this->lightningPass.push_back(std::move(pass));
    }

    {
      std::unique_ptr<gpu::RenderPass> pass = std::make_unique<gpu::RenderPass>();
      pass->passType = gpu::RenderPassType::BLOOMING_PASS;
      pass->name = "blooming extract Pass";
      this->bloomingExtractPass.push_back(std::move(pass));
    }

    {
      std::unique_ptr<gpu::RenderPass> pass = std::make_unique<gpu::RenderPass>();
      pass->passType = gpu::RenderPassType::BLOOMING_PASS;
      pass->name = "blooming blur Pass";
      this->bloomingBlurPass.push_back(std::move(pass));
    }

    {
      std::unique_ptr<gpu::RenderPass> pass = std::make_unique<gpu::RenderPass>();
      pass->passType = gpu::RenderPassType::TONEMAP_PASS;
      pass->name = "tone Mapping Pass";
      this->tonemappingPass.push_back(std::move(pass));
    }
    {
      std::unique_ptr<gpu::RenderPass> pass = std::make_unique<gpu::RenderPass>();
      pass->passType = gpu::RenderPassType::GAMMA_PASS;
      pass->name = "gamma collection pass";
      this->gammaPass.push_back(std::move(pass));
    }
    {
      std::unique_ptr<gpu::RenderPass> pass = std::make_unique<gpu::RenderPass>();
      pass->passType = gpu::RenderPassType::SWAPCHIAN;
      pass->name = "ImageRenderPass";
      this->swapchainRenderPass.push_back(std::move(pass));
    }
  }
}



