#include "renderer.hpp"
#include "render_pipeline.hpp"
#include "context.hpp"

Renderer::Renderer() = default;

Renderer::~Renderer()
{
}

void Renderer::init()
{
  pipeline_.buildPipeline();
  renderPass_.buildPass();
  frameViews_.init();
  frames_.resize(gpu::ctx__->pSwapChainContext->img__.size());
  films_.resize(gpu::ctx__->pSwapChainContext->img__.size());
  for (auto i = 0; i < gpu::ctx__->pSwapChainContext->img__.size(); i++)
  {
    films_[i].buildFrameResources(i);
    frames_[i] = std::make_unique<dag::FrameGraph>(i, &films_[i]);
    frames_[i]->pipeline_ = &pipeline_;
    frames_[i]->passPool_ = &renderPass_;
    frames_[i]->init();
  }
  frameViews_.capture(frames_[0].get());
}

void Renderer::render()
{
  frameViews_.show();
  ImGui::Render();
  gpu::ctx__->pScheduler->run(frames_[gpu::ctx__->renderingContext.inflightIndex__
                                [gpu::ctx__->renderingContext.currentFrame__]]->renderPasses_);
}
