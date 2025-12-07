#ifndef RENDERER_HPP
#define RENDERER_HPP
#include "render_pipeline.hpp"
#include "../resource/resource_manager.hpp"
#include "../GPU/gpu_context.hpp"
#include "render_target_film.hpp"
#include "ui/ui.hpp"

class RenderPassPool
{
  friend class FrameGraph;
  friend class UI;
  friend class EventManager;
  friend class Engine;

  public:
  RenderPassPool();
  ~RenderPassPool() = default;
  void init();
  void draw(VkCommandBuffer cmd, uint32_t currentFrame);
  void buildPass();
  flm::RenderTragetFilm offscreenFilm_;
  std::vector<std::unique_ptr<gpu::RenderPass>> depthOnlyPass;
  std::vector<std::unique_ptr<gpu::RenderPass>> gBufferPass;
  std::vector<std::unique_ptr<gpu::RenderPass>> uiDrawPass;
  std::vector<std::unique_ptr<gpu::RenderPass>> shadowPass;
  std::vector<std::unique_ptr<gpu::RenderPass>> lightningPass;
  std::vector<std::unique_ptr<gpu::RenderPass>> bloomingExtractPass;
  std::vector<std::unique_ptr<gpu::RenderPass>> bloomingBlurPass;
  std::vector<std::unique_ptr<gpu::RenderPass>> tonemappingPass;
  std::vector<std::unique_ptr<gpu::RenderPass>> gammaPass;
  std::vector<std::unique_ptr<gpu::RenderPass>> swapchainRenderPass;
  std::vector<std::unique_ptr<gpu::RenderPass>> dlss;
};
#endif
