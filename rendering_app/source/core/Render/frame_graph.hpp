#ifndef MYPROJECT_FRAME_GRAPH_HPP
#define MYPROJECT_FRAME_GRAPH_HPP
#include  "RenderPassPool.hpp"
#include  "render_target_film.hpp"
#include "context.hpp"
struct RenderNode
{
  int id;
  const char* name;
  std::vector<int> input_attr_id;
  std::vector<int> output_attr_id;
  gpu::Pipeline drawHandle;
  ImVec2 pos = ImVec2(0, 0);
  ImVec2 screenPos = ImVec2 (-1, -1);
  ImVec2 size = ImVec2(200, 200);
  uint32_t bindings;
  bool frame;
  bool posSeted =false;
};
namespace dag
{
  class FrameGraph
  {
    public:
    FrameGraph(uint32_t frameIndex,
               flm::RenderTragetFilm* film);
    void init();
    ~FrameGraph() = default ;
    void addDepthOnlyPass();
    void addGBufferWritePass();
    void addShadowPass();
    void addLightningPass();
    void addBloomingExtractPass();
    void addBloomingBlurPass();
    void addTonemapPass();
    void addGammaCorrectionPass();
    void addOffscreenRenderPass();
    void addUiDraw();
    void offscreenRenderPass();
    std::vector<RenderNode> nodes;
    void updateFrameConstant();
    void drawTexture(gpu::CommandBuffer command, uint32_t descriptorArrayIndex);
    void pushModelConstant(gpu::CommandBuffer command, ModelConstant* modelConstant);
    void pushFrameConstant(gpu::CommandBuffer cmdBuffer);

    uint32_t frameIndex_;
    Pipeline* pipeline_;
    std::vector<gpu::RenderPass*> uploadPasses_;
    std::vector<gpu::RenderPass> renderPasses_;
    flm::RenderTragetFilm* renderTargetFilm_;
    RenderPassPool* passPool_;
  };
}
#endif //MYPROJECT_FRAME_GRAPH_HPP
