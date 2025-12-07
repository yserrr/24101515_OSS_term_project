#ifndef MYPROJECT_RENDERER_HPP
#define MYPROJECT_RENDERER_HPP
#include "frame_graph.hpp"
#include "frame_graph_viewer.hpp"

class Renderer
{
  public:
  Renderer();
  ~Renderer() ;
  void init();
  void render();
  Pipeline pipeline_;
  RenderPassPool renderPass_;
  std::vector<flm::RenderTragetFilm> films_;
  std::vector<std::unique_ptr<dag::FrameGraph>> frames_;
  dag::FrameGraphViewer frameViews_;

};


#endif //MYPROJECT_RENDERER_HPP
