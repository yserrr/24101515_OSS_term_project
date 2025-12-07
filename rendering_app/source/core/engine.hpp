#ifndef ENGINE_HPP
#define ENGINE_HPP
#include <resource/importer.hpp>
#include <scene_graph/mesh.hpp>
#include <io/event_manager.hpp>
#include <resource/resource_manager.hpp>
#include "GPU/gpu_context.hpp"
#include "Render/renderer.hpp"

class Engine
{
  public:
  Engine();
  ~Engine();
  void init();
  void run();
  void drawUI();

  private:
  UI ui;
  EventManager eventManager_;
  ResourceManager resourceManager;
  Renderer renderer;
};

#endif //engine_hpp
