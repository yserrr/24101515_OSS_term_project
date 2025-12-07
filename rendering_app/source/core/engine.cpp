#define   STB_IMAGE_IMPLEMENTATION
#include <engine.hpp>
#include "util/unique.hpp"
#include "spdlog/spdlog.h"
#include "GPU/gpu_context.hpp"
#include "io/log_sink.hpp"
#include "io/io.hpp"
#include "../scene_graph/camera_state.hpp"
#include "Render/frame_graph_viewer.hpp"


Engine::Engine() = default;
Engine::~Engine() = default ;
namespace cpu
{
  std::unique_ptr<Context>ctx__ = std::make_unique<Context>();
}

void Engine::init()
{
  std::shared_ptr<sys::LogSink> UIsink = std::make_shared<sys::LogSink>();;
  // spdlog::set_default_logger(std::make_shared<spdlog::logger>("default", UIsink));
  // spdlog::set_level(spdlog::level::trace);
  gpu::ctx__->loadContext();
  mns::io__.init();
  ui.init();
  renderer.init();
  eventManager_.init();
  resourceManager.init();
  eventManager_.pResourcesManager_ = &resourceManager;
  eventManager_.ui = &ui;
  ui.setupStyle();
  ui.pResourceManager_ = &resourceManager;
  ui.sink_ = std::move(UIsink);

  //example model
  ///필요한 model 및 텍스쳐를 내부에서 코드로 올리면, 모델을 불러울 수 있습니다.
  ///std::string test = "C:/Users/dlwog/OneDrive/Desktop/VkMain-out/assets/textures/HAND_C.jpg";
  ///std::string test2 = "C:/Users/dlwog/OneDrive/Desktop/VkMain-out/assets/textures/HAND_N .jpg";
  ///std::string test3 = "C:/Users/dlwog/OneDrive/Desktop/VkMain-out/assets/textures/HAND_S.jpg";
  ///resourceManager.uploadTexture(test);
  ///resourceManager.uploadTexture(test2);
  ///resourceManager.uploadTexture(test3);
  ///resourceManager.uploadMesh("C:/Users/dlwog/OneDrive/Desktop/VkMain-out/assets/models/hand.fbx");
  ///auto& mesh = resourceManager.meshes_["C:/Users/dlwog/OneDrive/Desktop/VkMain-out/assets/models/hand.fbx"];
  ///resourceManager.addModel(mesh.get(), "test Model");
  ///auto& modle = resourceManager.models_["test Model"];
  ///modle->constant.albedoTextureIndex = resourceManager.textures_[test]->descriptorArrayIndex__;
  ///modle->constant.normalTextureIndex = resourceManager.textures_[test2]->descriptorArrayIndex__;
  ///modle->constant.roughnessTextureIndex = resourceManager.textures_[test3]->descriptorArrayIndex__;
  // resourceManager.uploadMesh("C:/Users/dlwog/OneDrive/Desktop/VkMain-out/assets/Iron_Man_Mark_44_Hulkbuster/Iron_Man_Mark_44_Hulkbuster_fbx.FBX");
  // resourceManager.uploadMesh("C:/Users/dlwog/OneDrive/Desktop/VkMain-out/assets/57-estancia_comedor_obj/room.obj");
}


void Engine::run()
{
  init();
  //uIRenderer->uploadImageToUI();
  while (!glfwWindowShouldClose(gpu::ctx__->windowh__))
  {
    glfwPollEvents();
    gpu::ctx__->pScheduler->nextFrame();
    ui.update();
    resourceManager.updateResource((gpu::ctx__->renderingContext.currentFrame__ + 1) %
                                   gpu::ctx__->renderingContext.maxInflight__);
    eventManager_.moveProcessEvent();
    renderer.render();
  }
}
