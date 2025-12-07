#ifndef UIRenderer_HPP
#define UIRenderer_HPP
//#include "../sculptor/sculptor_act.hpp"
#include "../../core/io/log_sink.hpp"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include "../resource/resource_manager.hpp"

enum class CallType
{
  Mesh,
  Texture,
  UPLOAD,
  UPDATE,
  FREE,
};

struct UICall
{
  std::string path;
  uint32_t bindingSlot = -1;
  CallType type;
};

struct UITexture
{
  VkDescriptorSet descriptorSet;
  bindingIndex index;
};

class UI
{
  friend class Engine;
  friend class EventManager;

  public:
  UI();
  void init();
  void update();
  void rec();
  void render();
  void drawcall();
  void drawTransition();
  void drawStateWindow(ImVec2 size);
  void drawToolBoxRight(ImVec2 size);
  void drawToolBoxLeft(ImVec2 size);
  void drawToolBoxUnder(ImVec2 size);
  void drawToolBoxUnder3(ImVec2 size);
  void drawToolBoxUnderTexture(ImVec2 size);
  void offScreenTargetSpace(ImVec2 size);
  void drawVertexState(ImVec2 size);
  void drawIndexState(ImVec2 size);
  void drawTextureState(ImVec2 size);
  void drawCameraState(ImVec2 size);
  void drawMaterialState(ImVec2 size);
  void drawMouseState(ImVec2 size);
  void drawFramebufferState();
  void drawFrame(ImVec2 size);
  void inputText();

  void drawLightUI(ImVec2 size);
  void drawCameraUI();
  void drawMaterialUI();
  void drawShaderUI();
  private:
  struct
  {
    std::vector<bool> updated;
    std::vector<UITexture> albedoTargets;
    std::vector<UITexture> normalTargets;
    std::vector<UITexture> positionTargets;
    std::vector<UITexture> specularTargets;
    std::vector<UITexture> depthTargets;
  }offscreenTagets;

  RenderPassBuilder* pPassBuilder;
  std::vector<UITexture> textureHandle;
  bool rightBox = false;
  void colorPickerColor();
  void setupStyle();
  void uploadImageToUI();
  bool smallUi_ = true; // 작은 창 유지 토글
  float color[4];
  float pos[3];
  bindingIndex selectedTextureBinding = 0;
  std::vector<std::string> uploadedMesh_;
  std::vector<UITexture> uiTextures_;
  std::vector<UICall> callStack_;
  std::shared_ptr<sys::LogSink> sink_;
  VkDescriptorSet backgroundDescriptor_;
  ResourceManager* pResourceManager_;
};

#endif
