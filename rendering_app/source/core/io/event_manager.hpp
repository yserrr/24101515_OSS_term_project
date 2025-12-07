#ifndef Enteraction__
#define Enteraction__
#include "../Render/RenderPassPool.hpp"

class EventManager
{
  friend class Engine;
  public:
  EventManager() = default;
  ~EventManager() = default;
  void init();
  void updateUITexture();
  void moveProcessEvent();
  void getKey();
  void getMouseEvent();
  void wheelUpdate();
  private:
  bool singleModel = true;
  //UserMode currentActor_ = UserMode::Sculptor;
  //std::unique_ptr<User> actor_;
  std::vector<bool> offscreenTarget;
  ResourceManager* pResourcesManager_;
  RenderPassBuilder* pRenderpassBuilder_;
  UI* ui;

  bool offscreenUpdateNeed = true;
  bool altPressed = false;
  bool leftButton = false;
  bool middleButton = false;
  bool rightButton = false;

  float yaw = -90.0f;
  float pitch = 0.0f;
  float distance = 5.0f;

  bool moved = false;
  double wheelDelta_ = 0;
  bool muliiViews = false;
  bool VR = false;
  bool resized = false;
  bool leftMousePressed = false;
  float sensitivity = 0.05f;
  double lastX = 0.0;
  double lastY = 0.0;
  double lastActionTime;
  bool mouseMoveState = true;
  const double actionCooldown = 0.2; // 200ms
  VkExtent2D currentExtent;
};
#endif //INTERACTION_HPP
