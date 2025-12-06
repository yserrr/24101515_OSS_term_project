#ifndef INTERACTION_HPP
#define INTERACTION_HPP
#include"renderer.hpp"
#include "sculptor/sculptor_act.hpp"
#include "event_manager_mode.hpp"
#include "imgui.h"

class SwapchainManager;

class EventManager{
  friend class App;

public:
  ImGuiKey glfwToImgui(int key);
  EventManager(GLFWwindow *window, Camera *mainCam, ResourceManager *resourceManager, VkExtent2D extent);
  void onKeyEvent(int key, int scancode, int action, int mods);
  void syncWithCam(Camera *cam);
  void setSwapchain(SwapchainManager *swapchainP);
  bool isResized();
  bool isMultiView();
  VkExtent2D getExtent();
  void getKey();
  void getMouseEvent();
  void wheelUpdate();
  void setRenderer(RenderingSystem *renderer);
  void selectActor();

private:
  static void keyCallbackWrapper(GLFWwindow *window, int key, int scancode, int action, int mods);
  static void mouseButtonCallbackWrapper(GLFWwindow *window, int button, int action, int mods);
  static void cursorPosCallbackWrapper(GLFWwindow *window, double xpos, double ypos);
  static void framebufferSizeCallback(GLFWwindow *window, int w, int h);
  static void scrollCallback(GLFWwindow *window, double xoffset, double yoffset);
  void onMouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
  void cursorPosCallback(GLFWwindow *window, double xpos, double ypos);
  void getViewIndex(double w, double h);

private:
  bool singleModel        = true;
  ActorMode currentActor_ = ActorMode::Sculptor;
  std::unique_ptr<Actor> actor_;
  SwapchainManager *swapchain_;
  ResourceManager *resourcesManager_;
  RenderingSystem *renderer_;
  GLFWwindow *window_;
  Camera *mainCam;
  bool altPressed   = false;
  bool leftButton   = false;
  bool middleButton = false;
  bool rightButton  = false;

  float yaw      = -90.0f;
  float pitch    = 0.0f;
  float distance = 5.0f;

  bool moved            = false;
  double wheelDelta_    = 0;
  bool muliiViews       = false;
  bool VR               = false;
  bool resized          = false;
  bool leftMousePressed = false;
  float sensitivity     = 0.05f;
  double lastX          = 0.0;
  double lastY          = 0.0;
  double lastActionTime;
  bool mouseMoveState         = true;
  const double actionCooldown = 0.2; // 200ms
  VkExtent2D currentExtent;
};
#endif //INTERACTION_HPP