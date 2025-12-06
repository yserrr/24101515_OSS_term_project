#include<event_manager.hpp>
#include "sculptor/sculptor_act.hpp"
#include "imgui_impl_glfw.h"
#include <imgui.h>


EventManager::EventManager(GLFWwindow *window, Camera *mainCam, ResourceManager *resourceManager, VkExtent2D extent) :
  window_(window),
  mainCam(mainCam),
  resourcesManager_(resourceManager),
  currentExtent(extent)

{
  syncWithCam(mainCam);
  glfwGetCursorPos(window_, &lastX, &lastY);
  glfwSetWindowUserPointer(window, this);
  glfwSetKeyCallback(window, keyCallbackWrapper);
  glfwSetCursorPosCallback(window, cursorPosCallbackWrapper);
  glfwSetMouseButtonCallback(window, mouseButtonCallbackWrapper);
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
  glfwSetScrollCallback(window, scrollCallback);

  lastActionTime = glfwGetTime();
  spdlog::info("interaction set up");
  selectActor();
}

void EventManager::onKeyEvent(int key, int scancode, int action, int mods)
{
  ImGuiIO &io       = ImGui::GetIO();
  ImGuiKey imguiKey = glfwToImgui(key);
  if (imguiKey != ImGuiKey_None)
  {
    bool isPressed = (action == GLFW_PRESS || action == GLFW_REPEAT);
    io.AddKeyEvent(imguiKey, isPressed);
  }
  io.SetKeyEventNativeData(imguiKey, key, scancode, mods);
  if (io.WantCaptureKeyboard)
  {
    return;
  }
  actor_->keyEvent(key, scancode, action, mods);
  if (action == GLFW_PRESS)
  {
    if (key == GLFW_KEY_1)
    {
      renderer_->polygonMode = VK_POLYGON_MODE_FILL;
    }
    if (key == GLFW_KEY_2)
    {
      renderer_->polygonMode = VK_POLYGON_MODE_LINE;
    }
    if (key == GLFW_KEY_3)
    {
      renderer_->depthTest = !renderer_->depthTest;
    }
    if (key == GLFW_KEY_V)
    {
      VR = !VR;
      if (VR)
      {
        renderer_->viewMode =ViewMode::VR;
      }
      else
      {
        renderer_->viewMode =ViewMode::SINGLE;
      }
    }
    if (key == GLFW_KEY_Q)
    {
      glfwSetWindowShouldClose(window_, GLFW_TRUE);
    }
    if (key == GLFW_KEY_SPACE)
    {
      muliiViews = !muliiViews;
      if (!muliiViews)
      {
        renderer_->viewMode = ViewMode::SINGLE;
        double mouseX, mouseY;
        glfwGetCursorPos(window_, &mouseX, &mouseY);
        getViewIndex(mouseX, mouseY);
      } else
      {
        renderer_->viewMode = ViewMode::MULTI;
      }
    }

    if (key == GLFW_KEY_T)
    {
      mouseMoveState = !mouseMoveState;
      if (mouseMoveState)
      {
        io.MousePos.x = lastX;
        io.MousePos.y = lastY;
      }
    }
  }
}

void EventManager::syncWithCam(Camera *cam)
{
  glm::quat quat{};
  mainCam->pos_         = glm::vec3(0.0f, 0.0f, 30.0f);
  mainCam->dir_         = glm::vec3(0.0f, 0.0f, -1.0f);
  mainCam->orientation_ = quat;
  mainCam->camUpdate();
  lastX = 0;
  lastY = 0;

  glfwGetCursorPos(window_, &lastX, &lastY);
  mainCam->addQuat(lastX, lastY);
  mainCam->currentExtent = currentExtent;
}

void EventManager::setSwapchain(SwapchainManager *swapchainP)
{
  ///currentExtent.height = swapchainP->getExtent().height;
  ///currentExtent.width  = swapchainP->getExtent().width;
  ///swapchain_= swapchainP;
}

bool EventManager::isResized()
{
  if (resized)
  {
    resized = false;
    return true;
  }
  return false;
}

bool EventManager::isMultiView()
{
  return muliiViews;
}

VkExtent2D EventManager::getExtent()
{
  return currentExtent;
}

void EventManager::getKey()
{
  ImGuiIO &io = ImGui::GetIO();
  if (io.WantCaptureKeyboard)
  {
    return;
  }
  actor_->getKey();
}

void EventManager::getMouseEvent()
{
  ImGuiIO &io = ImGui::GetIO();
  if (io.WantCaptureMouse)
  {
    return;
  }
  actor_->getMouseEvent();
}

void EventManager::wheelUpdate()
{
  ImGuiIO &io = ImGui::GetIO();
  if (io.WantCaptureMouse)
  {
    return;
  }

  actor_->getWheelUpdate(wheelDelta_);
  wheelDelta_ = 0.0f;
}

void EventManager::setRenderer(RenderingSystem *renderer)
{
  renderer_ = renderer;
}

void EventManager::selectActor()
{
  switch (currentActor_)
  {
    case(ActorMode::Sculptor):
    {
      actor_ = std::make_unique<SculptorMode>(window_,
                                              mainCam,
                                              &resourcesManager_->selectedModel,
                                              currentExtent);
      break;
    }
    case(ActorMode::FPS):
    {
      actor_ = std::make_unique<FPS>(window_,mainCam);
      break ;
    }
    default:
    {
    case(ActorMode::Editor):
      actor_ = std::make_unique<EditorMode>(window_,mainCam);
      break;
    }
  }
}

void EventManager::keyCallbackWrapper(GLFWwindow *window, int key, int scancode, int action, int mods)
{
  EventManager *self = static_cast<EventManager *>(glfwGetWindowUserPointer(window));
  if (self)
  {
    self->onKeyEvent(key, scancode, action, mods);
  }
}

void EventManager::mouseButtonCallbackWrapper(GLFWwindow *window, int button, int action, int mods)
{
  EventManager *self = static_cast<EventManager *>(glfwGetWindowUserPointer(window));
  if (self)
  {
    self->onMouseButtonCallback(window, button, action, mods);
  }
}

void EventManager::cursorPosCallbackWrapper(GLFWwindow *window, double xpos, double ypos)
{
  ImGuiIO &io   = ImGui::GetIO();
  io.MousePos.x = static_cast<float>(xpos);
  io.MousePos.y = static_cast<float>(ypos);

  io.AddMousePosEvent(xpos, ypos);
  if (io.WantCaptureMouse)
  {
    io.MousePos.x = xpos;
    io.MousePos.y = ypos;
    return;
  }

  EventManager *self = static_cast<EventManager *>(glfwGetWindowUserPointer(window));
  if (self)
  {
    self->cursorPosCallback(window, xpos, ypos);
  }
}

void EventManager::framebufferSizeCallback(GLFWwindow *window, int w, int h)
{
  EventManager *self = static_cast<EventManager *>(glfwGetWindowUserPointer(window));
  if (self)
  {
    self->resized              = true;
    self->currentExtent.height = h;
    self->currentExtent.width  = w;
  }
}

void EventManager::scrollCallback(GLFWwindow *window, double xoffset, double yoffset)
{
  ImGuiIO &io = ImGui::GetIO();
  io.AddMouseWheelEvent(xoffset, yoffset);
  if (io.WantCaptureMouse)
  {
    return;
  }
  EventManager *self = static_cast<EventManager *>(glfwGetWindowUserPointer(window));
  self->wheelDelta_ += yoffset;
}

void EventManager::onMouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
  ImGuiIO &io = ImGui::GetIO();
  io.AddMouseButtonEvent(button, action);
  if (io.WantCaptureMouse)
  {
    return;
  }
}

void EventManager::cursorPosCallback(GLFWwindow *window, double xpos, double ypos)
{
  float deltaX = static_cast<float>(xpos - lastX);
  float deltaY = static_cast<float>(ypos - lastY);
  lastX        = xpos;
  lastY        = ypos;
  if (!mouseMoveState) return;
  ImGuiIO &io   = ImGui::GetIO();
  io.MousePos.x = static_cast<float>(xpos);
  io.MousePos.y = static_cast<float>(ypos);
  if (io.WantCaptureMouse | moved)
  {
    return;
  }
  EventManager *self = static_cast<EventManager *>(glfwGetWindowUserPointer(window));
  actor_->cursorPosCallBack(deltaX, deltaY);
  moved = true;
}

void EventManager::getViewIndex(double w, double h)
{
  ImGuiIO &io = ImGui::GetIO();
  if (io.WantCaptureMouse)
  {
    return;
  }
  bool right = w >= (currentExtent.width / 2.0f);
  bool top   = h >= (currentExtent.height / 2.0f);
  int index  = 0;
  if (!right && !top) index = 0;
  else if (right && !top) index = 1;
  else if (!right && top) index = 2;
  else if (right && top) index = 3;
  //mainCam->setMainCam(index);
}



ImGuiKey EventManager::glfwToImgui(int key)
{
  switch (key)
  {
    case GLFW_KEY_TAB: return ImGuiKey_Tab;
    case GLFW_KEY_LEFT: return ImGuiKey_LeftArrow;
    case GLFW_KEY_RIGHT: return ImGuiKey_RightArrow;
    case GLFW_KEY_UP: return ImGuiKey_UpArrow;
    case GLFW_KEY_DOWN: return ImGuiKey_DownArrow;
    case GLFW_KEY_PAGE_UP: return ImGuiKey_PageUp;
    case GLFW_KEY_PAGE_DOWN: return ImGuiKey_PageDown;
    case GLFW_KEY_HOME: return ImGuiKey_Home;
    case GLFW_KEY_END: return ImGuiKey_End;
    case GLFW_KEY_INSERT: return ImGuiKey_Insert;
    case GLFW_KEY_DELETE: return ImGuiKey_Delete;
    case GLFW_KEY_BACKSPACE: return ImGuiKey_Backspace;
    case GLFW_KEY_SPACE: return ImGuiKey_Space;
    case GLFW_KEY_ENTER: return ImGuiKey_Enter;
    case GLFW_KEY_ESCAPE: return ImGuiKey_Escape;
    case GLFW_KEY_A: return ImGuiKey_A;
    case GLFW_KEY_C: return ImGuiKey_C;
    case GLFW_KEY_V: return ImGuiKey_V;
    case GLFW_KEY_X: return ImGuiKey_X;
    case GLFW_KEY_Y: return ImGuiKey_Y;
    case GLFW_KEY_Z: return ImGuiKey_Z;
    default: return ImGuiKey_None;
  }
}