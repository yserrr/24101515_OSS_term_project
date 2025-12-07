#include "../GPU/gpu_context.hpp"
#include "io.hpp"
#include "imgui_impl_glfw.h"

extern gpu::Context* ctx__;

namespace mns
{
  IoSystem io__;
  WSAD WSAD__;
  MouseButton mousebutton__;
  CursorState cursor__;
  IoSystem::IoSystem() = default;

  void IoSystem::init()
  {
    glfwSetWindowUserPointer(gpu::ctx__->windowh__, this);
    glfwSetKeyCallback(gpu::ctx__->windowh__, keyCallbackWrapper);
    glfwSetCursorPosCallback(gpu::ctx__->windowh__, cursorPosCallbackWrapper);
    glfwSetMouseButtonCallback(gpu::ctx__->windowh__, mouseButtonCallbackWrapper);
    glfwSetFramebufferSizeCallback(gpu::ctx__->windowh__, framebufferSizeCallback);
    glfwMakeContextCurrent(gpu::ctx__->windowh__);
    glfwSetScrollCallback(gpu::ctx__->windowh__, scrollCallback);
  }

  void IoSystem::keyCallbackWrapper(GLFWwindow* window, int key, int scancode, int action, int mods)
  {
    IoSystem* self = static_cast<IoSystem*>(glfwGetWindowUserPointer(window));
    if (self)
    {
      self->onKeyCallback(key, scancode, action, mods);
    }
  }

  void IoSystem::mouseButtonCallbackWrapper(GLFWwindow* window, int button, int action, int mods)
  {
    IoSystem* self = static_cast<IoSystem*>(glfwGetWindowUserPointer(window));
    ImGuiIO& io = ImGui::GetIO();
    if (action == GLFW_PRESS)
    {
      switch (button)
      {
        case (GLFW_MOUSE_BUTTON_LEFT):
        {
          self->mouseState__.leftButtonDown = true;
          break;
        }
        case (GLFW_MOUSE_BUTTON_RIGHT):
        {
          self->mouseState__.rightButtonDown = true;
          break;
        }
        case (GLFW_MOUSE_BUTTON_MIDDLE):
        {
          self->mouseState__.middleButtonDown = true;
          break;
        }
      }
    }
    if (action == GLFW_RELEASE)
    {
      switch (button)
      {
        case (GLFW_MOUSE_BUTTON_LEFT):
        {
          self->mouseState__.leftButtonDown = false;
          break;
        }
        case (GLFW_MOUSE_BUTTON_RIGHT):
        {
          self->mouseState__.rightButtonDown = false;
          break;
        }
        case (GLFW_MOUSE_BUTTON_MIDDLE):
        {
          self->mouseState__.middleButtonDown = false;
          break;
        }
      }
    }
    io.AddMouseButtonEvent(button, action);
  }

  void IoSystem::cursorPosCallbackWrapper(GLFWwindow* window, double xpos, double ypos)
  {
    IoSystem* self = static_cast<IoSystem*>(glfwGetWindowUserPointer(window));
    if (self)
    {
      self->cursorPosCallback(window, xpos, ypos);
    }
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos.x = static_cast<float>(xpos);
    io.MousePos.y = static_cast<float>(ypos);
    io.AddMousePosEvent(xpos, ypos);
    if (io.WantCaptureMouse)
    {
      io.MousePos.x = xpos;
      io.MousePos.y = ypos;
    }
  }

  void IoSystem::framebufferSizeCallback(GLFWwindow* window, int w, int h)
  {
    IoSystem* self = static_cast<IoSystem*>(glfwGetWindowUserPointer(window));
    self->windowState__.resized = true;
    self->windowState__.height = h;
    self->windowState__.width = w;
    self->dirty_ = true;
  }

  void IoSystem::scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
  {
    IoSystem* self = static_cast<IoSystem*>(glfwGetWindowUserPointer(window));
    self->mouseState__.whellDeltaX = xoffset - self->mouseState__.scrollXOffset;
    self->mouseState__.whellDeltaY = yoffset - self->mouseState__.scrollYOffset;

    self->mouseState__.scrollXOffset = xoffset;
    self->mouseState__.scrollYOffset = yoffset;

    self->dirty_ = true;
    ImGuiIO& io = ImGui::GetIO();
    io.AddMouseWheelEvent(xoffset, yoffset);
  }


  void IoSystem::cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
  {
    IoSystem* self = static_cast<IoSystem*>(glfwGetWindowUserPointer(window));
    self->mouseState__.xpos = static_cast<float>(xpos);
    self->mouseState__.ypos = static_cast<float>(ypos);
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos.x = static_cast<float>(xpos);
    io.MousePos.y = static_cast<float>(ypos);
    dirty_ = true;
  }

  ImGuiKey IoSystem::glfwToImgui(int key)
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

  void IoSystem::onKeyCallback(int key, int scancode, int action, int mods)
  {
    dirty_ = true;
    ImGuiIO& io = ImGui::GetIO();
    ImGuiKey imguiKey = glfwToImgui(key);
    if (imguiKey != ImGuiKey_None)
    {
      bool isPressed = (action == GLFW_PRESS || action == GLFW_REPEAT);
      io.AddKeyEvent(imguiKey, isPressed);
    }
    io.SetKeyEventNativeData(imguiKey, key, scancode, mods);
    if (action == GLFW_PRESS)
    {
      switch (key)
      {
        case (GLFW_KEY_SPACE):
        {
          keyState__.keySpace = true;
          break;
        }
        case (GLFW_KEY_1):
        {
          keyState__.key1 = true;
          break;
        }
        case (GLFW_KEY_2):
        {
          keyState__.key2 = true;
          break;
        }
        case (GLFW_KEY_3):
        {
          keyState__.key3 = true;
          break;
        }
        case (GLFW_KEY_4):
        {
          keyState__.key4 = true;
          break;
        }
        case (GLFW_KEY_5):
        {
          keyState__.key5 = true;
          break;
        }
        case (GLFW_KEY_6):
        {
          keyState__.key6 = true;
          break;
        }
        case (GLFW_KEY_7):
        {
          keyState__.key7 = true;
          break;
        }
        case (GLFW_KEY_8):
        {
          keyState__.key8 = true;
          break;
        }

        case (GLFW_KEY_9):
        {
          keyState__.key9 = true;
          break;
        }
        case (GLFW_KEY_A):
        {
          keyState__.keyA = true;
          break;
        }
        case (GLFW_KEY_B):
        {
          keyState__.keyB = true;
          break;
        }
        case (GLFW_KEY_C):
        {
          keyState__.keyC = true;
          break;
        }
        case (GLFW_KEY_D):
        {
          keyState__.keyD = true;
          break;
        }
        case (GLFW_KEY_E):
        {
          keyState__.keyE = true;
          break;
        }
        case (GLFW_KEY_F):
        {
          keyState__.keyF = true;
          break;
        }
        case (GLFW_KEY_G):
        {
          keyState__.keyG = true;
          break;
        }
        case (GLFW_KEY_H):
        {
          keyState__.keyH = true;
          break;
        }
        case (GLFW_KEY_I):
        {
          keyState__.keyI = true;
          break;
        }
        case (GLFW_KEY_J):
        {
          keyState__.keyJ = true;
          break;
        }
        case (GLFW_KEY_K):
        {
          keyState__.keyK = true;
          break;
        }
        case (GLFW_KEY_L):
        {
          keyState__.keyL = true;
          break;
        }
        case (GLFW_KEY_M):
        {
          keyState__.keyM = true;
          break;
        }
        case (GLFW_KEY_N):
        {
          keyState__.keyN = true;
          break;
        }
        case (GLFW_KEY_O):
        {
          keyState__.keyO = true;
          break;
        }
        case (GLFW_KEY_P):
        {
          keyState__.keyP = true;
          break;
        }
        case (GLFW_KEY_Q):
        {
          keyState__.keyQ = true;
          break;
        }
        case (GLFW_KEY_R):
        {
          keyState__.keyR = true;
          break;
        }
        case (GLFW_KEY_S):
        {
          keyState__.keyS = true;
          break;
        }
        case (GLFW_KEY_T):
        {
          keyState__.keyT = true;
          break;
        }
        case (GLFW_KEY_U):
        {
          keyState__.keyU = true;
          break;
        }
        case (GLFW_KEY_V):
        {
          keyState__.keyV = true;
          break;
        }
        case (GLFW_KEY_W):
        {
          keyState__.keyW = true;
          break;
        }
        case (GLFW_KEY_X):
        {
          keyState__.keyX = true;
          break;
        }
        case (GLFW_KEY_Y):
        {
          keyState__.keyY = true;
          break;
        }
        case (GLFW_KEY_Z):
        {
          keyState__.keyZ = true;
          break;
        }
      }
    }
    if (action == GLFW_RELEASE)
    {
      switch (key)
      {
        case (GLFW_KEY_SPACE):
        {
          keyState__.keySpace = false;
          break;
        }
        case (GLFW_KEY_1):
        {
          keyState__.key1 = false;
          break;
        }
        case (GLFW_KEY_2):
        {
          keyState__.key2 = false;
          break;
        }
        case (GLFW_KEY_3):
        {
          keyState__.key3 = false;
          break;
        }
        case (GLFW_KEY_4):
        {
          keyState__.key4 = false;
          break;
        }
        case (GLFW_KEY_5):
        {
          keyState__.key5 = false;
          break;
        }
        case (GLFW_KEY_6):
        {
          keyState__.key6 = false;
          break;
        }
        case (GLFW_KEY_7):
        {
          keyState__.key7 = false;
          break;
        }
        case (GLFW_KEY_8):
        {
          keyState__.key8 = false;
          break;
        }

        case (GLFW_KEY_9):
        {
          keyState__.key9 = false;
          break;
        }
        case (GLFW_KEY_A):
        {
          keyState__.keyA = false;
          break;
        }
        case (GLFW_KEY_B):
        {
          keyState__.keyB = false;
          break;
        }
        case (GLFW_KEY_C):
        {
          keyState__.keyC = false;
          break;
        }
        case (GLFW_KEY_D):
        {
          keyState__.keyD = false;
          break;
        }
        case (GLFW_KEY_E):
        {
          keyState__.keyE = false;
          break;
        }
        case (GLFW_KEY_F):
        {
          keyState__.keyF = false;
          break;
        }
        case (GLFW_KEY_G):
        {
          keyState__.keyG = false;
          break;
        }
        case (GLFW_KEY_H):
        {
          keyState__.keyH = false;
          break;
        }
        case (GLFW_KEY_I):
        {
          keyState__.keyI = false;
          break;
        }
        case (GLFW_KEY_J):
        {
          keyState__.keyJ = false;
          break;
        }
        case (GLFW_KEY_K):
        {
          keyState__.keyK = false;
          break;
        }
        case (GLFW_KEY_L):
        {
          keyState__.keyL = false;
          break;
        }
        case (GLFW_KEY_M):
        {
          keyState__.keyM = false;
          break;
        }
        case (GLFW_KEY_N):
        {
          keyState__.keyN = false;
          break;
        }
        case (GLFW_KEY_O):
        {
          keyState__.keyO = false;
          break;
        }
        case (GLFW_KEY_P):
        {
          keyState__.keyP = false;
          break;
        }
        case (GLFW_KEY_Q):
        {
          keyState__.keyQ = false;
          break;
        }
        case (GLFW_KEY_R):
        {
          keyState__.keyR = false;
          break;
        }
        case (GLFW_KEY_S):
        {
          keyState__.keyS = false;
          break;
        }
        case (GLFW_KEY_T):
        {
          keyState__.keyT = false;
          break;
        }
        case (GLFW_KEY_U):
        {
          keyState__.keyU = false;
          break;
        }
        case (GLFW_KEY_V):
        {
          keyState__.keyV = false;
          break;
        }
        case (GLFW_KEY_W):
        {
          keyState__.keyW = false;
          break;
        }
        case (GLFW_KEY_X):
        {
          keyState__.keyX = false;
          break;
        }
        case (GLFW_KEY_Y):
        {
          keyState__.keyY = false;
          break;
        }
        case (GLFW_KEY_Z):
        {
          keyState__.keyZ = false;
          break;
        }
      }
    }
  }
}
