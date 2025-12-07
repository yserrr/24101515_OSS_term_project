#ifndef IO_HPP
#define IO_HPP

#include "GLFW/glfw3.h"
#include "imgui.h"
namespace mns
{
  class IoSystem;
  extern IoSystem io__;
  struct WSAD;
  extern WSAD WSAD__;
  struct MouseButton;
  extern MouseButton mousebutton__;

  class IoSystem
  {
    public:
    IoSystem();
    void init();
    void onKeyCallback(int key, int scancode, int action, int mods);
    void getKey();
    void getMouseEvent();
    static void keyCallbackWrapper(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouseButtonCallbackWrapper(GLFWwindow* window, int button, int action, int mods);
    static void cursorPosCallbackWrapper(GLFWwindow* window, double xpos, double ypos);
    static void framebufferSizeCallback(GLFWwindow* window, int w, int h);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    void getViewIndex(double w, double h);
    ImGuiKey glfwToImgui(int key);
    bool dirty_ = false;

    struct
    {
      bool keySpace = false;
      bool keyCtrl = false;
      bool keyAlt = false;
      bool key0 = false;
      bool key1 = false;
      bool key2 = false;
      bool key3 = false;
      bool key4 = false;
      bool key5 = false;
      bool key6 = false;
      bool key7 = false;
      bool key8 = false;
      bool key9 = false;
      bool keyA = false;
      bool keyB = false;
      bool keyC = false;
      bool keyD = false;
      bool keyE = false;
      bool keyF = false;
      bool keyG = false;
      bool keyH = false;
      bool keyI = false;
      bool keyJ = false;
      bool keyK = false;
      bool keyL = false;
      bool keyM = false;
      bool keyN = false;
      bool keyO = false;
      bool keyP = false;
      bool keyQ = false;
      bool keyR = false;
      bool keyS = false;
      bool keyT = false;
      bool keyU = false;
      bool keyV = false;
      bool keyW = false;
      bool keyX = false;
      bool keyY = false;
      bool keyZ = false;
      bool keyEqual = false;
      bool keySemicolon = false;
    } keyState__;

    struct
    {
      double xpos = 0.0;
      double ypos = 0.0;
      double scrollXOffset = 0.0;
      double scrollYOffset = 0.0;

      double whellDeltaX = 0.0;
      double whellDeltaY = 0.0;

      bool leftButtonDown = false;
      bool rightButtonDown = false;
      bool middleButtonDown = false;
      double radius = 5;
    } mouseState__;

    struct
    {
      bool resized = false;
      int width = 0;
      int height = 0;
    } windowState__;
  };

  struct WSAD
  {
    const bool* W = &mns::io__.keyState__.keyW;
    const bool* S = &mns::io__.keyState__.keyS;
    const bool* D = &mns::io__.keyState__.keyD;
    const bool* A = &mns::io__.keyState__.keyA;
  };

  struct MouseButton
  {
    const bool* lButton = &mns::io__.mouseState__.leftButtonDown;
    const bool* mButton = &mns::io__.mouseState__.middleButtonDown;
    const bool* rButton = &mns::io__.mouseState__.rightButtonDown;
  };

  struct CursorState
  {
    const double* xPos = &mns::io__.mouseState__.xpos;
    const double* yPos = &mns::io__.mouseState__.ypos;
    const double* wheelXoffset = &mns::io__.mouseState__.scrollXOffset;
    const double* wheelYoffset = &mns::io__.mouseState__.scrollYOffset;
    const double* radius =  &mns::io__.mouseState__.radius;
  };
}
#endif //INTERACTION_HPP
