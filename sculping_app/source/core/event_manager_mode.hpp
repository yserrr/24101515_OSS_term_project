#ifndef MYPROJECT_EVENT_MANAGER_MODE_HPP
#define MYPROJECT_EVENT_MANAGER_MODE_HPP
#include "GLFW/glfw3.h"
#include  <common.hpp>
#include  <camera.hpp>
#include "sculptor/sculptor.hpp"

#include "FPS/camera_fps.hpp"

enum class ActorMode: uint32_t{
  Sculptor,
  Editor,
  FPS,
  Renderer,
};

struct Actor{
  Actor(GLFWwindow *window, Camera *camera) : window_(window), camera_(camera) {};
  virtual void act(VkCommandBuffer command) = 0;
  GLFWwindow *window_;
  std::unique_ptr<Sculptor> sculptor;
  float deltaX_     = 0.0;
  float deltaY_     = 0.0;
  float sensitivity = 0.01f;
  bool shoudAct     = false;
  Camera *camera_;
  virtual void keyEvent(int key, int scancode, int action, int mods) = 0;
  virtual void cursorPosCallBack(float deltaX, float deltaY) = 0;
  virtual void getKey() =0;
  virtual void getWheelUpdate(float deltaS) = 0;
  virtual void getMouseEvent() =0;
};

struct EditorMode : Actor{
  EditorMode(GLFWwindow *window, Camera *camera) : Actor(window, camera)
  {
    camera_ = dynamic_cast<EditorCam *>(camera_);
  };
  void act(VkCommandBuffer command) override {} ;
  virtual void keyEvent(int key, int scancode, int action, int mods) override {}
  virtual void getMouseEvent() override {}

  virtual void cursorPosCallBack(float deltaX, float deltaY) override
  {
    //camera_->addQuat(-deltaX * sensitivity, -deltaY * sensitivity);
    deltaX = 0;
    deltaY = 0;
  }

  virtual void getKey() override
  {
    if (glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS)
    {
      camera_->moveForward();
    }
    if (glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS)
    {
      camera_->moveLeft();
    }
    if (glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS)
    {
      camera_->moveRight();
    }
    if (glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS)
    {
      camera_->moveBackward();
    }
  }

  virtual void getWheelUpdate(float deltaS) override {};
};

struct FPS : Actor{
  FPS(GLFWwindow *window, Camera *camera) : Actor(window, camera)
  {
    camera_ = dynamic_cast<FPSCamera *>(camera_);
  };
  void act(VkCommandBuffer command) override {} ;
  virtual void keyEvent(int key, int scancode, int action, int mods) override {}
  virtual void getMouseEvent() override {}

  virtual void cursorPosCallBack(float deltaX, float deltaY) override
  {
    camera_->addQuat(-deltaX * sensitivity, -deltaY * sensitivity);
    deltaX = 0;
    deltaY = 0;
  }

  virtual void getKey() override
  {
    if (glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS)
    {
      camera_->moveForward();
    }
    if (glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS)
    {
      camera_->moveLeft();
    }
    if (glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS)
    {
      camera_->moveRight();
    }
    if (glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS)
    {
      camera_->moveBackward();
    }
  }

  virtual void getWheelUpdate(float deltaS) override {};
};

#endif //MYPROJECT_EVENT_MANAGER_MODE_HPP