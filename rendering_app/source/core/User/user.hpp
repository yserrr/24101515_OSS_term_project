#ifndef MYPROJECT_EVENT_MANAGER_MODE_HPP
#define MYPROJECT_EVENT_MANAGER_MODE_HPP

//#include "../sculptor/sculptor.hpp"

enum class UserMode: uint32_t
{
  Sculptor,
  Editor,
  FPS,
  Renderer,
};

struct User
{
  User();;
  virtual void keyEvent(int key, int scancode, int action, int mods) = 0;
  virtual void cursorPosCallBack(float deltaX, float deltaY) = 0;
  virtual void getKey() =0;
  virtual void getWheelUpdate(float deltaS) = 0;
  virtual void getMouseEvent() =0;
};


#endif //MYPROJECT_EVENT_MANAGER_MODE_HPP
