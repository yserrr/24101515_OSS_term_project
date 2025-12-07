#include "spdlog/spdlog.h"
#include"camera_state.hpp"
#include "io.hpp"
#pragma once


namespace mns
{
  struct WSAD;
  struct CursorState;
  extern mns::WSAD WSAD__;
  extern mns::CursorState cursor__;
}

namespace fn_cam
{
  void wsadMove(UserCamera* cam);
  void updateView(UserCamera* cam);
  void rotateFpsMode(UserCamera* cam);
  void rotateModel(UserCamera* cam);
  void rotateFpsMode(UserCamera* cam);
  void resetState(UserCamera* cam);
  void updateWheel(UserCamera* cam);
  void rotateEditing(UserCamera* cam);
  void noFn(UserCamera* cam);
}
