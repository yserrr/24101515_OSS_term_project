#include "camera_state.hpp"
#include "camera_func.hpp"
#include "io.hpp"

UserCamera::UserCamera()
  : camTransform_(glm::vec3(0.0f, 0.0f, 0.0f)),
    dir_(0.0f, 0.0f, -1.0f),
    up_(0.0f, 1.0f, 0.0f),
    delta(0.001),
    projection_{}
{
  projection_.farPlane = 1000.0f;
  switch (mode)
  {
    default:
    {
      moveProcess = fn_cam::wsadMove;
      updateViewMatrix = fn_cam::updateView;
      rotateProcess = fn_cam::rotateEditing;
      zoomProcess = fn_cam::noFn;
      resetState = fn_cam::resetState;
    }
  }
}

void UserCamera::updateUI()
{
  if (ImGui::Button("main camera")) this->uiState = !uiState;
  if (uiState)
  {
    ImGui::Begin("Camera:", &this->uiState);
    if (ImGui::Button("lock")) this->noUpdate = !noUpdate;
    ImGui::Text("camera state :");

    ImGui::SliderFloat("pos x: ", &this->camTransform_.position.x, -10.0f, 10.0f);
    ImGui::SliderFloat("pos y: ", &this->camTransform_.position.y, -10.0f, 10.0f);
    ImGui::SliderFloat("pos z: ", &this->camTransform_.position.z, -10.0f, 10.0f);
    ImGui::Separator();
    ImGui::SliderFloat("dir x: ", &this->dir_.x, 0.001, this->projection_.farPlane);
    ImGui::SliderFloat("dir y: ", &this->dir_.y, this->projection_.nearPlane, 1000);
    ImGui::SliderFloat("dir z: ", &this->dir_.z, 0.001, 10);

    ImGui::Separator();
    ImGui::Text("cam dir: %f, %f, %f :",
                this->dir_.x,
                this->dir_.y,
                this->dir_.z);
    ImGui::Separator();
    ImGui::SliderFloat("cam fov: ", &this->projection_.fov_, 0, 1);
    ImGui::SliderFloat("cam near: ", &this->projection_.nearPlane, 0.001, this->projection_.farPlane);
    ImGui::SliderFloat("cam far: ", &this->projection_.farPlane, this->projection_.nearPlane, 1000);
    ImGui::SliderFloat("cam aspect: ", &this->projection_.aspect, 0.001, 10);
    ImGui::End();
  }
}


void UserCamera::update()
{
  updateUI();
  moveProcess(this);
  rotateProcess(this);
  if (zoomProcess) zoomProcess(this);
  resetState(this);
  updateViewMatrix(this);
  this->cache.lastX = *mns::cursor__.xPos;
  this->cache.lastY = *mns::cursor__.yPos;
  this->cache.lastWheelOffsetX = *mns::cursor__.wheelXoffset;
  this->cache.lastWheelOffsetY = *mns::cursor__.wheelYoffset;
}
