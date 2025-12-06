#include<camera.hpp>

static float toRadians(float deg)
{
  return glm::radians(deg);
}

Camera::Camera(CamCI info)
  : pos_(0.0f, 0.0f, 0.0f),
    dir_(0.0f, 0.0f, -1.0f),
    up_(0.0f, 1.0f, 0.0f),
    fov_(info.fov),
    aspect(info.aspectRatio),
    nearPlane(info.nearPlane),
    farPlane(info.farPlane)

{
  right_ = glm::normalize(glm::cross(dir_, up_));
  right_ = glm::normalize(right_);
  camUpdate();
  spdlog::info("set camera");
}

void Camera::camUpdate()
{
  ubo.view   = getViewMatrix();
  ubo.proj   = getProjectionMatrix();
  ubo.camPos = pos_;
}

void Camera::setSpeed(float ds)
{
  delta += ds;
}

glm::mat4 Camera::getViewMatrix() const
{
  return glm::lookAt(pos_, pos_ + dir_, up_);
}

glm::mat4 Camera::getProjectionMatrix() const
{
  glm::mat4 pers = glm::perspective(glm::radians(fov_), aspect, nearPlane, farPlane);
  pers[1][1] *= -1.0f;
  return pers;
}

void Camera::onVR(bool left)
{
  float top         = nearPlane * tan(VRFov / 2);
  glm::vec3 leftEye = pos_;
  float bottom      = -top;
  float right       = top * aspect;
  float leftPoint   = -right;
  glm::mat4 frustom = glm::frustum(leftPoint, right, bottom, top, nearPlane, farPlane);
  ubo.proj          = frustom;

  glm::vec3 eyePos = pos_;
  if (left) eyePos -= ipdHalf * right_;
  if (!left)eyePos += ipdHalf * right_;
  glm::mat4 View = glm::lookAt(eyePos, eyePos + dir_, up_);
  ubo.view = View;
}

Ray Camera::generateRay(double posX, double posY)
{
  double xNdc = (2 * posX / currentExtent.width) - 1;
  double yNdc = 1 - (2 * posY / currentExtent.height);

  float tanFov     = tan(fov_ * 0.5f);
  float xView      = xNdc * aspect * tanFov;
  float yView      = yNdc * tanFov;
  glm::vec3 rayDir = glm::vec3(-xView, yView, -1.0f);
  rayDir           = glm::normalize(rayDir);
  Ray ray{pos_, rayDir};
  return ray;
}

void Camera::moveForward()
{
  pos_ += delta * dir_;
  camUpdate();
}

void Camera::moveBackward()
{
  pos_ -= delta * dir_;
  camUpdate();
}

void Camera::moveRight()
{
  pos_ += delta * right_;
}

void Camera::moveLeft()
{
  pos_ -= delta * right_;
}

void Camera::lookAt(glm::vec3 center)
{
  dir_ = glm::normalize(center - pos_);
  camUpdate();
}

void Camera::addFov(float dt)
{
  fov_ -= dt;
  camUpdate();
}

void Camera::addQuat(float dYawDeg, float dPitDeg)
{
  glm::quat pitchRotation = glm::angleAxis(toRadians(dPitDeg), right_);
  glm::quat yawRotation   = glm::angleAxis(toRadians(dYawDeg), glm::vec3(up_));
  orientation_            = yawRotation * pitchRotation * orientation_;
  orientation_            = glm::normalize(orientation_);
  dir_                    = glm::normalize(orientation_ * glm::vec3(0, 0, -1));
  camUpdate();
}