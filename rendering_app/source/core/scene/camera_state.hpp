#ifndef MYPROJECT_CAMERA_HPP
#define MYPROJECT_CAMERA_HPP
#include <string>
#include <optional>
#include <vector>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#include "util/transform.h"
#include "context.hpp"
#include "model.hpp"
enum class CameraUserMode
{
  Free,
  Follow,
  Orbit
};

struct ClipControl
{
  bool infiniteFar = false;
  bool reverseZ = true;
  bool yFlip = true;
};

enum class CameraMode : uint32_t
{
  PERSPECTIVE  = 0,
  ORTHOGRAPHIC = 1,
};

struct Ray
{
  glm::vec3 origin;
  glm::vec3 direction;
};

enum class Eye : uint32_t
{
  MONO  = 0,
  LEFT  = 1,
  RIGHT = 2
};

struct CameraUBO
{
  glm::mat4 view{};
  glm::mat4 proj{};

  glm::vec3 camPos{};
  float padding ;

  float fov_ = glm::radians(45.0f);
  float aspect = 20 / 12;
  float nearPlane = 0.01f;
  float farPlane = 1000.0f;

  glm::mat4 invView{};
  glm::mat4 invProj{};
};

struct ProjectionConfig
{
  float fov_ = glm::radians(45.0f);
  float aspect = 20 / 12;
  float nearPlane = 0.1f;
  float farPlane = 10000.0f;
};

struct MouseCache
{
  double lastX = 0;
  double lastY = 0;
  double lastWheelOffsetX = 0;
  double lastWheelOffsetY = 0;
};

class Viewport
{
  std::vector<Viewport> viewports_;
};


class UserCamera
{
  public:
  std::string name = "user main Camera";
  UserCamera();
  void updateUI();
  void update();
  void (*moveProcess)(UserCamera* cam);
  void (*rotateProcess)(UserCamera* cam);
  void (*zoomProcess)(UserCamera* cam);
  void (*camUtilityProcess)(UserCamera* cam);
  void (*updateViewMatrix)(UserCamera* cam);
  void (*resetState)(UserCamera* cam);
  void genRay();
  glm::vec3 dir_;
  glm::vec3 right_;
  glm::vec3 up_;
  Transform camTransform_;
  std::optional<Model*> selectModel;
  ProjectionConfig projection_{};
  CameraUserMode mode;
  MouseCache cache;
  float delta = 0.01;
  CameraUBO ubo;
  bool noUpdate = false;
  bool uiState = false;
};

class VRCamera
{
};

#endif //MYPROJECT_CAMERA_HPP
