//
// Created by ljh on 25. 9. 16..
//

#ifndef MYPROJECT_CAMERA_CFG_HPP
#define MYPROJECT_CAMERA_CFG_HPP

#include <glm/glm.hpp>
#include <string>

struct Ray{
  glm::vec3 origin;
  glm::vec3 direction;
};

enum class CameraMode : uint32_t{
  PERSPECTIVE  = 0,
  ORTHOGRAPHIC = 1,
};

enum class Eye : uint32_t{
  MONO  = 0,
  LEFT  = 1,
  RIGHT = 2
};

///@param
///@reverseZ:
///@infiniteFar:
///@ipdHalf : meter unit

struct ClipControl{
  bool infiniteFar = false;
  bool reverseZ    = true;
  bool yFlip       = true;
};

struct ViewExtent{
  uint32_t width;
  uint32_t height;
};

struct ViewConfig{
  std::string name;
  ViewExtent extent = {1280, 720};
  float fovDeg      = 45.0f;
  float aspect      = 16.0f / 9.0f;
  float nearPlane   = 0.1f;
  float farPlane    = 300.0f;
  CameraMode mode   = CameraMode::PERSPECTIVE;
  float orthoHeight = 10.0f;
  ClipControl clip{};
  glm::vec2 jitter = glm::vec2(0.0f);
  Eye eye          = Eye::MONO;
  float ipdHalf    = 0.032f;
};

namespace gpu{
  struct alignas(16) fps_cam{
    glm::mat4 view;
    glm::mat4 proj;
  };

  struct alignas(16) cam_processing{
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 invView;
    glm::mat4 invProj;
    glm::mat4 viewProj;
    glm::mat4 invViewProj;
    glm::vec4 camPos;
  };

  struct cam_desc{
    uint64_t stride;
    uint32_t camMax;
  };
}

struct ViewData{
  ViewConfig cfg;
  gpu::cam_processing camBuf{};
  gpu::cam_processing prevBuf{};
  bool dirty = false;
};

static_assert(sizeof(gpu::cam_processing) % 16 == 0, "CameraUBO must be 16-byte aligned");

#endif //MYPROJECT_CAMERA_CFG_HPP