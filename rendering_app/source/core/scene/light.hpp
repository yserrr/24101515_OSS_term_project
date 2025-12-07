#ifndef LIGHT_HPP
#define LIGHT_HPP
#include <vector>
#include "gpu_context.hpp"
#include "glm/glm.hpp"
#include "transform.h"

#define MAX_LIGHTS 4

enum class LightType: uint32_t
{
  Directional = 0,
  Point       = 1,
  Spot        = 2
};

struct Light
{
  Transform transform;
  LightType type;
  glm::vec3 color;
  float angle;
  float intensity;
};

struct GPULight
{
  glm::vec4 position;
  glm::vec4 direction;
  glm::vec4 color;
  glm::vec4 padding;
  glm::mat4 view;
  glm::mat4 proj;
};

struct lightUBO
{
  GPULight lights[MAX_LIGHTS];
  int lightCount;
  int padding1;
  int padding2;
  int padding3;
};

class LightBuilder
{
  public:
  LightBuilder();
  void build(const Light& light);
  void uploadData();
  void drawUI();
  gpu::GPUBuffer buffer;
  std::vector<Light> lights;
  bool uiState = false;
  lightUBO ubo;
};

#endif
