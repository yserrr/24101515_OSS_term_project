#define MAX_LIGHTS 16
#ifndef LIGHT_HPP
#define LIGHT_HPP
#include <glm/glm.hpp>
#include <common.hpp>
#include <buffer_builder.hpp>

enum class LightType: uint32_t{
  Directional = 0,
  Point       = 1,
  Spot        = 2
};

struct Light{
  glm::vec3 position;
  LightType type;
  glm::vec3 direction;
  float angle;
  glm::vec3 color;
  float intensity;
};

struct GPULight{
  glm::vec4 position;
  glm::vec4 direction;
  glm::vec4 color;
  glm::vec4 coner;
};

struct lightUBO{
  GPULight lights[MAX_LIGHTS];
  alignas(16) int lightCount;
};

struct LightBuilder{
  LightBuilder()
  {
    ubo.lightCount = 0;
  }

  void build(const Light &light)
  {
    if (ubo.lightCount >= MAX_LIGHTS) return;
    GPULight &gpu = ubo.lights[ubo.lightCount++];
    gpu.position  = glm::vec4(light.position[0],
                             light.position[1],
                             light.position[2],
                             static_cast<float>(light.type));
    gpu.direction = glm::vec4(glm::normalize(light.direction), light.angle);
    gpu.color     = glm::vec4(light.color, light.intensity);
  }
  void uploadData()
  {
    bufferContext.buffer->loadData(&ubo, sizeof(ubo));
  }
  BufferContext bufferContext;
  lightUBO ubo;
};

#endif