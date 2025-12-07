#ifndef MODEL_HPP
#define MODEL_HPP
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include "gpu_context.hpp"
#include "mesh_sub.hpp"
#include "vk_host_buffer.h"

#include "util/transform.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <../core/scene_graph/mesh.hpp>
#include <../core/scene_graph/material.hpp>

struct ModelConstant
{
  glm::mat4 modelMatrix = glm::mat4(1.0f);
  glm::vec4 color = glm::vec4(1.0f);
  int albedoTextureIndex = -1;
  int normalTextureIndex = -1;
  int roughnessTextureIndex = -1;
  int metalicTextureIndex = -1;
};

struct Model
{
  gpu::MeshBuffer* mesh = nullptr;
  Material* material = nullptr;
  ModelConstant constant{};
  Transform transform;
  std::string name;
  float rotateX;
  float rotateY;
  float rotateZ;
  bool uiState = false;
  void drawUIState();
};

#endif
