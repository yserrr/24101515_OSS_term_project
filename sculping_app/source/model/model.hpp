#ifndef MODEL_HPP
#define MODEL_HPP
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <mesh.hpp>
#include <material.hpp>
#include "mesh_sub.hpp"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
// static call ->batch 만들기
// sink구조 생각하면 될듯
struct Transform{
  glm::vec3 position = glm::vec3(0, 0, 0);
  glm::quat rotation{};
  glm::vec3 scale = glm::vec3(1, 1, 1);

  glm::mat4 getMatrix()
  {
    return glm::translate(glm::mat4(1.0), position) *
           glm::toMat4(rotation) *
           glm::scale(glm::mat4(1), scale);
  }
};

struct Model{
  Mesh *mesh = nullptr;
  Material *material = nullptr;
  Submesh submesh{};
  Transform transform;
  MaterialConstant constant{};
  std::string name;
};

#endif