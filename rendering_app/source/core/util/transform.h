//
// Created by dlwog on 25. 10. 22..
//

#ifndef MYPROJECT_TRANSFORM_H
#define MYPROJECT_TRANSFORM_H
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/quaternion.hpp"

struct Transform
{
  glm::vec3 position = glm::vec3(0, 0, 0);
  glm::quat rotation{};
  glm::vec3 scale = glm::vec3(1, 1, 1);
  glm::mat4 matrix = glm::mat4(1.0f);
  bool dirty;

  inline void rotate(float pitch, float yaw, glm::vec3 up, glm::vec3 right)
  {
    glm::vec3 pivot = position;
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, position);
    model = glm::translate(model, pivot);
    glm::quat pitchRotation = glm::angleAxis(glm::radians(pitch), right);
    glm::quat yawRotation = glm::angleAxis(glm::radians(yaw), up);
    this->rotation = yawRotation * -pitchRotation * this->rotation;
    this->rotation = glm::normalize(this->rotation);
    model = model * glm::toMat4(this->rotation);
    model = glm::translate(model, -pivot);
    this->matrix = model;
  }


  inline void rotate(float pitch, float yaw)
  {
    glm::quat pitchRotation = glm::angleAxis(glm::radians(pitch), glm::vec3(1, 0, 0));
    glm::quat yawRotation = glm::angleAxis(glm::radians(yaw), glm::vec3(0, 1, 0));
    this->rotation = yawRotation * -pitchRotation * this->rotation;
    this->rotation = glm::normalize(this->rotation);
  }

  inline void move(glm::vec3 dir, float delta)
  {
    position += dir * delta;
  }

  inline void update()
  {
    this->matrix = glm::translate(glm::mat4(1.0), position) *
      glm::toMat4(rotation) *
      glm::scale(glm::mat4(1), scale);
  }
};
#endif //MYPROJECT_TRANSFORM_H
