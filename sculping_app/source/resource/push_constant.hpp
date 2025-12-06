//
// Created by ljh on 25. 9. 18..
//

#ifndef MYPROJECT_PUSH_CONSTANT_HPP
#define MYPROJECT_PUSH_CONSTANT_HPP
#include "glm/glm.hpp"

struct MaterialConstant{
  glm::mat4 modelMatrix   = glm::mat4(1.0f);
  glm::vec4 color         = glm::vec4(1.0f);
  int albedoTextureIndex  = 0;
  int normalTextureIndex  = 0;
  int metalicTextureIndex = 0;
  int padding;
  glm::vec3 emissiveColor = glm::vec3(0.0f);
  float fpadding;

  float metallic    = 0.0f;
  float roughness   = 0.04f;
  float ao          = 1.0f;
  float emission    = 0.0f;
  float normalScale = 1.0f;

  float alphaCutoff = 0.0f;
  float paddingf;
  float padding2;
  float paddin;
};
#endif //MYPROJECT_PUSH_CONSTANT_HPP