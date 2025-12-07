#ifndef VERTEX_HPP
#define VERTEX_HPP
#include "glm/glm.hpp"


enum class VertexType:uint32_t{
  QUAD,
  PC,
  PUVN,
  PUVNTC,
  ALL,
  BACKGROUND
};

struct Vertex{};

struct Quad: Vertex {
  glm::vec3 point1;
  glm::vec3 point2;

};

struct VertexPC: Vertex {
  glm::vec3 position;
  glm::vec3 color;
};

struct VertexPN{
  glm::vec3 position;
  glm::vec3 normal;
};

struct VertexPUVN{
  glm::vec3 position;
  glm::vec2 uv;
  glm::vec3 normal;
};

struct VertexPUVNTC{
  glm::vec3 position;
  glm::vec2 uv;
  glm::vec3 normal;
  glm::vec3 tangent;
  glm::vec3 bitangent;
  glm::vec3 color;
};

struct VertexAll{
  glm::vec3 position;     // location = 0
  glm::vec3 normal;       // location = 1
  glm::vec2 uv;           // location = 2
  glm::vec3 tangent;      // location = 3
  glm::vec3 bitangent;    // location = 4
  glm::vec4 color;        // location = 5
  glm::ivec4 boneIndices; // location = 6
  glm::vec4 boneWeights;  // location = 7
};
  //vertexBindingDesc.binding   = 0;
  //vertexBindingDesc.stride    = sizeof(VertexAll);
  //vertexBindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
//
  //vertexAttributeDescriptions.resize(8);
  //vertexAttributeDescriptions[0].binding  = vertexBinding;
  //vertexAttributeDescriptions[0].location = 0;
  //vertexAttributeDescriptions[0].format   = VK_FORMAT_R32G32B32_SFLOAT; // position (vec3)
  //vertexAttributeDescriptions[0].offset   = offsetof(VertexAll, position);
//
  //vertexAttributeDescriptions[1].binding  = vertexBinding;
  //vertexAttributeDescriptions[1].location = 1;
  //vertexAttributeDescriptions[1].format   = VK_FORMAT_R32G32B32_SFLOAT; // normal (vec3)
  //vertexAttributeDescriptions[1].offset   = offsetof(VertexAll, normal);
//
  //vertexAttributeDescriptions[2].binding  = vertexBinding;
  //vertexAttributeDescriptions[2].location = 2;
  //vertexAttributeDescriptions[2].format   = VK_FORMAT_R32G32_SFLOAT; // uv (vec2)
  //vertexAttributeDescriptions[2].offset   = offsetof(VertexAll, uv);
//
  //vertexAttributeDescriptions[3].binding  = vertexBinding;
  //vertexAttributeDescriptions[3].location = 3;
  //vertexAttributeDescriptions[3].format   = VK_FORMAT_R32G32B32_SFLOAT; // tangent (vec3)
  //vertexAttributeDescriptions[3].offset   = offsetof(VertexAll, tangent);
//
  //vertexAttributeDescriptions[4].binding  = vertexBinding;
  //vertexAttributeDescriptions[4].location = 4;
  //vertexAttributeDescriptions[4].format   = VK_FORMAT_R32G32B32_SFLOAT; // bitangent (vec3)
  //vertexAttributeDescriptions[4].offset   = offsetof(VertexAll, bitangent);
//
  //vertexAttributeDescriptions[5].binding  = vertexBinding;
  //vertexAttributeDescriptions[5].location = 5;
  //vertexAttributeDescriptions[5].format   = VK_FORMAT_R32G32B32A32_SFLOAT; // color (vec4)
  //vertexAttributeDescriptions[5].offset   = offsetof(VertexAll, color);
//
  //vertexAttributeDescriptions[6].binding  = vertexBinding;
  //vertexAttributeDescriptions[6].location = 6;
  //vertexAttributeDescriptions[6].format   = VK_FORMAT_R32G32B32A32_SINT; // boneIndices (ivec4)
  //vertexAttributeDescriptions[6].offset   = offsetof(VertexAll, boneIndices);
//
  //vertexAttributeDescriptions[7].binding  = vertexBinding;
  //vertexAttributeDescriptions[7].location = 7;
  //vertexAttributeDescriptions[7].format   = VK_FORMAT_R32G32B32A32_SFLOAT; // boneWeights (vec4)
  //vertexAttributeDescriptions[7].offset   = offsetof(VertexAll, boneWeights);
//
  //vertexInputInfo.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  //vertexInputInfo.vertexBindingDescriptionCount   = 1;
  //vertexInputInfo.pVertexBindingDescriptions      = &vertexBindingDesc;
  //vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributeDescriptions.size());
  //vertexInputInfo.pVertexAttributeDescriptions    = vertexAttributeDescriptions.data();
#endif