#ifndef MYPROJECT_VK_VERTEX_ATTR_HPP
#define MYPROJECT_VK_VERTEX_ATTR_HPP
#include <cstdint>
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

namespace gpu{
  enum class VertexType: uint32_t{
    QUAD,
    PC,
    PUVN,
    PUVNTC,
    ALL,
    BACKGROUND
  };


  struct Quad{
    glm::vec3 point1;
    glm::vec3 point2;
  };

  struct VertexPC{
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
}



#endif //MYPROJECT_VK_VERTEX_ATTR_HPP