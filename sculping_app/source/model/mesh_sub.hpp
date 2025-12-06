//
// Created by ljh on 25. 9. 16..
//

#ifndef MYPROJECT_MESH_SUB_HPP
#define MYPROJECT_MESH_SUB_HPP

#include <glm/vec3.hpp>
#include <vector>
#include <vertex.hpp>

enum class Topology: uint32_t{
  POINT,
  LINE,
  TRIANGLE
};

enum class IndexType: uint32_t{
  UINT16,
  UINT32
};

struct Submesh{
  uint32_t startIndex   = 0;
  uint32_t indexCount   = 0;
  uint32_t vertexOffset = 0; // vertices 배열 내 오프셋
};

struct mesh_cmd{
  mesh_cmd(
      const VertexAll *vData,
      const uint32_t *iData,
      const std::vector<Submesh> &submeshes
    )
    : vData(vData),
      iData(iData),
      submeshes(submeshes) {}

  const VertexAll *vData;
  const uint32_t *iData;
  uint32_t indexCnt    = 0;
  uint32_t vertexCnt   = 0;
  uint64_t vBufferSize = 0;
  uint64_t iBufferSize = 0;
  uint64_t gblVBOffset = 0;
  uint64_t gblIBOffset = 0;
  Topology topology    = Topology::TRIANGLE;
  IndexType indexType  = IndexType::UINT32;
  const std::vector<Submesh> &submeshes;
};

#endif //MYPROJECT_MESH_SUB_HPP