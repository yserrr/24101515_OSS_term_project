//
// Created by ljh on 25. 9. 1..
//
#include <base_geometry.hpp>

MeshDesc GeometryFactory::generateCubes(glm::vec3 center, glm::vec3 size) {}

MeshDesc GeometryFactory::GenerateGrid(
    glm::vec3 center,
    int xCount,
    int yCount,
    float xGridSize,
    float yGridSize
  )
{
  MeshDesc descs;
  descs.vertices.reserve((xCount + 1) * (yCount + 1));
  //descs.primitives = PrimitiveType::LINE;
  float startX = center.x - (xGridSize * xCount) / 2.0f;
  float startY = center.y - (yGridSize * yCount) / 2.0f;
  float z      = center.z;
  for (int i = 0; i <= xCount; ++i)
  {
    for (int j = 0; j <= yCount; ++j)
    {
      VertexAll v;
      v.position = glm::vec3(startX + i * xGridSize, startY + j * yGridSize, z);
      descs.vertices.push_back(v);
    }
  }
  descs.name        = "grid";
  descs.numVertices = descs.vertices.size();
  return descs;
}

MeshDesc GeometryFactory::generateSpheres(
    glm::vec3 center,
    float radius,
    int stacks,
    int slices
  )
{
  MeshDesc desc;
  for (int i = 0; i <= stacks; ++i)
  {
    float v     = float(i) / stacks;
    float theta = v * M_PI;

    for (int j = 0; j <= slices; ++j)
    {
      float u   = float(j) / slices;
      float phi = u * 2.0f * M_PI;

      VertexAll vert{};
      vert.position[0] = center[0] + radius * sinf(theta) * cosf(phi);
      vert.position[1] = center[1] + radius * cosf(theta);
      vert.position[2] = center[2] + radius * sinf(theta) * sinf(phi);

      vert.normal[0] = vert.position[0];
      vert.normal[1] = vert.position[1];
      vert.normal[2] = vert.position[2];

      vert.uv[0] = u;
      vert.uv[1] = v;

      desc.vertices.push_back(vert);
    }
  }

  for (int i = 0; i < stacks; ++i)
  {
    for (int j = 0; j < slices; ++j)
    {
      uint32_t first  = i * (slices + 1) + j;
      uint32_t second = first + slices + 1;
      desc.indices.push_back(first);
      desc.indices.push_back(second);
      desc.indices.push_back(first + 1);
      desc.indices.push_back(second);
      desc.indices.push_back(second + 1);
      desc.indices.push_back(first + 1);
    }
  }
  desc.name        = "spheres";
  desc.numVertices = desc.vertices.size();
  desc.numIndices  = desc.indices.size();
  desc.uvChanels   = 1;
  //desc.primitives = PrimitiveType::TRIANGLE;
  return desc;
};