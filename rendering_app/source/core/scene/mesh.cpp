#include "mesh.hpp"

Mesh::~Mesh() {}

Mesh::Mesh(const std::vector<VertexAll> &vertices,
           const std::vector<uint32_t> &indices,
           gpu::VkMemoryAllocator &allocator) : vertices(vertices), indices(indices), allocator(&allocator)
{
  recenterMesh();
  reNomalCompute();
  vertexSize      = (sizeof(vertices[0])) * vertices.size();
  indiceSize      = sizeof(indices[0]) * indices.size();
  vertexBuffer    = std::make_unique<StaticBuffer>(allocator, vertexSize, BufferType::VERTEX);
  indexBuffer     = std::make_unique<StaticBuffer>(allocator, indiceSize, BufferType::INDEX);
  vertexBuffer->getStagingBuffer(vertices.data());
  indexBuffer->getStagingBuffer(indices.data());
  vertexBuffer->createMainBuffer();
  indexBuffer->createMainBuffer();
}

Mesh::Mesh(const std::vector<VertexAll>& vertices, const std::vector<uint32_t>& indices):allocator(nullptr)
{

}

void Mesh::copyBuffer(VkCommandBuffer commandBuffer) const
{
  vertexBuffer->copyBuffer(commandBuffer);
  indexBuffer->copyBuffer(commandBuffer);
}

void Mesh::bind(VkCommandBuffer commandBuffer)
{
  VkDeviceSize offsets[] = {0};
  VkBuffer vertexBufs[]  = {*(vertexBuffer->getBuffer())};
  vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBufs, offsets);
  vkCmdBindIndexBuffer(commandBuffer, *(indexBuffer->getBuffer()), 0, VK_INDEX_TYPE_UINT32);
}

void Mesh::draw(VkCommandBuffer commandBuffer) const
{
  vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
}


const std::vector<VertexAll> &Mesh::getVertices() const
{
  return vertices;
}

const std::vector<uint32_t> &Mesh::getIndices() const
{
  return indices;
}

void Mesh::recenterMesh()
{
  if (vertices.empty()) return;
  glm::vec3 centroid(0.0f);

  for (const auto &v: vertices)
  {
    centroid += v.position;
  }
  centroid /= static_cast<float>(vertices.size());
  for (auto &v: vertices)
  {
    v.position -= centroid;
  }
}

void Mesh::reNomalCompute()
{
  for (auto &v: vertices)
  {
    v.normal = glm::vec3(0.0f);
  }
  for (size_t i = 0; i < indices.size(); i += 3)
  {
    VertexAll &v0 = vertices[indices[i]];
    VertexAll &v1 = vertices[indices[i + 1]];
    VertexAll &v2 = vertices[indices[i + 2]];

    glm::vec3 normal = glm::normalize(glm::cross(v1.position - v0.position, v2.position - v0.position));
    v0.normal += normal;
    v1.normal += normal;
    v2.normal += normal;
  }
  for (auto &v: vertices)
  {
    v.normal = glm::normalize(v.normal);
  }
}