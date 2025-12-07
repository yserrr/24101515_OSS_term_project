#ifndef MESH_HPP
#define MESH_HPP
#include "../resource/vertex.hpp"
#include "material.hpp"
#include "vk_memory_allocator.hpp"
// tool box mesh -> simple
// rendering resource -> batch style
// don't optimize temp resource
struct Mesh{
  Mesh (gpu::VkMemoryAllocator & allocator) : allocator(&allocator){}
  Mesh(const std::vector<VertexAll> &vertices,
       const std::vector<uint32_t> &indices,
       gpu::VkMemoryAllocator &allocator);
  Mesh(const std::vector<VertexAll> &vertices,
       const std::vector<uint32_t> &indices
       );
  Mesh (Mesh &&other) = default;

  ~Mesh();
  bool selected = false;
  const std::vector<VertexAll> &getVertices() const;
  const std::vector<uint32_t> &getIndices() const;
  void dynMeshUpdate(VkCommandBuffer commandBuffer);
  void copyBuffer(VkCommandBuffer commandBuffer) const;
  void bind(VkCommandBuffer commandBuffer);
  void draw(VkCommandBuffer commandBuffer) const;
  void recenterMesh();
  void reNomalCompute();
  std::string name;
  std::vector<VertexAll> vertices;
  std::vector<uint32_t> indices;
  gpu::VkMemoryAllocator *allocator;
  VkDeviceSize vertexSize;
  VkDeviceSize indiceSize;
};

#endif //MESH_HPP