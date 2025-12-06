#ifndef MYPROJECT_GRID_HPP
#define MYPROJECT_GRID_HPP
#include <glm/glm.hpp>
#include <importer_desc.hpp>
#include <vertex.hpp>
#include <vector>

///todo: template-> select vertex structure
struct GeometryFactory{
  static MeshDesc generateQuad();
  static MeshDesc generateSpheres(
      glm::vec3 center,
      float radius,
      int stacks = 32,
      int slices = 32
    );
  static MeshDesc generateCubes(
      glm::vec3 center,
      glm::vec3 size
    );
  static MeshDesc GenerateGrid(
      glm::vec3 center,
      int xCount,
      int yCount,
      float xGridSize,
      float yGridSize
    );
};

#endif //MYPROJECT_GRID_HPP