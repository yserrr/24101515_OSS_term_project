#ifndef IMPORTER_HPP
#define IMPORTER_HPP 

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "importer_desc.hpp"
#include <mesh.hpp>
#include <stdexcept>
#include <vertex.hpp>

class ImporterEx{
public:
  ImporterEx() = default;
  ImportResult loadScene(const char *filepath);
  Mesh loadModel(const char *filepath, MemoryAllocator &allocator);

private:
  void processMaterialsAndTextures(const aiScene *scene, const std::string &folder, ImportResult &out);
  void processMeshes(const aiScene *scene, ImportResult &out);
  void processMeshesWithOnlyTriangles(const aiScene *scene, ImportResult &out);
  void processBones(aiMesh *mesh, std::vector<VertexAll> &vertices);
  int addTexture(ImportResult &out, TexUsage slot, const std::string &path, int embeddedIndex);
  int buildNodeRecursive(
      const aiScene *scene,
      const aiNode *node,
      int parent,
      const glm::mat4 &parentWorld,
      ImportResult &out
    );
  void buildNodeHierarchy(const aiScene *scene, ImportResult &out);
  void processAnimations(const aiScene *scene, ImportResult &out);
  void processCameras(const aiScene *scene, ImportResult &out);
  void processLights(const aiScene *scene, ImportResult &out);

private:
  static glm::mat4 ToGlm(const aiMatrix4x4 &m);
  static glm::vec3 ToGlm(const aiVector3D &v);
  static glm::quat ToGlm(const aiQuaternion &q);
};

#endif