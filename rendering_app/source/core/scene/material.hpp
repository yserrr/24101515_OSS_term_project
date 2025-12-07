#ifndef MATERIAL_builderHPP
#define MATERIAL_builderHPP
#include "gpu_context.hpp"
#define MATERIAL_MAX 32

struct Material
{
  std::string name;
  glm::vec4 baseColor = glm::vec4(1.0f);
  glm::vec3 emissiveColor = glm::vec3(0.0f);
  float metallic = 0.0f; //intense
  float roughness = 1.0f;
  float ao = 1.0f;
  float emission = 0.0f;
  float normalScale = 1.0f;
  float alphaCutoff = 0.5f;
  uint32_t albedoIndex = 0;
  uint32_t normalIndex = 0;
  uint32_t metallicTexIndex = 0;
  uint32_t roughnessTexIndex = 0;
  uint32_t aoTexIndex = 0;
  uint32_t emissionTexIndex = 0;
  uint32_t padding = 0;
};

class MaterialBuilder
{
  public:
  gpu::GPUBuffer buffer;
};

//void setMapUV(MaterialMap map, float tilingU, float tilingV, float offsetU, float offsetV);
//void setMapStrength(MaterialMap map, float s);
#endif
