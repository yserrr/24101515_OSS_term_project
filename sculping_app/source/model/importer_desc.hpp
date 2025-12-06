#ifndef MYPROJECT_IMPORT_DESC_HPP
#define MYPROJECT_IMPORT_DESC_HPP

#include <glm/mat4x4.hpp>
#include <string>
#include <vector>
#include <glm/gtc/quaternion.hpp>
#include <vertex.hpp>

enum class PrimitiveType: uint32_t{
  POINT    = 0x1,
  LINE     = 0x2,
  TRIANGLE = 0x4,
  POLYGON  = 0x8,
};

/// todo: setting primitive -> batch rendering
struct MeshDesc{
  std::string name;
  std::vector<VertexAll> vertices;
  std::vector<uint32_t> indices;
  uint32_t primitives;
  uint32_t numVertices;
  uint32_t numIndices;
  uint32_t uvChanels;
  int materialIndex = -1;
  unsigned int mPrimitiveTypes;
};

enum DescriptorBindFlags : uint32_t{
  ImageBaseColor       = 0x00000001,
  ImageNormalMap       = 0x00000002,
  ImageMetallicMap     = 0x00000004,
  ImageRoughnessMap    = 0x00000008,
  ImageAoMap           = 0x00000010,
  ImageEmissionMap     = 0x00000020,
  UseConstantBaseColor = 0x00000040,
  AlphaTest            = 0x00000100,
  AlphaBlend           = 0x00000200,
  DoubleSided          = 0x00000400,
  UseVertexColor       = 0x00000800,
};

inline DescriptorBindFlags operator|(const uint32_t &flag, const DescriptorBindFlags &desc)
{
  return static_cast<DescriptorBindFlags>(flag | static_cast<uint32_t>(desc));
}

inline DescriptorBindFlags operator|(const DescriptorBindFlags &l, const DescriptorBindFlags &r)
{
  return static_cast<DescriptorBindFlags>(static_cast<uint32_t>(l) | static_cast<uint32_t>(r));
}

inline DescriptorBindFlags operator&(const uint32_t &flag, const DescriptorBindFlags &desc)
{
  return static_cast<DescriptorBindFlags>(flag & static_cast<uint32_t>(desc));
}

struct alignas(16) MaterialDesc{
// x=metallic, y=roughness, z=ao, w=emissionScale
  glm::vec4 baseColor  = glm::vec4(1.0f);
  glm::vec4 params     = glm::vec4(0.0f);
  uint32_t flags       = 0;
  int32_t texAlbedo    = -1;
  int32_t texNormal    = -1;
  int32_t texMetallic  = -1;
  int32_t texRoughness = -1;
  int32_t texAo        = -1;
  int32_t texEmissive  = -1;
  int32_t padding      = 0;
};

enum class MaterialMap : uint32_t{
  // materail: index-> get text meta
  Albedo = 0,
  Normal,
  Metallic,
  Roughness,
  Ao,
  Emissive,
  Count
};

enum LightFlags : uint32_t{
  LightFlag_None       = 0x00000000,
  LightFlag_Static     = 0x00000002,
  LightFlag_CastShadow = 0x00000001,
};

struct LightDesc{
  std::string name;
  int type        = 0;
  glm::vec3 color = glm::vec3(1);
  glm::vec3 pos   = glm::vec3(0);
  glm::vec3 dir   = glm::vec3(0, -1, 0);
  float innerCone = 0.0f;
  float outerCone = 0.0f;
};

enum class LightSetBinding : uint32_t{
  LightUbo       = 0,
  IblIrradiance  = 1,
  IblPrefiltered = 2,
  IblBrdfLut     = 3,
  ShadowMaps     = 4,
};

struct CameraDesc{
  std::string name;
  float fovY       = 45.0f;
  float aspect     = 1.0f;
  float nearZ      = 0.1f;
  float farZ       = 1000.0f;
  glm::vec3 pos    = glm::vec3(0);
  glm::vec3 lookAt = glm::vec3(0, 0, -1);
  glm::vec3 up     = glm::vec3(0, 1, 0);
};

struct NodeDesc{
  std::string name;
  glm::mat4 local; // 로컬 변환
  glm::mat4 world; // 월드 변환
  std::vector<int> meshIndices;
  std::vector<int> children;
  int parent = -1;
};

// 애니메이션(간략화: 노드 트랜스폼 애니메이션)
struct VecKey{
  double t;
  glm::vec3 v;
};

struct QuatKey{
  double t;
  glm::quat q;
};

struct NodeAnim{
  std::string nodeName;
  std::vector<VecKey> positions;
  std::vector<QuatKey> rotations;
  std::vector<VecKey> scales;
};

struct AnimationClip{
  std::string name;
  double duration       = 0.0;
  double ticksPerSecond = 0.0;
  std::vector<NodeAnim> channels;
};

enum class TexUsage : uint32_t{
  ALBEDO,
  NORMAL,
  METAILIC,
  ROUGHNESS,
  AO,
  EMISSIVE,
};

struct TextureDesc{
  std::string path;
  int embeddedIndex = -1;
  TexUsage flag;
};

enum TexCreateFlags : uint32_t{
  TexFlag_None         = 0x00000000,
  TexFlag_GenerateMips = 0x00000001,
  TexFlag_FlipYOnLoad  = 0x00000002,
  TexFlag_ForceRGBA8   = 0x00000004,
  TexFlag_OwnSampler   = 0x00000008,
};

struct ImportResult{
  std::vector<MeshDesc> meshes;
  std::vector<MaterialDesc> materials;
  std::vector<TextureDesc> textures;
  std::vector<NodeDesc> nodes; // 0번이 루트
  std::vector<AnimationClip> animations;
  std::vector<LightDesc> lights;
  std::vector<CameraDesc> cameras;

  struct EmbeddedTex{
    std::string name;
    const uint8_t *data;
    size_t size;
    bool isCompressed;
  };

  std::vector<EmbeddedTex> embedded;
};

#endif //MYPROJECT_IMPORT_DESC_HPP