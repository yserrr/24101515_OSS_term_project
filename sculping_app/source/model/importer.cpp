//
// Created by ljh on 25. 9. 16..
//

// importer_ex.cpp
#include <stdexcept>
#include <glm/glm.hpp>
#include <vertex.hpp>
#include <glm/gtc/quaternion.hpp>
#include <common.hpp>
#include "importer.hpp"
#include"importer_desc.hpp"
#include <assimp/postprocess.h>
#include <filesystem>
using std::string;
namespace fs = std::filesystem;

ImportResult ImporterEx::loadScene(const char *filepath)
{
  ImportResult out;
  Assimp::Importer importer;
  // aiProcess_ValidateDataStructure // 디버그용
  const aiScene *scene = importer.ReadFile(
                                           filepath,
                                           aiProcess_Triangulate |
                                           aiProcess_JoinIdenticalVertices |
                                           aiProcess_GenNormals |
                                           aiProcess_CalcTangentSpace |
                                           aiProcess_LimitBoneWeights |
                                           aiProcess_ImproveCacheLocality |
                                           aiProcess_GenUVCoords |
                                           aiProcess_SortByPType
                                          );

  if (!scene || !scene->HasMeshes())
  {
    spdlog::error("Failed toe load scene: {}", importer.GetErrorString());
  }
  const string folder = std::filesystem::path(filepath).parent_path().string();
  if (scene->HasTextures())
  {
    out.embedded.reserve(scene->mNumTextures);
    //embedded texture
    for (unsigned i = 0; i < scene->mNumTextures; ++i)
    {
      const aiTexture *t = scene->mTextures[i];
      ImportResult::EmbeddedTex e{};
      e.name = t->mFilename.C_Str();
      if (t->mHeight == 0)
      {
        e.data         = reinterpret_cast<const uint8_t *>(t->pcData);
        e.size         = static_cast<size_t>(t->mWidth);
        e.isCompressed = true;
      } else
      {
        // 비압축 BGRA32 배열
        e.data         = reinterpret_cast<const uint8_t *>(t->pcData);
        e.size         = static_cast<size_t>(t->mWidth * t->mHeight * 4);
        e.isCompressed = false;
      }
      out.embedded.push_back(e);
    }
  }
  processMaterialsAndTextures(scene, folder, out);
  processMeshesWithOnlyTriangles(scene, out);
  buildNodeHierarchy(scene, out);
  processAnimations(scene, out);
  processLights(scene, out);
  processCameras(scene, out);
  spdlog::info("Import done: meshes={}, materials={}, textures={}, nodes={}, anims={}, lights={}, cameras={}",
               out.meshes.size(),
               out.materials.size(),
               out.textures.size(),
               out.nodes.size(),
               out.animations.size(),
               out.lights.size(),
               out.cameras.size());
  return out;
}

void ImporterEx::processMaterialsAndTextures(const aiScene *scene, const string &folder, ImportResult &out)
{
  //embedded texture
  out.materials.resize(scene->mNumMaterials);
  auto fetchTexture = [&](aiMaterial *mtl, aiTextureType type, TexUsage slot) -> int
  {
    if (mtl->GetTextureCount(type) == 0) return -1;
    aiString path;
    if (AI_SUCCESS != mtl->GetTexture(type, 0, &path)) return -1;
    std::string str = path.C_Str();

    size_t pos            = str.find_last_of("/\\");
    std::string filename  = (pos == std::string::npos) ? str : str.substr(pos + 1);
    std::string txtfolder = "/texture/";
    filename              = txtfolder + filename;
    string full           = folder + filename;
    if (str[0] == '*')
    {
      //embedded
      std::string emb    = "embedded:";
      unsigned int index = std::stoi(path.C_Str());
      str                = emb + str;
      return addTexture(out, slot, str, index);
    }
    return addTexture(out, slot, full, -1);
  };

  for (unsigned i = 0; i < scene->mNumMaterials; ++i)
  {
    aiMaterial *mtl = scene->mMaterials[i];
    MaterialDesc md{};
    // BaseColor (PBR) 또는 Diffuse
    aiColor4D base{};
    if (AI_SUCCESS == mtl->Get(AI_MATKEY_BASE_COLOR, base) ||
        AI_SUCCESS == mtl->Get(AI_MATKEY_COLOR_DIFFUSE, base))
    {
      md.baseColor = glm::vec4(base.r, base.g, base.b, base.a);
    }

    // Metallic/Roughness/AO/Emissive scale
    float metallic = 0.f, roughness = 1.f, ao = 1.f, emissive = 0.f;
    mtl->Get(AI_MATKEY_METALLIC_FACTOR, metallic);
    mtl->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness);
    aiColor3D emiColor(0, 0, 0);
    if (AI_SUCCESS == mtl->Get(AI_MATKEY_COLOR_EMISSIVE, emiColor))
    {
      emissive = std::max({emiColor.r, emiColor.g, emiColor.b});
    }
    md.params = glm::vec4(metallic, roughness, ao, emissive);

    // 텍스처(우선순위: PBR → 레거시)
    md.texAlbedo = fetchTexture(mtl, aiTextureType_BASE_COLOR, TexUsage::ALBEDO);
    md.texNormal = fetchTexture(mtl, aiTextureType_NORMALS, TexUsage::NORMAL);
    if (md.texAlbedo < 0) md.texAlbedo = fetchTexture(mtl, aiTextureType_DIFFUSE, TexUsage::ALBEDO);
    if (md.texNormal < 0) md.texNormal = fetchTexture(mtl, aiTextureType_HEIGHT, TexUsage::NORMAL);
    // 일부 에셋이 노말을 height에 넣음

    md.texMetallic  = fetchTexture(mtl, aiTextureType_METALNESS, TexUsage::METAILIC);
    md.texRoughness = fetchTexture(mtl, aiTextureType_DIFFUSE_ROUGHNESS, TexUsage::ROUGHNESS);
    md.texAo        = fetchTexture(mtl, aiTextureType_AMBIENT_OCCLUSION, TexUsage::AO);
    md.texEmissive  = fetchTexture(mtl, aiTextureType_EMISSIVE, TexUsage::EMISSIVE);

    // 플래그 구성
    if (md.texAlbedo >= 0) md.flags |= DescriptorBindFlags::ImageBaseColor;
    else md.flags |= DescriptorBindFlags::UseConstantBaseColor;
    if (md.texNormal >= 0) md.flags |= DescriptorBindFlags::ImageNormalMap;
    if (md.texMetallic >= 0) md.flags |= DescriptorBindFlags::ImageMetallicMap;
    if (md.texRoughness >= 0) md.flags |= DescriptorBindFlags::ImageRoughnessMap;
    if (md.texAo >= 0) md.flags |= DescriptorBindFlags::ImageAoMap;
    if (md.texEmissive >= 0) md.flags |= DescriptorBindFlags::ImageEmissionMap;

    // 더블사이드/블렌드 등(가능하면)
    int twoSided = 0;
    if (AI_SUCCESS == mtl->Get(AI_MATKEY_TWOSIDED, twoSided) && twoSided)
    {
      md.flags |= DescriptorBindFlags::DoubleSided;
    }

    out.materials[i] = md;
  }
}

int ImporterEx::addTexture(ImportResult &out, TexUsage slot, const string &path, int embeddedIndex)
{
  TextureDesc t{};
  t.flag          = slot;
  t.path          = path;
  t.embeddedIndex = embeddedIndex;
  out.textures.push_back(t);
  return static_cast<int>(out.textures.size() - 1);
}

void ImporterEx::processMeshes(const aiScene *scene, ImportResult &out)
{
  out.meshes.reserve(scene->mNumMeshes);
  for (unsigned m = 0; m < scene->mNumMeshes; ++m)
  {
    aiMesh *am = scene->mMeshes[m];
    MeshDesc md{};
    md.name          = am->mName.C_Str();
    md.materialIndex = static_cast<int>(am->mMaterialIndex);

    md.vertices.resize(am->mNumVertices);
    for (unsigned i = 0; i < am->mNumVertices; ++i)
    {
      VertexAll v{};
      // pos
      v.position[0] = am->mVertices[i].x;
      v.position[1] = am->mVertices[i].y;
      v.position[2] = am->mVertices[i].z;
      // nrm
      if (am->HasNormals())
      {
        v.normal[0] = am->mNormals[i].x;
        v.normal[1] = am->mNormals[i].y;
        v.normal[2] = am->mNormals[i].z;
      }
      // uv0
      if (am->HasTextureCoords(0))
      {
        v.uv[0] = am->mTextureCoords[0][i].x;
        v.uv[1] = am->mTextureCoords[0][i].y;
      } else
      {
        v.uv[0] = v.uv[1] = 0.f;
      }
      // tangent/bitangent
      if (am->HasTangentsAndBitangents())
      {
        v.tangent[0]   = am->mTangents[i].x;
        v.tangent[1]   = am->mTangents[i].y;
        v.tangent[2]   = am->mTangents[i].z;
        v.bitangent[0] = am->mBitangents[i].x;
        v.bitangent[1] = am->mBitangents[i].y;
        v.bitangent[2] = am->mBitangents[i].z;
      }
      // color0
      if (am->HasVertexColors(0))
      {
        v.color[0] = am->mColors[0][i].r;
        v.color[1] = am->mColors[0][i].g;
        v.color[2] = am->mColors[0][i].b;
        v.color[3] = am->mColors[0][i].a;
      } else
      {
        v.color[0] = v.color[1] = v.color[2] = 1.f;
        v.color[3] = 1.f;
      }
      // bone init
      for (int k = 0; k < 4; ++k)
      {
        v.boneIndices[k] = 0;
        v.boneWeights[k] = 0.f;
      }

      md.vertices[i] = v;
    }
    // indices
    md.indices.reserve(am->mNumFaces * 3);
    for (unsigned f = 0; f < am->mNumFaces; ++f)
    {
      const aiFace &face = am->mFaces[f];
      for (unsigned idx = 0; idx < face.mNumIndices; ++idx)
      {
        md.indices.push_back(face.mIndices[idx]);
      }
    }

    // bones
    if (am->HasBones()) processBones(am, md.vertices);

    out.meshes.push_back(std::move(md));
  }
}

///
void ImporterEx::processMeshesWithOnlyTriangles(const aiScene *scene, ImportResult &out)
{
  out.meshes.reserve(scene->mNumMeshes);
  for (unsigned m = 0; m < scene->mNumMeshes; ++m)
  {
    /// todo:
    ///  rendering mode setting with line
    ///  now: not triangle-> not draw
    ///  need to set up with dynamic topology
    ///  need: topology
    ///  -> need type binding pipeline and draw call
    aiMesh *am = scene->mMeshes[m];
    MeshDesc md{};
    md.name = am->mName.C_Str();

    md.materialIndex = static_cast<int>(am->mMaterialIndex);
    md.primitives    = am->mPrimitiveTypes;

    std::vector<VertexAll> vertices;
    std::vector<uint32_t> indices;
    for (unsigned f = 0; f < am->mNumFaces; ++f)
    {
      const aiFace &face = am->mFaces[f];
      if (face.mNumIndices != 3)
      {
        spdlog::info("passed, need to pipeline update for muliy geometry");
        continue;
      }
      uint32_t startIndex = static_cast<uint32_t>(vertices.size());
      //ALWAYS INDEX : 3 CASE
      for (unsigned idx = 0; idx < 3; ++idx)
      {
        const auto &srcV = am->mVertices[face.mIndices[idx]];
        VertexAll v{};
        v.position[0] = srcV.x;
        v.position[1] = srcV.y;
        v.position[2] = srcV.z;
        if (am->HasNormals())
        {
          v.normal[0] = am->mNormals[face.mIndices[idx]].x;
          v.normal[1] = am->mNormals[face.mIndices[idx]].y;
          v.normal[2] = am->mNormals[face.mIndices[idx]].z;
        }

        if (am->HasTextureCoords(0))
        {
          auto &uv = am->mTextureCoords[0][face.mIndices[idx]];
          v.uv[0]  = uv.x;
          v.uv[1]  = uv.y;
        } else
        {
          spdlog::info("no uv need to setting");
          v.uv[0] = v.uv[1] = 0.f;
        }
        if (am->HasTangentsAndBitangents())
        {
          v.tangent[0]   = am->mTangents[face.mIndices[idx]].x;
          v.tangent[1]   = am->mTangents[face.mIndices[idx]].y;
          v.tangent[2]   = am->mTangents[face.mIndices[idx]].z;
          v.bitangent[0] = am->mBitangents[face.mIndices[idx]].x;
          v.bitangent[1] = am->mBitangents[face.mIndices[idx]].y;
          v.bitangent[2] = am->mBitangents[face.mIndices[idx]].z;
        }
        if (am->HasVertexColors(0))
        {
          v.color[0] = am->mColors[0][face.mIndices[idx]].r;
          v.color[1] = am->mColors[0][face.mIndices[idx]].g;
          v.color[2] = am->mColors[0][face.mIndices[idx]].b;
          v.color[3] = am->mColors[0][face.mIndices[idx]].a;
        } else
        {
          v.color[0] = v.color[1] = v.color[2] = 1.f;
          v.color[3] = 1.f;
        }
        // bone init
        for (int k = 0; k < 4; ++k)
        {
          v.boneIndices[k] = 0;
          v.boneWeights[k] = 0.f;
        }
        vertices.push_back(v);
      }
      indices.push_back(startIndex);
      indices.push_back(startIndex + 1);
      indices.push_back(startIndex + 2);
    }
    md.vertices = vertices;
    md.indices  = indices;

    if (am->HasBones())
    {
      processBones(am, md.vertices);
    }
    out.meshes.push_back(std::move(md));
  }
}

// 본 가중치 4개 제한 + 정규화
void ImporterEx::processBones(aiMesh *mesh, std::vector<VertexAll> &vertices)
{
  struct BW{
    uint32_t i;
    float w;
  };
  std::vector<std::array<BW, 4> > accum(vertices.size());
  std::vector<int> counts(vertices.size(), 0);

  auto push = [&](uint32_t vtx, uint32_t boneId, float w)
  {
    auto &c = counts[vtx];
    if (w <= 0.f) return;
    if (c < 4)
    {
      accum[vtx][c++] = BW{boneId, w};
    } else
    {
      // 가장 작은 것과 비교 교체(간단)
      int minIdx = 0;
      for (int k = 1; k < 4; ++k) if (accum[vtx][k].w < accum[vtx][minIdx].w) minIdx = k;
      if (accum[vtx][minIdx].w < w) accum[vtx][minIdx] = BW{boneId, w};
    }
  };

  for (unsigned b = 0; b < mesh->mNumBones; ++b)
  {
    aiBone *bone = mesh->mBones[b];
    for (unsigned w = 0; w < bone->mNumWeights; ++w)
    {
      const aiVertexWeight &vw = bone->mWeights[w];
      push(vw.mVertexId, b, vw.mWeight);
    }
  }

  for (size_t i = 0; i < vertices.size(); ++i)
  {
    float sum = 0.f;
    for (int k = 0; k < 4; ++k)
    {
      vertices[i].boneIndices[k] = (k < counts[i]) ? accum[i][k].i : 0;
      vertices[i].boneWeights[k] = (k < counts[i]) ? accum[i][k].w : 0.f;
      sum += vertices[i].boneWeights[k];
    }
    if (sum > 0.f)
    {
      for (int k = 0; k < 4; ++k) vertices[i].boneWeights[k] /= sum;
    } else
    {
      vertices[i].boneWeights[0] = 1.f;
    }
  }
}

// 노드 트리
void ImporterEx::buildNodeHierarchy(const aiScene *scene, ImportResult &out)
{
  out.nodes.clear();
  out.nodes.reserve(128);
  const int root = buildNodeRecursive(scene, scene->mRootNode, -1, glm::mat4(1.0f), out);
  (void) root; // 0일 것
}

int ImporterEx::buildNodeRecursive(
    const aiScene *scene,
    const aiNode *node,
    int parent,
    const glm::mat4 &parentWorld,
    ImportResult &out
  )
{
  NodeDesc nd{};
  nd.name   = node->mName.C_Str();
  nd.local  = ToGlm(node->mTransformation);
  nd.world  = parentWorld * nd.local;
  nd.parent = parent;

  // 해당 노드에 연결된 mesh 인덱스들
  for (unsigned i = 0; i < node->mNumMeshes; ++i)
  {
    nd.meshIndices.push_back(static_cast<int>(node->mMeshes[i]));
  }

  int myIndex = static_cast<int>(out.nodes.size());
  out.nodes.push_back(nd);

  // 자식 순회
  for (unsigned c = 0; c < node->mNumChildren; ++c)
  {
    int childIdx = buildNodeRecursive(scene, node->mChildren[c], myIndex, out.nodes[myIndex].world, out);
    out.nodes[myIndex].children.push_back(childIdx);
  }
  return myIndex;
}

// 애니메이션
void ImporterEx::processAnimations(const aiScene *scene, ImportResult &out)
{
  if (!scene->HasAnimations()) return;
  out.animations.reserve(scene->mNumAnimations);

  for (unsigned a = 0; a < scene->mNumAnimations; ++a)
  {
    const aiAnimation *anim = scene->mAnimations[a];
    AnimationClip clip{};
    clip.name           = anim->mName.C_Str();
    clip.duration       = anim->mDuration;
    clip.ticksPerSecond = anim->mTicksPerSecond > 0.0 ? anim->mTicksPerSecond : 25.0;

    for (unsigned c = 0; c < anim->mNumChannels; ++c)
    {
      const aiNodeAnim *ch = anim->mChannels[c];
      NodeAnim na{};
      na.nodeName = ch->mNodeName.C_Str();

      na.positions.reserve(ch->mNumPositionKeys);
      for (unsigned i = 0; i < ch->mNumPositionKeys; ++i)
        na.positions.push_back({ch->mPositionKeys[i].mTime, ToGlm(ch->mPositionKeys[i].mValue)});

      na.rotations.reserve(ch->mNumRotationKeys);
      for (unsigned i = 0; i < ch->mNumRotationKeys; ++i)
        na.rotations.push_back({ch->mRotationKeys[i].mTime, ToGlm(ch->mRotationKeys[i].mValue)});

      na.scales.reserve(ch->mNumScalingKeys);
      for (unsigned i = 0; i < ch->mNumScalingKeys; ++i)
        na.scales.push_back({ch->mScalingKeys[i].mTime, ToGlm(ch->mScalingKeys[i].mValue)});

      clip.channels.push_back(std::move(na));
    }
    out.animations.push_back(std::move(clip));
  }
}

// 라이트
void ImporterEx::processLights(const aiScene *scene, ImportResult &out)
{
  for (unsigned i = 0; i < scene->mNumLights; ++i)
  {
    const aiLight *l = scene->mLights[i];
    LightDesc d{};
    d.name      = l->mName.C_Str();
    d.type      = static_cast<int>(l->mType);
    d.color     = glm::vec3(l->mColorDiffuse.r, l->mColorDiffuse.g, l->mColorDiffuse.b);
    d.pos       = ToGlm(l->mPosition);
    d.dir       = ToGlm(l->mDirection);
    d.innerCone = l->mAngleInnerCone;
    d.outerCone = l->mAngleOuterCone;
    out.lights.push_back(d);
  }
}

///simple 
Mesh ImporterEx::loadModel(const char *filepath, MemoryAllocator &allocator)
{
  Assimp::Importer importer;
  const aiScene *scene = importer.ReadFile(filepath,
                                           aiProcess_Triangulate |
                                           aiProcess_FlipUVs |
                                           aiProcess_CalcTangentSpace |
                                           aiProcess_GenNormals);

  if (!scene || !scene->HasMeshes())
  {
    spdlog::error("Failed to load model or no meshes found: {}", importer.GetErrorString());
  }

  std::vector<VertexAll> vertices;
  std::vector<uint32_t> indices;

  for (unsigned int m = 0; m < scene->mNumMeshes; ++m)
  {
    aiMesh *mesh = scene->mMeshes[m];

    for (unsigned int i = 0; i < mesh->mNumVertices; ++i)
    {
      VertexAll vertex{};

      // position
      vertex.position[0] = mesh->mVertices[i].x;
      vertex.position[1] = mesh->mVertices[i].y;
      vertex.position[2] = mesh->mVertices[i].z;

      // normal
      if (mesh->HasNormals())
      {
        vertex.normal[0] = mesh->mNormals[i].x;
        vertex.normal[1] = mesh->mNormals[i].y;
        vertex.normal[2] = mesh->mNormals[i].z;
      } else
      {
        vertex.normal[0] = vertex.normal[1] = vertex.normal[2] = 0.0f;
      }

      // texCoord (first UV channel only)
      if (mesh->HasTextureCoords(0))
      {
        vertex.uv[0] = mesh->mTextureCoords[0][i].x;
        vertex.uv[1] = mesh->mTextureCoords[0][i].y;
      } else
      {
        vertex.uv[0] = vertex.uv[1] = 0.0f;
      }

      // tangent
      if (mesh->HasTangentsAndBitangents())
      {
        vertex.tangent[0] = mesh->mTangents[i].x;
        vertex.tangent[1] = mesh->mTangents[i].y;
        vertex.tangent[2] = mesh->mTangents[i].z;

        vertex.bitangent[0] = mesh->mBitangents[i].x;
        vertex.bitangent[1] = mesh->mBitangents[i].y;
        vertex.bitangent[2] = mesh->mBitangents[i].z;
      } else
      {
        vertex.tangent[0]   = vertex.tangent[1]   = vertex.tangent[2]   = 0.0f;
        vertex.bitangent[0] = vertex.bitangent[1] = vertex.bitangent[2] = 0.0f;
      }

      // color (only first color set)
      if (mesh->HasVertexColors(0))
      {
        vertex.color[0] = mesh->mColors[0][i].r;
        vertex.color[1] = mesh->mColors[0][i].g;
        vertex.color[2] = mesh->mColors[0][i].b;
        vertex.color[3] = mesh->mColors[0][i].a;
      } else
      {
        vertex.color[0] = vertex.color[1] = vertex.color[2] = 1.0f; // 기본 흰색
        vertex.color[3] = 1.0f;
      }

      // boneIndices and boneWeights 초기화 (0으로)
      for (int j = 0; j < 4; ++j)
      {
        vertex.boneIndices[j] = 0;
        vertex.boneWeights[j] = 0.0f;
      }
      // 본 정보는 aiBone에서 따로 채워야 함. (여기선 생략)
      vertices.push_back(vertex);
    }
    // 인덱스
    for (unsigned int f = 0; f < mesh->mNumFaces; ++f)
    {
      aiFace face = mesh->mFaces[f];
      for (unsigned int idx = 0; idx < face.mNumIndices; ++idx)
      {
        indices.push_back(face.mIndices[idx]);
      }
    }
  }
  spdlog::info("import models");
  return Mesh(vertices, indices, allocator);
}

// 카메라
void ImporterEx::processCameras(const aiScene *scene, ImportResult &out)
{
  for (unsigned i = 0; i < scene->mNumCameras; ++i)
  {
    const aiCamera *c = scene->mCameras[i];
    CameraDesc d{};
    d.name   = c->mName.C_Str();
    d.fovY   = c->mHorizontalFOV; // 주의: 수평/수직 FOV 해석은 타겟 엔진 규칙에 맞춰 보정
    d.nearZ  = c->mClipPlaneNear;
    d.farZ   = c->mClipPlaneFar;
    d.pos    = ToGlm(c->mPosition);
    d.lookAt = ToGlm(c->mLookAt);
    d.up     = ToGlm(c->mUp);
    // aspect는 씬에 없을 수도 있으니 추후 렌더 타겟에서 설정
    out.cameras.push_back(d);
  }
}

// 유틸 변환
glm::mat4 ImporterEx::ToGlm(const aiMatrix4x4 &m)
{
  return glm::mat4(
                   m.a1,
                   m.b1,
                   m.c1,
                   m.d1,
                   m.a2,
                   m.b2,
                   m.c2,
                   m.d2,
                   m.a3,
                   m.b3,
                   m.c3,
                   m.d3,
                   m.a4,
                   m.b4,
                   m.c4,
                   m.d4
                  );
}

glm::vec3 ImporterEx::ToGlm(const aiVector3D &v)
{
  return glm::vec3(v.x, v.y, v.z);
}

glm::quat ImporterEx::ToGlm(const aiQuaternion &q)
{
  return glm::quat(q.w, q.x, q.y, q.z);
}