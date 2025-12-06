#ifndef RESOURCE_MANAGER
#define RESOURCE_MANAGER
#include <../resource/texture.hpp>
#include <../model/mesh.hpp>
#include <light.hpp>
#include <common.hpp>
#include <../resource/descriptor_manager.hpp>
#include <importer.hpp>
#include <material.hpp>
#include <camera.hpp>
#include "buffer_builder.hpp"
#include <unordered_map>
#include "descriptor_uploader.hpp"
#include "push_constant.hpp"
#include "model.hpp"
#include <pipeline_pool.hpp>
#include <shader_pool.hpp>

using Key = std::string;

struct ResourceManagerCreateInfo{
  VkDevice device            = VK_NULL_HANDLE;
  MemoryAllocator *allocator = nullptr;
  uint32_t maxInflight       = 3;
};

using bindingIndex = uint32_t;

class ResourceManager{
  friend class App;
  friend class RenderingSystem;
  friend class UIRenderer;

public:
  ResourceManager(const ResourceManagerCreateInfo &info);
  ~ResourceManager();
  void updateDescriptorSet(uint32_t currentFrame);
  void uploadMesh(VkCommandBuffer command, std::string path);
  void uploadTexture(VkCommandBuffer command, std::string path);
  VulkanTexture *getTextures(bindingIndex index);

  void setLight();
  LightBuilder lightBuilder;
  std::vector<BufferContext> mainCamBuffers_;
  std::vector<VulkanTexture *> uploadedTexures;
  std::unique_ptr<DescriptorManager> descriptorManager;

  Model selectedModel{};
  Camera *getCamera();
  ImporterEx importer_;
  uint32_t maxInflight_ = 3;
  bindingIndex currentBinding_ = 0;
private:
  bool shouldUpdate;
  VkDevice device_;
  MemoryAllocator &allocator_;
  VkDeviceSize minUboAlign_;
  VkPhysicalDeviceProperties physicalDeviceProperties;
  DescriptorUploader descriptorUploader_;
  VkBindlessDescriptor bindlessDescirptor_;
  std::vector<VulkanTexture *> uiNeedTextures;
  std::unique_ptr<UBOBuilder> uboBuilder_;
  std::unique_ptr<PipelinePool> pipelinePool_;
  std::unique_ptr<ShaderPool> fragShaderPool;
  std::unique_ptr<ShaderPool> vertexShaderPool;
  std::unordered_map<Key, std::unique_ptr<Mesh> > meshes_;
  std::unordered_map<Key, std::unique_ptr<Model> > models_;
  std::unordered_map<Key, std::unique_ptr<Material> > materials_;


  std::vector <std::unique_ptr<VulkanTexture> > textures_;
  std::unique_ptr<SamplerBuilder> samplerBuilder_;
  std::shared_ptr<VulkanTexture> nomal;
  std::unique_ptr<Camera> camera;
};

#endif