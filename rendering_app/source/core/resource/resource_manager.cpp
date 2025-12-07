#include <filesystem>
#include <fstream>
#include "resource_manager.hpp"

#include "engine.hpp"
#include "../Render/RenderPassPool.hpp"
#include "../GPU/gpu_context.hpp"
extern cpu::Context ctx__;
ResourceManager::ResourceManager() = default;
ResourceManager::~ResourceManager() = default;

void ResourceManager::init()
{
  // std::unique_ptr<Material> base = std::make_unique<Material>();
  // base->name = "base";
  // materials_[base->name] = std::move(base);
  auto lightBuf =
    gpu::VkGraphBuilder::buildHostBuffer(sizeof(lightUBO), BufferType::UNIFORM);
  lightBuilder.buffer = lightBuf;
  Light light;
  light.color = glm::vec4(1.0f);
  light.transform.position = glm::vec3(0.0f, 0.0f, 4.5f);
  lightBuilder.build(light);
  lightBuilder.uploadData();
  lightBuilder.buffer.sets.resize(gpu::ctx__->renderingContext.maxInflight__);
  currentCamBuffer.resize(gpu::ctx__->renderingContext.maxInflight__);
  for (int i = 0; i < gpu::ctx__->renderingContext.maxInflight__; i++)
  {
    currentCamBuffer[i] = gpu::VkGraphBuilder::buildHostBuffer(sizeof(CameraUBO),
                                                               BufferType::UNIFORM);
    currentCamBuffer[i].sets.resize(gpu::ctx__->renderingContext.maxInflight__);
    currentCamBuffer[i].sets[i] = gpu::ctx__->pDescriptorAllocator->descriptorSets[i];
    gpu::ctx__->pDescriptorAllocator->writeUbo(currentCamBuffer[i].bufferh_,
                                               currentCamBuffer[i].size_,
                                               currentCamBuffer[i].sets[i],
                                               gpu::CAMERA_BINDING,
                                               0,
                                               1);
    lightBuilder.buffer.sets[i] = currentCamBuffer[i].sets[i];
    gpu::ctx__->pDescriptorAllocator->writeUbo(lightBuilder.buffer.bufferh_,
                                               lightBuilder.buffer.size_,
                                               lightBuilder.buffer.sets[i],
                                               gpu::GLOBAL_LIGHT,
                                               0,
                                               1);
    gpu::ctx__->pDescriptorAllocator->update();
  }
}


void ResourceManager::updateResource(uint32_t update)
{
  drawResourceBox();
  drawUploadedMesh();
  camera.update();
  lightBuilder.drawUI();
  currentCamBuffer[update].data_ = &(camera.ubo);
  currentCamBuffer[update].size_ = sizeof(camera.ubo);
  currentCamBuffer[update].uploadData();
  lightBuilder.uploadData();
  drawModelState();
}

void ResourceManager::addModel(gpu::MeshBuffer* meshBuffer,
                               std::string name)
{
  std::unique_ptr<Model> model = std::make_unique<Model>();
  model->name = name;
  model->mesh = meshBuffer;
  cpu::ctx__->drawHandle_.push_back(model.get());
  models_[name] = std::move(model);
}

void ResourceManager::uploadMesh(std::string path)
{
  std::string modelPath = path;
  spdlog::info("Loading {} ", modelPath.c_str());
  auto tempMesh = importer_.loadModel(modelPath.c_str());
  gpu::ctx__->pResourceAllocator->buildMeshNode(tempMesh.get());
  tempMesh->nodeName_ = path;
  meshes_[path] = std::move(tempMesh);
}

void ResourceManager::uploadTexture(std::string path)
{
  std::unique_ptr<gpu::VkTexture> texture = std::make_unique<gpu::VkTexture>();
  texture->type_ = gpu::ResourceType::TEXTURE;
  texture->filepath__ = path;
  auto ptr = texture.get();
  gpu::ctx__->pResourceAllocator->buildTexture(texture.get());
  texture->descriptorSet__ = gpu::ctx__->pDescriptorAllocator->descriptorSets;
  gpu::ctx__->pDescriptorAllocator->uploadBindlessTextureSet(texture.get());
  gpu::ctx__->pDescriptorAllocator->update();
  textures_[path] = std::move(texture);
}


void ResourceManager::drawResourceBox()
{
  {
    ImVec2 dispSize = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos(ImVec2(dispSize.x - dispSize.x / 9, 0));
    ImGui::SetNextWindowSize(ImVec2(dispSize.x / 9, dispSize.y / 6 * 5));
    if (ImGui::Begin("ASSET BOX",
                     nullptr))
    {
      ImGui::Text("MODEL FOLDER : ");
      ImGui::BeginChild("Model folder");
      {
        for (const auto& entry : std::filesystem::directory_iterator(ASSET_MODELS_DIR))
        {
          if (entry.is_regular_file())
          {
            if (ImGui::Button(entry.path().filename().string().c_str()))
            {
              uploadMesh(std::string(ASSET_MODELS_DIR) + entry.path().filename().string());
            }
          }
        }
      }
      ImGui::Separator();
      ImGui::Text("TEXTURE FOLDER : ");
      ImGui::BeginChild("Texture folder");
      {
        {
          for (const auto& entry : std::filesystem::directory_iterator(ASSET_TEXTURES_DIR))
          {
            if (entry.is_regular_file())
            {
              if (ImGui::Button(entry.path().filename().string().c_str()))
              {
                uploadTexture(entry.path().filename().string());
              }
            }
          }
          ImGui::EndChild();
        }
      }
      ImGui::EndChild();
    }
  }
  ImGui::End();
}

void ResourceManager::drawModelState()
{
  for (auto& model : this->models_)
  {
    auto m = model.second.get();
    m->drawUIState();
    if (m->uiState)
    {
      this->camera.selectModel = m;
    }
  }
}

void ResourceManager::drawUploadedMesh()
{
  if (ImGui::Begin("Mesh edditor",
                   nullptr))
  {
    ImGui::Text("uploaded Mesh: ");
    for (auto& mesh : meshes_)
    {
      if (ImGui::Button(mesh.first.c_str()))
      {
        addModel(mesh.second.get(), mesh.first);
      }
    }
  }

  ImGui::End();
}
