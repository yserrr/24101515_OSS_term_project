#include<resource_pool.hpp>
#include <filesystem>

VulkanTexture *ResourceManager::getTextures(bindingIndex index)
{
  if (currentBinding_ > index)
  {
    return textures_[index].get();
  }
  return nullptr;
}

void ResourceManager::setLight() {}

Camera *ResourceManager::getCamera()
{
  return camera.get();
}

void ResourceManager::uploadMesh(VkCommandBuffer command, std::string path)
{
  std::string modelPath = std::string(ASSET_MODELS_DIR) + path;
  spdlog::info("Loading {} ", modelPath.c_str());
  Mesh tempMesh              = importer_.loadModel(modelPath.c_str(), allocator_);
  tempMesh.name              = path;
  std::unique_ptr<Mesh> mesh = std::make_unique<Mesh>(tempMesh.getVertices(), tempMesh.getIndices(), allocator_);

  mesh->copyBuffer(command);
  mesh->selected = true;
  if (selectedModel.mesh != nullptr)
  {
    selectedModel.mesh->selected = false;
    if (mesh->name == selectedModel.mesh->name)
    {
      mesh->name = selectedModel.mesh->name + "t";
    }
  }
  selectedModel.mesh     = mesh.get();
  selectedModel.material = materials_["base"].get();
  meshes_[path]          = std::move(mesh);
}

void ResourceManager::uploadTexture(VkCommandBuffer command, std::string path)
{
  TextureCreateInfo textureInfo;
  textureInfo.device      = device_;
  textureInfo.sampler     = samplerBuilder_->get();
  std::string texturePath = std::string(ASSET_TEXTURES_DIR) + path;
  textureInfo.filename    = texturePath.c_str();
  textureInfo.allocator   = &allocator_;
  auto texture            = std::make_unique<VulkanTexture>(textureInfo);
  std::filesystem::path assetPath(path);
  if (assetPath.extension() == ".ktx")
  {
//    texture->loadKtx (command);
  }
  else  {
    texture->loadImage(command);
  }
  if (!texture->uploadedReady)
  {
    spdlog::info("fail to load image");
    return;
  }
  texture->bindigIndex = currentBinding_;
  texture->copyBufferToImage(command);
  texture->descriptor = bindlessDescirptor_;
  texture->uploadDescriptor(bindlessDescirptor_, currentBinding_);
  uiNeedTextures.push_back(texture.get());

  textures_[currentBinding_] = std::move(texture);
  currentBinding_++;
}

void ResourceManager::updateDescriptorSet(uint32_t currentFrame)
{
  mainCamBuffers_[currentFrame].buffer->loadData(&(camera->ubo), sizeof(camera->ubo));
}