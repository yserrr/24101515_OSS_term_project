#include "resource_pool.hpp"
#include "camera_cfg.hpp"
#include "descriptor_layout_config.hpp"

ResourceManager::ResourceManager(const ResourceManagerCreateInfo &info) :
  device_(info.device),
  allocator_(*info.allocator),
  maxInflight_(info.maxInflight),
  descriptorUploader_(info.device)
{
  vkGetPhysicalDeviceProperties(allocator_.getPhysicalDevice(), &physicalDeviceProperties);
  textures_.resize(1024);
  uiNeedTextures.reserve(1024);
  minUboAlign_    = physicalDeviceProperties.limits.minUniformBufferOffsetAlignment;
  samplerBuilder_ = std::make_unique<SamplerBuilder>(device_);

  fragShaderPool                 = std::make_unique<ShaderPool>(device_);
  vertexShaderPool               = std::make_unique<ShaderPool>(device_);
  pipelinePool_                  = std::make_unique<PipelinePool>(device_);
  std::unique_ptr<Material> base = std::make_unique<Material>();
  base->name                     = "base";
  materials_[base->name]         = std::move(base);

  CamCI cam_ci{};
  cam_ci.fov         = glm::radians(350.0f);
  cam_ci.aspectRatio = 2400 / 1200;
  cam_ci.nearPlane   = 1;
  cam_ci.farPlane    = 256.0;
  camera             = std::make_unique<EditorCam>(cam_ci);

  descriptorManager   = std::make_unique<DescriptorManager>(allocator_.getDevice());
  bindlessDescirptor_ = descriptorManager->bindlessSet();

  uboBuilder_ = std::make_unique<UBOBuilder>(allocator_, BufferType::UNIFORM, AccessPolicy::HostPreferred);
  auto lightBuf = uboBuilder_->buildBuffer(sizeof(lightUBO));
  lightBuilder.bufferContext = lightBuf;

  Light light;
  light.direction = glm::vec3(0, 0, 0);
  light.position = {3.0, 3.0, 3.0};
  light.type      = LightType::Point;
  light.intensity = 0.5;
  light.color     = glm::vec3(0.3, 0.2, 0.5);
  lightBuilder.build(light);

  lightBuilder.bufferContext.buffer->loadData(&lightBuilder.ubo, sizeof(lightBuilder.ubo));
  mainCamBuffers_.resize(info.maxInflight);
  for (int i = 0; i < maxInflight_; i++)
  {
    VkDescriptorSet set              = descriptorManager->allocateSet();
    mainCamBuffers_[i]               = uboBuilder_->buildBuffer(sizeof(camera->ubo));
    mainCamBuffers_[i].descriptorSet = set;
    mainCamBuffers_[i].bindingIndex  = gpu::CAMERA_BINDING;

    descriptorUploader_.UploadUboSet(*mainCamBuffers_[i].buffer->getBuffer(),
                                     sizeof(camera->ubo),
                                     set,
                                     gpu::CAMERA_BINDING);

    descriptorUploader_.UploadUboSet(*lightBuf.buffer->getBuffer(),
                                     sizeof(lightUBO),
                                     set,
                                     gpu::LOCAL_LIGHT_BINDING);
  }
}

ResourceManager::~ResourceManager() = default;