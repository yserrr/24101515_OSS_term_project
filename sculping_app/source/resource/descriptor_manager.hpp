#include<common.hpp>
#include <../resource/descriptor_pool.hpp>

#include "buffer_builder.hpp"
#ifndef DESCRIPTORPOOLMANAGER_HPP
#define DESCRIPTORPOOLMANAGER_HPP

#define UBO_MAX     100
#define TEXTURE_MAX 1024
#define LIGHT_MAX   128

struct DescriptorContext{
  VkDescriptorSet descriptorSet;
  VkDescriptorSetLayout layout;
  std::vector<VkWriteDescriptorSet> writeDescriptorSets;
  std::vector<VkDescriptorImageInfo> imageInfos;
};

class DescriptorManager{
public:
  DescriptorManager(VkDevice device);
  ~DescriptorManager();
  VkDescriptorSet bindlessSet();
  std::vector<VkDescriptorSetLayout> *getLayouts();
  std::vector<VkDescriptorSet> getSets();
  VkDescriptorSet allocateSet();
  VkDescriptorSetLayout getVert();
  VkDescriptorSetLayout getFrag();
  VkDescriptorSetLayout getLight();

private:
  VkDevice device;
  std::unique_ptr<CameraDescriptorPool> cameraPool;
  std::unique_ptr<TextureDescriptorPool> texturePool;
  std::unique_ptr<LightDescriptorPool> lightPool;
  VkDescriptorSetLayout vertexLayout;  //temp cam
  VkDescriptorSetLayout textureLayout; //temp light ;
  VkDescriptorSetLayout lightLayout;   //both
  std::vector<VkDescriptorSetLayout> layouts;
  std::vector<VkDescriptorSet> descriptors;
  VkDescriptorSet cameraSet;
  VkDescriptorSet textureSet;
  VkDescriptorSet lightSet;
};

#endif //DESCRIPTORMANAGER_HPP