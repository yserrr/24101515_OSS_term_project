#ifndef PIPELINE_POOL_HPP
#define PIPELINE_POOL_HPP
#include <stdexcept>
#include <vector>
#include <common.hpp>
#include <vertex.hpp>
#include <unordered_map>
#include "pipeline_program.hpp"
#include <fstream>

///@param flags -> reserved field, but vk1.3 -> don't use, just 0
class PipelinePool{
public:
  PipelinePool(VkDevice device);
  ~PipelinePool();
  VkPipeline createPipeline(PipelineProgram program);
  VkPipeline getPipeline(PipelineProgram program) const;
  VkPipelineLayout createPipelineLayout(VkDescriptorSetLayout *descriptorLayoutData,
                                        uint32_t layoutCount);
  void createComputePipeline(const std::vector<VkDescriptorSetLayout> &descriptorSetLayouts,
                             VkShaderModule computeShader);

private:
  void buildFragmentPipeline(VkPipelineShaderStageCreateInfo &shaderStateCi,
                             VkShaderModule fragModule);

  void buildVertexPipeline(VkPipelineShaderStageCreateInfo &shaderStateCi,
                           VkShaderModule vertexModule);

  void buildVertexDescriptor(VertexType type,
                             VkVertexInputBindingDescription &vertexBindingDesc,
                             std::vector<VkVertexInputAttributeDescription> &vertexAttributeDescriptions,
                             VkPipelineVertexInputStateCreateInfo &vertexInputInfo,
                             uint32_t vertexBinding = 0);

  void buildDepthStencilPipeline(VkPipelineDepthStencilStateCreateInfo &depthStencilCi,
                                 VkBool32 depthTestEnable   = VK_TRUE,
                                 VkBool32 depthWriteEnable  = VK_TRUE,
                                 VkBool32 stencilTestEnable = VK_FALSE,
                                 VkCompareOp depthCompareOp =
                                 VK_COMPARE_OP_LESS_OR_EQUAL);

  void buildAssemblyPipeline(VkPipelineInputAssemblyStateCreateInfo &inputAssembly,
                             VkPrimitiveTopology topology                  = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                             VkBool32 primitiveRestartEnable               = VK_FALSE,
                             VkPipelineInputAssemblyStateCreateFlags flags = 0);

  void buildDynamicPipelineDyscriptor(VkPipelineDynamicStateCreateInfo &dynamicStateCi,
                                      std::vector<VkDynamicState> &dynamicStates,
                                      VkPipelineViewportStateCreateInfo &viewportStateCi,
                                      uint32_t viewCount                      = 1,
                                      VkBool32 dynamicStencilTestEnable       = VK_FALSE,
                                      VkBool32 dynamicStateDepthCompare       = VK_FALSE,
                                      VkBool32 dynamicStateVetexStride        = VK_FALSE,
                                      VkPipelineDynamicStateCreateFlags flags = 0);

  void buildRasterizationPipeline(VkPipelineRasterizationStateCreateInfo &rasterizeCi,
                                  VkCullModeFlags cullMode                      = VK_CULL_MODE_BACK_BIT,
                                  VkPolygonMode mode                            = VK_POLYGON_MODE_FILL,
                                  VkFrontFace front                             = VK_FRONT_FACE_CLOCKWISE,
                                  VkPipelineRasterizationStateCreateFlags flags = 0);

  void buildMultiSamplingPipeline(VkPipelineMultisampleStateCreateInfo &multiSamplingCi,
                                  VkSampleCountFlagBits samples               = VK_SAMPLE_COUNT_1_BIT,
                                  VkPipelineMultisampleStateCreateFlags flags = 0);

  void buildColorBlendingAttachment(VkPipelineColorBlendAttachmentState &colorBlendAttachment,
                                    uint32_t             = COLOR_FLAG_ALL,
                                    VkBool32 blendEnable = VK_FALSE);

  void buildColorBlendingPipeline(VkPipelineColorBlendStateCreateInfo &colorBlendingCi,
                                  VkPipelineColorBlendAttachmentState &attachment,
                                  uint32_t attachmentCount = 1);

  void buildDynamicRenderingPipeline(VkPipelineRenderingCreateInfo &dynamicRendering,
                                     VkFormat *colorAttachmentFormats,
                                     uint32_t colorAttachmentCount = 1,
                                     uint32_t viewMask             = 0,
                                     VkFormat depthFormat          = VK_FORMAT_D32_SFLOAT,
                                     VkFormat stencilAttachment    = VK_FORMAT_UNDEFINED
    );
  VkPipeline createPipeline(VertexType type,
                            VkShaderModule vertexModule,
                            VkShaderModule fragModule,
                            VkFormat *colorAttachmentFormats,
                            VkRenderPass renderPass,
                            VkPipelineLayout pipelineLayout,
                            uint32_t colorAttachmentCount = 1,
                            uint32_t viewMask             = 0,
                            VkFormat depthFormat          = VK_FORMAT_D32_SFLOAT,
                            VkFormat stencilAttachment    = VK_FORMAT_UNDEFINED,
                            VkPrimitiveTopology topology  = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                            VkCullModeFlags cullMode      = VK_CULL_MODE_NONE,
                            VkBool32 depthTestEnable      = VK_TRUE,
                            VkBool32 depthWriteEnable     = VK_TRUE,
                            VkBool32 stencilTestEnable    = VK_FALSE,
                            VkCompareOp depthCompareOp    = VK_COMPARE_OP_LESS_OR_EQUAL);

private:
  VkDevice device_;
  VkRenderPass renderPass_;
  VkPipelineLayout computePipelineLayout_;
  VkPipelineCache oldPipelineCache_;

  std::unordered_map<PipelineProgram, VkPipeline, PipelineHash> pipelineHash_{};
  std::vector<uint8_t> loadPipelineCache(const std::string &filename);
};

#endif