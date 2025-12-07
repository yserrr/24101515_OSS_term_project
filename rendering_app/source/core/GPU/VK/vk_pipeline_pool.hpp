#ifndef VK_PIPELINE_POOL_HPP
#define VK_PIPELINE_POOL_HPP

#include <vulkan/vulkan.h>
#include <unordered_map>
#include <stdexcept>
#include <fstream>
#include <cstdint>
#include <vector>
#include <string>
#include "vk_common.hpp"
#include "vk_vertex_attr.hpp"

namespace gpu
{
  class VkContext;
  constexpr uint32_t COLOR_FLAG_ALL = VK_COLOR_COMPONENT_R_BIT |
    VK_COLOR_COMPONENT_G_BIT |
    VK_COLOR_COMPONENT_B_BIT |
    VK_COLOR_COMPONENT_A_BIT;

  struct VkConstant
  {
    glm::mat4 mat1;
    glm::mat4 mat2;
  };

  enum class RenderingAttachmentType
  {
    DEPTH,
    SWAPCHAIN,
    G_BUFFER,
    LIGHTNING,
    BLOOMING,
    TONEMAP,
    GAMMA_CORRECTION,
  };

  class VkPipelineProgram
  {
    public :
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkShaderModule vertShaderModule = VK_NULL_HANDLE;
    VkShaderModule fragShaderModule = VK_NULL_HANDLE;
    RenderingAttachmentType renderingType = RenderingAttachmentType::SWAPCHAIN;
    VertexType vertexType = VertexType::ALL;
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT;
    VkFrontFace frontFace = VK_FRONT_FACE_CLOCKWISE;

    bool operator==(const VkPipelineProgram& other) const
    {
      return pipelineLayout == other.pipelineLayout &&
        vertShaderModule == other.vertShaderModule &&
        fragShaderModule == other.fragShaderModule &&
        topology == other.topology;
    }
  };

  struct PipelineHash
  {
    std::size_t operator()(const VkPipelineProgram& program) const
    {
      return std::hash<VkShaderModule>()(program.vertShaderModule) ^
        (std::hash<VkShaderModule>()(program.fragShaderModule) << 1);
    }
  };


  ///@param flags -> reserved field, but vk1.3 -> don't use, just 0
  class VkPipelinePool
  {
    public:
    VkPipelinePool(VkContext* pCtxt);
    ~VkPipelinePool();
    VkPipeline createPipeline(VkPipelineProgram program);
    VkPipeline getPipeline(VkPipelineProgram program) const;
    VkPipeline buildRTPipeline();
    VkPipelineLayout createPipelineLayout(VkDescriptorSetLayout* descriptorLayoutData,
                                          uint32_t layoutCount);
    void createComputePipeline(const std::vector<VkDescriptorSetLayout>& descriptorSetLayouts,
                               VkShaderModule computeShader);

    private:
    void buildFragmentPipeline(VkPipelineShaderStageCreateInfo& shaderStateCi,
                               VkShaderModule fragModule);
    void buildVertexPipeline(VkPipelineShaderStageCreateInfo& shaderStateCi,
                             VkShaderModule vertexModule);
    void buildVertexDescriptor(VertexType type,
                               VkVertexInputBindingDescription& vertexBindingDesc,
                               std::vector<VkVertexInputAttributeDescription>& vertexAttributeDescriptions,
                               VkPipelineVertexInputStateCreateInfo& vertexInputInfo,
                               uint32_t vertexBinding = 0);
    void buildDepthStencilPipeline(VkPipelineDepthStencilStateCreateInfo& depthStencilCi,
                                   VkBool32 depthTestEnable = VK_TRUE,
                                   VkBool32 depthWriteEnable = VK_TRUE,
                                   VkBool32 stencilTestEnable = VK_FALSE,
                                   VkCompareOp depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL);

    void buildAssemblyPipeline(VkPipelineInputAssemblyStateCreateInfo& inputAssembly,
                               VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                               VkBool32 primitiveRestartEnable = VK_FALSE,
                               VkPipelineInputAssemblyStateCreateFlags flags = 0);

    void buildDynamicPipelineDyscriptor(VkPipelineDynamicStateCreateInfo& dynamicStateCi,
                                        std::vector<VkDynamicState>& dynamicStates,
                                        VkPipelineViewportStateCreateInfo& viewportStateCi,
                                        uint32_t viewCount = 1,
                                        VkBool32 dynamicStencilTestEnable = VK_FALSE,
                                        VkBool32 dynamicStateDepthCompare = VK_FALSE,
                                        VkBool32 dynamicStateVetexStride = VK_FALSE,
                                        VkPipelineDynamicStateCreateFlags flags = 0);

    void buildRasterizationPipeline(VkPipelineRasterizationStateCreateInfo& rasterizeCi,
                                    VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT,
                                    VkPolygonMode mode = VK_POLYGON_MODE_FILL,
                                    VkFrontFace front = VK_FRONT_FACE_CLOCKWISE,
                                    VkPipelineRasterizationStateCreateFlags flags = 0);

    void buildMultiSamplingPipeline(VkPipelineMultisampleStateCreateInfo& multiSamplingCi,
                                    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT,
                                    VkPipelineMultisampleStateCreateFlags flags = 0);


    VkPipelineColorBlendAttachmentState buildColorBlendingAttachment(
      uint32_t = COLOR_FLAG_ALL,
      VkBool32 blendEnable = VK_FALSE);

    void buildColorBlendingPipeline(VkPipelineColorBlendStateCreateInfo& colorBlendingCi,
                                    VkPipelineColorBlendAttachmentState* attachment,
                                    uint32_t attachmentCount = 1);

    void buildDynamicRenderingPipeline(VkPipelineRenderingCreateInfo& dynamicRendering,
                                       VkFormat* colorAttachmentFormats,
                                       uint32_t colorAttachmentCount = 1,
                                       uint32_t viewMask = 0,
                                       VkFormat depthFormat = VK_FORMAT_D32_SFLOAT,
                                       VkFormat stencilAttachment = VK_FORMAT_UNDEFINED);

    VkPipeline createPipeline(VertexType type,
                              VkShaderModule vertexModule,
                              VkShaderModule fragModule,
                              RenderingAttachmentType attachment,
                              VkPipelineLayout pipelineLayout,
                              uint32_t viewMask = 0,
                              VkFormat depthFormat = VK_FORMAT_D32_SFLOAT,
                              VkFormat stencilAttachment = VK_FORMAT_UNDEFINED,
                              VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                              VkCullModeFlags cullMode = VK_CULL_MODE_NONE,
                              VkBool32 depthTestEnable = VK_TRUE,
                              VkBool32 depthWriteEnable = VK_TRUE,
                              VkBool32 stencilTestEnable = VK_FALSE,
                              VkCompareOp depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL);

    private:
    VkContext* pCtxt_;
    VkDevice device_;
    VkPipelineLayout computePipelineLayout_;
    VkPipelineCache oldPipelineCache_;
    std::unordered_map<VkPipelineProgram, VkPipeline, PipelineHash> pipelineHash_{};
    std::vector<uint8_t> loadPipelineCache(const std::string& filename);
  };
}

#endif
