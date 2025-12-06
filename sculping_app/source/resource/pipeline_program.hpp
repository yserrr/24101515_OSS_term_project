#ifndef MYPROJECT_PIPELINE_CFG_HPP
#define MYPROJECT_PIPELINE_CFG_HPP

#include <cstdint>
#include <string>
#include <vulkan/vulkan.h>
#include "vertex.hpp"

struct PipelineProgram{
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkRenderPass renderPass         = VK_NULL_HANDLE;
  VkShaderModule vertShaderModule = VK_NULL_HANDLE;
  VkShaderModule fragShaderModule = VK_NULL_HANDLE;
  VkFormat *pColorAttachmentFormat = nullptr;
  VertexType vertexType           = VertexType::ALL;
  VkPrimitiveTopology topology    = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkCullModeFlags cullMode        = VK_CULL_MODE_BACK_BIT;
  VkFrontFace frontFace         = VK_FRONT_FACE_CLOCKWISE;
  bool operator==(const PipelineProgram &other) const
  {
    return pipelineLayout == other.pipelineLayout &&
           vertShaderModule == other.vertShaderModule &&
           fragShaderModule == other.fragShaderModule &&
           topology == other.topology;
  }
};

struct PipelineHash{
  std::size_t operator()(const PipelineProgram &program) const
  {
    return std::hash<VkShaderModule>()(program.vertShaderModule) ^
           (std::hash<VkShaderModule>()(program.fragShaderModule) << 1);
  }
};

constexpr uint32_t COLOR_FLAG_ALL = VK_COLOR_COMPONENT_R_BIT |
                                    VK_COLOR_COMPONENT_G_BIT |
                                    VK_COLOR_COMPONENT_B_BIT |
                                    VK_COLOR_COMPONENT_A_BIT;

#endif //MYPROJECT_PIPELINE_CFG_HPP