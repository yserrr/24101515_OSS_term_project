//
// Created by ljh on 25. 9. 17..
//

#include "renderer.hpp"

RenderingSystem::RenderingSystem(RenderInitInfo info) : device_h(info.device_h),
                                                        renderpass_h(info.renderPass),
                                                        framebufferManager(info.frameManager),
                                                        swapchain(info.swapchain),
                                                        imageManager(info.imageManager),
                                                        allocator(*info.allocator),
                                                        pResourceManager(*info.resourceManager),
                                                        pDescriptorSetLayouts(info.pDescriptorSetLayouts),
                                                        descriptorLayoutCount(info.descriptorSetLayoutCount)
{
  vkCmdSetPolygonModeEXT = (PFN_vkCmdSetPolygonModeEXT) vkGetDeviceProcAddr(device_h, "vkCmdSetPolygonModeEXT");
  if (!vkCmdSetPolygonModeEXT)
  {
    throw std::runtime_error("vkCmdSetPolygonModeEXT not available!");
  }
  VkShaderModule vShader = pResourceManager.fragShaderPool->getShader(vertPath, shaderc_vertex_shader);
  VkShaderModule fshader = pResourceManager.fragShaderPool->getShader(fragPath, shaderc_fragment_shader);

  pipelineLayout_h = pResourceManager.pipelinePool_->createPipelineLayout(pDescriptorSetLayouts, descriptorLayoutCount);
  PipelineProgram program{};
  VkFormat swapchainfFormat      = (swapchain->getFormat());
  program.pColorAttachmentFormat = &swapchainfFormat;
  program.renderPass             = renderpass_h;
  program.pipelineLayout         = pipelineLayout_h;
  program.vertexType             = VertexType::ALL;
  program.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  program.vertShaderModule       = vShader;
  program.fragShaderModule       = fshader;
  pipeline_h                     = pResourceManager.pipelinePool_->createPipeline(program);

  VkShaderModule vBackground = pResourceManager.vertexShaderPool->getShader(VertBackPath, shaderc_vertex_shader);
  VkShaderModule fBackground = pResourceManager.fragShaderPool->getShader(fragBackPath, shaderc_fragment_shader);

  program.vertShaderModule = vBackground;
  program.fragShaderModule = fBackground;
  program.vertexType       = VertexType::BACKGROUND;
  backgroundPipeline_      = pResourceManager.pipelinePool_->createPipeline(program);
  drawBackground           = VK_TRUE;
}

void RenderingSystem::pushConstant(VkCommandBuffer command)
{
  vkCmdPushConstants(
                     command,
                     pipelineLayout_h,
                     VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                     0,
                     sizeof(MaterialConstant),
                     &pResourceManager.selectedModel.constant
                    );
  //spdlog::trace("albedo index: {} ",pResourceManager.selectedModel.constant.albedoTextureIndex) ;
  //spdlog::trace("metalic index: {} ",pResourceManager.selectedModel.constant.metalicTextureIndex) ;
  //spdlog::trace("normal index: {} ",pResourceManager.selectedModel.constant.normalTextureIndex) ;
}
