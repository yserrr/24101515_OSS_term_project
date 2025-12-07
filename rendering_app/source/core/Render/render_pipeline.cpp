#include "render_pipeline.hpp"
#include "cast.hpp"

void Pipeline::buildPipeline()
{
  cmdSetPolygonMode =
    (PFN_vkCmdSetPolygonModeEXT)::vkGetDeviceProcAddr(gpu::ctx__->deviceh__, "vkCmdSetPolygonModeEXT");
  if (!cmdSetPolygonMode)
  {
    throw std::runtime_error("vkCmdSetPolygonModeEXT not available!");
  }
  pipelineLayout_h = gpu::ctx__->pPipelinePool->createPipelineLayout(&gpu::ctx__->pDescriptorAllocator->defaultLayout,
                                                                     1);
  std::uintptr_t vShader = gpu::iShd__->getShader(MESH_VERTEX, shaderc_vertex_shader);
  std::uintptr_t depthOnlyWrite = gpu::iShd__->getShader(DEPTH_WRITE, shaderc_fragment_shader);

  std::uintptr_t vQuad = gpu::iShd__->getShader(VERTEX_QUAD, shaderc_vertex_shader);

  std::uintptr_t gBufferWrite_ = gpu::iShd__->getShader(G_BUFFER_WRITE,
                                                        shaderc_fragment_shader);

  std::uintptr_t lightnintWrite = gpu::iShd__->getShader(LIGHT_WRITE,
                                                         shaderc_fragment_shader);

  std::uintptr_t bloomingExtractWrite = gpu::iShd__->getShader(BLOOMING_EXTRACT_WRITE,
                                                               shaderc_fragment_shader);


  std::uintptr_t bloomingBlurWrite = gpu::iShd__->getShader(BLOOMING_BLUR_WRITE,
                                                            shaderc_fragment_shader);

  std::uintptr_t toneMapping = gpu::iShd__->getShader(TONE_MAPPING,
                                                      shaderc_fragment_shader);

  std::uintptr_t gamma = gpu::iShd__->getShader(GAMMA,
                                                shaderc_fragment_shader);

  std::uintptr_t depthRender = gpu::iShd__->getShader(FRAG_DEPTH_RENDER,
                                                      shaderc_fragment_shader);

  std::uintptr_t offscreenRender = gpu::iShd__->getShader(FRAG_OFF_SCREEN_RENDER,
                                                          shaderc_fragment_shader);


  gpu::PipelineProgram program{};
  {
    program.renderingType = gpu::RenderingAttachmentType::DEPTH;
    program.pipelineLayout = pipelineLayout_h;
    program.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    program.vertShaderModule = cast<VkShaderModule>(vShader);
    program.fragShaderModule = cast<VkShaderModule>(depthOnlyWrite);
    program.vertexType = gpu::VertexType::ALL;
    depthWritePipeline__ = gpu::ctx__->pPipelinePool->createPipeline(program);
  }
  {
    //todo: shadow write vertex shader setting
    program.renderingType = gpu::RenderingAttachmentType::DEPTH;
    program.pipelineLayout = pipelineLayout_h;
    program.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    program.vertShaderModule = cast<VkShaderModule>(vShader);
    program.fragShaderModule = cast<VkShaderModule>(depthOnlyWrite);
    program.vertexType = gpu::VertexType::ALL;
    shadowWritePipeline__ = gpu::ctx__->pPipelinePool->createPipeline(program);
  }
  {
    program.renderingType = gpu::RenderingAttachmentType::G_BUFFER;
    program.pipelineLayout = pipelineLayout_h;
    program.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    program.vertShaderModule = cast<VkShaderModule>(vShader);
    program.fragShaderModule = cast<VkShaderModule>(gBufferWrite_);
    program.vertexType = gpu::VertexType::ALL;
    program.renderingType = gpu::RenderingAttachmentType::G_BUFFER;
    gBufferWritePipeline__ = gpu::ctx__->pPipelinePool->createPipeline(program);
  }

  {
    program.renderingType = gpu::RenderingAttachmentType::LIGHTNING;
    program.pipelineLayout = pipelineLayout_h;
    program.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    program.vertShaderModule = cast<VkShaderModule>(vQuad);
    program.fragShaderModule = cast<VkShaderModule>(lightnintWrite);
    program.vertexType = gpu::VertexType::BACKGROUND;
    lightningWritePipeline__ = gpu::ctx__->pPipelinePool->createPipeline(program);
  }


  {
    program.renderingType = gpu::RenderingAttachmentType::BLOOMING;
    program.pipelineLayout = pipelineLayout_h;
    program.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    program.vertShaderModule = cast<VkShaderModule>(vQuad);
    program.fragShaderModule = cast<VkShaderModule>(bloomingExtractWrite);
    program.vertexType = gpu::VertexType::BACKGROUND;
    bloomingExtractWritePipeline__ = gpu::ctx__->pPipelinePool->createPipeline(program);
  }

  {
    program.renderingType = gpu::RenderingAttachmentType::BLOOMING;
    program.pipelineLayout = pipelineLayout_h;
    program.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    program.vertShaderModule = cast<VkShaderModule>(vQuad);
    program.fragShaderModule = cast<VkShaderModule>(bloomingBlurWrite);
    program.vertexType = gpu::VertexType::BACKGROUND;
    bloomingBlurWritePipeline__ = gpu::ctx__->pPipelinePool->createPipeline(program);
  }
  {
    program.renderingType = gpu::RenderingAttachmentType::TONEMAP;
    program.pipelineLayout = pipelineLayout_h;
    program.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    program.vertShaderModule = cast<VkShaderModule>(vQuad);
    program.fragShaderModule = cast<VkShaderModule>(toneMapping);
    program.vertexType = gpu::VertexType::BACKGROUND;
    tonemappingWritePipeline__ = gpu::ctx__->pPipelinePool->createPipeline(program);
  }

  {
    program.renderingType = gpu::RenderingAttachmentType::GAMMA_CORRECTION;
    program.pipelineLayout = pipelineLayout_h;
    program.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    program.vertShaderModule = cast<VkShaderModule>(vQuad);
    program.fragShaderModule = cast<VkShaderModule>(gamma);
    program.vertexType = gpu::VertexType::BACKGROUND;
    gammaWritePipeline__ = gpu::ctx__->pPipelinePool->createPipeline(program);
  }


  {
    program.renderingType = gpu::RenderingAttachmentType::SWAPCHAIN;
    program.pipelineLayout = pipelineLayout_h;
    program.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    program.vertShaderModule = cast<VkShaderModule>(vQuad);
    program.fragShaderModule = cast<VkShaderModule>(depthRender);
    program.vertexType = gpu::VertexType::BACKGROUND;
    depthRenderingPipeline__ = gpu::ctx__->pPipelinePool->createPipeline(program);
  }
  {
    program.renderingType = gpu::RenderingAttachmentType::SWAPCHAIN;
    program.pipelineLayout = pipelineLayout_h;
    program.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    program.vertShaderModule = cast<VkShaderModule>(vQuad);
    program.fragShaderModule = cast<VkShaderModule>(offscreenRender);
    program.vertexType = gpu::VertexType::BACKGROUND;
    program.renderingType = gpu::RenderingAttachmentType::SWAPCHAIN;
    offScreenTexture = gpu::ctx__->pPipelinePool->createPipeline(program);
  }

  drawBackground = VK_TRUE;
}

void Pipeline::bindPipeline(gpu::CommandBuffer cmd)
{
  cmdSetPolygonMode(cmd, polygonMode);
  vkCmdSetDepthTestEnable(cmd, depthTest);
}
