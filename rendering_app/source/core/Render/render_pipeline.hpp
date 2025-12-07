#ifndef MYPROJECT_RENDER_PIPELINE_HPP
#define MYPROJECT_RENDER_PIPELINE_HPP
#include "gpu_context.hpp"
class EventManager;
class RenderPassBuilder;

class Pipeline
{
  friend class ::RenderPassBuilder;
  friend class ::EventManager;

  public:
  void buildPipeline();
  void bindPipeline(gpu::CommandBuffer cmd);
  gpu::Pipeline depthRenderingPipeline__;
  gpu::Pipeline offScreenTexture;
  gpu::Pipeline gBufferWritePipeline__;
  gpu::Pipeline depthWritePipeline__;
  gpu::Pipeline shadowWritePipeline__;
  gpu::Pipeline lightningWritePipeline__;
  gpu::Pipeline bloomingExtractWritePipeline__;
  gpu::Pipeline bloomingBlurWritePipeline__;
  gpu::Pipeline tonemappingWritePipeline__;
  gpu::Pipeline gammaWritePipeline__;
  gpu::PipelineLayout pipelineLayout_h;

  std::string DEPTH_WRITE =
    "C:/Users/dlwog/OneDrive/Desktop/VkMain-out/source/shader/depth_write.frag";
  std::string G_BUFFER_WRITE =
    "C:/Users/dlwog/OneDrive/Desktop/VkMain-out/source/shader/def_geo_write.frag";
  std::string LIGHT_WRITE =
    "C:/Users/dlwog/OneDrive/Desktop/VkMain-out/source/shader/def_lightning_write.frag";

  std::string BLOOMING_EXTRACT_WRITE =
    "C:/Users/dlwog/OneDrive/Desktop/VkMain-out/source/shader/post_blooming_extract.frag";
  std::string BLOOMING_BLUR_WRITE =
    "C:/Users/dlwog/OneDrive/Desktop/VkMain-out/source/shader/post_blooming_blur.frag";

  std::string TONE_MAPPING =
    "C:/Users/dlwog/OneDrive/Desktop/VkMain-out/source/shader/post_tone_mapping.frag";
  std::string GAMMA =
    "C:/Users/dlwog/OneDrive/Desktop/VkMain-out/source/shader/post_gamma.frag";


  std::string FRAG_OFF_SCREEN_RENDER =
    "C:/Users/dlwog/OneDrive/Desktop/VkMain-out/source/shader/off_screen_render.frag";

  std::string FRAG_DEPTH_RENDER =
    "C:/Users/dlwog/OneDrive/Desktop/VkMain-out/source/shader/def_depthRender.frag";
  std::string MESH_VERTEX =
    "C:/Users/dlwog/OneDrive/Desktop/VkMain-out/source/shader/vertex.vert";
  std::string VERTEX_QUAD =
    "C:/Users/dlwog/OneDrive/Desktop/VkMain-out/source/shader/quad.vert";
  PFN_vkCmdSetPolygonModeEXT cmdSetPolygonMode;
  VkPolygonMode polygonMode = VK_POLYGON_MODE_FILL;
  VkBool32 depthTest = VK_TRUE;
  VkBool32 drawBackground = VK_TRUE;
};


#endif //MYPROJECT_RENDER_PIPELINE_HPP
