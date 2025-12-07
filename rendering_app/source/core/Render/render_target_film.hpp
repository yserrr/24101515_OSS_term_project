//
// Created by dlwog on 25. 10. 28..
//

#ifndef MYPROJECT_RENDER_TRAGET_FILM_HPP
#define MYPROJECT_RENDER_TRAGET_FILM_HPP
#include <vector>
#include <unique.hpp>
#include "gpu_context.hpp"

class RenderPassBuilder;

namespace flm
{
  class RenderTragetFilm
  {
    public:
    void buildFrameResources(uint32_t index);
    void updateFrameConstant();
    void clearGbuffer();
    void clearDepthBuffer();
    void clearShadowBuffer();
    void clearLightBuffer();
    void clearBloomingBuffer();
    std::unique_ptr<gpu::FrameAttachment> swapchainAttachment_;
    std::unique_ptr<gpu::FrameAttachment> depthAttachmentHandle_;
    std::unique_ptr<gpu::FrameAttachment> gBufferAlbedoHandle_;
    std::unique_ptr<gpu::FrameAttachment> gBufferNormalHandle_;
    std::unique_ptr<gpu::FrameAttachment> gBufferPositionHandle_;
    std::unique_ptr<gpu::FrameAttachment> gBufferRoughnessHandle_;
    std::unique_ptr<gpu::FrameAttachment> gBufferMetalicHandle_;
    std::unique_ptr<gpu::FrameAttachment> lightningAttachmentHandle_;
    std::unique_ptr<gpu::FrameAttachment> shadowAttachmentHandleXp_;
    std::unique_ptr<gpu::FrameAttachment> shadowAttachmentHandleXm_;
    std::unique_ptr<gpu::FrameAttachment> shadowAttachmentHandleYp_;
    std::unique_ptr<gpu::FrameAttachment> shadowAttachmentHandleYm_;
    std::unique_ptr<gpu::FrameAttachment> shadowAttachmentHandleZp_;
    std::unique_ptr<gpu::FrameAttachment> shadowAttachmentHandleZm_;
    std::unique_ptr<gpu::FrameAttachment> bloomingExtractAttachment_;
    std::unique_ptr<gpu::FrameAttachment> bloomingBlurAttachment_;
    std::unique_ptr<gpu::FrameAttachment> toneMappingAttachment_;
    std::unique_ptr<gpu::FrameAttachment> gammaAttachment_;

    struct FrameAttachment
    {
      int gBufferPositon = 0;
      int gBufferNormal = 0;
      int gBufferAlbedo = 0;
      int gBufferRoughness = 0;

      int DepthBuffer = 0;
      int bloomingExtract = 0;
      int lightningBuffer = 0;
      int shadowXp = -1;

      int shadowXm = -1;
      int shadowYp = -1;
      int shadowYm = -1;
      int shadowZp = -1;

      int shadowZm = -1;
      int bloomingBlur = -1;
      int tonemapping = -1;
      int gamma = -1;

      int cubeTex = -1;
    } renderAttachment;
  };
}


#endif //MYPROJECT_RENDER_TRAGET_FILM_HPP
