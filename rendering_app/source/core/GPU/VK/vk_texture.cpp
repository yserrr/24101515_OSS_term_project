#include "vk_context.hpp"
#include "vk_texture.hpp"
extern gpu::VkContext ctx__;
gpu::VkFrameAttachment::VkFrameAttachment()
{
  descriptorSet__.resize(ctx__->renderingContext.maxInflight__, VK_NULL_HANDLE);
}
