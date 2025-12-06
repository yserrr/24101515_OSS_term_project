#include <memory>
#include "vk_context.h"
#include "vk_device.h"
#include "vk_queue.h"

void vk_begin_ctx(vk_device& device, vk_context& subctx)
{
  VK_LOG_DEBUG("v_vk_ctx_begin(" << device->name << ")");
  if (subctx->s != nullptr)
  {
    vk_ctx_end(subctx);
  }

  subctx->seqs.push_back({mmlVKBeginSubMission(device, *subctx->p)});
  subctx->s = subctx->seqs[subctx->seqs.size() - 1].data();
}

void vk_ctx_end(vk_context& ctx)
{
  VK_LOG_DEBUG("v_vk_ctx_end(" << ctx << ", " << ctx->seqs.size() << ")");
  if (ctx->s == nullptr)
  {
    return;
  }

  ctx->s->buffer.end();
  ctx->s = nullptr;
}

vk_context vk_create_temp_ctx(MmlCommandPool& p)
{
  vk_context result = std::make_shared<vk_context_struct>();
  VK_LOG_DEBUG("v_vk_create_temporary_context(" << result << ")");
  result->p = &p;
  return result;
}


