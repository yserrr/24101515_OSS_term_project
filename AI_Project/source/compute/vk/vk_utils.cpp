#include "vk_util.h"

uint32_t get_subgroup_size(const std::string& pipeline_name, const vk_device_architecture& arch)
{
  for (const auto& config : gpu_pipeline_configs)
  {
    if (config.arch == arch)
    {
      auto pipIt = config.pipelines.find(pipeline_name);
      if (pipIt != config.pipelines.end())
      {
        return pipIt->second;
      }
      std::vector<std::pair<std::string, uint32_t>> sorted_pipelines(config.pipelines.begin(), config.pipelines.end());
      std::sort(sorted_pipelines.begin(), sorted_pipelines.end(),
                [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
      for (const auto& entry : sorted_pipelines)
      {
        if (pipeline_name.find(entry.first) != std::string::npos)
        {
          return entry.second;
        }
      }
      return config.default_subgroup_size;
    }
  }
  return 0; // If no matching configuration is found
}

uint32_t get_fa_num_small_rows(FaCodePath path)
{
  if (path == FA_COOPMAT2)
  {
    return flash_attention_num_small_rows;
  }
  else
  {
    return scalar_flash_attention_num_small_rows;
  }
}
std::array<uint32_t, 2> fa_rows_cols(FaCodePath path, uint32_t hsk, uint32_t hsv, uint32_t clamp, v_data_type type,
                                            bool small_rows)
{
  v_UNUSED(clamp);
  v_UNUSED(hsv);

  if (path == FA_SCALAR)
  {
    if (small_rows)
    {
      return {scalar_flash_attention_num_small_rows, 64};
    }
    else
    {
      if ((hsv | hsk) & 8)
      {
        // HSV/HSK not being a multiple of 16 makes D_split smaller, which makes cols_per_iter
        // larger, and Bc needs to be >= cols_per_thread. 64 is large enough, 32 is not.
        return {get_fa_scalar_num_large_rows(hsv), 64};
      }
      else
      {
        return {get_fa_scalar_num_large_rows(hsv), 32};
      }
    }
  }

  if (path == FA_COOPMAT1)
  {
    if (small_rows)
    {
      return {scalar_flash_attention_num_small_rows, scalar_flash_attention_Bc};
    }
    else
    {
      return {coopmat1_flash_attention_num_large_rows, scalar_flash_attention_Bc};
    }
  }

  // small rows, large cols
  if (small_rows)
  {
    return {get_fa_num_small_rows(FA_COOPMAT2), 32};
  }

  // small cols to reduce register count
  if (v_is_quantized(type) || hsk >= 256 || hsv >= 256)
  {
    if (hsk >= 512 || hsv >= 512)
    {
      return {32, 32};
    }
    else
    {
      return {64, 32};
    }
  }
  return {64, 64};
}


vk_pipeline v_vk_get_to_fp16(vk_backend_ctx* ctx, v_data_type type)
{
  VK_LOG_DEBUG("v_vk_get_to_fp16()");
  switch (type)
  {
  case v_TYPE_F32:
  case v_TYPE_Q4_0:
  case v_TYPE_Q4_1:
  case v_TYPE_Q5_0:
  case v_TYPE_Q5_1:
  case v_TYPE_Q8_0:
  case v_TYPE_Q2_K:
  case v_TYPE_Q3_K:
  case v_TYPE_Q4_K:
  case v_TYPE_Q5_K:
  case v_TYPE_Q6_K:
  case v_TYPE_IQ1_S:
  case v_TYPE_IQ1_M:
  case v_TYPE_IQ2_XXS:
  case v_TYPE_IQ2_XS:
  case v_TYPE_IQ2_S:
  case v_TYPE_IQ3_XXS:
  case v_TYPE_IQ3_S:
  case v_TYPE_IQ4_XS:
  case v_TYPE_IQ4_NL:
  case v_TYPE_MXFP4:
    break;
  default:
    return nullptr;
  }

  return ctx->device->pipeline_dequant[type];
}
uint32_t get_fa_scalar_num_large_rows(uint32_t hsv)
{
  if (hsv >= 192)
  {
    return 2;
  }
  else
  {
    return 8;
  }
}

uint32_t find_properties(const vk::PhysicalDeviceMemoryProperties* mem_props, vk::MemoryRequirements* mem_req,
                               vk::MemoryPropertyFlags flags)
{
  for (uint32_t i = 0; i < mem_props->memoryTypeCount; ++i)
  {
    vk::MemoryType memory_type = mem_props->memoryTypes[i];
    if ((mem_req->memoryTypeBits & ((uint64_t)1 << i)) &&
        (flags & memory_type.propertyFlags) == flags &&
        mem_props->memoryHeaps[memory_type.heapIndex].size >= mem_req->size)
    {
      return static_cast<int32_t>(i);
    }
  }
  return UINT32_MAX;
}
void vk_sync_buffers(vk_backend_ctx* ctx, vk_context& subctx)
{
  VK_LOG_DEBUG("v_vk_sync_buffers()");

  const bool transfer_queue = subctx->p->q->transfer_only;

  if (ctx)
  {
    ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_split_k_need_sync = false;
  }

  subctx->s->buffer.pipelineBarrier(
    subctx->p->q->stage_flags,
    subctx->p->q->stage_flags,
    {},
    {
      {
        {
          !transfer_queue
            ? (vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferRead |
               vk::AccessFlagBits::eTransferWrite)
            : (vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite)
        },
        {
          !transfer_queue
            ? (vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferRead |
               vk::AccessFlagBits::eTransferWrite)
            : (vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite)
        }
      }
    },
    {},
    {}
  );
}
