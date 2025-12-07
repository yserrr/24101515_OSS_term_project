#include <filesystem>
#include <vulkan/vulkan.h>
#include <sys/stat.h>
#include <algorithm>
#include <unordered_set>
#include <deque>
#include "vk_context.hpp"
#include "vk_discard_pool.hpp"
#include "vk_resource_allocator.hpp"
#include "vk_graph_builder.hpp"
#include "context.hpp"


namespace gpu
{
  extern VkContext* ctx__;
  // expect base layout
  //: ex (g-buffer -> shadow -> lighting) declare order
  //  tracking last writer -> if need barrier, insert
  //
  //
  //todo : if need,
  //       insert write->reader dependency
  VkGraphBuilder::VkGraphBuilder(VkContext* pCtxt) :
    pCtxt(pCtxt), semaphorePool_(pCtxt)
  {
  }

  void VkGraphBuilder::deffered()
  {
    ///todo:
    /// 1.pass: write->reader depenency check
    ///  2.row sort and find prologue pass, epilogue pass
    ///  3.for first prologue pass to epilogue pass,
    ///   4.find not usable pass and cull
    ///   5.pass cull and merge pass with async compute pass,
    ///     make wave front pass
    ///   6.frame resource memory friend setting
    /// ->topological sort
    /// ->compile pass
    //std::vector<VkPass*>& passes = pCtxt->uploadedPass;
    // for (auto* pass : passes):
    //write->reader dependency
    //for (gpu::VkPass* pass : passes)
    //{
    //  for (auto& write__ : pass->write__)
    //  {
    //    maskingTimeline(write__);
    //    for (auto& reader__ : write__->reader__)
    //    {
    //      if (pass != reader__)
    //      {
    //        //W->R
    //        reader__->dependency__.insert(pass);
    //        pass->dependent__.insert(reader__);
    //      }
    //    }
    //    //W->W
    //  }
    /// std::unordered_set<VkResource*> firstWrite
    /// std::unordered_set<VkResource*> lastProducted
    /// std::vector<VkPass*> passStack ;
    /// /// pass-> multi queue and product cover
    /// for(auto last : lastProducted):
    ///   for (auto writer: last->write):
    ///    RECURSIVE_FIND_WRITER()
    ///     if(not find)->cullpass
    ///
    /// this time: pass is culled and row was sorted
    /// std::unordered_set<VkResource*> useResource
    /// std::unordered_map<memType*, VkFriendMemory*> memFriend
    /// for(auto pass: passStack) :
    ///   ( auto write: pass->write  )
    ///   if (memFriend.find(write->memType))
    ///    -> write->memFriend =
    ///   useResource->insert(write)
    ///   write->lastwriter = pass
    ///   write->timelineMasking = masking
    ///   memFriend[write]  = memFriend
    ///
    ///for(pass: passStack)
    ///  for(read: pass->read)
    ///     collectBarrier(read)
    ///  for(write: pass->write
    ///     collectBarrier(write)
    ///  pass->write,read, sync,pass Write RenderPassType setting
    ///  compiledPass.pushBack()
    ///   std::deque<VkPass*> ready;
    ///
    ///
    ///
    ///
    /// for (auto* pass : pCtxt->uploadedPass)
    /// {
    ///   pass->linkCount = pass->dependency__.size();
    ///   if (pass->dependency__.size() == 0)
    ///   {
    ///     pass->dependency__.clear();
    ///     ready.push_back(pass);
    ///   }
    /// }
    /// while (!ready.empty())
    /// {
    ///   VkPass* pass = ready.front();
    ///   ready.pop_front();
    ///   compilePass(pass);
    ///   for (auto* postPass : pass->dependent__)
    ///   {
    ///     postPass->linkCount--;
    ///     if (postPass->linkCount == 0)
    ///     {
    ///       ready.push_back(postPass);
    ///     }
    ///   }
    ///   pass->dependent__.clear();
    /// }
    ///
    ///
    ///
    ///
    ///
    ///VkDependencyInfo
    /// depencency= >collected barriers register
    ///VkPass pass
    /// pass.type = dependencyPass
    /// pass.lambda= buildDependencyPass ;
    /// compiledPass.push_back(dependencyPass)
    ///VkPass pass. presentpass pushback
    //}
  }

  VkDeviceSize alignUp(VkDeviceSize v, VkDeviceSize a)
  {
    return (v + (a - 1)) & ~(a - 1);
  }

  VkHostBuffer VkGraphBuilder::buildHostBuffer(VkDeviceSize size, BufferType bufferType)
  {
    VkBufferUsageFlags flags;
    switch (bufferType)
    {
      case BufferType::VERTEX:
        flags = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
          VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        break;
      case BufferType::INDEX:
        flags = VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
          VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        break;
      case BufferType::UNIFORM:
        flags = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        size = alignUp(size,
                       ctx__->deviceProperties__.limits.minUniformBufferOffsetAlignment);
        break;
      case BufferType::STORAGE:
        flags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
          VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        size = alignUp(size,
                       ctx__->deviceProperties__.limits.minStorageBufferOffsetAlignment);
        break;
      case BufferType::STAGE:
        flags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
          VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }

    VkHostBuffer hostBuffer;
    hostBuffer.size_ = size;
    hostBuffer.usage__ = flags;
    hostBuffer.nodeId_ = ctx__->nodeId_;
    hostBuffer.bufferh_ = ctx__->pResourceAllocator->buildBufferHandle(size, flags);
    hostBuffer.allocation__ = ctx__->pResourceAllocator->mBindBuffer(hostBuffer.bufferh_,
                                                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    hostBuffer.allocated__ = true;
    return hostBuffer;
  }

  std::vector<VkPass> VkGraphBuilder::build(std::vector<VkPass*>& passes,
                                            uint32_t frameIndex)
  {
    compiledPasses_.clear();
    for (auto* pass : passes)
    {
      for (auto* read__ : pass->read__)
      {
        read__->currentAccessMask__ = VK_ACCESS_NONE;
        read__->currentPipeline__ = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        if (read__->lastWriter__ != nullptr && pass != read__->lastWriter__)
        {
          pass->dependency__.insert(read__->lastWriter__);
          read__->lastWriter__->dependent__.insert(pass);
        }
      }
      for (auto* write__ : pass->write__)
      {
        if (write__->lastWriter__ != nullptr && pass != write__->lastWriter__)
        {
          pass->dependency__.insert(write__->lastWriter__);
          write__->lastWriter__->dependent__.insert(pass);
        }
        maskingTimeline(write__);
        write__->lastWriter__ = pass;
      }
    }
    std::deque<VkPass*> ready;
    for (auto* pass : passes)
    {
      pass->linkCount = pass->dependency__.size();
      if (pass->dependency__.size() == 0)
      {
        pass->dependency__.clear();
        ready.push_back(pass);
      }
    }
    while (!ready.empty())
    {
      VkPass* pass = ready.front();
      ready.pop_front();
      compilePass(pass);
      for (auto* postPass : pass->dependent__)
      {
        postPass->linkCount--;
        if (postPass->linkCount == 0)
        {
          ready.push_back(postPass);
        }
      }
      pass->dependent__.clear();
    }
    //std::vector<VkResource*> frameResources;
    //for (auto& node : frameNodes_)
    //{
    //  frameResources.push_back(node);
    //}

    VkPass barrierPass;
    barrierPass.passType = VkRenderPassType::BARRIER_PASS;
    auto* swapchain = pCtxt->pSwapChainContext->swapchainAttachment__[frameIndex];
    barrierPass.execute = buildImageBarrier(swapchain->currentAccessMask__,
                                            0,
                                            swapchain->currentPipeline__,
                                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                            swapchain->currentLayout__,
                                            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                            pCtxt->graphicsFamailyIdx__,
                                            pCtxt->graphicsFamailyIdx__,
                                            swapchain);
    swapchain->currentLayout__ = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    swapchain->currentAccessMask__ = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    swapchain->currentPipeline__ = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    compiledPasses_.push_back(barrierPass);
    //pCtxt->pDiscardPool->registerResource(frameResources);
    return compiledPasses_;
  }

  ///resource usage-> allocate
  /// memory bind(triming -> allocate)
  void VkGraphBuilder::allocate(VkResource* _read)
  {
    if (!_read->allocated__)
    {
      if (_read->type_ == gpu::ResourceType::IMAGE)
      {
        VkFrameAttachment* frameImage = reinterpret_cast<gpu::VkFrameAttachment*>(_read);
        frameImage->usage__ = getResourceUsage(frameImage->usage_);
        pCtxt->pResourceAllocator->buildFrameAttachment(frameImage);
      }
      if (_read->type_ == ResourceType::BUFFER)
      {
        VkHostBuffer* frameBuffer = reinterpret_cast<gpu::VkHostBuffer*>(_read);
        frameBuffer->usage__ = getResourceUsage(frameBuffer->usage_);
        frameBuffer->bufferh_ = pCtxt->pResourceAllocator->buildBufferHandle(frameBuffer->size_,
                                                                             frameBuffer->usage__);
      }
      if (_read->type_ == ResourceType::MESH)
      {
        VkMeshBuffer* mesh = reinterpret_cast<gpu::VkMeshBuffer*>(_read);
        pCtxt->pResourceAllocator->buildMeshNode(mesh);
      }
      if (_read->type_ == ResourceType::TEXTURE)
      {
        VkTexture* frameImage = reinterpret_cast<gpu::VkTexture*>(_read);
        pCtxt->pResourceAllocator->buildTexture(frameImage);
      }
    }
    _read->allocated__ = true;
  }


  void VkGraphBuilder::maskingTimeline(VkResource* node)
  {
    if (node->type_ == gpu::ResourceType::IMAGE)
    {
      VkFrameAttachment* frameImage = reinterpret_cast<gpu::VkFrameAttachment*>(node);
      if (frameImage->usage_ & (gpu::ResourceUsage::G_BUFFER |
        gpu::ResourceUsage::LIGHTNING_BUFFER))
      {
        frameImage->writeAccessMask__ = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        frameImage->writePipeline__ = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        frameImage->writeLayout__ = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      }
      if (frameImage->usage_ & (gpu::ResourceUsage::DEPTH_STENCIL_ATTACHMENT |
        gpu::ResourceUsage::SHADOW_BUFFER))
      {
        frameImage->writeAccessMask__ = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        frameImage->writePipeline__ = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
          VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        frameImage->writeLayout__ = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      }
      if (frameImage->usage_ & gpu::ResourceUsage::TEXTURE)
      {
        frameImage->writeAccessMask__ = VK_ACCESS_TRANSFER_WRITE_BIT;
        frameImage->writePipeline__ = VK_PIPELINE_STAGE_TRANSFER_BIT;
        frameImage->writeLayout__ = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      }
    }
    if (node->type_ == gpu::ResourceType::BUFFER)
    {
      if (node->usage_ & (gpu::ResourceUsage::MESH_BUFFER |
        gpu::ResourceUsage::VERTEX_BUFFER))
      {
        VkMeshBuffer* buffer = reinterpret_cast<gpu::VkMeshBuffer*>(node);
        buffer->writeAccessMask__ = VK_ACCESS_TRANSFER_WRITE_BIT;
        buffer->writePipeline__ = VK_PIPELINE_STAGE_TRANSFER_BIT;
      }
      if (node->usage_ & (gpu::ResourceUsage::UNIFORM_BUFFER |
        gpu::ResourceUsage::SHADER_STORAGE_BUFFER))
      {
        //todo:
        //imple with shader
      }
    }
  }

  VkImageLayout gpu::VkGraphBuilder::decideReadImageLayout(gpu::ResourceUsage usage)
  {
    uint32_t usage_ = static_cast<uint32_t>(usage);
    if (usage & (gpu::ResourceUsage::G_BUFFER |
      gpu::ResourceUsage::LIGHTNING_BUFFER |
      gpu::ResourceUsage::FORWARD |
      gpu::ResourceUsage::POST_PROCESS))
    {
      return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    if (usage & (gpu::ResourceUsage::DEPTH_STENCIL_ATTACHMENT |
      gpu::ResourceUsage::SHADOW_BUFFER))
    {
      return VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL;
    }
    if (usage & (gpu::ResourceUsage::TRANSFER |
      gpu::ResourceUsage::TEXTURE))
    {
      return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    }
    throw std::runtime_error("unsupported resource usage");
  }

  uint32_t gpu::VkGraphBuilder::getResourceUsage(gpu::ResourceUsage usage)
  {
    uint32_t usage_ = static_cast<uint32_t>(usage);
    if (usage_ & (gpu::ResourceUsage::MESH_BUFFER |
      gpu::ResourceUsage::VERTEX_BUFFER))
    {
      return static_cast<VkBufferUsageFlags>(
        static_cast<uint32_t>(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) |
        static_cast<uint32_t>(VK_BUFFER_USAGE_INDEX_BUFFER_BIT) |
        static_cast<uint32_t>(VK_BUFFER_USAGE_TRANSFER_DST_BIT)
      );
    }
    if (usage_ & (gpu::ResourceUsage::DEPTH_STENCIL_ATTACHMENT |
      gpu::ResourceUsage::SHADOW_BUFFER))
    {
      return VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT;
    }
    if (usage_ & (gpu::ResourceUsage::G_BUFFER |
      ResourceUsage::POST_PROCESS |
      ResourceUsage::FORWARD |
      ResourceUsage::LIGHTNING_BUFFER))
    {
      return VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT;
    }
    if (usage_ & (gpu::ResourceUsage::UNIFORM_BUFFER))
    {
      return VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }
    if (usage_ & (gpu::ResourceUsage::SHADER_STORAGE_BUFFER))
    {
      return VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }
    if (usage_ & (gpu::ResourceUsage::TEXTURE))
    {
      return VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT;
    }
    throw std::runtime_error("unsupported resource usage");
  }

  uint32_t VkGraphBuilder::decideReadPipeline(gpu::ResourceUsage readUsage)
  {
    if (readUsage & (gpu::ResourceUsage::MESH_BUFFER |
      gpu::ResourceUsage::VERTEX_BUFFER))
    {
      return VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
    }
    if (readUsage & (gpu::ResourceUsage::DEPTH_STENCIL_ATTACHMENT |
      gpu::ResourceUsage::SHADOW_BUFFER))
    {
      return VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    if (readUsage & (gpu::ResourceUsage::G_BUFFER |
      ResourceUsage::POST_PROCESS |
      ResourceUsage::FORWARD |
      ResourceUsage::LIGHTNING_BUFFER))
    {
      return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }

    if (readUsage & (gpu::ResourceUsage::UNIFORM_BUFFER |
      gpu::ResourceUsage::SHADER_STORAGE_BUFFER))
    {
      return VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    if (readUsage & (gpu::ResourceUsage::TEXTURE))
    {
      return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    throw std::runtime_error("unsupported resource usage");
  }


  uint32_t gpu::VkGraphBuilder::decideReadAccessMask(gpu::ResourceUsage readUsage)
  {
    if (readUsage & (gpu::ResourceUsage::MESH_BUFFER |
      gpu::ResourceUsage::VERTEX_BUFFER))
    {
      return VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    }
    if (readUsage & (gpu::ResourceUsage::DEPTH_STENCIL_ATTACHMENT |
      gpu::ResourceUsage::SHADOW_BUFFER))
    {
      return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    }
    if (readUsage & (gpu::ResourceUsage::LIGHTNING_BUFFER |
      gpu::ResourceUsage::G_BUFFER |
      gpu::ResourceUsage::TEXTURE |
      gpu::ResourceUsage::POST_PROCESS))
    {
      return VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    }

    if (readUsage & (gpu::ResourceUsage::MESH_BUFFER |
      gpu::ResourceUsage::VERTEX_BUFFER))
    {
      return VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    }

    throw std::runtime_error("unsupported resource usage");
  }

  void gpu::VkGraphBuilder::compilePass(VkPass* renderPass)
  {
    for (auto* _read : renderPass->read__)
    {
      allocate(_read);
      if (_read->type_ & (gpu::ResourceType::IMAGE | gpu::ResourceType::TEXTURE))
      {
        gpu::VkFrameAttachment* readImage = reinterpret_cast<gpu::VkFrameAttachment*>(_read);
        readSync(readImage);
        insertResolve(readImage);
        if (readImage->descriptorSet__[pCtxt->renderingContext.currentFrame__] == VK_NULL_HANDLE)
        {
          readImage->descriptorSet__ = pCtxt->pDescriptorAllocator->descriptorSets;
          pCtxt->pDescriptorAllocator->uploadBindlessTextureSet(readImage);
          pCtxt->pDescriptorAllocator->update();
        }
      }
      if (_read->type_ == gpu::ResourceType::BUFFER)
      {
        VkHostBuffer* buffer = reinterpret_cast<VkHostBuffer*>(_read);
        readSync(buffer);
      }
    }
    for (auto* write : renderPass->write__)
    {
      if ((write->allocated__) == false)
      {
        allocate(write);
      }
      if (write->type_ & (gpu::ResourceType::IMAGE))
      {
        gpu::VkFrameAttachment* frameImage = reinterpret_cast<gpu::VkFrameAttachment*>(write);
        writeSync(frameImage);
        if (frameImage->usage_ & (gpu::ResourceUsage::G_BUFFER |
          gpu::ResourceUsage::LIGHTNING_BUFFER))
        {
          if (!renderPass->passParameter__.writen__)
          {
            VkRenderingAttachmentInfo colorAttachment{
              .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
              .imageView = frameImage->imageView__,
              .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
              .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
              .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            };
            colorAttachment.clearValue.color = renderPass->passParameter__.clearColor__;
            if (frameImage->writen__)
            {
              colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            }
            frameImage->writen__ = true;
            renderPass->passParameter__.colorAttachment__.push_back(colorAttachment);
          }
        }

        if (frameImage->usage_ & (gpu::ResourceUsage::DEPTH_STENCIL_ATTACHMENT |
          gpu::ResourceUsage::SHADOW_BUFFER))
        {
          if (!renderPass->passParameter__.writen__)
          {
            VkRenderingAttachmentInfo depthAttachment{};
            depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depthAttachment.imageView = frameImage->imageView__;
            depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL_KHR;
            depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depthAttachment.clearValue.color = renderPass->passParameter__.clearColor__;
            depthAttachment.clearValue.depthStencil = {1.0f, 0};
            renderPass->passParameter__.depthAttachment__ = depthAttachment;
          }
        }
        if (write->type_ == gpu::ResourceType::BUFFER)
        {
          VkHostBuffer* buffer = reinterpret_cast<VkHostBuffer*>(write);
          writeSync(buffer);
        }
      }
    }

    if (renderPass->execute != nullptr)
    {
      compiledPasses_.push_back(*renderPass);
    }
  }


  void VkGraphBuilder::readSync(VkFrameAttachment* image)
  {
    VkImageLayout readLayout = decideReadImageLayout(image->usage_);
    VkAccessFlags readAccessMask = decideReadAccessMask(image->usage_);
    VkPipelineStageFlags readPipeline = decideReadPipeline(image->usage_);

    if (image->currentLayout__ != readLayout ||
      image->currentAccessMask__ != readAccessMask ||
      image->currentPipeline__ != readPipeline)
    {
      VkPass barrierPass;
      barrierPass.passType = VkRenderPassType::BARRIER_PASS;
      barrierPass.execute = buildImageBarrier(image->currentAccessMask__,
                                              readAccessMask,
                                              image->currentPipeline__,
                                              readPipeline,
                                              image->currentLayout__,
                                              readLayout,
                                              pCtxt->graphicsFamailyIdx__,
                                              pCtxt->graphicsFamailyIdx__,
                                              image);

      image->currentLayout__ = readLayout;
      image->currentAccessMask__ = readAccessMask;
      image->currentPipeline__ = readPipeline;
      compiledPasses_.push_back(barrierPass);
      spdlog::trace("barrier pass insert");
    }
  }


  void VkGraphBuilder::writeSync(VkFrameAttachment* image)
  {
    if ((image->currentLayout__ != image->writeLayout__) ||
      (image->currentAccessMask__ != image->writeAccessMask__) ||
      (image->currentPipeline__ != image->writePipeline__) ||
      image->lastWriter__ != nullptr) //who write-> sync
    {
      VkPass barrierPass;
      barrierPass.passType = VkRenderPassType::BARRIER_PASS;
      barrierPass.execute = buildImageBarrier(image->currentAccessMask__,
                                              image->writeAccessMask__,
                                              image->currentPipeline__,
                                              image->writePipeline__,
                                              image->currentLayout__,
                                              image->writeLayout__,
                                              pCtxt->graphicsFamailyIdx__,
                                              pCtxt->graphicsFamailyIdx__,
                                              image);
      image->currentLayout__ = image->writeLayout__;
      image->currentAccessMask__ = image->writeAccessMask__;
      image->currentPipeline__ = image->writePipeline__;
      compiledPasses_.push_back(barrierPass);
    }
  }

  void VkGraphBuilder::writeSync(VkHostBuffer* fBuffer)
  {
  }

  void VkGraphBuilder::readSync(VkHostBuffer* fBuffer)
  {
  }

  void VkGraphBuilder::insertResolve(VkFrameAttachment* image)
  {
  }


  std::function<void(VkCommandBuffer cmd)> VkGraphBuilder::buildBufferBarrier(VkAccessFlags srcAccessMask,
                                                                              VkAccessFlags dstAccessMask,
                                                                              VkPipelineStageFlags srcStageMask,
                                                                              VkPipelineStageFlags dstStageMask,
                                                                              VkHostBuffer& frameBuffer,
                                                                              uint32_t srcQFamily,
                                                                              uint32_t dstQFamily,
                                                                              VkHostBuffer* buffer)
  {
    std::function<void(VkCommandBuffer cmd)> lambda;
    lambda = [&buffer,
        srcAccessMask,
        dstAccessMask,
        srcStageMask,
        dstStageMask,
        srcQFamily,
        dstQFamily](VkCommandBuffer cmd)
      {
        spdlog::debug(" resource dst StageMast {}", static_cast<uint32_t>(dstStageMask));
        spdlog::debug(" resource dst AccessMask {}", static_cast<uint32_t>(dstAccessMask));
        VkDeviceSize offsets = 0;
        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = srcAccessMask;
        barrier.dstAccessMask = dstAccessMask;
        barrier.srcQueueFamilyIndex = srcQFamily;
        barrier.dstQueueFamilyIndex = dstQFamily;
        barrier.buffer = buffer->bufferh_;
        barrier.size = buffer->size_;
        barrier.offset = offsets;
        vkCmdPipelineBarrier(
                             cmd,
                             srcStageMask,
                             dstStageMask,
                             0,
                             0,
                             nullptr,
                             1,
                             &barrier,
                             0,
                             nullptr
                            );
      };
    return lambda;
  }

  std::function<void(VkCommandBuffer cmd)> VkGraphBuilder::buildImageBarrier(VkAccessFlags srcAccessMask,
                                                                             VkAccessFlags dstAccessMask,
                                                                             VkPipelineStageFlags srcStageMask,
                                                                             VkPipelineStageFlags dstStageMask,
                                                                             VkImageLayout srcImageLayout,
                                                                             VkImageLayout dstImageLayout,
                                                                             uint32_t srcQFamily,
                                                                             uint32_t dstQFamily,
                                                                             gpu::VkFrameAttachment* srcImage)
  {
    std::function<void(VkCommandBuffer cmd)> lambda;

    lambda = [srcImage,
        srcAccessMask,
        dstAccessMask,
        srcImageLayout,
        dstImageLayout,
        srcStageMask,
        dstStageMask,
        srcQFamily,
        dstQFamily](VkCommandBuffer cmd)
      {
        spdlog::debug(" resource usage {}", static_cast<uint32_t>(dstImageLayout));
        spdlog::debug(" resource usage {}", static_cast<uint32_t>(dstStageMask));
        spdlog::debug(" resource usage {}", static_cast<uint32_t>(dstImageLayout));
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = srcImageLayout;
        barrier.newLayout = dstImageLayout;
        barrier.srcQueueFamilyIndex = srcQFamily;
        barrier.dstQueueFamilyIndex = dstQFamily;
        barrier.image = srcImage->imageh__;
        barrier.subresourceRange.aspectMask = srcImage->aspectMask__;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = srcImage->levelCount__;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.srcAccessMask = srcAccessMask;
        barrier.dstAccessMask = dstAccessMask;
        vkCmdPipelineBarrier(
                             cmd,
                             srcStageMask,
                             dstStageMask,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &barrier
                            );
      };
    return lambda;
  }

  std::function<void(VkCommandBuffer cmd)> VkGraphBuilder::buildBufferCopyToImage(gpu::VkHostBuffer* srcBuffer,
    gpu::VkFrameAttachment* dstImage)
  {
    std::function<void(VkCommandBuffer cmd)> lambda;
    lambda = [srcBuffer,
        dstImage](VkCommandBuffer cmd)
      {
        spdlog::debug(" resource usage {}", static_cast<uint32_t>(dstImage->usage_));

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = dstImage->aspectMask__;
        region.imageSubresource.mipLevel = dstImage->mipLevels__;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = dstImage->levelCount__;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {
          dstImage->width__,
          dstImage->height__,
          1
        };
        spdlog::debug("call to command buffer to trenslate to texture");
        vkCmdCopyBufferToImage(
                               cmd,
                               srcBuffer->bufferh_,
                               dstImage->imageh__,
                               dstImage->writeLayout__,
                               1,
                               &region
                              );
      };
    return lambda;
  }

  std::function<void(VkCommandBuffer cmd)> VkGraphBuilder::buildBufferCopyToBuffer(gpu::VkHostBuffer* srcBuffer,
    gpu::VkHostBuffer* dstBuffer)
  {
    std::function<void(VkCommandBuffer cmd)> lambda;
    lambda = [&srcBuffer,
        &dstBuffer](VkCommandBuffer cmd)
      {
        spdlog::debug(" resource usage {}", static_cast<uint32_t>(dstBuffer->usage_));
        VkDeviceSize offsets = 0;
        VkBufferCopy bufferCopy{};
        bufferCopy.srcOffset = 0;
        bufferCopy.dstOffset = 0;
        bufferCopy.size = dstBuffer->size_;
        vkCmdCopyBuffer(cmd,
                        srcBuffer->bufferh_,
                        dstBuffer->bufferh_,
                        1,
                        &bufferCopy);
      };
    return lambda;
  }


  std::unique_ptr<VkFrameAttachment> VkGraphBuilder::buildSwapchainAttachment(int index)
  {
    std::unique_ptr<VkFrameAttachment> swapchainImage = std::make_unique<VkFrameAttachment>();
    swapchainImage->allocated__ = true;
    swapchainImage->format__ = ctx__->pSwapChainContext->imgFormat__;
    swapchainImage->imageh__ = ctx__->pSwapChainContext->img__[index];
    swapchainImage->imageView__ = ctx__->pSwapChainContext->imgView__[index];
    swapchainImage->levelCount__ = 1;
    swapchainImage->type_ = ResourceType::IMAGE;
    swapchainImage->usage_ = ResourceUsage::G_BUFFER;
    swapchainImage->lifetime = VkResourceLifetime::PERSISTENT;
    swapchainImage->aspectMask__ = VK_IMAGE_ASPECT_COLOR_BIT;
    swapchainImage->lastWriter__ = nullptr;
    swapchainImage->height__ = ctx__->pSwapChainContext->extent__.height;
    swapchainImage->width__ = ctx__->pSwapChainContext->extent__.width;
    swapchainImage->nodeId_ = ctx__->nodeId_++;
    swapchainImage->currentLayout__ = VK_IMAGE_LAYOUT_UNDEFINED;
    swapchainImage->currentPipeline__ = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    swapchainImage->currentAccessMask__ = VK_ACCESS_NONE;
    ctx__->pSwapChainContext->swapchainAttachment__[index] = swapchainImage.get();
    return std::move(swapchainImage);
  }

  std::unique_ptr<gpu::VkFrameAttachment> VkGraphBuilder::buildDepthAttachment()
  {
    std::unique_ptr<gpu::VkFrameAttachment> depth = std::make_unique<gpu::VkFrameAttachment>();
    depth->type_ = gpu::ResourceType::IMAGE;
    depth->usage_ = gpu::ResourceUsage::DEPTH_STENCIL_ATTACHMENT;
    depth->aspectMask__ = VK_IMAGE_ASPECT_DEPTH_BIT;
    depth->format__ = VK_FORMAT_D32_SFLOAT;
    depth->height__ = gpu::ctx__->pSwapChainContext->extent__.height;
    depth->width__ = gpu::ctx__->pSwapChainContext->extent__.width;
    depth->mSpace_ = gpu::MemorySpace::DEVICE_LOCAL;
    depth->lifetime = gpu::VkResourceLifetime::FRAME;
    depth->mipLevels__ = 1;
    depth->levelCount__ = 1;
    return std::move(depth);
  }

  std::unique_ptr<gpu::VkFrameAttachment> VkGraphBuilder::buildFrameAttachment(uint32_t format)
  {
    mns::uptr<gpu::VkFrameAttachment> gBuffer = mns::mUptr<gpu::VkFrameAttachment>();
    gBuffer->type_ = gpu::ResourceType::IMAGE;
    gBuffer->usage_ = gpu::ResourceUsage::G_BUFFER;
    gBuffer->aspectMask__ = VK_IMAGE_ASPECT_COLOR_BIT;
    gBuffer->format__ = static_cast<VkFormat>(format);
    gBuffer->height__ = gpu::ctx__->pSwapChainContext->extent__.height;
    gBuffer->width__ = gpu::ctx__->pSwapChainContext->extent__.width;
    gBuffer->mSpace_ = gpu::MemorySpace::DEVICE_LOCAL;
    gBuffer->lifetime = gpu::VkResourceLifetime::FRAME;
    gBuffer->mipLevels__ = 1;
    gBuffer->levelCount__ = 1;
    return std::move(gBuffer);
  }
}
