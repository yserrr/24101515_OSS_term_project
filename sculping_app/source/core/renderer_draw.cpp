#include<renderer.hpp>

void RenderingSystem::setUp(VkCommandBuffer cmd)
{
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_h); // 파이프라인 바인딩
  vkCmdSetPolygonModeEXT(cmd, polygonMode);
  vkCmdSetDepthTestEnable(cmd, depthTest);
  camera->camUpdate();
  viewport.x        = 0.0f;
  viewport.y        = 0.0f;
  viewport.width    = (float) swapchain->getExtent().width;
  viewport.height   = (float) swapchain->getExtent().height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  scissor.offset = {0, 0};
  scissor.extent = swapchain->getExtent();
  vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void RenderingSystem::draw(VkCommandBuffer cmd, uint32_t currenFrame)
{
  if (drawBackground)
  {
    vkCmdSetPolygonModeEXT(cmd, polygonMode);
    vkCmdSetDepthTestEnable(cmd, depthTest);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, backgroundPipeline_);
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = (float) swapchain->getExtent().width;
    viewport.height   = (float) swapchain->getExtent().height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    scissor.offset = {0, 0};
    scissor.extent = swapchain->getExtent();
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdDraw(cmd, 6, 1, 0, 0);
  }

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_h);
  vkCmdSetPolygonModeEXT(cmd, polygonMode);
  vkCmdSetDepthTestEnable(cmd, depthTest);
  switch (viewMode)
  {
    case (ViewMode::SINGLE):
    {
      pushConstant(cmd);
      camera->camUpdate();
      viewport.x        = 0.0f;
      viewport.y        = 0.0f;
      viewport.width    = (float) swapchain->getExtent().width;
      viewport.height   = (float) swapchain->getExtent().height;
      viewport.minDepth = 0.0f;
      viewport.maxDepth = 1.0f;
      vkCmdSetViewport(cmd, 0, 1, &viewport);

      scissor.offset = {0, 0};
      scissor.extent = swapchain->getExtent();
      vkCmdSetScissor(cmd, 0, 1, &scissor);

      for (auto &mesh: pResourceManager.meshes_)
      {
        if (mesh.second == nullptr) return;
        mesh.second->bind(cmd);
        mesh.second->draw(cmd);
      }
      break;
    }
    case (ViewMode::MULTI):
    {
      auto extent      = swapchain->getExtent();
      float halfWidth  = extent.width / 2.0f;
      float halfHeight = extent.height / 2.0f;
      for (uint32_t i = 0; i < 4; i++)
      {
        viewport.width    = halfWidth;
        viewport.height   = halfHeight;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        viewport.x = (i % 2) * halfWidth;
        viewport.y = (i / 2) * halfHeight;
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        pushConstant(cmd);
        scissor.offset = {static_cast<int32_t>(viewport.x), static_cast<int32_t>(viewport.y)};
        scissor.extent = {static_cast<uint32_t>(viewport.width), static_cast<uint32_t>(viewport.height)};
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        for (auto &mesh: pResourceManager.meshes_)
        {
          mesh.second->bind(cmd);
          mesh.second->draw(cmd);
        }
      }
      break;
    }
    case (ViewMode::VR):
    {
      pushConstant(cmd);
      camera->onVR(true);
      pResourceManager.updateDescriptorSet(currenFrame);
      viewport.x        = 0.0f;
      viewport.y        = 0.0f;
      viewport.width    = (float) swapchain->getExtent().width / 2;
      viewport.height   = (float) swapchain->getExtent().height;
      viewport.minDepth = 0.0f;
      viewport.maxDepth = 1.0f;
      vkCmdSetViewport(cmd, 0, 1, &viewport);

      scissor.offset = {0, 0};
      scissor.extent = swapchain->getExtent();
      vkCmdSetScissor(cmd, 0, 1, &scissor);

      for (auto &mesh: pResourceManager.meshes_)
      {
        if (mesh.second == nullptr) return;
        mesh.second->bind(cmd);
        mesh.second->draw(cmd);
      }
      pushConstant(cmd);
      camera->onVR(false);
      pResourceManager.updateDescriptorSet(currenFrame);
      viewport.x        = swapchain->getExtent().width / 2;
      viewport.y        = 0.0f;
      viewport.width    = (float) swapchain->getExtent().width / 2;
      viewport.height   = (float) swapchain->getExtent().height;
      viewport.minDepth = 0.0f;
      viewport.maxDepth = 1.0f;
      vkCmdSetViewport(cmd, 0, 1, &viewport);

      scissor.offset = {0, 0};
      scissor.extent = swapchain->getExtent();
      vkCmdSetScissor(cmd, 0, 1, &scissor);

      for (auto &mesh: pResourceManager.meshes_)
      {
        if (mesh.second == nullptr) return;
        mesh.second->bind(cmd);
        mesh.second->draw(cmd);
      }
      break;
    }
    default:
      break;
  }
}