#include "vk_resource.hpp"

void gpu::VkMeshBuffer::draw(VkCommandBuffer cmd)
{
  VkDeviceSize offset[] = {0};
  vkCmdBindVertexBuffers(cmd,
                         0,
                         1,
                         &vertexBuffer__,
                         offset);
  vkCmdBindIndexBuffer(cmd,
                       indexBuffer__,
                       0,
                       VK_INDEX_TYPE_UINT32);
  vkCmdDrawIndexed(cmd,
                   indices.size(),
                   1,
                   0,
                   0,
                   0);
}
