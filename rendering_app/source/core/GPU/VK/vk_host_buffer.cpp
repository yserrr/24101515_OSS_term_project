#include "vk_host_buffer.h"
#include "vk_context.hpp"
extern gpu::VkContext* ctx__;

void gpu::VkHostBuffer::uploadData()
{
  void* bufferData;
  vkMapMemory(ctx__->deviceh__, this->allocation__.memory__,
              this->allocation__.offset__,
              this->allocation__.size, 0, &bufferData);
  memcpy(bufferData, this->data_, (size_t)this->size_);
  vkUnmapMemory(ctx__->deviceh__, this->allocation__.memory__);
}
