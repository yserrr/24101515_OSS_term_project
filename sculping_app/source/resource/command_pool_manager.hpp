#ifndef COMMANDBUFFER_HPP
#define COMMANDBUFFER_HPP
#include "common.hpp"

//commandBuffer에서 command list 전체 관리 및 렌더링 작업 진행
struct CommandPoolManagerCreateInfo{
  VkDevice device;
  VkCommandPool commandPool;
  uint32_t frameCount;
};

class CommandPoolManager{
public:
  CommandPoolManager(const CommandPoolManagerCreateInfo &info);

  ~CommandPoolManager();

  VkCommandBuffer record(uint32_t imageIndex);

  void endRecord(uint32_t imageIndex);

private:
  VkDevice device;
  VkCommandPool commandPool;
  std::vector<VkCommandBuffer> commandBuffers;
};

#endif