#include "common.hpp"
#ifndef COMMANDPOOL_HPP
#define COMMANDPOOL_HPP

struct CommandPoolCreateInfo{
  VkDevice device;
  uint32_t queueFamilyIndex;
  VkCommandPoolCreateFlags flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
};

class CommandPool{
public:
  CommandPool(CommandPoolCreateInfo info);
  ~CommandPool();
  VkCommandPool get() const;
  void reset();

private:
  VkDevice device;
  VkCommandPool commandPool;
};

struct CommandManagerConfig{
  uint32_t primaryReservePerFrame               = 1;
  VkCommandBufferUsageFlags defaultPrimaryBegin = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  bool enableSecondary                          = false;

  uint32_t secondaryThreads                   = 0;
  uint32_t secondaryQueueFamilyIndex          = 0;
  VkCommandPoolCreateFlags secondaryPoolFlags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                                                VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

  VkCommandBufferUsageFlags defaultSecondaryBegin = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;

  bool ownSingleTimePool                       = false;
  uint32_t singleTimeQueueFamilyIndex          = 0;
  VkCommandPoolCreateFlags singleTimePoolFlags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                                                 VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
};

struct FrameBucket{
  std::vector<VkCommandBuffer> primaries;
  uint32_t cursor           = 0;
  VkCommandBuffer lastBegun = VK_NULL_HANDLE;
};

struct SecondaryBucket{
  VkCommandPool pool = VK_NULL_HANDLE;
  std::vector<VkCommandBuffer> cache;
  std::unique_ptr<std::mutex> mtx;
};

#endif