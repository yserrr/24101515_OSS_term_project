#ifndef FENCE_HPP
#define FENCE_HPP
#include <semaphore_pool.hpp>

class FencePool{
public:
  FencePool(SignalCreateInfo info, bool signaled = false);
  ~FencePool();
  VkFence get(uint32_t currentFrame) const;
  void wait(uint32_t currentFrame, uint64_t timeout = UINT64_MAX);
  void reset(uint32_t currentFrame);
  void recreate();

private:
  VkDevice device;
  std::vector<VkFence> fences;
  uint32_t MAX_FRAMES_IN_FLIGHT;
};

#endif