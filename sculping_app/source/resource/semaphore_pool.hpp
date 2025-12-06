#ifndef SEMAPHORE_HPP
#define SEMAPHORE_HPP
#include <common.hpp>

struct SignalCreateInfo{
  VkDevice device;
  uint32_t MAX_FRAMES_IN_FLIGHT;
};

class SemaphorePool{
public:
  SemaphorePool(SignalCreateInfo &info);
  ~SemaphorePool();

  VkSemaphore get(uint32_t currentFrame) const;
  void recreate();

private:
  VkDevice device;
  uint32_t MAX_FRAMES_IN_FLIGHT;
  std::vector<VkSemaphore> semaphores;
};
#endif //semaphore_hpp