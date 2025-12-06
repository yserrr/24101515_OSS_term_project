#ifndef MEMORYALLOCATOR
#define MEMORYALLOCATOR
#include <memory_pool.hpp>

#ifndef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_TRACE
#define spdlog_trace(...) spdlog::trace(__VA_ARGS__)
#else
#define spdlog_trace(...) (void)0
#endif

#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG
#define spdlog_debug(...) spdlog::debug(__VA_ARGS__)
#else
#define spdlog_debug(...) (void)0
#endif

#define VK_VALID(handle) ((handle)!= VK_NULL_HANDLE)
#ifndef VK_ALLOCATOR_MIN_POOL_CHUNK_SIZE
#define VK_ALLOCATOR_MIN_POOL_CHUNK_SIZE (256 * 1024 * 1024)
#endif

class MemoryAllocator{
public:
  MemoryAllocator(VkPhysicalDevice physicalDevice,
                  VkDevice         device);

  ~MemoryAllocator();

  Allocation allocate(VkMemoryRequirements  requirements,
                      VkMemoryPropertyFlags desiredFlags,
                      const std::string &   debugName = "GenericAllocation");

  void     free(Allocation allocation, VkDeviceSize size);
  VkResult createBuffer(const VkBufferCreateInfo *pCreateInfo,
                        VkMemoryPropertyFlags     desiredFlags,
                        VkBuffer *                pBuffer,
                        Allocation *              pAllocation,
                        const std::string &       debugName);
  VkResult createImage(const VkImageCreateInfo *pCreateInfo,
                       VkMemoryPropertyFlags    desiredFlags,
                       VkImage *                pImage,
                       Allocation *             pAllocation,
                       const std::string &      debugName);
  void destroyBuffer(VkBuffer          buffer,
                     const Allocation &allocation);
  void destroyImage(VkImage           image,
                    const Allocation &allocation);

  VkDevice         getDevice();
  VkPhysicalDevice getPhysicalDevice();

private:
  std::vector<MemoryPool *> pools;
  VkPhysicalDevice          physicalDevice;
  VkDevice                  device;

  uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
};

#endif //MEMORYALLOCATOR_HPP