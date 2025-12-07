#ifndef VK_MEMORYALLOCATOR
#define VK_MEMORYALLOCATOR

#include "vk_memory_pool.hpp"
#include "vk_resource.hpp"

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
namespace gpu
{
  class VkContext;

  class VkMemoryAllocator
  {
    public:
    VkMemoryAllocator(gpu::VkContext* pCtxt);
    ~VkMemoryAllocator();
    VkAllocation allocate(VkMemoryRequirements requirements,
                          VkMemoryPropertyFlags desiredFlags,
                          const std::string& debugName = "GenericAllocation");
    void free(VkAllocation allocation, VkDeviceSize size);

    private:
    std::vector<VkMemoryPool*> pools;
    VkContext* pCtxt;

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
  };
}

#endif //MEMORYALLOCATOR_HPP
