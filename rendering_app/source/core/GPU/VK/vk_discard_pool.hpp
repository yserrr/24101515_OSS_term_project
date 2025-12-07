#ifndef MYPROJECT_RESOURCE_TRACKER_HPP
#define MYPROJECT_RESOURCE_TRACKER_HPP
#include <memory>
#include <unordered_map>
#include <queue>
#include "vk_resource.hpp"
#include "vk_memory_pool.hpp"

namespace gpu
{
  class VkContext;
  class VkDiscardPool{
  public:
    VkDiscardPool(VkContext* pCtxt);
    ~VkDiscardPool() = default;
    VkContext* pCtxt_;
    void clean();
    void registerResource(std::vector<VkResource*>& frameResource);
    std::function<void()> buildDiscardRec();

    struct DiscardHandle{
      VkResource* handler;
      std::function<void()> deleteFunction;
    };
    std::unordered_map<VkResourceId, DiscardHandle> handleHash_;
    std::vector<std::vector<DiscardHandle>> frameDiscardPool_;
  };
}


#endif //MYPROJECT_RESOURCE_TRACKER_HPP
