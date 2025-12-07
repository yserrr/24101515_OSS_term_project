#ifndef MYPROJECT_ENGINE_VK_DEVICE_HPP
#define MYPROJECT_ENGINE_VK_DEVICE_HPP
#include "vk_pipeline_pool.hpp"

#ifdef NDEBUG
const bool ENABLE_VALIDATION_LAYERS = false;
#else
const bool ENABLE_VALIDATION_LAYERS = true;
#endif
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <cstdint>
#include <memory>
#include <optional>

#include "vk_descriptor_allocator.hpp"
#include "vk_resource_allocator.hpp"
#include "vk_memory_allocator.hpp"
#include "vk_discard_pool.hpp"
#include "vk_descriptor_layout_builder.hpp"
#include "vk_graph_builder.hpp"
#include "vk_resource.hpp"
#include "vk_swapchain.hpp"
#include "vk_scheduler.hpp"


#include "../gpu_context.hpp"

namespace gpu
{
  class VkPass;

  class VkContext : public Context
  {
    public:
    VkContext() = default ;
    virtual void loadContext() override;
    GLFWwindow* windowh__;
    VkInstance instanceh__;
    VkSurfaceKHR surfaceh__;
    VkPhysicalDevice physicalDeviceh__;
    VkDevice deviceh__;
    VkQueue graphicsQh__;
    VkQueue computeQh__;
    VkQueue presentQh__;
    uint32_t graphicsFamailyIdx__;
    uint32_t presentFamilyIdx__;
    VkRenderPass uiRenderPassh__;
    VkExtent2D initExtent__{2000, 1200};

    VkBool32 dirty_ = VK_FALSE;
    VkBool32 useDynamicRendering = VK_TRUE;
    VkPhysicalDeviceProperties deviceProperties__;
    std::vector<gpu::VkPass*> uploadedPass;
    std::vector<gpu::VkPass> transitionPass;
    std::unordered_map<VkPassId, VkPass*> passHash_;
    std::vector<std::unique_ptr<gpu::VkResource>> nodes_;
    std::unique_ptr<gpu::VkPipelinePool> pPipelinePool;
    std::unique_ptr<gpu::VkMemoryAllocator> pMemoryAllocator;
    std::unique_ptr<gpu::VkSwapchainContext> pSwapChainContext;
    std::unique_ptr<gpu::VkResourceAllocator> pResourceAllocator;
    std::unique_ptr<gpu::VkDiscardPool> pDiscardPool;
    std::unique_ptr<gpu::VkDescriptorAllocator> pDescriptorAllocator;
    std::unique_ptr<gpu::VkDescriptorLayoutBuilder> pLayoutBuilder_;
    std::unique_ptr<gpu::VkGraphBuilder> pGraphBuilder;
    std::unique_ptr<gpu::VkScheduler> pScheduler;

    struct
    {
      std::vector<VkMeshBuffer*> batches;
    } batch;

    struct
    {
      uint32_t maxInflight__ = 2;
      std::vector<uint32_t> inflightIndex__;
      uint32_t currentFrame__ = 0;
    } renderingContext;

    VkResourceId nodeId_ = 0;
    VkPassId passId_ = 0;

    private:
    void loadDeviceContext();
    void loadImGuiGPUContext();
    // ~VkContext();
  };
}

#endif //MYPROJECT_ENGINE_VK_DEVICE_HPP
