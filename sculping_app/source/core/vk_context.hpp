#ifndef MYPROJECT_ENGINE_VK_DEVICE_HPP
#define MYPROJECT_ENGINE_VK_DEVICE_HPP

#include <optional>
#include <string>
#include <cstdint>

#ifdef NDEBUG
const bool ENABLE_VALIDATION_LAYERS = false;
#else
const bool ENABLE_VALIDATION_LAYERS = true;
#endif


struct InstanceCreateInfo{
  std::string applicationName = "myEngine";
  uint32_t applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  std::string engineName      = "noEngine";
  uint32_t engineVersion      = VK_MAKE_VERSION(1, 0, 0);
  uint32_t apiVersion         = VK_API_VERSION_1_3;
  bool enableValidation       = ENABLE_VALIDATION_LAYERS;
};

struct QueueFamilyIndices{
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;
  std::optional<uint32_t> transferFamily;
  std::optional<uint32_t> computeFamily;
  std::optional<uint32_t> sparseFamily;

  bool isComplete() const
  {
    return graphicsFamily.has_value() && presentFamily.has_value();
  }
};

struct VkContext{
  GLFWwindow *window_h;
  VkInstance instance_h;
  VkSurfaceKHR surface_h;
  VkPhysicalDevice physical_device_h;
  VkDevice device_h;
  VkSwapchainKHR swapchain_h;
  VkFormat imageformat_h;
  VkRenderPass renderpass_h;
  VkCommandPool command_pool_h;
  VkQueue present_q;
  VkQueue graphics_q;
  uint32_t graphics_family;
  uint32_t present_family;
};
#endif //MYPROJECT_ENGINE_VK_DEVICE_HPP