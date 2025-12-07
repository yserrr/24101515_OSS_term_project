#include <array>
#include "vk_graph_builder.hpp"
#include "vk_context.hpp"
#include "vk_common.hpp"
#include "vk_swapchain.hpp"
#include "vk_resource_allocator.hpp"
#include "vk_descriptor_allocator.hpp"
#include "vk_scheduler.hpp"
#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include "imnodes.h"


namespace gpu
{
  const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
  };
  PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT_ptr = nullptr;
  PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT_ptr = nullptr;
  PFN_vkSetDebugUtilsObjectNameEXT g_pfnSetDebugUtilsObjectNameEXT = nullptr;
  PFN_vkSetDebugUtilsObjectTagEXT g_pfnSetDebugUtilsObjectTagEXT = nullptr;


  static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData);

  void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);

  struct InstanceCreateInfo
  {
    std::string applicationName = "myEngine";
    uint32_t applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    std::string engineName = "noEngine";
    uint32_t engineVersion = VK_MAKE_VERSION(1, 0, 0);
    uint32_t apiVersion = VK_API_VERSION_1_3;
    bool enableValidation = ENABLE_VALIDATION_LAYERS;
  };

  struct QueueFamilyIndices
  {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    std::optional<uint32_t> transferFamily;
    std::optional<uint32_t> computeFamily;
    std::optional<uint32_t> sparseFamily;

    bool isComplete() const
    {
      return graphicsFamily.has_value() && presentFamily.has_value() &&  computeFamily.has_value();
    }
  };


  void VkContext::loadContext()
  {
    loadDeviceContext();
    pSwapChainContext = std::make_unique<gpu::VkSwapchainContext>(this);
    pMemoryAllocator = std::make_unique<gpu::VkMemoryAllocator>(this);
    pResourceAllocator = std::make_unique<gpu::VkResourceAllocator>(this);
    pLayoutBuilder_ = std::make_unique<gpu::VkDescriptorLayoutBuilder>(this);
    pDescriptorAllocator = std::make_unique<gpu::VkDescriptorAllocator>(this);
    pDiscardPool = std::make_unique<gpu::VkDiscardPool>(this);
    pPipelinePool = std::make_unique<gpu::VkPipelinePool>(this);
    pGraphBuilder = std::make_unique<gpu::VkGraphBuilder>(this);
    pScheduler = std::make_unique<gpu::VkScheduler>(this);
    loadImGuiGPUContext();
  }

  void gpu::VkContext::loadDeviceContext()
  {
    GLFWwindow* window_h;
    VkInstance instance_h;
    VkSurfaceKHR surface_h;
    VkPhysicalDevice physical_device_h;
    VkDevice device_h;
    VkQueue present_q;
    VkQueue graphics_q;
    VkQueue compute_q;
    uint32_t compute_family;
    uint32_t graphics_family;
    uint32_t present_family;
    VkExtent2D extent = {2400, 1200};
    const char* title = "vulkan";

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE); //need to change
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    window_h = glfwCreateWindow(extent.width, extent.height, title, nullptr, nullptr);
    GLFWmonitor* primary = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(primary);
    glfwMakeContextCurrent(window_h);
    int winWidth = extent.width;
    int winHeight = extent.height;
    int centerX = (mode->width - winWidth) / 2;
    int centerY = (mode->height - winHeight) / 2;

    glfwSetWindowPos(window_h, centerX, centerY);

    InstanceCreateInfo instance_info{};
    bool enableValidation = instance_info.enableValidation;

    VkApplicationInfo appInfo{
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = instance_info.applicationName.c_str(),
      .applicationVersion = instance_info.applicationVersion,
      .pEngineName = instance_info.engineName.c_str(),
      .engineVersion = instance_info.engineVersion,
      .apiVersion = instance_info.apiVersion
    };

    std::vector<const char*> enableExtensions;
    uint32_t glfwExtensionCount = 0;

    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> requiredExtensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidation)
    {
      requiredExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    spdlog::debug("Required Extensions:");
    for (const char* ext : requiredExtensions)
    {
      spdlog::debug("  {}", ext);
    }

    uint32_t instanceExtensionCount;
    vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(instanceExtensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount, availableExtensions.data());
    spdlog::debug("Available Instance Extensions:");

    for (const auto& extension : availableExtensions)
    {
      spdlog::debug("  {}", extension.extensionName);
    }

    for (const char* requiredExtension : requiredExtensions)
    {
      bool found = false;
      for (const auto& availableExtension : availableExtensions)
      {
        if (strcmp(requiredExtension, availableExtension.extensionName) == 0)
        {
          found = true;
          break;
        }
      }
      if (!found)
      {
        spdlog::error("Required instance extension '{}' not found.", requiredExtension);
        //throw std::runtime_error("Required requiredExtensions not supported!");
      }
    }
    enableExtensions = requiredExtensions;

    VkDebugUtilsMessengerEXT debugMessenger;
    VkInstanceCreateInfo vkCreateInfo{};
    vkCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    vkCreateInfo.pApplicationInfo = &appInfo;
    vkCreateInfo.ppEnabledExtensionNames = enableExtensions.data();
    vkCreateInfo.enabledExtensionCount = static_cast<uint32_t>(enableExtensions.size());

    if (enableValidation)
    {
      //validation layer check
      uint32_t layerCount;
      vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
      std::vector<VkLayerProperties> availableLayers(layerCount);
      vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
      spdlog::debug("Available Validation Layers:");
      for (const auto& layer : availableLayers)
      {
        spdlog::debug("  {}", layer.layerName);
      }

      for (const char* layerName : validationLayers)
      {
        bool layerFound = false;
        for (const auto& layerProperties : availableLayers)
        {
          if (strcmp(layerName, layerProperties.layerName) == 0)
          {
            layerFound = true;
            break;
          }
        }
        if (!layerFound)
        {
          spdlog::error("Validation layer '{}' requested but not found.", layerName);
          ////throw std::runtime_error("Validation layer not supported!");
        }
      }
      std::vector<const char*> enabledLayers = validationLayers;
      vkCreateInfo.enabledLayerCount = static_cast<uint32_t>(enabledLayers.size());
      vkCreateInfo.ppEnabledLayerNames = enabledLayers.data();
      VkDebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfoLocal{};
      populateDebugMessengerCreateInfo(debugMessengerCreateInfoLocal);
      vkCreateInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugMessengerCreateInfoLocal;
      VkResult result = vkCreateInstance(&vkCreateInfo, nullptr, &instance_h);
      if (result != VK_SUCCESS)
      {
        spdlog::error("Failed to create VkInstance! Result: {}", static_cast<int>(result));
        //throw std::runtime_error("Failed to create Vulkan instance!");
      }
    }
    else
    {
      vkCreateInfo.enabledLayerCount = 0;
      vkCreateInfo.pNext = nullptr;
      VkResult result = vkCreateInstance(&vkCreateInfo, nullptr, &instance_h);
      if (result != VK_SUCCESS)
      {
        spdlog::error("Failed to create VkInstance! Result: {}", static_cast<int>(result));
        //throw std::runtime_error("Failed to create Vulkan instance!");
      }
    }

    spdlog::info("Vulkan Instance created successfully.");
    if (enableValidation)
    {
      VkDebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo{};
      populateDebugMessengerCreateInfo(debugMessengerCreateInfo);
      vkCreateDebugUtilsMessengerEXT_ptr =
        reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance_h,
             "vkCreateDebugUtilsMessengerEXT"));
      vkDestroyDebugUtilsMessengerEXT_ptr
        = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(instance_h,
             "vkDestroyDebugUtilsMessengerEXT"));
      g_pfnSetDebugUtilsObjectNameEXT
        = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetInstanceProcAddr(instance_h, "vkSetDebugUtilsObjectNameEXT");
      g_pfnSetDebugUtilsObjectTagEXT
        = (PFN_vkSetDebugUtilsObjectTagEXT)vkGetInstanceProcAddr(instance_h, "vkSetDebugUtilsObjectTagEXT");

      if (vkCreateDebugUtilsMessengerEXT_ptr == nullptr)
      {
        spdlog::warn("Failed to load vkCreateDebugUtilsMessengerEXT function pointer.");
        //throw std::runtime_error("Failed to load vkCreateDebugUtilsMessengerEXT!");
      }

      VkResult debugSetupResult = vkCreateDebugUtilsMessengerEXT_ptr(instance_h,
                                                                     &debugMessengerCreateInfo,
                                                                     nullptr,
                                                                     &debugMessenger);
      if (debugSetupResult != VK_SUCCESS)
      {
        spdlog::error("Failed to set up debug messenger! Result: {}", static_cast<int>(debugSetupResult));
        //throw std::runtime_error("Failed to set up debug messenger!");
      }
      spdlog::info("Debug messenger setup successfully.");
    }
    VK_ASSERT(glfwCreateWindowSurface(instance_h, window_h, nullptr, &surface_h));

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_h, &deviceCount, nullptr);

    if (deviceCount == 0)
    {
      //throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> vk_physical_devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_h, &deviceCount, vk_physical_devices.data());

    int score = 0;
    for (const VkPhysicalDevice& physical_device : vk_physical_devices)
    {
      uint32_t physicalDeviceExtensionCount = 0;
      vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &physicalDeviceExtensionCount, nullptr);

      std::vector<VkExtensionProperties> deviceAvailableExtensions(physicalDeviceExtensionCount);
      vkEnumerateDeviceExtensionProperties(physical_device,
                                           nullptr,
                                           &physicalDeviceExtensionCount,
                                           deviceAvailableExtensions.data());

      std::set<std::string> engineRequiredExtension = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
        VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME,
        VK_EXT_SHADER_OBJECT_EXTENSION_NAME,
        VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,


      };

      for (const auto& extension : availableExtensions)
      {
        engineRequiredExtension.erase(extension.extensionName);
      }

      if (engineRequiredExtension.empty())
      {
        continue;
      }

      QueueFamilyIndices indices;
      uint32_t queueFamilyCount = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queueFamilyCount, nullptr);

      std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
      vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queueFamilyCount, queueFamilies.data());
      for (uint32_t i = 0; i < queueFamilyCount; ++i)
      {
        const auto& queueFamily = queueFamilies[i];

        if (queueFamily.queueFlags & static_cast<uint32_t>(VK_QUEUE_GRAPHICS_BIT))
        {
          indices.graphicsFamily = i;
        }

        if (queueFamily.queueFlags & static_cast<uint32_t>(VK_QUEUE_TRANSFER_BIT))
        {
          indices.computeFamily = i;
        }

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, i, surface_h, &presentSupport);
        if (presentSupport)
        {
          indices.presentFamily = i;
        }
        if (indices.isComplete())
        {
          break;
        }
      }

      if (!indices.isComplete())
      {
        //throw std::runtime_error("Queue family fail to find");
      }

      VkPhysicalDeviceProperties props;
      vkGetPhysicalDeviceProperties(physical_device, &props);

      VkPhysicalDeviceFeatures features;
      vkGetPhysicalDeviceFeatures(physical_device, &features);
      uint32_t tempScore = 0;

      if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
      {
        tempScore += 1000;
      }
      tempScore += props.limits.maxImageDimension2D;

      if (tempScore > score)
      {
        physical_device_h = physical_device;
        score = tempScore;
        graphics_family = indices.graphicsFamily.value();
        compute_family  = indices.computeFamily.value();
        present_family = indices.presentFamily.value();
      }
    }

    if (physical_device_h == VK_NULL_HANDLE)
    {
      //throw std::runtime_error("failed to find a suitable GPU!");
    }

    if (enableValidation)
    {
      VkPhysicalDeviceProperties properties;
      vkGetPhysicalDeviceProperties(physical_device_h, &properties);

      spdlog::debug("Physical device name: {}", properties.deviceName);
      spdlog::debug("API version: {}.{}.{}",
                    VK_VERSION_MAJOR(properties.apiVersion),
                    VK_VERSION_MINOR(properties.apiVersion),
                    VK_VERSION_PATCH(properties.apiVersion));
      spdlog::debug("Vendor ID: 0x{:04X}", properties.vendorID);
      spdlog::debug("Device ID: 0x{:04X}", properties.deviceID);

      spdlog::debug("Max 2D image dimension: {}", properties.limits.maxImageDimension2D);
      spdlog::debug("Max 3D image dimension: {}", properties.limits.maxImageDimension3D);
      spdlog::debug("Max cube image dimension: {}", properties.limits.maxImageDimensionCube);
      spdlog::debug("Max push constant : {}", properties.limits.maxPushConstantsSize);
      spdlog::debug("Max color attachment: {}", properties.limits.maxColorAttachments);
      spdlog::debug("Max sampler Lod Bias: {}", properties.limits.maxSamplerLodBias);
      spdlog::debug("Max UBO range : {}", properties.limits.maxUniformBufferRange);
      spdlog::debug("Max SSBO range : {}", properties.limits.maxStorageBufferRange);
      spdlog::debug("Max ViewPort : {}", properties.limits.maxViewports);

      VkPhysicalDeviceMemoryProperties memProperties;
      vkGetPhysicalDeviceMemoryProperties(physical_device_h, &memProperties);
      VkDeviceSize totalSize = 0;
      for (uint32_t i = 0; i < memProperties.memoryHeapCount; ++i)
      {
        if (memProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
          totalSize += memProperties.memoryHeaps[i].size;
      }
      spdlog::info("Total GPU memory size: {} MB", (totalSize / (1024.0 * 1024.0)));
    }

    spdlog::info("LogicalDevice: init");

    std::set<uint32_t> uniqueQueueFamilies = {
      graphics_family,
      compute_family,
      present_family
    };

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    float queuePriority = 1.0f;

    for (uint32_t queueFamily : uniqueQueueFamilies)
    {
      VkDeviceQueueCreateInfo queueCreateInfo{};
      queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queueCreateInfo.queueFamilyIndex = queueFamily;
      queueCreateInfo.queueCount = 1;
      queueCreateInfo.pQueuePriorities = &queuePriority;
      queueCreateInfos.push_back(queueCreateInfo);
    }

    const std::vector<const char*> requiredDeviceExtensions = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME,
      VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME,
      VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME,
      VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME,
      VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
      VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
      VK_NV_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME,
      VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
      VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
      VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
      VK_EXT_MESH_SHADER_EXTENSION_NAME,
      VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME,
      VK_KHR_PRESENT_WAIT_EXTENSION_NAME,
      VK_KHR_PRESENT_ID_EXTENSION_NAME,
      VK_KHR_MAINTENANCE1_EXTENSION_NAME,
    };
    uint32_t deviceExtensionCount;
    vkEnumerateDeviceExtensionProperties(physical_device_h, nullptr, &deviceExtensionCount, nullptr);
    std::vector<VkExtensionProperties> deviceAvailableExtensions(deviceExtensionCount);
    vkEnumerateDeviceExtensionProperties(physical_device_h,
                                         nullptr,
                                         &deviceExtensionCount,
                                         deviceAvailableExtensions.data());

    std::vector<const char*> enabledExtensions;
    for (const char* extensionName : requiredDeviceExtensions)
    {
      bool found = false;
      for (const auto& ext : deviceAvailableExtensions)
      {
        if (std::strcmp(extensionName, ext.extensionName) == 0)
        {
          found = true;
          enabledExtensions.push_back(extensionName);
          break;
        }
      }
      if (!found)
      {
        //throw std::runtime_error(
        //     "LogicalDevice: Required device extension not supported: " +
        //     std::string(extensionName));
      }
    }
    const char* descriptorIndexingExtName = VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME;
    const char* push_desc = "VK_KHR_push_descriptor";

    bool descriptorIndexingExtFound = false;
    for (const auto& ext : deviceAvailableExtensions)
    {
      if (std::strcmp(descriptorIndexingExtName, ext.extensionName) == 0)
      {
        descriptorIndexingExtFound = true;
        break;
      }
    }
    if (descriptorIndexingExtFound)
    {
      enabledExtensions.push_back(descriptorIndexingExtName);
      enabledExtensions.push_back(push_desc);
      spdlog::info("LogicalDevice: {} extension found and enabled.", descriptorIndexingExtName);
    }
    else
    {
      spdlog::warn("LogicalDevice: {} extension not found. Bindless features may not be available.",
                   descriptorIndexingExtName);
    }

    VkPhysicalDeviceDescriptorIndexingFeatures indexingFeatures{};
    indexingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
    indexingFeatures.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    indexingFeatures.runtimeDescriptorArray = VK_TRUE;
    indexingFeatures.descriptorBindingPartiallyBound = VK_TRUE;
    indexingFeatures.descriptorBindingVariableDescriptorCount = VK_TRUE;
    indexingFeatures.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE;
    indexingFeatures.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    indexingFeatures.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
    indexingFeatures.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    indexingFeatures.descriptorBindingUniformTexelBufferUpdateAfterBind = VK_TRUE;
    indexingFeatures.descriptorBindingStorageTexelBufferUpdateAfterBind = VK_TRUE;

    VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamicRenderingFeatures = {};
    dynamicRenderingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR;
    dynamicRenderingFeatures.dynamicRendering = VK_TRUE;

    VkPhysicalDeviceExtendedDynamicState2FeaturesEXT dynamicPipeline2Features = {};
    dynamicPipeline2Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT;
    dynamicPipeline2Features.extendedDynamicState2 = VK_TRUE;

    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT dynamicPipeline3Features{};
    dynamicPipeline3Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT;
    dynamicPipeline3Features.extendedDynamicState3PolygonMode = VK_TRUE;

    VkPhysicalDeviceFeatures2 deviceFeatures2{};
    deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    deviceFeatures2.features.robustBufferAccess = VK_TRUE;


    VkPhysicalDeviceBufferDeviceAddressFeatures addressFeature;
    addressFeature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
      addressFeature.bufferDeviceAddress = VK_TRUE;

    VkPhysicalDeviceDynamicRenderingFeatures dynamicReadFeature;
    dynamicReadFeature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES;
    dynamicReadFeature.dynamicRendering = VK_TRUE;

    VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT swapchainMaintenance1Features = {};
    swapchainMaintenance1Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT;
    swapchainMaintenance1Features.swapchainMaintenance1 = true;

    VkPhysicalDevicePresentWaitFeaturesKHR presentWait{};
    presentWait.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR;
    presentWait.presentWait = VK_TRUE;

    VkPhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT unusedRenderingFeatures{};
    unusedRenderingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_FEATURES_EXT;
    unusedRenderingFeatures.dynamicRenderingUnusedAttachments = VK_TRUE;

    deviceFeatures2.pNext = &indexingFeatures;
    indexingFeatures.pNext = &dynamicRenderingFeatures;
    dynamicRenderingFeatures.pNext = &dynamicPipeline2Features;
    dynamicPipeline2Features.pNext = &dynamicPipeline3Features;
    dynamicPipeline3Features.pNext = &addressFeature;
    addressFeature.pNext = &dynamicReadFeature;
    dynamicReadFeature.pNext = &swapchainMaintenance1Features;
    swapchainMaintenance1Features.pNext = &presentWait;
    presentWait.pNext = &unusedRenderingFeatures;
    unusedRenderingFeatures.pNext = nullptr;
    vkGetPhysicalDeviceFeatures2(physical_device_h, &deviceFeatures2);

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
    createInfo.ppEnabledExtensionNames = enabledExtensions.data();
    createInfo.pEnabledFeatures = nullptr;
    createInfo.pNext = &deviceFeatures2;

    VK_ASSERT(vkCreateDevice(physical_device_h, &createInfo, nullptr, &device_h));
    spdlog::info("LogicalDevice: Device created successfully.");

    vkGetDeviceQueue(device_h, graphics_family, 0, &graphics_q);
    vkGetDeviceQueue(device_h, compute_family, 0 , &compute_q);
    vkGetDeviceQueue(device_h, present_family, 0, &present_q);
    spdlog::info("LogicalDevice: Queues obtained.");

    windowh__ = window_h;
    instanceh__ = instance_h;
    deviceh__ = device_h;
    surfaceh__ = surface_h;
    physicalDeviceh__ = physical_device_h;
    graphicsQh__ = graphics_q;
    computeQh__ = compute_q;
    presentQh__ = present_q;
    graphicsFamailyIdx__ = graphics_family;
    presentFamilyIdx__ = graphics_family;
    vkGetPhysicalDeviceProperties(this->physicalDeviceh__,
                                  &this->deviceProperties__);
  }

  void VkContext::loadImGuiGPUContext()
  {
    IMGUI_CHECKVERSION();
    auto context = ImGui::CreateContext();
    ImGui::StyleColorsDark();

    VkDescriptorPoolSize pool_sizes[] = {
      {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}
    };
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
    pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;
    VkDescriptorPool pool;

    if (vkCreateDescriptorPool(deviceh__, &pool_info, nullptr, &pool) != VK_SUCCESS)
    {
      //throw std::runtime_error("fail to create Imgui Pool");
    }

    VkAttachmentDescription UI{};
    UI.format = pSwapChainContext->imgFormat__;
    UI.samples = VK_SAMPLE_COUNT_1_BIT;
    UI.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    UI.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    UI.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    UI.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    UI.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    UI.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference uiColorAttachmentRef{};
    uiColorAttachmentRef.attachment = 0;
    uiColorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    std::array<VkAttachmentDescription, 1> attachments = {UI};
    VkSubpassDescription uiSubpass{};
    uiSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    uiSubpass.colorAttachmentCount = 1;
    uiSubpass.pColorAttachments = &uiColorAttachmentRef;
    uiSubpass.pDepthStencilAttachment = nullptr;
    std::array<VkSubpassDependency, 1> dependencies{};

    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = 0;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    std::array<VkSubpassDescription, 1> subpasses = {uiSubpass};
    renderPassInfo.pSubpasses = subpasses.data();
    renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
    renderPassInfo.pDependencies = dependencies.data();
    VkRenderPass uiPass = VK_NULL_HANDLE;
    if (vkCreateRenderPass(deviceh__, &renderPassInfo, nullptr, &uiPass) != VK_SUCCESS)
    {
      //throw std::runtime_error("failed to create render pass!");
    }
    uiRenderPassh__ = uiPass;
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physicalDeviceh__, &properties);
    ImGui_ImplVulkan_InitInfo UIinfo = {};
    UIinfo.Instance = instanceh__;
    UIinfo.PhysicalDevice = physicalDeviceh__;
    UIinfo.Device = deviceh__;
    UIinfo.QueueFamily = graphicsFamailyIdx__;
    UIinfo.Queue = graphicsQh__;
    UIinfo.PipelineCache = VK_NULL_HANDLE;
    UIinfo.DescriptorPool = pool;
    UIinfo.Allocator = nullptr;
    UIinfo.MinImageCount = 2;
    UIinfo.ImageCount = this->pSwapChainContext->img__.size();
    UIinfo.CheckVkResultFn = nullptr;
    UIinfo.RenderPass = uiPass;
    //UIinfo.RenderPass = VK_NULL_HANDLE;
    UIinfo.ApiVersion = properties.apiVersion;
    UIinfo.Subpass = 0;
    UIinfo.UseDynamicRendering = VK_FALSE;
    UIinfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    //UIinfo.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    ImGui_ImplGlfw_InitForVulkan(windowh__, true);
    ImGui_ImplVulkan_Init(&UIinfo);
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontDefault();
    IM_ASSERT(io.Fonts != nullptr);
    spdlog::info("start imgui");
  }

  //   // VkContext::~VkContext()
  // {
  //   graphBuilder..reset();
  //   pMemoryAllocator.reset();
  //   pSwapChainContext.reset();
  //   pResourceAllocator.reset();
  //   pDiscardPool.reset();
  //   pDescriptorAllocator.reset();
  //   vkDestroyDevice(deviceh__, nullptr);
  //   vkDestroySurfaceKHR(instanceh__, surfaceh__, nullptr);
  //   vkDestroyInstance(instanceh__, nullptr);
  //   glfwDestroyWindow(windowh__);
  // }

  VkBool32 debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                         VkDebugUtilsMessageTypeFlagsEXT messageType,
                         const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                         void* pUserData)
  {
    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    {
      spdlog::error("Validation Layer ERROR: {}", pCallbackData->pMessage);
    }
    else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    {
      spdlog::warn("Validation Layer WARNING: {}", pCallbackData->pMessage);
    }
    else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
    {
      spdlog::info("Validation Layer INFO: {}", pCallbackData->pMessage);
    }
    else
    {
      spdlog::debug("Validation Layer DEBUG: {}", pCallbackData->pMessage);
    }
    return VK_FALSE;
  }

  void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
  {
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
      static_cast<uint32_t>(VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) |
      static_cast<uint32_t>(VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) |
      static_cast<uint32_t>(VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT);
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
      static_cast<uint32_t>(VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) |
      static_cast<uint32_t>(VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT);
    createInfo.pfnUserCallback = debugCallback;
    createInfo.pUserData = nullptr;
  }
}
