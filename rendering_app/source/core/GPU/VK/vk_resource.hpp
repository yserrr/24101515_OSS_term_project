#ifndef MYPROJECT_RENDER_GRAPH_NODE_HPP
#define MYPROJECT_RENDER_GRAPH_NODE_HPP
#include <cstdint>
#include <vector>
#include <string>
#include <vulkan/vulkan.h>
#include <functional>
#include <iterator>
#include <optional>
#include <queue>
#include <glm/glm.hpp>
#include <unordered_set>
#include "enum_op.hpp"
#include "vk_memory_pool.hpp"
#include "vk_pipeline_pool.hpp"
#include "../flag.hpp"
#include "../resource/vertex.hpp"
struct VertexAll;

namespace gpu
{
  using VkRenderPassType = RenderPassType;
  using VkDescriptorFlag = DescriptorFlag;
  using VkResourceUsage = ResourceUsage;
  using VkResourceType = ResourceType;
  using VkMemorySpace = MemorySpace;
  using VkSubMesh = SubMesh;


  constexpr uint32_t VK_VERTEX_BINDING = 0;
  constexpr uint32_t VK_INDEX_BINDING = 0;
  constexpr uint32_t VK_INSTANCE_BINDING = 1;
  constexpr uint32_t VK_GLOBAL_LAYOUT = 0;
  constexpr uint32_t VK_BINDLESS_TEXTURE = 0;
  constexpr uint32_t VK_CAMERA_BINDING = 1;
  constexpr uint32_t VK_GLOBAL_LIGHT = 2;
  constexpr uint32_t VK_LOCAL_LIGHT = 3;
  constexpr uint32_t VK_SHADOW_LAYOUT = 4;
  constexpr uint32_t VK_TEXTURE_LAYOUT = 1;
  constexpr uint32_t VK_ALBEDO_BINDING = 0;
  constexpr uint32_t VK_NORMAL_BINDING = 1;
  constexpr uint32_t VK_ROUGHNESS_BINDING = 2;
  constexpr uint32_t VK_BINDELSS_TEXTURE_ARRAY_COUNT = 512;
  constexpr uint32_t VK_DYNAMIC_UBO_ARRAY_COUNT = 8;

  using VkPassId = uint32_t;
  using VkResourceId = uint32_t;

  class VkPass;


  enum class VkResourceLifetime
  {
    TRANSFER   = 0,
    PERSISTENT = 1,
    FRAME      = 2
  };

  enum class VkComputePassCMD
  {
    ///todo:
    ///compute pass lambda build dispatcher
  };

  class VkResource
  {
    public:
    friend class VkGraphBuilder;
    friend class VkGraphBuilder;
    friend class VkResourceAllocator;
    friend class VkGraph;
    friend class VkDiscardPool;
    VkBool32 masked = false;
    std::string nodeName_;
    uint32_t descriptorSetId;
    uint32_t descriptorBindingId;
    ShaderFlag shaderFlag;
    DescriptorFlag descriptorFlag;
    ResourceType type_;
    ResourceUsage usage_;
    MemorySpace mSpace_;
    VkResourceLifetime lifetime = VkResourceLifetime::PERSISTENT;
    bool hostUpdate__ = false;
    bool dirty__ = false;
    VkPass* lastWriter__ = nullptr;

    private:
    VkMemoryPropertyFlags mFlags_ = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    uint32_t nodeId_ = 0;
    bool allocated__ = false;
    uint32_t referenceCount_ = 0;
    uint32_t currentPipeline__ = VK_PIPELINE_STAGE_TRANSFER_BIT;
    uint32_t writePipeline__ = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkAccessFlags writeAccessMask__ = VK_ACCESS_MEMORY_READ_BIT;
    VkAccessFlags currentAccessMask__ = VK_ACCESS_NONE;
  };

  struct VkMeshConstant
  {
    glm::mat4 modelMatrix_;
    std::int32_t albedoTexture = -1;
    std::int32_t normalTexture = -1;
    std::int32_t roughnessTexture = -1;
    std::int32_t metalnessTexture = -1;
  };

  class VkMeshBuffer : public VkResource
  {
    public:
    void draw(VkCommandBuffer cmd);
    void* vData__;
    void* iData__;
    uint32_t indexCount__;
    VkBuffer vertexBuffer__;
    VkBuffer indexBuffer__;
    std::vector<VertexAll> vertex;
    std::vector<uint32_t> indices;
    VkDeviceSize vSize__;
    VkDeviceSize iSize__;
    VkAllocation vAllocation__;
    VkAllocation iAllocation__;
    VkPipeline pipeline__;
    VkPipelineLayout pipelineLayout__;
    VkAccessFlagBits writeAccess__ = VK_ACCESS_TRANSFER_WRITE_BIT;
    VkPipelineStageFlagBits writePipelineState__ = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    gpu::VkConstant constant__;
    std::vector<SubMesh> submeshes__;
    std::vector<VkDescriptorSet> descriptorSet__;
    VkBool32 dirty__;
  };


  class VkShaderNode : public VkResource
  {
    VkShaderStageFlagBits stage;
    VkPipelineLayout layout;
    VkPipeline pipeline;
    std::vector<VkDescriptorSetLayout> descriptorLayouts;
    std::string entryPoint;
    std::vector<uint32_t> codeHash; // Shader module 캐싱용
    bool compiled = false;          // pipeline 생성 여부
    bool dirty = false;             // 코드/파라미터 변경 여부
  };
}


#endif //MYPROJECT_RENDER_GRAPH_NODE_HPP
