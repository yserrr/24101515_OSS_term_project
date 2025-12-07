#ifndef MYPROJECT_NODE_HPP
#define MYPROJECT_NODE_HPP
#include "enum_op.hpp"

namespace gpu
{
  using NodeId = uint32_t;
  using PassId = uint32_t;

  //setting only default
  enum class descriptor_usage: uint32_t
  {
    UBO                    = 0x00000001,
    DYNAMIC_UBO            = 0x00000002,
    SSBO                   = 0x00000004,
    SAMPLER                = 0x00000008,
    COMBINED_IMAGE_SAMPLER = 0x00000010,
    TEXTURE                = 0x00000040,
    TEXTURE_BINDLESS       = 0x00000080,
    NOT_DEFINED            = 0x00000100,
    SHADER_VERTEX          = 0x00000200,
    SHADER_GEOMETRY        = 0x00000400,
    SHADER_FRAGMENT        = 0x00000800,
    SHADER_COMPUTE         = 0x00001000,
    BASE                   = 0x00002000,
  }; //can append raytracing shader structure

  enum class ResourceType: uint32_t
  {
    BUFFER   = 1 << 0,
    IMAGE    = 1 << 1,
    LAYOUT   = 1 << 2,
    PIPELINE = 1 << 3,
    SHADER   = 1 << 4,
    MESH     = 1 << 5,
    TEXTURE  = 1 << 6,
  };

  enum class MemorySpace: uint32_t
  {
    DEVICE_LOCAL = 0,
    HOST_VISIBLE = 1,
    AUTO         = 2
  };

  enum class ResourceUsage : uint32_t
  {
    //index buffer depend on vertex buffer-> integrated
    TEXTURE                  = 0x00000001,
    DEPTH_STENCIL_ATTACHMENT = 0x00000002,
    CONSTANT                 = 0X00000004,
    SHADOW_BUFFER            = 0x00000008,
    LIGHTNING_BUFFER         = 0X00000010,
    G_BUFFER                 = 0x00000020,
    FORWARD                  = 0x00000040,
    SWAPCHAIN_IMAGE          = 0X00000080,
    STAGING                  = 0x00000100,
    VERTEX_BUFFER            = 0x00000200,
    MESH_BUFFER              = 0x00000400,
    INSTANCE_BUFFER          = 0x00000800,
    SHADER_STORAGE_BUFFER    = 0x00001000,
    UNIFORM_BUFFER           = 0x00002000,
    SHADER_RESOURCE          = 0x00004000,
    COMPUTE_RESOURCE         = 0x00008000,
    POST_PROCESS             = 0x00100000,
    TRANSFER                 = 0x00200000,
  };

  enum class RenderPassType : uint32_t
  {
    G_BUFFER_PASS     = 0x00000001,
    SHADOW_PASS       = 0x00000002,
    FORWARD_PASS      = 0x00000004,
    LIGHTING_PASS     = 0x00000008,
    POST_PROCESS_PASS = 0x00000010,
    CUSTOM_PASS       = 0x00000020,
    COMPUTE_PASS      = 0x00000040,
    BARRIER_PASS      = 0x00000080,
    COPY_PASS         = 0x00000100,
    RESOLVE_PASS      = 0x00000200,
    PRESENT_PASS      = 0x00000400,
    RAYTRACING_PASS   = 0x00000800,
    UI                = 0x00001000,
    DEPTH_PASS        = 0x00002000,
    SWAPCHIAN         = 0x00004000,
    BLOOMING_PASS     = 0x00008000,
    TONEMAP_PASS      = 0x00010000,
    GAMMA_PASS        = 0x00020000,
  };

  enum class DescriptorFlag : uint32_t
  {
    UBO                    = 0x00000001,
    DYNAMIC_UBO            = 0x00000002,
    SSBO                   = 0x00000004,
    SAMPLER                = 0x00000008,
    COMBINED_IMAGE_SAMPLER = 0x00000010,
    TEXTURE                = 0x00000040,
    TEXTURE_BINDLESS       = 0x00000080,
    NOT_DEFINED            = 0x00000100,
    SHADER_VERTEX          = 0x00000200,
    SHADER_GEOMETRY        = 0x00000400,
    SHADER_FRAGMENT        = 0x00000800,
    SHADER_COMPUTE         = 0x00001000,
    BASE                   = 0x00002000,
  }; //can append raytracing shader structure

  enum class ShaderFlag : uint32_t
  {
    VertexShader   = 0x00000001,
    FragmentShader = 0x00000002,
    ComputeShader  = 0x00000004,
  };


  struct SubMesh
  {
    uint32_t startIndex = 0;
    uint32_t indexCount = 0;
    uint32_t vertexOffset = 0;
  };


  constexpr uint32_t GRAPHICS_PASS = RenderPassType::G_BUFFER_PASS |
    RenderPassType::SHADOW_PASS |
    RenderPassType::FORWARD_PASS |
    RenderPassType::LIGHTING_PASS |
    RenderPassType::PRESENT_PASS;

  constexpr uint32_t COMPUTE_PASS = RenderPassType::COMPUTE_PASS |
    RenderPassType::POST_PROCESS_PASS |
    RenderPassType::CUSTOM_PASS |
    RenderPassType::RAYTRACING_PASS;

  constexpr uint32_t TRANSFER_PASS = RenderPassType::COPY_PASS |
    RenderPassType::RESOLVE_PASS;


  constexpr uint32_t VERTEX_BINDING = 0;
  constexpr uint32_t INDEX_BINDING = 0;
  constexpr uint32_t INSTANCE_BINDING = 1;
  constexpr uint32_t GLOBAL_LAYOUT = 0;
  constexpr uint32_t CAMERA_BINDING = 0;
  constexpr uint32_t GLOBAL_LIGHT = 1;
  constexpr uint32_t LOCAL_LIGHT = 3;
  constexpr uint32_t SHADOW_LAYOUT = 4;
  constexpr uint32_t TEXTURE_LAYOUT = 1;
  constexpr uint32_t BINDLESS_TEXTURE = 2;
  constexpr uint32_t ALBEDO_BINDING = 0;
  constexpr uint32_t NORMAL_BINDING = 1;
  constexpr uint32_t ROUGHNESS_BINDING = 2;
  constexpr uint32_t BINDELSS_TEXTURE_ARRAY_COUNT = 1024;
  constexpr uint32_t DYNAMIC_UBO_ARRAY_COUNT = 8;
}


#endif //MYPROJECT_NODE_HPP
