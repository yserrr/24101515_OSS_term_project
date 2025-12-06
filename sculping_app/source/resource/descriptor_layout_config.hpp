#ifndef MYPROJECT_DESCRIPTOR_CONFIG_HPP
#define MYPROJECT_DESCRIPTOR_CONFIG_HPP
#include <cstdint>

namespace gpu{
  //setting only default
  enum class descriptor_usage: uint32_t{
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

  constexpr uint32_t VERTEX_BINDING   = 0;
  constexpr uint32_t INDEX_BINDING    = 0;
  constexpr uint32_t INSTANCE_BINDING = 1;

  constexpr uint32_t UBO_LAYOUT           = 1;
  constexpr uint32_t CAMERA_BINDING       = 0;
  constexpr uint32_t LOCAL_LIGHT_BINDING  = 1;
  constexpr uint32_t GLOBAL_LIGHT_BINDING = 2;

  constexpr uint32_t TEXTURE_LAYOUT    = 1;
  constexpr uint32_t BINDLESS_BINDING  = 0;
  constexpr uint32_t ALBEDO_BINDING    = 1;
  constexpr uint32_t NORMAL_BINDING    = 2;
  constexpr uint32_t ROUGHNESS_BINDING = 3;
  inline uint32_t operator&(uint32_t a, gpu::descriptor_usage b)
  {
    return a & static_cast<unsigned int>(b);
  }

  inline constexpr uint32_t operator|(gpu::descriptor_usage a, gpu::descriptor_usage b)
  {
    return static_cast<unsigned int>(a) | static_cast<unsigned int>(b);
  }

  inline constexpr uint32_t operator|(uint32_t a, gpu::descriptor_usage b)
  {
    return a | static_cast<unsigned int>(b);
  }


};

#endif //MYPROJECT_DESCRIPTOR_CONFIG_HPP