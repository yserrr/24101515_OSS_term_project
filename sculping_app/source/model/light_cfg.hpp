//
// Created by ljh on 25. 9. 16..
//

#ifndef MYPROJECT_LIGHT_CFG_HPP
#define MYPROJECT_LIGHT_CFG_HPP
#include<glm/vec3.hpp>
#include<glm/vec4.hpp>
#include<glm/mat4x4.hpp>
#include<vulkan/vulkan.h>
#include<vector>

#ifndef MAX_LIGHTS
#define MAX_LIGHTS 16
#endif

#ifndef MAX_CASCADES
#define MAX_CASCADES 4
#endif

enum class LightType: uint32_t{
  DIRECTIONAL = 0,
  POINT       = 1,
  SPOT        = 2
};

namespace cpu{
  struct light{
    LightType type      = LightType::POINT;
    uint32_t flags      = LightFlag_None;
    glm::vec3 position  = glm::vec3(0.0f);
    glm::vec3 direction = glm::vec3(0.0f, -1.0f, 0.0f);
    float range         = 10.0f;
    float innerDeg      = 15.0f; // Spot 내부 각(도)
    glm::vec3 color     = glm::vec3(1.0f);
    float intensity     = 1.0f;  // 조도 스케일(엔진 규칙에 맞춰 정의)
    float outerDeg      = 25.0f; // Spot 외곽 각(도) — 셰이더에선 cos로 사용
    float falloff       = 1.0f;  // 스팟 가장자리 감쇠 또는 포인트 거리 감쇠 지수 등
    float pad0          = 0.0f;
    float pad1          = 0.0f;
    bool dirty          = false;
  };
}

namespace gpu{
  struct alignas(16) light{
    glm::vec4 position;  // xyz=pos, w=type
    glm::vec4 direction; // xyz=dir, w=innerCos
    glm::vec4 color;     // rgb=color, a=intensity
    glm::vec4 params;    // x=range, y=outerCos, z=falloff, w=flags
  };

  struct alignas(16) light_batch{
    gpu::light lights[MAX_LIGHTS]; // 64 * MAX_LIGHTS bytes
    uint32_t lightCount;
    uint32_t padding1;
    uint32_t padding2;
    uint32_t padding3;
  };

  struct alignas(16) global_light_param{
    glm::vec4 ambient;  // rgb=ambientColor, a=intensity
    glm::vec4 iblScale; // x=irradianceScale, y=prefilteredScale, z=brdfLutScale(옵션), w=reserved
  };

  static_assert(sizeof(gpu::light) == 64, "LightGPU must be 64 bytes");

  struct alignas(16) cascade_shadow_batch{
    glm::mat4 viewProj[MAX_CASCADES]; // 카스케이드별 VP
    glm::vec4 splitDepths;            // x,y,z,w 또는 (z0,z1,z2,z3)
    uint32_t cascadeCount;
  };
}

///struct IBLDescriptors{
///  VkDescriptorImageInfo irradiance{}; // cube
///  VkDescriptorImageInfo prefiltered{}; // cube
///  VkDescriptorImageInfo brdfLut{}; // 2D
///  bool valid = false;
///};
///
///struct ShadowDescriptors{
///  // 방향광 캐스케이드(예: 4장)
///  std::vector<VkDescriptorImageInfo> dirCascades;
///  bool valid = false;
///};
#endif //MYPROJECT_LIGHT_CFG_HPP