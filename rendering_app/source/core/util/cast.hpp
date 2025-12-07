//
// Created by dlwog on 25. 10. 23..
//

#ifndef MYPROJECT_CAST_HPP
#define MYPROJECT_CAST_HPP

#include <cstdint>

template <typename T>
T cast(std::uintptr_t ptr)
{
  return reinterpret_cast<T>(ptr);
}

template <typename T>
std::uintptr_t cast(T type)
{
  return reinterpret_cast<std::uintptr_t>(type);
}

//template<typename HandleType>
//void beginRendering(HandleType handle) {
//  // 내부적으로 HandleType에 맞게 처리
//  if constexpr (std::is_same_v<HandleType, VkPipelineLayout>) {
//    vkCmdBindPipelineLayout(cmdBuffer, handle);
//  } else if constexpr (std::is_same_v<HandleType, VkPipeline>) {
//    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, handle);
//  }
//  // 다른 handle 타입 추가 가능
//}
#endif //MYPROJECT_CAST_HPP
