#pragma once
#include <cstdint>
#include <type_traits>

namespace gpu {

  // Enum class 전용 operator|
  template<typename Enum, typename = std::enable_if_t<std::is_enum_v<Enum>>>
  constexpr uint32_t operator|(Enum lhs, Enum rhs) noexcept {
    return static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs);
  }

  // Enum | uint32_t
  template<typename Enum, typename = std::enable_if_t<std::is_enum_v<Enum>>>
  constexpr uint32_t operator|(Enum lhs, uint32_t rhs) noexcept {
    return static_cast<uint32_t>(lhs) | rhs;
  }

  // uint32_t | Enum
  template<typename Enum, typename = std::enable_if_t<std::is_enum_v<Enum>>>
  constexpr uint32_t operator|(uint32_t lhs, Enum rhs) noexcept {
    return lhs | static_cast<uint32_t>(rhs);
  }

  // 필요하면 operator&도 동일하게 정의 가능
  template<typename Enum, typename = std::enable_if_t<std::is_enum_v<Enum>>>
  constexpr uint32_t operator&(Enum lhs, Enum rhs) noexcept {
    return static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs);
  }

  template<typename Enum, typename = std::enable_if_t<std::is_enum_v<Enum>>>
  constexpr uint32_t operator&(Enum lhs, uint32_t rhs) noexcept {
    return static_cast<uint32_t>(lhs) & rhs;
  }

  template<typename Enum, typename = std::enable_if_t<std::is_enum_v<Enum>>>
  constexpr uint32_t operator&(uint32_t lhs, Enum rhs) noexcept {
    return lhs & static_cast<uint32_t>(rhs);
  }

} // namespace gpu
