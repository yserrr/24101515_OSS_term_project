//
// Created by ljh on 25. 9. 18..
//

#include "formater.hpp"

void UIFormatter::format(const spdlog::details::log_msg &msg, spdlog::memory_buf_t &dest)
{
  fmt::format_to(std::back_inserter(dest),
                 "[{}] {}\n",
                 spdlog::level::to_string_view(msg.level),
                 fmt::to_string(msg.payload));
}

std::unique_ptr<spdlog::formatter> UIFormatter::clone() const
{
  return std::make_unique<UIFormatter>();
}