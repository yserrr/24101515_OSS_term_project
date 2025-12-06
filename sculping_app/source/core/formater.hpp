//
// Created by ljh on 25. 9. 18..
//

#ifndef MYPROJECT_FORMATER_HPP
#define MYPROJECT_FORMATER_HPP
#include <spdlog/spdlog.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/details/log_msg.h>
#include <spdlog/fmt/fmt.h>

class UIFormatter : public spdlog::formatter{
public:
  void format(const spdlog::details::log_msg &msg, spdlog::memory_buf_t &dest) override;

  std::unique_ptr<spdlog::formatter> clone() const override;
};
#endif //MYPROJECT_FORMATER_HPP