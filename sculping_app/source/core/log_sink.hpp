//
// Created by ljh on 25. 9. 18..
//

#ifndef MYPROJECT_LOG_TRACER_HPP
#define MYPROJECT_LOG_TRACER_HPP
// spdlog_sink_capture.h
#include <spdlog/sinks/sink.h>
#include <spdlog/details/null_mutex.h>
#include <spdlog/details/log_msg.h>
#include <spdlog/fmt/bin_to_hex.h>
#include <spdlog/pattern_formatter.h>

#include <mutex>
#include <deque>
#include <string>
#include <memory>

class UILogSink : public spdlog::sinks::sink{
  friend class UIRenderer;

public:
  UILogSink();

  void log(const spdlog::details::log_msg &msg) override;
  void set_pattern(const std::string &pattern) override;
  void set_formatter(std::unique_ptr<spdlog::formatter> sink_formatter) override;
  void flush() override;
  void set_max_items(size_t m);
  std::deque<std::string> snapshot();
  void clear();

private:
  std::deque<std::string> buffer_;
  std::unique_ptr<spdlog::formatter> formatter_;
  std::mutex mutex_;
  size_t max_items_ = 10000;
};

#endif //MYPROJECT_LOG_TRACER_HPP