//
// Created by ljh on 25. 9. 18..
//
#include "log_sink.hpp"

namespace sys
{
  void UIFormatter::format(const spdlog::details::log_msg& msg, spdlog::memory_buf_t& dest)
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

  LogSink::LogSink()
  {
    set_formatter(nullptr);
  }

  void LogSink::log(const spdlog::details::log_msg& msg)
  {
    spdlog::memory_buf_t formatted;
    {
      std::lock_guard<std::mutex> lk(mutex_);
      formatter_->format(msg, formatted);
    }
    std::string s(formatted.data(), formatted.size());
    {
      std::lock_guard<std::mutex> lk(mutex_);
      if (buffer_.size() >= max_items_)
      {
        buffer_.pop_front();
      }
      buffer_.push_back(std::move(s));
    }
  }

  void LogSink::flush()
  {
  }

  void LogSink::set_pattern(const std::string& pattern)
  {
  }

  void LogSink::set_formatter(std::unique_ptr<spdlog::formatter> sink_formatter)
  {
    formatter_ = std::make_unique<UIFormatter>();
  }

  std::deque<std::string> LogSink::snapshot()
  {
    std::lock_guard<std::mutex> lk(mutex_);
    return buffer_;
  }

  void LogSink::clear()
  {
    std::lock_guard<std::mutex> lk(mutex_);
    buffer_.clear();
  }

  inline void LogSink::set_max_items(size_t m)
  {
    max_items_ = m;
  }
}
