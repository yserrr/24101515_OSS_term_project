#include "engine.hpp"
#include "spdlog/spdlog.h"
#include <spdlog/sinks/stdout_color_sinks.h>
#ifdef FORCE_UTF8_CONSOLE
#include <windows.h>
struct Utf8ConsoleInit {
  Utf8ConsoleInit() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
  }
} _utf8_console_init;
#endif

int main()
{
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
  auto logger = spdlog::stdout_color_mt("console");
  logger->set_pattern("[%H:%M:%S] [%^%l%$] %v");
  logger->info("Hello from spdlog!");

  // or use as default
  spdlog::set_default_logger(logger);
  spdlog::info("Default logger active");

  spdlog::info("한글 로그 출력 테스트 ");
  Engine engine;
  engine.run();
}
