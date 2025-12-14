
/// All code is adapted from ggml for personal educational purposes.(study, clone coding)
/// Core code under license is sourced from ggml (https://github.com/ggerganov/ggml)

#pragma once

#include "v_header.hpp"
#define V_LOG(...)       v_log_internal(v_LOG_LEVEL_NONE , __VA_ARGS__)
#define V_LOG_WARN(...)  v_log_internal(v_LOG_LEVEL_WARN , __VA_ARGS__)
#define V_LOG_ERROR(...) v_log_internal(v_LOG_LEVEL_ERROR, __VA_ARGS__)
#define V_LOG_DEBUG(...) v_log_internal(v_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define V_LOG_CONT(...)  v_log_internal(v_LOG_LEVEL_CONT , __VA_ARGS__)
#define V_LOG_INFO(...)  v_log_internal(v_LOG_LEVEL_INFO , __VA_ARGS__)
#define V_DEBUG 10

#if (V_DEBUG >= 1)
#define v_PRINT_DEBUG(...) V_LOG_DEBUG(__VA_ARGS__)
#else
#define PRINT_DEBUG(...)
#endif

#if (V_DEBUG >= 5)
#define v_PRINT_DEBUG_5(...) V_LOG_DEBUG(__VA_ARGS__)
#else
#define v_PRINT_DEBUG_5(...)
#endif

#if (V_DEBUG >= 10)
#define v_PRINT_DEBUG_10(...) V_LOG_DEBUG(__VA_ARGS__)
#else
#define v_PRINT_DEBUG_10(...)
#endif
// TODO these functions were sandwiched in the old optimization interface, is there a better place for them?
typedef void (*v_log_callback)(v_log_level level, const char* text, void* user_data);

// Set callback for all future logging events.
// If this is not called, or NULL is supplied, everything is output on stderr.
void set_log(v_log_callback log_callback, void* user_data);
void v_time_init(void);
int64_t v_time_ms(void);
int64_t v_time_us(void);
int64_t v_cycles(void);
int64_t v_cycles_per_ms(void);

struct v_logger_state {
  v_log_callback log_callback;
  void* log_callback_user_data;
};
