#include <cstdint>
#include <ctime>
#include "v_log.hpp"
#include "ggml-impl.hpp"


#if defined(_MSC_VER) || defined(__MINGW32__)
static int64_t timer_freq, timer_start;

void v_time_init(void) {
  LARGE_INTEGER t;
  QueryPerformanceFrequency(&t);
  timer_freq = t.QuadPart;
  // The multiplication by 1000 or 1000000 below can cause an overflow if timer_freq
  // and the uptime is high enough.
  // We subtract the program start time to reduce the likelihood of that happening.
  QueryPerformanceCounter(&t);
  timer_start = t.QuadPart;
}

int64_t v_time_ms(void) {
  LARGE_INTEGER t;
  QueryPerformanceCounter(&t);
  return ((t.QuadPart - timer_start) * 1000) / timer_freq;
}

int64_t v_time_us(void) {
  LARGE_INTEGER t;
  QueryPerformanceCounter(&t);
  return ((t.QuadPart - timer_start) * 1000000) / timer_freq;
}
#else
void v_time_init(void) {}
int64_t v_time_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000 + (int64_t)ts.tv_nsec / 1000000;
}

int64_t v_time_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000 + (int64_t)ts.tv_nsec / 1000;
}
#endif

int64_t v_cycles(void) {
  return std::clock();
}

int64_t v_cycles_per_ms(void) {
  return CLOCKS_PER_SEC / 1000;
}

const char* v_status_to_string(v_status status) {
  switch (status) {
    case V_STATUS_ALLOC_FAILED: return "V status: error (failed to allocate memory)";
    case V_STATUS_FAILED: return "V status: error (operation failed)";
    case V_STATUS_SUCCESS: return "V status: success";
    case V_STATUS_ABORTED: return "V status: warning (operation aborted)";
  }
  return "V status: unknown";
}
static v_abort_callback_t g_abort_callback = nullptr;
// Set the abort callback (passing null will restore original abort functionality: printing a message to stdout)
V_API v_abort_callback_t v_set_abort_callback(v_abort_callback_t callback) {
  v_abort_callback_t ret_val = g_abort_callback;
  g_abort_callback           = callback;
  return ret_val;
}

void v_abort(const char* file, int line, const char* fmt, ...) {
  fflush(stdout);

  char message[2048];
  int offset = snprintf(message, sizeof(message), "%s:%d: ", file, line);

  va_list args;
  va_start(args, fmt);
  vsnprintf(message + offset, sizeof(message) - offset, fmt, args);
  va_end(args);

  if (g_abort_callback) {
    g_abort_callback(message);
  } else {
    // default: print error and backtrace to stderr
    fprintf(stderr, "%s\n", message);
    v_print_backtrace();
  }

  abort();
}

// v_print_backtrace is registered with std::set_terminate by ggml.cpp

//
// logging
//


static struct v_logger_state g_logger_state = {v_log_callback_default, nullptr};

static void v_log_internal_v(enum v_log_level level, const char* format, va_list args) {
  if (format == nullptr) {
    return;
  }
  va_list args_copy;
  va_copy(args_copy, args);
  char buffer[128];
  int len = vsnprintf(buffer, 128, format, args);
  if (len < 128) {
    g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
  } else {
    char* buffer2 = (char*)calloc(len + 1, sizeof(char));
    vsnprintf(buffer2, len + 1, format, args_copy);
    buffer2[len] = 0;
    g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
    free(buffer2);
  }
  va_end(args_copy);
}

void v_log_internal(v_log_level level, const char* format, ...) {
  va_list args;
  va_start(args, format);
  v_log_internal_v(level, format, args);
  va_end(args);
}

void v_log_callback_default(v_log_level level, const char* text, void* user_data) {
  (void)level;
  (void)user_data;
  fputs(text, stderr);
  fflush(stderr);
}

void set_log(v_log_callback log_callback, void* user_data) {
  g_logger_state.log_callback           = log_callback ? log_callback : v_log_callback_default;
  g_logger_state.log_callback_user_data = user_data;
}