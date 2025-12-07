#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC
#include "v-backend.h"
#include "ggml-impl.h"
#include "v.h"
#include "v_hash.h"
#include "v_tensor.hpp"
// FIXME: required here for quantization functions
#include <mutex>

#include "v_quants.h"
#include "v_util.h"
#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h> // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <signal.h>
#if defined(__gnu_linux__)
#include <syscall.h>
#endif

#if defined(__APPLE__)
#include <unistd.h>
#include <mach/mach.h>
#include <TargetConditionals.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#define UNUSED v_UNUSED

#if defined(_MSC_VER)
#define m512bh(p) p
#define m512i(p) p
#else
#define m512bh(p) (__m512bh)(p)
#define m512i(p) (__m512i)(p)
#endif

#if defined(__linux__) || \
    defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
    (defined(__APPLE__) && !TARGET_OS_TV && !TARGET_OS_WATCH)

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif

#if defined(__ANDROID__)
#include <unwind.h>
#include <dlfcn.h>
#include <stdio.h>

struct backtrace_state {
  void** current;
  void** end;
};

static _Unwind_Reason_Code unwind_callback(struct _Unwind_Context* context, void* arg) {
  struct backtrace_state* state = (struct backtrace_state*)arg;
  uintptr_t pc                  = _Unwind_GetIP(context);
  if (pc) {
    if (state->current == state->end) {
      return _URC_END_OF_STACK;
    }
    else {
      *state->current++ = (void*)pc;
    }
  }
  return _URC_NO_REASON;
}

static void v_print_backtrace_symbols(void) {
  const int max = 100;
  void* buffer[max];

  struct backtrace_state state = {buffer, buffer + max};
  _Unwind_Backtrace(unwind_callback, &state);
  int count = state.current - buffer;

  for (int idx = 0; idx < count; ++idx) {
    const void* addr   = buffer[idx];
    const char* symbol = "";

    Dl_info info;
    if (dladdr(addr, &info) && info.dli_sname) {
      symbol = info.dli_sname;
    }

    fprintf(stderr, "%d: %p %s\n", idx, addr, symbol);
  }
}
#elif defined(__linux__) && defined(__GLIBC__)
#include <execinfo.h>
static void v_print_backtrace_symbols(void) {
  void* trace[100];
  int nptrs = backtrace(trace, sizeof(trace) / sizeof(trace[0]));
  backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#else
static void v_print_backtrace_symbols(void) {
  // platform not supported
}
#endif

void v_print_backtrace(void) {
  const char* v_NO_BACKTRACE = getenv("v_NO_BACKTRACE");
  if (v_NO_BACKTRACE) {
    return;
  }
#if defined(__linux__)
FILE* f        = fopen("/proc/self/status", "r");
size_t size    = 0;
char* line     = NULL;
ssize_t length = 0;
    while ((length = getline(&line, &size, f)) > 0) {
        if (!strncmp(line, "TracerPid:", sizeof("TracerPid:") - 1) &&
            (length != sizeof("TracerPid:\t0\n") - 1 || line[length - 2] != '0')) {
            // Already being debugged, and the breakpoint is the later abort()
            free(line);
            fclose(f);
            return;
        }
    }
free (line);
fclose (f);
int lock[2] = {-1, -1};
(void) !pipe (lock); // Don't start gdb until after PR_SET_PTRACER
#endif
const int parent_pid = getpid();
const int child_pid  = fork();
    if (child_pid<0) { // error
#if defined(__linux__)
close (lock[1]);
close (lock[0]);
#endif
return;
    } else if (child_pid== 0) { // child
        char attach[32];
        snprintf(attach, sizeof(attach), "attach %d", parent_pid);
#if defined(__linux__)
close (lock[1]);
(void) !read (lock[0], lock, 1);
close (lock[0]);
#endif
// try gdb
execlp ("gdb", "gdb", "--batch",
            "-ex", "set style enabled on",
            "-ex", attach,
            "-ex", "bt -frame-info source-and-location",
            "-ex", "detach",
            "-ex", "quit",
(char*) NULL);
// try lldb
execlp ("lldb", "lldb", "--batch",
            "-o", "bt",
            "-o", "quit",
            "-p", &attach [sizeof("attach ") - 1],
(char*) NULL);
// gdb failed, fallback to backtrace_symbols
v_print_backtrace_symbols();
_Exit (0);
    } else { // parent
#if defined(__linux__)
prctl(PR_SET_PTRACER, child_pid);
close (lock[1]);
close (lock[0]);
#endif
waitpid(child_pid, NULL, 0);
    }
}
#else
void v_print_backtrace(void) {
  // platform not supported
}
#endif


#include "ggml-impl.h"
#include <cstdlib>
#include <exception>


static std::terminate_handler previous_terminate_handler;

v_NORETURN static void v_uncaught_exception() {
  v_print_backtrace();
  if (previous_terminate_handler) {
    previous_terminate_handler();
  }
  abort(); // unreachable unless previous_terminate_handler was nullptr
}

static bool v_uncaught_exception_init = [] {
  const char* v_NO_BACKTRACE = getenv("v_NO_BACKTRACE");
  if (v_NO_BACKTRACE) {
    return false;
  }
  const auto prev{std::get_terminate()};
  V_ASSERT(prev != v_uncaught_exception);
  previous_terminate_handler = prev;
  std::set_terminate(v_uncaught_exception);
  return true;
}();


static v_abort_callback_t g_abort_callback = NULL;

// Set the abort callback (passing null will restore original abort functionality: printing a message to stdout)
v_API v_abort_callback_t v_set_abort_callback(v_abort_callback_t callback) {
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
  }
  else {
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

struct v_logger_state {
  v_log_callback log_callback;
  void* log_callback_user_data;
};

static struct v_logger_state g_logger_state = {mml_log_callback_default, NULL};

static void v_log_internal_v(enum v_log_level level, const char* format, va_list args) {
  if (format == NULL) {
    return;
  }
  va_list args_copy;
  va_copy(args_copy, args);
  char buffer[128];
  int len = vsnprintf(buffer, 128, format, args);
  if (len < 128) {
    g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
  }
  else {
    char* buffer2 = (char*)calloc(len + 1, sizeof(char));
    vsnprintf(buffer2, len + 1, format, args_copy);
    buffer2[len] = 0;
    g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
    free(buffer2);
  }
  va_end(args_copy);
}

void v_log_internal(enum v_log_level level, const char* format, ...) {
  va_list args;
  va_start(args, format);
  v_log_internal_v(level, format, args);
  va_end(args);
}

void mml_log_callback_default(enum v_log_level level, const char* text, void* user_data) {
  (void)level;
  (void)user_data;
  fputs(text, stderr);
  fflush(stderr);
}


void* v_aligned_malloc(size_t size) {
  #if defined(__s390x__)
  const int alignment = 256;
  #else
  const int alignment = 64;
  #endif

  #if defined(_MSC_VER) || defined(__MINGW32__)
  return _aligned_malloc(size, alignment);
  #else
  if (size == 0) {
    v_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for v_aligned_malloc!\n");
    return NULL;
  }
  void* aligned_memory = NULL;
  if (result != 0) {
    // Handle allocation failure
    const char* error_desc = "unknown allocation error";
    switch (result) {
      case EINVAL:
        error_desc = "invalid alignment value";
        break;
      case ENOMEM:
        error_desc = "insufficient memory";
        break;
    }
    v_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, size / (1024.0 * 1024.0));
    return NULL;
  }
  return aligned_memory;
  #endif
}

void v_aligned_free(void* ptr, size_t size) {
  v_UNUSED(size);
  _aligned_free(ptr);
}

//todo : check
inline static void* v_malloc(size_t size) {
  if (size == 0) {
    v_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for v_malloc!\n");
    return NULL;
  }
  void* result = malloc(size);
  if (result == NULL) {
    v_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
    v_ABORT("fatal error");
  }
  return result;
}

// calloc
inline static void* v_calloc(size_t num, size_t size) {
  if (num == 0 || size == 0) {
    v_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for v_calloc!\n");
    return NULL;
  }
  void* result = calloc(num, size);
  if (result == NULL) {
    v_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
    v_ABORT("fatal error");
  }
  return result;
}

#define v_MALLOC(size)      v_malloc(size)
#define v_CALLOC(num, size) v_calloc(num, size)

#define v_FREE(ptr) free(ptr)

const char* v_status_to_string(enum v_status status) {
  switch (status) {
    case v_STATUS_ALLOC_FAILED: return "GGML status: error (failed to allocate memory)";
    case v_STATUS_FAILED: return "GGML status: error (operation failed)";
    case v_STATUS_SUCCESS: return "GGML status: success";
    case v_STATUS_ABORTED: return "GGML status: warning (operation aborted)";
  }

  return "GGML status: unknown";
}

float v_fp16_to_fp32(v_fp16_t x) {
  #define v_fp16_to_fp32 do_not_use__v_fp16_to_fp32__in_ggml
  return v_FP16_TO_FP32(x);
}

v_fp16_t v_fp32_to_fp16(float x) {
  #define v_fp32_to_fp16 do_not_use__v_fp32_to_fp16__in_ggml
  return v_FP32_TO_FP16(x);
}

float v_bf16_to_fp32(v_bf16_t x) {
  #define v_bf16_to_fp32 do_not_use__v_bf16_to_fp32__in_ggml
  return v_BF16_TO_FP32(x); // it just left shifts
}

v_bf16_t v_fp32_to_bf16(float x) {
  #define v_fp32_to_bf16 do_not_use__v_fp32_to_bf16__in_ggml
  return v_FP32_TO_BF16(x);
}

void v_fp16_to_fp32_row(const v_fp16_t* x, float* y, int64_t n) {
  for (int64_t i = 0; i < n; i++) {
    y[i] = v_FP16_TO_FP32(x[i]);
  }
}

void v_fp32_to_fp16_row(const float* x, v_fp16_t* y, int64_t n) {
  int i = 0;
  for (; i < n; ++i) {
    y[i] = v_FP32_TO_FP16(x[i]);
  }
}

void v_bf16_to_fp32_row(const v_bf16_t* x, float* y, int64_t n) {
  int i = 0;
  for (; i < n; ++i) {
    y[i] = v_BF16_TO_FP32(x[i]);
  }
}

void v_fp32_to_bf16_row_ref(const float* x, v_bf16_t* y, int64_t n) {
  for (int i = 0; i < n; i++) {
    y[i] = v_compute_fp32_to_bf16(x[i]);
  }
}

void v_fp32_to_bf16_row(const float* x, v_bf16_t* y, int64_t n) {
  int i = 0;
  #if defined(__AVX512BF16__)
  // subnormals are flushed to zero on this platform
  for (; i + 32 <= n; i += 32) {
    _mm512_storeu_si512(
      (__m512i*)(y + i),
      m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16),
                                _mm512_loadu_ps(x + i))));
  }
  #endif
  for (; i < n; i++) {
    y[i] = v_FP32_TO_BF16(x[i]);
  }
}


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
  return clock();
}

int64_t v_cycles_per_ms(void) {
  return CLOCKS_PER_SEC / 1000;
}

//
// cross-platform UTF-8 file paths
//

#ifdef _WIN32
static wchar_t* v_mbstowcs(const char* mbs) {
  int wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, NULL, 0);
  if (!wlen) {
    errno = EINVAL;
    return NULL;
  }

  wchar_t* wbuf = (wchar_t*)v_MALLOC(wlen * sizeof(wchar_t));
  wlen          = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, wbuf, wlen);
  if (!wlen) {
    v_FREE(wbuf);
    errno = EINVAL;
    return NULL;
  }

  return wbuf;
}
#endif

FILE* v_fopen(const char* fname, const char* mode) {
  #ifdef _WIN32
  FILE* file = NULL;
  // convert fname (UTF-8)
  wchar_t* wfname = v_mbstowcs(fname);
  if (wfname) {
    // convert mode (ANSI)
    wchar_t* wmode   = (wchar_t*)v_MALLOC((strlen(mode) + 1) * sizeof(wchar_t));
    wchar_t* wmode_p = wmode;
    do {
      *wmode_p++ = (wchar_t)*mode;
    }
    while (*mode++);

    // open file
    file = _wfopen(wfname, wmode);

    v_FREE(wfname);
    v_FREE(wmode);
  }
  return file;
  #else
  return fopen(fname, mode);
  #endif
}


static_assert(v_OP_COUNT == 90, "v_OP_COUNT != 90");
static_assert(v_OP_POOL_COUNT == 2, "v_OP_POOL_COUNT != 2");
static_assert(v_GLU_OP_COUNT == 6, "v_GLU_OP_COUNT != 6");
static_assert(sizeof(v_object) % v_MEM_ALIGN == 0,
              "v_object size must be a multiple of v_MEM_ALIGN");
static_assert(sizeof(v_tensor) % v_MEM_ALIGN == 0,
              "v_tensor size must be a multiple of v_MEM_ALIGN");


void v_print_objects(const struct v_ctx* ctx) {
  struct v_object* obj = ctx->objects_begin;
  LOG_INFO("%s: objects in context %p:\n", __func__, (const void *) ctx);

  while (obj != NULL) {
    v_print_object(obj);
    obj = obj->next;
  }

  LOG_INFO("%s: --- end ---\n", __func__);
}

size_t v_graph_overhead_custom(size_t size, bool grads) {
  return MML_OBJECT_SIZE + MML_PAD(v_graph_nbyte(size, grads), v_MEM_ALIGN);
}

size_t num_bytes(const v_tensor* tensor) {
  for (int i = 0; i < V_MAX_DIMS; ++i) {
    if (tensor->ne[i] <= 0) {
      return 0;
    }
  }

  size_t nbytes;
  const size_t blck_size = block_size(tensor->type);
  if (blck_size == 1) {
    nbytes = v_type_size(tensor->type);
    for (int i = 0; i < V_MAX_DIMS; ++i) {
      nbytes += (tensor->ne[i] - 1) * tensor->nb[i];
    }
  }
  else {
    nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;
    for (int i = 1; i < V_MAX_DIMS; ++i) {
      nbytes += (tensor->ne[i] - 1) * tensor->nb[i];
    }
  }

  return nbytes;
}

size_t v_nbytes_pad(const v_tensor* tensor) {
  return MML_PAD(num_bytes(tensor), v_MEM_ALIGN);
}


size_t v_row_size(enum v_data_type type, int64_t ne) {
  assert(ne % block_size(type) == 0);
  return v_type_size(type) * ne / block_size(type);
}


const char* v_op_desc(const v_tensor* t) {
  if (t->op == v_OP_UNARY) {
    enum v_unary_op uop = v_get_unary_op(t);
    return v_unary_op_name(uop);
  }
  if (t->op == v_OP_GLU) {
    enum v_glu_op gop = v_get_glu_op(t);
    return v_glu_op_name(gop);
  }
  return v_op_name(t->op);
}


size_t v_tensor_over_head(void) {
  return MML_OBJECT_SIZE + v_TENSOR_SIZE;
}


bool v_is_contiguous_n(const v_tensor* tensor, int n) {
  size_t next_nb = v_type_size(tensor->type);
  if (tensor->ne[0] != block_size(tensor->type) && tensor->nb[0] != next_nb) {
    return false;
  }
  next_nb *= tensor->ne[0] / block_size(tensor->type);
  for (int i = 1; i < V_MAX_DIMS; i++) {
    if (tensor->ne[i] != 1) {
      if (i > n) {
        if (tensor->nb[i] != next_nb) {
          return false;
        }
        next_nb *= tensor->ne[i];
      }
      else {
        // this dimension does not need to be contiguous
        next_nb = tensor->ne[i] * tensor->nb[i];
      }
    }
  }
  return true;
}


bool is_empty(const v_tensor* tensor) {
  for (int i = 0; i < V_MAX_DIMS; ++i) {
    if (tensor->ne[i] == 0) {
      // empty if any dimension has no elements
      return true;
    }
  }
  return false;
}


////////////////////////////////////////////////////////////////////////////////


v_tensor* new_tensor_impl(struct v_ctx* ctx,
                          enum v_data_type type,
                          int n_dims,
                          const int64_t* ne,
                          v_tensor* view_src,
                          size_t view_offs) {
  V_ASSERT(type >= 0 && type < v_TYPE_COUNT);
  V_ASSERT(n_dims >= 1 && n_dims <= V_MAX_DIMS);
  // find the base tensor and absolute offset
  if (view_src != nullptr && view_src->view_src != nullptr) {
    view_offs += view_src->view_offs;
    view_src = view_src->view_src;
  }
  size_t data_size = v_row_size(type, ne[0]);
  for (int i = 1; i < n_dims; i++) {
    data_size *= ne[i];
  }

  V_ASSERT(view_src == NULL || data_size == 0 || data_size + view_offs <= num_bytes(view_src));

  void* data = view_src != NULL
                 ? view_src->data
                 : NULL;
  if (data != NULL) {
    data = (char*)data + view_offs;
  }

  size_t obj_alloc_size = 0;

  if (view_src == NULL && !ctx->no_alloc) {
    // allocate tensor data in the context's memory pool
    obj_alloc_size = data_size;
  }

  struct v_object* const obj_new = v_new_object(ctx, MML_TENSOR, v_TENSOR_SIZE + obj_alloc_size);
  V_ASSERT(obj_new);

  v_tensor* result  = (v_tensor*)((char*)ctx->mem_buffer + obj_new->offs);
  (*result).type    = type;
  (*result).buffer  = NULL;
  (*result).ne[0]   = 1,
    (*result).ne[1] = 1,
    (*result).ne[2] = 1,
    (*result).ne[3] = 1;
  (*result).nb[0]   = 0,
    (*result).nb[1] = 0,
    (*result).nb[2] = 0,
    (*result).nb[3] = 0;
  (*result).op      = v_OP_NONE;
  (*result).flags   = 0,
    //(*result).src = {NULL},
    (*result).view_src  = view_src,
    (*result).view_offs = view_offs,
    (*result).data      = obj_alloc_size > 0
                            ? (void*)(result + 1)
                            : data,
    //(*result).name = {},
    (*result).extra = NULL;
  //(*result).padding = {0};
  std::fill(std::begin((*result).op_params), std::end((*result).op_params), 0);
  std::fill(std::begin((*result).src), std::end((*result).src), nullptr);

  //(*result).src = {NULL},
  for (int i = 0; i < n_dims; i++) {
    result->ne[i] = ne[i];
  }
  result->nb[0] = v_type_size(type);
  result->nb[1] = result->nb[0] * (result->ne[0] / block_size(type));
  for (int i = 2; i < V_MAX_DIMS; i++) {
    result->nb[i] = result->nb[i - 1] * result->ne[i - 1];
  }
  ctx->n_objects++;
  return result;
}


void v_unravel_index(const v_tensor* tensor, int64_t i, int64_t* i0, int64_t* i1, int64_t* i2, int64_t* i3) {
  const int64_t ne2 = tensor->ne[2];
  const int64_t ne1 = tensor->ne[1];
  const int64_t ne0 = tensor->ne[0];

  const int64_t i3_ = (i / (ne2 * ne1 * ne0));
  const int64_t i2_ = (i - i3_ * ne2 * ne1 * ne0) / (ne1 * ne0);
  const int64_t i1_ = (i - i3_ * ne2 * ne1 * ne0 - i2_ * ne1 * ne0) / ne0;
  const int64_t i0_ = (i - i3_ * ne2 * ne1 * ne0 - i2_ * ne1 * ne0 - i1_ * ne0);

  if (i0) {
    *i0 = i0_;
  }
  if (i1) {
    *i1 = i1_;
  }
  if (i2) {
    *i2 = i2_;
  }
  if (i3) {
    *i3 = i3_;
  }
}


v_tensor* v_set_name(v_tensor* tensor, const char* name) {
  size_t i;
  for (i = 0; i < sizeof(tensor->name) - 1 && name[i] != '\0'; i++) {
    tensor->name[i] = name[i];
  }
  tensor->name[i] = '\0';
  return tensor;
}

v_tensor* v_format_name(v_tensor* tensor, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vsnprintf(tensor->name, sizeof(tensor->name), fmt, args);
  va_end(args);
  return tensor;
}

v_tensor* v_tensor_view(struct v_ctx* ctx,
                        v_tensor* src) {
  v_tensor* result = new_tensor_impl(ctx, src->type, V_MAX_DIMS, src->ne, src, 0);
  v_format_name(result, "%s (view)", src->name);
  for (int i = 0; i < V_MAX_DIMS; i++) {
    result->nb[i] = src->nb[i];
  }
  return result;
}


v_tensor* v_acc_imple(struct v_ctx* ctx,
                      v_tensor* a,
                      v_tensor* b,
                      size_t nb1,
                      size_t nb2,
                      size_t nb3,
                      size_t offset,
                      bool inplace) {
  V_ASSERT(nelements(b) <= nelements(a));
  V_ASSERT(v_is_contiguous(a));
  V_ASSERT(a->type == v_TYPE_F32);
  V_ASSERT(b->type == v_TYPE_F32);
  v_tensor* result = inplace
                       ? v_tensor_view(ctx, a)
                       : v_dup_tensor(ctx, a);

  int32_t params[] = {
    static_cast<int32_t>(nb1),
    static_cast<int32_t>(nb2),
    static_cast<int32_t>(nb3),
    static_cast<int32_t>(offset),
    (inplace
       ? 1
       : 0)
  };
  v_set_op_params(result, params, sizeof(params));

  result->op     = v_OP_ACC;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}


void v_mul_mat_set_prec(v_tensor* a,
                        enum v_prec prec) {
  V_ASSERT(a->op == v_OP_MUL_MAT);

  const int32_t prec_i32 = (int32_t)prec;

  v_set_op_params_i32(a, 0, prec_i32);
}


v_tensor* set_impl(struct v_ctx* ctx,
                   v_tensor* a,
                   v_tensor* b,
                   size_t nb1,
                   size_t nb2,
                   size_t nb3,
                   size_t offset,
                   bool inplace) {
  V_ASSERT(nelements(a) >= nelements(b));

  // make a view of the destination
  v_tensor* result = inplace
                       ? v_tensor_view(ctx, a)
                       : v_dup_tensor(ctx, a);

  V_ASSERT(offset < (size_t)(1 << 30));
  int32_t params[] = {
    (int32_t)nb1,
    (int32_t)nb2,
    (int32_t)nb3,
    (int32_t)offset,
    (int32_t)inplace
      ? 1
      : 0
  };
  v_set_op_params(result, params, sizeof(params));

  result->op     = v_OP_SET;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}


v_tensor* v_cpy(struct v_ctx* ctx,
                v_tensor* a,
                v_tensor* b) {
  V_ASSERT(nelements(a) == nelements(b));
  // make a view of the destination
  v_tensor* result = v_tensor_view(ctx, b);
  if (strlen(b->name) > 0) {
    v_format_name(result, "%s (copy of %s)", b->name, a->name);
  }
  else {
    v_format_name(result, "%s (copy)", a->name);
  }
  result->op     = v_OP_CPY;
  result->src[0] = a;
  result->src[1] = b;
  return result;
}


v_API v_tensor* v_cont_1d(
  struct v_ctx* ctx,
  v_tensor* a,
  int64_t ne0) {
  return v_cont_4d(ctx, a, ne0, 1, 1, 1);
}

v_API v_tensor* v_cont_2d(
  struct v_ctx* ctx,
  v_tensor* a,
  int64_t ne0,
  int64_t ne1) {
  return v_cont_4d(ctx, a, ne0, ne1, 1, 1);
}

v_API v_tensor* v_cont_3d(
  struct v_ctx* ctx,
  v_tensor* a,
  int64_t ne0,
  int64_t ne1,
  int64_t ne2) {
  return v_cont_4d(ctx, a, ne0, ne1, ne2, 1);
}

v_tensor* v_cont_4d(
  struct v_ctx* ctx,
  v_tensor* a,
  int64_t ne0,
  int64_t ne1,
  int64_t ne2,
  int64_t ne3) {
  V_ASSERT(nelements(a) == (ne0*ne1*ne2*ne3));

  v_tensor* result = v_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
  v_format_name(result, "%s (cont)", a->name);

  result->op     = v_OP_CONT;
  result->src[0] = a;

  return result;
}

// v_reshape

v_tensor* v_reshape(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b) {
  V_ASSERT(v_is_contiguous(a));
  // as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
  V_ASSERT(nelements(a) == nelements(b));

  v_tensor* result = new_tensor_impl(ctx, a->type, V_MAX_DIMS, b->ne, a, 0);
  v_format_name(result, "%s (reshaped)", a->name);

  result->op     = v_OP_RESHAPE;
  result->src[0] = a;

  return result;
}

v_tensor* v_reshape_1d(
  struct v_ctx* ctx,
  v_tensor* a,
  int64_t ne0) {
  V_ASSERT(v_is_contiguous(a));
  V_ASSERT(nelements(a) == ne0);

  const int64_t ne[1] = {ne0};
  v_tensor* result    = new_tensor_impl(ctx, a->type, 1, ne, a, 0);
  v_format_name(result, "%s (reshaped)", a->name);

  result->op     = v_OP_RESHAPE;
  result->src[0] = a;

  return result;
}

v_tensor* v_reshape_2d(struct v_ctx* ctx,
                       v_tensor* a,
                       int64_t ne0,
                       int64_t ne1) {
  V_ASSERT(v_is_contiguous(a));
  V_ASSERT(nelements(a) == ne0*ne1);

  const int64_t ne[2] = {ne0, ne1};
  v_tensor* result    = new_tensor_impl(ctx, a->type, 2, ne, a, 0);
  v_format_name(result, "%s (reshaped)", a->name);

  result->op     = v_OP_RESHAPE;
  result->src[0] = a;

  return result;
}

v_tensor* v_reshape_3d(
  struct v_ctx* ctx,
  v_tensor* a,
  int64_t ne0,
  int64_t ne1,
  int64_t ne2) {
  V_ASSERT(v_is_contiguous(a));
  V_ASSERT(nelements(a) == ne0*ne1*ne2);

  const int64_t ne[3] = {ne0, ne1, ne2};
  v_tensor* result    = new_tensor_impl(ctx, a->type, 3, ne, a, 0);
  v_format_name(result, "%s (reshaped)", a->name);

  result->op     = v_OP_RESHAPE;
  result->src[0] = a;

  return result;
}

v_tensor* v_reshape_4d(
  struct v_ctx* ctx,
  v_tensor* a,
  int64_t ne0,
  int64_t ne1,
  int64_t ne2,
  int64_t ne3) {
  V_ASSERT(v_is_contiguous(a));
  V_ASSERT(nelements(a) == ne0*ne1*ne2*ne3);

  const int64_t ne[4] = {ne0, ne1, ne2, ne3};
  v_tensor* result    = new_tensor_impl(ctx, a->type, 4, ne, a, 0);
  v_format_name(result, "%s (reshaped)", a->name);

  result->op     = v_OP_RESHAPE;
  result->src[0] = a;

  return result;
}

static v_tensor* v_view_impl(
  struct v_ctx* ctx,
  v_tensor* a,
  int n_dims,
  const int64_t* ne,
  size_t offset) {
  v_tensor* result = new_tensor_impl(ctx, a->type, n_dims, ne, a, offset);
  v_format_name(result, "%s (view)", a->name);
  v_set_op_params(result, &offset, sizeof(offset));
  result->op     = V_OP_VIEW;
  result->src[0] = a;
  return result;
}

v_tensor* v_view_1d(struct v_ctx* ctx,
                    v_tensor* a,
                    int64_t ne0,
                    size_t offset) {
  v_tensor* result = v_view_impl(ctx, a, 1, &ne0, offset);
  return result;
}

v_tensor* v_view_2d(struct v_ctx* ctx,
                    v_tensor* a,
                    int64_t ne0,
                    int64_t ne1,
                    size_t nb1,
                    size_t offset) {
  const int64_t ne[2] = {ne0, ne1};
  v_tensor* result    = v_view_impl(ctx, a, 2, ne, offset);
  result->nb[1]       = nb1;
  result->nb[2]       = result->nb[1] * ne1;
  result->nb[3]       = result->nb[2];
  return result;
}

v_tensor* v_view_3d(struct v_ctx* ctx,
                    v_tensor* a,
                    int64_t ne0,
                    int64_t ne1,
                    int64_t ne2,
                    size_t nb1,
                    size_t nb2,
                    size_t offset) {
  const int64_t ne[3] = {ne0, ne1, ne2};
  v_tensor* result    = v_view_impl(ctx, a, 3, ne, offset);
  result->nb[1]       = nb1;
  result->nb[2]       = nb2;
  result->nb[3]       = result->nb[2] * ne2;
  return result;
}

v_tensor* v_view_4d(struct v_ctx* ctx,
                    v_tensor* a,
                    int64_t ne0,
                    int64_t ne1,
                    int64_t ne2,
                    int64_t ne3,
                    size_t nb1,
                    size_t nb2,
                    size_t nb3,
                    size_t offset) {
  const int64_t ne[4] = {ne0, ne1, ne2, ne3};

  v_tensor* result = v_view_impl(ctx, a, 4, ne, offset);

  result->nb[1] = nb1;
  result->nb[2] = nb2;
  result->nb[3] = nb3;

  return result;
}


static v_tensor* v_diag_mask_inf_impl(
  struct v_ctx* ctx,
  v_tensor* a,
  int n_past,
  bool inplace) {
  v_tensor* result = inplace
                       ? v_tensor_view(ctx, a)
                       : v_dup_tensor(ctx, a);

  int32_t params[] = {n_past};
  v_set_op_params(result, params, sizeof(params));

  result->op     = V_OP_DIAG_MASK_INF;
  result->src[0] = a;

  return result;
}

v_tensor* v_diag_mask_inf(
  struct v_ctx* ctx,
  v_tensor* a,
  int n_past) {
  return v_diag_mask_inf_impl(ctx, a, n_past, false);
}

v_tensor* v_diag_mask_inf_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  int n_past) {
  return v_diag_mask_inf_impl(ctx, a, n_past, true);
}

// v_diag_mask_zero

v_tensor* v_diag_mask_zero_impl(struct v_ctx* ctx,
                                v_tensor* a,
                                int n_past,
                                bool inplace) {
  v_tensor* result = inplace
                       ? v_tensor_view(ctx, a)
                       : v_dup_tensor(ctx, a);

  int32_t params[] = {n_past};
  v_set_op_params(result, params, sizeof(params));

  result->op     = V_OP_DIAG_MASK_ZERO;
  result->src[0] = a;

  return result;
}

v_tensor* v_diag_mask_zero(
  struct v_ctx* ctx,
  v_tensor* a,
  int n_past) {
  return v_diag_mask_zero_impl(ctx, a, n_past, false);
}

v_tensor* v_diag_mask_zero_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  int n_past) {
  return v_diag_mask_zero_impl(ctx, a, n_past, true);
}

// v_soft_max

static v_tensor* v_soft_max_impl(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* mask,
  float scale,
  float max_bias,
  bool inplace) {
  V_ASSERT(v_is_contiguous(a));

  if (mask) {
    V_ASSERT(mask->type == v_TYPE_F16 || mask->type == v_TYPE_F32);
    V_ASSERT(v_is_contiguous(mask));
    V_ASSERT(mask->ne[0] == a->ne[0]);
    V_ASSERT(mask->ne[1] >= a->ne[1]);
    V_ASSERT(a->ne[2]%mask->ne[2] == 0);
    V_ASSERT(a->ne[3]%mask->ne[3] == 0);
  }

  if (max_bias > 0.0f) {
    V_ASSERT(mask);
  }

  v_tensor* result = inplace
                       ? v_tensor_view(ctx, a)
                       : v_dup_tensor(ctx, a);

  float params[] = {scale, max_bias};
  v_set_op_params(result, params, sizeof(params));

  result->op     = V_OP_SOFT_MAX;
  result->src[0] = a;
  result->src[1] = mask;

  return result;
}

v_tensor* v_soft_max(
  struct v_ctx* ctx,
  v_tensor* a) {
  return v_soft_max_impl(ctx, a, NULL, 1.0f, 0.0f, false);
}

v_tensor* v_soft_max_inplace(
  struct v_ctx* ctx,
  v_tensor* a) {
  return v_soft_max_impl(ctx, a, NULL, 1.0f, 0.0f, true);
}

v_tensor* v_soft_max_ext(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* mask,
  float scale,
  float max_bias) {
  return v_soft_max_impl(ctx, a, mask, scale, max_bias, false);
}

v_tensor* v_soft_max_ext_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* mask,
  float scale,
  float max_bias) {
  return v_soft_max_impl(ctx, a, mask, scale, max_bias, true);
}

void v_soft_max_add_sinks(
  v_tensor* a,
  v_tensor* sinks) {
  if (!sinks) {
    a->src[2] = NULL;
    return;
  }

  V_ASSERT(a->op == V_OP_SOFT_MAX);
  V_ASSERT(a->src[2] == NULL);
  V_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
  V_ASSERT(sinks->type == v_TYPE_F32);

  a->src[2] = sinks;
}

// v_soft_max_ext_back

static v_tensor* v_soft_max_ext_back_impl(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  float scale,
  float max_bias,
  bool inplace) {
  v_tensor* result = inplace
                       ? v_tensor_view(ctx, a)
                       : v_dup_tensor(ctx, a);

  result->op     = v_OP_SOFT_MAX_BACK;
  result->src[0] = a;
  result->src[1] = b;

  memcpy((float*)result->op_params + 0, &scale, sizeof(float));
  memcpy((float*)result->op_params + 1, &max_bias, sizeof(float));

  return result;
}

v_tensor* v_soft_max_ext_back(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  float scale,
  float max_bias) {
  return v_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, false);
}

v_tensor* v_soft_max_ext_back_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  float scale,
  float max_bias) {
  return v_soft_max_ext_back_impl(ctx, a, b, scale, max_bias, true);
}

// v_rope

static v_tensor* v_rope_impl(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  v_tensor* c,
  int n_dims,
  int sections[v_MROPE_SECTIONS],
  int mode,
  int n_ctx_orig,
  float freq_base,
  float freq_scale,
  float ext_factor,
  float attn_factor,
  float beta_fast,
  float beta_slow,
  bool inplace) {
  V_ASSERT((mode & 1) == 0 && "mode & 1 == 1 is no longer supported");

  V_ASSERT(v_is_vector(b));
  V_ASSERT(b->type == v_TYPE_I32);

  bool mrope_used = mode & v_ROPE_TYPE_MROPE;
  if (mrope_used) {
    V_ASSERT(a->ne[2] * 4 == b->ne[0]); // mrope expecting 4 position ids per token
  }
  else {
    V_ASSERT(a->ne[2] == b->ne[0]);
  }

  if (c) {
    V_ASSERT(c->type == v_TYPE_F32);
    V_ASSERT(c->ne[0] >= n_dims / 2);
  }

  v_tensor* result = inplace
                       ? v_tensor_view(ctx, a)
                       : v_dup_tensor(ctx, a);

  int32_t params[15] = {
    /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig
  };
  memcpy(params + 5, &freq_base, sizeof(float));
  memcpy(params + 6, &freq_scale, sizeof(float));
  memcpy(params + 7, &ext_factor, sizeof(float));
  memcpy(params + 8, &attn_factor, sizeof(float));
  memcpy(params + 9, &beta_fast, sizeof(float));
  memcpy(params + 10, &beta_slow, sizeof(float));
  if (mrope_used && sections) {
    memcpy(params + 11, sections, sizeof(int32_t) * v_MROPE_SECTIONS);
  }
  else {
    memset(params + 11, 0, sizeof(int32_t) * v_MROPE_SECTIONS);
  }
  v_set_op_params(result, params, sizeof(params));

  result->op     = V_OP_ROPE;
  result->src[0] = a;
  result->src[1] = b;
  result->src[2] = c;

  return result;
}

v_tensor* v_rope(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  int n_dims,
  int mode) {
  return v_rope_impl(
    ctx,
    a,
    b,
    NULL,
    n_dims,
    NULL,
    mode,
    0,
    10000.0f,
    1.0f,
    0.0f,
    1.0f,
    0.0f,
    0.0f,
    false
  );
}

v_tensor* v_rope_multi(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  v_tensor* c,
  int n_dims,
  int sections[v_MROPE_SECTIONS],
  int mode,
  int n_ctx_orig,
  float freq_base,
  float freq_scale,
  float ext_factor,
  float attn_factor,
  float beta_fast,
  float beta_slow) {
  return v_rope_impl(
    ctx,
    a,
    b,
    c,
    n_dims,
    sections,
    mode,
    n_ctx_orig,
    freq_base,
    freq_scale,
    ext_factor,
    attn_factor,
    beta_fast,
    beta_slow,
    false
  );
}

v_tensor* v_rope_multi_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  v_tensor* c,
  int n_dims,
  int sections[v_MROPE_SECTIONS],
  int mode,
  int n_ctx_orig,
  float freq_base,
  float freq_scale,
  float ext_factor,
  float attn_factor,
  float beta_fast,
  float beta_slow) {
  return v_rope_impl(
    ctx,
    a,
    b,
    c,
    n_dims,
    sections,
    mode,
    n_ctx_orig,
    freq_base,
    freq_scale,
    ext_factor,
    attn_factor,
    beta_fast,
    beta_slow,
    true
  );
}

v_tensor* v_rope_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  int n_dims,
  int mode) {
  return v_rope_impl(
    ctx,
    a,
    b,
    NULL,
    n_dims,
    NULL,
    mode,
    0,
    10000.0f,
    1.0f,
    0.0f,
    1.0f,
    0.0f,
    0.0f,
    true
  );
}

v_tensor* v_rope_ext(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  v_tensor* c,
  int n_dims,
  int mode,
  int n_ctx_orig,
  float freq_base,
  float freq_scale,
  float ext_factor,
  float attn_factor,
  float beta_fast,
  float beta_slow) {
  return v_rope_impl(
    ctx,
    a,
    b,
    c,
    n_dims,
    NULL,
    mode,
    n_ctx_orig,
    freq_base,
    freq_scale,
    ext_factor,
    attn_factor,
    beta_fast,
    beta_slow,
    false
  );
}

v_tensor* v_rope_ext_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  v_tensor* c,
  int n_dims,
  int mode,
  int n_ctx_orig,
  float freq_base,
  float freq_scale,
  float ext_factor,
  float attn_factor,
  float beta_fast,
  float beta_slow) {
  return v_rope_impl(
    ctx,
    a,
    b,
    c,
    n_dims,
    NULL,
    mode,
    n_ctx_orig,
    freq_base,
    freq_scale,
    ext_factor,
    attn_factor,
    beta_fast,
    beta_slow,
    true
  );
}

v_tensor* v_rope_custom(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  int n_dims,
  int mode,
  int n_ctx_orig,
  float freq_base,
  float freq_scale,
  float ext_factor,
  float attn_factor,
  float beta_fast,
  float beta_slow) {
  return v_rope_impl(
    ctx,
    a,
    b,
    NULL,
    n_dims,
    NULL,
    mode,
    n_ctx_orig,
    freq_base,
    freq_scale,
    ext_factor,
    attn_factor,
    beta_fast,
    beta_slow,
    false
  );
}

v_tensor* v_rope_custom_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  int n_dims,
  int mode,
  int n_ctx_orig,
  float freq_base,
  float freq_scale,
  float ext_factor,
  float attn_factor,
  float beta_fast,
  float beta_slow) {
  return v_rope_impl(
    ctx,
    a,
    b,
    NULL,
    n_dims,
    NULL,
    mode,
    n_ctx_orig,
    freq_base,
    freq_scale,
    ext_factor,
    attn_factor,
    beta_fast,
    beta_slow,
    true
  );
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
static float v_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
  return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float)M_PI)) / (2 * logf(base));
}

void v_rope_yarn_corr_dims(
  int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
  // start and end correction dims
  float start = floorf(v_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
  float end   = ceilf(v_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
  dims[0]     = MAX(0, start);
  dims[1]     = MIN(n_dims - 1, end);
}

// v_rope_back

v_tensor* v_rope_ext_back(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  v_tensor* c,
  int n_dims,
  int mode,
  int n_ctx_orig,
  float freq_base,
  float freq_scale,
  float ext_factor,
  float attn_factor,
  float beta_fast,
  float beta_slow) {
  v_tensor* result = v_rope_ext(
    ctx,
    a,
    b,
    c,
    n_dims,
    mode,
    n_ctx_orig,
    freq_base,
    freq_scale,
    ext_factor,
    attn_factor,
    beta_fast,
    beta_slow);
  result->op = v_OP_ROPE_BACK;
  return result;
}

v_tensor* v_rope_multi_back(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  v_tensor* c,
  int n_dims,
  int sections[4],
  int mode,
  int n_ctx_orig,
  float freq_base,
  float freq_scale,
  float ext_factor,
  float attn_factor,
  float beta_fast,
  float beta_slow) {
  v_tensor* result = v_rope_multi(
    ctx,
    a,
    b,
    c,
    n_dims,
    sections,
    mode,
    n_ctx_orig,
    freq_base,
    freq_scale,
    ext_factor,
    attn_factor,
    beta_fast,
    beta_slow);
  result->op = v_OP_ROPE_BACK;
  return result;
}

// v_clamp

v_tensor* v_clamp(
  struct v_ctx* ctx,
  v_tensor* a,
  float min,
  float max) {
  // TODO: when implement backward, fix this:
  v_tensor* result = v_tensor_view(ctx, a);

  float params[] = {min, max};
  v_set_op_params(result, params, sizeof(params));

  result->op     = v_OP_CLAMP;
  result->src[0] = a;

  return result;
}


// v_pad

v_tensor* v_pad(struct v_ctx* ctx,
                v_tensor* a,
                int p0,
                int p1,
                int p2,
                int p3) {
  return v_pad_ext(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

v_tensor* v_pad_ext(
  struct v_ctx* ctx,
  v_tensor* a,
  int lp0,
  int rp0,
  int lp1,
  int rp1,
  int lp2,
  int rp2,
  int lp3,
  int rp3
) {
  v_tensor* result = v_new_tensor_4d(ctx,
                                     a->type,
                                     a->ne[0] + lp0 + rp0,
                                     a->ne[1] + lp1 + rp1,
                                     a->ne[2] + lp2 + rp2,
                                     a->ne[3] + lp3 + rp3);

  v_set_op_params_i32(result, 0, lp0);
  v_set_op_params_i32(result, 1, rp0);
  v_set_op_params_i32(result, 2, lp1);
  v_set_op_params_i32(result, 3, rp1);
  v_set_op_params_i32(result, 4, lp2);
  v_set_op_params_i32(result, 5, rp2);
  v_set_op_params_i32(result, 6, lp3);
  v_set_op_params_i32(result, 7, rp3);


  result->op     = v_OP_PAD;
  result->src[0] = a;

  return result;
}

// v_pad_reflect_1d
v_tensor* v_pad_reflect_1d(struct v_ctx* ctx,
                           v_tensor* a,
                           int p0,
                           int p1) {
  V_ASSERT(p0 >= 0);
  V_ASSERT(p1 >= 0);

  V_ASSERT(p0 < a->ne[0]); // padding length on each size must be less than the
  V_ASSERT(p1 < a->ne[0]); // existing length of the dimension being padded

  V_ASSERT(v_is_contiguous(a));
  V_ASSERT(a->type == v_TYPE_F32);

  v_tensor* result = v_new_tensor_4d(ctx,
                                     a->type,
                                     a->ne[0] + p0 + p1,
                                     a->ne[1],
                                     a->ne[2],
                                     a->ne[3]);

  int32_t params[] = {p0, p1};
  v_set_op_params(result, params, sizeof(params));

  result->op     = v_OP_PAD_REFLECT_1D;
  result->src[0] = a;

  return result;
}

// v_roll

v_tensor* v_roll(struct v_ctx* ctx,
                 v_tensor* a,
                 int shift0,
                 int shift1,
                 int shift2,
                 int shift3) {
  V_ASSERT(a->nb[0] == v_type_size(a->type));
  V_ASSERT(abs(shift0) < a->ne[0]);
  V_ASSERT(abs(shift1) < a->ne[1]);
  V_ASSERT(abs(shift2) < a->ne[2]);
  V_ASSERT(abs(shift3) < a->ne[3]);

  v_tensor* result = v_dup_tensor(ctx, a);

  v_set_op_params_i32(result, 0, shift0);
  v_set_op_params_i32(result, 1, shift1);
  v_set_op_params_i32(result, 2, shift2);
  v_set_op_params_i32(result, 3, shift3);

  result->op     = v_OP_ROLL;
  result->src[0] = a;

  return result;
}

// v_arange

v_tensor* v_arange(struct v_ctx* ctx,
                   float start,
                   float stop,
                   float step) {
  V_ASSERT(stop > start);

  const int64_t steps = (int64_t)ceilf((stop - start) / step);

  v_tensor* result = v_new_tensor_1d(ctx, v_TYPE_F32, steps);

  v_set_op_params_f32(result, 0, start);
  v_set_op_params_f32(result, 1, stop);
  v_set_op_params_f32(result, 2, step);

  result->op = v_OP_ARANGE;

  return result;
}

// v_timestep_embedding

v_tensor* v_timestep_embedding(struct v_ctx* ctx,
                               v_tensor* timesteps,
                               int dim,
                               int max_period) {
  v_tensor* result = v_new_tensor_2d(ctx, v_TYPE_F32, dim, timesteps->ne[0]);

  v_set_op_params_i32(result, 0, dim);
  v_set_op_params_i32(result, 1, max_period);

  result->op     = v_OP_TIMESTEP_EMBEDDING;
  result->src[0] = timesteps;

  return result;
}

// v_argsort
v_tensor* v_argsort(struct v_ctx* ctx,
                    v_tensor* a,
                    enum v_sort_order order) {
  V_ASSERT(a->ne[0] <= INT32_MAX);
  v_tensor* result = v_new_tensor(ctx, v_TYPE_I32, V_MAX_DIMS, a->ne);
  v_set_op_params_i32(result, 0, (int32_t)order);
  result->op     = v_OP_ARGSORT;
  result->src[0] = a;
  return result;
}

// v_top_k
v_tensor* v_top_k(struct v_ctx* ctx,
                  v_tensor* a,
                  int k) {
  V_ASSERT(a->ne[0] >= k);

  v_tensor* result = v_argsort(ctx, a, v_SORT_ORDER_DESC);
  result           = v_view_4d(ctx,
                               result,
                               k,
                               result->ne[1],
                               result->ne[2],
                               result->ne[3],
                               result->nb[1],
                               result->nb[2],
                               result->nb[3],
                               0);
  return result;
}

// v_flash_attn_ext

v_tensor* v_flash_attn_ext(struct v_ctx* ctx,
                           v_tensor* q,
                           v_tensor* k,
                           v_tensor* v,
                           v_tensor* mask,
                           float scale,
                           float max_bias,
                           float logit_softcap) {
  V_ASSERT(can_mul_mat(k, q));
  // TODO: check if vT can be multiplied by (k*qT)

  V_ASSERT(q->ne[3] == k->ne[3]);
  V_ASSERT(q->ne[3] == v->ne[3]);

  if (mask) {
    V_ASSERT(v_is_contiguous(mask));
    V_ASSERT(mask->ne[1] >= MML_PAD(q->ne[1], v_KQ_MASK_PAD) &&
      "the Flash-Attention kernel requires the mask to be padded to v_KQ_MASK_PAD and at least n_queries big");
    //v_ASSERT(v_can_repeat_rows(mask, qk));

    V_ASSERT(q->ne[2] % mask->ne[2] == 0);
    V_ASSERT(q->ne[3] % mask->ne[3] == 0);
  }

  if (max_bias > 0.0f) {
    V_ASSERT(mask);
  }

  // permute(0, 2, 1, 3)
  int64_t ne[4]    = {v->ne[0], q->ne[2], q->ne[1], q->ne[3]};
  v_tensor* result = v_new_tensor(ctx, v_TYPE_F32, 4, ne);

  float params[] = {scale, max_bias, logit_softcap};
  v_set_op_params(result, params, sizeof(params));

  result->op     = v_OP_FLASH_ATTN_EXT;
  result->src[0] = q;
  result->src[1] = k;
  result->src[2] = v;
  result->src[3] = mask;

  return result;
}

void v_flash_attn_ext_set_prec(
  v_tensor* a,
  enum v_prec prec) {
  V_ASSERT(a->op == v_OP_FLASH_ATTN_EXT);

  const int32_t prec_i32 = (int32_t)prec;

  v_set_op_params_i32(a, 3, prec_i32); // scale is on first pos, max_bias on second
}

enum v_prec v_flash_attn_ext_get_prec(
  const v_tensor* a) {
  V_ASSERT(a->op == v_OP_FLASH_ATTN_EXT);

  const int32_t prec_i32 = v_get_op_params_i32(a, 3);

  return (enum v_prec)prec_i32;
}

void v_flash_attn_ext_add_sinks(
  v_tensor* a,
  v_tensor* sinks) {
  if (!sinks) {
    a->src[4] = NULL;
    return;
  }
  V_ASSERT(a->op == v_OP_FLASH_ATTN_EXT);
  V_ASSERT(a->src[4] == NULL);
  V_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
  V_ASSERT(sinks->type == v_TYPE_F32);
  a->src[4] = sinks;
}

// v_flash_attn_back
v_tensor* v_flash_attn_back(struct v_ctx* ctx,
                            v_tensor* q,
                            v_tensor* k,
                            v_tensor* v,
                            v_tensor* d,
                            bool masked) {
  v_ABORT("TODO: adapt to v_flash_attn_ext() changes");

  V_ASSERT(can_mul_mat(k, q));
  // TODO: check if vT can be multiplied by (k*qT)

  // d shape [D,N,ne2,ne3]
  // q shape [D,N,ne2,ne3]
  // k shape [D,M,kvne2,ne3]
  // v shape [M,D,kvne2,ne3]

  const int64_t D     = q->ne[0];
  const int64_t N     = q->ne[1];
  const int64_t M     = k->ne[1];
  const int64_t ne2   = q->ne[2];
  const int64_t ne3   = q->ne[3];
  const int64_t kvne2 = k->ne[2];
  V_ASSERT(k->ne[0] == D);
  V_ASSERT(v->ne[0] == M);
  V_ASSERT(v->ne[1] == D);
  V_ASSERT(d->ne[0] == D);
  V_ASSERT(d->ne[1] == N);
  V_ASSERT(k->ne[2] == kvne2);
  V_ASSERT(k->ne[3] == ne3);
  V_ASSERT(v->ne[2] == kvne2);
  V_ASSERT(v->ne[3] == ne3);
  V_ASSERT(d->ne[2] == ne2);
  V_ASSERT(d->ne[3] == ne3);

  V_ASSERT(ne2 % kvne2 == 0);

  // store gradients of q, k and v as continuous tensors concatenated in result.
  // note: v and gradv are actually transposed, i.e. v->ne[0] != D.
  const int64_t elem_q = nelements(q);
  const int64_t elem_k = nelements(k);
  const int64_t elem_v = nelements(v);

  enum v_data_type result_type = v_TYPE_F32;
  V_ASSERT(block_size(result_type) == 1);
  const size_t tsize = v_type_size(result_type);

  const size_t offs_q = 0;
  const size_t offs_k = offs_q + MML_PAD(elem_q * tsize, v_MEM_ALIGN);
  const size_t offs_v = offs_k + MML_PAD(elem_k * tsize, v_MEM_ALIGN);
  const size_t end    = offs_v + MML_PAD(elem_v * tsize, v_MEM_ALIGN);

  const size_t nelements = (end + tsize - 1) / tsize;

  v_tensor* result = v_new_tensor_1d(ctx, v_TYPE_F32, nelements);

  int32_t masked_i = masked
                       ? 1
                       : 0;
  v_set_op_params(result, &masked_i, sizeof(masked_i));

  result->op     = v_OP_FLASH_ATTN_BACK;
  result->src[0] = q;
  result->src[1] = k;
  result->src[2] = v;
  result->src[3] = d;

  return result;
}

// v_ssm_conv

v_tensor* v_ssm_conv(
  struct v_ctx* ctx,
  v_tensor* sx,
  v_tensor* c) {
  V_ASSERT(v_is_3d(sx));
  V_ASSERT(v_is_matrix(c));

  const int64_t d_conv  = c->ne[0];
  const int64_t d_inner = c->ne[1];
  const int64_t n_t     = sx->ne[0] - d_conv + 1; // tokens per sequence
  const int64_t n_s     = sx->ne[2];

  // TODO: maybe support other strides than 1?
  V_ASSERT(sx->ne[0] == d_conv - 1 + n_t);
  V_ASSERT(sx->ne[1] == d_inner);
  V_ASSERT(n_t >= 0);

  v_tensor* result = v_new_tensor_3d(ctx, v_TYPE_F32, d_inner, n_t, n_s);

  result->op     = v_OP_SSM_CONV;
  result->src[0] = sx;
  result->src[1] = c;

  return result;
}

// v_ssm_scan

v_tensor* v_ssm_scan(
  struct v_ctx* ctx,
  v_tensor* s,
  v_tensor* x,
  v_tensor* dt,
  v_tensor* A,
  v_tensor* B,
  v_tensor* C,
  v_tensor* ids) {
  V_ASSERT(v_is_contiguous(s));
  V_ASSERT(v_is_contiguous(dt));
  V_ASSERT(v_is_contiguous(A));
  V_ASSERT(x->nb[0] == v_type_size(x->type));
  V_ASSERT(B->nb[0] == v_type_size(B->type));
  V_ASSERT(C->nb[0] == v_type_size(C->type));
  V_ASSERT(x->nb[1] == x->ne[0]*x->nb[0]);
  V_ASSERT(B->nb[1] == B->ne[0]*B->nb[0]);
  V_ASSERT(C->nb[1] == C->ne[0]*C->nb[0]);
  V_ASSERT(v_are_same_shape(B, C));
  V_ASSERT(ids->type == v_TYPE_I32);

  {
    const int64_t d_state      = s->ne[0];
    const int64_t head_dim     = x->ne[0];
    const int64_t n_head       = x->ne[1];
    const int64_t n_seq_tokens = x->ne[2];
    const int64_t n_seqs       = x->ne[3];

    V_ASSERT(dt->ne[0] == n_head);
    V_ASSERT(dt->ne[1] == n_seq_tokens);
    V_ASSERT(dt->ne[2] == n_seqs);
    V_ASSERT(v_is_3d(dt));
    V_ASSERT(s->ne[1] == head_dim);
    V_ASSERT(s->ne[2] == n_head);
    V_ASSERT(B->ne[0] == d_state);
    V_ASSERT(B->ne[2] == n_seq_tokens);
    V_ASSERT(B->ne[3] == n_seqs);
    V_ASSERT(ids->ne[0] == n_seqs);
    V_ASSERT(v_is_vector(ids));
    V_ASSERT(A->ne[1] == n_head);
    V_ASSERT(v_is_matrix(A));

    if (A->ne[0] != 1) {
      // Mamba-1 has more granular decay factors
      V_ASSERT(A->ne[0] == d_state);
    }
  }

  // concatenated y + ssm_states
  v_tensor* result = v_new_tensor_1d(ctx,
                                     v_TYPE_F32,
                                     nelements(x) + s->ne[0] * s->ne[1] * s->ne[2] * ids->ne[0]);

  result->op     = v_OP_SSM_SCAN;
  result->src[0] = s;
  result->src[1] = x;
  result->src[2] = dt;
  result->src[3] = A;
  result->src[4] = B;
  result->src[5] = C;
  result->src[6] = ids;

  return result;
}

// v_win_part
v_tensor* v_win_part(
  struct v_ctx* ctx,
  v_tensor* a,
  int w) {
  V_ASSERT(a->ne[3] == 1);
  V_ASSERT(a->type == v_TYPE_F32);

  // padding
  const int px = (w - a->ne[1] % w) % w;
  const int py = (w - a->ne[2] % w) % w;

  const int npx = (px + a->ne[1]) / w;
  const int npy = (py + a->ne[2]) / w;
  const int np  = npx * npy;

  const int64_t ne[4] = {a->ne[0], w, w, np,};
  v_tensor* result    = v_new_tensor(ctx, v_TYPE_F32, 4, ne);

  int32_t params[] = {npx, npy, w};
  v_set_op_params(result, params, sizeof(params));

  result->op     = v_OP_WIN_PART;
  result->src[0] = a;

  return result;
}

// v_win_unpart

v_tensor* v_win_unpart(
  struct v_ctx* ctx,
  v_tensor* a,
  int w0,
  int h0,
  int w) {
  V_ASSERT(a->type == v_TYPE_F32);

  const int64_t ne[4] = {a->ne[0], w0, h0, 1,};
  v_tensor* result    = v_new_tensor(ctx, v_TYPE_F32, 3, ne);

  int32_t params[] = {w};
  v_set_op_params(result, params, sizeof(params));

  result->op     = v_OP_WIN_UNPART;
  result->src[0] = a;

  return result;
}

// v_get_rel_pos

v_tensor* v_get_rel_pos(
  struct v_ctx* ctx,
  v_tensor* a,
  int qh,
  int kh) {
  V_ASSERT(qh == kh);
  V_ASSERT(2*MAX(qh, kh) - 1 == a->ne[1]);

  const int64_t ne[4] = {a->ne[0], kh, qh, 1,};
  v_tensor* result    = v_new_tensor(ctx, v_TYPE_F16, 3, ne);

  result->op     = v_OP_GET_REL_POS;
  result->src[0] = a;

  return result;
}

// v_add_rel_pos

static v_tensor* v_add_rel_pos_impl(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* pw,
  v_tensor* ph,
  bool inplace) {
  V_ASSERT(v_are_same_shape(pw, ph));
  V_ASSERT(v_is_contiguous(a));
  V_ASSERT(v_is_contiguous(pw));
  V_ASSERT(v_is_contiguous(ph));
  V_ASSERT(ph->type == v_TYPE_F32);
  V_ASSERT(pw->type == v_TYPE_F32);
  V_ASSERT(pw->ne[3] == a->ne[2]);
  V_ASSERT(pw->ne[0]*pw->ne[0] == a->ne[0]);
  V_ASSERT(pw->ne[1]*pw->ne[2] == a->ne[1]);

  v_tensor* result = inplace
                       ? v_tensor_view(ctx, a)
                       : v_dup_tensor(ctx, a);
  v_set_op_params_i32(result, 0, inplace
                                   ? 1
                                   : 0);

  result->op     = v_OP_ADD_REL_POS;
  result->src[0] = a;
  result->src[1] = pw;
  result->src[2] = ph;

  return result;
}

v_tensor* v_add_rel_pos(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* pw,
  v_tensor* ph) {
  return v_add_rel_pos_impl(ctx, a, pw, ph, false);
}

v_tensor* v_add_rel_pos_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* pw,
  v_tensor* ph) {
  return v_add_rel_pos_impl(ctx, a, pw, ph, true);
}

// v_rwkv_wkv6
v_tensor* v_rwkv_wkv6(
  struct v_ctx* ctx,
  v_tensor* k,
  v_tensor* v,
  v_tensor* r,
  v_tensor* tf,
  v_tensor* td,
  v_tensor* state) {
  V_ASSERT(v_is_contiguous(k));
  V_ASSERT(v_is_contiguous(v));
  V_ASSERT(v_is_contiguous(r));
  V_ASSERT(v_is_contiguous(tf));
  V_ASSERT(v_is_contiguous(td));
  V_ASSERT(v_is_contiguous(state));

  const int64_t S        = k->ne[0];
  const int64_t H        = k->ne[1];
  const int64_t n_tokens = k->ne[2];
  const int64_t n_seqs   = state->ne[1];
  {
    V_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
    V_ASSERT(r->ne[0] == S && r->ne[1] == H && r->ne[2] == n_tokens);
    V_ASSERT(td->ne[0] == S && td->ne[1] == H && td->ne[2] == n_tokens);
    V_ASSERT(nelements(state) == S * S * H * n_seqs);
  }

  // concat output and new_state
  const int64_t ne[4] = {S * H, n_tokens + S * n_seqs, 1, 1};
  v_tensor* result    = v_new_tensor(ctx, v_TYPE_F32, 4, ne);

  result->op     = v_OP_RWKV_WKV6;
  result->src[0] = k;
  result->src[1] = v;
  result->src[2] = r;
  result->src[3] = tf;
  result->src[4] = td;
  result->src[5] = state;

  return result;
}

// v_gated_linear_attn

v_tensor* v_gated_linear_attn(
  struct v_ctx* ctx,
  v_tensor* k,
  v_tensor* v,
  v_tensor* q,
  v_tensor* g,
  v_tensor* state,
  float scale) {
  V_ASSERT(v_is_contiguous(k));
  V_ASSERT(v_is_contiguous(v));
  V_ASSERT(v_is_contiguous(q));
  V_ASSERT(v_is_contiguous(g));
  V_ASSERT(v_is_contiguous(state));

  const int64_t S        = k->ne[0];
  const int64_t H        = k->ne[1];
  const int64_t n_tokens = k->ne[2];
  const int64_t n_seqs   = state->ne[1];
  {
    V_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
    V_ASSERT(q->ne[0] == S && q->ne[1] == H && q->ne[2] == n_tokens);
    V_ASSERT(g->ne[0] == S && g->ne[1] == H && g->ne[2] == n_tokens);
    V_ASSERT(nelements(state) == S * S * H * n_seqs);
  }

  // concat output and new_state
  const int64_t ne[4] = {S * H, n_tokens + S * n_seqs, 1, 1};
  v_tensor* result    = v_new_tensor(ctx, v_TYPE_F32, 4, ne);
  v_set_op_params_f32(result, 0, scale);
  result->op     = v_OP_GATED_LINEAR_ATTN;
  result->src[0] = k;
  result->src[1] = v;
  result->src[2] = q;
  result->src[3] = g;
  result->src[4] = state;
  return result;
}

// v_rwkv_wkv7

v_tensor* v_rwkv_wkv7(
  struct v_ctx* ctx,
  v_tensor* r,
  v_tensor* w,
  v_tensor* k,
  v_tensor* v,
  v_tensor* a,
  v_tensor* b,
  v_tensor* state) {
  V_ASSERT(v_is_contiguous(r));
  V_ASSERT(v_is_contiguous(w));
  V_ASSERT(v_is_contiguous(k));
  V_ASSERT(v_is_contiguous(v));
  V_ASSERT(v_is_contiguous(a));
  V_ASSERT(v_is_contiguous(b));
  V_ASSERT(v_is_contiguous(state));

  const int64_t S        = k->ne[0];
  const int64_t H        = k->ne[1];
  const int64_t n_tokens = k->ne[2];
  const int64_t n_seqs   = state->ne[1];
  {
    V_ASSERT(w->ne[0] == S && w->ne[1] == H && w->ne[2] == n_tokens);
    V_ASSERT(k->ne[0] == S && k->ne[1] == H && k->ne[2] == n_tokens);
    V_ASSERT(v->ne[0] == S && v->ne[1] == H && v->ne[2] == n_tokens);
    V_ASSERT(a->ne[0] == S && a->ne[1] == H && a->ne[2] == n_tokens);
    V_ASSERT(b->ne[0] == S && b->ne[1] == H && b->ne[2] == n_tokens);
    V_ASSERT(nelements(state) == S * S * H * n_seqs);
  }

  // concat output and new_state
  const int64_t ne[4] = {S * H, n_tokens + S * n_seqs, 1, 1};
  v_tensor* result    = v_new_tensor(ctx, v_TYPE_F32, 4, ne);

  result->op     = v_OP_RWKV_WKV7;
  result->src[0] = r;
  result->src[1] = w;
  result->src[2] = k;
  result->src[3] = v;
  result->src[4] = a;
  result->src[5] = b;
  result->src[6] = state;

  return result;
}


v_tensor* v_map_custom1_impl(struct v_ctx* ctx,
                             v_tensor* a,
                             const v_custom1_op_t fun,
                             int n_tasks,
                             void* userdata,
                             bool inplace) {
  V_ASSERT(n_tasks == v_N_TASKS_MAX || n_tasks > 0);

  v_tensor* result = inplace
                       ? v_tensor_view(ctx, a)
                       : v_dup_tensor(ctx, a);

  struct v_map_custom1_op_params params = {
    /*.fun      =*/ fun,
    /*.n_tasks  =*/ n_tasks,
    /*.userdata =*/ userdata
  };
  v_set_op_params(result, &params, sizeof(params));

  result->op     = v_OP_MAP_CUSTOM1;
  result->src[0] = a;

  return result;
}


v_tensor* v_map_custom2_impl(struct v_ctx* ctx,
                             v_tensor* a,
                             v_tensor* b,
                             const v_custom2_op_t fun,
                             int n_tasks,
                             void* userdata,
                             bool inplace) {
  V_ASSERT(n_tasks == v_N_TASKS_MAX || n_tasks > 0);

  v_tensor* result = inplace
                       ? v_tensor_view(ctx, a)
                       : v_dup_tensor(ctx, a);

  struct v_map_custom2_op_params params = {
    /*.fun      =*/ fun,
    /*.n_tasks  =*/ n_tasks,
    /*.userdata =*/ userdata
  };
  v_set_op_params(result, &params, sizeof(params));

  result->op     = v_OP_MAP_CUSTOM2;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}


v_tensor* v_map_custom3_impl(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  v_tensor* c,
  const v_custom3_op_t fun,
  int n_tasks,
  void* userdata,
  bool inplace) {
  V_ASSERT(n_tasks == v_N_TASKS_MAX || n_tasks > 0);
  v_tensor* result = inplace
                       ? v_tensor_view(ctx, a)
                       : v_dup_tensor(ctx, a);
  struct v_map_custom3_op_params params = {
    /*.fun      =*/ fun,
    /*.n_tasks  =*/ n_tasks,
    /*.userdata =*/ userdata
  };
  v_set_op_params(result, &params, sizeof(params));
  result->op     = v_OP_MAP_CUSTOM3;
  result->src[0] = a;
  result->src[1] = b;
  result->src[2] = c;
  return result;
}


v_tensor* v_custom_4d(struct v_ctx* ctx,
                      enum v_data_type type,
                      int64_t ne0,
                      int64_t ne1,
                      int64_t ne2,
                      int64_t ne3,
                      v_tensor* * args,
                      int n_args,
                      v_custom_op_t fun,
                      int n_tasks,
                      void* userdata) {
  V_ASSERT(n_args < v_MAX_SRC);
  v_tensor* result                 = v_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
  struct v_custom_op_params params = {
    /*.fun      =*/ fun,
    /*.n_tasks  =*/ n_tasks,
    /*.userdata =*/ userdata
  };
  v_set_op_params(result, &params, sizeof(params));
  result->op = v_OP_CUSTOM;
  for (int i = 0; i < n_args; i++) {
    result->src[i] = args[i];
  }
  return result;
}

v_tensor* v_custom_inplace(struct v_ctx* ctx,
                           v_tensor* a,
                           v_tensor* * args,
                           int n_args,
                           v_custom_op_t fun,
                           int n_tasks,
                           void* userdata) {
  V_ASSERT(n_args < v_MAX_SRC - 1);
  v_tensor* result = v_tensor_view(ctx, a);

  struct v_custom_op_params params = {
    /*.fun      =*/ fun,
    /*.n_tasks  =*/ n_tasks,
    /*.userdata =*/ userdata
  };
  v_set_op_params(result, &params, sizeof(params));

  result->op     = v_OP_CUSTOM;
  result->src[0] = a;
  for (int i = 0; i < n_args; i++) {
    result->src[i + 1] = args[i];
  }
  return result;
}

v_tensor* v_cross_entropy_loss(struct v_ctx* ctx,
                               v_tensor* a,
                               v_tensor* b) {
  V_ASSERT(v_are_same_shape(a, b));
  v_tensor* result = v_new_tensor_1d(ctx, a->type, 1);
  result->op       = v_OP_CROSS_ENTROPY_LOSS;
  result->src[0]   = a;
  result->src[1]   = b;
  return result;
}

// v_cross_entropy_loss_back
v_tensor* v_cross_entropy_loss_back(v_ctx* ctx,
                                    v_tensor* a,
                                    v_tensor* b,
                                    v_tensor* c) {
  V_ASSERT(v_is_scalar(a));
  V_ASSERT(v_are_same_shape(b, c));

  v_tensor* result = v_dup_tensor(ctx, b);

  result->op     = v_OP_CROSS_ENTROPY_LOSS_BACK;
  result->src[0] = a;
  result->src[1] = b;
  result->src[2] = c;

  return result;
}

// opt_step_adamw
v_tensor* v_opt_step_adamw(struct v_ctx* ctx,
                           v_tensor* a,
                           v_tensor* grad,
                           v_tensor* m,
                           v_tensor* v,
                           v_tensor* adamw_params) {
  V_ASSERT(a->flags & TENSOR_FLAG_PARAM);
  V_ASSERT(v_are_same_shape(a, grad));
  V_ASSERT(v_are_same_shape(a, m));
  V_ASSERT(v_are_same_shape(a, v));
  V_ASSERT(adamw_params->type == v_TYPE_F32);
  V_ASSERT(nelements(adamw_params) == 7);
  v_tensor* result = v_tensor_view(ctx, a);
  result->op       = v_OP_OPT_STEP_ADAMW;
  result->src[0]   = a;
  result->src[1]   = grad;
  result->src[2]   = m;
  result->src[3]   = v;
  result->src[4]   = adamw_params;
  return result;
}

// opt_step_sgd
v_tensor* v_opt_step_sgd(struct v_ctx* ctx,
                         v_tensor* a,
                         v_tensor* grad,
                         v_tensor* params) {
  V_ASSERT(a->flags & TENSOR_FLAG_PARAM);
  V_ASSERT(v_are_same_shape(a, grad));
  V_ASSERT(params->type == v_TYPE_F32);
  V_ASSERT(nelements(params) == 2);
  v_tensor* result = v_tensor_view(ctx, a);
  result->op       = v_OP_OPT_STEP_SGD;
  result->src[0]   = a;
  result->src[1]   = grad;
  result->src[2]   = params;
  return result;
}

////////////////////////////////////////////////////////////////////////////////

struct v_hash_set v_hash_set_new(size_t size) {
  size = v_hash_size(size);
  struct v_hash_set result;
  result.size = size;
  result.keys = (v_tensor* *)v_MALLOC(sizeof(v_tensor *) * size);
  result.used = (uint32_t*)v_CALLOC(v_bitset_size(size), sizeof(unsigned));
  return result;
}


void v_hash_set_free(struct v_hash_set* hash_set) {
  v_FREE(hash_set->used);
  v_FREE(hash_set->keys);
}

size_t v_hash_size(size_t min_sz) {
  // next primes after powers of two
  static const size_t primes[] = {
    2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
    2053, 4099, 8209, 16411, 32771, 65537, 131101,
    262147, 524309, 1048583, 2097169, 4194319, 8388617,
    16777259, 33554467, 67108879, 134217757, 268435459,
    536870923, 1073741827, 2147483659
  };
  static const size_t n_primes = sizeof(primes) / sizeof(primes[0]);
  // find the smallest prime that is larger or equal than min_sz
  size_t l = 0;
  size_t r = n_primes;
  while (l < r) {
    size_t m = (l + r) / 2;
    if (primes[m] < min_sz) {
      l = m + 1;
    }
    else {
      r = m;
    }
  }
  size_t sz = l < n_primes
                ? primes[l]
                : min_sz | 1;
  return sz;
}


struct hash_map* v_new_hash_map(size_t size) {
  struct hash_map* result = (struct hash_map*)v_MALLOC(sizeof(struct hash_map));
  result->set             = v_hash_set_new(size);
  result->vals            = (v_tensor* *)v_CALLOC(result->set.size, sizeof(v_tensor *));
  return result;
}

void v_hash_map_free(struct hash_map* map) {
  v_hash_set_free(&map->set);
  v_FREE(map->vals);
  v_FREE(map);
}


size_t v_visit_parents(struct v_cgraph* cgraph, v_tensor* node) {
  // check if already visited
  size_t node_hash_pos = find_hash(&cgraph->visited_hash_set, node);
  #ifdef MEM_DEBUG
  std::cout << node_hash_pos << std::endl;
  #endif
  V_ASSERT(node_hash_pos != v_HASHSET_FULL);
  if (!v_bit_set_get(cgraph->visited_hash_set.used, node_hash_pos)) {
    // This is the first time we see this node in the current graph.
    cgraph->visited_hash_set.keys[node_hash_pos] = node;
    v_bitset_set(cgraph->visited_hash_set.used, node_hash_pos);
    cgraph->use_counts[node_hash_pos] = 0;
  }
  else {
    // already visited
    return node_hash_pos;
  }

  for (int i = 0; i < v_MAX_SRC; ++i) {
    const int k = (cgraph->order == v_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT)
                    ? i
                    : (cgraph->order == v_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT)
                    ? (v_MAX_SRC - 1 - i)
                    :
                    /* unknown order, just fall back to using i */ i;

    v_tensor* src = node->src[k];
    if (src) {
      size_t src_hash_pos = v_visit_parents(cgraph, src);
      // Update the use count for this operand.
      cgraph->use_counts[src_hash_pos]++;
    }
  }

  if (node->op == v_OP_NONE && !(node->flags & TENSOR_FLAG_PARAM)) {
    // reached a leaf node, not part of the gradient graph (e.g. a constant)
    V_ASSERT(cgraph->n_leafs < cgraph->size);
    if (strlen(node->name) == 0) {
      v_format_name(node, "leaf_%d", cgraph->n_leafs);
    }

    cgraph->leafs[cgraph->n_leafs] = node;
    cgraph->n_leafs++;
  }
  else {
    V_ASSERT(cgraph->n_nodes < cgraph->size);
    if (strlen(node->name) == 0) {
      v_format_name(node, "node_%d", cgraph->n_nodes);
    }
    cgraph->nodes[cgraph->n_nodes] = node;
    cgraph->n_nodes++;
  }

  return node_hash_pos;
}


void v_build_foward_expand(struct v_cgraph* cgraph, v_tensor* tensor) {
  const int n0 = cgraph->n_nodes;
  v_visit_parents(cgraph, tensor);
  const int n_new = cgraph->n_nodes - n0;
  #ifdef DEBUG
  printf("%s: visited %d new nodes\n", __func__, n_new);
  #endif
  if (n_new > 0) {
    if (cgraph->nodes[cgraph->n_nodes - 1] != tensor) {
      throw std::runtime_error("cgraph forward expand tensor be last node");
    }
  }
}

void v_build_backward_expend(struct v_ctx* ctx,
                             struct v_cgraph* cgraph,
                             v_tensor* * grad_accs) {
  V_ASSERT(cgraph->n_nodes > 0);
  V_ASSERT(cgraph->grads);
  V_ASSERT(cgraph->grad_accs);

  const int n_nodes_f = cgraph->n_nodes;
  memset(cgraph->grads, 0, cgraph->visited_hash_set.size * sizeof(v_tensor*));
  memset(cgraph->grad_accs, 0, cgraph->visited_hash_set.size * sizeof(v_tensor*));
  bool* grads_needed = (bool*)calloc(cgraph->visited_hash_set.size, sizeof(bool));
  {
    bool any_params = false;
    bool any_loss   = false;
    for (int i = 0; i < n_nodes_f; ++i) {
      v_tensor* node = cgraph->nodes[i];
      any_params     = any_params || (node->flags & TENSOR_FLAG_PARAM);
      any_loss       = any_loss || (node->flags & TENSOR_FLAG_LOSS);
    }
    //V_ASSERT(any_params && "no trainable parameters found, did you forget to call v_set_param?");
    //V_ASSERT(any_loss && "no training loss found, did you forget to call v_set_loss?");
  }

  for (int i = 0; i < n_nodes_f; ++i) {
    v_tensor* node = cgraph->nodes[i];

    if (node->type == v_TYPE_I32) {
      continue;
    }

    bool node_needs_grad       = (node->flags & TENSOR_FLAG_PARAM) || (node->flags & TENSOR_FLAG_LOSS);
    bool ignore_src[v_MAX_SRC] = {false};
    switch (node->op) {
      // gradients in node->src[0] for one reason or another have no effect on output gradients
      case V_OP_IM2COL: // only used for its shape
      case v_OP_IM2COL_BACK: // same as IM2COL
        ignore_src[0] = true;
        break;
      case v_OP_UNARY: {
        const enum v_unary_op uop = v_get_unary_op(node);
        // SGN and STEP unary ops are piecewise constant
        if (uop == v_UNARY_OP_SGN || uop == v_UNARY_OP_STEP) {
          ignore_src[0] = true;
        }
      }
      break;

      // gradients in node->src[1] for one reason or another have no effect on output gradients
      case v_OP_CPY: // gradients in CPY target are irrelevant
      case v_OP_GET_ROWS: // row indices not differentiable
      case V_OP_GET_ROWS_BACK: // same as for GET_ROWS
      case V_OP_ROPE: // positions not differentiable
        ignore_src[1] = true;
        break;

      default:
        break;
    }
    for (int j = 0; j < v_MAX_SRC; ++j) {
      if (!node->src[j] || ignore_src[j] || !grads_needed[find_hash(&cgraph->visited_hash_set, node->src[j])]) {
        continue;
      }
      V_ASSERT(node->src[j]->type == v_TYPE_F32 || node->src[j]->type == v_TYPE_F16);
      node_needs_grad = true;
      break;
    }
    if (!node_needs_grad) {
      continue;
    }

    // inplace operations are currently not supported
    V_ASSERT(!node->view_src || node->op == v_OP_CPY || node->op == V_OP_VIEW ||
      node->op == v_OP_RESHAPE || node->op == V_OP_PERMUTE || node->op == v_OP_TRANSPOSE);

    const size_t ihash = find_hash(&cgraph->visited_hash_set, node);
    V_ASSERT(ihash != v_HASHSET_FULL);
    V_ASSERT(v_bit_set_get(cgraph->visited_hash_set.used, ihash));
    if (grad_accs && grad_accs[i]) {
      cgraph->grad_accs[ihash] = grad_accs[i];
      cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
    }
    else if (node->flags & TENSOR_FLAG_LOSS) {
      // loss tensors always need a gradient accumulator
      cgraph->grad_accs[ihash] = v_new_tensor(ctx, v_TYPE_F32, V_MAX_DIMS, node->ne);
      cgraph->grads[ihash]     = cgraph->grad_accs[ihash];
    }
    grads_needed[ihash] = true;
  }

  for (int i = n_nodes_f - 1; i >= 0; --i) {
    v_tensor* node = cgraph->nodes[i];
    printf("node name: %s \n", v_op_name(node->op));
    v_compute_backward(ctx, cgraph, i, grads_needed);
  }
  free(grads_needed);
}


void v_graph_reset(struct v_cgraph* cgraph) {
  if (!cgraph) {
    return;
  }
  V_ASSERT(cgraph->grads != NULL);

  for (int i = 0; i < cgraph->n_nodes; i++) {
    v_tensor* node     = cgraph->nodes[i];
    v_tensor* grad_acc = v_graph_get_grad_acc(cgraph, node);

    if (node->op == v_OP_OPT_STEP_ADAMW) {
      // clear momenta
      v_set_zero(node->src[2]);
      v_set_zero(node->src[3]);
    }

    // initial gradients of loss should be 1, 0 otherwise
    if (grad_acc) {
      if (node->flags & TENSOR_FLAG_LOSS) {
        V_ASSERT(grad_acc->type == v_TYPE_F32);
        V_ASSERT(v_is_scalar(grad_acc));
        const float onef = 1.0f;
        if (grad_acc->buffer) {
          v_set_backend_tensor(grad_acc, &onef, 0, sizeof(float));
        }
        else {
          V_ASSERT(grad_acc->data);
          *((float*)grad_acc->data) = onef;
        }
      }
      else {
        v_set_zero(grad_acc);
      }
    }
  }
}


v_tensor* v_graph_get_tensor(const struct v_cgraph* cgraph, const char* name) {
  for (int i = 0; i < cgraph->n_leafs; i++) {
    v_tensor* leaf = cgraph->leafs[i];

    if (strcmp(leaf->name, name) == 0) {
      return leaf;
    }
  }

  for (int i = 0; i < cgraph->n_nodes; i++) {
    v_tensor* node = cgraph->nodes[i];

    if (strcmp(node->name, name) == 0) {
      return node;
    }
  }

  return NULL;
}


void v_print_graph(const struct v_cgraph* cgraph) {
  LOG_INFO("=== GRAPH ===\n");
  LOG_INFO("n_nodes = %d\n", cgraph->n_nodes);
  for (int i = 0; i < cgraph->n_nodes; i++) {
    v_tensor* node = cgraph->nodes[i];
    LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
             i,
             node->ne[0],
             node->ne[1],
             node->ne[2],
             v_op_name(node->op),
             (node->flags & TENSOR_FLAG_PARAM) ? "x" :
             v_graph_get_grad(cgraph, node) ? "g" : " ");
  }

  LOG_INFO("n_leafs = %d\n", cgraph->n_leafs);
  for (int i = 0; i < cgraph->n_leafs; i++) {
    v_tensor* node = cgraph->leafs[i];

    LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
             i,
             node->ne[0],
             node->ne[1],
             v_op_name(node->op),
             get_name(node));
  }
  LOG_INFO("========================================\n");
}

static int v_node_list_find_tensor(const struct v_cgraph* cgraph,
                                   const int* idxs,
                                   int count,
                                   const v_tensor* tensor) {
  V_ASSERT(cgraph && idxs);
  for (int i = 0; i < count; ++i) {
    const int node_idx = idxs[i];
    if (node_idx >= cgraph->n_nodes) {
      return -1;
    }
    if (cgraph->nodes[node_idx] == tensor) {
      return i;
    }
  }
  return -1;
}

bool v_can_fuse_subgraph_ext(const struct v_cgraph* cgraph,
                             const int* node_idxs,
                             int count,
                             const enum v_operation* ops,
                             const int* outputs,
                             int num_outputs) {
  V_ASSERT(outputs && num_outputs > 0);

  for (int i = 0; i < count; ++i) {
    if (node_idxs[i] >= cgraph->n_nodes) {
      return false;
    }

    const v_tensor* node = cgraph->nodes[node_idxs[i]];

    if (node->op != ops[i]) {
      return false;
    }

    if (v_node_list_find_tensor(cgraph, outputs, num_outputs, node) != -1) {
      continue;
    }

    if (node->flags & TENSOR_FLAG_OUTPUT) {
      return false;
    }

    int subgraph_uses = 0;
    for (int j = i + 1; j < count; ++j) {
      const v_tensor* other_node = cgraph->nodes[node_idxs[j]];
      for (int src_idx = 0; src_idx < v_MAX_SRC; src_idx++) {
        if (other_node->src[src_idx] == node) {
          subgraph_uses++;
        }
      }
    }

    if (subgraph_uses != v_node_get_use_count(cgraph, node_idxs[i])) {
      return false;
    }

    // if node is a view, check if the view_src and all it's parent view_srcs are within the subgraph
    v_tensor* view_src = node->view_src;
    while (view_src) {
      if (v_node_list_find_tensor(cgraph, node_idxs, count, view_src) == -1) {
        return false;
      }
      view_src = view_src->view_src;
    }
  }

  return true;
}

// check if node is part of the graph
static bool v_graph_find(const struct v_cgraph* cgraph, const v_tensor* node) {
  if (cgraph == NULL) {
    return true;
  }

  for (int i = 0; i < cgraph->n_nodes; i++) {
    if (cgraph->nodes[i] == node) {
      return true;
    }
  }

  return false;
}


static v_tensor* v_graph_get_parent(const struct v_cgraph* cgraph, const v_tensor* node) {
  for (int i = 0; i < cgraph->n_nodes; i++) {
    v_tensor* parent = cgraph->nodes[i];
    v_tensor* grad   = v_graph_get_grad(cgraph, parent);

    if (grad == node) {
      return parent;
    }
  }

  return NULL;
}

static void v_graph_dump_dot_node_edge(FILE* fp, const struct v_cgraph* gb, v_tensor* node,
                                       v_tensor* parent, const char* label) {
  v_tensor* gparent  = v_graph_get_parent(gb, node);
  v_tensor* gparent0 = v_graph_get_parent(gb, parent);
  fprintf(fp,
          "  \"%p\" -> \"%p\" [ arrowhead = %s; style = %s; label = \"%s\"; ]\n",
          gparent0
            ? (void*)gparent0
            : (void*)parent,
          gparent
            ? (void*)gparent
            : (void*)node,
          gparent
            ? "empty"
            : "vee",
          gparent
            ? "dashed"
            : "solid",
          label);
}

static void v_graph_dump_dot_leaf_edge(FILE* fp, v_tensor* node, v_tensor* parent,
                                       const char* label) {
  fprintf(fp,
          "  \"%p\" -> \"%p\" [ label = \"%s\"; ]\n",
          (void*)parent,
          (void*)node,
          label);
}

void v_graph_dump_dot(const struct v_cgraph* gb, const struct v_cgraph* gf, const char* filename) {
  char color[16];
  FILE* fp = v_fopen(filename, "w");
  V_ASSERT(fp);
  fprintf(fp, "digraph G {\n");
  fprintf(fp, "  newrank = true;\n");
  fprintf(fp, "  rankdir = TB;\n");
  for (int i = 0; i < gb->n_nodes; i++) {
    v_tensor* node = gb->nodes[i];
    v_tensor* grad = v_graph_get_grad(gb, node);

    if (v_graph_get_parent(gb, node) != NULL) {
      continue;
    }

    if (node->flags & TENSOR_FLAG_PARAM) {
      snprintf(color, sizeof(color), "yellow");
    }
    else if (grad) {
      if (v_graph_find(gf, node)) {
        snprintf(color, sizeof(color), "green");
      }
      else {
        snprintf(color, sizeof(color), "lightblue");
      }
    }
    else {
      snprintf(color, sizeof(color), "white");
    }

    fprintf(fp,
            "  \"%p\" [ "
            "style = filled; fillcolor = %s; shape = record; "
            "label=\"",
            (void*)node,
            color);

    if (strlen(node->name) > 0) {
      fprintf(fp, "%s (%s)|", node->name, v_type_name(node->type));
    }
    else {
      fprintf(fp, "(%s)|", v_type_name(node->type));
    }

    if (v_is_matrix(node)) {
      fprintf(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, node->ne[0], node->ne[1], v_op_symbol(node->op));
    }
    else {
      fprintf(fp,
              "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s",
              i,
              node->ne[0],
              node->ne[1],
              node->ne[2],
              v_op_symbol(node->op));
    }

    if (grad) {
      fprintf(fp, " | <g>%s\"; ]\n", v_op_symbol(grad->op));
    }
    else {
      fprintf(fp, "\"; ]\n");
    }
  }

  for (int i = 0; i < gb->n_leafs; i++) {
    v_tensor* node = gb->leafs[i];

    snprintf(color, sizeof(color), "pink");

    fprintf(fp,
            "  \"%p\" [ "
            "style = filled; fillcolor = %s; shape = record; "
            "label=\"<x>",
            (void*)node,
            color);

    if (strlen(node->name) > 0) {
      fprintf(fp, "%s (%s)|", node->name, v_type_name(node->type));
    }
    else {
      fprintf(fp, "(%s)|", v_type_name(node->type));
    }

    fprintf(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, node->ne[0], node->ne[1]);
    if (nelements(node) < 5 && node->data != NULL) {
      fprintf(fp, " | (");
      for (int j = 0; j < nelements(node); j++) {
        // FIXME: use ggml-backend to obtain the tensor data
        //if (node->type == v_TYPE_I8 || node->type == v_TYPE_I16 || node->type == v_TYPE_I32) {
        //    fprintf(fp, "%d", v_get_i32_1d(node, j));
        //}
        //else if (node->type == v_TYPE_F32 ||
        //         node->type == v_TYPE_F16 ||
        //         node->type == v_TYPE_BF16) {
        //    fprintf(fp, "%.1e", (double)v_get_f32_1d(node, j));
        //}
        //else
        {
          fprintf(fp, "#");
        }
        if (j < nelements(node) - 1) {
          fprintf(fp, ", ");
        }
      }
      fprintf(fp, ")");
    }
    fprintf(fp, "\"; ]\n");
  }

  for (int i = 0; i < gb->n_nodes; i++) {
    v_tensor* node = gb->nodes[i];

    for (int j = 0; j < v_MAX_SRC; j++) {
      if (node->src[j]) {
        char label[16];
        snprintf(label, sizeof(label), "src %d", j);
        v_graph_dump_dot_node_edge(fp, gb, node, node->src[j], label);
      }
    }
  }

  for (int i = 0; i < gb->n_leafs; i++) {
    v_tensor* node = gb->leafs[i];

    for (int j = 0; j < v_MAX_SRC; j++) {
      if (node->src[j]) {
        char label[16];
        snprintf(label, sizeof(label), "src %d", j);
        v_graph_dump_dot_leaf_edge(fp, node, node->src[j], label);
      }
    }
  }

  fprintf(fp, "}\n");

  fclose(fp);

  LOG_INFO("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

void v_critical_section_start();
void v_critical_section_end();


void v_quantize_init(enum v_data_type type) {
  v_critical_section_start();

  switch (type) {
    case v_TYPE_IQ2_XXS:
    case v_TYPE_IQ2_XS:
    case v_TYPE_IQ2_S:
    case v_TYPE_IQ1_S:
    case v_TYPE_IQ1_M: iq2xs_init_impl(type);
      break;
    case v_TYPE_IQ3_XXS: iq3xs_init_impl(256);
      break;
    case v_TYPE_IQ3_S: iq3xs_init_impl(512);
      break;
    default: // nothing
      break;
  }

  v_critical_section_end();
}

void v_quantize_free(void) {
  v_critical_section_start();

  iq2xs_free_impl(v_TYPE_IQ2_XXS);
  iq2xs_free_impl(v_TYPE_IQ2_XS);
  iq2xs_free_impl(v_TYPE_IQ1_S);
  iq3xs_free_impl(256);

  v_critical_section_end();
}

void set_log(v_log_callback log_callback, void* user_data) {
  g_logger_state.log_callback = log_callback
                                  ? log_callback
                                  : mml_log_callback_default;
  g_logger_state.log_callback_user_data = user_data;
}

std::mutex v_critical_section_mutex;

void v_critical_section_start() {
  v_critical_section_mutex.lock();
}

void v_critical_section_end(void) {
  v_critical_section_mutex.unlock();
}
