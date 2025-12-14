#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC
#include "v_backend.hpp"
#include "ggml-impl.hpp"
#include "v.hpp"
#include "v_hash.hpp"
#include "v_tensor.hpp"
// FIXME: required here for quantization functions
#include <mutex>

#include "v_quants.hpp"
#include "v_util.hpp"
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

#define UNUSED V_UNUSED

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
    } else {
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
char* line     = nullptr;
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
(char*) nullptr);
// try lldb
execlp ("lldb", "lldb", "--batch",
            "-o", "bt",
            "-o", "quit",
            "-p", &attach [sizeof("attach ") - 1],
(char*) nullptr);
// gdb failed, fallback to backtrace_symbols
v_print_backtrace_symbols();
_Exit (0);
    } else { // parent
#if defined(__linux__)
prctl(PR_SET_PTRACER, child_pid);
close (lock[1]);
close (lock[0]);
#endif
waitpid(child_pid, nullptr, 0);
    }
}
#else
void v_print_backtrace(void) {
  // platform not supported
}
#endif


#include <cstdlib>
#include <exception>


void v_aligned_free(void* ptr, size_t size) {
  V_UNUSED(size);
  _aligned_free(ptr);
}

//todo : check
inline static void* v_malloc(size_t size) {
  if (size == 0) {
    V_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for v_malloc!\n");
    return nullptr;
  }
  void* result = malloc(size);
  if (result == nullptr) {
    V_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
    V_ABORT("fatal error");
  }
  return result;
}


#define v_MALLOC(size)      v_malloc(size)
#define v_CALLOC(num, size) v_calloc(num, size)

#define v_FREE(ptr) free(ptr)


//
// cross-platform UTF-8 file paths
//

#ifdef _WIN32
static wchar_t* v_mbstowcs(const char* mbs) {
  int wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, nullptr, 0);
  if (!wlen) {
    errno = EINVAL;
    return nullptr;
  }

  wchar_t* wbuf = (wchar_t*)v_MALLOC(wlen * sizeof(wchar_t));
  wlen          = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, wbuf, wlen);
  if (!wlen) {
    v_FREE(wbuf);
    errno = EINVAL;
    return nullptr;
  }

  return wbuf;
}
#endif

FILE* v_fopen(const char* fname, const char* mode) {
  #ifdef _WIN32
  FILE* file = nullptr;
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


size_t v_graph_overhead_custom(size_t size, bool grads) {
  return V_OBJECT_SIZE + V_PAD(sizeof(v_cgraph), V_MEM_ALIGN);
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
  } else {
    nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;
    for (int i = 1; i < V_MAX_DIMS; ++i) {
      nbytes += (tensor->ne[i] - 1) * tensor->nb[i];
    }
  }

  return nbytes;
}

size_t v_nbytes_pad(const v_tensor* tensor) {
  return V_PAD(num_bytes(tensor), V_MEM_ALIGN);
}


size_t v_row_size(v_data_type type, int64_t ne) {
  assert(ne % block_size(type) == 0);
  return v_type_size(type) * ne / block_size(type);
}


const char* v_op_desc(const v_tensor* t) {
  if (t->op == v_OP_UNARY) {
    v_unary_op uop = v_get_unary_op(t);
    return v_unary_op_name(uop);
  }
  if (t->op == v_OP_GLU) {
    v_glu_op gop = v_get_glu_op(t);
    return v_glu_op_name(gop);
  }
  return v_op_name(t->op);
}


size_t v_tensor_over_head(void) {
  return V_OBJECT_SIZE + V_TENSOR_SIZE;
}


void v_unravel_index(const v_tensor* tensor, int64_t i, int64_t* i0, int64_t* i1, int64_t* i2, int64_t* i3) {
  const int64_t ne2 = tensor->ne[2];
  const int64_t ne1 = tensor->ne[1];
  const int64_t ne0 = tensor->ne[0];
  const int64_t i3_ = (i / (ne2 * ne1 * ne0));
  const int64_t i2_ = (i - i3_ * ne2 * ne1 * ne0) / (ne1 * ne0);
  const int64_t i1_ = (i - i3_ * ne2 * ne1 * ne0 - i2_ * ne1 * ne0) / ne0;
  const int64_t i0_ = (i - i3_ * ne2 * ne1 * ne0 - i2_ * ne1 * ne0 - i1_ * ne0);
  if (i0) *i0 = i0_;
  if (i1) *i1 = i1_;
  if (i2) *i2 = i2_;
  if (i3) *i3 = i3_;
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
  vsnprintf(tensor->name.data(), sizeof(tensor->name.data()), fmt, args);
  va_end(args);
  return tensor;
}


v_tensor* v_set_impl(v_ctx* ctx,
                     v_tensor* a, v_tensor* b,
                     size_t nb1, size_t nb2, size_t nb3,
                     size_t offset,
                     bool inplace) {
  V_ASSERT(nelements(a) >= nelements(b));
  // make a view of the destination
  v_tensor* result = inplace ? v_tensor_view(ctx, a) : v_dup_tensor(ctx, a);
  V_ASSERT(offset < static_cast<size_t>(1 << 30));
  int32_t params[] = {
    static_cast<int32_t>(nb1),
    static_cast<int32_t>(nb2),
    static_cast<int32_t>(nb3),
    static_cast<int32_t>(offset),
    static_cast<int32_t>(inplace) ? 1 : 0
  };
  v_set_op_params(result, params, sizeof(params));
  result->op     = v_OP_SET;
  result->src[0] = a;
  result->src[1] = b;
  return result;
}


v_tensor* v_cpy(v_ctx* ctx,
                v_tensor* a, v_tensor* b) {
  V_ASSERT(nelements(a) == nelements(b));
  // make a view of the destination
  v_tensor* result = v_tensor_view(ctx, b);
  if (strlen(b->name.data()) > 0) {
    v_format_name(result, "%s (copy of %s)", b->name.data(), a->name.data());
  } else {
    v_format_name(result, "%s (copy)", a->name.data());
  }
  result->op     = V_OP_CPY;
  result->src[0] = a;
  result->src[1] = b;
  return result;
}


static v_tensor* v_diag_mask_inf_impl(v_ctx* ctx,
                                      v_tensor* a,
                                      int n_past,
                                      bool inplace) {
  v_tensor* result = inplace ? v_tensor_view(ctx, a) : v_dup_tensor(ctx, a);
  int32_t params[] = {n_past};
  v_set_op_params(result, params, sizeof(params));
  result->op     = V_OP_DIAG_MASK_INF;
  result->src[0] = a;

  return result;
}

v_tensor* v_diag_mask_inf(v_ctx* ctx, v_tensor* a, int n_past) {
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
  v_ctx* ctx,
  v_tensor* a) {
  return v_soft_max_impl(ctx, a, nullptr, 1.0f, 0.0f, false);
}

v_tensor* v_soft_max_inplace(
  struct v_ctx* ctx,
  v_tensor* a) {
  return v_soft_max_impl(ctx, a, nullptr, 1.0f, 0.0f, true);
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
    a->src[2] = nullptr;
    return;
  }

  V_ASSERT(a->op == V_OP_SOFT_MAX);
  V_ASSERT(a->src[2] == nullptr);
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

  memcpy((float*)result->op_params.data() + 0, &scale, sizeof(float));
  memcpy((float*)result->op_params.data() + 1, &max_bias, sizeof(float));

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
  int sections[V_MROPE_SECTIONS],
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

  V_ASSERT(b->is_vector());
  V_ASSERT(b->type == v_TYPE_I32);

  bool mrope_used = mode & V_ROPE_TYPE_MROPE;
  if (mrope_used) {
    V_ASSERT(a->ne[2] * 4 == b->ne[0]); // mrope expecting 4 position ids per token
  } else {
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
    memcpy(params + 11, sections, sizeof(int32_t) * V_MROPE_SECTIONS);
  } else {
    memset(params + 11, 0, sizeof(int32_t) * V_MROPE_SECTIONS);
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
    nullptr,
    n_dims,
    nullptr,
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
  int sections[V_MROPE_SECTIONS],
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
  int sections[V_MROPE_SECTIONS],
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
    nullptr,
    n_dims,
    nullptr,
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
    nullptr,
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
    nullptr,
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
    nullptr,
    n_dims,
    nullptr,
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

v_tensor* v_rope_custom_inplace(v_ctx* ctx,
                                v_tensor* a, v_tensor* b,
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
    nullptr,
    n_dims,
    nullptr,
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

v_tensor* v_clamp(v_ctx* ctx,
                  v_tensor* a,
                  float min, float max) {
  // TODO: when implement backward, fix this:
  v_tensor* result = v_tensor_view(ctx, a);
  float params[]   = {min, max};
  v_set_op_params(result, params, sizeof(params));
  result->op     = V_OP_CLAMP;
  result->src[0] = a;
  return result;
}


// v_pad

v_tensor* v_pad(v_ctx* ctx,
                v_tensor* a,
                int p0, int p1, int p2, int p3) {
  return v_pad_ext(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3);
}

v_tensor* v_pad_ext(v_ctx* ctx,
                    v_tensor* a,
                    int lp0, int rp0,
                    int lp1, int rp1,
                    int lp2, int rp2,
                    int lp3, int rp3) {
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
v_tensor* v_pad_reflect_1d(v_ctx* ctx,
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
v_tensor* v_roll(v_ctx* ctx,
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

v_tensor* v_arange(v_ctx* ctx,
                   float start, float stop, float step) {
  V_ASSERT(stop > start);

  const int64_t steps = ceilf((stop - start) / step);
  v_tensor* result    = v_new_tensor_1d(ctx, v_TYPE_F32, steps);

  v_set_op_params_f32(result, 0, start);
  v_set_op_params_f32(result, 1, stop);
  v_set_op_params_f32(result, 2, step);

  result->op = v_OP_ARANGE;

  return result;
}

// v_timestep_embedding
v_tensor* v_timestep_embedding(v_ctx* ctx, v_tensor* timesteps,
                               int dim, int max_period) {
  v_tensor* result = v_new_tensor_2d(ctx, v_TYPE_F32, dim, timesteps->ne[0]);

  v_set_op_params_i32(result, 0, dim);
  v_set_op_params_i32(result, 1, max_period);

  result->op     = v_OP_TIMESTEP_EMBEDDING;
  result->src[0] = timesteps;

  return result;
}

// v_argsort
v_tensor* v_argsort(v_ctx* ctx,
                    v_tensor* a,
                    v_sort_order order) {
  V_ASSERT(a->ne[0] <= INT32_MAX);
  v_tensor* result = v_new_tensor(ctx, v_TYPE_I32, V_MAX_DIMS, a->ne.data());
  v_set_op_params_i32(result, 0, order);
  result->op     = v_OP_ARGSORT;
  result->src[0] = a;
  return result;
}

// v_top_k
v_tensor* v_top_k(v_ctx* ctx,
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
    V_ASSERT(mask->ne[1] >= V_PAD(q->ne[1], v_KQ_MASK_PAD) &&
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
    a->src[4] = nullptr;
    return;
  }
  V_ASSERT(a->op == v_OP_FLASH_ATTN_EXT);
  V_ASSERT(a->src[4] == nullptr);
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
  V_ABORT("TODO: adapt to v_flash_attn_ext() changes");

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
  const size_t offs_k = offs_q + V_PAD(elem_q * tsize, V_MEM_ALIGN);
  const size_t offs_v = offs_k + V_PAD(elem_k * tsize, V_MEM_ALIGN);
  const size_t end    = offs_v + V_PAD(elem_v * tsize, V_MEM_ALIGN);

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

v_tensor* v_add_rel_pos_inplace(v_ctx* ctx,
                                v_tensor* a,
                                v_tensor* pw, v_tensor* ph) {
  return v_add_rel_pos_impl(ctx, a, pw, ph, true);
}

// v_rwkv_wkv6
v_tensor* v_rwkv_wkv6(v_ctx* ctx,
                      v_tensor* k, v_tensor* v,
                      v_tensor* r,
                      v_tensor* tf, v_tensor* td,
                      v_tensor* state) {
  V_ASSERT((k)->is_contiguous());
  V_ASSERT((v)->is_contiguous());
  V_ASSERT((r)->is_contiguous());
  V_ASSERT((tf)->is_contiguous());
  V_ASSERT((td)->is_contiguous());
  V_ASSERT((state)->is_contiguous());

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
  V_ASSERT((r)->is_contiguous());
  V_ASSERT((w)->is_contiguous());
  V_ASSERT((k)->is_contiguous());
  V_ASSERT((v)->is_contiguous());
  V_ASSERT((a)->is_contiguous());
  V_ASSERT((b)->is_contiguous());
  V_ASSERT((state)->is_contiguous());

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


v_tensor* v_map_custom1_impl(v_ctx* ctx,
                             v_tensor* a,
                             const v_custom1_op_t fun,
                             int n_tasks,
                             void* userdata,
                             bool inplace) {
  V_ASSERT(n_tasks == v_N_TASKS_MAX || n_tasks > 0);
  v_tensor* result = inplace ? v_tensor_view(ctx, a) : v_dup_tensor(ctx, a);
  v_map_custom1_op_params params = {
    /*.fun      =*/ fun,
    /*.n_tasks  =*/ n_tasks,
    /*.userdata =*/ userdata
  };
  v_set_op_params(result, &params, sizeof(params));

  result->op     = v_OP_MAP_CUSTOM1;
  result->src[0] = a;

  return result;
}


v_tensor* v_map_custom2_impl(v_ctx* ctx,
                             v_tensor* a, v_tensor* b,
                             const v_custom2_op_t fun,
                             int n_tasks,
                             void* userdata,
                             bool inplace) {
  V_ASSERT(n_tasks == v_N_TASKS_MAX || n_tasks > 0);

  v_tensor* result = inplace
                       ? v_tensor_view(ctx, a)
                       : v_dup_tensor(ctx, a);

  v_map_custom2_op_params params = {
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


v_tensor* v_custom_4d(v_ctx* ctx,
                      v_data_type type,
                      int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                      v_tensor* * args,
                      int n_args,
                      v_custom_op_t fun,
                      int n_tasks,
                      void* userdata) {
  V_ASSERT(n_args < V_MAX_SRC);
  v_tensor* result          = v_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
  v_custom_op_params params = {
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

v_tensor* v_custom_inplace(v_ctx* ctx,
                           v_tensor* a,
                           v_tensor* * args,
                           int n_args,
                           v_custom_op_t fun,
                           int n_tasks,
                           void* userdata) {
  V_ASSERT(n_args < V_MAX_SRC - 1);
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

v_tensor* v_cross_entropy_loss(v_ctx* ctx,
                               v_tensor* a, v_tensor* b) {
  V_ASSERT(v_are_same_shape(a, b));
  v_tensor* result = v_new_tensor_1d(ctx, a->type, 1);
  result->op       = V_OP_CROSS_ENTROPY_LOSS;
  result->src[0]   = a;
  result->src[1]   = b;
  return result;
}

// v_cross_entropy_loss_back
v_tensor* v_cross_entropy_loss_back(v_ctx* ctx,
                                    v_tensor* a, v_tensor* b, v_tensor* c) {
  V_ASSERT(a->is_scalar());
  V_ASSERT(v_are_same_shape(b, c));
  v_tensor* result = v_dup_tensor(ctx, b);

  result->op     = v_OP_CROSS_ENTROPY_LOSS_BACK;
  result->src[0] = a;
  result->src[1] = b;
  result->src[2] = c;

  return result;
}

// opt_step_adamw
v_tensor* v_opt_step_adamw(v_ctx* ctx,
                           v_tensor* a, v_tensor* grad,
                           v_tensor* m, v_tensor* v,
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
v_tensor* v_opt_step_sgd(v_ctx* ctx,
                         v_tensor* a, v_tensor* grad,
                         v_tensor* params) {
  V_ASSERT(a->flags & TENSOR_FLAG_PARAM);
  V_ASSERT(v_are_same_shape(a, grad));
  V_ASSERT(params->type == v_TYPE_F32);
  V_ASSERT(params->num_elements() == 2);
  v_tensor* result = v_tensor_view(ctx, a);
  result->op       = v_OP_OPT_STEP_SGD;
  result->src[0]   = a;
  result->src[1]   = grad;
  result->src[2]   = params;
  return result;
}


void v_critical_section_start();
void v_critical_section_end();

void v_quantize_free(void) {
  v_critical_section_start();
  iq2xs_free_impl(v_TYPE_IQ2_XXS);
  iq2xs_free_impl(v_TYPE_IQ2_XS);
  iq2xs_free_impl(v_TYPE_IQ1_S);
  iq3xs_free_impl(256);
  v_critical_section_end();
}


std::mutex v_critical_section_mutex;

void v_critical_section_start() {
  v_critical_section_mutex.lock();
}

void v_critical_section_end() {
  v_critical_section_mutex.unlock();
}
