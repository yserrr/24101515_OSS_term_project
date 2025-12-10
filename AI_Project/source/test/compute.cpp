#include "v.hpp"
#include "v_allocator.hpp"
#include "v-backend.hpp"
#include "v_vk.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include "extern/pybind11/pybind11.h"
#include "pybind11/embed.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <random>
#include <string>
#include <utility>

#include "v_propagation.hpp"
#include "v_util.hpp"
#include "vk_common.h"
#include "vk_test.h"
#define FORCE_UTF8_CONSOLE
#ifdef FORCE_UTF8_CONSOLE
#include <windows.h>
//#define MEM_DEBUG

struct utf8_console {
  utf8_console() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
  }
} _utf8_console_init;
#endif

static void log_callback_default(v_log_level level, const char* text, void* user_data) {
  (void)level;
  (void)user_data;
  fputs(text,
        stderr);
  fflush(stderr);
}

struct simple_model {
  v_tensor* input{};
  v_tensor* out1;
  v_tensor* out2;
  v_tensor* out3;
  v_tensor* w{};
  v_tensor* b{};
  v_tensor* logit{};
  v_tensor* label{};
  v_tensor* grad_b{};
  v_tensor* grad_w{};
  v_tensor* grad_acc{};
  v_tensor* loss{};
  v_backend_t backend{};
  v_backend_sched_t sched{};
  std::vector<uint8_t> buf;
};

const int rows_A                = 4, cols_A = 2;
float matrix_A[rows_A * cols_A] = {
  2, 8,
  5, 1,
  4, 2,
  8, 6
};

const int rows_B = 3, cols_B = 2;

float weight_mat[rows_B * cols_B] = {
  10, 5,
  9, 9,
  5, 4
};
// [ 60.00 55.00 50.00 110.00
//  90.00 54.00 54.00 126.00
//  42.00 29.00 28.00 64.00]
//  ggml example의 simple backend 를 가져왔습니다. 원본 코드구조와 거의 동일합니다.
void v_init_model(simple_model& model) {
  set_log(log_callback_default, nullptr);
  model.backend        = backend_vk_init(0);
  v_backend_t backends = {model.backend};
  model.sched          = v_sched_new(backends, nullptr, 1, v_DEFAULT_GRAPH_SIZE, true);
}

// build the compute graph to perform input matrix multiplication
struct v_cgraph* build_graph(simple_model& model) {
  size_t buf_size = v_tensor_over_head() * v_DEFAULT_GRAPH_SIZE + graph_overhead();
  model.buf.resize(buf_size);
  v_init_param params0 = {
    /*.mem_size   =*/ buf_size,
    /*.mem_buffer =*/ model.buf.data(),
    /*.no_alloc   =*/ true, // the tensors will be allocated later
  };
  v_ctx* ctx   = v_ctx_init(params0);
  v_cgraph* gf = new_graph(ctx);
  model.input  = v_new_tensor_2d(ctx, v_TYPE_F32, cols_A, rows_A);
  model.w      = v_new_tensor_2d(ctx, v_TYPE_F32, cols_B, rows_B);
  model.b      = v_new_tensor_1d(ctx, v_TYPE_F32, 4);
  model.logit  = v_new_tensor_1d(ctx, v_TYPE_F32, 1);
  model.out1   = v_matmul(ctx, model.input, model.w);
  v_build_foward_expand(gf, model.out1);
  model.out3 = v_add(ctx, model.out1, model.b);
  model.out3 = v_log(ctx, model.out3);
  model.out3 = v_log(ctx, model.out3);

  model.logit = v_sum(ctx, model.out3);
  v_build_foward_expand(gf, model.out3);
  model.loss = v_sqr(ctx, model.logit);
  v_build_foward_expand(gf, model.loss);
  free_ctx(ctx);
  return gf;
}

struct v_tensor* compute(simple_model& model, struct v_cgraph* gf) {
  float backendData[] = {
    1, 2, 3, 4,
  };
  v_sched_reset(model.sched);
  v_sched_alloc_graph(model.sched, gf);
  v_set_backend_tensor(model.input, matrix_A, 0, num_bytes(model.input));
  v_set_backend_tensor(model.w, weight_mat, 0, num_bytes(model.w));
  model.b = v_set_zero(model.b);
  v_set_backend_tensor(model.b, backendData, 0, num_bytes(model.b));
  v_sched_graph_compute(model.sched, gf);
  return v_graph_node(gf, -1);
}

int main(void) {
  v_time_init();
  simple_model model;
  v_init_model(model);
  struct v_cgraph* gf     = build_graph(model);
  struct v_tensor* result = compute(model, gf);

  v_sched_free(model.sched);
  v_backend_free(model.backend);
  return 0;
}
