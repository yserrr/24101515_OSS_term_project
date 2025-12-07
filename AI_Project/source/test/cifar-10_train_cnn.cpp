#include "v.h"
#include "v_allocator.h"
#include "v-backend.h"
#include "v_vk.h"
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
#include "v_util.h"
#include <fstream>
#include <random>
#include <string>
#include <utility>
#include "v_opt_common.hpp"
#include "v_opt_ctx.hpp"
#include "v_opt_dataset.hpp"
#include <omp.h>

namespace py = pybind11;
#define CIFAR_NTRAIN 50000
#define CIFAR_NTEST  10000
#define CIFAR_NINPUT  (32*32*3)
#define CIFAR_NCLASSES 10
#define CIFAR_NHIDDEN  1000
#define CIFAR_NBATCH_LOGICAL  5000
#define CIFAR_NBATCH_PHYSICAL 5000
#define CIFAR_NCHANEL  =3
#define CIFAR_CNN_NCB 8
#define FORCE_UTF8_CONSOLE
#ifdef FORCE_UTF8_CONSOLE
#include <windows.h>

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
  fputs(text, stderr);
  fflush(stderr);
}

int main() {
  std::vector<v_tensor*> init_tensors;
  struct {
    std::string arch;
    v_backend_sched_t backend_sched;
    v_backend_t backend;
    int nbatch_logical  = CIFAR_NBATCH_LOGICAL;
    int nbatch_physical = CIFAR_NBATCH_PHYSICAL;
    v_tensor* images    = nullptr;
    v_tensor* kernel_1  = nullptr;
    v_tensor* bias_1    = nullptr;
    v_tensor* kernel_2  = nullptr;
    v_tensor* bias_2    = nullptr;
    v_tensor* dense_w   = nullptr;
    v_tensor* dense_b   = nullptr;

    v_tensor* logits              = nullptr;
    v_ctx* ctx_static             = nullptr;
    v_ctx* ctx_compute            = nullptr;
    v_backend_buffer_t buf_static = nullptr;
  } model;

  auto vk         = backend_vk_init(0);
  v_backend_t a[] = {vk};

  auto backend_sched  = v_sched_new(*a, nullptr, 1,v_DEFAULT_GRAPH_SIZE, false, true);
  model.backend_sched = backend_sched;

  int num_tensors = 10;
  v_init_param params{
    /*.mem_size   =*/ v_tensor_over_head() * 1024 * 1024,
    /*.mem_buffer =*/ NULL,
    /*.no_alloc   =*/ true,
  };
  const size_t size_meta = v_DEFAULT_GRAPH_SIZE * v_tensor_over_head() + 10 * graph_overhead();
  model.ctx_compute      = v_ctx_init(params);

  py::scoped_interpreter guard{};
  py::module_ torch      = py::module_::import("torch");
  py::module_ datasets   = py::module_::import("torchvision.datasets");
  py::module_ transforms = py::module_::import("torchvision.transforms");
  py::module_ np         = py::module_::import("numpy");
  auto transform         = transforms.attr("Compose")(py::make_tuple(
    transforms.attr("ToTensor")(),
    transforms.attr("Normalize")(
      py::make_tuple(0.5f, 0.5f, 0.5f), // mean
      py::make_tuple(0.5f, 0.5f, 0.5f) // std
    )
  ));


  auto cifar_train = datasets.attr("CIFAR10")(
    "./data",
    py::arg("train")     = true,
    py::arg("download")  = true,
    py::arg("transform") = transform
  );

  py::object DataLoader = torch.attr("utils").attr("data").attr("DataLoader");
  auto loader           = DataLoader(
    cifar_train,
    py::arg("batch_size") = 1,
    py::arg("shuffle")    = true
  );

  std::vector<std::vector<float>> images;
  std::vector<long> labels;

  for (auto batch : loader) {
    py::tuple pair  = batch.cast<py::tuple>();
    py::object imgs = pair[0];
    py::object lbls = pair[1];

    py::object np_imgs     = imgs.attr("contiguous")().attr("cpu")().attr("numpy")();
    py::array_t<float> arr = np_imgs.cast<py::array_t<float>>();
    py::buffer_info buf    = arr.request();

    float* ptr  = static_cast<float*>(buf.ptr);
    size_t size = 1;
    for (auto s : buf.shape) size *= s;

    std::vector<float> data(ptr, ptr + size);
    images.push_back(std::move(data));

    std::vector<long> lbl_vec = lbls.attr("cpu")().attr("numpy")().cast<std::vector<long>>();
    labels.insert(labels.end(), lbl_vec.begin(), lbl_vec.end());
  }
  v_time_init();

  v_opt_data_set_t dataset = v_opt_dataset_init(v_TYPE_F32,
                                                v_TYPE_F32,
                                                CIFAR_NINPUT,
                                                CIFAR_NCLASSES,
                                                CIFAR_NTRAIN,
                                                1);

  v_tensor* data  = dataset->getDataset();
  v_tensor* label = dataset->getLabels();
  float* buf      = v_get_tdata_f32(data);
  float* lbuf     = v_get_tdata_f32(label);

  #pragma omp parallel for
  for (int64_t iex = 0; iex < data->ne[1]; ++iex) {
    for (int64_t i = 0; i < CIFAR_NINPUT; ++i) {
      buf[iex * CIFAR_NINPUT + i] = images[iex][i];
    }
  }
  #pragma omp parallel for
  for (int64_t iex = 0; iex < label->ne[1]; ++iex) {
    long long actual_class = labels[iex];
    for (int64_t i = 0; i < CIFAR_NCLASSES; ++i) {
      lbuf[iex * CIFAR_NCLASSES + i] = (i == actual_class)
                                         ? 1.0f
                                         : 0.0f;
    }
  }
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<float> nd{0.0f, 1e-2f};
  {
    const size_t size_meta = 1024 * 1024 * 30 * v_tensor_over_head();
    v_init_param params    = {
      /*.mem_size   =*/ size_meta,
      /*.mem_buffer =*/ nullptr,
      /*.no_alloc   =*/ true,
    };
    model.ctx_static = v_ctx_init(params);
  }
  model.kernel_1 = v_new_tensor_4d(model.ctx_static, v_TYPE_F32, 3, 3, 3, 8);
  model.bias_1   = v_new_tensor_3d(model.ctx_static, v_TYPE_F32, 1, 1, 8);
  model.kernel_2 = v_new_tensor_4d(model.ctx_static, v_TYPE_F32, 3, 3, 8, 16);
  model.bias_2   = v_new_tensor_3d(model.ctx_static, v_TYPE_F32, 1, 1, 16);
  model.dense_w  = v_new_tensor_2d(model.ctx_static, v_TYPE_F32, 8 * 8 * 16, 10);
  model.dense_b  = v_new_tensor_1d(model.ctx_static, v_TYPE_F32, 10);


  init_tensors.push_back(model.kernel_1);
  init_tensors.push_back(model.bias_1);
  init_tensors.push_back(model.kernel_2);
  init_tensors.push_back(model.bias_2);
  init_tensors.push_back(model.dense_w);
  init_tensors.push_back(model.dense_b);

  model.images = v_new_tensor_2d(model.ctx_static, v_TYPE_F32, CIFAR_NINPUT,CIFAR_NBATCH_PHYSICAL);
  v_tensor* conv_input = v_reshape_4d(model.ctx_compute, model.images, 32, 32, 3, model.images->ne[1]);
  v_set_name(model.images, "images");
  v_set_inputs(model.images);
  model.buf_static = v_backend_alloc_ctx_tensors(model.ctx_static, vk);

  for (v_tensor* t : init_tensors) {
    V_ASSERT(t->type == v_TYPE_F32);
    const int64_t ne = nelements(t);
    std::vector<float> tmp(ne);
    for (int64_t i = 0; i < ne; ++i) { tmp[i] = nd(gen); }
    v_set_backend_tensor(t, tmp.data(), 0, num_bytes(t));
  }

  v_set_params(model.kernel_1);
  v_set_params(model.bias_1);
  v_set_params(model.kernel_2);
  v_set_params(model.bias_2);
  v_set_params(model.dense_w);
  v_set_params(model.dense_b);
  v_tensor* conv_out1 = v_relu(model.ctx_compute,
                               v_add(model.ctx_compute,
                                     v_conv_2d(model.ctx_compute,
                                               model.kernel_1,
                                               conv_input, 1, 1, 1, 1, 1, 1),
                                     model.bias_1));
  v_tensor* conv_in2  = v_pool_2d(model.ctx_compute, conv_out1, V_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
  v_tensor* conv_out2 = v_relu(model.ctx_compute,
                               v_add(model.ctx_compute,
                                     v_conv_2d(model.ctx_compute, model.kernel_2, conv_in2, 1, 1, 1, 1, 1, 1),
                                     model.bias_2));
  v_tensor* dense_in = v_pool_2d(model.ctx_compute, conv_out2, V_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
  dense_in           = v_reshape_2d(model.ctx_compute,
                          v_mem_cont(model.ctx_compute, v_permute(model.ctx_compute, dense_in, 1, 2, 0, 3))
                          ,64 * 16,CIFAR_NBATCH_PHYSICAL);


  model.logits = v_add(model.ctx_compute,
                       v_matmul(model.ctx_compute,
                                model.dense_w,
                                dense_in),
                       model.dense_b);

  ///not impled yet, need:
  /// im2col :
  ///   backpropagation check
  ///   pool2d_back_check
  ///   not impled back_propagation kernels yet


  v_opt_fit(model.backend_sched,
            model.ctx_compute,
            model.images,
            model.logits,
            dataset,
            V_OPT_LOSS_CROSS_ENTROPY,
            V_OPTIMIZER_TYPE_ADAMW,
            v_opt_get_default_optimizer_params,
            10000,
            model.nbatch_logical,
            0.05f,
            false);
}
