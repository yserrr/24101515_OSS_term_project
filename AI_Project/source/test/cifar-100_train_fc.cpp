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
#include "source/AI/Model.hpp"
#include "extern/pybind11/pybind11.h"
#include "pybind11/embed.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include <algorithm>
#include <cstdint>
#include <random>
#include <utility>
#include "v_opt_common.hpp"
#include "v_opt_ctx.hpp"
#include "v_opt_dataset.hpp"
#include <omp.h>

namespace py = pybind11;

#define CIFAR_NTRAIN 50000
#define CIFAR_NTEST  10000
#define CIFAR_NINPUT  (32*32*3)
#define CIFAR_NCLASSES 100
#define CIFAR_NHIDDEN  3000
#define CIFAR_NBATCH_LOGICAL  5000
#define CIFAR_NBATCH_PHYSICAL 5000

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
    int nbatch_logical            = CIFAR_NBATCH_LOGICAL;
    int nbatch_physical           = CIFAR_NBATCH_PHYSICAL;
    v_tensor* images              = nullptr;
    v_tensor* logits              = nullptr;
    v_tensor* fc1_weight          = nullptr;
    v_tensor* fc1_bias            = nullptr;
    v_tensor* fc2_weight          = nullptr;
    v_tensor* fc2_bias            = nullptr;
    v_ctx* ctx_static             = nullptr;
    v_ctx* ctx_compute            = nullptr;
    v_backend_buffer_t buf_static = nullptr;
  } model;

  auto vk         = backend_vk_init(0);
  v_backend_t a[] = {vk};

  auto backend_sched  = v_sched_new(*a, nullptr, 1,GGML_DEFAULT_GRAPH_SIZE, false, true);
  model.backend_sched = backend_sched;

  int num_tensors = 10;
  v_init_param params{
    /*.mem_size   =*/ v_tensor_over_head() * 1024 * 1024,
    /*.mem_buffer =*/ NULL,
    /*.no_alloc   =*/ true,
  };
  const size_t size_meta = GGML_DEFAULT_GRAPH_SIZE * v_tensor_over_head() + 10 * graph_overhead();
  model.ctx_compute      = v_ctx_init(params);

  py::scoped_interpreter guard{};
  py::module_ torch      = py::module_::import("torch");
  py::module_ datasets   = py::module_::import("torchvision.datasets");
  py::module_ transforms = py::module_::import("torchvision.transforms");
  py::module_ np         = py::module_::import("numpy");

  auto transform         = transforms.attr("Compose")(py::make_tuple(
    transforms.attr("ToTensor")(),
    transforms.attr("Normalize")(
      py::make_tuple(0.5071f, 0.4865f, 0.4409f),
      py::make_tuple(0.2673f, 0.2564f, 0.2761f)
    )
  ));

  auto cifar_train = datasets.attr("CIFAR100")(
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
  images.reserve(CIFAR_NTRAIN);
  labels.reserve(CIFAR_NTRAIN);

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

    if ((int)images.size() >= CIFAR_NTRAIN) break;
  }
  v_time_init();

  v_opt_data_set_t dataset = v_opt_dataset_init(GGML_TYPE_F32,
                                                GGML_TYPE_F32,
                                                CIFAR_NINPUT,
                                                CIFAR_NCLASSES,
                                                CIFAR_NTRAIN,
                                                1);

  v_tensor* data  = dataset->getDataset();
  v_tensor* label = dataset->getLabels();
  float* buf      = v_get_tdata_f32(data);
  float* lbuf     = v_get_tdata_f32(label);
  int64_t nfill = std::min<int64_t>(data->ne[1], (int64_t)images.size());

  #pragma omp parallel for
  for (int64_t iex = 0; iex < nfill; ++iex) {
    for (int64_t i = 0; i < CIFAR_NINPUT; ++i) {
      buf[iex * CIFAR_NINPUT + i] = images[iex][i];
    }
    // 남아있는 경우 0으로 패딩할 필요 없음(위에서 nfill로 제한)
  }

  #pragma omp parallel for
  for (int64_t iex = 0; iex < nfill; ++iex) {
    long long actual_class = labels[iex];
    for (int64_t i = 0; i < CIFAR_NCLASSES; ++i) {
      lbuf[iex * CIFAR_NCLASSES + i] = (i == actual_class) ? 1.0f : 0.0f;
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

  model.fc1_weight = v_new_tensor_2d(model.ctx_static, GGML_TYPE_F32, CIFAR_NINPUT, CIFAR_NHIDDEN);
  model.fc1_bias   = v_new_tensor_1d(model.ctx_static, GGML_TYPE_F32, CIFAR_NHIDDEN);
  model.fc2_weight = v_new_tensor_2d(model.ctx_static, GGML_TYPE_F32, CIFAR_NHIDDEN, CIFAR_NCLASSES);
  model.fc2_bias   = v_new_tensor_1d(model.ctx_static, GGML_TYPE_F32, CIFAR_NCLASSES);

  init_tensors.push_back(model.fc1_weight);
  init_tensors.push_back(model.fc1_bias);
  init_tensors.push_back(model.fc2_weight);
  init_tensors.push_back(model.fc2_bias);
  model.images = v_new_tensor_2d(model.ctx_static, GGML_TYPE_F32, CIFAR_NINPUT, CIFAR_NBATCH_PHYSICAL);
  v_set_name(model.images, "images");
  v_set_inputs(model.images);

  model.buf_static = v_backend_alloc_ctx_tensors(model.ctx_static, vk);
  for (v_tensor* t : init_tensors) {
    V_ASSERT(t->type == GGML_TYPE_F32);
    const int64_t ne = nelements(t);
    std::vector<float> tmp(ne);
    for (int64_t i = 0; i < ne; ++i) { tmp[i] = nd(gen); }
    v_set_backend_tensor(t, tmp.data(), 0, num_bytes(t));
  }
  v_set_params(model.fc1_weight);
  v_set_params(model.fc1_bias);
  v_set_params(model.fc2_weight);
  v_set_params(model.fc2_bias);

  v_tensor* fc1 = v_relu(model.ctx_compute,
                         v_add(model.ctx_compute,
                               v_matmul(model.ctx_compute,
                                        model.fc1_weight,
                                        model.images),
                               model.fc1_bias));

  model.logits = v_add(model.ctx_compute,
                       v_matmul(model.ctx_compute,
                                model.fc2_weight,
                                fc1),
                       model.fc2_bias);

  // optimizer 설정: epochs/steps 등은 v_opt_fit 내부에서 사용하는 인자임
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

  return 0;
}
