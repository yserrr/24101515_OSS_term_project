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
#include "source/AI/Model.hpp"
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
#include "v_opt_common.hpp"
#include "v_opt_ctx.hpp"
#include "v_opt_dataset.hpp"
#include <omp.h>
namespace py = pybind11;
#define MNIST_NTRAIN 60000
#define MNIST_NTEST  10000
#define MNIST_NBATCH_LOGICAL  15000
#define MNIST_NBATCH_PHYSICAL 15000

#define MNIST_HW       28
#define MNIST_NINPUT   (MNIST_HW*MNIST_HW)
#define MNIST_NCLASSES 10
#define MNIST_NHIDDEN  500
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
    int nbatch_logical             = MNIST_NBATCH_LOGICAL;
    int nbatch_physical            = MNIST_NBATCH_PHYSICAL;
    v_tensor* images               = nullptr;
    v_tensor* descriminator_in     = nullptr;
    v_tensor* descriminator_w1     = nullptr;
    v_tensor* descriminator_b1     = nullptr;
    v_tensor* descriminator_relu   = nullptr;
    v_tensor* descriminator_w2     = nullptr;
    v_tensor* descriminator_b2     = nullptr;
    v_tensor* descriminator_logits = nullptr;

    v_tensor* generator_in   = nullptr;
    v_tensor* generator_w    = nullptr;
    v_tensor* generator_b    = nullptr;
    v_tensor* generator_relu = nullptr;

    ///discriminator:
    /// D_labels = [0,1]
    ///  x -> D(x)  = D_predX;
    ///  x_loss = cross_entropy(x,D(x))
    ///  z_loss = cross_entropy D(G(z) ,0)
    ///  D_loss =x_loss + z_loss
    ///  need: for opt_epoch
    ///    not impled yet


    v_ctx* ctx_static             = nullptr;
    v_ctx* ctx_compute            = nullptr;
    v_backend_buffer_t buf_static = nullptr;
  } model;

  auto vk         = backend_vk_init(0);
  v_backend_t a[] = {vk};

  auto backend_sched  = v_sched_new(*a, nullptr, 1, v_DEFAULT_GRAPH_SIZE, false, true);
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
  // torch, torchvision import
  py::module_ torch      = py::module_::import("torch");
  py::module_ datasets   = py::module_::import("torchvision.datasets");
  py::module_ transforms = py::module_::import("torchvision.transforms");
  py::module_ np         = py::module_::import("numpy");

  auto transform = transforms.attr("Compose")(py::make_tuple(
    transforms.attr("ToTensor")()
  ));

  auto mnist_train = datasets.attr("MNIST")(
    "./data",
    py::arg("train")     = true,
    py::arg("download")  = true,
    py::arg("transform") = transform
  );

  py::object DataLoader = torch.attr("utils").attr("data").attr("DataLoader");
  auto loader           = DataLoader(
    mnist_train,
    py::arg("batch_size") = 1,
    py::arg("shuffle")    = false
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

    std::vector<long> lbl_vec = lbls.attr("cpu")().attr("numpy")().cast<py::array_t<long>>().cast<std::vector<long>>();
    labels.insert(labels.end(), lbl_vec.begin(), lbl_vec.end());
  }
  v_time_init();
  v_opt_data_set_t dataset = v_opt_dataset_init(v_TYPE_F32,
                                                v_TYPE_F32,
                                                MNIST_NINPUT,
                                                MNIST_NCLASSES,
                                                MNIST_NTRAIN,
                                                /*ndata_shard =*/
                                                1);

  v_tensor* data  = dataset->getDataset();
  v_tensor* label = dataset->getLabels();
  float* buf      = v_get_tdata_f32(data);
  float* lbuf     = v_get_tdata_f32(label);
  #pragma omp parallel for
  for (int64_t iex = 0; iex < data->ne[1]; ++iex) {
    for (int64_t i = 0; i < MNIST_NINPUT; ++i) {
      buf[iex * MNIST_NINPUT + i] = images[iex][i];
    }
  }
  #pragma omp parallel for
  for (int64_t iex = 0; iex < label->ne[1]; ++iex) {
    long long actual_class = (long long)labels[iex];
    //lbuf[iex * MNIST_NCLASSES] = float(actual_class);
    for (int64_t i = 0; i < MNIST_NCLASSES; ++i) {
      lbuf[iex * MNIST_NCLASSES + i] = (i == actual_class)
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
}
