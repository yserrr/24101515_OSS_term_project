#include "pybind11/embed.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
namespace py = pybind11;
int main()
{
  py::scoped_interpreter guard{};
  // torch, torchvision import
  py::module torch = py::module::import("torch");
  py::module datasets = py::module::import("torchvision.datasets");
  py::module transforms = py::module::import("torchvision.transforms");
  py::module np = py::module::import("numpy");
  py::module plt = py::module::import("matplotlib.pyplot");

  auto transform = transforms.attr("Compose")(py::make_tuple(
    transforms.attr("ToTensor")()
  ));

  auto mnist_train = datasets.attr("MNIST")(
    "./data",
    py::arg("train") = true,
    py::arg("download") = true,
    py::arg("transform") = transform
  );

  py::object DataLoader = torch.attr("utils").attr("data").attr("DataLoader");
  auto loader = DataLoader(
    mnist_train,
    py::arg("batch_size") = 1000,
    py::arg("shuffle") = false
  );

  std::vector<std::vector<float>> images;
  std::vector<long> labels;
  for (auto batch : loader)
  {
    py::tuple pair = batch.cast<py::tuple>();
    py::object imgs = pair[0];
    py::object lbls = pair[1];

    py::object np_imgs = imgs.attr("contiguous")().attr("cpu")().attr("numpy")();
    py::array_t<float> arr = np_imgs.cast<py::array_t<float>>();
    py::buffer_info buf = arr.request();

    float* ptr = static_cast<float*>(buf.ptr);
    size_t size = 1;
    for (auto s : buf.shape) size *= s;

    std::vector<float> data(ptr, ptr + size);
    images.push_back(std::move(data));

    std::vector<long> lbl_vec = lbls.attr("cpu")().attr("numpy")().cast<py::array_t<long>>().cast<std::vector<long>>();
    labels.insert(labels.end(), lbl_vec.begin(), lbl_vec.end());
  }

  int n = 10; // 예시 10장
  for (int i = 0; i < n; ++i)
  {
    plt.attr("subplot")(2, 5, i + 1); // 2x5 그리드
    py::array_t<float> arr({28, 28}, images[i].data());
    plt.attr("imshow")(arr, py::arg("cmap") = "gray");
    plt.attr("title");
    plt.attr("axis")("off");
  }
  plt.attr("show")();
}
