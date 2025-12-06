#include "ggml-opt.h"
#include "mnist_common.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif
int main(int argc, char** argv)
{
  if (argc != 5 && argc != 6)
  {
    fprintf(
      stderr,
      "Usage: %s mnist-fc mnist-fc-f32.gguf data/MNIST/raw/train-images-idx3-ubyte data/MNIST/raw/train-labels-idx1-ubyte [CPU/CUDA0]\n",
      argv[0]);
    exit(0);
  }
  ggml_opt_dataset_t dataset = ggml_opt_dataset_init(GGML_TYPE_F32,
                                                     GGML_TYPE_F32,
                                                     MNIST_NINPUT,
                                                     MNIST_NCLASSES,
                                                     MNIST_NTRAIN, /*ndata_shard =*/ 10);


  mnist_model model = mnist_model_init_random(argv[1],
    argc >= 6 ? argv[5] : "", MNIST_NBATCH_LOGICAL,
                                              MNIST_NBATCH_PHYSICAL);

  mnist_model_build(model);

  mnist_model_train(model, dataset, /*nepoch =*/ 30, /*val_split =*/ 0.05f);

  mnist_model_save(model, argv[2]);
}
