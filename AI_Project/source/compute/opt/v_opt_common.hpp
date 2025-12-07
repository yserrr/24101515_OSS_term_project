#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cinttypes>
#include <map>
#include <random>
#include <vector>

#include "v.h"
#include "v-backend.h"
#include <stdint.h>
#include "v.h"
#include "v_allocator.h"
#include "v-backend.h"
#include "ggml-impl.h"
#include "v_vk.h"
typedef struct v_opt_params (*v_opt_get_optimizer_params)(void* userdata);
// returns the default optimizer params (constant, hard-coded values)
// userdata is not used
v_API v_opt_params v_opt_get_default_optimizer_params(void* userdata);
// casts userdata to v_opt_optimizer_params and returns it
v_API v_opt_params v_opt_get_constant_optimizer_params(void* userdata);


struct v_opt_dataset;
struct v_opt_ctx;
struct v_opt_result;
typedef v_opt_dataset* v_opt_data_set_t;
typedef v_opt_ctx* v_opt_context_t;
typedef v_opt_result* v_opt_result_t;

enum v_opt_build_type {
  V_OPT_TYPE_FORWARD    = 10,
  V_OPT_BUILD_TYPE_GRAD = 20,
  V_OPT_TYPE_OPT        = 30,
};

enum v_opt_type {
  V_OPTIMIZER_TYPE_ADAMW,
  V_OPTIMIZER_TYPE_SGD,
  V_OPIMIZER_TYPE_COUNT
};

// parameters that control which optimizer is used and how said optimizer tries to find the minimal loss
struct v_opt_params {
  struct {
    float alpha; // learning rate
    float beta1; // first AdamW momentum
    float beta2; // second AdamW momentum
    float eps; // epsilon for numerical stability
    float wd; // weight decay - 0.0f to disable
  } adamw;

  struct {
    float alpha; // learning rate
    float wd; // weight decay
  } sgd;
};

enum v_opt_loss_type {
  V_OPT_LOSS_MEAN,
  V_OPT_LOSS_SUM,
  V_OPT_LOSS_CROSS_ENTROPY,
  V_OPT_LOSS_SQUARED_ERROR,
};


struct v_opt_struct {
  v_backend_sched_t backend_sched;
  // by default the forward graph needs to be reconstructed for each eval
  // if ctx_compute, inputs, and outputs are set the graphs are instead allocated statically
  v_ctx* ctx_compute;
  v_tensor* inputs;
  v_tensor* outputs;
  v_opt_loss_type loss_type;
  v_opt_build_type build_type;
  int32_t opt_period; // after how many gradient accumulation steps an optimizer step should be done
  v_opt_get_optimizer_params get_opt_pars; // callback for calculating optimizer parameters
  void* get_opt_pars_ud; // userdata for calculating optimizer parameters
  // only v_OPT_OPTIMIZER_TYPE_ADAMW needs m, v momenta per parameter tensor
  v_opt_type optimizer;
};


// ====== Dataset ======
v_API v_opt_data_set_t v_opt_dataset_init(enum v_data_type type_data__, // the type for the internal data tensor
                                          enum v_data_type type_label__, // the type for the internal labels tensor
                                          int64_t ne_datapoint__, // number of elements per datapoint
                                          int64_t ne_label__, // number of elements per label
                                          int64_t ndata__, // total number of datapoints/labels
                                          int64_t ndata_shard__);
// number of datapoints/labels per shard (unit at which the dataset is shuffled/copied)
v_API void v_opt_data_set_free(v_opt_data_set_t dataset);

// get underlying tensors that store the data
v_API int64_t v_opt_dataset_num_data(v_opt_data_set_t dataset);
v_API struct v_tensor* v_opt_dataset_datas(v_opt_data_set_t dataset); // shape = [ne_datapoint, ndata]
v_API struct v_tensor* v_opt_dataset_labels(v_opt_data_set_t dataset); // shape = [nd_label,     ndata]
// shuffle idata first datapoints from dataset with RNG from opt_ctx, shuffle all datapoints if idata is negative
v_API void v_opt_data_set_shuffle(v_opt_context_t opt_ctx, v_opt_data_set_t dataset, int64_t idata);
// get batch at position ibatch from dataset and copy the data to data_batch and labels_batch
v_API void v_opt_dataset_get_batch(v_opt_data_set_t dataset,
                                   struct v_tensor* data_batch, // shape = [ne_datapoint, ndata_batch]
                                   struct v_tensor* labels_batch, // shape = [ne_label,     ndata_batch]
                                   int64_t ibatch);

v_API void v_opt_dataset_get_batch_host(v_opt_data_set_t dataset,
                                        void* data_batch,
                                        size_t nb_data_batch,
                                        void* labels_batch,
                                        int64_t ibatch);


// get parameters for an optimization context with defaults set where possible
// parameters for which no sensible defaults exist are supplied as arguments to this function
v_API struct v_opt_struct v_opt_default_params(v_backend_sched_t backend_sched,
                                               enum v_opt_loss_type loss_type);
v_API v_opt_context_t v_opt_init(struct v_opt_struct params);
// set gradients to zero, initilize loss, and optionally reset the optimizer
v_API void v_opt_reset(v_opt_context_t opt_ctx, bool optimizer);

// get underlying tensors that store data
// if not using static graphs these pointers become invalid with the next call to v_opt_alloc
v_API struct v_tensor* v_opt_inputs(v_opt_context_t opt_ctx); // forward graph input tensor
v_API struct v_tensor* v_opt_outputs(v_opt_context_t opt_ctx); // forward graph output tensor
v_API struct v_tensor* v_opt_labels(v_opt_context_t opt_ctx); // labels to compare outputs against
v_API struct v_tensor* v_opt_loss(v_opt_context_t opt_ctx); // scalar tensor that contains the loss
v_API struct v_tensor* v_opt_pred(v_opt_context_t opt_ctx); // predictions made by outputs
v_API struct v_tensor* v_opt_ncorrect(v_opt_context_t opt_ctx);
// number of matching predictions between outputs and labels

// get the gradient accumulator for a node from the forward graph
v_API struct v_tensor* v_opt_grad_acc(v_opt_context_t opt_ctx, struct v_tensor* node);

v_API enum v_opt_type v_opt_context_optimizer_type(v_opt_context_t);
//TODO consistent naming scheme

v_API const char* v_opt_optimizer_name(enum v_opt_type);

// ====== Optimization Result ======

v_API v_opt_result_t v_opt_result_init(void);
v_API void v_opt_result_free(v_opt_result_t result);
v_API void v_opt_result_reset(v_opt_result_t result);

// get data from result, uncertainties are optional and can be ignored by passing NULL
v_API void v_opt_result_ndata(v_opt_result_t result, int64_t* ndata); // writes 1 value, number of datapoints
v_API void v_opt_result_loss(v_opt_result_t result, double* loss, double* unc); // writes 1 value
v_API void v_opt_result_pred(v_opt_result_t result, int32_t* pred); // writes ndata values
v_API void v_opt_result_accurancy(v_opt_result_t result, double* accuracy, double* unc); // writes 1 value


// allocate the next graph for evaluation, either forward or forward + backward
// must be called exactly once prior to calling v_opt_eval
v_API void v_opt_alloc(v_opt_context_t opt_ctx, bool backward);

// do forward pass, increment result if not NULL, do backward pass if allocated
v_API void v_opt_evaluate(v_opt_context_t opt_ctx, v_opt_result_t result);

// ====== Intended Usage ======
//
// 1. Select the appropriate loss for your problem.
// 2. Create a dataset and set the data for the "data" tensor. Also set the "labels" tensor if your loss needs them.
//    Setting the shard size to 1 will be fine, it's the granularity with which data is shuffled/loaded (bigger values are faster).

// 3. Create a GGML graph for your model with no_alloc == true. Use two separate contexts for the tensors.
//    The first context should contain the model parameters and inputs and be allocated statically in user code.
//    The second context should contain all other tensors and will be (re)allocated automatically.
//    Due to this automated allocation the data of the second context is not defined when accessed in user code.
//    Note that the second dimension of the inputs/outputs are interpreted as the number of datapoints in those tensors.
// 4. Call v_opt_fit. If you need more control you can use v_opt_epoch instead.
// signature for a callback while evaluating opt_ctx on dataset, called after an evaluation
typedef void (*v_opt_epoch_callback)(
  bool train, // true after training evaluation, false after validation evaluation
  v_opt_context_t opt_ctx,
  v_opt_data_set_t dataset,
  v_opt_result_t result, // result associated with the dataset subsection
  int64_t ibatch, // number of batches that have been evaluated so far
  int64_t ibatch_max, // total number of batches in this dataset subsection
  int64_t t_start_us); // time at which the evaluation on the dataset subsection was started

// do training on front of dataset, do evaluation only on back of dataset
v_API void v_opt_epoch(
  v_opt_context_t opt_ctx__,
  v_opt_data_set_t dataset__,
  v_opt_result_t result_train__, // result to increment during training, ignored if NULL
  v_opt_result_t result_eval__, // result to increment during evaluation, ignored if NULL
  int64_t idata_split__, // data index at which to split training and evaluation
  v_opt_epoch_callback callback_train__,
  v_opt_epoch_callback callback_eval__);

// callback that prints a progress bar on stderr
v_API void v_opt_epoch_callback_progress_bar(
  bool train,
  v_opt_context_t opt_ctx,
  v_opt_data_set_t dataset,
  v_opt_result_t result,
  int64_t ibatch,
  int64_t ibatch_max,
  int64_t t_start_us);

// fit model defined by inputs and outputs to dataset
v_API void v_opt_fit(
  v_backend_sched_t backend_sched, // backend scheduler for constructing the compute graphs
  struct v_ctx* ctx_compute, // context with temporarily allocated tensors to calculate the outputs
  struct v_tensor* inputs, // input tensor with shape [ne_datapoint, ndata_batch]
  struct v_tensor* outputs, // output tensor, must have shape [ne_label, ndata_batch] if labels are used
  v_opt_data_set_t dataset, // dataset with data and optionally also labels
  enum v_opt_loss_type loss_type, // loss to minimize
  enum v_opt_type optimizer, // sgd or adamw
  v_opt_get_optimizer_params get_opt_pars,
  // callback to get optimizer params, userdata is pointer to epoch (of type int64_t)
  int64_t nepoch, // how many times the dataset should be iterated over
  int64_t nbatch_logical, // datapoints optimizer step, must be a multiple of ndata_batch in inputs/outputs
  float val_split, // fraction of the dataset to use for validation, must be in [0.0f, 1.0f)
  bool silent);
