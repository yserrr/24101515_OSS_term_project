#include "v_opt_dataset.hpp"
#include "v_opt_ctx.hpp"

v_opt_data_set_t v_opt_dataset_init(enum v_data_type type_data__,
                                   enum v_data_type type_label__,
                                   int64_t ne_datapoint__,
                                   int64_t ne_label__,
                                   int64_t ndata__,
                                   int64_t ndata_shard__) {
  V_ASSERT(ne_datapoint__ > 0);
  V_ASSERT(ne_label__ >= 0);
  V_ASSERT(ndata__ > 0);
  V_ASSERT(ndata_shard__ > 0);
  v_opt_data_set_t result = new v_opt_dataset;
  result->ndata__        = ndata__;
  result->ndata_shard    = ndata_shard__;
  {
    struct v_init_param params = {
      .mem_size = 2 * v_tensor_over_head(),
      .mem_buffer = nullptr,
      .no_alloc = true,
    };
    result->ctx = v_ctx_init(params);
  }
  result->data = v_new_tensor_2d(result->ctx,
                                 type_data__,
                                 ne_datapoint__,
                                 ndata__);

  result->nbs_data = num_bytes(result->data) * ndata_shard__ / ndata__;

  if (ne_label__ > 0) {
    result->labels = v_new_tensor_2d(result->ctx,
                                     type_label__,
                                     ne_label__,
                                     ndata__);

    result->nbs_labels = num_bytes(result->labels) * ndata_shard__ / ndata__;
  }
  else {
    result->labels     = nullptr;
    result->nbs_labels = 0;
  }

  result->buf           = v_backend_alloc_ctx_tensor_from_buffer_t(result->ctx, vk_host_buffer_type());
  const int64_t nshards = ndata__ / ndata_shard__;
  result->permutation.resize(nshards);
  for (int64_t i = 0; i < nshards; ++i) { result->permutation[i] = i; }
  return result;
}


struct v_tensor* v_opt_dataset_datas(v_opt_data_set_t dataset) { return dataset->data; }

struct v_tensor* v_opt_dataset_labels(v_opt_data_set_t dataset) { return dataset->labels; }


void ggml_opt_dataset_get_batch_host(v_opt_data_set_t dataset,
                                     void* data_batch,
                                     size_t nb_data_batch,
                                     void* labels_batch,
                                     int64_t ibatch) {
  V_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
  V_ASSERT(nb_data_batch % dataset->nbs_data == 0);
  const int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;
  V_ASSERT((ibatch + 1) * shards_per_batch <= int64_t(dataset->permutation.size()));
  for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
    const int64_t ishard = dataset->permutation[ibatch * shards_per_batch + ishard_batch];
    const char* ptr_data = (const char*)dataset->data->data + ishard * dataset->nbs_data;
    char* ptr_data_batch = (char*)data_batch + ishard_batch * dataset->nbs_data;
    memcpy(ptr_data_batch, ptr_data, dataset->nbs_data);
    if (!labels_batch) { continue; }
    const char* ptr_labels = (const char*)dataset->labels->data + ishard * dataset->nbs_labels;
    char* ptr_labels_batch = (char*)labels_batch + ishard_batch * dataset->nbs_labels;
    memcpy(ptr_labels_batch, ptr_labels, dataset->nbs_labels);
  }
}

void v_opt_dataset::shuffle(v_opt_ctx* ctx, int64_t idata) {
  V_ASSERT(idata <=ndata__);
  if (idata < 0) {
    std::shuffle(permutation.begin(),
                 permutation.end(),
                 ctx->rng);
    return;
  }
  V_ASSERT(idata % ndata_shard == 0);
  const int64_t ishard_max = idata / ndata_shard;
  std::shuffle(permutation.begin(),
               permutation.begin() + ishard_max,
               ctx->rng);
}

void v_opt_dataset::getBatch(struct v_tensor* data_batch, struct v_tensor* labels_batch, int64_t ibatch) {
  auto dataset = this;

  V_ASSERT(data_batch && v_is_contiguous(data_batch));
  V_ASSERT(!labels_batch || v_is_contiguous(labels_batch));
  V_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
  V_ASSERT(data_batch->type == dataset->data->type);
  V_ASSERT(!labels_batch || labels_batch->type == dataset->labels->type);

  const size_t nb_data_batch = num_bytes(data_batch);
  V_ASSERT(nb_data_batch % dataset->nbs_data == 0);
  const int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;
  if (labels_batch) {
    const size_t nb_labels_batch = num_bytes(labels_batch);
    V_ASSERT(nb_labels_batch == shards_per_batch * dataset->nbs_labels);
  }
  V_ASSERT((ibatch + 1) * shards_per_batch <= int64_t(dataset->permutation.size()));
  for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
    const int64_t ishard = dataset->permutation[ibatch * shards_per_batch + ishard_batch];

    const char* ptr_data = (const char*)dataset->data->data + ishard * dataset->nbs_data;
    v_set_backend_tensor(data_batch,
                         ptr_data,
                         ishard_batch * dataset->nbs_data,
                         dataset->nbs_data);
    if (!labels_batch) { continue; }
    const char* ptr_labels = (const char*)dataset->labels->data
      + ishard * dataset->nbs_labels;
    v_set_backend_tensor(labels_batch,
                         ptr_labels,
                         ishard_batch * dataset->nbs_labels,
                         dataset->nbs_labels);
  }
}
