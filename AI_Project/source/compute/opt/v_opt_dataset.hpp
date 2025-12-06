#ifndef MYPROJECT_MML_OPT_DATASET_HPP
#define MYPROJECT_MML_OPT_DATASET_HPP
#include "v_opt_common.hpp"

struct v_opt_dataset
{
  struct v_ctx* ctx = nullptr;
  v_backend_buffer_t buf = nullptr;
  struct v_tensor* data = nullptr;
  struct v_tensor* labels = nullptr;
  int64_t ndata__ = -1;
  int64_t ndata_shard = -1;
  size_t nbs_data = -1;
  size_t nbs_labels = -1;
  std::vector<int64_t> permutation;
  void shuffle(v_opt_ctx* ctx, int64_t idata);

  void free()
  {
    v_backend_buffer_free(buf);
    free_ctx(ctx);
    delete this;
  }

  uint64_t ndata()
  {
    return ndata__;
  }

  void get_batch(struct v_tensor* data_batch,
                struct v_tensor* labels_batch,
                int64_t ibatch);
  struct v_tensor* getDataset()
  {
    return data;
  }
  struct v_tensor* getLabels()
  {
    return labels;
  }
};


#endif //MYPROJECT_MML_OPT_DATASET_HPP
