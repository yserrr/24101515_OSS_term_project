//
// Created by dlwog on 25. 11. 18..
//

#ifndef MYPROJECT_MML_OPT_CONTEXT_HPP
#define MYPROJECT_MML_OPT_CONTEXT_HPP
#include "v_opt_common.hpp"
#include "v_opt_result.hpp"

/// todo:
///  class setting
///  update setting for ctx
struct v_opt_ctx {
  v_opt_ctx() = default;
  v_opt_ctx(struct v_opt_struct params);
  v_backend_sched_t backend_sched = nullptr;
  v_cgraph* allocated_graph       = nullptr;
  v_cgraph* allocated_graph_copy  = nullptr;
  struct v_ctx* ctx_static        = nullptr;
  struct v_ctx* ctx_gpu           = nullptr;
  struct v_ctx* ctx_compute       = nullptr;
  struct v_ctx* ctx_copy          = nullptr;
  v_backend_buffer_t buf_static   = nullptr;
  v_backend_buffer_t buf_host     = nullptr;
  std::mt19937 rng;
  enum v_opt_loss_type loss_type;
  enum v_opt_build_type build_type;
  enum v_opt_build_type build_type_alloc;
  struct v_tensor* inputs   = nullptr;
  struct v_tensor* outputs  = nullptr;
  struct v_tensor* labels   = nullptr;
  struct v_tensor* loss     = nullptr;
  struct v_tensor* pred     = nullptr;
  struct v_tensor* ncorrect = nullptr;
  struct v_cgraph* gf       = nullptr;
  struct v_cgraph* gb_grad  = nullptr;
  struct v_cgraph* gb_opt   = nullptr;
  bool static_graphs        = false;
  bool eval_ready           = false;
  std::vector<struct v_tensor*> grad_accs;
  std::vector<struct v_tensor*> grad_m;
  std::vector<struct v_tensor*> grad_v;
  int64_t iter                            = 1;
  int32_t opt_period                      = 1;
  int32_t opt_i                           = 0;
  bool loss_per_datapoint                 = false;
  v_opt_get_optimizer_params get_opt_pars = nullptr;
  void* get_opt_pars_ud                   = nullptr;
  struct v_tensor* opt_step_params__      = nullptr; // Stores output of get_opt_pars.
  enum v_opt_type optimizer               = V_OPTIMIZER_TYPE_ADAMW;
  void allocate(bool backward = false);
  void build();
  v_tensor* getInput() { return inputs; }
  v_tensor* getOutput() { return outputs; }
  v_tensor* getLabels() { return labels; }

  v_tensor* getLoss() { return loss; }

  bool isStaticGraph() { return static_graphs; }

  enum v_opt_type getOptType() { return optimizer; }

  const char* get_opt_name() {
    switch (optimizer) {
    case V_OPTIMIZER_TYPE_ADAMW:
      return "adamw";
    case V_OPTIMIZER_TYPE_SGD:
      return "sgd";
    default:
      return "undefined";
    };
  }
  void free() {
    v_backend_buffer_free(this->buf_static);
    v_backend_buffer_free(this->buf_host);
    free_ctx(this->ctx_static);
    free_ctx(this->ctx_gpu);
    delete this;
  }
};

#endif //MYPROJECT_MML_OPT_CONTEXT_HPP
