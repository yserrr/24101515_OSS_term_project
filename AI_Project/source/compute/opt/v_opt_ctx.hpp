#pragma once
/// All code is adapted from ggml for personal educational purposes.(study, clone coding)
/// Core code under license is sourced from ggml (https://github.com/ggerganov/ggml)


#include "v_opt_common.hpp"
#include "v_opt_result.hpp"
/// todo:
///  class setting
///  update setting for ctx
struct v_opt_ctx {
  v_opt_ctx() = default;
  v_opt_ctx(v_opt_struct params);
  v_backend_sched_t backend_sched = nullptr;
  v_cgraph* allocated_graph       = nullptr;
  v_cgraph* allocated_graph_copy  = nullptr;
  v_ctx* ctx_static               = nullptr;
  v_ctx* ctx_gpu                  = nullptr;
  v_ctx* ctx_compute              = nullptr;
  v_ctx* ctx_copy                 = nullptr;
  v_backend_buffer_t buf_static   = nullptr;
  v_backend_buffer_t buf_host     = nullptr;
  v_opt_loss_type loss_type;
  v_opt_build_type build_type;
  v_opt_build_type build_type_alloc;
  std::mt19937 rng;
  v_tensor* inputs   = nullptr;
  v_tensor* outputs  = nullptr;
  v_tensor* labels   = nullptr;
  v_tensor* loss     = nullptr;
  v_tensor* pred     = nullptr;
  v_tensor* ncorrect = nullptr;
  v_cgraph* gf       = nullptr;
  v_cgraph* gb_grad  = nullptr;
  v_cgraph* gb_opt   = nullptr;
  bool static_graphs = false;
  bool eval_ready    = false;
  std::vector<v_tensor*> grad_accs;
  std::vector<v_tensor*> grad_m;
  std::vector<v_tensor*> grad_v;
  int64_t iter                            = 1;
  int32_t opt_period                      = 1;
  int32_t opt_i                           = 0;
  bool loss_per_datapoint                 = false;
  v_opt_get_optimizer_params get_opt_pars = nullptr;
  void* get_opt_pars_ud                   = nullptr;
  v_tensor* opt_step_params__             = nullptr; // Stores output of get_opt_pars.
  v_opt_type optimizer                    = V_OPTIMIZER_TYPE_ADAMW;
  void allocate(bool backward = false);
  void build();
  v_tensor* get_input() { return inputs; }
  v_tensor* get_output() { return outputs; }
  v_tensor* get_labels() { return labels; }
  v_tensor* get_loss() { return loss; }
  bool is_static_graph() { return static_graphs; }
  v_opt_type get_opt_type() { return optimizer; }
  const char* get_opt_name();
  void free();
};

