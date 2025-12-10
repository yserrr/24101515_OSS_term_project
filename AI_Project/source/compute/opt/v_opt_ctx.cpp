#include "v_opt_common.hpp"
#include "v_opt_ctx.hpp"
#include "v_propagation.hpp"
#include "v_opt_dataset.hpp"
#include "v_opt_result.hpp"
#include "v_util.hpp"

void v_opt_reset(v_opt_ctx* opt_ctx, bool optimizer) {
  if (optimizer) {
    v_graph_reset(opt_ctx->gb_opt);
    opt_ctx->iter = 1;
  }
  else v_graph_reset(opt_ctx->gb_grad);
}


v_tensor* v_map_tensor(std::map<v_tensor*, v_tensor*>& tensor_map, v_ctx* ctx, v_tensor* tensor) {
  if (!tensor) return nullptr;
  if (tensor_map.find(tensor) != tensor_map.end()) return tensor_map[tensor];
  v_tensor* new_tensor = v_dup_tensor(ctx, tensor);
  tensor_map[tensor]   = new_tensor;
  new_tensor->op       = tensor->op;
  for (int i = 0; i < V_MAX_DIMS; i++) { new_tensor->nb[i] = tensor->nb[i]; }
  new_tensor->flags = tensor->flags;

  memcpy(new_tensor->op_params.data(), tensor->op_params.data(), sizeof(tensor->op_params));
  strcpy(new_tensor->name.data(), tensor->name.data());
  new_tensor->data      = tensor->data;
  new_tensor->buffer    = tensor->buffer;
  new_tensor->view_offs = tensor->view_offs;
  new_tensor->view_src  = v_map_tensor(tensor_map, ctx, tensor->view_src);
  for (int i = 0; i < V_MAX_SRC; i++) {
    new_tensor->src[i] = v_map_tensor(tensor_map,
                                      ctx,
                                      tensor->src[i]);
  }

  return new_tensor;
}


v_opt_ctx* v_opt_init(v_opt_struct params) {
  v_opt_ctx* result        = new struct v_opt_ctx();
  result->backend_sched    = params.backend_sched;
  result->ctx_compute      = params.ctx_compute;
  result->loss_type        = params.loss_type;
  result->build_type       = params.build_type;
  result->build_type_alloc = params.build_type;
  result->inputs           = params.inputs;
  result->outputs          = params.outputs;
  result->opt_period       = params.opt_period;
  result->get_opt_pars     = params.get_opt_pars;
  result->get_opt_pars_ud  = params.get_opt_pars_ud;
  result->optimizer        = params.optimizer;
  V_ASSERT(result->opt_period >= 1);
  result->static_graphs = result->ctx_compute;

  if (!result->static_graphs) {
    V_ASSERT(!result->inputs);
    V_ASSERT(!result->outputs);
    return result;
  }

  V_ASSERT(result->inputs);
  V_ASSERT(result->outputs);
  result->gf = v_new_graph_custom(result->ctx_compute, v_DEFAULT_GRAPH_SIZE, /*grads =*/ true); // Forward pass.
  v_build_foward_expand(result->gf, result->outputs);
  result->build();
  return result;
}


void v_opt_evaluate(v_opt_ctx* opt_ctx,
                    v_opt_result_t result) {
  V_ASSERT(opt_ctx->eval_ready);
  if (opt_ctx->allocated_graph == opt_ctx->gb_opt) {
    const v_opt_params& opt_pars = opt_ctx->get_opt_pars(opt_ctx->get_opt_pars_ud);
    switch (opt_ctx->optimizer) {
      case V_OPTIMIZER_TYPE_ADAMW: {
        V_ASSERT(opt_pars.adamw.alpha > 0.0f);
        V_ASSERT(opt_pars.adamw.beta1 >= 0.0f);
        V_ASSERT(opt_pars.adamw.beta1 <= 1.0f);
        V_ASSERT(opt_pars.adamw.beta2 >= 0.0f);
        V_ASSERT(opt_pars.adamw.beta2 <= 1.0f);
        V_ASSERT(opt_pars.adamw.eps >= 0.0f);
        V_ASSERT(opt_pars.adamw.wd >= 0.0f);
        V_ASSERT(opt_pars.adamw.wd <= 1.0f);
        const float beta1h = 1.0f / (1.0f - powf(opt_pars.adamw.beta1, opt_ctx->iter));
        const float beta2h = 1.0f / (1.0f - powf(opt_pars.adamw.beta2, opt_ctx->iter));
        std::array<float, 7> adamw_data{};
        adamw_data[0] = opt_pars.adamw.alpha;
        adamw_data[1] = opt_pars.adamw.beta1;
        adamw_data[2] = opt_pars.adamw.beta2;
        adamw_data[3] = opt_pars.adamw.eps;
        adamw_data[4] = opt_pars.adamw.wd;
        adamw_data[5] = beta1h;
        adamw_data[6] = beta2h;
        v_set_backend_tensor(opt_ctx->opt_step_params__, adamw_data.data(), 0, num_bytes(opt_ctx->opt_step_params__));
      }
      break;
      case V_OPTIMIZER_TYPE_SGD: {
        V_ASSERT(opt_pars.sgd.alpha > 0.0f);
        V_ASSERT(opt_pars.sgd.wd >= 0.0f);
        V_ASSERT(opt_pars.sgd.wd <= 1.0f);
        float sgd[] = {opt_pars.sgd.alpha, opt_pars.sgd.wd};
        v_set_backend_tensor(opt_ctx->opt_step_params__, sgd, 0, 8);
      }
      break;
      default:
        v_ABORT("fatal error");
    }
  }
  v_sched_graph_compute(opt_ctx->backend_sched,
                        opt_ctx->allocated_graph_copy);

  opt_ctx->iter += opt_ctx->allocated_graph == opt_ctx->gb_opt;
  opt_ctx->opt_i = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
  if (!opt_ctx->static_graphs) {
    opt_ctx->gf                   = nullptr;
    opt_ctx->gb_grad              = nullptr;
    opt_ctx->gb_opt               = nullptr;
    opt_ctx->allocated_graph      = nullptr;
    opt_ctx->allocated_graph_copy = nullptr;
  }
  opt_ctx->eval_ready = false;
  if (!result) return;
  if (result->ndata == 0) {
    result->loss_per_datapoint = opt_ctx->loss_per_datapoint;
    result->opt_period         = opt_ctx->opt_period;
  }
  else {
    V_ASSERT(result->loss_per_datapoint == opt_ctx->loss_per_datapoint);
    V_ASSERT(result->opt_period == opt_ctx->opt_period);
  }
  const int64_t ndata = opt_ctx->outputs->ne[1];
  V_ASSERT(result->ndata == ndata * int64_t(result->loss.size()) && "varying batch size not supported");
  result->ndata += ndata;
  V_ASSERT(v_is_scalar(opt_ctx->loss));
  V_ASSERT(opt_ctx->loss->type == v_TYPE_F32);
  float loss;
  v_get_backend_tensor(opt_ctx->loss, &loss, 0, num_bytes(opt_ctx->loss));
  result->loss.push_back(loss);
  if (opt_ctx->pred) {
    V_ASSERT(opt_ctx->pred->type == v_TYPE_I32);
    std::vector<int32_t> pred(ndata);
    //v_print_tensor2d(opt_ctx->pred);
    v_get_backend_tensor(opt_ctx->pred, pred.data(), 0, num_bytes(opt_ctx->pred));
    result->pred.insert(result->pred.end(), pred.begin(), pred.end());
  }
  if (!opt_ctx->ncorrect || result->ncorrect < 0) {
    result->ncorrect = -1;
    return;
  }
  V_ASSERT(v_is_scalar(opt_ctx->ncorrect));
  V_ASSERT(opt_ctx->ncorrect->type == v_TYPE_I64);
  int64_t ncorrect;
  v_get_backend_tensor(opt_ctx->ncorrect, &ncorrect, 0, num_bytes(opt_ctx->ncorrect));
  result->ncorrect += ncorrect;
}

struct v_opt_params v_opt_get_default_optimizer_params(void* userdata) {
  V_UNUSED(userdata);
  v_opt_params result{};
  result.adamw.alpha = 1e-3;
  result.adamw.beta1 = 0.95f;
  result.adamw.beta2 = 0.9995f;
  result.adamw.eps   = 1e-4;
  result.adamw.wd    = 0.0000;

  result.sgd.alpha = 1e-3f;
  result.sgd.wd    = 0.0001;
  return result;
}


v_opt_params v_opt_get_constant_optimizer_params(void* userdata) { return *static_cast<v_opt_params*>(userdata); }
v_opt_struct v_opt_default_params(v_backend_sched_t backend_sched, v_opt_loss_type loss_type) {
  return {
    /*backend_sched   =*/ backend_sched,
    /*ctx_compute     =*/ nullptr,
    /*inputs          =*/ nullptr,
    /*logits          =*/ nullptr,
    /*loss_type       =*/ loss_type,
    /*build_type      =*/ V_OPT_TYPE_OPT,
    /*opt_period      =*/ 1,
    /*get_opt_pars    =*/ v_opt_get_default_optimizer_params,
    /*get_opt_pars_ud =*/ nullptr,
    /*optimizer       =*/ V_OPTIMIZER_TYPE_ADAMW,
  };
}


void v_opt_epoch(v_opt_ctx* opt_ctx__,
                 v_opt_data_set_t dataset__,
                 v_opt_result_t result_train__,
                 v_opt_result_t result_eval__,
                 int64_t idata_split__,
                 v_opt_epoch_callback callback_train__,
                 v_opt_epoch_callback callback_eval__) {
  V_ASSERT((opt_ctx__)->isStaticGraph() && "v_opt_epoch requires static graphs");
  v_tensor* inputs = opt_ctx__->getInput();
  v_tensor* labels = opt_ctx__->getLabels();
  v_tensor* data   = v_opt_dataset_datas(dataset__);

  V_ASSERT(data->ne[0] == inputs->ne[0]);

  const int64_t ndata       = data->ne[1];
  const int64_t ndata_batch = inputs->ne[1];
  V_ASSERT(data->ne[1] % inputs->ne[1] == 0);
  const int64_t nbatches = ndata / ndata_batch;

  idata_split__ = idata_split__ < 0 ? ndata : idata_split__;

  V_ASSERT(idata_split__ % ndata_batch == 0);

  const int64_t ibatch_split = idata_split__ / ndata_batch;
  int64_t batch_idx          = 0;
  int64_t t_loop_start       = v_time_us();
  for (; batch_idx < ibatch_split; ++batch_idx) {
    opt_ctx__->allocate(/*backward =*/ true);

    dataset__->get_batch(inputs, labels, batch_idx);

    v_opt_evaluate(opt_ctx__,
                   result_train__);

    if (callback_train__) {
      callback_train__(true,
                       opt_ctx__,
                       dataset__,
                       result_train__,
                       batch_idx + 1,
                       ibatch_split,
                       t_loop_start);
    }
  }
  t_loop_start = v_time_us();
  for (; batch_idx < nbatches; ++batch_idx) {
    opt_ctx__->allocate(/*backward =*/ false);
    dataset__->get_batch(inputs, labels, batch_idx);
    v_opt_evaluate(opt_ctx__,
                   result_eval__);
    if (callback_eval__) {
      callback_eval__(false,
                      opt_ctx__,
                      dataset__,
                      result_eval__,
                      batch_idx + 1 - ibatch_split,
                      nbatches - ibatch_split,
                      t_loop_start);
    }
  }
}

void v_opt_epoch_callback_progress_bar(bool train,
                                       v_opt_ctx* opt_ctx,
                                       v_opt_data_set_t dataset,
                                       v_opt_result_t result,
                                       int64_t batch_idx,
                                       int64_t ibatch_max,
                                       int64_t t_start_us) {
  fprintf(stderr, "%s[", train
                           ? "train: "
                           : "val:   ");
  // The progress bar consists of partially filled blocks, unicode has 8 separate fill levels.
  constexpr int64_t bar_length = 8;
  const int64_t ibatch8        = 8 * batch_idx;
  for (int64_t j = 0; j < bar_length; ++j) {
    if (ibatch_max * (8 * j + 8) / bar_length < ibatch8) fprintf(stderr, "\u2588"); // full block
    else if (ibatch_max * (8 * j + 7) / bar_length < ibatch8) fprintf(stderr, "\u2589"); // 7/8 filled
    else if (ibatch_max * (8 * j + 6) / bar_length < ibatch8) fprintf(stderr, "\u258A"); // 6/8 filled
    else if (ibatch_max * (8 * j + 5) / bar_length < ibatch8) fprintf(stderr, "\u258B"); // 5/8 filled
    else if (ibatch_max * (8 * j + 4) / bar_length < ibatch8) fprintf(stderr, "\u258C"); // 4/8 filled
    else if (ibatch_max * (8 * j + 3) / bar_length < ibatch8) fprintf(stderr, "\u258D"); // 3/8 filled
    else if (ibatch_max * (8 * j + 2) / bar_length < ibatch8) fprintf(stderr, "\u258E"); // 2/8 filled
    else if (ibatch_max * (8 * j + 1) / bar_length < ibatch8) fprintf(stderr, "\u258F"); // 1/8 filled
    else fprintf(stderr, " ");
  }

  const int64_t batch_size = (opt_ctx)->getInput()->ne[1];
  const int64_t idata      = batch_idx * batch_size;
  const int64_t idata_max  = ibatch_max * batch_size;

  double loss;
  double loss_unc;
  v_opt_result_loss(result, &loss, &loss_unc);

  double accuracy;
  double accuracy_unc;

  v_opt_result_accurancy(result, &accuracy, &accuracy_unc);

  const int64_t t_ibatch_us = v_time_us() - t_start_us;
  int64_t t_ibatch_s        = t_ibatch_us / 1000000;
  const int64_t t_ibatch_h  = t_ibatch_s / 3600;
  t_ibatch_s -= t_ibatch_h * 3600;
  const int64_t t_ibatch_m = t_ibatch_s / 60;
  t_ibatch_s -= t_ibatch_m * 60;

  const int64_t t_eta_us = t_ibatch_us * (ibatch_max - batch_idx) / batch_idx;
  int64_t t_eta_s        = t_eta_us / 1000000;
  const int64_t t_eta_h  = t_eta_s / 3600;
  t_eta_s -= t_eta_h * 3600;
  const int64_t t_eta_m = t_eta_s / 60;
  t_eta_s -= t_eta_m * 60;

  fprintf(stderr,
          "] data=%07" PRId64 "/%07" PRId64 " loss=%.5lf±%.5lf acc=%.2lf±%.2lf%% "
          "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " \r",
          idata,
          idata_max,
          loss,
          loss_unc,
          100.0 * accuracy,
          100.0 * accuracy_unc,
          t_ibatch_h,
          t_ibatch_m,
          t_ibatch_s,
          t_eta_h,
          t_eta_m,
          t_eta_s);

  if (batch_idx == ibatch_max) fprintf(stderr, "\n");
  fflush(stderr);

  V_UNUSED(dataset);
}

void v_opt_fit(v_backend_sched_t backend_sched,
               v_ctx* ctx_compute,
               v_tensor* inputs,
               v_tensor* outputs,
               v_opt_data_set_t dataset,
               v_opt_loss_type loss_type,
               v_opt_type optimizer,
               v_opt_get_optimizer_params get_opt_pars,
               int64_t nepoch,
               int64_t nbatch_logical,
               float val_split,
               bool silent) {
  v_time_init();
  const int64_t t_start_us = v_time_us();
  const int64_t ndata      = v_opt_dataset_datas(dataset)->ne[1];

  const int64_t nbatch_physical = inputs->ne[1];
  V_ASSERT(ndata % nbatch_logical == 0);
  V_ASSERT(nbatch_logical % nbatch_physical == 0);
  const int64_t opt_period       = nbatch_logical / nbatch_physical;
  const int64_t nbatches_logical = ndata / nbatch_logical;
  V_ASSERT(val_split >= 0.0f);
  V_ASSERT(val_split < 1.0f);
  const int64_t ibatch_split = int64_t(((1.0f - val_split) * nbatches_logical)) * opt_period;
  // train <-> val split index (physical)
  const int64_t idata_split = ibatch_split * nbatch_physical;
  int64_t epoch             = 1;

  v_opt_struct params    = v_opt_default_params(backend_sched, loss_type);
  params.ctx_compute     = ctx_compute;
  params.inputs          = inputs;
  params.outputs         = outputs;
  params.opt_period      = opt_period;
  params.get_opt_pars    = get_opt_pars;
  params.get_opt_pars_ud = &epoch;
  params.optimizer       = optimizer;
  v_opt_ctx* opt_ctx     = v_opt_init(params);

  if (nbatch_logical < ndata) dataset->shuffle(opt_ctx, -1);

  v_opt_result_t result_train         = v_opt_result_init();
  v_opt_result_t result_val           = v_opt_result_init();
  v_opt_epoch_callback epoch_callback = silent
                                          ? nullptr
                                          : v_opt_epoch_callback_progress_bar;
  for (; epoch <= nepoch; ++epoch) {
    if (nbatch_logical < idata_split) { dataset->shuffle(opt_ctx, -idata_split); }
    result_train->reset();
    result_val->reset();
    if (!silent) { fprintf(stderr, "%s: epoch %04" PRId64 "/%04" PRId64 ":\n", __func__, epoch, nepoch); }
    v_opt_epoch(opt_ctx,
                dataset,
                result_train,
                result_val,
                idata_split,
                epoch_callback,
                epoch_callback);
    if (!silent) { fprintf(stderr, "\n"); }
  }
  if (!silent) {
    int64_t t_total_s       = (v_time_us() - t_start_us) / 1000000;
    const int64_t t_total_h = t_total_s / 3600;
    t_total_s -= t_total_h * 3600;
    const int64_t t_total_m = t_total_s / 60;
    t_total_s -= t_total_m * 60;
    fprintf(stderr,
            "%s: training took %02" PRId64 ":%02" PRId64 ":%02" PRId64 "\n",
            __func__,
            t_total_h,
            t_total_m,
            t_total_s);
  }
  opt_ctx->free();
  result_train->reset();
  result_val->reset();
}


v_opt_ctx::v_opt_ctx(struct v_opt_struct params) {
  v_opt_ctx* result        = new struct v_opt_ctx;
  result->backend_sched    = params.backend_sched;
  result->ctx_compute      = params.ctx_compute;
  result->loss_type        = params.loss_type;
  result->build_type       = params.build_type;
  result->build_type_alloc = params.build_type;
  result->inputs           = params.inputs;
  result->outputs          = params.outputs;
  result->opt_period       = params.opt_period;
  result->get_opt_pars     = params.get_opt_pars;
  result->get_opt_pars_ud  = params.get_opt_pars_ud;
  result->optimizer        = params.optimizer;

  V_ASSERT(result->opt_period >= 1);

  result->static_graphs = result->ctx_compute;

  if (!result->static_graphs) {
    V_ASSERT(!result->inputs);
    V_ASSERT(!result->outputs);
  }

  V_ASSERT(result->inputs);
  V_ASSERT(result->outputs);

  result->gf = v_new_graph_custom(result->ctx_compute, v_DEFAULT_GRAPH_SIZE, /*grads =*/ true); // Forward pass.
  v_build_foward_expand(result->gf, result->outputs);
  result->build();
}

void v_opt_ctx::allocate(bool backward) {
  auto opt_ctx = this;
  V_ASSERT(!opt_ctx->eval_ready);
  if (opt_ctx->build_type == V_OPT_TYPE_OPT &&
    opt_ctx->opt_period > 1 &&
    opt_ctx->opt_i == 0) { v_graph_reset(opt_ctx->gb_grad); }
  if (backward) {
    const int32_t opt_i_next = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
    opt_ctx->build_type      = (opt_i_next == 0)
                            ? V_OPT_TYPE_OPT
                            : V_OPT_BUILD_TYPE_GRAD;
  }
  else { opt_ctx->build_type = V_OPT_TYPE_FORWARD; }
  if (!opt_ctx->static_graphs) { opt_ctx->build(); }
  struct v_cgraph* graph = nullptr;
  switch (opt_ctx->build_type) {
    case V_OPT_TYPE_FORWARD: { graph = opt_ctx->gf; }
    break;
    case V_OPT_BUILD_TYPE_GRAD: { graph = opt_ctx->gb_grad; }
    break;
    case V_OPT_TYPE_OPT: { graph = opt_ctx->gb_opt; }
    break;
  }
  V_ASSERT(graph);

  if (opt_ctx->allocated_graph == graph) {
    opt_ctx->eval_ready = true;
    return;
  }

  v_sched_reset(opt_ctx->backend_sched); // clear allocation of previous graph
  if (opt_ctx->static_graphs) {
    v_init_param params = {
      /*.mem_size   =*/ graph->size * v_tensor_over_head() + v_graph_overhead_custom(graph->size, graph->grads),
      /*.mem_buffer =*/ nullptr,
      /*.no_alloc   =*/ true,
    };
    free_ctx(opt_ctx->ctx_copy);
    opt_ctx->ctx_copy = v_ctx_init(params);
    auto ctx          = opt_ctx->ctx_copy;
    auto src          = graph;
    std::map<v_tensor*, v_tensor*> tensor_map;
    v_cgraph* dst = v_new_graph_custom(ctx, src->size, /*grads =*/ true);
    for (int i = 0; i < src->n_leafs; i++) { v_build_foward_expand(dst, v_map_tensor(tensor_map, ctx, src->leafs[i])); }
    V_ASSERT(dst->n_leafs == src->n_leafs);
    for (int i = 0; i < src->n_nodes; i++) { v_build_foward_expand(dst, v_map_tensor(tensor_map, ctx, src->nodes[i])); }
    V_ASSERT(dst->n_nodes == src->n_nodes);
    for (int i = 0; i < src->n_nodes; ++i) {
      const size_t igrad_src = src->visited_hash_set.find_hash(src->nodes[i]);
      const size_t igrad_dst = dst->visited_hash_set.find_hash(dst->nodes[i]);

      V_ASSERT(igrad_src != V_HASHSET_FULL);
      //V_ASSERT(src->visited_hash_set.get_bitset(igrad_src));
      //V_ASSERT(igrad_dst != V_HASHSET_FULL);
      //V_ASSERT(dst->visited_hash_set.get_bitset(igrad_dst));

      dst->grads[igrad_dst]     = src->grads[igrad_src];
      dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
    }
    opt_ctx->allocated_graph_copy = dst;
  }
  else { opt_ctx->allocated_graph_copy = graph; }
  v_sched_alloc_graph(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
  opt_ctx->allocated_graph = graph;
  opt_ctx->eval_ready      = true;
}


void v_opt_ctx::build() {
  auto opt_ctx = this;
  V_ASSERT(opt_ctx->ctx_compute && "no compute context set, either use static graphs or set one with v_opt_prepare_alloc");
  V_ASSERT((!opt_ctx->static_graphs || opt_ctx->inputs->data) &&"when using static graphs the inputs must be allocated statically");
  const enum v_opt_type optimizer = opt_ctx->optimizer;
  const bool accumulate           = opt_ctx->build_type_alloc >= V_OPT_BUILD_TYPE_GRAD &&
    !(opt_ctx->static_graphs && opt_ctx->build_type_alloc == V_OPT_TYPE_OPT && opt_ctx->
      opt_period == 1);

  const bool need_momenta = opt_ctx->build_type_alloc == V_OPT_TYPE_OPT &&
    opt_ctx->optimizer == V_OPTIMIZER_TYPE_ADAMW;

  v_set_inputs(opt_ctx->inputs);
  v_set_outputs(opt_ctx->outputs);

  int n_param = 0;
  for (int i = 0; i < opt_ctx->gf->n_nodes; ++i) {
    const struct v_tensor* node = opt_ctx->gf->nodes[i];
    if (node->flags & TENSOR_FLAG_PARAM) { n_param++; }
    V_ASSERT(!(node->flags & TENSOR_FLAG_LOSS) && "support for extra loss terms not implemented");
  }
  if (!opt_ctx->ctx_static) {
    // The static context is used_bits__ for:
    //   - gradients (1 per loss, 1 tensor per param if using gradient accumulation)
    //   - optimizer momenta (2 tensors per param)
    //   - labels (if using static graphs)
    //   - loss (if using static graphs, up to 5 tensors)
    //   - pred (if using static graphs)
    //   - ncorrect (if using static graphs, 2 tensors).
    constexpr size_t n_loss        = 1;
    const size_t tensors_per_param = (accumulate
                                        ? 1
                                        : 0) + (need_momenta
                                                  ? 2
                                                  : 0);
    const size_t tensors_const = opt_ctx->static_graphs
                                   ? 9
                                   : 0;
    const size_t size_meta     = (n_loss + tensors_per_param * n_param + tensors_const) * v_tensor_over_head();
    struct v_init_param params = {
      /*.mem_size   =*/ size_meta,
      /*.mem_buffer =*/ nullptr,
      /*.no_alloc   =*/ true,
    };
    opt_ctx->ctx_static = v_ctx_init(params);
  }
  V_ASSERT(opt_ctx->build_type <= opt_ctx->build_type_alloc);
  {
    // The cpu context is allocated statically if using static graphs, dynamically otherwise.
    // It is used_bits__ for:
    //   - optimizer parameters (1 shared for all optimizer invocations)
    const size_t size_meta = 1 * v_tensor_over_head();
    v_init_param params    = {
      /*.mem_size   =*/ size_meta,
      /*.mem_buffer =*/ nullptr,
      /*.no_alloc   =*/ true,
    };
    free_ctx(opt_ctx->ctx_gpu);
    opt_ctx->ctx_gpu = v_ctx_init(params);
    v_backend_buffer_free(opt_ctx->buf_host);
    opt_ctx->buf_host = nullptr;
  }

  v_ctx* ctx_results = opt_ctx->static_graphs
                         ? opt_ctx->ctx_static
                         : opt_ctx->ctx_compute;

  switch (opt_ctx->loss_type) {
    case V_OPT_LOSS_MEAN: {
      opt_ctx->loss = v_sum(ctx_results, opt_ctx->outputs);
      v_set_name(opt_ctx->loss, "loss_sum");
      const float scale = 1.0f / (opt_ctx->opt_period * nelements(opt_ctx->outputs));
      opt_ctx->loss     = v_scale(ctx_results, opt_ctx->loss, scale);
      v_set_name(opt_ctx->loss, "loss_mean");
      opt_ctx->loss_per_datapoint = true;
      break;
    }
    case V_OPT_LOSS_SUM: {
      opt_ctx->loss = v_sum(ctx_results, opt_ctx->outputs);
      v_set_name(opt_ctx->loss, "loss_sum");
      opt_ctx->loss_per_datapoint = false;
      break;
    }
    case V_OPT_LOSS_CROSS_ENTROPY: {
      opt_ctx->labels = v_dup_tensor(ctx_results, opt_ctx->outputs);

      v_set_inputs(opt_ctx->labels);
      v_set_name(opt_ctx->labels, "labels_one_hot");

      opt_ctx->loss = v_soft_max(ctx_results, opt_ctx->outputs);
      v_build_foward_expand(opt_ctx->gf, opt_ctx->loss);
      v_set_name(opt_ctx->loss, "predicted_probabilities");

      opt_ctx->loss = v_log(ctx_results, opt_ctx->loss);
      v_build_foward_expand(opt_ctx->gf, opt_ctx->loss);
      v_set_name(opt_ctx->loss, "log_probabilities");

      opt_ctx->loss = v_mul(ctx_results, opt_ctx->labels, opt_ctx->loss);
      v_build_foward_expand(opt_ctx->gf, opt_ctx->loss);
      v_set_name(opt_ctx->loss, "weighted_log_probabilities");

      opt_ctx->loss = v_sum(ctx_results, opt_ctx->loss);
      v_build_foward_expand(opt_ctx->gf, opt_ctx->loss);
      v_set_name(opt_ctx->loss, "loss_sum_negative_log_likelihood");

      const float scale = -1.0f / opt_ctx->inputs->ne[1] * opt_ctx->opt_period;
      opt_ctx->loss     = v_scale(ctx_results, opt_ctx->loss, scale);
      //predict check
      v_build_foward_expand(opt_ctx->gf, opt_ctx->loss);
      opt_ctx->loss_per_datapoint = true;
      opt_ctx->pred               = v_argmax(ctx_results, opt_ctx->outputs);
      v_set_name(opt_ctx->pred, "pred");
      v_set_outputs(opt_ctx->pred);
      v_build_foward_expand(opt_ctx->gf, opt_ctx->pred);
      opt_ctx->ncorrect = v_count_equal(ctx_results, opt_ctx->pred, v_argmax(ctx_results, opt_ctx->labels));
      v_set_name(opt_ctx->ncorrect, "ncorrect");
      v_set_outputs(opt_ctx->ncorrect);
      v_build_foward_expand(opt_ctx->gf, opt_ctx->ncorrect);
      v_print_graph(opt_ctx->gf);
      break;
    }

    case V_OPT_LOSS_SQUARED_ERROR: {
      opt_ctx->labels = v_dup_tensor(ctx_results, opt_ctx->outputs);
      v_set_inputs(opt_ctx->labels);
      v_set_name(opt_ctx->labels, "labels");

      opt_ctx->loss = v_sub(ctx_results, opt_ctx->outputs, opt_ctx->labels);
      v_build_foward_expand(opt_ctx->gf, opt_ctx->loss);
      v_set_name(opt_ctx->loss, "loss_error");

      opt_ctx->loss = v_sqr(ctx_results, opt_ctx->loss);
      v_build_foward_expand(opt_ctx->gf, opt_ctx->loss);
      v_set_name(opt_ctx->loss, "loss_squared_error");

      opt_ctx->loss = v_sum(ctx_results, opt_ctx->loss);
      v_set_name(opt_ctx->loss, "loss_sum_squared_error");
      v_build_foward_expand(opt_ctx->gf, opt_ctx->loss);

      const float scale = 1.0f / (opt_ctx->opt_period * nelements(opt_ctx->inputs));
      opt_ctx->loss     = v_scale(ctx_results, opt_ctx->loss, scale);
      v_build_foward_expand(opt_ctx->gf, opt_ctx->loss);

      v_set_name(opt_ctx->loss, "loss_mean_squared_error");
      opt_ctx->loss_per_datapoint = true;
      break;
    }
  }
  v_set_outputs(opt_ctx->loss);
  v_set_loss(opt_ctx->loss);
  v_build_foward_expand(opt_ctx->gf, opt_ctx->loss);


  if (opt_ctx->buf_static) { if (opt_ctx->build_type == V_OPT_TYPE_FORWARD) { return; } }
  else if (opt_ctx->build_type_alloc == V_OPT_TYPE_FORWARD) {
    opt_ctx->buf_static = v_backend_alloc_ctx_tensors(
      opt_ctx->ctx_static,
      v_sched_get_backend(opt_ctx->backend_sched, 0));
    return;
  }

  if (opt_ctx->grad_accs.empty()) {
    V_ASSERT(opt_ctx->build_type_alloc >= V_OPT_BUILD_TYPE_GRAD);
    const int n_nodes = opt_ctx->gf->n_nodes;
    opt_ctx->grad_accs.resize(n_nodes);
    for (int i = 0; i < n_nodes; ++i) {
      v_tensor* node = opt_ctx->gf->nodes[i];
      if ((accumulate && (node->flags & TENSOR_FLAG_PARAM)) || (node->flags & TENSOR_FLAG_LOSS)) { opt_ctx->grad_accs[i] = v_new_tensor(opt_ctx->ctx_static, v_TYPE_F32, V_MAX_DIMS, node->ne.data()); }
      else { opt_ctx->grad_accs[i] = nullptr; }
    }

    if (need_momenta && opt_ctx->build_type_alloc >= V_OPT_TYPE_OPT) {
      opt_ctx->grad_m.resize(n_nodes);
      opt_ctx->grad_v.resize(n_nodes);
      for (int i = 0; i < n_nodes; ++i) {
        v_tensor* node = opt_ctx->gf->nodes[i];
        if (node->flags & TENSOR_FLAG_PARAM) {
          opt_ctx->grad_m[i] = v_new_tensor(opt_ctx->ctx_static, v_TYPE_F32, V_MAX_DIMS, node->ne.data());
          opt_ctx->grad_v[i] = v_new_tensor(opt_ctx->ctx_static, v_TYPE_F32, V_MAX_DIMS, node->ne.data());
        }
        else {
          opt_ctx->grad_m[i] = nullptr;
          opt_ctx->grad_v[i] = nullptr;
        }
      }
    }
  }

  // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
  opt_ctx->gb_grad = v_graph_duplicate(opt_ctx->ctx_compute,
                                       opt_ctx->gf,
                                       /*force_grads =*/
                                       true);

  v_build_backward_expend(opt_ctx->ctx_compute,
                          opt_ctx->gb_grad,
                          opt_ctx->grad_accs.data());

  if (opt_ctx->buf_static) { if (opt_ctx->build_type == V_OPT_BUILD_TYPE_GRAD) { return; } }
  else if (opt_ctx->build_type_alloc == V_OPT_BUILD_TYPE_GRAD) {
    opt_ctx->buf_static = v_backend_alloc_ctx_tensors(opt_ctx->ctx_static,
                                                      v_sched_get_backend(opt_ctx->backend_sched, 0));
    v_graph_reset(opt_ctx->gb_grad);
  }
  V_ASSERT(opt_ctx->build_type_alloc == V_OPT_TYPE_OPT);
  // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
  opt_ctx->gb_opt            = v_graph_duplicate(opt_ctx->ctx_compute, opt_ctx->gb_grad, /*force_grads =*/ true);
  opt_ctx->opt_step_params__ = v_new_tensor_1d(opt_ctx->ctx_gpu,
                                               v_TYPE_F32,
                                               need_momenta
                                                 ? 7
                                                 : 2);
  v_tensor* adamw_params = opt_ctx->opt_step_params__;
  v_set_inputs(adamw_params);
  const char* optimizer_name = opt_ctx->get_opt_name();
  v_format_name(adamw_params, "%s_params", optimizer_name);
  for (int i = opt_ctx->gf->n_nodes - 1; i >= 0; --i) {
    struct v_tensor* node = opt_ctx->gb_opt->nodes[i];
    struct v_tensor* grad = v_graph_get_grad(opt_ctx->gb_opt, node);
    if (grad && (node->flags & TENSOR_FLAG_PARAM)) {
      struct v_tensor* m = nullptr;
      struct v_tensor* v = nullptr;
      if (need_momenta) {
        m = opt_ctx->grad_m[i];
        v = opt_ctx->grad_v[i];
        v_format_name(m, "AdamW m for %s", node->name);
        v_format_name(v, "AdamW v for %s", node->name);
      }
      v_tensor* opt_step;
      switch (optimizer) {
        case V_OPTIMIZER_TYPE_ADAMW:
          v_norm_l2(opt_ctx->ctx_compute, grad, 1e-3);
          opt_step = v_opt_step_adamw(opt_ctx->ctx_compute, node, grad, m, v, adamw_params);
          break;
        case V_OPTIMIZER_TYPE_SGD:
          v_norm_l2(opt_ctx->ctx_compute, grad, 1e-1);
          opt_step = v_opt_step_sgd(opt_ctx->ctx_compute, node, grad, adamw_params);
          break;
        default:
          v_ABORT("fatal error");
      }
      v_format_name(opt_step, "%s step for %s", optimizer_name, node->name);
      v_build_foward_expand(opt_ctx->gb_opt, opt_step);
    }
  }

  if (!opt_ctx->buf_static) {
    opt_ctx->buf_static = v_backend_alloc_ctx_tensors(opt_ctx->ctx_static,
                                                      v_sched_get_backend(opt_ctx->backend_sched, 0));
    v_graph_reset(opt_ctx->gb_opt);
  }
  opt_ctx->buf_static = v_backend_alloc_ctx_tensor_from_buffer_t(opt_ctx->ctx_gpu, vk_device_buffer_type(0));
}
