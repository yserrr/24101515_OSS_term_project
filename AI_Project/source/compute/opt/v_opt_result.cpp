#include "v_opt_result.hpp"

#include <iostream>
#include <ostream>

#include "v_opt_ctx.hpp"

v_opt_result_t v_opt_result_init() { return new v_opt_result; }

void ggml_opt_result_free(v_opt_result_t result) { delete result; }


void v_opt_result_loss(v_opt_result_t result, double* loss, double* unc) {

  const int64_t nbatches = result->loss.size();
  if (nbatches == 0) {
    *loss = 0.0;
    *unc  = NAN;
    return;
  }
  double sum         = 0.0;
  double sum_squared = 0.0;
  for (const float& loss : result->loss) {
    const float loss_scaled = result->loss_per_datapoint
                                ? loss * result->opt_period
                                : loss;
    sum += loss_scaled;
    sum_squared += loss_scaled * loss_scaled;
  }
  const double mean = sum / nbatches;
  *loss             = result->loss_per_datapoint
                        ? mean
                        : sum;
  if (!unc) { return; }

  if (nbatches < 2) {
    *unc = NAN;
    return;
  }
  //if (*loss>13) {
  //  throw std::runtime_error("loss NAN check");
  //}
  const double var_sum = sum_squared / nbatches - mean * mean;
  // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
  *unc = result->loss_per_datapoint
           ? sqrt(var_sum / (nbatches - 1))
           : sqrt(var_sum * nbatches / (nbatches - 1));
}

void v_opt_result_pred(v_opt_result_t result, int32_t* pred) {
  for (size_t i = 0; i < result->pred.size(); ++i) {
    pred[i] = result->pred[i];
    std::cout << i << ": " << result->pred[i] << std::endl;
  }
}

void v_opt_result_accurancy(v_opt_result_t result, double* accuracy, double* unc) {
  *accuracy = result->ncorrect >= 0
                ? double(result->ncorrect) / double(result->ndata)
                : NAN;
  if (!unc) { return; }

  *unc = result->ncorrect >= 0 && result->ndata >= 2
           ? sqrt((*accuracy) * (1.0 - (*accuracy)) / double(result->ndata - 1))
           : NAN;
}
