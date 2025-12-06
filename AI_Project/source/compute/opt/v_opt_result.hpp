

#ifndef MYPROJECT_MML_OPT_RESULT_HPP
#define MYPROJECT_MML_OPT_RESULT_HPP
#include "v_opt_common.hpp"


struct v_opt_result
{
  int64_t ndata = 0;
  std::vector<float> loss;
  std::vector<int32_t> pred;
  int64_t ncorrect = 0;
  int64_t opt_period = -1;
  bool loss_per_datapoint = false;
  void reset()
  {
    this->ndata = 0;
    this->loss.clear();
    this->pred.clear();
    this->ncorrect = 0;
  }
};


#endif //MYPROJECT_MML_OPT_RESULT_HPP