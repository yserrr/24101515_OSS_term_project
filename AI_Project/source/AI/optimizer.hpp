//
// Created by dlwog on 25. 11. 10..
//

#ifndef MYPROJECT_OPTIMIZER_HPP
#define MYPROJECT_OPTIMIZER_HPP
#include "mml_opt.hpp"

namespace opt
{
  class MmlOptimizer
  {
  public:
    MmlOptimizer(MmlLossType eLossType,
                 MmlOptimizerBuildType eBuildType,
                 MmlOptimizerType otType
    );




    MmlOptimizerParameters optParameters;
    MmlOptimizeBuildType optBuildType;
    MmlOptimizerType optType;
    MmlLossType lossType;
  };
}


#endif //MYPROJECT_OPTIMIZER_HPP
