//
// Created by dlwog on 25. 11. 20..
//

#ifndef MYPROJECT_MML_TYPE_HPP
#define MYPROJECT_MML_TYPE_HPP
#include <array>
#include "ggml-common.h"
#include "v.h"

class MmlTypeTrait
{
  public:
  std::array<v_type_traits, v_TYPE_COUNT> traits;
  MmlTypeTrait();
};


#endif //MYPROJECT_MML_TYPE_HPP
