//
// Created by dlwog on 25. 11. 20..
//

#ifndef MYPROJECT_MML_TYPE_HPP
#define MYPROJECT_MML_TYPE_HPP
#include <array>
#include "v_common.hpp"
#include "ggml-common.h"
#include "v.hpp"

class v_type_trait {
public:
  std::array<v_type_traits, v_TYPE_COUNT> traits;
  v_type_trait();
};


#endif //MYPROJECT_MML_TYPE_HPP
