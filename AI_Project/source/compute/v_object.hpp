#ifndef MYPROJECT_V_OBJECT_HPP
#define MYPROJECT_V_OBJECT_HPP
#include "v_common.hpp"

struct v_object {
  size_t offs;
  size_t size;
  v_object* next;
  v_object_type type;
  char padding[4];
};

static_assert(sizeof(v_object) % v_MEM_ALIGN == 0, "v_object size must be a multiple of v_MEM_ALIGN");
#endif //MYPROJECT_V_OBJECT_HPP
