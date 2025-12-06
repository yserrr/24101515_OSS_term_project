//
// Created by dlwog on 25. 11. 13..
//

#ifndef MYPROJECT_MML_FN_H
#define MYPROJECT_MML_FN_H

#include "v-backend.h"
#include "v.h"


void v_print_tensor2d(v_tensor* t);
void v_print_t_buffer(v_tensor* t);
void v_build_foward_expand(struct v_cgraph* cgraph, struct v_tensor* tensor);
void v_compute_backword(struct v_ctx* ctx, struct v_cgraph* cgraph, int i, const bool* grads_needed);
#endif //MYPROJECT_MML_FN_H
