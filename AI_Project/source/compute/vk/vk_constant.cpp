#include "vk_constant.h"


vk_op_unary_push_constants vk_op_unary_push_constants_init(const v_tensor* src0, const v_tensor* dst, int64_t ne) {
  V_ASSERT(ne != 0 || (nelements(src0) == nelements(dst)));
  ne = ne != 0
         ? ne
         : nelements(dst);
  V_ASSERT(ne <= (int64_t)std::numeric_limits<uint32_t>::max());

  vk_op_unary_push_constants p{};
  p.ne = (uint32_t)ne;

  size_t src0_tsize = v_type_size(src0->type);
  p.ne00            = (uint32_t)src0->ne[0];
  p.ne01            = (uint32_t)src0->ne[1];
  p.ne02            = (uint32_t)src0->ne[2];
  p.ne03            = (uint32_t)src0->ne[3];
  p.nb00            = (uint32_t)(src0->nb[0] / src0_tsize);
  p.nb01            = (uint32_t)(src0->nb[1] / src0_tsize);
  p.nb02            = (uint32_t)(src0->nb[2] / src0_tsize);
  p.nb03            = (uint32_t)(src0->nb[3] / src0_tsize);

  size_t dst_tsize = v_type_size(dst->type);
  p.ne10           = (uint32_t)dst->ne[0];
  p.ne11           = (uint32_t)dst->ne[1];
  p.ne12           = (uint32_t)dst->ne[2];
  p.ne13           = (uint32_t)dst->ne[3];
  p.nb10           = (uint32_t)(dst->nb[0] / dst_tsize);
  p.nb11           = (uint32_t)(dst->nb[1] / dst_tsize);
  p.nb12           = (uint32_t)(dst->nb[2] / dst_tsize);
  p.nb13           = (uint32_t)(dst->nb[3] / dst_tsize);

  return p; // offsets are initialized later in v_vk_op
}


vk_op_pad_push_constants vk_op_pad_push_constants_init(const v_tensor* src0, const v_tensor* dst) {
  int64_t ne = nelements(dst);
  V_ASSERT(ne <= (int64_t)std::numeric_limits<uint32_t>::max());

  vk_op_pad_push_constants p{};
  p.ne = (uint32_t)ne;

  size_t src0_tsize = v_type_size(src0->type);
  p.ne00            = (uint32_t)src0->ne[0];
  p.ne01            = (uint32_t)src0->ne[1];
  p.ne02            = (uint32_t)src0->ne[2];
  p.ne03            = (uint32_t)src0->ne[3];
  p.nb00            = (uint32_t)(src0->nb[0] / src0_tsize);
  p.nb01            = (uint32_t)(src0->nb[1] / src0_tsize);
  p.nb02            = (uint32_t)(src0->nb[2] / src0_tsize);
  p.nb03            = (uint32_t)(src0->nb[3] / src0_tsize);

  size_t dst_tsize = v_type_size(dst->type);
  p.ne10           = (uint32_t)dst->ne[0];
  p.ne11           = (uint32_t)dst->ne[1];
  p.ne12           = (uint32_t)dst->ne[2];
  p.ne13           = (uint32_t)dst->ne[3];
  p.nb10           = (uint32_t)(dst->nb[0] / dst_tsize);
  p.nb11           = (uint32_t)(dst->nb[1] / dst_tsize);
  p.nb12           = (uint32_t)(dst->nb[2] / dst_tsize);
  p.nb13           = (uint32_t)(dst->nb[3] / dst_tsize);

  p.lp0 = dst->op_params[0];
  p.rp0 = dst->op_params[1];
  p.lp1 = dst->op_params[2];
  p.rp1 = dst->op_params[3];
  p.lp2 = dst->op_params[4];
  p.rp2 = dst->op_params[5];
  p.lp3 = dst->op_params[6];
  p.rp3 = dst->op_params[7];

  return p; // fastdiv values and offsets are initialized later in v_vk_op
}
vk_op_sum_rows_push_constants vk_op_sum_rows_push_constants_init(const v_tensor* src, const v_tensor* dst,
                                                                 int64_t n_cols) {
  uint32_t type_size              = (uint32_t)v_type_size(src->type);
  vk_op_sum_rows_push_constants p = {};
  p.n_cols                        = (uint32_t)n_cols;
  p.ne01                          = (uint32_t)src->ne[1];
  p.ne02                          = (uint32_t)src->ne[2];
  p.nb01                          = (uint32_t)src->nb[1] / type_size;
  p.nb02                          = (uint32_t)src->nb[2] / type_size;
  p.nb03                          = (uint32_t)src->nb[3] / type_size;
  p.nb11                          = (uint32_t)dst->nb[1] / type_size;
  p.nb12                          = (uint32_t)dst->nb[2] / type_size;
  p.nb13                          = (uint32_t)dst->nb[3] / type_size;
  p.weight                        = 1.0f;
  return p;
}

