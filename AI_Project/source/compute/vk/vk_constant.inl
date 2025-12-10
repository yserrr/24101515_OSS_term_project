
template <typename T>
inline size_t push_constant_size(const T& t) {
  static_assert(std::is_class<T>::value, "T must be a struct/class");
  V_UNUSED(t);
  return sizeof(T);
}

template <typename T>
inline size_t push_constant_size(const std::vector<T>& t) {
  V_UNUSED(t);
  return sizeof(T) * t.size();
}

template <typename T, uint32_t N>
inline size_t push_constant_size(const std::array<T, N>& t) {
  V_UNUSED(t);
  return sizeof(T) * N;
}

template <typename T>
const T* push_constant_data(const T& t) {
  static_assert(std::is_class<T>::value, "T must be a struct/class");
  return &t;
}

template <typename T>
const T* push_constant_data(const std::vector<T>& t) { return t.data(); }

template <typename T, uint32_t N>
const T* push_constant_data(const std::array<T, N>& t) { return t.data(); }



template <typename T>
inline void init_pushconst_tensor_offsets(vk_backend_ctx* ctx, T& p, const v_tensor* src0, const v_tensor* src1,
                                          const v_tensor* src2, v_tensor* dst) {
  V_UNUSED(p);
  V_UNUSED(src0);
  V_UNUSED(src1);
  V_UNUSED(src2);
  V_UNUSED(dst);
  static_assert(!std::is_const<T>::value, "unexpected type");
  V_ASSERT(!src0 || get_misalign_bytes(ctx, src0) == 0);
  V_ASSERT(!src1 || get_misalign_bytes(ctx, src1) == 0);
  V_ASSERT(!src2 || get_misalign_bytes(ctx, src2) == 0);
  V_ASSERT(!dst || get_misalign_bytes(ctx, dst) == 0);
}

template <>
inline void init_pushconst_tensor_offsets(vk_backend_ctx* ctx, vk_op_unary_push_constants& p, const v_tensor* src0,
                                          const v_tensor* src1, const v_tensor* src2, v_tensor* dst) {
  const uint32_t a_offset = get_misalign_bytes(ctx, src0) / v_type_size(src0->type);
  const uint32_t d_offset = get_misalign_bytes(ctx, dst) / v_type_size(dst->type);

  p.misalign_offsets = (a_offset << 16) | d_offset;

  V_UNUSED(src1);
  V_UNUSED(src2);
}

template <>
inline void init_pushconst_tensor_offsets(vk_backend_ctx* ctx, vk_op_sum_rows_push_constants& p,
                                          const v_tensor* src0, const v_tensor* src1, const v_tensor* src2,
                                          v_tensor* dst) {
  const uint32_t a_offset = get_misalign_bytes(ctx, src0) / v_type_size(src0->type);
  const uint32_t d_offset = get_misalign_bytes(ctx, dst) / v_type_size(dst->type);
  p.misalign_offsets      = (a_offset << 16) | d_offset;
  V_UNUSED(src1);
  V_UNUSED(src2);
}

template <>
inline void init_pushconst_tensor_offsets(vk_backend_ctx* ctx, vk_op_pad_push_constants& p, const v_tensor* src0,
                                          const v_tensor* src1, const v_tensor* src2, v_tensor* dst) {
  const uint32_t a_offset = get_misalign_bytes(ctx, src0) / v_type_size(src0->type);
  const uint32_t d_offset = get_misalign_bytes(ctx, dst) / v_type_size(dst->type);

  p.misalign_offsets = (a_offset << 16) | d_offset;

  V_UNUSED(src1);
  V_UNUSED(src2);
}

template <>
inline void init_pushconst_tensor_offsets(vk_backend_ctx* ctx, vk_op_im2col_3d_push_constants& p,
                                          const v_tensor* src0, const v_tensor* src1, const v_tensor* src2,
                                          v_tensor* dst) {
  const uint32_t a_offset = get_misalign_bytes(ctx, src1) / v_type_size(src1->type);
  const uint32_t d_offset = get_misalign_bytes(ctx, dst) / v_type_size(dst->type);

  p.misalign_offsets = (a_offset << 16) | d_offset;

  V_UNUSED(src0);
  V_UNUSED(src2);
}

template <>
inline void init_pushconst_tensor_offsets(vk_backend_ctx* ctx, vk_op_binary_push_constants& p,
                                          const v_tensor* src0, const v_tensor* src1, const v_tensor* src2,
                                          v_tensor* dst) {
  const uint32_t a_offset = get_misalign_bytes(ctx, src0) / v_type_size(src0->type);
  const uint32_t b_offset = get_misalign_bytes(ctx, src1) / v_type_size(src1->type);
  const uint32_t d_offset = get_misalign_bytes(ctx, dst) / v_type_size(dst->type);

  V_ASSERT(dst->op != v_OP_GET_ROWS || (a_offset == 0 && b_offset == 0 && d_offset == 0));

  p.misalign_offsets = (a_offset << 16) | (b_offset << 8) | d_offset;

  V_UNUSED(src2);
}

template <>
inline void init_pushconst_tensor_offsets(vk_backend_ctx* ctx, vk_op_upscale_push_constants& p,
                                          const v_tensor* src0, const v_tensor* src1, const v_tensor* src2,
                                          v_tensor* dst) {
  const uint32_t a_offset = get_misalign_bytes(ctx, src0) / v_type_size(src0->type);
  const uint32_t d_offset = get_misalign_bytes(ctx, dst) / v_type_size(dst->type);

  p.a_offset = a_offset;
  p.d_offset = d_offset;

  V_UNUSED(src1);
  V_UNUSED(src2);
}


template <typename T>
inline void init_pushconst_fastdiv(T& p) {
  V_UNUSED(p);
  static_assert(!std::is_const<T>::value, "unexpected type");
}

template <>
inline void init_pushconst_fastdiv(vk_op_sum_rows_push_constants& p) {
  init_fastdiv_values(p.ne01 * p.ne02, p.ne0_12mp, p.ne0_12L);
  init_fastdiv_values(p.ne01, p.ne0_1mp, p.ne0_1L);
}

template <>
inline void init_pushconst_fastdiv(vk_op_unary_push_constants& p) {
  // Compute magic values to divide by these six numbers.
  init_fastdiv_values(p.ne02 * p.ne01 * p.ne00, p.ne0_012mp, p.ne0_012L);
  init_fastdiv_values(p.ne01 * p.ne00, p.ne0_01mp, p.ne0_01L);
  init_fastdiv_values(p.ne00, p.ne0_0mp, p.ne0_0L);
  init_fastdiv_values(p.ne12 * p.ne11 * p.ne10, p.ne1_012mp, p.ne1_012L);
  init_fastdiv_values(p.ne11 * p.ne10, p.ne1_01mp, p.ne1_01L);
  init_fastdiv_values(p.ne10, p.ne1_0mp, p.ne1_0L);
}

template <>
inline void init_pushconst_fastdiv(vk_op_conv2d_push_constants& p) {
  // Compute magic values to divide by KW, KW*KH, OW, OW*OH
  init_fastdiv_values(p.KW, p.KWmp, p.KWL);
  init_fastdiv_values(p.KW * p.KH, p.KWKHmp, p.KWKHL);
  init_fastdiv_values(p.OW, p.OWmp, p.OWL);
  init_fastdiv_values(p.OW * p.OH, p.OWOHmp, p.OWOHL);
}

template <>
inline void init_pushconst_fastdiv(vk_op_conv_transpose_2d_push_constants& p) {
  // Compute magic values to divide by KW, KW*KH, OW, OW*OH, s0, s1
  init_fastdiv_values(p.KW, p.KWmp, p.KWL);
  init_fastdiv_values(p.KW * p.KH, p.KWKHmp, p.KWKHL);
  init_fastdiv_values(p.OW, p.OWmp, p.OWL);
  init_fastdiv_values(p.OW * p.OH, p.OWOHmp, p.OWOHL);
  init_fastdiv_values(p.s0, p.s0mp, p.s0L);
  init_fastdiv_values(p.s1, p.s1mp, p.s1L);
}