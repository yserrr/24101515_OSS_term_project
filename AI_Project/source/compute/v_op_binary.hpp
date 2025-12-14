#pragma once

V_API v_tensor* v_sub_impl(v_ctx* ctx, v_tensor* a, v_tensor* b, bool inplace);
V_API v_tensor* v_acc_imple(v_ctx* ctx, v_tensor* a, v_tensor* b, size_t nb1, size_t nb2, size_t nb3, size_t offset, bool inplace);
V_API v_tensor* v_mul_impl(v_ctx* ctx, v_tensor* a, v_tensor* b, bool inplace);
V_API v_tensor* v_add_cast_impl(v_ctx* ctx, v_tensor* a, v_tensor* b, v_data_type type);
V_API v_tensor* v_add1_impl(v_ctx* ctx, v_tensor* a, v_tensor* b, bool inplace);
V_API v_tensor* v_div_impl(v_ctx* ctx, v_tensor* a, v_tensor* b, bool inplace);
V_API v_tensor* v_add_imple(v_ctx* ctx, v_tensor* a, v_tensor* b, bool inplace);
V_API v_tensor* v_map_custom3_impl(v_ctx* ctx, v_tensor* a, v_tensor* b, v_tensor* c, const v_custom3_op_t fun, int n_tasks, void* userdata, bool inplace);
V_API v_tensor* v_scale_impl(v_ctx* ctx, v_tensor* a, float s, float b, bool inplace);
V_API v_tensor* v_diag(v_ctx* ctx, v_tensor* a);
V_API v_tensor* v_diag_mask_zero_impl(v_ctx* ctx, v_tensor* a, int n_past, bool inplace);
V_API v_tensor* v_map_custom1_impl(v_ctx* ctx, v_tensor* a, v_custom1_op_t fun, int n_tasks, void* userdata, bool inplace);
V_API v_tensor* v_map_custom2_impl(v_ctx* ctx, v_tensor* a, v_tensor* b, v_custom2_op_t fun, int n_tasks, void* userdata, bool inplace);
V_API v_tensor* v_sqr_impl(v_ctx* ctx, v_tensor* a, bool inplace);
V_API v_tensor* v_sqrt_impl(v_ctx* ctx, v_tensor* a, bool inplace);
V_API v_tensor* v_interpolate_impl(v_ctx* ctx, v_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, uint32_t mode);
