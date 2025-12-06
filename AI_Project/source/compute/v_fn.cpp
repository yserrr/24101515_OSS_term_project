#include "v_fn.h"
#include "v-backend.h"
#include "ggml-impl.h"
#include "v_vk.h"
#include "v.h"

void vk_host_buffer_memset_tensor(v_backend_buffer_t buffer, struct v_tensor* tensor, uint8_t value, size_t offset, size_t size);

void v_backend_tensor_memset(struct v_tensor* tensor, uint8_t value, size_t offset, size_t size) {
  V_ASSERT(tensor);
  v_backend_buffer_t buf = tensor->view_src
                             ? tensor->view_src->buffer
                             : tensor->buffer;
  if (size == 0) { return; }
  V_ASSERT(buf != NULL && "tensor buffer not set");
  V_ASSERT(tensor->data != NULL && "tensor not allocated");
  V_ASSERT(offset + size <= num_bytes(tensor) && "tensor write out of bounds");
  buf->buft->host
    ? vk_host_buffer_memset_tensor(buf, tensor, value, offset, size)
    : vk_device_buffer_memset_tensor(buf, tensor, value, offset, size);
}

// utility functions to change gradients
// isrc is the index of tensor in cgraph->visited_has_set.keys
// the corresponding gradient (accumulators) are also at position isrc
// if tensor has a gradient accumulator, modify that accumulator in-place
// else if there is no gradient for tensor, set the corresponding value
// else, just add/subtract/etc. the gradients

static void v_add_or_set(struct v_ctx* ctx,
                         struct v_cgraph* cgraph,
                         size_t isrc,
                         struct v_tensor* tensor) {
  struct v_tensor* src = cgraph->visited_hash_set.keys[isrc];
  V_ASSERT(src);
  if (cgraph->grads[isrc]) { cgraph->grads[isrc] = v_add_imple(ctx, cgraph->grads[isrc], tensor, /*inplace =*/ cgraph->grad_accs[isrc]); }
  else { cgraph->grads[isrc] = tensor; }
  v_format_name(cgraph->grads[isrc], "grad for %s", src->name);
  v_build_foward_expand(cgraph, cgraph->grads[isrc]);
}

static void v_acc_or_set(
  struct v_ctx* ctx,
  struct v_cgraph* cgraph,
  size_t isrc,
  struct v_tensor* tensor,
  const size_t nb1,
  const size_t nb2,
  const size_t nb3,
  const size_t offset) {
  struct v_tensor* src = cgraph->visited_hash_set.keys[isrc];
  V_ASSERT(src);
  if (cgraph->grads[isrc]) {
    cgraph->grads[isrc] = v_acc_imple(ctx,
                                      cgraph->grads[isrc],
                                      tensor,
                                      nb1,
                                      nb2,
                                      nb3,
                                      offset,
                                      cgraph->grad_accs[isrc]);
  }
  else {
    struct v_tensor* a_zero = v_scale(ctx, src, 0.0f); // FIXME this is going to produce NaN if a contains inf/NaN
    cgraph->grads[isrc]     = v_acc_imple(ctx, a_zero, tensor, nb1, nb2, nb3, offset, false);
  }
  v_format_name(cgraph->grads[isrc], "grad for %s", cgraph->visited_hash_set.keys[isrc]->name);
  v_build_foward_expand(cgraph, cgraph->grads[isrc]);
}

static void v_add1_or_set(
  struct v_ctx* ctx,
  struct v_cgraph* cgraph,
  size_t isrc,
  struct v_tensor* tensor) {
  struct v_tensor* src = cgraph->visited_hash_set.keys[isrc];
  V_ASSERT(src);
  if (cgraph->grads[isrc]) { cgraph->grads[isrc] = add1_impl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]); }
  else { cgraph->grads[isrc] = v_repeat(ctx, tensor, src); }
  v_format_name(cgraph->grads[isrc], "grad for %s", src->name);
  v_build_foward_expand(cgraph, cgraph->grads[isrc]);
}

static void v_sub_or_set(
  struct v_ctx* ctx,
  struct v_cgraph* cgraph,
  size_t isrc,
  struct v_tensor* tensor) {
  struct v_tensor* src = cgraph->visited_hash_set.keys[isrc];
  V_ASSERT(src);
  //if (cgraph->grads[isrc])
  //{
  cgraph->grads[isrc] = mmlSubImpl(ctx, cgraph->grads[isrc], tensor, cgraph->grad_accs[isrc]);
  //}
  //else
  //{
  //  cgraph->grads[isrc] = v_neg(ctx, tensor);
  //}
  v_format_name(cgraph->grads[isrc], "grad for %s", src->name);
  v_build_foward_expand(cgraph, cgraph->grads[isrc]);
}


void v_print_tensor2d(v_tensor* t) {
  std::vector<float> out_data(nelements(t));
  v_get_backend_tensor(t, out_data.data(), 0, num_bytes(t));
  printf("size : [%d , %d] :\n", static_cast<int>(t->ne[0]), static_cast<int>(t->ne[1]));
  printf("buffer size: [%llu] : \n", static_cast<size_t>(t->buffer->size));
  printf("offset : [%lld], size: [%d] :\n", reinterpret_cast<size_t>(static_cast<const char*>(t->data)),
         static_cast<int>(num_bytes(t)));
  printf("tensor op: [%d] \n", t->op);
  printf("name : %s \n", t->name);
  for (int j = 0; j < t->ne[1] /* rows */; j++) {
    if (j > 0) { printf("\n"); }
    for (int i = 0; i < t->ne[0] /* cols */; i++) {
      printf(" %.2f",
             out_data[j * t->ne[0] + i]);
    }
  }
  printf(" \n");
  printf(" \n");
}

void v_print_t_buffer(v_tensor* t) {
  auto buf = t->buffer;
  printf("tensor name: [%s]\n", t->name);
  printf("tensor ctx: [%llu], size: [%d]\n",
         reinterpret_cast<size_t>(t->data), static_cast<int>(buf->size));
  //std::vector<float> out_data(nelements(t));
  //v_get_backend_tensor(t, out_data.data(), 0, buf->size);
  //for ( int i=0; i<(buf->size/4); i++)
  //  printf("%.2f ",((const char*)t->data + 4 * i));
}

void v_compute_backword(struct v_ctx* ctx,
                        struct v_cgraph* cgraph,
                        int i,
                        const bool* grads_needed) {
  printf("backword compute called\n");

  struct v_tensor* tensor = cgraph->nodes[i];
  struct v_tensor* grad   = v_graph_get_grad(cgraph, tensor);

  if (!grad) return;

  struct v_tensor* src0       = tensor->src[0];
  struct v_tensor* src1       = tensor->src[1];
  struct v_tensor* src2       = tensor->src[2];
  struct v_hash_set* hash_set = &cgraph->visited_hash_set;
  const size_t isrc0          = src0
                                  ? find_hash(hash_set, src0)
                                  : (size_t)-1;
  const size_t isrc1 = src1
                         ? find_hash(hash_set, src1)
                         : (size_t)-1;
  const size_t isrc2 = src2
                         ? find_hash(hash_set, src2)
                         : (size_t)-1;

  const bool src0_needs_grads = src0 && isrc0 != v_HASHSET_FULL && v_bit_set_get(hash_set->used, isrc0) &&
    grads_needed[isrc0];
  const bool src1_needs_grads = src1 && isrc1 != v_HASHSET_FULL && v_bit_set_get(hash_set->used, isrc1) &&
    grads_needed[isrc1];
  const bool src2_needs_grads = src2 && isrc2 != v_HASHSET_FULL && v_bit_set_get(hash_set->used, isrc2) &&
    grads_needed[isrc2];

  switch (tensor->op) {
    case v_OP_DUP: { if (src0_needs_grads) { v_add_or_set(ctx, cgraph, isrc0, grad); } }
    break;
    case v_OP_ADD: {
      if (src0_needs_grads) { v_add_or_set(ctx, cgraph, isrc0, grad); }
      if (src1_needs_grads) {
        struct v_tensor* tmp = grad;
        if (!v_are_same_shape(src0, src1)) { tmp = v_repeat_back(ctx, tmp, src1); }
        v_add_or_set(ctx, cgraph, isrc1, tmp);
      }
    }
    break;
    case v_OP_ADD1: {
      if (src0_needs_grads) { v_add_or_set(ctx, cgraph, isrc0, grad); }
      if (src1_needs_grads) {
        v_add_or_set(ctx, cgraph, isrc1, v_mean(ctx, grad)); // TODO: should probably be sum instead of mean
      }
    }
    break;
    case v_OP_ACC: {
      if (src0_needs_grads) { v_add_or_set(ctx, cgraph, isrc0, grad); }
      if (src1_needs_grads) {
        const size_t nb1                  = ((int32_t*)tensor->op_params)[0];
        const size_t nb2                  = ((int32_t*)tensor->op_params)[1];
        const size_t nb3                  = ((int32_t*)tensor->op_params)[2];
        const size_t offset               = ((int32_t*)tensor->op_params)[3];
        struct v_tensor* tensor_grad_view = v_view_4d(ctx,
                                                      grad,
                                                      src1->ne[0],
                                                      src1->ne[1],
                                                      src1->ne[2],
                                                      src1->ne[3],
                                                      nb1,
                                                      nb2,
                                                      nb3,
                                                      offset);

        v_add_or_set(ctx, cgraph, isrc1, v_reshape(ctx, v_mem_cont(ctx, tensor_grad_view), src1));
      }
    }
    break;
    case v_OP_SUB: {
      if (src0_needs_grads) { v_add_or_set(ctx, cgraph, isrc0, grad); }
      if (src1_needs_grads) { v_sub_or_set(ctx, cgraph, isrc1, grad); }
    }
    break;
    case v_OP_MUL: {
      if (src0_needs_grads) { v_add_or_set(ctx, cgraph, isrc0, v_mul(ctx, grad, src1)); }
      if (src1_needs_grads) {
        struct v_tensor* tmp = v_mul(ctx, src0, grad);
        if (!v_are_same_shape(src0, src1)) { tmp = v_repeat_back(ctx, tmp, src1); }
        v_add_or_set(ctx, cgraph, isrc1, tmp);
      }
    }
    break;
    case v_OP_DIV: {
      if (src0_needs_grads) { v_add_or_set(ctx, cgraph, isrc0, v_div(ctx, grad, src1)); }
      if (src1_needs_grads) { v_sub_or_set(ctx, cgraph, isrc1, v_mul(ctx, grad, v_div(ctx, tensor, src1))); }
    }
    break;
    case v_OP_SQR: { if (src0_needs_grads) { v_add_or_set(ctx, cgraph, isrc0, v_scale(ctx, v_mul(ctx, src0, grad), 2.0f)); } }
    break;
    case v_OP_SQRT: { if (src0_needs_grads) { v_add_or_set(ctx, cgraph, isrc0, v_scale(ctx, v_div(ctx, grad, tensor), 0.5f)); } }
    break;
    case v_OP_LOG: { if (src0_needs_grads) { v_add_or_set(ctx, cgraph, isrc0, v_div(ctx, grad, src0)); } }
    break;
    case v_OP_SIN: { if (src0_needs_grads) { v_add_or_set(ctx, cgraph, isrc0, v_mul(ctx, grad, v_cos(ctx, src0))); } }
    break;
    case v_OP_COS: { if (src0_needs_grads) { v_sub_or_set(ctx, cgraph, isrc0, v_mul(ctx, grad, v_sin(ctx, src0))); } }
    break;
    case v_OP_SUM: { if (src0_needs_grads) { v_add1_or_set(ctx, cgraph, isrc0, grad); } }
    break;
    case v_OP_SUM_ROWS: { if (src0_needs_grads) { v_add_or_set(ctx, cgraph, isrc0, v_repeat(ctx, grad, src0)); } }
    break;
    case V_OP_MEAN: { if (src0_needs_grads) { v_add1_or_set(ctx, cgraph, isrc0, v_scale_impl(ctx, grad, 1.0f / src0->ne[0], 0.0, false)); } }
    break;
    case v_OP_REPEAT: { if (src0_needs_grads) { v_add_or_set(ctx, cgraph, isrc0, v_repeat_back(ctx, grad, src0)); } }
    break;
    case v_OP_REPEAT_BACK: { if (src0_needs_grads) { v_add_or_set(ctx, cgraph, isrc0, v_repeat(ctx, grad, src0)); } }
    break;
    case v_OP_RMS_NORM: {
      if (src0_needs_grads) {
        float eps;
        memcpy(&eps, tensor->op_params, sizeof(float));
        v_add_or_set(ctx, cgraph, isrc0, v_rms_norm_back(ctx, grad, src0, eps));
      }
    }
    break;
    case v_OP_MUL_MAT: {
      // https://cs231n.github.io/optimization-2/#staged
      // # forward pass
      // s0 = np.random.randn(5, 10)
      // s1 = np.random.randn(10, 3)
      // t = s0.dot(s1)

      // # now suppose we had the gradient on t from above in the circuit
      // dt = np.random.randn(*t.shape) # same shape as t
      // ds0 = dt.dot(s1.T) #.T gives the transpose of the matrix
      // ds1 = t.T.dot(dt)

      // tensor.shape [m,p,qq,rr]
      // src0.shape   [n,m,q1,r1]
      // src1.shape   [n,p,qq,rr]
      // grad         [m,p,qq,rr]
      if (src0_needs_grads) {
        printf("mat_mul called \n ");
        V_ASSERT(grad->ne[2] == src1->ne[2]);
        V_ASSERT(grad->ne[3] == src1->ne[3]);
        //src0 : [n,m, q1,r1]
        //src1 : [n,p, qq,rr]
        //grad : [m,p, qq,rr]
        //out prod: [n,m, qq,rr]
        // src1 can't transpose
        auto tSrc1 = v_reshape_2d(ctx, v_mem_cont(ctx, v_transpose(ctx, src1)),
                                  src1->ne[1],
                                  src1->ne[0]);
        auto tGrad           = v_transpose(ctx, v_mem_cont(ctx, grad));
        struct v_tensor* tmp = v_matmul(ctx,
                                        tSrc1,
                                        tGrad);
        // [n,p,qq,rr]
        // struct MmlTensor* tmp = v_out_prod(ctx, // [n,m,qq,rr]
        //                                      src1, // [p,n, qq,rr]
        //                                      grad); // [p,m,qq,rr]

        if (!v_are_same_shape(tmp, src0)) {
          V_ASSERT(tmp->ne[0] == src0->ne[0]);
          V_ASSERT(tmp->ne[1] == src0->ne[1]);
          V_ASSERT(tmp->ne[3] == 1);

          const int64_t nr2 = tmp->ne[2] / src0->ne[2];
          const size_t nb2  = tmp->nb[2] * nr2;
          const size_t nb3  = tmp->nb[2];

          tmp = v_view_4d(ctx, tmp, src0->ne[0], src0->ne[1], src0->ne[2], nr2, tmp->nb[1], nb2, nb3, 0);
          tmp = v_repeat_back(ctx, tmp, src0);
        }
        v_add_or_set(ctx, cgraph, isrc0, tmp);
      }
      if (src1_needs_grads) {
        // v_mul_mat(ctx,                   // [n,p,qq,rr]
        //     v_cont(ctx,                  // [m,n,q1,r1]
        //         v_transpose(ctx, src0)), // [m,n,q1,r1]
        //     grad),                          // [m,p,qq,rr]
        // when src0 is bigger than tensor->grad (this is mostly the case in llama),
        // avoid transpose of src0, rather transpose smaller tensor->grad
        // and then use v_out_prod
        //v_out_prod(ctx,      // [n,p,qq,rr]
        //    src0,               // [n,m,q1,r1]
        //    v_transpose(ctx, // [p,m,qq,rr]
        //        grad)));        // [m,p,qq,rr]
        // v_mul_mat(ctx,                   // [n,p,qq,rr]
        //     v_cont(ctx,                  // [m,n,q1,r1]
        //         v_transpose(ctx, src0)), // [m,n,q1,r1]
        //     grad),                          // [m,p,qq,rr]
        v_add_or_set(ctx,
                     cgraph,
                     isrc1,
                     v_matmul(ctx,
                              // [n,p,qq,rr]
                              v_mem_cont(ctx,
                                         // [m,n,q1,r1]
                                         v_transpose(ctx, src0)),
                              // [m,n,q1,r1]
                              grad)); // [m,p,qq,rr]
      }
    }
    break;
    case V_OP_SCALE: {
      if (src0_needs_grads) {
        float s;
        memcpy(&s, tensor->op_params, sizeof(float));
        v_add_or_set(ctx, cgraph, isrc0, v_scale_impl(ctx, grad, s, 0.0, false));
      }
    }
    break;
    case v_OP_SET: {
      const size_t nb1    = ((const int32_t*)tensor->op_params)[0];
      const size_t nb2    = ((const int32_t*)tensor->op_params)[1];
      const size_t nb3    = ((const int32_t*)tensor->op_params)[2];
      const size_t offset = ((const int32_t*)tensor->op_params)[3];

      struct v_tensor* tensor_grad_view = NULL;

      if (src0_needs_grads || src1_needs_grads) {
        V_ASSERT(src0->type == tensor->type);
        V_ASSERT(!cgraph->grads[isrc0] || cgraph->grads[isrc0]->type == grad->type);
        V_ASSERT(!cgraph->grads[isrc1] || !src1_needs_grads || cgraph->grads[isrc1]->type == grad->type);

        tensor_grad_view = v_view_4d(ctx,
                                     grad,
                                     src1->ne[0],
                                     src1->ne[1],
                                     src1->ne[2],
                                     src1->ne[3],
                                     nb1,
                                     nb2,
                                     nb3,
                                     offset);
      }

      if (src0_needs_grads) {
        struct v_tensor* tmp = v_sub(ctx,
                                     v_set_zero(tensor_grad_view),
                                     tensor_grad_view);

        //struct MmlTensor* tmp = v_neg(ctx, tensor_grad_view);
        v_add_or_set(ctx, cgraph, isrc0, v_acc_imple(ctx, grad, tmp, nb1, nb2, nb3, offset, false));
      }

      if (src1_needs_grads) { v_add_or_set(ctx, cgraph, isrc1, v_reshape(ctx, v_mem_cont(ctx, tensor_grad_view), src1)); }
    }
    break;
    case v_OP_CPY: {
      // cpy overwrites value of src1 by src0 and returns view(src1)
      // the overwriting is mathematically equivalent to:
      // tensor = src0 * 1 + src1 * 0
      if (src0_needs_grads) {
        // dsrc0 = dtensor * 1
        v_add_or_set(ctx, cgraph, isrc0, v_reshape(ctx, grad, src0));
      }
      if (src1_needs_grads) {
        // dsrc1 = dtensor * 0 -> noop
      }
    }
    break;
    case v_OP_CONT: {
      // same as cpy
      if (src0_needs_grads) {
        V_ASSERT(!cgraph->grads[isrc0] || v_is_contiguous(cgraph->grads[isrc0]));
        V_ASSERT(v_is_contiguous(grad));
        V_ASSERT(nelements(tensor) == nelements(src0));
        v_add_or_set(ctx,
                     cgraph,
                     isrc0,
                     v_are_same_shape(tensor, src0)
                       ? grad
                       : v_reshape(ctx, grad, src0));
      }
    }
    break;
    case v_OP_RESHAPE: {
      if (src0_needs_grads) {
        struct v_tensor* grad_cont = v_is_contiguous(grad)
                                       ? grad
                                       : v_mem_cont(ctx, grad);
        v_add_or_set(ctx, cgraph, isrc0, v_reshape(ctx, grad_cont, src0));
      }
    }
    break;
    case V_OP_VIEW: {
      if (src0_needs_grads) {
        size_t offset;

        memcpy(&offset, tensor->op_params, sizeof(offset));

        size_t nb1 = tensor->nb[1];
        size_t nb2 = tensor->nb[2];
        size_t nb3 = tensor->nb[3];

        if (cgraph->grads[isrc0] && src0->type != cgraph->grads[isrc0]->type) {
          // gradient is typically F32, but src0 could be other type
          size_t ng = v_element_size(cgraph->grads[isrc0]);
          size_t n0 = v_element_size(src0);
          V_ASSERT(offset % n0 == 0);
          V_ASSERT(nb1 % n0 == 0);
          V_ASSERT(nb2 % n0 == 0);
          V_ASSERT(nb3 % n0 == 0);
          offset = (offset / n0) * ng;
          nb1    = (nb1 / n0) * ng;
          nb2    = (nb2 / n0) * ng;
          nb3    = (nb3 / n0) * ng;
        }

        v_acc_or_set(ctx, cgraph, isrc0, grad, nb1, nb2, nb3, offset);
      }
    }
    break;
    case V_OP_PERMUTE: {
      if (src0_needs_grads) {
        const int32_t* axes = (const int32_t*)tensor->op_params;
        const int axis0     = axes[0] & 0x3;
        const int axis1     = axes[1] & 0x3;
        const int axis2     = axes[2] & 0x3;
        const int axis3     = axes[3] & 0x3;
        int axb[4]          = {0, 0, 0, 0}; // axes backward
        axb[axis0]          = 0;
        axb[axis1]          = 1;
        axb[axis2]          = 2;
        axb[axis3]          = 3;
        v_add_or_set(ctx, cgraph, isrc0, v_permute(ctx, grad, axb[0], axb[1], axb[2], axb[3]));
      }
    }
    break;
    case v_OP_TRANSPOSE: { if (src0_needs_grads) { v_add_or_set(ctx, cgraph, isrc0, v_transpose(ctx, grad)); } }
    break;
    case v_OP_GET_ROWS: {
      if (src0_needs_grads) { v_add_or_set(ctx, cgraph, isrc0, v_get_rows_back(ctx, grad, src1, src0)); }
      if (src1_needs_grads) {
        // noop
      }
    }
    break;
    case V_OP_DIAG_MASK_INF: {
      if (src0_needs_grads) {
        /* v_diag_mask_inf_impl() shouldn't be here */
        /* ref:  https://github.com/ggerganov/llama.cpp/pull/4203#discussion_r1412377992 */
        const int n_past = ((const int32_t*)tensor->op_params)[0];
        v_add_or_set(ctx, cgraph, isrc0, v_diag_mask_zero_impl(ctx, grad, n_past, false));
      }
    }
    break;
    case V_OP_DIAG_MASK_ZERO: {
      if (src0_needs_grads) {
        const int n_past = ((const int32_t*)tensor->op_params)[0];
        v_add_or_set(ctx, cgraph, isrc0, v_diag_mask_zero_impl(ctx, grad, n_past, false));
      }
    }
    break;
    case V_OP_SOFT_MAX: {
      if (src0_needs_grads) {
        float scale    = 1.0f;
        float max_bias = 0.0f;
        memcpy(&scale, (const float*)tensor->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float*)tensor->op_params + 1, sizeof(float));
        v_add_or_set(ctx, cgraph, isrc0, v_soft_max_ext_back(ctx, grad, tensor, scale, max_bias));
      }
      V_ASSERT((!src1 || !src1_needs_grads) && "backward pass for softmax mask not implemented");
    }
    break;
    case V_OP_ROPE: {
      if (src0_needs_grads) {
        //const int n_past = ((int32_t *) tensor->op_params)[0];
        const int n_dims = ((const int32_t*)tensor->op_params)[1];
        const int mode   = ((const int32_t*)tensor->op_params)[2];
        //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
        const int n_ctx_orig = ((const int32_t*)tensor->op_params)[4];
        float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
        int sections[4] = {0, 0, 0, 0};

        memcpy(&freq_base, (const float*)tensor->op_params + 5, sizeof(float));
        memcpy(&freq_scale, (const float*)tensor->op_params + 6, sizeof(float));
        memcpy(&ext_factor, (const float*)tensor->op_params + 7, sizeof(float));
        memcpy(&attn_factor, (const float*)tensor->op_params + 8, sizeof(float));
        memcpy(&beta_fast, (const float*)tensor->op_params + 9, sizeof(float));
        memcpy(&beta_slow, (const float*)tensor->op_params + 10, sizeof(float));
        memcpy(&sections, tensor->op_params + 11, sizeof(sections));

        struct v_tensor* rope_back = grad->ne[2] == src1->ne[0]
                                       ? v_rope_ext_back(ctx,
                                                         grad,
                                                         src1,
                                                         src2,
                                                         n_dims,
                                                         mode,
                                                         n_ctx_orig,
                                                         freq_base,
                                                         freq_scale,
                                                         ext_factor,
                                                         attn_factor,
                                                         beta_fast,
                                                         beta_slow)
                                       : v_rope_multi_back(ctx,
                                                           grad,
                                                           src1,
                                                           src2,
                                                           n_dims,
                                                           sections,
                                                           mode,
                                                           n_ctx_orig,
                                                           freq_base,
                                                           freq_scale,
                                                           ext_factor,
                                                           attn_factor,
                                                           beta_fast,
                                                           beta_slow);
        v_add_or_set(ctx, cgraph, isrc0, rope_back);
      }
      V_ASSERT((!src2 || !src2_needs_grads) && "gradients for freq factors not implemented");
    }
    break;
    case V_OP_IM2COL: {
      if (src1_needs_grads) {
        const int32_t s0 = v_get_op_params_i32(tensor, 0);
        const int32_t s1 = v_get_op_params_i32(tensor, 1);
        const int32_t p0 = v_get_op_params_i32(tensor, 2);
        const int32_t p1 = v_get_op_params_i32(tensor, 3);
        const int32_t d0 = v_get_op_params_i32(tensor, 4);
        const int32_t d1 = v_get_op_params_i32(tensor, 5);
        const bool is_2D = v_get_op_params_i32(tensor, 6) == 1;
        v_add_or_set(ctx, cgraph, isrc1, v_im2col_back(ctx, grad, src0, src1->ne, s0, s1, p0, p1, d0, d1, is_2D));
      }
    }
    break;
    case V_OP_POOL_2D: {
      if (src0_needs_grads) {
        const auto op = static_cast<v_op_pool>(v_get_op_params_i32(tensor, 0));
        const auto k0 = static_cast<int32_t>(v_get_op_params_i32(tensor, 1));
        const auto k1 = static_cast<int32_t>(v_get_op_params_i32(tensor, 2));
        const auto s0 = static_cast<int32_t>(v_get_op_params_i32(tensor, 3));
        const auto s1 = static_cast<int32_t>(v_get_op_params_i32(tensor, 4));
        const auto p0 = static_cast<int32_t>(v_get_op_params_i32(tensor, 5));
        const auto p1 = static_cast<int32_t>(v_get_op_params_i32(tensor, 6));
        v_add_or_set(ctx, cgraph, isrc0, v_pool_2d_back(ctx, grad, src0, op, k0, k1, s0, s1, p0, p1));
      }
    }
    break;
    case v_OP_WIN_PART:
    case v_OP_WIN_UNPART:
    case v_OP_UNARY: {
      switch (v_get_unary_op(tensor)) {
        case v_UNARY_OP_ABS: { if (src0_needs_grads) { v_add_or_set(ctx, cgraph, isrc0, v_mul(ctx, v_sgn(ctx, src0), grad)); } }
        break;
        case v_UNARY_OP_SGN: {
          // noop
        }
        break;
        case v_UNARY_OP_NEG: { if (src0_needs_grads) { v_sub_or_set(ctx, cgraph, isrc0, grad); } }
        break;
        case v_UNARY_OP_STEP: {
          // noop
        }
        break;
        case v_UNARY_OP_RELU: { if (src0_needs_grads) { v_add_or_set(ctx, cgraph, isrc0, v_mul(ctx, v_step(ctx, src0), grad)); } }
        break;
        case v_UNARY_OP_SILU: { if (src0_needs_grads) { v_add_or_set(ctx, cgraph, isrc0, v_silu_back(ctx, grad, src0)); } }
        break;
        case v_UNARY_OP_EXP: { if (src0_needs_grads) { v_add_or_set(ctx, cgraph, isrc0, v_mul(ctx, tensor, grad)); } }
        case v_UNARY_OP_LOG: { if (src0_needs_grads) { v_add_or_set(ctx, cgraph, isrc0, v_div(ctx, grad, src0)); } }
        break;
        default: {
          fprintf(stderr,
                  "%s: unsupported unary op for backward pass: %s\n",
                  __func__,
                  v_unary_op_name(v_get_unary_op(tensor)));
          v_ABORT("fatal error");
        } //break;
      }
    }
    break;
    case v_OP_CROSS_ENTROPY_LOSS: {
      if (src0_needs_grads) { v_add_or_set(ctx, cgraph, isrc0, v_cross_entropy_loss_back(ctx, grad, src0, src1)); }
      V_ASSERT(!src1_needs_grads && "backward pass for labels not implemented");
    }
    break;
    case v_OP_GLU: {
      switch (v_get_glu_op(tensor)) {
        case v_GLU_OP_SWIGLU: {
          if (src0_needs_grads) {
            V_ASSERT(src1 && "backward pass only implemented for split swiglu");
            v_add_or_set(ctx, cgraph, isrc0, v_silu_back(ctx, v_mul(ctx, grad, src1), src0));
          }
          if (src1_needs_grads) { v_add_or_set(ctx, cgraph, isrc1, v_mul(ctx, v_silu(ctx, src0), grad)); }
        }
        break;
        default: { v_ABORT("unsupported glu op for backward pass: %s", v_glu_op_name(v_get_glu_op(tensor))); } //break;
      }
    }
    break;
    case v_OP_NONE: {
      // noop
    }
    break;
    case v_OP_COUNT:
    default: { v_ABORT("%s: unsupported ggml op for backward pass: %s\n", __func__, v_op_name(tensor->op)); } //break;
  }

  V_ASSERT(!src0_needs_grads || v_are_same_shape(src0, cgraph->grads[isrc0]));
  V_ASSERT(!src1_needs_grads || v_are_same_shape(src1, cgraph->grads[isrc1]));
  V_ASSERT(!src2_needs_grads || v_are_same_shape(src2, cgraph->grads[isrc2]));
}

void vk_host_buffer_get_tensor(v_backend_buffer_t buffer, const struct v_tensor* tensor,
                               void* data, size_t offset, size_t size);

void v_get_backend_tensor(const struct v_tensor* tensor,
                          void* data,
                          size_t offset,
                          size_t size) {
  V_ASSERT(tensor);
  v_backend_buffer_t buf = tensor->view_src
                             ? tensor->view_src->buffer
                             : tensor->buffer;
  if (size == 0) { return; }
  V_ASSERT(buf != NULL && "tensor buffer not set");
  V_ASSERT(tensor->data != NULL && "tensor not allocated");
  V_ASSERT(offset + size <= num_bytes(tensor) && "tensor read out of bounds");
  buf->buft->host
    ? vk_host_buffer_get_tensor(buf, tensor, data, offset, size)
    : vk_device_buffer_get_tensor(buf, tensor, data, offset, size);
}

void vk_host_buffer_set_tensor(v_backend_buffer_t buffer, struct v_tensor* tensor,
                               const void* data, size_t offset, size_t size);

void v_set_backend_tensor(struct v_tensor* tensor, const void* data, size_t offset, size_t size) {
  V_ASSERT(tensor);
  V_ASSERT(tensor->data != NULL && "tensor not allocated");
  V_ASSERT(offset + size <= num_bytes(tensor) && "tensor write out of bounds");
  v_backend_buffer_t buf = tensor->view_src
                             ? tensor->view_src->buffer
                             : tensor->buffer;
  if (size == 0) return;

  V_ASSERT(buf != NULL && "tensor buffer not set");
  V_ASSERT(tensor->data != NULL && "tensor not allocated");
  V_ASSERT(offset + size <= num_bytes(tensor) && "tensor write out of bounds");
  buf->buft->host
    ? vk_host_buffer_set_tensor(buf, tensor, data, offset, size)
    : vk_device_buffer_set_tensor(buf, tensor, data, offset, size);
}
