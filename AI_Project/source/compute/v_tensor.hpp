/// All code is adapted from ggml for personal educational purposes.(study, clone coding)
/// Core code under license is sourced from ggml (https://github.com/ggerganov/ggml)
#pragma once
#include <array>
#include "v_header.hpp"


/**
 * @brief Tensor object representing multi-dimensional data.
 * The Tensor class stores the data buffer, dimensions, strides,
 * and provides utility functions to create views and perform operations without copying data.
 */

struct v_tensor {
  v_data_type type;
  v_backend_buffer* buffer;
  std::array<int64_t, V_MAX_DIMS> ne;             ///< numbers of elements;
  std::array<int64_t, V_MAX_DIMS> nb;             ///< dim bytes;
  std::array<v_tensor*, V_MAX_SRC> src;           ///< tensor node source
  std::array<int32_t, V_NUM_OP_PARAMS> op_params; ///< operation paramters
  std::array<char, V_MAX_NAME> name;
  v_operation op; ///< computing operation node
  v_tensor* view_src;
  int32_t flags;
  size_t view_offs;
  void* data;
  char padding[8];
  void set_name(const char* name);
  void set_inputs();
  void set_outputs();
  void set_params();
  bool is_transposed();
  bool is_permuted();
  bool is_scalar();
  bool is_vector();
  bool is_matrix();
  bool is_3d();
  bool is_contiguous();
  bool is_contiguous_0();           // same as v_is_contiguous()
  bool is_contiguous_1();           // contiguous for dims >= 1
  bool is_contiguous_2();           // contiguous for dims >= 2
  bool is_contiguously_allocated(); // returns whether the tensor elements are allocated as one contiguous block of memory (no gaps, but permutation ok)
  bool is_contiguous_channels();    // true for tensor that is stored in memory as CxWxHxN and has been permuted to WxHxCxN
  // true if the elements in dimension 0 are contiguous, or there is just 1 block of elements
  bool is_contiguous_rows();
  bool is_empty() const;
};

/**
 * @brief Creates New tensor.
 * for all new tensor object is allocated in ctx memory
 * @param ctx    ptr to the computation context
 * @param type   data types
 * @param n_dims Number of elements dimension 0
 * @param ne     Number of elements along each dimension
 * @return       new Tensor object
 */
v_tensor* v_new_tensor(v_ctx* ctx, v_data_type type, int n_dims, const int64_t* ne);
//creates 1d tensor. call v_new_tensor()
v_tensor* v_new_tensor_1d(v_ctx* ctx, v_data_type type, int64_t ne0);
//creates 2d tensor. call v_new_tensor()
v_tensor* v_new_tensor_2d(v_ctx* ctx, v_data_type type, int64_t ne0, int64_t ne1);
//creates 3d tensor. call v_new_tensor()
v_tensor* v_new_tensor_3d(v_ctx* ctx, v_data_type type, int64_t ne0, int64_t ne1, int64_t ne2);
//creates 4d tensor. call v_new_tensor()
v_tensor* v_new_tensor_4d(v_ctx* ctx, v_data_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);
//creates a tensor. call new object to ctx
v_tensor* v_new_tensor_impl(v_ctx* ctx, v_data_type type, int n_dims, const int64_t* ne, v_tensor* view_src, size_t view_offs);
//create tensor with same structure of source
v_tensor* v_dup_tensor(v_ctx* ctx, const v_tensor* src);


/**
 * @brief Creates a view on an existing tensor.
 * This function does not copy data; it creates a lightweight view
 * with specified dimensions, strides, and offset.
 * @param ctx       Pointer to the computation context
 * @param a         Source tensor
 * @param n_dims    Number of dims
 * @param ne        tensor dim_elements size_ptr
 * @param offset    Byte offset from the start of the source tensor
 * @return          Pointer to the view tensor (does not own the data)
 */
v_tensor* v_view_impl(v_ctx* ctx, v_tensor* a, int n_dims, const int64_t* ne, size_t offset);
//Creates a view on an existing tensor. call v_view_impl()
v_tensor* v_view_1d(v_ctx* ctx, v_tensor* a, int64_t ne0, size_t offset);
//Creates a view on an existing tensor. call v_view_impl()
v_tensor* v_view_2d(v_ctx* ctx, v_tensor* a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset);
//Creates a view on an existing tensor. call v_view_impl()
v_tensor* v_view_3d(v_ctx* ctx, v_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t offset);
//Creates a view on an existing tensor. call v_view_impl()
v_tensor* v_view_4d(v_ctx* ctx, v_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t offset);
//Creates a view on an existing tensor with same stride, same dims
v_tensor* v_tensor_view(v_ctx* ctx, v_tensor* src);
/**
 * @brief Creates a view on an existing tensor reshaped.
 * as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
 * @param ctx       Pointer to the computation context
 * @param a         src tensor
 * @param b         dst tensor view of new shape
 * @return          Pointer to the view b specifies the new shape
 */
// TODO: when we start computing gradient, make a copy instead of view
v_tensor* v_reshape(v_ctx* ctx, v_tensor* a, v_tensor* b);
//Creates view tensor and return with reshaped 1d. call v_reshape()
v_tensor* v_reshape_1d(v_ctx* ctx, v_tensor* a, int64_t ne0);
//Creates view tensor and return with reshaped 2d. call v_reshape()
v_tensor* v_reshape_2d(v_ctx* ctx, v_tensor* a, int64_t ne0, int64_t ne1);
//Creates view tensor and return with reshaped 3d. call v_reshape()
v_tensor* v_reshape_3d(v_ctx* ctx, v_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2);
//Creates view tensor and return with reshaped 4d. call v_reshape()
v_tensor* v_reshape_4d(v_ctx* ctx, v_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

//util functions return shape t0 == t1
bool v_are_same_shape(const v_tensor* t0, const v_tensor* t1);
//util functions return stride t0 == t1
bool v_are_same_stride(const v_tensor* t0, const v_tensor* t1);

/**
 * @brief create new tensor with new shape
 * duplicate and make new shape of tensors to mem contiguous
 * all dst dim element must same with source
 * @param ctx       ptr to the computation context
 * @param a         src tensor
 * @return          ptr to the tensor specifies the new shape, with mem contiguous
 */
v_tensor* v_cont(v_ctx* ctx, v_tensor* a);
//util function return v_cont_4d with src
v_tensor* v_cont_1d(v_ctx* ctx, v_tensor* a, int64_t ne0);
//util function return v_cont_4d with src
v_tensor* v_cont_2d(v_ctx* ctx, v_tensor* a, int64_t ne0, int64_t ne1);
//util function return v_cont_4d with src
v_tensor* v_cont_3d(v_ctx* ctx, v_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2);
//operations s.t create new tensor with new shape, mem contigious
v_tensor* v_cont_4d(v_ctx* ctx, v_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);


v_tensor* v_permute(v_ctx* ctx, v_tensor* a, int axis0, int axis1, int axis2, int axis3);
// alias for v_permute(ctx, a, 1, 0, 2, 3)
v_tensor* v_transpose(v_ctx* ctx, v_tensor* a);
v_tensor* v_get_rows(
  struct v_ctx* ctx,
  v_tensor* a,  // data
  v_tensor* b); // row indices

V_API v_tensor* v_get_rows_back(
  struct v_ctx* ctx,
  v_tensor* a,  // gradients of v_get_rows result
  v_tensor* b,  // row indices
  v_tensor* c); // data for v_get_rows, only used_bits__ for its shape

inline constexpr size_t V_TENSOR_SIZE = sizeof(v_tensor);
static_assert(V_TENSOR_SIZE % V_MEM_ALIGN == 0, "v_tensor size must be a multiple of V_MEM_ALIGN");
