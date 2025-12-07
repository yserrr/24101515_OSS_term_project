/// All code is adapted from ggml for educational purposes.(study, clone coding)
/// Core code under license is sourced from ggml (https://github.com/ggerganov/ggml)

#pragma once
#define v_API extern
#ifdef __GNUC__
#    define v_DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define v_DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define v_DEPRECATED(func, hint) func
#endif

#ifndef __GNUC__
#    define v_ATTRIBUTE_FORMAT(...)
#elif defined(__MINGW32__) && !defined(__clang__)
#    define v_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#    define v_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define v_FILE_MAGIC   0x67676d6c // "ggml"
#define v_FILE_VERSION 2

#define v_QNT_VERSION        2    // bump this on quantization format changes
#define v_QNT_VERSION_FACTOR 1000 // do not change this

#define v_MAX_PARAMS         2048
#define v_MAX_SRC            10
#define v_MAX_N_THREADS      512


#define v_DEFAULT_N_THREADS  4
#define v_DEFAULT_GRAPH_SIZE 2048

#if UINTPTR_MAX == 0xFFFFFFFF
#define v_MEM_ALIGN 4
#else
#define v_MEM_ALIGN 16
#endif

#define v_EXIT_SUCCESS 0
#define v_EXIT_ABORTED 1

// TODO: convert to enum https://github.com/ggml-org/llama.cpp/pull/16187#discussion_r2388538726
#define v_ROPE_TYPE_NORMAL 0
#define v_ROPE_TYPE_NEOX   2
#define v_ROPE_TYPE_MROPE  8
#define v_ROPE_TYPE_VISION 24

#define v_MROPE_SECTIONS   4

#define v_UNUSED(x) (void)(x)

#define v_UNUSED_VARS(...) do { (void)sizeof((__VA_ARGS__, 0)); } while(0)

#define MML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

#ifndef NDEBUG
#   define v_UNREACHABLE() do { fprintf(stderr, "statement should be unreachable\n"); abort(); } while(0)
#elif defined(__GNUC__)
#   define v_UNREACHABLE() __builtin_unreachable()
#elif defined(_MSC_VER)
#   define v_UNREACHABLE() __assume(0)
#else
#   define v_UNREACHABLE() ((void) 0)
#endif

#ifdef __cplusplus
#   define v_NORETURN [[noreturn]]
#elif defined(_MSC_VER)
#   define v_NORETURN __declspec(noreturn)
#else
#   define v_NORETURN _Noreturn
#endif

#define v_ABORT(...) v_abort(__FILE__, __LINE__, __VA_ARGS__)
#define V_ASSERT(x) if (!(x)) v_ABORT("v_ASSERT(%s) failed", #x)


#define v_TENSOR_LOCALS_1(type, prefix, pointer, array) \
    const type prefix##0 = (pointer) ? (pointer)->array[0] : 0; \
    v_UNUSED(prefix##0);
#define v_TENSOR_LOCALS_2(type, prefix, pointer, array) \
    v_TENSOR_LOCALS_1    (type, prefix, pointer, array) \
    const type prefix##1 = (pointer) ? (pointer)->array[1] : 0; \
    v_UNUSED(prefix##1);
#define v_TENSOR_LOCALS_3(type, prefix, pointer, array) \
    v_TENSOR_LOCALS_2    (type, prefix, pointer, array) \
    const type prefix##2 = (pointer) ? (pointer)->array[2] : 0; \
    v_UNUSED(prefix##2);
#define v_TENSOR_LOCALS(type, prefix, pointer, array) \
    v_TENSOR_LOCALS_3  (type, prefix, pointer, array) \
    const type prefix##3 = (pointer) ? (pointer)->array[3] : 0; \
    v_UNUSED(prefix##3);

#define v_TENSOR_UNARY_OP_LOCALS \
    v_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    v_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    v_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    v_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

#define v_TENSOR_BINARY_OP_LOCALS \
    v_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    v_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    v_TENSOR_LOCALS(int64_t, ne1, src1, ne) \
    v_TENSOR_LOCALS(size_t,  nb1, src1, nb) \
    v_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    v_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

#define v_TENSOR_TERNARY_OP_LOCALS \
    v_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    v_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    v_TENSOR_LOCALS(int64_t, ne1, src1, ne) \
    v_TENSOR_LOCALS(size_t,  nb1, src1, nb) \
    v_TENSOR_LOCALS(int64_t, ne2, src2, ne) \
    v_TENSOR_LOCALS(size_t,  nb2, src2, nb) \
    v_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    v_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

#define v_TENSOR_BINARY_OP_LOCALS01 \
    v_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    v_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    v_TENSOR_LOCALS(int64_t, ne1, src1, ne) \
    v_TENSOR_LOCALS(size_t,  nb1, src1, nb)

// Function type used in fatal error callbacks
typedef void (*v_abort_callback_t)(const char* error_message);

// Set the abort callback (passing null will restore original abort functionality: printing a message to stdout)
// Returns the old callback for chaining
v_API v_abort_callback_t v_set_abort_callback(v_abort_callback_t callback);

v_NORETURN v_ATTRIBUTE_FORMAT(3, 4)
v_API void v_abort(const char* file, int line, const char* fmt, ...);

enum v_status {
  v_STATUS_ALLOC_FAILED = -2,
  v_STATUS_FAILED       = -1,
  v_STATUS_SUCCESS      = 0,
  v_STATUS_ABORTED      = 1,
};

// get v_status name string
v_API const char* v_status_to_string(enum v_status status);
// ieee 754-2008 half-precision float16
// todo: make this not an integral type
typedef uint16_t v_fp16_t;
v_API float v_fp16_to_fp32(v_fp16_t);
v_API v_fp16_t v_fp32_to_fp16(float);
v_API void v_fp16_to_fp32_row(const v_fp16_t*, float*, int64_t);
v_API void v_fp32_to_fp16_row(const float*, v_fp16_t*, int64_t);
#include "v_log.hpp"

// google brain half-precision bfloat16
typedef struct {
  uint16_t bits;
} v_bf16_t;

v_API v_bf16_t v_fp32_to_bf16(float);
v_API float v_bf16_to_fp32(v_bf16_t); // consider just doing << 16
v_API void v_bf16_to_fp32_row(const v_bf16_t*, float*, int64_t);
v_API void v_fp32_to_bf16_row_ref(const float*, v_bf16_t*, int64_t);
v_API void v_fp32_to_bf16_row(const float*, v_bf16_t*, int64_t);

struct v_object;
struct v_ctx;
struct v_cgraph;

// NOTE: always add types at the end of the enum to keep backward compatibility
enum v_data_type {
  v_TYPE_F32  = 0,
  v_TYPE_F16  = 1,
  v_TYPE_Q4_0 = 2,
  v_TYPE_Q4_1 = 3,
  // v_TYPE_Q4_2 = 4, support has been removed
  // v_TYPE_Q4_3 = 5, support has been removed
  v_TYPE_Q5_0    = 6,
  v_TYPE_Q5_1    = 7,
  v_TYPE_Q8_0    = 8,
  v_TYPE_Q8_1    = 9,
  v_TYPE_Q2_K    = 10,
  v_TYPE_Q3_K    = 11,
  v_TYPE_Q4_K    = 12,
  v_TYPE_Q5_K    = 13,
  v_TYPE_Q6_K    = 14,
  v_TYPE_Q8_K    = 15,
  v_TYPE_IQ2_XXS = 16,
  v_TYPE_IQ2_XS  = 17,
  v_TYPE_IQ3_XXS = 18,
  v_TYPE_IQ1_S   = 19,
  v_TYPE_IQ4_NL  = 20,
  v_TYPE_IQ3_S   = 21,
  v_TYPE_IQ2_S   = 22,
  v_TYPE_IQ4_XS  = 23,
  v_TYPE_I8      = 24,
  v_TYPE_I16     = 25,
  v_TYPE_I32     = 26,
  v_TYPE_I64     = 27,
  v_TYPE_F64     = 28,
  v_TYPE_IQ1_M   = 29,
  v_TYPE_BF16    = 30,
  // v_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
  // v_TYPE_Q4_0_4_8 = 32,
  // v_TYPE_Q4_0_8_8 = 33,
  v_TYPE_TQ1_0 = 34,
  v_TYPE_TQ2_0 = 35,
  // v_TYPE_IQ4_NL_4_4 = 36,
  // v_TYPE_IQ4_NL_4_8 = 37,
  // v_TYPE_IQ4_NL_8_8 = 38,
  v_TYPE_MXFP4 = 39, // MXFP4 (1 block)
  v_TYPE_COUNT = 40,
};

enum v_prec {
  v_PREC_DEFAULT = 0, // stored as v_tensor.op_params, 0 by default
  v_PREC_F32     = 10,
};

enum v_ftype {
  v_FTYPE_UNKNOWN              = -1,
  v_FTYPE_ALL_F32              = 0,
  v_FTYPE_MOSTLY_F16           = 1, // except 1d tensors
  v_FTYPE_MOSTLY_Q4_0          = 2, // except 1d tensors
  v_FTYPE_MOSTLY_Q4_1          = 3, // except 1d tensors
  v_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
  v_FTYPE_MOSTLY_Q8_0          = 7, // except 1d tensors
  v_FTYPE_MOSTLY_Q5_0          = 8, // except 1d tensors
  v_FTYPE_MOSTLY_Q5_1          = 9, // except 1d tensors
  v_FTYPE_MOSTLY_Q2_K          = 10, // except 1d tensors
  v_FTYPE_MOSTLY_Q3_K          = 11, // except 1d tensors
  v_FTYPE_MOSTLY_Q4_K          = 12, // except 1d tensors
  v_FTYPE_MOSTLY_Q5_K          = 13, // except 1d tensors
  v_FTYPE_MOSTLY_Q6_K          = 14, // except 1d tensors
  v_FTYPE_MOSTLY_IQ2_XXS       = 15, // except 1d tensors
  v_FTYPE_MOSTLY_IQ2_XS        = 16, // except 1d tensors
  v_FTYPE_MOSTLY_IQ3_XXS       = 17, // except 1d tensors
  v_FTYPE_MOSTLY_IQ1_S         = 18, // except 1d tensors
  v_FTYPE_MOSTLY_IQ4_NL        = 19, // except 1d tensors
  v_FTYPE_MOSTLY_IQ3_S         = 20, // except 1d tensors
  v_FTYPE_MOSTLY_IQ2_S         = 21, // except 1d tensors
  v_FTYPE_MOSTLY_IQ4_XS        = 22, // except 1d tensors
  v_FTYPE_MOSTLY_IQ1_M         = 23, // except 1d tensors
  v_FTYPE_MOSTLY_BF16          = 24, // except 1d tensors
  v_FTYPE_MOSTLY_MXFP4         = 25, // except 1d tensors
};


// model file types

#include "v_op.hpp"


enum v_log_level {
  v_LOG_LEVEL_NONE  = 0,
  v_LOG_LEVEL_DEBUG = 1,
  v_LOG_LEVEL_INFO  = 2,
  v_LOG_LEVEL_WARN  = 3,
  v_LOG_LEVEL_ERROR = 4,
  v_LOG_LEVEL_CONT  = 5, // continue previous log
};

#include "v_object.hpp"
#include "v_ctx.hpp"


struct v_init_param {
  // memory pool
  size_t mem_size; // bytes
  void* mem_buffer; // if NULL, memory will be allocated internally
  bool no_alloc; // don't allocate memory for the tensor data
};

#include "v_tensor.hpp"
#include "v_graph.hpp"
// n-dimensional tensor


static const size_t v_TENSOR_SIZE = sizeof(v_tensor);

// Abort callback
// If not NULL, called before ggml computation
// If it returns true, the computation is aborted
typedef bool (*v_abort_callback)(void* data);


#ifdef __cplusplus
// restrict not standard in C++
#    if defined(__GNUC__)
#        define v_RESTRICT __restrict__
#    elif defined(__clang__)
#        define v_RESTRICT __restrict
#    elif defined(_MSC_VER)
#        define v_RESTRICT  __restrict
#    else
#        define v_RESTRICT
#    endif
#else
#    if defined (_MSC_VER) && (__STDC_VERSION__ < 201112L)
#        define v_RESTRICT __restrict
#    else
#        define v_RESTRICT restrict
#    endif
#endif

typedef void (*v_to_float_t)(const void* v_RESTRICT x, float* v_RESTRICT y, int64_t k);
typedef void (*v_from_float_t)(const float* v_RESTRICT x, void* v_RESTRICT y, int64_t k);

struct v_type_traits {
  const char* type_name;
  int64_t blck_size;
  int64_t blck_size_interleave; // interleave elements in blocks
  size_t type_size;
  bool is_quantized;
  v_to_float_t to_float;
  v_from_float_t from_float_ref;
};


// misc

//v_API const char* v_version(void);
//v_API const char* v_commit(void);

v_API void v_time_init(void); // call this once at the beginning of the program
v_API int64_t v_time_ms(void);
v_API int64_t v_time_us(void);
v_API int64_t v_cycles(void);
v_API int64_t v_cycles_per_ms(void);

// accepts a UTF-8 path, even on Windows
v_API FILE* v_fopen(const char* fname, const char* mode);

v_API void v_print_object(const struct v_object* obj);
v_API void v_print_objects(const struct v_ctx* ctx);

v_tensor* new_tensor_impl(struct v_ctx* ctx,
                          enum v_data_type type,
                          int n_dims,
                          const int64_t* ne,
                          v_tensor* view_src,
                          size_t view_offs);
v_API int64_t nelements(const v_tensor* tensor);
v_API int64_t v_nrows(const v_tensor* tensor);
v_API size_t num_bytes(const v_tensor* tensor);
v_API size_t v_nbytes_pad(const v_tensor* tensor); // same as v_nbytes() but padded to v_MEM_ALIGN

v_API int64_t block_size(enum v_data_type type);
v_API size_t v_type_size(enum v_data_type type); // size in bytes for all elements in a block
v_API size_t v_row_size(enum v_data_type type, int64_t ne); // size in bytes for all elements in a row

v_DEPRECATED(
  v_API double v_type_sizef(enum v_data_type type),
  // v_type_size()/v_blck_size() as float
  "use v_row_size() instead");

v_API const char* v_type_name(enum v_data_type type);
v_API const char* v_op_name(enum v_operation op);
v_API const char* v_op_symbol(enum v_operation op);

v_API const char* v_unary_op_name(enum v_unary_op op);
v_API const char* v_glu_op_name(enum v_glu_op op);
v_API const char* v_op_desc(const v_tensor* t); // unary or op name

v_API size_t v_element_size(const v_tensor* tensor);

v_API bool v_is_quantized(enum v_data_type type);

// TODO: temporary until model loading of ggml examples is refactored
v_API enum v_data_type v_ftype_to_v_type(enum v_ftype ftype);

v_API bool v_is_transposed(const v_tensor* tensor);
v_API bool v_is_permuted(const v_tensor* tensor);
v_API bool is_empty(const v_tensor* tensor);
v_API bool v_is_scalar(const v_tensor* tensor);
v_API bool v_is_vector(const v_tensor* tensor);
v_API bool v_is_matrix(const v_tensor* tensor);
v_API bool v_is_3d(const v_tensor* tensor);

// returns whether the tensor elements can be iterated over with a flattened index (no gaps, no permutation)
v_API bool v_is_contiguous(const v_tensor* tensor);
v_API bool v_is_contiguous_0(const v_tensor* tensor); // same as v_is_contiguous()
v_API bool v_is_contiguous_1(const v_tensor* tensor); // contiguous for dims >= 1
v_API bool v_is_contiguous_2(const v_tensor* tensor); // contiguous for dims >= 2

// returns whether the tensor elements are allocated as one contiguous block of memory (no gaps, but permutation ok)
v_API bool v_is_contiguously_allocated(const v_tensor* tensor);

// true for tensor that is stored in memory as CxWxHxN and has been permuted to WxHxCxN
v_API bool v_is_contiguous_channels(const v_tensor* tensor);

// true if the elements in dimension 0 are contiguous, or there is just 1 block of elements
v_API bool v_is_contiguous_rows(const v_tensor* tensor);

v_API bool v_are_same_shape(const v_tensor* t0, const v_tensor* t1);
v_API bool v_are_same_stride(const v_tensor* t0, const v_tensor* t1);

v_API bool can_repeat(const v_tensor* t0, const v_tensor* t1);

// use this to compute the memory overhead of a tensor
v_API size_t v_tensor_over_head(void);

v_API bool v_validate_row_data(enum v_data_type type, const void* data, size_t nbytes);
struct v_object* v_new_object(struct v_ctx* ctx,
                              enum v_object_type type,
                              size_t size);
// main

v_API struct v_ctx* v_ctx_init(struct v_init_param params);
v_API void v_reset(struct v_ctx* ctx);
v_API void free_ctx(struct v_ctx* ctx);

v_API void* v_get_mem_buffer(const struct v_ctx* ctx);
v_API size_t v_get_mem_size(const struct v_ctx* ctx);
v_API size_t v_get_max_tensor_size(const struct v_ctx* ctx);

v_API v_tensor* v_new_tensor(
  struct v_ctx* ctx,
  enum v_data_type type,
  int n_dims,
  const int64_t* ne);

v_API v_tensor* v_new_tensor_1d(
  struct v_ctx* ctx,
  enum v_data_type type,
  int64_t ne0);

v_API v_tensor* v_new_tensor_2d(
  struct v_ctx* ctx,
  enum v_data_type type,
  int64_t ne0,
  int64_t ne1);

v_API v_tensor* v_new_tensor_3d(
  struct v_ctx* ctx,
  enum v_data_type type,
  int64_t ne0,
  int64_t ne1,
  int64_t ne2);

v_API v_tensor* v_new_tensor_4d(
  struct v_ctx* ctx,
  enum v_data_type type,
  int64_t ne0,
  int64_t ne1,
  int64_t ne2,
  int64_t ne3);

v_API void* v_new_buffer(struct v_ctx* ctx, size_t nbytes);

v_API v_tensor* v_dup_tensor(struct v_ctx* ctx, const v_tensor* src);
v_API v_tensor* v_tensor_view(struct v_ctx* ctx, v_tensor* src);

// Context tensor enumeration and lookup
v_API v_tensor* v_get_first_tensor(const struct v_ctx* ctx);
v_API v_tensor* v_get_next_tensor(const struct v_ctx* ctx, v_tensor* tensor);
v_API v_tensor* v_get_tensor_name(struct v_ctx* ctx, const char* name);

// Converts a flat index into coordinates
v_API void v_unravel_index(const v_tensor* tensor, int64_t i, int64_t* i0, int64_t* i1, int64_t* i2,
                           int64_t* i3);

v_API enum v_unary_op v_get_unary_op(const v_tensor* tensor);
v_API enum v_glu_op v_get_glu_op(const v_tensor* tensor);

v_API void* v_get_data(const v_tensor* tensor);
v_API float* v_get_tdata_f32(const v_tensor* tensor);

v_API const char* get_name(const v_tensor* tensor);
v_API v_tensor* v_set_name(v_tensor* tensor, const char* name);
v_ATTRIBUTE_FORMAT(2, 3)
v_API v_tensor* v_format_name(v_tensor* tensor, const char* fmt, ...);

v_API void v_set_inputs(v_tensor* tensor);
v_API void v_set_outputs(v_tensor* tensor);
v_API void v_set_params(v_tensor* tensor);
v_API void v_set_loss(v_tensor* tensor);


v_tensor* v_dup_impl(struct v_ctx* ctx,
                     v_tensor* a,
                     bool inplace);

v_API v_tensor* v_dup(
  struct v_ctx* ctx,
  v_tensor* a);

// in-place, returns view(a)
v_API v_tensor* v_dup_inplace(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_add(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b);

v_API v_tensor* v_add_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b);

v_API v_tensor* add_cast(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  enum v_data_type type);

// dst[i0, i1, i2] = a[i0, i1, i2] + b[i0, ids[i1, i2]]
v_API v_tensor* add_id(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  v_tensor* ids);

v_API v_tensor* add1(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b);

v_API v_tensor* v_add1_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b);


v_API v_tensor* v_sub(struct v_ctx* ctx,
                      v_tensor* a,
                      v_tensor* b);

v_API v_tensor* v_sub_inplace(struct v_ctx* ctx,
                              v_tensor* a,
                              v_tensor* b);

v_API v_tensor* v_mul(struct v_ctx* ctx,
                      v_tensor* a,
                      v_tensor* b);

v_API v_tensor* v_mul_inplace(struct v_ctx* ctx,
                              v_tensor* a,
                              v_tensor* b);

v_API v_tensor* v_div(struct v_ctx* ctx,
                      v_tensor* a,
                      v_tensor* b);

v_API v_tensor* v_div_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b);


v_tensor* set_impl(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  size_t nb1,
  size_t nb2,
  size_t nb3,
  size_t offset,
  bool inplace);

v_API v_tensor* v_sqr(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_sqr_inplace(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_sqrt(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_sqrt_inplace(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_log(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_log_inplace(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_sin(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_sin_inplace(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_cos(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* cos_inplace(
  struct v_ctx* ctx,
  v_tensor* a);

// return scalar
v_API v_tensor* v_sum(
  struct v_ctx* ctx,
  v_tensor* a);

// sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]
v_API v_tensor* v_sum_rows(
  struct v_ctx* ctx,
  v_tensor* a);

// mean along rows
v_API v_tensor* v_mean(
  struct v_ctx* ctx,
  v_tensor* a);

// argmax along rows
v_API v_tensor* v_argmax(
  struct v_ctx* ctx,
  v_tensor* a);

// count number of equal elements in a and b
v_API v_tensor* v_count_equal(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b);

// if a is the same shape as b, and a is not parameter, return a
// otherwise, return a new tensor: repeat(a) to fit in b
v_API v_tensor* v_repeat(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b);

// repeat a to the specified shape
v_API v_tensor* v_repeat_4d(
  struct v_ctx* ctx,
  v_tensor* a,
  int64_t ne0,
  int64_t ne1,
  int64_t ne2,
  int64_t ne3);

// sums repetitions in a into shape of b
v_API v_tensor* v_repeat_back(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b); // sum up values that are adjacent in dims > 0 instead of repeated with same stride

// concat a and b along dim
// used in stable-diffusion
v_API v_tensor* v_concat(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  int dim);

v_API v_tensor* v_abs(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_abs_inplace(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_sgn(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_sgn_inplace(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_neg(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_neg_inplace(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_step(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_step_inplace(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_tanh(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_tanh_inplace(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_elu(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_elu_inplace(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_relu(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_leaky_relu(
  struct v_ctx* ctx,
  v_tensor* a, float negative_slope, bool inplace);

v_API v_tensor* v_relu_inplace(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_sigmoid(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_sigmoid_inplace(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_gelu(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_gelu_inplace(
  struct v_ctx* ctx,
  v_tensor* a);

// GELU using erf (error function) when possible
// some backends may fallback to approximation based on Abramowitz and Stegun formula
v_API v_tensor* v_gelu_erf(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_gelu_erf_inplace(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_gelu_quick(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_gelu_quick_inplace(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_silu(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_silu_inplace(
  struct v_ctx* ctx,
  v_tensor* a);

// a - x
// b - dy
v_API v_tensor* v_silu_back(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b);

// hardswish(x) = x * relu6(x + 3) / 6
v_API v_tensor* v_hardswish(
  struct v_ctx* ctx,
  v_tensor* a);

// hardsigmoid(x) = relu6(x + 3) / 6
v_API v_tensor* v_hardsigmoid(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_exp(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_exp_inplace(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_floor(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_floor_inplace(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_ceil(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_ceil_inplace(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_round(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_round_inplace(
  struct v_ctx* ctx,
  v_tensor* a);

/**
* Truncates the fractional part of each element in the tensor (towards zero).
* For example: trunc(3.7) = 3.0, trunc(-2.9) = -2.0
* Similar to std::trunc in C/C++.
*/

v_API v_tensor* v_trunc(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_trunc_inplace(
  struct v_ctx* ctx,
  v_tensor* a);


// xIELU activation function
// x = x * (c_a(alpha_n) + c_b(alpha_p, beta) * sigmoid(beta * x)) + eps * (x > 0)
// where c_a = softplus and c_b(a, b) = softplus(a) + b are constraining functions
// that constrain the positive and negative source alpha values respectively
v_API v_tensor* v_xielu(
  struct v_ctx* ctx,
  v_tensor* a,
  float alpha_n,
  float alpha_p,
  float beta,
  float eps);

// gated linear unit ops
// A: n columns, r rows,
// result is n / 2 columns, r rows,
// expects gate in second half of row, unless swapped is true
v_API v_tensor* v_glu(
  struct v_ctx* ctx,
  v_tensor* a,
  enum v_glu_op op,
  bool swapped);

v_API v_tensor* v_reglu(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_reglu_swapped(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_geglu(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_geglu_swapped(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_swiglu(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_swiglu_swapped(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_geglu_erf(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_geglu_erf_swapped(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_geglu_quick(
  struct v_ctx* ctx,
  v_tensor* a);

v_API v_tensor* v_geglu_quick_swapped(
  struct v_ctx* ctx,
  v_tensor* a);

// A: n columns, r rows,
// B: n columns, r rows,
v_API v_tensor* v_glu_split(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  enum v_glu_op op);

v_API v_tensor* v_reglu_split(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b);

v_API v_tensor* v_geglu_split(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b);

v_API v_tensor* v_swiglu_split(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b);

v_API v_tensor* v_geglu_erf_split(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b);

v_API v_tensor* v_geglu_quick_split(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b);

v_API v_tensor* v_swiglu_oai(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  float alpha,
  float limit);

// normalize along rows
v_API v_tensor* v_norm(
  struct v_ctx* ctx,
  v_tensor* a,
  float eps);

v_API v_tensor* mmlNormInplace(
  struct v_ctx* ctx,
  v_tensor* a,
  float eps);

v_API v_tensor* v_rms_norm(
  struct v_ctx* ctx,
  v_tensor* a,
  float eps);

v_API v_tensor* v_rms_norm_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  float eps);

// group normalize along ne0*ne1*n_groups
// used in stable-diffusion
v_API v_tensor* v_group_norm(
  struct v_ctx* ctx,
  v_tensor* a,
  int n_groups,
  float eps);

v_API v_tensor* v_group_norm_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  int n_groups,
  float eps);

// l2 normalize along rows
// used in rwkv v7
v_API v_tensor* v_norm_l2(
  struct v_ctx* ctx,
  v_tensor* a,
  float eps);

v_API v_tensor* v_l2_norm_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  float eps);

// a - x
// b - dy
v_API v_tensor* v_rms_norm_back(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  float eps);

// A: k columns, n rows => [ne03, ne02, n, k]
// B: k columns, m rows  (i.e. we transpose it internally) => [ne03 * x, ne02 * y, m, k]
// result is n columns, m rows => [ne03 * x, ne02 * y, m, n]
v_API v_tensor* v_matmul(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b);

// change the precision of a matrix multiplication
// set to v_PREC_F32 for higher precision (useful for phi-2)
v_API void v_mul_mat_set_prec(
  v_tensor* a,
  enum v_prec prec);

// indirect matrix multiplication
v_API v_tensor* mmlMatrixMulId(
  struct v_ctx* ctx,
  v_tensor* as,
  v_tensor* b,
  v_tensor* ids);

// A: m columns, n rows,
// B: p columns, n rows,
// result is m columns, p rows
v_API v_tensor* v_out_prod(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b);

//
// operations on tensors without backpropagation
//

v_API v_tensor* v_scale(
  struct v_ctx* ctx,
  v_tensor* a,
  float s);

// in-place, returns view(a)
v_API v_tensor* v_scale_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  float s);

// x = s * a + b
v_API v_tensor* v_scale_bias(
  struct v_ctx* ctx,
  v_tensor* a,
  float s,
  float b);

v_API v_tensor* v_scale_bias_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  float s,
  float b);

// b -> view(a,offset,nb1,nb2,3), return modified a
v_API v_tensor* v_set(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  size_t nb1,
  size_t nb2,
  size_t nb3,
  size_t offset); // in bytes
bool v_is_contiguous_n(const v_tensor* tensor, int n);
v_tensor* v_group_norm_impl(struct v_ctx* ctx,
                            v_tensor* a,
                            int n_groups,
                            float eps,
                            bool inplace);

// b -> view(a,offset,nb1,nb2,3), return view(a)
v_API v_tensor* v_set_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  size_t nb1,
  size_t nb2,
  size_t nb3,
  size_t offset); // in bytes

v_API v_tensor* v_set_1d(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  size_t offset); // in bytes

v_API v_tensor* v_set_1d_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  size_t offset); // in bytes

// b -> view(a,offset,nb1,nb2,3), return modified a
v_API v_tensor* v_set_2d(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  size_t nb1,
  size_t offset); // in bytes

// b -> view(a,offset,nb1,nb2,3), return view(a)
v_API v_tensor* v_set_2d_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  size_t nb1,
  size_t offset); // in bytes

// a -> b, return view(b)
v_API v_tensor* v_cpy(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b);

// note: casting from f32 to i32 will discard the fractional part
v_API v_tensor* v_cast(
  struct v_ctx* ctx,
  v_tensor* a,
  enum v_data_type type);

// make contiguous
v_API v_tensor* v_mem_cont(
  struct v_ctx* ctx,
  v_tensor* a);

// make contiguous, with new shape
v_API v_tensor* v_cont_1d(
  struct v_ctx* ctx,
  v_tensor* a,
  int64_t ne0);

v_API v_tensor* v_cont_2d(
  struct v_ctx* ctx,
  v_tensor* a,
  int64_t ne0,
  int64_t ne1);

v_API v_tensor* v_cont_3d(
  struct v_ctx* ctx,
  v_tensor* a,
  int64_t ne0,
  int64_t ne1,
  int64_t ne2);

v_API v_tensor* v_cont_4d(
  struct v_ctx* ctx,
  v_tensor* a,
  int64_t ne0,
  int64_t ne1,
  int64_t ne2,
  int64_t ne3);

// return view(a), b specifies the new shape
// TODO: when we start computing gradient, make a copy instead of view
v_API v_tensor* v_reshape(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b);

// return view(a)
// TODO: when we start computing gradient, make a copy instead of view
v_API v_tensor* v_reshape_1d(
  struct v_ctx* ctx,
  v_tensor* a,
  int64_t ne0);

v_API v_tensor* v_reshape_2d(
  struct v_ctx* ctx,
  v_tensor* a,
  int64_t ne0,
  int64_t ne1);

// return view(a)
// TODO: when we start computing gradient, make a copy instead of view
v_API v_tensor* v_reshape_3d(
  struct v_ctx* ctx,
  v_tensor* a,
  int64_t ne0,
  int64_t ne1,
  int64_t ne2);

v_API v_tensor* v_reshape_4d(
  struct v_ctx* ctx,
  v_tensor* a,
  int64_t ne0,
  int64_t ne1,
  int64_t ne2,
  int64_t ne3);

// offset in bytes
v_API v_tensor* v_view_1d(
  struct v_ctx* ctx,
  v_tensor* a,
  int64_t ne0,
  size_t offset);

v_API v_tensor* v_view_2d(
  struct v_ctx* ctx,
  v_tensor* a,
  int64_t ne0,
  int64_t ne1,
  size_t nb1, // row stride in bytes
  size_t offset);

v_API v_tensor* v_view_3d(
  struct v_ctx* ctx,
  v_tensor* a,
  int64_t ne0,
  int64_t ne1,
  int64_t ne2,
  size_t nb1, // row   stride in bytes
  size_t nb2, // slice stride in bytes
  size_t offset);

v_API v_tensor* v_view_4d(
  struct v_ctx* ctx,
  v_tensor* a,
  int64_t ne0,
  int64_t ne1,
  int64_t ne2,
  int64_t ne3,
  size_t nb1, // row   stride in bytes
  size_t nb2, // slice stride in bytes
  size_t nb3,
  size_t offset);

v_API v_tensor* v_permute(
  struct v_ctx* ctx,
  v_tensor* a,
  int axis0,
  int axis1,
  int axis2,
  int axis3);

// alias for v_permute(ctx, a, 1, 0, 2, 3)
v_API v_tensor* v_transpose(
  struct v_ctx* ctx,
  v_tensor* a);

// supports 4D a:
// a     [n_embd, ne1, ne2, ne3]
// b I32 [n_rows, ne2, ne3, 1]
//
// return [n_embd, n_rows, ne2, ne3]
v_API v_tensor* v_get_rows(
  struct v_ctx* ctx,
  v_tensor* a, // data
  v_tensor* b); // row indices

v_API v_tensor* v_get_rows_back(
  struct v_ctx* ctx,
  v_tensor* a, // gradients of v_get_rows result
  v_tensor* b, // row indices
  v_tensor* c); // data for v_get_rows, only used for its shape

// a TD  [n_embd, ne1,    ne2,    ne3]
// b TS  [n_embd, n_rows, ne02,   ne03] | ne02 == ne2, ne03 == ne3
// c I64 [n_rows, ne11,   ne12,   1]    | c[i] in [0, ne1)
//
// undefined behavior if destination rows overlap
//
// broadcast:
//   ne2 % ne11 == 0
//   ne3 % ne12 == 0
//
// return view(a)
v_API v_tensor* v_set_rows(struct v_ctx* ctx,
                           v_tensor* a, // destination
                           v_tensor* b, // source
                           v_tensor* c); // row indices
v_tensor* v_glu_impl(struct v_ctx* ctx,
                     v_tensor* a,
                     v_tensor* b,
                     enum v_glu_op op,
                     bool swapped);

v_tensor* v_norm_impl(struct v_ctx* ctx,
                      v_tensor* a,
                      float eps,
                      bool inplace);

v_tensor* v_rms_norm_impl(struct v_ctx* ctx,
                          v_tensor* a,
                          float eps,
                          bool inplace);

v_tensor* v_l2_norm_impl(struct v_ctx* ctx,
                         v_tensor* a,
                         float eps,
                         bool inplace);
v_API v_tensor* v_diag(
  struct v_ctx* ctx,
  v_tensor* a);

// set elements above the diagonal to -INF
v_API v_tensor* v_diag_mask_inf(
  struct v_ctx* ctx,
  v_tensor* a,
  int n_past);

// in-place, returns view(a)
v_API v_tensor* v_diag_mask_inf_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  int n_past);

// set elements above the diagonal to 0
v_API v_tensor* v_diag_mask_zero(
  struct v_ctx* ctx,
  v_tensor* a,
  int n_past);

// in-place, returns view(a)
v_API v_tensor* v_diag_mask_zero_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  int n_past);

v_API v_tensor* v_soft_max(
  struct v_ctx* ctx,
  v_tensor* a);

// in-place, returns view(a)
v_API v_tensor* v_soft_max_inplace(
  struct v_ctx* ctx,
  v_tensor* a);

// a    [ne0, ne01, ne02, ne03]
// mask [ne0, ne11, ne12, ne13] | ne11 >= ne01, F16 or F32, optional
//
// broadcast:
//   ne02 % ne12 == 0
//   ne03 % ne13 == 0
//
// fused soft_max(a*scale + mask*(ALiBi slope))
// max_bias = 0.0f for no ALiBi
v_API v_tensor* v_soft_max_ext(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* mask,
  float scale,
  float max_bias);

v_API v_tensor* v_soft_max_ext_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* mask,
  float scale,
  float max_bias);

v_API void v_soft_max_add_sinks(
  v_tensor* a,
  v_tensor* sinks);

v_API v_tensor* v_soft_max_ext_back(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  float scale,
  float max_bias);

// in-place, returns view(a)
v_API v_tensor* v_soft_max_ext_back_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  float scale,
  float max_bias);

// rotary position embedding
// if (mode & 1) - skip n_past elements (NOT SUPPORTED)
// if (mode & v_ROPE_TYPE_NEOX) - GPT-NeoX style
//
// b is an int32 vector with size a->ne[2], it contains the positions
v_API v_tensor* v_rope(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  int n_dims,
  int mode);

// in-place, returns view(a)
v_API v_tensor* v_rope_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  int n_dims,
  int mode);

// custom RoPE
// c is freq factors (e.g. phi3-128k), (optional)
v_API v_tensor* v_rope_ext(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  v_tensor* c,
  int n_dims,
  int mode,
  int n_ctx_orig,
  float freq_base,
  float freq_scale,
  float ext_factor,
  float attn_factor,
  float beta_fast,
  float beta_slow);

v_API v_tensor* v_rope_multi(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  v_tensor* c,
  int n_dims,
  int sections[v_MROPE_SECTIONS],
  int mode,
  int n_ctx_orig,
  float freq_base,
  float freq_scale,
  float ext_factor,
  float attn_factor,
  float beta_fast,
  float beta_slow);

// in-place, returns view(a)
v_API v_tensor* v_rope_ext_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  v_tensor* c,
  int n_dims,
  int mode,
  int n_ctx_orig,
  float freq_base,
  float freq_scale,
  float ext_factor,
  float attn_factor,
  float beta_fast,
  float beta_slow);

v_API v_tensor* v_rope_multi_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  v_tensor* c,
  int n_dims,
  int sections[v_MROPE_SECTIONS],
  int mode,
  int n_ctx_orig,
  float freq_base,
  float freq_scale,
  float ext_factor,
  float attn_factor,
  float beta_fast,
  float beta_slow);

v_DEPRECATED(v_API v_tensor * v_rope_custom(
               struct v_ctx * ctx,
               v_tensor * a,
               v_tensor * b,
               int n_dims,
               int mode,
               int n_ctx_orig,
               float freq_base,
               float freq_scale,
               float ext_factor,
               float attn_factor,
               float beta_fast,
               float beta_slow),
             "use v_rope_ext instead");

v_DEPRECATED(v_API v_tensor * v_rope_custom_inplace(
               struct v_ctx * ctx,
               v_tensor * a,
               v_tensor * b,
               int n_dims,
               int mode,
               int n_ctx_orig,
               float freq_base,
               float freq_scale,
               float ext_factor,
               float attn_factor,
               float beta_fast,
               float beta_slow),
             "use v_rope_ext_inplace instead");

// compute correction dims for YaRN RoPE scaling
v_API void v_rope_yarn_corr_dims(
  int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]);

// rotary position embedding backward, i.e compute dx from dy
// a - dy
v_API v_tensor* v_rope_ext_back(
  struct v_ctx* ctx,
  v_tensor* a, // gradients of v_rope result
  v_tensor* b, // positions
  v_tensor* c, // freq factors
  int n_dims,
  int mode,
  int n_ctx_orig,
  float freq_base,
  float freq_scale,
  float ext_factor,
  float attn_factor,
  float beta_fast,
  float beta_slow);

v_API v_tensor* v_rope_multi_back(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  v_tensor* c,
  int n_dims,
  int sections[4],
  int mode,
  int n_ctx_orig,
  float freq_base,
  float freq_scale,
  float ext_factor,
  float attn_factor,
  float beta_fast,
  float beta_slow);


// clamp
// in-place, returns view(a)
v_API v_tensor* v_clamp(
  struct v_ctx* ctx,
  v_tensor* a,
  float min,
  float max);

// im2col
// converts data into a format that effectively results in a convolution when combined with matrix multiplication
v_API v_tensor* v_im2col(
  struct v_ctx* ctx,
  v_tensor* a, // convolution kernel
  v_tensor* b, // data
  int s0, // stride dimension 0
  int s1, // stride dimension 1
  int p0, // padding dimension 0
  int p1, // padding dimension 1
  int d0, // dilation dimension 0
  int d1, // dilation dimension 1
  bool is_2D,
  enum v_data_type dst_type);

v_API v_tensor* v_im2col_back(
  struct v_ctx* ctx,
  v_tensor* a, // convolution kernel
  v_tensor* b, // gradient of im2col output
  int64_t* ne, // shape of im2col input
  int s0, // stride dimension 0
  int s1, // stride dimension 1
  int p0, // padding dimension 0
  int p1, // padding dimension 1
  int d0, // dilation dimension 0
  int d1, // dilation dimension 1
  bool is_2D);

v_API v_tensor* v_conv_1d(
  struct v_ctx* ctx,
  v_tensor* a, // convolution kernel
  v_tensor* b, // data
  int s0, // stride
  int p0, // padding
  int d0); // dilation

// conv_1d with padding = half
// alias for v_conv_1d(a, b, s, a->ne[0]/2, d)
v_API v_tensor* v_conv_1d_ph(
  struct v_ctx* ctx,
  v_tensor* a, // convolution kernel
  v_tensor* b, // data
  int s, // stride
  int d); // dilation

// depthwise
// TODO: this is very likely wrong for some cases! - needs more testing
v_API v_tensor* v_conv_1d_dw(
  struct v_ctx* ctx,
  v_tensor* a, // convolution kernel
  v_tensor* b, // data
  int s0, // stride
  int p0, // padding
  int d0); // dilation

v_API v_tensor* v_conv_1d_dw_ph(
  struct v_ctx* ctx,
  v_tensor* a, // convolution kernel
  v_tensor* b, // data
  int s0, // stride
  int d0); // dilation

v_API v_tensor* v_conv_transpose_1d(
  struct v_ctx* ctx,
  v_tensor* a, // convolution kernel
  v_tensor* b, // data
  int s0, // stride
  int p0, // padding
  int d0); // dilation
v_tensor* v_log_impl(struct v_ctx* ctx,
                     v_tensor* a,
                     bool inplace);

v_API v_tensor* v_conv_2d(
  struct v_ctx* ctx,
  v_tensor* a, // convolution kernel
  v_tensor* b, // data
  int s0, // stride dimension 0
  int s1, // stride dimension 1
  int p0, // padding dimension 0
  int p1, // padding dimension 1
  int d0, // dilation dimension 0
  int d1); // dilation dimension 1

v_API v_tensor* v_im2col_3d(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  int64_t IC,
  int s0, // stride width
  int s1, // stride height
  int s2, // stride depth
  int p0, // padding width
  int p1, // padding height
  int p2, // padding depth
  int d0, // dilation width
  int d1, // dilation height
  int d2, // dilation depth
  enum v_data_type dst_type);

// a: [OC*IC, KD, KH, KW]
// b: [N*IC, ID, IH, IW]
// result: [N*OC, OD, OH, OW]
v_API v_tensor* v_conv_3d(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  int64_t IC,
  int s0, // stride width
  int s1, // stride height
  int s2, // stride depth
  int p0, // padding width
  int p1, // padding height
  int p2, // padding depth
  int d0, // dilation width
  int d1, // dilation height
  int d2 // dilation depth
);

// kernel size is a->ne[0] x a->ne[1]
// stride is equal to kernel size
// padding is zero
// example:
// a:     16   16    3  768
// b:   1024 1024    3    1
// res:   64   64  768    1
// used in sam
v_API v_tensor* v_conv_2d_sk_p0(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b);

// kernel size is a->ne[0] x a->ne[1]
// stride is 1
// padding is half
// example:
// a:      3    3    256  256
// b:     64   64    256    1
// res:   64   64    256    1
// used in sam
v_API v_tensor* v_conv_2d_s1_ph(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b);

// depthwise (via im2col and mul_mat)
v_API v_tensor* v_conv_2d_dw(
  struct v_ctx* ctx,
  v_tensor* a, // convolution kernel
  v_tensor* b, // data
  int s0, // stride dimension 0
  int s1, // stride dimension 1
  int p0, // padding dimension 0
  int p1, // padding dimension 1
  int d0, // dilation dimension 0
  int d1); // dilation dimension 1

// Depthwise 2D convolution
// may be faster than v_conv_2d_dw, but not available in all backends
// a:   KW    KH    1    C    convolution kernel
// b:   W     H     C    N    input data
// res: W_out H_out C    N
v_API v_tensor* v_conv_2d_dw_direct(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  int stride0,
  int stride1,
  int pad0,
  int pad1,
  int dilation0,
  int dilation1);

v_API v_tensor* v_conv_transpose_2d_p0(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  int stride);

v_API v_tensor* v_conv_2d_direct(
  struct v_ctx* ctx,
  v_tensor* a, // convolution kernel [KW, KH, IC, OC]
  v_tensor* b, // input data [W, H, C, N]
  int s0, // stride dimension 0
  int s1, // stride dimension 1
  int p0, // padding dimension 0
  int p1, // padding dimension 1
  int d0, // dilation dimension 0
  int d1); // dilation dimension 1

v_API v_tensor* v_conv_3d_direct(
  struct v_ctx* ctx,
  v_tensor* a, // kernel [KW, KH, KD, IC * OC]
  v_tensor* b, // input  [W, H, D, C * N]
  int s0, // stride
  int s1,
  int s2,
  int p0, // padding
  int p1,
  int p2,
  int d0, // dilation
  int d1,
  int d2,
  int n_channels,
  int n_batch,
  int n_channels_out);

enum v_op_pool {
  V_OP_POOL_MAX,
  V_OP_POOL_AVG,
  v_OP_POOL_COUNT,
};

v_API v_tensor* v_pool_1d(
  struct v_ctx* ctx,
  v_tensor* a,
  enum v_op_pool op,
  int k0, // kernel size
  int s0, // stride
  int p0); // padding

// the result will have 2*p0 padding for the first dimension
// and 2*p1 padding for the second dimension
v_API v_tensor* v_pool_2d(
  struct v_ctx* ctx,
  v_tensor* a,
  enum v_op_pool op,
  int k0,
  int k1,
  int s0,
  int s1,
  float p0,
  float p1);

v_API v_tensor* v_pool_2d_back(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* af, // "a"/input used in forward pass
  enum v_op_pool op,
  int k0,
  int k1,
  int s0,
  int s1,
  float p0,
  float p1);

enum v_scale_mode {
  v_SCALE_MODE_NEAREST  = 0,
  v_SCALE_MODE_BILINEAR = 1,

  v_SCALE_MODE_COUNT
};

enum v_scale_flag {
  v_SCALE_FLAG_ALIGN_CORNERS = (1 << 8)
};

// interpolate
// multiplies ne0 and ne1 by scale factor
v_API v_tensor* v_upscale(
  struct v_ctx* ctx,
  v_tensor* a,
  int scale_factor,
  enum v_scale_mode mode);

// interpolate
// interpolate scale to specified dimensions
v_DEPRECATED(v_API v_tensor * v_upscale_ext(
               struct v_ctx * ctx,
               v_tensor * a,
               int ne0,
               int ne1,
               int ne2,
               int ne3,
               enum v_scale_mode mode),
             "use v_interpolate instead");

// Up- or downsamples the input to the specified size.
// 2D scale modes (eg. bilinear) are applied to the first two dimensions.
v_API v_tensor* v_interpolate(
  struct v_ctx* ctx,
  v_tensor* a,
  int64_t ne0,
  int64_t ne1,
  int64_t ne2,
  int64_t ne3,
  uint32_t mode); // v_scale_mode [ | v_scale_flag...]

// pad each dimension with zeros: [x, ..., x] -> [x, ..., x, 0, ..., 0]
v_API v_tensor* v_pad(
  struct v_ctx* ctx,
  v_tensor* a,
  int p0,
  int p1,
  int p2,
  int p3);

v_API v_tensor* v_pad_ext(
  struct v_ctx* ctx,
  v_tensor* a,
  int lp0,
  int rp0,
  int lp1,
  int rp1,
  int lp2,
  int rp2,
  int lp3,
  int rp3
);

// pad each dimension with reflection: [a, b, c, d] -> [b, a, b, c, d, c]
v_API v_tensor* v_pad_reflect_1d(
  struct v_ctx* ctx,
  v_tensor* a,
  int p0,
  int p1);

// Move tensor elements by an offset given for each dimension. Elements that
// are shifted beyond the last position are wrapped around to the beginning.
v_API v_tensor* v_roll(
  struct v_ctx* ctx,
  v_tensor* a,
  int shift0,
  int shift1,
  int shift2,
  int shift3);


// Ref: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py#L151
// timesteps: [N,]
// return: [N, dim]
v_API v_tensor* v_timestep_embedding(
  struct v_ctx* ctx,
  v_tensor* timesteps,
  int dim,
  int max_period);

// sort rows
enum v_sort_order {
  v_SORT_ORDER_ASC,
  v_SORT_ORDER_DESC,
};

v_API v_tensor* v_argsort(
  struct v_ctx* ctx,
  v_tensor* a,
  enum v_sort_order order);

v_API v_tensor* v_arange(
  struct v_ctx* ctx,
  float start,
  float stop,
  float step);

// top k elements per row
v_API v_tensor* v_top_k(
  struct v_ctx* ctx,
  v_tensor* a,
  int k);

#define v_KQ_MASK_PAD 64

// q:    [n_embd_k, n_batch,     n_head,    ne3 ]
// k:    [n_embd_k, n_kv,        n_head_kv, ne3 ]
// v:    [n_embd_v, n_kv,        n_head_kv, ne3 ] !! not transposed !!
// mask: [n_kv,     n_batch_pad, ne32,      ne33] !! n_batch_pad = v_PAD(n_batch, v_KQ_MASK_PAD) !!
// res:  [n_embd_v, n_head,      n_batch,   ne3 ] !! permuted !!
//
// broadcast:
//   n_head % n_head_kv == 0
//   n_head % ne32      == 0
//   ne3    % ne33      == 0
//
v_API v_tensor* v_flash_attn_ext(
  struct v_ctx* ctx,
  v_tensor* q,
  v_tensor* k,
  v_tensor* v,
  v_tensor* mask,
  float scale,
  float max_bias,
  float logit_softcap);

v_API void v_flash_attn_ext_set_prec(
  v_tensor* a,
  enum v_prec prec);

v_API enum v_prec v_flash_attn_ext_get_prec(
  const v_tensor* a);

v_API void v_flash_attn_ext_add_sinks(
  v_tensor* a,
  v_tensor* sinks);

// TODO: needs to be adapted to v_flash_attn_ext
v_API v_tensor* v_flash_attn_back(
  struct v_ctx* ctx,
  v_tensor* q,
  v_tensor* k,
  v_tensor* v,
  v_tensor* d,
  bool masked);

v_API v_tensor* v_ssm_conv(
  struct v_ctx* ctx,
  v_tensor* sx,
  v_tensor* c);

v_API v_tensor* v_ssm_scan(
  struct v_ctx* ctx,
  v_tensor* s,
  v_tensor* x,
  v_tensor* dt,
  v_tensor* A,
  v_tensor* B,
  v_tensor* C,
  v_tensor* ids);

// partition into non-overlapping windows with padding if needed
// example:
// a:   768   64   64    1
// w:    14
// res: 768   14   14    25
// used in sam
v_API v_tensor* v_win_part(
  struct v_ctx* ctx,
  v_tensor* a,
  int w);

// reverse of v_win_part
// used in sam
v_API v_tensor* v_win_unpart(
  struct v_ctx* ctx,
  v_tensor* a,
  int w0,
  int h0,
  int w);

v_API v_tensor* v_unary(
  struct v_ctx* ctx,
  v_tensor* a,
  enum v_unary_op op);

v_API v_tensor* v_unary_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  enum v_unary_op op);

// used in sam
v_API v_tensor* v_get_rel_pos(
  struct v_ctx* ctx,
  v_tensor* a,
  int qh,
  int kh);

// used in sam
v_API v_tensor* v_add_rel_pos(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* pw,
  v_tensor* ph);

v_API v_tensor* v_add_rel_pos_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* pw,
  v_tensor* ph);

v_API v_tensor* v_rwkv_wkv6(
  struct v_ctx* ctx,
  v_tensor* k,
  v_tensor* v,
  v_tensor* r,
  v_tensor* tf,
  v_tensor* td,
  v_tensor* state);

v_API v_tensor* v_gated_linear_attn(
  struct v_ctx* ctx,
  v_tensor* k,
  v_tensor* v,
  v_tensor* q,
  v_tensor* g,
  v_tensor* state,
  float scale);

v_API v_tensor* v_rwkv_wkv7(
  struct v_ctx* ctx,
  v_tensor* r,
  v_tensor* w,
  v_tensor* k,
  v_tensor* v,
  v_tensor* a,
  v_tensor* b,
  v_tensor* state);

// custom operators

typedef void (*v_custom1_op_t)(v_tensor* dst, const v_tensor* a, int ith, int nth,
                               void* userdata);
typedef void (*v_custom2_op_t)(v_tensor* dst, const v_tensor* a, const v_tensor* b,
                               int ith, int nth, void* userdata);
typedef void (*v_custom3_op_t)(v_tensor* dst, const v_tensor* a, const v_tensor* b,
                               const v_tensor* c, int ith, int nth, void* userdata);

#define v_N_TASKS_MAX (-1)
// n_tasks == v_N_TASKS_MAX means to use max number of tasks

v_API v_tensor* v_map_custom1(
  struct v_ctx* ctx,
  v_tensor* a,
  v_custom1_op_t fun,
  int n_tasks,
  void* userdata);

v_API v_tensor* v_map_custom1_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  v_custom1_op_t fun,
  int n_tasks,
  void* userdata);

v_API v_tensor* v_map_custom2(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  v_custom2_op_t fun,
  int n_tasks,
  void* userdata);

v_API v_tensor* v_map_custom2_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  v_custom2_op_t fun,
  int n_tasks,
  void* userdata);

v_API v_tensor* v_map_custom3(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  v_tensor* c,
  v_custom3_op_t fun,
  int n_tasks,
  void* userdata);

v_API v_tensor* v_map_custom3_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* b,
  v_tensor* c,
  v_custom3_op_t fun,
  int n_tasks,
  void* userdata);

typedef void (*v_custom_op_t)(v_tensor* dst, int ith, int nth, void* userdata);

v_API v_tensor* v_custom_4d(
  struct v_ctx* ctx,
  enum v_data_type type,
  int64_t ne0,
  int64_t ne1,
  int64_t ne2,
  int64_t ne3,
  v_tensor* * args,
  int n_args,
  v_custom_op_t fun,
  int n_tasks,
  void* userdata);

v_API v_tensor* v_custom_inplace(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* * args,
  int n_args,
  v_custom_op_t fun,
  int n_tasks,
  void* userdata);

// loss function

v_API v_tensor* v_cross_entropy_loss(
  struct v_ctx* ctx,
  v_tensor* a, // logits
  v_tensor* b); // labels

v_API v_tensor* v_cross_entropy_loss_back(
  struct v_ctx* ctx,
  v_tensor* a, // logits
  v_tensor* b, // labels
  v_tensor* c); // gradients of cross_entropy_loss result

// AdamW optimizer step
// Paper: https://arxiv.org/pdf/1711.05101v3.pdf
// PyTorch: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
v_API v_tensor* v_opt_step_adamw(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* grad,
  v_tensor* m,
  v_tensor* v,
  v_tensor* adamw_params); // parameters such as the learning rate

// stochastic gradient descent step (with weight decay)
v_API v_tensor* v_opt_step_sgd(
  struct v_ctx* ctx,
  v_tensor* a,
  v_tensor* grad,
  v_tensor* sgd_params); // alpha, weight decay


//
// automatic differentiation
//

void v_build_foward_expand(struct v_cgraph* cgraph, v_tensor* tensor);
v_API void v_build_backward_expend(
  struct v_ctx* ctx, // context for gradient computation
  struct v_cgraph* cgraph,
  v_tensor* * grad_accs);

// graph allocation in a context
v_API struct v_cgraph* new_graph(struct v_ctx* ctx);
// size = v_DEFAULT_GRAPH_SIZE, grads = false
v_API struct v_cgraph* v_graph_duplicate(struct v_ctx* ctx, struct v_cgraph* cgraph,
                                         bool force_grads);
v_API void v_graph_reset(struct v_cgraph* cgraph);
// set regular grads + optimizer momenta to 0, set loss grad to 1
v_API void v_graph_clear(struct v_cgraph* cgraph);

v_API int v_graph_size(struct v_cgraph* cgraph);
v_API v_tensor* v_graph_node(struct v_cgraph* cgraph, int i);
// if i < 0, returns nodes[n_nodes + i]
v_API v_tensor* * v_graph_nodes(struct v_cgraph* cgraph);
v_API int v_graph_n_nodes(struct v_cgraph* cgraph);

v_API void v_graph_add_node(struct v_cgraph* cgraph, v_tensor* tensor);

v_API size_t graph_overhead(void);
v_API v_tensor* v_graph_get_tensor(const struct v_cgraph* cgraph, const char* name);
v_API v_tensor* v_graph_get_grad(const struct v_cgraph* cgraph, const v_tensor* node);
v_API v_tensor* v_graph_get_grad_acc(const struct v_cgraph* cgraph, const v_tensor* node);

// print info and performance information for the graph
v_API void v_print_graph(const struct v_cgraph* cgraph);

// dump the graph into a file using the dot format
v_API void v_graph_dump_dot(const struct v_cgraph* gb, const struct v_cgraph* gf,
                            const char* filename);

// TODO these functions were sandwiched in the old optimization interface, is there a better place for them?
typedef void (*v_log_callback)(enum v_log_level level, const char* text, void* user_data);

// Set callback for all future logging events.
// If this is not called, or NULL is supplied, everything is output on stderr.
v_API void set_log(v_log_callback log_callback, void* user_data);
v_API v_tensor* v_set_zero(v_tensor* tensor);


// quantization
//
// - v_quantize_init can be called multiple times with the same type
//   it will only initialize the quantization tables for the first call or after v_quantize_free
//   automatically called by v_quantize_chunk for convenience
//
// - v_quantize_free will free any memory allocated by v_quantize_init
//   call this at the end of the program to avoid memory leaks
//
// note: these are thread-safe
//
v_API void v_quantize_init(enum v_data_type type);
v_API void v_quantize_free(void);

// some quantization type cannot be used without an importance matrix
v_API bool v_quantize_requires_imatrix(enum v_data_type type);

// calls v_quantize_init internally (i.e. can allocate memory)
v_API size_t v_quantize_chunk(
  enum v_data_type type,
  const float* src,
  void* dst,
  int64_t start,
  int64_t nrows,
  int64_t n_per_row,
  const float* imatrix);


static const size_t MML_OBJECT_SIZE = sizeof(struct v_object);
v_tensor* mmlSubImpl(struct v_ctx* ctx,
                     v_tensor* a,
                     v_tensor* b,
                     bool inplace);

v_tensor* v_acc_imple(struct v_ctx* ctx,
                      v_tensor* a,
                      v_tensor* b,
                      size_t nb1,
                      size_t nb2,
                      size_t nb3,
                      size_t offset,
                      bool inplace);
v_tensor* v_cos_impl(struct v_ctx* ctx,
                     v_tensor* a,
                     bool inplace);

v_tensor* v_sin_impl(struct v_ctx* ctx,
                     v_tensor* a,
                     bool inplace);
void v_compute_backward(struct v_ctx* ctx,
                        struct v_cgraph* cgraph,
                        int i,
                        const bool* grads_needed);
v_tensor* v_mul_impl(struct v_ctx* ctx,
                     v_tensor* a,
                     v_tensor* b,
                     bool inplace);

v_tensor* add_cast_impl(struct v_ctx* ctx,
                        v_tensor* a,
                        v_tensor* b,
                        enum v_data_type type);
v_tensor* add1_impl(struct v_ctx* ctx,
                    v_tensor* a,
                    v_tensor* b,
                    bool inplace);

v_tensor* v_div_impl(struct v_ctx* ctx,
                     v_tensor* a,
                     v_tensor* b,
                     bool inplace);
v_tensor* v_add_imple(struct v_ctx* ctx,
                      v_tensor* a,
                      v_tensor* b,
                      bool inplace);
v_tensor* v_map_custom3_impl(struct v_ctx* ctx,
                             v_tensor* a,
                             v_tensor* b,
                             v_tensor* c,
                             const v_custom3_op_t fun,
                             int n_tasks,
                             void* userdata,
                             bool inplace);
v_tensor* v_scale_impl(struct v_ctx* ctx,
                       v_tensor* a,
                       float s,
                       float b,
                       bool inplace);
v_tensor* v_diag(struct v_ctx* ctx,
                 v_tensor* a);
v_tensor* v_diag_mask_zero_impl(struct v_ctx* ctx,
                                v_tensor* a,
                                int n_past,
                                bool inplace);

v_API const struct v_type_traits* v_get_type_traits(enum v_data_type type);

// ggml threadpool
// TODO: currently, only a few functions are in the base ggml API, while the rest are in the CPU backend
// the goal should be to create an API that other backends can use move everything to the ggml base
// scheduling priorities
enum v_sched_priority {
  v_SCHED_PRIO_LOW = -1,
  v_SCHED_PRIO_NORMAL,
  v_SCHED_PRIO_MEDIUM,
  v_SCHED_PRIO_HIGH,
  v_SCHED_PRIO_REALTIME
};

v_tensor* v_unary_impl(struct v_ctx* ctx,
                       v_tensor* a,
                       enum v_unary_op op,
                       bool inplace);
v_tensor* v_map_custom1_impl(struct v_ctx* ctx,
                             v_tensor* a,
                             const v_custom1_op_t fun,
                             int n_tasks,
                             void* userdata,
                             bool inplace);
v_tensor* v_map_custom2_impl(struct v_ctx* ctx,
                             v_tensor* a,
                             v_tensor* b,
                             const v_custom2_op_t fun,
                             int n_tasks,
                             void* userdata,
                             bool inplace);
v_tensor* v_sqr_impl(struct v_ctx* ctx,
                     v_tensor* a,
                     bool inplace);
v_tensor* v_sqrt_impl(struct v_ctx* ctx,
                      v_tensor* a,
                      bool inplace);

v_tensor* v_interpolate_impl(struct v_ctx* ctx,
                             v_tensor* a,
                             int64_t ne0,
                             int64_t ne1,
                             int64_t ne2,
                             int64_t ne3,
                             uint32_t mode);
