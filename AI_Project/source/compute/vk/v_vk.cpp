#include "vk_common.h"
#include "vk_pipeline.hpp"
#include "vk_queue.hpp"
#include "vk_device.hpp"
#include "vk_constant.h"
#include "vk_buffer.h"
#include "vk_context.h"
#include "vk_util.hpp"
#include "v_util.hpp"
#include "vk_op_f32.hpp"
#include "vk_comp.hpp"


//#define v_VULKAN_MEMORY_DEBUG
#ifdef v_VULKAN_MEMORY_DEBUG
class vk_memory_logger;
#endif
class vk_perf_logger;
#ifdef v_VULKAN_MEMORY_DEBUG
std::mutex log_mutex;

void vk_memory_logger::log_allocation(vk_buffer_ref buf_ref, size_t size) {
  std::lock_guard<std::mutex> guard(log_mutex);
  vk_buffer buf          = buf_ref.lock();
  const bool device      = bool(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eDeviceLocal);
  const std::string type = device
                             ? "device"
                             : "host";
  allocations[buf->buffer] = size;
  total_device += device
                    ? size
                    : 0;
  total_host += device
                  ? 0
                  : size;
  VK_LOG_MEMORY(
    buf->device->name << ": +" << format_size(size) << " " << type << " at " << buf->buffer << ". Total device: " <<
    format_size(total_device) << ", total host: " << format_size(total_host));
}

void vk_memory_logger::log_deallocation(vk_buffer_ref buf_ref) {
  if (buf_ref.expired() || buf_ref.lock()->size == 0) { return; }

  std::lock_guard<std::mutex> guard(log_mutex);
  vk_buffer buf     = buf_ref.lock();
  const bool device = bool(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eDeviceLocal);
  std::string type  = device
                       ? "device"
                       : "host";
  auto it = allocations.find(buf->buffer);
  total_device -= device
                    ? it->second
                    : 0;
  total_host -= device
                  ? 0
                  : it->second;
  if (it != allocations.end()) {
    VK_LOG_MEMORY(
      buf->device->name << ": -" << format_size(it->second) << " " << type << " at " << buf->buffer <<
      ". Total device: " << format_size(total_device) << ", total host: " << format_size(total_host));
    allocations.erase(it);
  }
  else {
    VK_LOG_MEMORY(
      "ERROR " << buf->device->name << ": Attempted to deallocate unknown " << type << " memory at " << buf->buffer);
  }
}
#endif // v_VULKAN_MEMORY_DEBUG
//#define v_VULKAN_CHECK_RESULTS
#ifdef v_VULKAN_CHECK_RESULTS
size_t vk_skip_checks;
size_t vk_output_tensor;

void vk_print_tensor(const v_tensor* tensor, const char* name);
void vk_check_results_0(vk_backend_ctx* ctx, v_cgraph* cgraph, int tensor_idx);
void vk_check_results_1(vk_backend_ctx* ctx, v_cgraph* cgraph, int tensor_idx);
#endif


void vk_backend_free(v_backend_t backend);


// variables to track number of compiles in progress


// backend interface

#define UNUSED V_UNUSED


#ifndef MYPROJECT_MML_VK_TEST_H
#define MYPROJECT_MML_VK_TEST_H
#define v_VULKAN_CHECK_RESULT
#include "v_vk.hpp"
#include "vk_device.hpp"
#include "vk_buffer.h"
#ifdef v_VULKAN_RUN_TESTS
static void v_vk_print_matrix_area(const void* data, v_data_type type, int ne0, int ne1, int i0, int i1, int i2) {
  if (type != v_TYPE_F32 && type != v_TYPE_F16) { return; }
  i0 = std::max(i0, 5);
  i1 = std::max(i1, 5);
  i2 = std::max(i2, 0);
  fprintf(stderr, "         ");
  for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) { fprintf(stderr, "%7d ", idx1); }
  fprintf(stderr, "\n");
  for (int idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
    fprintf(stderr, "%7d: ", idx0);
    for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
      if (idx0 >= 0 && idx0 < ne0 && idx1 >= 0 && idx1 < ne1) {
        float val;
        if (type == v_TYPE_F32) { val = *((const float*)data + i2 * ne1 * ne0 + idx1 * ne0 + idx0); }
        else if (type == v_TYPE_F16) { val = v_fp16_to_fp32(*((const v_fp16_t*)data + i2 * ne1 * ne0 + idx1 * ne0 + idx0)); }
        else { v_ABORT("fatal error"); }
        fprintf(stderr, "% 7.2f ", val);
      }
      else { fprintf(stderr, "        "); }
    }
    fprintf(stderr, "\n");
  }
}

template <typename X_TYPE, typename Y_TYPE>
static void v_vk_test_matmul(vk_backend_ctx* ctx, size_t m, size_t n, size_t k, size_t batch, size_t num_it,
                             int split_k, int shader_size) {
  VK_LOG_DEBUG(
    "v_vk_test_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << num_it << ", " << split_k << ", "
    << shader_size << ")");
  const size_t x_ne = m * k * batch;
  const size_t y_ne = k * n * batch;
  const size_t d_ne = m * n * batch;

  vk_pipeline p;
  std::string shname;
  if (shader_size == 0) {
    if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f32->a_s;
      shname = "F32_ALIGNED_S";
    }
    else if (std::is_same<float, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f32_f16->a_s;
      shname = "F32_F16_ALIGNED_S";
    }
    else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f16_f32.f32acc->a_s;
      shname = "F16_F32_ALIGNED_S";
    }
    else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f16.f32acc->a_s;
      shname = "F16_ALIGNED_S";
    }
    else { v_ABORT("fatal error"); }
  }
  else if (shader_size == 1) {
    if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f32->a_m;
      shname = "F32_ALIGNED_M";
    }
    else if (std::is_same<float, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f32_f16->a_m;
      shname = "F32_F16_ALIGNED_M";
    }
    else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f16_f32.f32acc->a_m;
      shname = "F16_F32_ALIGNED_M";
    }
    else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f16.f32acc->a_m;
      shname = "F16_ALIGNED_M";
    }
    else { v_ABORT("fatal error"); }
  }
  else if (shader_size == 2) {
    if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f32->a_l;
      shname = "F32_ALIGNED_L";
    }
    else if (std::is_same<float, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f32_f16->a_l;
      shname = "F32_F16_ALIGNED_L";
    }
    else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f16_f32.f32acc->a_l;
      shname = "F16_F32_ALIGNED_L";
    }
    else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
      p      = ctx->device->pipeline_matmul_f16.f32acc->a_l;
      shname = "F16_ALIGNED_L";
    }
    else { v_ABORT("fatal error"); }
  }
  else { V_ASSERT(0); }

  const size_t kpad = vk_align_size(k, p->align);

  if (k != kpad) {
    if (shader_size == 0) {
      if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f32->s;
        shname = "F32_S";
      }
      else if (std::is_same<float, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f32_f16->s;
        shname = "F32_F16_S";
      }
      else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f16_f32.f32acc->s;
        shname = "F16_F32_S";
      }
      else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f16.f32acc->s;
        shname = "F16_S";
      }
    }
    else if (shader_size == 1) {
      if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f32->m;
        shname = "F32_M";
      }
      else if (std::is_same<float, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f32_f16->m;
        shname = "F32_F16_M";
      }
      else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f16_f32.f32acc->m;
        shname = "F16_F32_M";
      }
      else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f16.f32acc->m;
        shname = "F16_M";
      }
    }
    else if (shader_size == 2) {
      if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f32->l;
        shname = "F32_L";
      }
      else if (std::is_same<float, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f32_f16->l;
        shname = "F32_F16_L";
      }
      else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f16_f32.f32acc->l;
        shname = "F16_F32_L";
      }
      else if (std::is_same<v_fp16_t, X_TYPE>() && std::is_same<v_fp16_t, Y_TYPE>()) {
        p      = ctx->device->pipeline_matmul_f16.f32acc->l;
        shname = "F16_L";
      }
    }
  }

  v_pipeline_request_descriptor_sets(ctx, p, num_it);
  if (split_k > 1) {
    v_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_split_k_reduce, num_it);

    if (ctx->prealloc_split_k == nullptr || ctx->prealloc_split_k->size < sizeof(float) * d_ne * split_k) {
      // Resize buffer
      if (ctx->prealloc_split_k != nullptr) { vk_destroy_buffer(ctx->prealloc_split_k); }
      ctx->prealloc_split_k = vk_create_buffer_check(ctx->device,
                                                     sizeof(float) * d_ne * split_k,
                                                     {vk::MemoryPropertyFlagBits::eDeviceLocal});
    }
  }

  if (ctx->device->need_compiles) { vk_load_shaders(ctx->device); }

  vk_pipeline_allocate_descriptor_sets(ctx);

  vk_buffer d_X = vk_create_buffer_check(ctx->device,
                                         sizeof(X_TYPE) * x_ne,
                                         {vk::MemoryPropertyFlagBits::eDeviceLocal});
  vk_buffer d_Y = vk_create_buffer_check(ctx->device,
                                         sizeof(Y_TYPE) * y_ne,
                                         {vk::MemoryPropertyFlagBits::eDeviceLocal});
  vk_buffer d_D = vk_create_buffer_check(ctx->device,
                                         sizeof(float) * d_ne,
                                         {vk::MemoryPropertyFlagBits::eDeviceLocal});

  X_TYPE* x = (X_TYPE*)malloc(sizeof(X_TYPE) * x_ne);
  Y_TYPE* y = (Y_TYPE*)malloc(sizeof(Y_TYPE) * y_ne);
  float* d  = (float*)malloc(sizeof(float) * d_ne);

  for (size_t i = 0; i < x_ne; i++) {
    if (std::is_same<float, X_TYPE>()) {
      x[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
      // x[i] = 1.0f;
      // x[i] = i + 1;
      // x[i] = (i % k == i / k) ? 1.0f : 0.0f;
    }
    else if (std::is_same<v_fp16_t, X_TYPE>()) {
      x[i] = v_fp32_to_fp16((rand() / (float)RAND_MAX) * 2.0f - 1.0f);
      // x[i] = v_fp32_to_fp16(1.0f);
      // x[i] = v_fp32_to_fp16(i + 1);
      // x[i] = v_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
    }
    else { v_ABORT("fatal error"); }
  }
  for (size_t i = 0; i < y_ne; i++) {
    if (std::is_same<float, Y_TYPE>()) {
      y[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
      // y[i] = (i % k == i / k) ? 1.0f : 0.0f;
      // y[i] = i + 1;
    }
    else if (std::is_same<v_fp16_t, Y_TYPE>()) {
      y[i] = v_fp32_to_fp16((rand() / (float)RAND_MAX) * 2.0f - 1.0f);
      // y[i] = v_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
      // y[i] = v_fp32_to_fp16(i + 1);
    }
    else { v_ABORT("fatal error"); }
  }

  v_vk_buffer_write(d_X, 0, x, sizeof(X_TYPE) * k * m * batch);
  v_vk_buffer_write(d_Y, 0, y, sizeof(Y_TYPE) * k * n * batch);

  vk_context subctx = vk_create_context(ctx, ctx->compute_cmd_pool);
  vk_begin_ctx(ctx->device, subctx);
  for (size_t i = 0; i < num_it; i++) {
    v_vk_matmul(
      ctx,
      subctx,
      p,
      v_vk_subbuffer(ctx, d_X),
      v_vk_subbuffer(ctx, d_Y),
      v_vk_subbuffer(ctx, d_D),
      v_vk_subbuffer(ctx, ctx->prealloc_split_k),
      m,
      n,
      k,
      k,
      k,
      m,
      k * m,
      k * n,
      m * n,
      split_k,
      batch,
      batch,
      batch,
      1,
      1,
      n
    );
  }
  vk_ctx_end(subctx);
  v_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);
  v_vk_command_pool_cleanup(ctx->device, ctx->transfer_cmd_pool);
  vk_destroy_buffer(d_X);
  vk_destroy_buffer(d_Y);
  vk_destroy_buffer(d_D);

  free(x);
  free(y);
  free(d);
}

static void v_vk_print_tensor_area(const v_tensor* tensor, int i0, int i1, int i2, int i3) {
  if (tensor->type != v_TYPE_F32 && tensor->type != v_TYPE_F16) { return; }
  i0 = std::max(i0, 5);
  i1 = std::max(i1, 5);
  i2 = std::max(i2, 0);
  i3 = std::max(i3, 0);
  fprintf(stderr, "         ");
  for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) { fprintf(stderr, "%7d ", idx1); }
  fprintf(stderr, "\n");
  for (int idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
    fprintf(stderr, "%7d: ", idx0);
    for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
      if (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3
        >= 0 && i3 < tensor->ne[3]) {
        float val;
        if (tensor->type == v_TYPE_F32) {
          val = *(float*)((char*)tensor->data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + idx1 * tensor->nb[1] + idx0 *
            tensor->nb[0]);
        }
        else if (tensor->type == v_TYPE_F16) {
          val = v_fp16_to_fp32(
            *(v_fp16_t*)((char*)tensor->data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + idx1 * tensor->nb[1] + idx0
              * tensor->nb[0]));
        }
        else { v_ABORT("fatal error"); }
        fprintf(stderr, "% 7.2f ", val);
      }
      else { fprintf(stderr, "        "); }
    }
    fprintf(stderr, "\n");
  }
}

static void v_vk_quantize_data(const float* from, void* to, size_t ne, v_data_type quant) { v_quantize_chunk(quant, from, to, 0, 1, ne, nullptr); }

static void v_vk_dequantize_data(const void* from, float* to, size_t ne, v_data_type quant) {
  if (quant == v_TYPE_F32) {
    memcpy(to, from, sizeof(float) * ne);
    return;
  }

  const auto* tt = v_get_type_traits(quant);

  v_to_float_t dequant_fn = tt->to_float;

  dequant_fn(from, to, ne);
}

static void v_vk_test_dequant(vk_backend_ctx* ctx, size_t ne, v_data_type quant) {
  VK_LOG_DEBUG("v_vk_test_dequant(" << ne << ")");
  const size_t x_sz     = sizeof(float) * ne;
  const size_t x_sz_f16 = sizeof(v_fp16_t) * ne;
  const size_t qx_sz    = ne * v_type_size(quant) / block_size(quant);
  float* x              = (float*)malloc(x_sz);
  void* qx              = malloc(qx_sz);
  vk_buffer qx_buf      = vk_create_buffer_check(ctx->device, qx_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
  vk_buffer x_buf       = vk_create_buffer_check(ctx->device, x_sz_f16, {vk::MemoryPropertyFlagBits::eDeviceLocal});
  float* x_ref          = (float*)malloc(x_sz);
  v_fp16_t* x_chk       = (v_fp16_t*)malloc(x_sz_f16);

  for (size_t i = 0; i < ne; i++) { x[i] = rand() / (float)RAND_MAX; }

  vk_pipeline p = v_vk_get_to_fp16(ctx, quant);

  v_vk_quantize_data(x, qx, ne, quant);
  v_vk_dequantize_data(qx, x_ref, ne, quant);

  v_pipeline_request_descriptor_sets(ctx, p, 1);

  if (ctx->device->need_compiles) { vk_load_shaders(ctx->device); }

  vk_pipeline_allocate_descriptor_sets(ctx);

  v_vk_buffer_write(qx_buf, 0, qx, qx_sz);

  vk_context subctx = vk_create_context(ctx, ctx->compute_cmd_pool);
  //v_vk_ctx_begin(ctx->device, subctx);
  const std::vector<uint32_t> pc = {1, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne};
  v_vk_dispatch_pipeline(ctx,
                         subctx,
                         p,
                         {vk_sub_buffer{qx_buf, 0, qx_sz}, vk_sub_buffer{x_buf, 0, x_sz_f16}},
                         pc,
                         {(uint32_t)ne, 1, 1});
  //v_vk_ctx_end(subctx);
  //
  //auto begin = std::chrono::high_resolution_clock::now();
  //
  //v_vk_submit(subctx, ctx->fence);
  //VK_CHECK(ctx->device->device.waitForFences({ctx->fence}, true, UINT64_MAX), "v_vk_test_dequant waitForFences");
  //ctx->device->device.resetFences({ctx->fence});
  //v_vk_queue_command_pools_cleanup(ctx->device);
  //
  //auto end = std::chrono::high_resolution_clock::now();
  //
  //double ms_dequant = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0;
  //mmlVKBufferRead(x_buf, 0, x_chk, x_sz_f16);
  //
  //int first_err = -1;
  //
  //double avg_err = 0.0;
  //for (size_t i = 0; i < ne; i++)
  //{
  //  double error = std::fabs(x_ref[i] - v_fp16_to_fp32(x_chk[i]));
  //  avg_err += error;
  //
  //  if (first_err < 0 && error > 0.05)
  //  {
  //    first_err = i;
  //  }
  //}
  //
  //avg_err /= ne;
  //
  //std::cerr << "TEST DEQUANT " << v_type_name(quant) << " time=" << ms_dequant << "ms avg_err=" << avg_err <<
  //  std::endl;
  //
  //if (avg_err > 0.1)
  //{
  //  std::cerr << "first_error = " << first_err << std::endl;
  //  std::cerr << "Actual result: " << std::endl << std::endl;
  //  for (int i = std::max(0, first_err - 5); i < std::min((int)ne, first_err + 5); i++)
  //  {
  //    std::cerr << v_fp16_to_fp32(x_chk[i]) << ", ";
  //  }
  //  std::cerr << std::endl << "Expected result: " << std::endl << std::endl;
  //  for (int i = std::max(0, first_err - 5); i < std::min((int)ne, first_err + 5); i++)
  //  {
  //    std::cerr << x_ref[i] << ", ";
  //  }
  //  std::cerr << std::endl;
  //}
  //
  //mmlVKDestroyBuffer(x_buf);
  //mmlVKDestroyBuffer(qx_buf);
  //
  //free(x);
  //free(qx);
  //free(x_ref);
  //free(x_chk);
}

#endif
// checks
//#define v_VULKAN_CHECK_RESULTS
//#include <iostream>
#ifdef v_VULKAN_CHECK_RESULTS
void v_vk_print_graph_origin(const v_tensor* tensor, std::vector<const v_tensor*>& done, int level = 0) {
  if (std::find(done.begin(), done.end(), tensor) != done.end() || level > 10) { return; }
  for (int j = 0; j < level; j++) { std::cerr << " "; }
  std::cerr << v_op_name(tensor->op) << " gpu=" << (tensor->extra != nullptr) << std::endl;
  done.push_back(tensor);
  for (int i = 0; i < v_MAX_SRC; i++) { if (tensor->src[i] != nullptr) { v_vk_print_graph_origin(tensor->src[i], done, level + 1); } }
}

void v_vk_print_tensor_area(const v_tensor* tensor, const void* data, int i0, int i1, int i2, int i3) {
  if (tensor->type != v_TYPE_F32 && tensor->type != v_TYPE_F16 && tensor->type != v_TYPE_I32) { return; }
  i0 = std::max(i0, 5);
  i1 = std::max(i1, 5);
  i2 = std::max(i2, 0);
  i3 = std::max(i3, 0);
  fprintf(stderr, "         ");
  for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) { fprintf(stderr, "%7d ", idx1); }
  fprintf(stderr, "\n");
  for (int idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
    fprintf(stderr, "%7d: ", idx0);
    for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
      if (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3
        >= 0 && i3 < tensor->ne[3]) {
        float val;
        if (tensor->type == v_TYPE_F32) {
          val = *(const float*)((const char*)data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + idx1 * tensor->nb[1] +
            idx0 * tensor->nb[0]);
        }
        else if (tensor->type == v_TYPE_F16) {
          val = v_fp16_to_fp32(
            *(const v_fp16_t*)((const char*)data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + idx1 * tensor->nb[1] +
              idx0 * tensor->nb[0]));
        }
        else if (tensor->type == v_TYPE_I32) {
          val = *(const int32_t*)((const char*)data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + idx1 * tensor->nb[1] +
            idx0 * tensor->nb[0]);
        }
        else { v_ABORT("fatal error"); }
        fprintf(stderr, "% 7.7f ", val);
      }
      else { fprintf(stderr, "        "); }
    }
    fprintf(stderr, "\n");
  }
}

void vk_print_tensor(const v_tensor* tensor, const char* name) {
  void* tensor_data = tensor->data;
  const bool is_gpu = tensor->buffer != nullptr && v_backend_buffer_is_vk(tensor->buffer);
  if (is_gpu) {
    const size_t tensor_size         = num_bytes(tensor);
    tensor_data                      = malloc(tensor_size);
    v_backend_vk_buffer_ctx* buf_ctx = (v_backend_vk_buffer_ctx*)tensor->buffer->context;
    vk_buffer buffer_gpu             = buf_ctx->dev_buffer;
    vk_read_buffer(buffer_gpu, vk_tensor_offset(tensor) + tensor->view_offs, tensor_data, tensor_size);
  }

  std::cerr << "TENSOR CHECK " << name << " (" << tensor->name << "): " << v_op_name(tensor->op) << std::endl;
  std::cerr << "tensor=" << tensor << " tensor->type: " << v_type_name(tensor->type) << " ne0=" << tensor->ne[0] <<
    " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] <<
    " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << std::endl;
  if (tensor->src[0] != nullptr) {
    std::cerr << "tensor->src[0]=" << tensor->src[0] << " name=" << tensor->src[0]->name << " op=" <<
      v_op_name(tensor->src[0]->op) << " type=" << v_type_name(tensor->src[0]->type) << " ne0=" << tensor->src[0]
      ->ne[0] << " nb0=" << tensor->src[0]->nb[0] << " ne1=" << tensor->src[0]->ne[1] << " nb1=" << tensor->src[0]->nb[
        1] << " ne2=" << tensor->src[0]->ne[2] << " nb2=" << tensor->src[0]->nb[2] << " ne3=" << tensor->src[0]->ne[3]
      << " nb3=" << tensor->src[0]->nb[3] << std::endl;
  }
  if (tensor->src[1] != nullptr) {
    std::cerr << "tensor->src[1]=" << tensor->src[1] << " name=" << tensor->src[1]->name << " op=" <<
      v_op_name(tensor->src[1]->op) << " type=" << v_type_name(tensor->src[1]->type) << " ne0=" << tensor->src[1]
      ->ne[0] << " nb0=" << tensor->src[1]->nb[0] << " ne1=" << tensor->src[1]->ne[1] << " nb1=" << tensor->src[1]->nb[
        1] << " ne2=" << tensor->src[1]->ne[2] << " nb2=" << tensor->src[1]->nb[2] << " ne3=" << tensor->src[1]->ne[3]
      << " nb3=" << tensor->src[1]->nb[3] << std::endl;
  }
  std::cerr << std::endl << "Result:" << std::endl;
  v_vk_print_tensor_area(tensor, tensor_data, 5, 5, 0, 0);
  std::cerr << std::endl;
  std::vector<const v_tensor*> done;
  v_vk_print_graph_origin(tensor, done);
  if (is_gpu) { free(tensor_data); }
}

void* comp_result;
size_t comp_size;
size_t comp_nb[4];
size_t check_counter = 0;

void vk_check_results_0(vk_backend_ctx* ctx, v_cgraph* cgraph, int tensor_idx) {
  v_tensor* tensor = cgraph->nodes[tensor_idx];
  if (tensor->op == v_OP_TRANSPOSE || tensor->op == v_OP_SET_ROWS) { return; }

  bool fused_rms_norm_mul = false;
  int rms_norm_idx        = -1;
  if (ctx->num_additional_fused_ops == 1 &&
    tensor->op == v_OP_RMS_NORM &&
    cgraph->nodes[tensor_idx + 1]->op == v_OP_MUL) {
    fused_rms_norm_mul = true;
    tensor             = cgraph->nodes[tensor_idx + 1];
  }

  check_counter++;
  if (!(vk_output_tensor > 0 && vk_output_tensor == check_counter) && check_counter <= vk_skip_checks) { return; }

  VK_LOG_DEBUG("v_vk_check_results_0(" << tensor->name << ")");

  v_tensor* src0 = tensor->src[0];
  v_tensor* src1 = tensor->src[1];

  struct v_init_param iparams = {
    /*.mem_size   =*/ 2ul * 1024ul * 1024ul * 1024ul,
    /*.mem_buffer =*/ NULL,
    /*.no_alloc   =*/ false,
  };

  struct v_ctx* v_ctx = v_ctx_init(iparams);

  std::array<struct v_tensor*, v_MAX_SRC> src_clone = {
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr
  };
  std::array<size_t, v_MAX_SRC> src_size  = {};
  std::array<void*, v_MAX_SRC> src_buffer = {};
  const char* srci_name[v_MAX_SRC]        = {
    "src0", "src1", "src2", "src3", "src4", "src5", "src6", "src7", "src8", "src9"
  };

  struct v_tensor* tensor_clone = nullptr;

  for (int i = 0; i < v_MAX_SRC; i++) {
    v_tensor* srci = tensor->src[i];
    if (fused_rms_norm_mul) {
      rms_norm_idx = tensor->src[0]->op == v_OP_RMS_NORM
                       ? 0
                       : 1;
      v_tensor* rms_norm = tensor->src[rms_norm_idx];
      switch (i) {
        case 0: srci = rms_norm->src[0];
          break;
        case 1: srci = tensor->src[1 - rms_norm_idx];
          break;
        default: continue;
      }
    }
    if (srci == nullptr) { continue; }
    v_tensor* srci_clone = v_dup_tensor(v_ctx, srci);
    size_t srci_size     = num_bytes(srci);

    src_clone[i]  = srci_clone;
    src_size[i]   = num_bytes(srci);
    src_buffer[i] = malloc(srci_size);

    srci_clone->data = src_buffer[i];
    if (v_backend_buffer_is_host(srci->buffer)) {
      memcpy(srci_clone->data, srci->data, srci_size);
      memcpy(srci_clone->nb, srci->nb, sizeof(size_t) * 4);
    }
    else if (v_backend_buffer_is_vk(srci->buffer)) {
      v_backend_vk_buffer_ctx* buf_ctx = (v_backend_vk_buffer_ctx*)srci->buffer->context;
      vk_buffer& buffer_gpu            = buf_ctx->dev_buffer;
      uint64_t offset                  = vk_tensor_offset(srci) + srci->view_offs;
      if (!v_is_contiguous(srci) && v_vk_dim01_contiguous(srci)) {
        for (int i3 = 0; i3 < srci->ne[3]; i3++) {
          for (int i2 = 0; i2 < srci->ne[2]; i2++) {
            const int idx = i3 * srci->ne[2] + i2;
            vk_read_buffer(buffer_gpu,
                           offset + idx * srci->nb[2],
                           ((char*)srci_clone->data + idx * srci_clone->nb[2]),
                           srci->ne[1] * srci->nb[1]);
          }
        }

        srci_clone->nb[0] = srci->nb[0];
        srci_clone->nb[1] = srci->nb[1];
        for (int i = 2; i < 4; i++) { srci_clone->nb[i] = srci_clone->nb[i - 1] * srci_clone->ne[i - 1]; }
      }
      else {
        if (offset + srci_size >= buffer_gpu->size) { srci_size = buffer_gpu->size - offset; }
        vk_read_buffer(buffer_gpu, offset, srci_clone->data, srci_size);
        memcpy(srci_clone->nb, srci->nb, sizeof(size_t) * 4);
      }
    }
    else { v_ABORT("fatal error"); }

    if (vk_output_tensor > 0 && vk_output_tensor == check_counter) { vk_print_tensor(srci, srci_name[i]); }
  }

  if (tensor->op == v_OP_FLASH_ATTN_EXT) {
    const float* params = (const float*)tensor->op_params;
    tensor_clone        = v_flash_attn_ext(v_ctx,
                                    src_clone[0],
                                    src_clone[1],
                                    src_clone[2],
                                    src_clone[3],
                                    params[0],
                                    params[1],
                                    params[2]);
    if (src_clone[4]) { v_flash_attn_ext_add_sinks(tensor_clone, src_clone[4]); }
  }
  else if (tensor->op == V_OP_MUL_MAT) { tensor_clone = v_matmul(v_ctx, src_clone[0], src_clone[1]); }
  else if (tensor->op == v_OP_MUL_MAT_ID) { tensor_clone = v_mat_mul_id(v_ctx, src_clone[0], src_clone[1], src_clone[2]); }
  else if (tensor->op == v_OP_SUB) { tensor_clone = v_sub(v_ctx, src_clone[0], src_clone[1]); }
  else if (tensor->op == v_OP_MUL) {
    if (fused_rms_norm_mul) {
      tensor_clone = v_rms_norm(v_ctx, src_clone[0], *(float*)tensor->src[rms_norm_idx]->op_params);
      tensor_clone = v_mul(v_ctx, tensor_clone, src_clone[1 - rms_norm_idx]);
    }
    else { tensor_clone = v_mul(v_ctx, src_clone[0], src_clone[1]); }
  }
  else if (tensor->op == v_OP_DIV) { tensor_clone = v_div(v_ctx, src_clone[0], src_clone[1]); }
  else if (tensor->op == v_OP_CONCAT) { tensor_clone = v_concat(v_ctx, src_clone[0], src_clone[1], *(int*)tensor->op_params); }
  else if (tensor->op == v_OP_UPSCALE) {
    tensor_clone = v_interpolate(v_ctx,
                                 src_clone[0],
                                 tensor->ne[0],
                                 tensor->ne[1],
                                 tensor->ne[2],
                                 tensor->ne[3],
                                 (v_scale_mode)tensor->op_params[0]);
  }
  else if (tensor->op == V_OP_SCALE) {
    const float* params = (const float*)tensor->op_params;
    tensor_clone        = v_scale_bias(v_ctx, src_clone[0], params[0], params[1]);
  }
  else if (tensor->op == V_OP_SQR) { tensor_clone = v_sqr(v_ctx, src_clone[0]); }
  else if (tensor->op == v_OP_SQRT) { tensor_clone = v_sqrt(v_ctx, src_clone[0]); }
  else if (tensor->op == V_OP_SIN) { tensor_clone = v_sin(v_ctx, src_clone[0]); }
  else if (tensor->op == V_OP_COS) { tensor_clone = v_cos(v_ctx, src_clone[0]); }
  else if (tensor->op == V_OP_CLAMP) {
    const float* params = (const float*)tensor->op_params;
    tensor_clone        = v_clamp(v_ctx, src_clone[0], params[0], params[1]);
  }
  else if (tensor->op == v_OP_PAD) {
    tensor_clone = v_pad_ext(v_ctx,
                             src_clone[0],
                             tensor->op_params[0],
                             tensor->op_params[1],
                             tensor->op_params[2],
                             tensor->op_params[3],
                             tensor->op_params[4],
                             tensor->op_params[5],
                             tensor->op_params[6],
                             tensor->op_params[7]);
  }
  else if (tensor->op == v_OP_REPEAT) { tensor_clone = v_repeat(v_ctx, src_clone[0], tensor); }
  else if (tensor->op == v_OP_REPEAT_BACK) { tensor_clone = v_repeat_back(v_ctx, src_clone[0], tensor); }
  else if (tensor->op == v_OP_ADD) { tensor_clone = v_add(v_ctx, src_clone[0], src_clone[1]); }
  else if (tensor->op == v_OP_ACC) {
    tensor_clone = v_acc(v_ctx,
                         src_clone[0],
                         src_clone[1],
                         tensor->op_params[0],
                         tensor->op_params[1],
                         tensor->op_params[2],
                         tensor->op_params[3]);
  }
  else if (tensor->op == v_OP_NORM) { tensor_clone = v_norm(v_ctx, src_clone[0], *(float*)tensor->op_params); }
  else if (tensor->op == v_OP_GROUP_NORM) {
    const float* float_params = (const float*)tensor->op_params;
    tensor_clone              = v_group_norm(v_ctx, src_clone[0], tensor->op_params[0], float_params[1]);
  }
  else if (tensor->op == v_OP_RMS_NORM) { tensor_clone = v_rms_norm(v_ctx, src_clone[0], *(float*)tensor->op_params); }
  else if (tensor->op == v_OP_RMS_NORM_BACK) {
    const float eps = ((float*)tensor->op_params)[0];
    tensor_clone    = v_rms_norm_back(v_ctx, src_clone[0], src_clone[1], eps);
  }
  else if (tensor->op == v_OP_SILU_BACK) { tensor_clone = v_silu_back(v_ctx, src_clone[0], src_clone[1]); }
  else if (tensor->op == v_OP_L2_NORM) {
    const float eps = ((float*)tensor->op_params)[0];
    tensor_clone    = v_norm_l2(v_ctx, src_clone[0], eps);
  }
  else if (tensor->op == V_OP_SOFT_MAX) {
    if (src1 != nullptr) {
      const float* params = (const float*)tensor->op_params;
      tensor_clone        = v_soft_max_ext(v_ctx, src_clone[0], src_clone[1], params[0], params[1]);
    }
    else { tensor_clone = v_soft_max(v_ctx, src_clone[0]); }
  }
  else if (tensor->op == v_OP_SOFT_MAX_BACK) {
    tensor_clone = v_soft_max_ext_back(v_ctx,
                                       src_clone[0],
                                       src_clone[1],
                                       ((float*)tensor->op_params)[0],
                                       ((float*)tensor->op_params)[1]);
  }
  else if (tensor->op == V_OP_DIAG_MASK_INF) { tensor_clone = v_diag_mask_inf(v_ctx, src_clone[0], tensor->op_params[0]); }
  else if (tensor->op == V_OP_ROPE || tensor->op == v_OP_ROPE_BACK) {
    const int n_dims = ((int32_t*)tensor->op_params)[1];
    const int mode   = ((int32_t*)tensor->op_params)[2];
    //const int n_ctx_ggml       = ((int32_t *) tensor->op_params)[3];
    const int n_ctx_orig_ggml = ((int32_t*)tensor->op_params)[4];
    const float freq_base     = ((float*)tensor->op_params)[5];
    const float freq_scale    = ((float*)tensor->op_params)[6];
    const float ext_factor    = ((float*)tensor->op_params)[7];
    const float attn_factor   = ((float*)tensor->op_params)[8];
    const float beta_fast     = ((float*)tensor->op_params)[9];
    const float beta_slow     = ((float*)tensor->op_params)[10];
    if (mode & v_ROPE_TYPE_MROPE) {
      int32_t* sections = ((int32_t*)tensor->op_params) + 11;
      if (tensor->op == V_OP_ROPE) {
        tensor_clone = v_rope_multi(v_ctx,
                                    src_clone[0],
                                    src_clone[1],
                                    src_clone[2],
                                    n_dims,
                                    sections,
                                    mode,
                                    n_ctx_orig_ggml,
                                    freq_base,
                                    freq_scale,
                                    ext_factor,
                                    attn_factor,
                                    beta_fast,
                                    beta_slow);
      }
      else {
        tensor_clone = v_rope_multi_back(v_ctx,
                                         src_clone[0],
                                         src_clone[1],
                                         src_clone[2],
                                         n_dims,
                                         sections,
                                         mode,
                                         n_ctx_orig_ggml,
                                         freq_base,
                                         freq_scale,
                                         ext_factor,
                                         attn_factor,
                                         beta_fast,
                                         beta_slow);
      }
    }
    else {
      if (tensor->op == V_OP_ROPE) {
        tensor_clone = v_rope_ext(v_ctx,
                                  src_clone[0],
                                  src_clone[1],
                                  src_clone[2],
                                  n_dims,
                                  mode,
                                  n_ctx_orig_ggml,
                                  freq_base,
                                  freq_scale,
                                  ext_factor,
                                  attn_factor,
                                  beta_fast,
                                  beta_slow);
      }
      else {
        tensor_clone = v_rope_ext_back(v_ctx,
                                       src_clone[0],
                                       src_clone[1],
                                       src_clone[2],
                                       n_dims,
                                       mode,
                                       n_ctx_orig_ggml,
                                       freq_base,
                                       freq_scale,
                                       ext_factor,
                                       attn_factor,
                                       beta_fast,
                                       beta_slow);
      }
    }
  }
  else if (tensor->op == v_OP_UNARY) {
    switch (v_get_unary_op(tensor)) {
      case v_UNARY_OP_EXP:
        tensor_clone = v_exp(v_ctx, src_clone[0]);
        break;
      case v_UNARY_OP_SILU:
        tensor_clone = v_silu(v_ctx, src_clone[0]);
        break;
      case v_UNARY_OP_GELU:
        tensor_clone = v_gelu(v_ctx, src_clone[0]);
        break;
      case v_UNARY_OP_GELU_ERF:
        tensor_clone = v_gelu_erf(v_ctx, src_clone[0]);
        break;
      case v_UNARY_OP_GELU_QUICK:
        tensor_clone = v_gelu_quick(v_ctx, src_clone[0]);
        break;
      case v_UNARY_OP_RELU:
        tensor_clone = v_relu(v_ctx, src_clone[0]);
        break;
      case v_UNARY_OP_TANH:
        tensor_clone = v_tanh(v_ctx, src_clone[0]);
        break;
      case v_UNARY_OP_SIGMOID:
        tensor_clone = v_sigmoid(v_ctx, src_clone[0]);
        break;
      case V_UNARY_OP_LOG:
        tensor_clone = v_log(v_ctx, src_clone[0]);
        break;
      case v_UNARY_OP_HARDSIGMOID:
        tensor_clone = v_hardsigmoid(v_ctx, src_clone[0]);
        break;
      case v_UNARY_OP_HARDSWISH:
        tensor_clone = v_hardswish(v_ctx, src_clone[0]);
        break;
      default:
        std::cerr << "Missing vk_check_results OP: " << v_op_name(tensor->op) << std::endl;
        v_ABORT("fatal error");
    }
  }
  else if (tensor->op == v_OP_GLU) {
    if (src_clone[1] == nullptr) { tensor_clone = v_glu(v_ctx, src_clone[0], (v_glu_op)tensor->op_params[0], tensor->op_params[1]); }
    else { tensor_clone = v_glu_split(v_ctx, src_clone[0], src_clone[1], (v_glu_op)tensor->op_params[0]); }
    v_set_op_params_i32(tensor_clone, 2, v_get_op_params_i32(tensor, 2));
    v_set_op_params_i32(tensor_clone, 3, v_get_op_params_i32(tensor, 3));
  }
  else if (tensor->op == V_OP_CPY || tensor->op == v_OP_DUP) {
    if (src1 == nullptr) {
      tensor_clone       = v_dup(v_ctx, src_clone[0]);
      tensor_clone->type = tensor->type;
    }
    else { tensor_clone = v_cpy(v_ctx, src_clone[0], src_clone[1]); }
  }
  else if (tensor->op == V_OP_CONT) { tensor_clone = v_cont_4d(v_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]); }
  else if (tensor->op == V_OP_RESHAPE) { tensor_clone = v_reshape_4d(v_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]); }
  else if (tensor->op == V_OP_VIEW) {
    tensor_clone = v_view_4d(v_ctx,
                             src_clone[0],
                             tensor->ne[0],
                             tensor->ne[1],
                             tensor->ne[2],
                             tensor->ne[3],
                             tensor->nb[1],
                             tensor->nb[2],
                             tensor->nb[3],
                             ((int32_t*)tensor->op_params)[0]);
  }
  else if (tensor->op == V_OP_PERMUTE) {
    int32_t* params = (int32_t*)tensor->op_params;
    tensor_clone    = v_permute(v_ctx, src_clone[0], params[0], params[1], params[2], params[3]);
  }
  else if (tensor->op == v_OP_TRANSPOSE) { tensor_clone = v_transpose(v_ctx, src_clone[0]); }
  else if (tensor->op == v_OP_GET_ROWS) { tensor_clone = v_get_rows(v_ctx, src_clone[0], src_clone[1]); }
  else if (tensor->op == v_OP_ARGSORT) { tensor_clone = v_argsort(v_ctx, src_clone[0], (v_sort_order)*(int*)tensor->op_params); }
  else if (tensor->op == v_OP_SUM) { tensor_clone = v_sum(v_ctx, src_clone[0]); }
  else if (tensor->op == v_OP_SUM_ROWS) { tensor_clone = v_sum_rows(v_ctx, src_clone[0]); }
  else if (tensor->op == V_OP_MEAN) { tensor_clone = v_mean(v_ctx, src_clone[0]); }
  else if (tensor->op == V_OP_ARGMAX) { tensor_clone = v_argmax(v_ctx, src_clone[0]); }
  else if (tensor->op == v_OP_COUNT_EQUAL) { tensor_clone = v_count_equal(v_ctx, src_clone[0], src_clone[1]); }
  else if (tensor->op == V_OP_IM2COL) {
    const int32_t s0 = tensor->op_params[0];
    const int32_t s1 = tensor->op_params[1];
    const int32_t p0 = tensor->op_params[2];
    const int32_t p1 = tensor->op_params[3];
    const int32_t d0 = tensor->op_params[4];
    const int32_t d1 = tensor->op_params[5];

    const bool is_2D = tensor->op_params[6] == 1;
    tensor_clone     = v_im2col(v_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1, is_2D, tensor->type);
  }
  else if (tensor->op == v_OP_IM2COL_3D) {
    const int32_t s0 = tensor->op_params[0];
    const int32_t s1 = tensor->op_params[1];
    const int32_t s2 = tensor->op_params[2];
    const int32_t p0 = tensor->op_params[3];
    const int32_t p1 = tensor->op_params[4];
    const int32_t p2 = tensor->op_params[5];
    const int32_t d0 = tensor->op_params[6];
    const int32_t d1 = tensor->op_params[7];
    const int32_t d2 = tensor->op_params[8];
    const int32_t IC = tensor->op_params[9];

    tensor_clone = v_im2col_3d(v_ctx,
                               src_clone[0],
                               src_clone[1],
                               IC,
                               s0,
                               s1,
                               s2,
                               p0,
                               p1,
                               p2,
                               d0,
                               d1,
                               d2,
                               tensor->type);
  }
  else if (tensor->op == v_OP_TIMESTEP_EMBEDDING) {
    const int32_t dim        = tensor->op_params[0];
    const int32_t max_period = tensor->op_params[1];
    tensor_clone             = v_timestep_embedding(v_ctx, src_clone[0], dim, max_period);
  }
  else if (tensor->op == v_OP_CONV_TRANSPOSE_1D) {
    const int32_t s0 = tensor->op_params[0];
    const int32_t p0 = tensor->op_params[1];
    const int32_t d0 = tensor->op_params[2];
    tensor_clone     = v_conv_transpose_1d(v_ctx, src_clone[0], src_clone[1], s0, p0, d0);
  }
  else if (tensor->op == V_OP_POOL_2D) {
    enum v_op_pool op = static_cast<v_op_pool>(tensor->op_params[0]);
    const int32_t k0  = tensor->op_params[1];
    const int32_t k1  = tensor->op_params[2];
    const int32_t s0  = tensor->op_params[3];
    const int32_t s1  = tensor->op_params[4];
    const int32_t p0  = tensor->op_params[5];
    const int32_t p1  = tensor->op_params[6];

    tensor_clone = v_pool_2d(v_ctx, src_clone[0], op, k0, k1, s0, s1, p0, p1);
  }
  else if (tensor->op == v_OP_CONV_2D) {
    const int32_t s0 = tensor->op_params[0];
    const int32_t s1 = tensor->op_params[1];
    const int32_t p0 = tensor->op_params[2];
    const int32_t p1 = tensor->op_params[3];
    const int32_t d0 = tensor->op_params[4];
    const int32_t d1 = tensor->op_params[5];
    tensor_clone     = v_conv_2d(v_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1);
  }
  else if (tensor->op == v_OP_CONV_TRANSPOSE_2D) {
    const int32_t s = tensor->op_params[0];
    tensor_clone    = v_conv_transpose_2d_p0(v_ctx, src_clone[0], src_clone[1], s);
  }
  else if (tensor->op == V_OP_LEAKY_RELU) {
    const float* op_params = (const float*)tensor->op_params;
    tensor_clone           = v_leaky_relu(v_ctx, src_clone[0], op_params[0], false);
  }
  else if (tensor->op == v_OP_RWKV_WKV6) {
    tensor_clone = v_rwkv_wkv6(v_ctx,
                               src_clone[0],
                               src_clone[1],
                               src_clone[2],
                               src_clone[3],
                               src_clone[4],
                               src_clone[5]);
  }
  else if (tensor->op == v_OP_RWKV_WKV7) {
    tensor_clone = v_rwkv_wkv7(v_ctx,
                               src_clone[0],
                               src_clone[1],
                               src_clone[2],
                               src_clone[3],
                               src_clone[4],
                               src_clone[5],
                               src_clone[6]);
  }
  else if (tensor->op == v_OP_OPT_STEP_ADAMW) {
    src_clone[0]->flags = src0->flags;
    tensor_clone        = v_opt_step_adamw(v_ctx,
                                    src_clone[0],
                                    src_clone[1],
                                    src_clone[2],
                                    src_clone[3],
                                    src_clone[4]);
  }
  else if (tensor->op == v_OP_OPT_STEP_SGD) {
    src_clone[0]->flags = src0->flags;
    tensor_clone        = v_opt_step_sgd(v_ctx,
                                  src_clone[0],
                                  src_clone[1],
                                  src_clone[2]);
  }
  else if (tensor->op == v_OP_ADD_ID) { tensor_clone = v_add_id(v_ctx, src_clone[0], src_clone[1], src_clone[2]); }
  else if (tensor->op == v_OP_SSM_SCAN) {
    tensor_clone = v_ssm_scan(v_ctx,
                              src_clone[0],
                              src_clone[1],
                              src_clone[2],
                              src_clone[3],
                              src_clone[4],
                              src_clone[5],
                              src_clone[6]);
  }
  else if (tensor->op == V_OP_SSM_CONV) { tensor_clone = v_ssm_conv(v_ctx, src_clone[0], src_clone[1]); }
  else {
    std::cerr << "Missing vk_check_results OP: " << v_op_name(tensor->op) << std::endl;
    v_ABORT("fatal error");
  }

  //v_cgraph* cgraph_cpu = new_graph(v_ctx);
  //v_build_foward_expand(cgraph_cpu, tensor_clone);
  //
  //v_graph_compute_with_ctx(v_ctx, cgraph_cpu, 8);

  if (vk_output_tensor > 0 && vk_output_tensor == check_counter) { vk_print_tensor(tensor_clone, "tensor_clone"); }

  comp_size = num_bytes(tensor_clone);

  comp_result = malloc(comp_size);
  memcpy(comp_result, tensor_clone->data, comp_size);
  memcpy(comp_nb, tensor_clone->nb, sizeof(size_t) * 4);

  for (int i = 0; i < v_MAX_SRC; i++) { if (src_buffer[i] != nullptr) { free(src_buffer[i]); } }


  VK_LOG_DEBUG("END v_vk_check_results_0(" << tensor->name << ")");
}

void vk_check_results_1(vk_backend_ctx* ctx, v_cgraph* cgraph, int tensor_idx) {
  v_tensor* tensor = cgraph->nodes[tensor_idx];
  if (tensor->op == v_OP_TRANSPOSE || tensor->op == v_OP_SET_ROWS) { return; }
  if (ctx->num_additional_fused_ops == 1 &&
    tensor->op == v_OP_RMS_NORM &&
    cgraph->nodes[tensor_idx + 1]->op == v_OP_MUL) { tensor = cgraph->nodes[tensor_idx + 1]; }

  if (!(vk_output_tensor > 0 && vk_output_tensor == check_counter) && check_counter <= vk_skip_checks) { return; }

  VK_LOG_DEBUG("v_vk_check_results_1(" << tensor->name << ")");

  v_tensor* src0 = tensor->src[0];
  v_tensor* src1 = tensor->src[1];
  v_tensor* src2
  v_free_ctx(v_ctx);= tensor->src[2];
  v_tensor* src3 = tensor->src[3];

  void* tensor_data = tensor->data;

  if (v_backend_buffer_is_vk(tensor->buffer)) {
    size_t tensor_size = num_bytes(tensor);
    tensor_data        = malloc(tensor_size);

    v_backend_vk_buffer_ctx* buf_ctx = (v_backend_vk_buffer_ctx*)tensor->buffer->context;

    vk_buffer& buffer_gpu = buf_ctx->dev_buffer;
    uint64_t offset       = vk_tensor_offset(tensor) + tensor->view_offs;
    if (offset + tensor_size >= buffer_gpu->size) { tensor_size = buffer_gpu->size - offset; }

    vk_read_buffer(buffer_gpu, offset, tensor_data, tensor_size);
  }

  float first_error_result       = -1.0f;
  float first_error_correct      = -1.0f;
  std::array<int, 4> first_error = {-1, -1, -1, -1};
  double avg_err                 = 0.0;
  size_t counter                 = 0;

  for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
    for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
      for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
        for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
          const bool buffer_size_fit = i3 * comp_nb[3] + i2 * comp_nb[2] + i1 * comp_nb[1] + i0 * comp_nb[0] <
            comp_size;
          float correct = 0.0f;
          float result  = 0.0f;

          if (buffer_size_fit) {
            if (tensor->type == v_TYPE_F32) {
              correct = *(float*)((char*)comp_result + i3 * comp_nb[3] + i2 * comp_nb[2] + i1 * comp_nb[1] + i0 *
                comp_nb[0]);
              result = *(float*)((char*)tensor_data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0
                * tensor->nb[0]);
            }
            else if (tensor->type == v_TYPE_F16) {
              correct = v_fp16_to_fp32(
                *(v_fp16_t*)((char*)comp_result + i3 * comp_nb[3] + i2 * comp_nb[2] + i1 * comp_nb[1] + i0 * comp_nb[
                  0]));
              result = v_fp16_to_fp32(
                *(v_fp16_t*)((char*)tensor_data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 *
                  tensor->nb[0]));
            }
            else if (tensor->type == v_TYPE_BF16) {
              correct = v_bf16_to_fp32(
                *(v_bf16_t*)((char*)comp_result + i3 * comp_nb[3] + i2 * comp_nb[2] + i1 * comp_nb[1] + i0 * comp_nb[
                  0]));
              result = v_bf16_to_fp32(
                *(v_bf16_t*)((char*)tensor_data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] + i0 *
                  tensor->nb[0]));
            }
            else if (tensor->type == v_TYPE_I32) {
              correct = *(int32_t*)((char*)comp_result + i3 * comp_nb[3] + i2 * comp_nb[2] + i1 * comp_nb[1] + i0 *
                comp_nb[0]);
              result = *(int32_t*)((char*)tensor_data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] +
                i0 * tensor->nb[0]);
            }
            else if (tensor->type == v_TYPE_I64) {
              correct = *(int64_t*)((char*)comp_result + i3 * comp_nb[3] + i2 * comp_nb[2] + i1 * comp_nb[1] + i0 *
                comp_nb[0]);
              result = *(int64_t*)((char*)tensor_data + i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1] +
                i0 * tensor->nb[0]);
            }
            else { std::cerr << "Results check not implemented for type " << v_type_name(tensor->type) << std::endl; }
          }
          else {
            std::cerr << "Missing debug code for type " << v_type_name(tensor->type) << std::endl;
            v_ABORT("fatal error");
          }

          if ((std::isnan(correct) != std::isnan(result)) || (std::isinf(correct) != std::isinf(result)) || !
            buffer_size_fit) {
            std::cerr << "ERROR: Invalid value in " << v_op_name(tensor->op) << " i3=" << i3 << " i2=" << i2 <<
              " i1=" << i1 << " i0=" << i0 << " result=" << result << " correct=" << correct << " avg_err=" << (avg_err
                / counter)
              << std::endl;
            std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " <<
              v_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor
              ->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" <<
              tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
            if (src0 != nullptr) {
              std::cerr << "src0=" << src0 << " src0->name=" << src0->name << " op=" << v_op_name(src0->op) <<
                " type=" << v_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" <<
                src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" <<
                src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::endl;
            }
            if (src1 != nullptr) {
              std::cerr << "src1=" << src1 << " src1->name=" << src1->name << " op=" << v_op_name(src1->op) <<
                " type=" << v_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" <<
                src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" <<
                src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::endl;
            }
            if (src2 != nullptr) {
              std::cerr << "src2=" << src2 << " src2->name=" << src2->name << " op=" << v_op_name(src2->op) <<
                " type=" << v_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" <<
                src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" <<
                src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::endl;
            }
            if (src3 != nullptr) {
              std::cerr << "src3=" << src3 << " src3->name=" << src3->name << " op=" << v_op_name(src3->op) <<
                " type=" << v_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" <<
                src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" <<
                src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::endl;
            }
            //std::cerr << "First error: result=" << first_error_result << " correct=" << first_error_correct << " i3=" <<
            //  first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] <<
            //  std::endl;
            std::cerr << std::endl << "Result:" << std::endl;
            v_vk_print_tensor_area(tensor, tensor_data, i0, i1, i2, i3);
            //std::cerr << std::endl << "Correct:" << std::endl//;
            //v_vk_print_tensor_area(tensor, comp_result, i0, i1, i2, i3);
            //std::cerr << std::endl;
            std::vector<const v_tensor*> done;
            v_vk_print_graph_origin(tensor, done);
            v_ABORT("fatal error");
          }
          const double denom = std::fabs(correct) > 1.0f
                                 ? (std::fabs(correct) > 1e-8
                                      ? std::fabs(correct)
                                      : 1e-8)
                                 : 1.0f;
          if (first_error[0] == -1 && std::fabs(correct - result) / denom > 0.5) {
            first_error[0]      = i0;
            first_error[1]      = i1;
            first_error[2]      = i2;
            first_error[3]      = i3;
            first_error_result  = result;
            first_error_correct = correct;
          }

          // Special case, value is infinite, avoid NaN result in avg_err
          // NaN also appears in results, if both are nan error is 0
          if (!std::isinf(correct) && !std::isinf(result) && !std::isnan(correct) && !std::isnan(result)) { avg_err += std::fabs(correct - result) / denom; }
          counter++;
        }
      }
    }
  }

  avg_err /= counter;

  if (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
    ///std::cerr << "TENSOR CHECK: avg_err=" << avg_err << " in " << v_op_name(tensor->op) << " (check " <<
    ///  check_counter << ")" << std::endl;
    ///std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " <<
    ///  v_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1]
    ///  << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] <<
    ///  " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
    if (src0 != nullptr) {
      std::cerr << "src0=" << src0 << " op=" << v_op_name(src0->op) << " type=" << v_type_name(src0->type) <<
        " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2="
        << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" <<
        src0->view_offs << std::endl;
    }
    if (src1 != nullptr) {
      std::cerr << "src1=" << src1 << " op=" << v_op_name(src1->op) << " type=" << v_type_name(src1->type) <<
        " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2="
        << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" <<
        src1->view_offs << std::endl;
    }
    if (src2 != nullptr) {
      std::cerr << "src2=" << src2 << " op=" << v_op_name(src2->op) << " type=" << v_type_name(src2->type) <<
        " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2="
        << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" <<
        src2->view_offs << std::endl;
    }
    if (src3 != nullptr) {
      std::cerr << "src3=" << src3 << " op=" << v_op_name(src3->op) << " type=" << v_type_name(src3->type) <<
        " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2="
        << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" <<
        src3->view_offs << std::endl;
    }
    //std::cerr << "First error: result=" << first_error_result << " correct=" << first_error_correct << " i3=" <<
    //  first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
    //std::cerr << std::endl << "Result:" << std::endl;
    v_vk_print_tensor_area(tensor, tensor_data, 5, 5, 0, 0);
    //std::cerr << std::endl << "Correct:" << std::endl;
    //v_vk_print_tensor_area(tensor, comp_result, 5, 5, 0, 0);
    std::cerr << std::endl;
    std::vector<const v_tensor*> done;
    v_vk_print_graph_origin(tensor, done);
  }

  if (avg_err > 0.5 || std::isnan(avg_err)) {
    ///std::cerr << "ERROR: avg_err=" << avg_err << " in " << v_op_name(tensor->op) << " (check " << check_counter <<
    ///  ")" << std::endl;
    ///std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " <<
    ///  v_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1]
    ///  << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] <<
    ///  " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
    if (src0 != nullptr) {
      std::cerr << "src0=" << src0 << " op=" << v_op_name(src0->op) << " type=" << v_type_name(src0->type) <<
        " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2="
        << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" <<
        src0->view_offs << std::endl;
    }
    if (src1 != nullptr) {
      std::cerr << "src1=" << src1 << " op=" << v_op_name(src1->op) << " type=" << v_type_name(src1->type) <<
        " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2="
        << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" <<
        src1->view_offs << std::endl;
    }
    if (src2 != nullptr) {
      std::cerr << "src2=" << src2 << " op=" << v_op_name(src2->op) << " type=" << v_type_name(src2->type) <<
        " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2="
        << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" <<
        src2->view_offs << std::endl;
    }
    if (src3 != nullptr) {
      std::cerr << "src3=" << src3 << " op=" << v_op_name(src3->op) << " type=" << v_type_name(src3->type) <<
        " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2="
        << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" <<
        src3->view_offs << std::endl;
    }
    //std::cerr << "First error: result=" << first_error_result << " correct=" << first_error_correct << " i3=" <<
    //  first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
    //std::cerr << std::endl << "Result:" << std::endl;
    v_vk_print_tensor_area(tensor, tensor_data, first_error[0], first_error[1], first_error[2], first_error[3]);
    //std::cerr << std::endl << "Correct:" << std::endl;
    //v_vk_print_tensor_area(tensor, comp_result, first_error[0], first_error[1], first_error[2], first_error[3]);
    std::cerr << std::endl;
    std::vector<const v_tensor*> done;
    //v_vk_print_graph_origin(tensor, done);
    //V_ABORT("fatal error");
  }
  else {
    std::cerr << check_counter << " " << tensor->name << " op=" << v_op_name(tensor->op) << " avg_err=" << avg_err <<
      std::endl;
  }

  free(comp_result);
  comp_result = nullptr;
  comp_size   = 0;

  if (v_backend_buffer_is_vk(tensor->buffer)) { free(tensor_data); }

  VK_LOG_DEBUG("END v_vk_check_results_1(" << tensor->name << ")");
}
#endif
#endif //MYPROJECT_MML_VK_TEST_H
