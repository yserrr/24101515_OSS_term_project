#pragma once
#include "v.h"
#include "v-backend.h"

#define v_VK_NAME "Vulkan"
#define v_VK_MAX_DEVICES 16
#include <string>

struct vk_device_ctx {
  size_t device;
  std::string name;
  std::string description;
  bool is_integrated_gpu;
  std::string pci_bus_id;
};

v_backend_t backend_vk_init(size_t dev_num);
void vk_device_buffer_set_tensor(v_backend_buffer_t buffer, v_tensor* tensor, const void* data, size_t offset, size_t size);
size_t vk_device_buffer_get_align(v_backend_buffer_type_t buft);
void vk_free_buffer(v_backend_buffer_t buffer);
int vk_get_device_count(void);
void vk_get_device_description(int device, char* description, size_t description_size);
void vk_get_device_memory(int device, size_t* free, size_t* total);
void* vk_device_buffer_get_base(v_backend_buffer_t buffer);
bool vk_buffer_cpy_tensor(v_backend_buffer_t buffer, const v_tensor* src, v_tensor* dst);
void vk_buffer_clear(v_backend_buffer_t buffer, uint8_t value);
const char* vk_name(v_backend_t backend);
void vk_graph_optimize(v_backend_t backend, struct v_cgraph* graph);
size_t vk_host_buffer_get_align(v_backend_buffer_type_t buft);
enum v_status vk_buffer_init_tensor(v_backend_buffer_t buffer, const v_tensor* tensor);

v_backend_buffer_type_t vk_device_buffer_type(size_t dev_num);
v_backend_buffer_type_t vk_host_buffer_type(void);
void vk_device_buffer_get_tensor(v_backend_buffer_t buffer, const v_tensor* tensor, void* data, size_t offset, size_t size);
v_status vk_graph_compute(v_backend_t backend, v_cgraph* cgraph);


// extension need  VK_KHR_compute_shader_derivatives
