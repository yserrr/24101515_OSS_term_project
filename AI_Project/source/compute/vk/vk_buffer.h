#ifndef MYPROJECT_VK_BUFFER_H
#define MYPROJECT_VK_BUFFER_H
#include "vk_common.h"
struct vk_staging_memcpy;
struct vk_sub_buffer;
struct vk_staging_memset;

void vk_read_buffer(vk_buffer& src, size_t offset, void* dst, size_t size);
void vk_get_host_buffer(vk_device& device, const void* ptr, vk_buffer& buf, size_t& buf_offset);
void vk_buffer_read2d_async(vk_context subctx, vk_buffer& src, size_t offset, void* dst, size_t spitch, size_t dpitch, size_t width, size_t height, bool sync_staging = false);
void vk_deffered_memcpy(void* dst, const void* src, size_t size, std::vector<vk_staging_memcpy>* memcpys = nullptr);
void vk_ensure_sync_staging_buffer(vk_device& device, size_t size);
void vk_deffered_memset(void* dst, uint32_t val, size_t size, std::vector<vk_staging_memset>* memsets = nullptr);
void vk_buffer_read_async(vk_context subctx, vk_buffer& src, size_t offset, void* dst, size_t size, bool sync_staging = false);
vk_buffer vk_create_buffer(vk_device& device, size_t size, const std::initializer_list<vk::MemoryPropertyFlags>& req_flags_list);
vk_buffer vk_create_buffer_check(vk_device& device, size_t size, vk::MemoryPropertyFlags req_flags, vk::MemoryPropertyFlags fallback_flags = vk::MemoryPropertyFlags(0));
vk_sub_buffer v_vk_subbuffer(const vk_backend_ctx* ctx, const vk_buffer& buf, size_t offset = 0);
void vk_device_buffer_memset_tensor(v_backend_buffer_t buffer, v_tensor* tensor, uint8_t value, size_t offset, size_t size);
void vk_buffer_memset(vk_buffer& dst, size_t offset, uint32_t c, size_t size);
void vk_buffer_memset_async(vk_context& ctx, vk_buffer& dst, size_t offset, uint32_t c, size_t size);
void v_vk_buffer_copy(vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size);
void v_vk_buffer_write(vk_buffer& dst, size_t offset, const void* src, size_t size);
void v_vk_buffer_copy_async(vk_context& ctx, vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size);
void v_vk_buffer_write_async(vk_context subctx, vk_buffer& dst, size_t offset, const void* src, size_t size, bool sync_staging = false);
void v_vk_buffer_write_2d(vk_buffer& dst, size_t offset, const void* src, size_t spitch, size_t width, size_t height);
void v_vk_buffer_write_2d_async(vk_context subctx, vk_buffer& dst, size_t offset, const void* src, size_t spitch, size_t width, size_t height, bool sync_staging = false);
void* vk_host_malloc(vk_device& device, size_t size);
void vk_host_free(vk_device& device, void* ptr);
void v_vk_buffer_write_nc_async(vk_backend_ctx* ctx, vk_context& subctx, vk_buffer& dst, size_t offset, v_tensor* const tensor, bool sync_staging = false);
void vk_host_buffer_free(v_backend_buffer_t buffer);
bool vk_buffer_cpy_tensor(v_backend_buffer_t buffer, v_tensor* const src, v_tensor* dst);
void vk_destroy_buffer(vk_buffer& buf);
const char* vk_host_buffer_name(v_backend_buffer_type_t buft);

VkDeviceSize v_vk_get_max_buffer_range(const vk_backend_ctx* ctx, const vk_buffer& buf,
                                       const VkDeviceSize offset);

struct vk_buffer_type_context {
  std::string name;
  vk_device device;
};

struct vk_buffer_struct {
  vk::Buffer buffer              = VK_NULL_HANDLE;
  vk::DeviceMemory device_memory = VK_NULL_HANDLE;
  vk::MemoryPropertyFlags memory_property_flags;
  void* ptr;
  size_t size = 0;
  vk::DeviceAddress bda_addr{};
  vk_device device;
  ~vk_buffer_struct();
};

struct vk_sub_buffer {
  vk_buffer buffer;
  uint64_t offset;
  uint64_t size;
  operator vk::DescriptorBufferInfo() const { return {buffer->buffer, offset, size}; }
};

// Allow pre-recording command buffers
struct vk_staging_memcpy {
  vk_staging_memcpy(void* _dst, const void* _src, size_t _n) : dst(_dst), src(_src), n(_n) {}

  void* dst;
  const void* src;
  size_t n;
};

struct vk_staging_memset {
  vk_staging_memset(void* _dst, uint32_t _val, size_t _n) : dst(_dst), val(_val), n(_n) {}

  void* dst;
  uint32_t val;
  size_t n;
};

void vk_destroy_buffer(vk_buffer& buf);

struct v_backend_vk_buffer_ctx {
  vk_device_ref device;
  vk_buffer dev_buffer;
  std::string name;

  v_backend_vk_buffer_ctx(vk_device_ref device, vk_buffer&& dev_buffer, std::string& name) :
    device(device),
    dev_buffer(dev_buffer),
    name(name) {}

  ~v_backend_vk_buffer_ctx() { vk_destroy_buffer(dev_buffer); }
};

const char* vk_device_buffer_name(v_backend_buffer_type_t buft);

size_t vk_device_buffer_get_align(v_backend_buffer_type_t buft);

size_t vk_device_buffer_get_max_size(v_backend_buffer_type_t buft);

size_t vk_device_buffer_get_alloc_size(v_backend_buffer_type_t buft, const v_tensor* tensor);

v_backend_buffer_type_t vk_device_buffer_type(size_t dev_num);


v_backend_buffer_t vk_host_buffer_alloc(v_backend_buffer_type_t buft,
                                        size_t size);

size_t vk_host_buffer_get_align(v_backend_buffer_type_t buft);

size_t vk_host_buffer_get_align();

size_t vk_host_buffer_get_max_size(v_backend_buffer_type_t buft);



v_backend_buffer_type_t vk_host_buffer_type();

#endif //MYPROJECT_VK_BUFFER_H
