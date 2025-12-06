#ifndef MYPROJECT_VK_BUFFER_H
#define MYPROJECT_VK_BUFFER_H
#include "vk_common.h"

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

inline void vk_destroy_buffer(vk_buffer& buf) {
  if (buf == nullptr) { return; }

  #ifdef v_VULKAN_MEMORY_DEBUG
  if (buf->device != nullptr) { buf->device->memory_logger->log_deallocation(buf); }
  #endif

  buf.reset();
}

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

void mmlVKBufferRead(vk_buffer& src,
                     size_t offset,
                     void* dst,
                     size_t size);

void vk_get_host_buffer(vk_device& device,
                        const void* ptr,
                        vk_buffer& buf,
                        size_t& buf_offset);
void vk_buffer_read2d_async(vk_context subctx, vk_buffer& src, size_t offset, void* dst, size_t spitch,
                            size_t dpitch, size_t width, size_t height, bool sync_staging = false);

vk_buffer vk_create_buffer_check(vk_device& device, size_t size, vk::MemoryPropertyFlags req_flags,
                                 vk::MemoryPropertyFlags fallback_flags = vk::MemoryPropertyFlags(0));
vk_buffer vk_create_buffer(vk_device& device, size_t size,
                           const std::initializer_list<vk::MemoryPropertyFlags>& req_flags_list);
void vk_ensure_sync_staging_buffer(vk_device& device, size_t size);
void vk_deffered_memcpy(void* dst, const void* src, size_t size, std::vector<vk_staging_memcpy>* memcpys = nullptr);
void vk_deffered_memset(void* dst, uint32_t val, size_t size, std::vector<vk_staging_memset>* memsets = nullptr);

inline void vk_buffer_read_async(vk_context subctx, vk_buffer& src, size_t offset, void* dst, size_t size,
                                 bool sync_staging = false) { return vk_buffer_read2d_async(subctx, src, offset, dst, size, size, size, 1, sync_staging); }


#endif //MYPROJECT_VK_BUFFER_H
