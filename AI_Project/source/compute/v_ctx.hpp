#pragma once
// assert that pointer is aligned to V_MEM_ALIGN
#define V_ASSERT_ALIGNED(ptr) \
V_ASSERT(((uintptr_t) (ptr))%V_MEM_ALIGN == 0)

struct v_init_param {
  size_t mem_size;  // bytes
  void* mem_buffer; // if NULL, memory will be allocated internally
  bool no_alloc;    // don't allocate memory for the tensor data
};

struct v_object {
  size_t offs;
  size_t size;
  v_object* next;
  v_object_type type;
  char padding[4];
  void print();
};

struct v_ctx {
  size_t mem_size;
  void* mem_buffer;
  bool mem_buffer_owned;
  bool no_alloc;
  int n_objects;
  v_object* objects_begin;
  v_object* objects_end;

  v_tensor* get_first_tensor() const;
  v_tensor* get_next_tensor(v_tensor* tensor) const;
  size_t get_max_tensor_size();
  void reset();
  void print_objects() const;
};

static_assert(sizeof(v_object) % V_MEM_ALIGN == 0, "v_object size must be a multiple of V_MEM_ALIGN");
