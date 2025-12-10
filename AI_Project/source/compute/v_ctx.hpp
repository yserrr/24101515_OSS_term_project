//
// Created by dlwog on 25. 11. 20..
//

#ifndef MYPROJECT_MML_CONTEXT_HPP
#define MYPROJECT_MML_CONTEXT_HPP
// assert that pointer is aligned to v_MEM_ALIGN
#define v_ASSERT_ALIGNED(ptr) \
V_ASSERT(((uintptr_t) (ptr))%v_MEM_ALIGN == 0)
struct v_ctx
{
  size_t mem_size;
  void* mem_buffer;
  bool mem_buffer_owned;
  bool no_alloc;
  int n_objects;
  v_object* objects_begin;
  v_object* objects_end;
  void reset()
  {
    this->n_objects = 0;
    this->objects_begin = NULL;
    this->objects_end = NULL;
  }
};



#endif //MYPROJECT_MML_CONTEXT_HPP
