
#ifndef MYPROJECT_MML_OBJECT_HPP
#define MYPROJECT_MML_OBJECT_HPP

enum v_object_type
{
  MML_TENSOR,
  MML_GRAPH,
  MML_BUFFER
};

struct v_object
{
  size_t offs;
  size_t size;
  struct v_object* next;
  enum v_object_type type;
  char padding[4];
};
#endif //MYPROJECT_MML_OBJECT_HPP