//
// Created by dlwog on 25. 11. 16..
//

#ifndef MYPROJECT_MML_HASH_H
#define MYPROJECT_MML_HASH_H
#include <stdint.h>


typedef uint32_t v_bitset_t;

struct v_hash_set
{
  size_t size;
  uint32_t* used; // whether or not the keys are in use i.e. set
  struct v_tensor** keys; // actual tensors in the set, keys[i] is only defined if v_bitset_get(used, i)
};

struct hash_map
{
  struct v_hash_set set;
  struct v_tensor** vals;
};

void v_hash_map_free(struct hash_map* map);
struct hash_map* v_new_hash_map(size_t size);
size_t v_visit_parents(struct v_cgraph* cgraph, struct v_tensor* node);
size_t v_hash_size(size_t min_sz);
void v_hash_set_free(struct v_hash_set* hash_set);
size_t v_visit_parents(struct v_cgraph* cgraph, struct v_tensor* node);

#endif //MYPROJECT_MML_HASH_H
