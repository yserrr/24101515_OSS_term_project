#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "common.h"
#include "hittable.h"
#include "aabb.h"


class HittableList : public Hittable {
public:
    std::vector<std::shared_ptr<Hittable>> objects;
    HittableList() {}
    HittableList(std::shared_ptr<Hittable> object) { add(object); }
    void clear() { objects.clear(); }


    void add(std::shared_ptr<Hittable> object) {
        objects.push_back(object);
        bbox = AABB(bbox,object-> bounding_box());
    }
    /*
    hit function:
    get record class and ray parameter direction size 
    if ray (parameter in) then find closest object 
     */
    bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const override {
        HitRecord temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (const auto& object : objects) {
            if (object->hit(r, ray_t, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
        return hit_anything;
    }
    
    AABB bounding_box() const override {return bbox;}
    
    private:
    AABB bbox;
};


#endif