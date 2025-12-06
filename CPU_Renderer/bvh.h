#ifndef BVH_H
#define BVH_H 

#include "aabb.h"
#include "hittable.h"
#include "hittable_list.h"

class BvhNode: public Hittable{
public : 
    BvhNode(HittableList list): BvhNode(list.objects,0, list.objects.size()){ 
    // There's a C++ subtlety here. This constructor (without span indices) creates an
    // implicit copy of the hittable list, which we will modify. The lifetime of the copied
    // list only extends until this constructor exits. That's OK, because we only need to
    // persist the resulting bounding volume hierarchy.
    }
    // Build the bounding box of the span of source objects.
    // size_t -> unsigned integer type based on cpu
    // list -> bbox append and find space and sort for optimize
    BvhNode(std::vector<std::shared_ptr<Hittable>>& objects, size_t start , size_t end){
        bbox = AABB::empty; 
        for (size_t it=start; it < end; it++) 
            bbox = AABB(bbox, objects[it]->bounding_box());

        int axis = bbox.longestAxis();
        auto comparator = (axis == 0) ? box_x_compare
                        : (axis == 1) ? box_y_compare
                                      : box_z_compare;

        size_t objectSpan = end - start;
        if(objectSpan == 1) left = right = objects[start];
        else if(objectSpan == 2){
            left  = objects[start];
            right = objects[start+1];
        }else{
            std::sort(std::begin(objects) +start, std::begin(objects) +end, comparator);
            size_t mid = start + objectSpan/2;
            left       = std::make_shared<BvhNode>(objects, start, mid);
            right      = std::make_shared<BvhNode>(objects, mid,   end);
        }
    }
    //size_t: cpu bit 
    bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const override{
        if(!bbox.hit(r, ray_t)) return false; 
        bool hit_left  = left-> hit(r, ray_t, rec);
        bool hit_right = right->hit(r, Interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);
        return hit_left || hit_right;
    }

    AABB bounding_box() const override{return bbox;}

private: 
    std::shared_ptr<Hittable> left;
    std::shared_ptr<Hittable> right;
    AABB bbox;

    //box axis compare function  
    static bool boxCompare
    (const std:: shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b, int axisIndex) {
        auto aAxisInterval = a->bounding_box().axisInterval(axisIndex);
        auto bAxisInterval = b->bounding_box().axisInterval(axisIndex);
        return aAxisInterval.min > bAxisInterval.min;
    }

    static bool box_x_compare 
    (const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b) {
        return boxCompare(a, b, 0);
    }

    static bool box_y_compare 
    (const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b) {
        return boxCompare(a, b, 1);
    }

    static bool box_z_compare 
    (const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b) {
        return boxCompare(a, b, 2);
    }
};


#endif //BVH_H
