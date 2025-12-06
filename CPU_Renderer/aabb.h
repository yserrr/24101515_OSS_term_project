#ifndef AABB_H
#define AABB_H
#include"common.h" 
class AABB {
public: 
    //box size setting
    Interval x,y,z; 
    AABB() {} 
    // The default AABB is empty, since intervals are empty by default.
    // overloading: interval on 3 dimension setting
    AABB(const Interval& x, const Interval& y, const Interval z)
    : x(x), y(y), z(z)
    {
        padToMinimums();
    }
    
    // Treat the two points a and b as extrema for the bounding box
    // Don't require a particular minimum/maximum coordinate order.
    // 2 point only structure of box (like openGL 2 point -> matrix view setting )
    AABB(const Point3& a, const Point3& b){
        x = (a[0] <= b[0]) ? Interval(a[0], b[0]) : Interval(b[0], a[0]);
        y = (a[1] <= b[1]) ? Interval(a[1], b[1]) : Interval(b[1], a[1]);
        z = (a[2] <= b[2]) ? Interval(a[2], b[2]) : Interval(b[2], a[2]);
        padToMinimums();
    }
    //find maximum box for interval
    AABB(const AABB& box0, const AABB& box1){
        x= Interval(box0.x, box1.x);
        y= Interval(box0.y, box1.y);
        z= Interval(box0.z, box1.z);
    }

    // int to Axis
    const Interval& axisInterval(int n) const {
        if (n == 1) return y;
        if (n == 2) return z;
        return x;
    }

    bool hit(const Ray& r, Interval ray_t) const {
        const Point3& rayOrigin = r.origin();
        const Vec3&   rayDir    = r.direction();
        //test for each axis
        for (int axis = 0; axis < 3; axis++) {
            const Interval& ax = axisInterval(axis);
            const double adinv = 1.0 / rayDir[axis];

            auto t0 = (ax.min - rayOrigin[axis]) * adinv;
            auto t1 = (ax.max - rayOrigin[axis]) * adinv;

            if (t0 < t1) {
                if (t0 > ray_t.min) ray_t.min = t0;
                if (t1 < ray_t.max) ray_t.max = t1;
            } else {
                if (t1 > ray_t.min) ray_t.min = t1;
                if (t0 < ray_t.max) ray_t.max = t0;
            }
            if (ray_t.max <= ray_t.min) return false;
        }
        return true;
    }
    // Returns the index of the longest axis of the bounding box.
    // return x: 0, y: 1, z: 2
    int longestAxis() const {
        return  x.size()>z.size() ? (x.size()>y.size()? 0: 2):
                                    (y.size()>z.size()? 1: 2);
    }
    static const AABB empty; 
    static const AABB space;
private:
    void padToMinimums() {
        // Adjust the AABB so that no side is narrower than some delta, padding if necessary.
        // mininum size of box setting
        // If box is so thin like paper, it minium size of thickness
        double delta = 0.0001;
        if (x.size() < delta) x = x.expand(delta);
        if (y.size() < delta) y = y.expand(delta);
        if (z.size() < delta) z = z.expand(delta);
    }
};

// empty box: base;
const AABB AABB::empty= AABB(
                        Interval::empty,    
                        Interval::empty,    
                        Interval::empty);

// infinity space box 
const AABB AABB::space= AABB(
                        Interval::space,    
                        Interval::space,    
                        Interval::space);

//basic operator overloading  
AABB operator+(const AABB& bbox, const Vec3& offset) {
    return AABB(bbox.x + offset.x(), bbox.y + offset.y(), bbox.z + offset.z());
}   
AABB operator+(const Vec3& offset, const AABB& bbox) {
    return bbox + offset;
}

#endif //AABB_H