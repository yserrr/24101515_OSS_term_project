#ifndef HITTABLE_H
#define HITTABLE_H 

#include "aabb.h"
//predefine
class Material;
class Hittable;
class HitRecord; 



//visible object virtual Super class 
class Hittable{
    public: 
        virtual ~Hittable()= default; 
        virtual bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const = 0;
        virtual AABB bounding_box() const = 0; //hittable box bound setting 
};

//object hit componenent
class HitRecord{ 
public: 
    std::shared_ptr<Material> mat;      // Hit Material Type
    Vec3   normal;                      // nomal vector 
    Point3 p;                           // hit point: p 
    double t;                           // hit point parameter t
    double u;                           // spherical coordinate system axis 
    double v;                           // spherical coordinate system axis
    bool   frontFace;                  // face setting front face
    // Sets the hit record normal vector.
    // NOTE: the parameter `outward_normal` is assumed to have unit length.
    // front face setting : not front -> flip
    void set_face_normal(const Ray& r, const Vec3& outward_normal) {
        frontFace  = dot(r.direction(), outward_normal)< 0; 
        normal     = frontFace ? outward_normal : -outward_normal;
    }
}; 


#endif // HIITABLE_H