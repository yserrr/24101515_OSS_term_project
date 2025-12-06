#ifndef QUAD_H
#define QUAD_H

#include "hittable.h"
#include "hittable_list.h"

class Quad : public Hittable { 
public:
    //coordinate and u,v
    //u,v constitute the plane 
    Quad(const Point3& Q, const Vec3& u, const Vec3& v, std::shared_ptr<Material> mat)
        : Q(Q), u(u), v(v), mat(mat)
    {
        Vec3 n = cross(u, v);
        normal = unitVector(n);
        D = dot(normal, Q);
        w = n / dot(n,n);
        setBoundingBox();
    }

    // Compute the bounding box of all four vertices.
    virtual void setBoundingBox() {
        AABB bbox_diagonal1 = AABB(Q, Q + u + v);
        AABB bbox_diagonal2 = AABB(Q + u, Q + v);
        bbox = AABB(bbox_diagonal1, bbox_diagonal2);
    }

    bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const override {
        double denom = dot(normal, r.direction());

        if (std::fabs(denom) < 1e-8) //Dot Product Near zero, ray is parallel to the plane. 
            return false;

        double t = (D - dot(normal, r.origin())) / denom; 
        if (!ray_t.contains(t)) // Return false if the hit point parameter t is outside the ray interval.
            return false;
            
        // Determine if the hit point lies within the planar shape using its plane coordinates.
        Vec3 hitPoint       = r.at(t);
        Vec3 hitPointVector = hitPoint - Q; //place setting
        auto alpha = dot(w, cross(hitPointVector, v));
        auto beta  = dot(w, cross(u, hitPointVector));

        // Ray hits the 2D shape; set the rest of the hit record and return true.
        if (!is_interior(alpha, beta, rec)) 
            return false;
        //record the hit history
        rec.t   = t; 
        rec.p   = hitPoint; 
        rec.mat = mat;
        //rec.set_face_normal(r, normal);
        return true;
    }
    
AABB bounding_box()
     const override { return bbox; }

virtual bool is_interior(double a, double b, HitRecord& rec) const {
    Interval unit_interval = Interval(0, 1);
    // Given the hit point in plane coordinates, return false if it is outside the
    // primitive, otherwise set the hit record UV coordinates and return true.

    if (!unit_interval.contains(a) || !unit_interval.contains(b))
        return false;

    rec.u = a;
    rec.v = b;
    return true;
}

private:
    Point3 Q;
    Vec3 u, v , w;
    Vec3 normal;
    AABB bbox;
    std::shared_ptr<Material> mat;
    double D;
};



// Returns the 3D box (six sides) that contains the two opposite vertices a & b.
// Box needs only minimum 2point
inline std::shared_ptr<HittableList> box(
    const Point3& a,
    const Point3& b, 
    std::shared_ptr<Material> mat)
{
    auto sides = std::make_shared<HittableList>();

    // Construct the two opposite vertices with the minimum and maximum coordinates.
    auto min = Point3(std::fmin(a.x(),b.x()), std::fmin(a.y(),b.y()), std::fmin(a.z(),b.z()));
    auto max = Point3(std::fmax(a.x(),b.x()), std::fmax(a.y(),b.y()), std::fmax(a.z(),b.z()));
    auto dx = Vec3(max.x() - min.x(), 0, 0);
    auto dy = Vec3(0, max.y() - min.y(), 0);
    auto dz = Vec3(0, 0, max.z() - min.z());
    sides->add(std::make_shared<Quad>(Point3(min.x(), min.y(), max.z()),  dx,  dy, mat)); // front
    sides->add(std::make_shared<Quad>(Point3(max.x(), min.y(), max.z()), -dz,  dy, mat)); // right
    sides->add(std::make_shared<Quad>(Point3(max.x(), min.y(), min.z()), -dx,  dy, mat)); // back
    sides->add(std::make_shared<Quad>(Point3(min.x(), min.y(), min.z()),  dz,  dy, mat)); // left
    sides->add(std::make_shared<Quad>(Point3(min.x(), max.y(), max.z()),  dx, -dz, mat)); // top
    sides->add(std::make_shared<Quad>(Point3(min.x(), min.y(), min.z()),  dx,  dz, mat)); // bottom
    return sides;
}

#endif