#ifndef RAY_SCATTER_H
#define RAY_SCATTER_H

#include "hittable.h"
#include "material.h"

class constantMedium : public Hittable {
public:
    constantMedium(std::shared_ptr<Hittable> boundary, double density, std::shared_ptr<TextureManager> tex)
        : boundary(boundary), negInvDensity(-1/density),
          phaseFunction(std::make_shared<Isotropic>(tex))
  {}

  constantMedium(std::shared_ptr<Hittable> boundary, double density, const Color& albedo)
    : boundary(boundary), negInvDensity(-1/density),
      phaseFunction(std::make_shared<Isotropic>(albedo))
  {}

  bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const override {
      HitRecord rec1, rec2;

      if (!boundary->hit(r, Interval::space, rec1))
          return false;

      if (!boundary->hit(r, Interval(rec1.t+0.0001, INF), rec2))
          return false;

      if (rec1.t < ray_t.min) rec1.t = ray_t.min;
      if (rec2.t > ray_t.max) rec2.t = ray_t.max;

      if (rec1.t >= rec2.t)
          return false;

      if (rec1.t < 0)
          rec1.t = 0;

      auto rayLength               = r.direction().length();
      auto distanceInsideBoundary = (rec2.t - rec1.t) * rayLength;
      auto hit_distance             = negInvDensity * std::log(randomDouble());

      if (hit_distance > distanceInsideBoundary)
          return false;

      rec.t = rec1.t + hit_distance / rayLength;
      rec.p = r.at(rec.t);

      rec.normal = Vec3(1,0,0);  // arbitrary
      rec.frontFace = true;     // also arbitrary
      rec.mat = phaseFunction;

      return true;
    }

    AABB bounding_box() const override { return boundary->bounding_box(); }

private:
    std::shared_ptr<Hittable> boundary;
    double negInvDensity;
    std::shared_ptr<Material> phaseFunction;
};

#endif// RAY_SCATTER_H