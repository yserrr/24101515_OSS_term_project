#ifndef RAY_H
#define RAY_H

#include "vec3.h"
//origin+ t* direction 
class Ray {
public:
    Ray() {}
    Ray(const Point3& origin, const Vec3& direction, double time): 
    orig(origin), dir(direction), tm(time) {}
    
    Ray(const Point3& origin, const Vec3& direction): 
    Ray(origin, direction, 0) {}

    const Point3& origin() const  { 
        return orig; 
    }

    const Vec3& direction() const { 
        return dir; }
    double time() const { return tm; }

    Point3 at(double t) const {
        return orig + t*dir;
    }

  private:
    Point3 orig;
    Vec3 dir;
    double tm;
};

#endif