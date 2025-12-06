#ifndef COMMON_H 
#define COMMON_H
//common lib and header 
#include<algorithm> 
#include<iostream> 
#include<memory>
#include<vector>
#include<cmath>

#include "interval.h"
#include "color.h"
#include "vec3.h"
#include "ray.h"

//Constant
//floating point number all 1 number-> infinity
const double PI = M_PI; 

//inline macro
inline double degreesToRadians(double degrees){
    return degrees *PI / 180.0; 
}

// Returns a random integer in [min,max].
inline int randomInt(int min, int max) {
    return int(randomDouble(min, max+1));
}

#endif //COMMON_H

