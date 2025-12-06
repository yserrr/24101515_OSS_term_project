#ifndef INTERVAL_H
#define INTERVAL_H

#include<limits>
const double INF = std::numeric_limits<double>::infinity(); //const infitiy
 
// interval class : check threshold 
class Interval{ 
public:
    double min, max;
    Interval(): min(INF), max(-INF){} //default construct :: empty 
    Interval(double min,const double& max): min(min), max(max){} 

    double size() const { 
        return max - min; 
    }
    
    bool contains(double x) const {
        return min <= x && x<= max; 
    }

    bool surrounds(double x) const { 
        return min<x && x< max ; 
    }
    
    double clamp(double x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    Interval expand(double delta) const {
        auto padding = delta/2;
        return Interval(min - padding, max + padding);
    }

    //create the interval tightly enclosing the two input intervals.
    Interval(const Interval& a, const Interval& b){
        min = a.min<= b.min ? a.min:b.min;
        max = a.max>= b.max ? a.max:b.max;
    }

    static const Interval empty; 
    static const Interval space; 
};

//operator overriding:
//interval move
Interval operator+(const Interval& ival, double displacement) {
    return Interval(ival.min + displacement, ival.max + displacement);
}
//r case overriding
Interval operator+(double displacement, const Interval& ival) {
    return ival + displacement;
}

const Interval Interval::empty= Interval(); 
const Interval Interval::space= Interval(-INF, INF); 
#endif