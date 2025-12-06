#ifndef PERLIN_H
#define PERLIN_H

#include "common.h"
class Perlin {
public:
    Perlin() {
        for (int i = 0; i < pointCount; i++) 
            randfloat[i] = randomDouble();

        perlinGeneratePerm(permX);
        perlinGeneratePerm(permY);
        perlinGeneratePerm(permZ);
    }

    double noise(const Point3& p) const {
        auto u = p.x() - std::floor(p.x());
        auto v = p.y() - std::floor(p.y());
        auto w = p.z() - std::floor(p.z());

        u = u*u*(3-2*u);
        v = v*v*(3-2*v);
        w = w*w*(3-2*w);
        
        auto i = int(std::floor(p.x()));
        auto j = int(std::floor(p.y()));
        auto k = int(std::floor(p.z()));
        double c[2][2][2];

        for (int di=0; di < 2; di++)
            for (int dj=0; dj < 2; dj++)
                for (int dk=0; dk < 2; dk++)
                    c[di][dj][dk] = randfloat[
                        permX[(i+di) & 255] ^
                        permY[(j+dj) & 255] ^
                        permZ[(k+dk) & 255]
                    ];

        return trilinear_interp(c, u, v, w);
    }

private:
    static const int pointCount = 256;
    double randfloat[pointCount];
    int permX[pointCount];
    int permY[pointCount];
    int permZ[pointCount];

    static double trilinear_interp(double c[2][2][2], double u, double v, double w) {
        auto accum = 0.0;
        for (int i=0; i < 2; i++)
            for (int j=0; j < 2; j++)
                for (int k=0; k < 2; k++)
                    accum += (i*u + (1-i)*(1-u))
                            * (j*v + (1-j)*(1-v))
                            * (k*w + (1-k)*(1-w))
                            * c[i][j][k];

        return accum;
    }

    static void perlinGeneratePerm(int* p) {
        for (int i = 0; i < pointCount; i++)
            p[i] = i;

        permute(p, pointCount);
    }

    static void permute(int* p, int n) {
        for (int i = n-1; i > 0; i--) {
            int target = randomInt(0, i);
            int tmp = p[i];
            p[i] = p[target];
            p[target] = tmp;
        }
    }
};

#endif //PERLIN_H
