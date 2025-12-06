#ifndef TEXTURE_H
#define TEXUTRE_H
#include "common.h"
#include "image.h"
#include "perlin.h"
//texture virtual class  
//can use parent's Taxonomy System inherit 
class TextureManager { 
public : 
    virtual ~TextureManager()= default; 
    virtual Color value(double u, double v, const Point3& p) const = 0 ; 
}; 


//SingleColor albedo Texture
class SolidColor: public TextureManager{ 
public: 
    //default: 
    SolidColor(const Color& albedo): albedo(albedo){} 
    SolidColor(double r, double g, double b): 
                SolidColor(Color(r, g, b)) {}

    Color value(double u, double v, const Point3& p) const override {
        return albedo;
    }

private: 
    Color albedo;
};

//simple CheckerTexture 
class CheckerTexture : public TextureManager { 
public: 
    CheckerTexture(double scale, std::shared_ptr<TextureManager> even, std::shared_ptr<TextureManager> odd)
    : invScale(1.0 / scale), even(even), odd(odd) {}

    CheckerTexture(double scale, const Color& c1, const Color& c2)
    : CheckerTexture(scale, std::make_shared<SolidColor>(c1), std::make_shared<SolidColor>(c2)) {}

    Color value(double u, double v, const Point3& p) const override {
        auto xInteger = int(std::floor(invScale * p.x()));
        auto yInteger = int(std::floor(invScale * p.y()));
        auto zInteger = int(std::floor(invScale * p.z()));

        bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;

        return isEven ? even->value(u, v, p) : odd->value(u, v, p);
    }

private:
    double invScale;
    std::shared_ptr<TextureManager> even;
    std::shared_ptr<TextureManager> odd;
};




class ImageTexture : public TextureManager {
    public:
    ImageTexture(const char* filename) : image(filename) {}

    Color value(double u, double v, const Point3& p) const override {
        // If we have no texture data, then return solid cyan as a debugging aid.
        if (image.height() <= 0) return Color(0,1,1);

        // Clamp input texture coordinates to [0,1] x [1,0]
        u = Interval(0,1).clamp(u);
        v = 1.0 - Interval(0,1).clamp(v);  // Flip V to image coordinates

        auto i = int(u * image.width());
        auto j = int(v * image.height());
        auto pixel = image.pixel_data(i,j);

        auto color_scale = 1.0 / 255.0;
        return Color(color_scale*pixel[0], color_scale*pixel[1], color_scale*pixel[2]);
    }

private:
    rtw_image image;
}; 

/*class NoiseTexture : public Texture {
public:
    NoiseTexture(double scale) : scale(scale) {}

    Color value(double u, double v, const Point3& p) const override {
        return Color(1,1,1) * noise.noise(p);
    }

private:
    Perlin noise;
    double scale; 
};
*/


#endif //TEXTURE_H;
