#ifndef MATERIAL_H 
#define MATERIAL_H 

#include "common.h"
#include "texture.h"
#include "hittable.h"

class Material{
public: 
    virtual ~Material() = default; 
    virtual bool scatter(
                            const   Ray&       rayIn ,   
                            const   HitRecord& rec,
                                    Color&     attenuation,  
                                    Ray&       scatteredRay
                        )   const
    {
        return false ;
    }  
    //default: no color
    virtual Color emitted(double u, double v, const Point3& p) const {
        return Color(0,0,0);
    }
}; 

//diffuse material 
class Lambertian: public Material{
public: 
    Lambertian(const Color& albedo): 
        tex(std::make_shared<SolidColor>(albedo)) {} //single color albedo map

    Lambertian(std::shared_ptr<TextureManager> tex) : tex(tex)  {} //load texture

    bool scatter(
                    const Ray&        rayIn, 
                    const HitRecord&  rec, 
                          Color&      attenuation, 
                          Ray&        scatteredRay
                )   const override 
    {
        Vec3 newDir = rec.normal + randomUnitVector(); //get scattered vector
        if (newDir.nearZero())    //n+v = 0 case: too small vector size padding
            newDir= rec.normal;

        scatteredRay   = Ray(rec.p, newDir ,rayIn.time()); 
        attenuation    = tex-> value(rec.u,rec.v ,rec.p);
        return true;
    }
private:
    std::shared_ptr<TextureManager> tex; 
};

//fuzz metal class
class Metal : public Material {
public:
    Metal(const Color& albedo, double fuzz): 
        albedo(albedo), fuzz(fuzz<1? fuzz:1 ) {}

    bool scatter(
                    const Ray&         rayIn, 
                    const HitRecord&   rec,
                            Color&     attenuation, 
                            Ray&       scatteredRay
                )   const override 
    {
        //compute reflected direction
        Vec3 reflectedDir = reflect(rayIn.direction(), rec.normal);
             reflectedDir = unitVector(reflectedDir) + (fuzz*randomUnitVector());
        scatteredRay = Ray(rec.p, reflectedDir, rayIn.time());
        attenuation  = albedo;
        return (dot(scatteredRay.direction(), rec.normal)> 0);
    }

private:
    Color albedo;
    double fuzz; 
};


//like glass, has opacity
class Dielectric : public Material {
public:
    Dielectric(double refractionIndex) : refractionIndex(refractionIndex) {}

    bool scatter(
                    const Ray&         rayIn, 
                    const HitRecord&   rec, 
                          Color&       attenuation, 
                          Ray&         scatteredRay
                )   const override 
        {
        attenuation = Color(1.0, 1.0, 1.0);
        double ri = rec.frontFace ? (1.0 / refractionIndex) : refractionIndex;

        Vec3 unitDirection  = unitVector(rayIn.direction());
        
        double cosTheta     = std::fmin(dot(-unitDirection, rec.normal), 1.0);
        double sinTheta     = std::sqrt(1.0 - cosTheta*cosTheta); 

        bool cannot_refract = ri * sinTheta > 1.0;
        Vec3 direction;

        if (cannot_refract || reflectance(cosTheta, ri) > randomDouble())
            direction = reflect(unitDirection, rec.normal);
        else
            direction = refract(unitDirection, rec.normal, ri);

        scatteredRay = Ray(rec.p, direction,rayIn.time());
        return true;
    }

private:
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media 
    // think snail's low
    double refractionIndex;

    // Use Schlick's approximation for reflectance.
    static double reflectance(double cosine, double refractionIndex) {
        double r0 = (1 - refractionIndex) / (1 + refractionIndex);
               r0 = r0*r0;

        return r0 + (1-r0)*std::pow((1 - cosine),5);
    }
};

//Emit Diffuse light Material
//Use singleColor Text and add;
class DiffuseLight : public Material {

public:
    DiffuseLight(std::shared_ptr<TextureManager> tex) : tex(tex) {}
    DiffuseLight(const Color& emit) : tex(std::make_shared<SolidColor>(emit)) {}
    Color emitted(double u, double v, const Point3& p) const override {
        return tex->value(u, v, p);
    }
    
private:
    std::shared_ptr<TextureManager> tex;
};



class Isotropic : public Material {
public:
    Isotropic(const Color& albedo): tex(std::make_shared<SolidColor>(albedo)) {}
    Isotropic(std::shared_ptr<TextureManager> tex): tex(tex) {}
  
    bool scatter(
                const Ray&       rayIn, 
                const HitRecord& rec,
                      Color&     attenuation, 
                      Ray&       scatteredRay
                    )
                const override 
    {
          scatteredRay = Ray(rec.p, randomUnitVector(), rayIn.time());
          attenuation  = tex->value(rec.u, rec.v, rec.p);
          return true;
    }
  
private:
    std:: shared_ptr<TextureManager> tex;
};
  
#endif  //MATERIAL_H