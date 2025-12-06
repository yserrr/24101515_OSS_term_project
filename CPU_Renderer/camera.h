#ifndef CAMERA_H 
#define CAMERA_H
#include "material.h"
#include "common.h"
#include "hittable_list.h"
#include <fstream>
#include <sstream> 
#include <ctime>
///todo: setting with module connection
/// ->rendering call

class BaseCamera{
public: 
    double aspectRatio      = 1.0;  // Ratio of image width over height
    int    imageWidth       = 100;  // Rendered image width in pixel count
    int    samplesPerPixel  = 10;   // Count of random samples for each pixel
    int    maxDepth         = 10; 
    Color  background;              // Scene background color default vector = (0,0,0)

    double vfov             = 90;              // Vertical view angle (field of view)
    Point3 lookfrom         = Point3(0,0,0);   // Point camera is looking from
    Point3 lookat           = Point3(0,0,-1);  // Point camera is looking at
    Vec3   vup              = Vec3(0,1,0);     // CameraManager-relative "up" direction
    double defocusAngle     = 0;               // Variation angle of rays through each pixel
    double focalLength      = 10;               // Distance from camera lookfrom point to plane of perfect focus
    //rendering funtion : compute Ray and draw
    void render(const Hittable& world) {
        initialize();
        (*rendered) << "P3\n" << imageWidth << ' ' << imageHeight << "\n255\n"; //PPM format header 
        for (int j = 0; j < imageHeight; j++) {
            std::clog << "\rScanlines remaining: " << (imageHeight - j) << ' ' << std::flush;
            for (int i = 0; i < imageWidth; i++){
                Color pixelColor(0,0,0); 
                for (int k= 0; k<samplesPerPixel; k++){
                    Ray r = getRay(i,j);
                    pixelColor += rayColor(r, maxDepth, world);
                }
                pixelColor /= samplesPerPixel;
                writeColor((*rendered), pixelColor);
            }
        }
        std::clog << "\rDone.                 \n";
    }

    private:
    int    imageHeight;          // Rendered image height fixed(use width)
    double pixelSamplesScale;    // Color scale factor for a sum of pixel samples
    Vec3   u, v, w;              // CameraManager frame basis vectors
    Point3 center;               // CameraManager center
    Point3 pixelStart;           // Location of pixel 0, 0
    Vec3   pixelDu;              // Offset to pixel to the right
    Vec3   pixelDv;              // Offset to pixel below
    Vec3   defocusDu;            // Defocus disk horizontal radius
    Vec3   defocusDv;            // Defocus disk vertical radius
    
    std::unique_ptr<std::ofstream> rendered= nullptr; //render image
    
    void initialize(){
        //file name setting
        char time_buffer[100];
        std::time_t now = std::time(nullptr);
        std::strftime(time_buffer, sizeof(time_buffer), "%Y%m%d_%H%M%S", std::localtime(&now));
        std::string   file_name=   std::string("render_")+ time_buffer+ ".ppm";

        // Open output file setting
        rendered = std::make_unique<std::ofstream>(file_name);
        if (!rendered->is_open()) {
            rendered.reset();
            std::cerr << "Error: Could not open file " << file_name << " for writing.\n";
            std::exit(1);
        }
        
        imageHeight = int(imageWidth / aspectRatio);
        imageHeight = (imageHeight < 1) ? 1 : imageHeight;
        
        //lens setting
        // Determine viewport dimensions.
        center = lookfrom;
        double theta          = degreesToRadians(vfov);
        double h              = std::tan(theta/2);
        double focalLength    = (lookfrom- lookat).length(); //OA vector size
        double viewportHeight = 2 * h * focalLength;
        double viewportWidth  = viewportHeight* (double(imageWidth)/imageHeight);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unitVector(lookfrom - lookat);
        u = unitVector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        Vec3 viewport_u = viewportWidth  *   u;    // Vector across viewport horizontal edge
        Vec3 viewport_v = viewportHeight * (-v);   // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixelDu = viewport_u / imageWidth;
        pixelDv = viewport_v / imageHeight;

        // Calculate the location of the upper left pixel.
        Vec3 viewportUpperLeft = center - (focalLength * w) - viewport_u/2 - viewport_v/2;
             pixelStart        = viewportUpperLeft + 0.5 * (pixelDu + pixelDv);
        // Calculate the camera defocus disk basis vectors.
        auto defocusRadius = focalLength * std::tan(degreesToRadians(defocusAngle / 2));
        defocusDu          = u * defocusRadius;
        defocusDv          = v * defocusRadius;
        
    }

    // Construct a camera ray originating from the defocus disk and directed at a randomly
    // sampled point around the pixel location i, j.
    const Ray getRay(int i, int j){
        Vec3 offset = sampleSquare() ; 
        Vec3 pixelSample= pixelStart;
             pixelSample += (i+offset.x())*pixelDu; 
             pixelSample += (j+offset.y())*pixelDv; 

        Point3 ray_origin = (defocusAngle <= 0) ? center : defocusDiskSample();
        double ray_time= randomDouble();
        Vec3 rDirection = pixelSample - center; 
        return Ray(center,rDirection, ray_time);
    }

    // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
    const Vec3 sampleSquare() const {
        return Vec3(randomDouble() - 0.5, randomDouble() - 0.5, 0);
    }

    //ray to color
    Color rayColor(const Ray& r, int depth,  const Hittable& world){
        if (depth <= 0) return Color(0,0,0); //If we've exceeded the ray bounce limit, no more light is gathered.
        HitRecord rec;

        if (!world.hit(r, Interval(0.001, INF),rec)) return background; //no hit
        Ray   scattered;
        Color attenuation; //albedo
        Color colorFromEmission = rec.mat-> emitted(rec.u,rec.v, rec.p); 

        //not scatter case : only emit or no black
        //scatter          : recursive to find other hit ray
        if (!rec.mat->scatter(r, rec, attenuation, scattered)) return colorFromEmission;
        Color colorFromScatter = attenuation * rayColor(scattered, depth-1, world);

        return colorFromEmission + colorFromScatter;
    }
    // Returns a random point in the camera defocus disk
    const Point3 defocusDiskSample()  {    
        auto p = randomInUnitDisk();
        return center + (p[0] * defocusDu) + (p[1] * defocusDv);
    }
};

#endif //CAMERA_H