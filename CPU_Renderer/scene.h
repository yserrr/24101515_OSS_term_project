#include "bvh.h"
#include "quad.h"
#include "camera.h"
#include "sphere.h"
#include "material.h"
#include "ray_scatter.h"
#include "transformation.h"

void sampleShpere() {
    HittableList world;

    auto checker = std::make_shared<CheckerTexture>(0.32, Color(.2, .3, .1), Color(.9, .9, .9));
    world.add(std::make_shared<Sphere>(Point3(0,-1000,0), 1000, std::make_shared<Lambertian>(checker)));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = randomDouble();
            Point3 center(a + 0.9*randomDouble(), 0.2, b + 0.9*randomDouble());

            if ((center - Point3(4, 0.2, 0)).length() > 0.9) {
                std::shared_ptr<Material> sphereMaterial;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo    = Color::random() * Color::random();
                    sphereMaterial = std::make_shared<Lambertian>(albedo);
                    auto center2   = center + Vec3(0, randomDouble(0,.5), 0);
                    world.add(std::make_shared<Sphere>(center, center2, 0.2, sphereMaterial));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo    = Color::random(0.5, 1);
                    auto fuzz      = randomDouble(0, 0.5);
                    sphereMaterial = std::make_shared<Metal>(albedo, fuzz);
                    world.add(std::make_shared<Sphere>(center, 0.2, sphereMaterial));
                } else {
                    // glass
                    sphereMaterial = std::make_shared<Dielectric>(1.5);
                    world.add(std:: make_shared<Sphere>(center, 0.2, sphereMaterial));
                }
            }
        }
    }

    auto material1 = std::make_shared<Dielectric>(1.5);
    world.add(std::make_shared<Sphere>(Point3(0, 1, 0), 1.0, material1));

    auto material2 = std::make_shared<Lambertian>(Color(0.4, 0.2, 0.1));
    world.add(std::make_shared<Sphere>(Point3(-4, 1, 0), 1.0, material2));

    auto material3 = std::make_shared<Metal>(Color(0.7, 0.6, 0.5), 0.0);
    world.add(std::make_shared<Sphere>(Point3(4, 1, 0), 1.0, material3));

    BaseCamera cam;
    world = HittableList(std:: make_shared<BvhNode>(world));


    cam.aspectRatio      = 16.0 / 9.0;
    cam.imageWidth       = 400;
    cam.samplesPerPixel  = 10;
    cam.maxDepth         = 4;
    cam.background       = Color(0.70, 0.80, 1.00);
    
    cam.vfov     = 20;
    cam.lookfrom = Point3(13,2,3);
    cam.lookat   = Point3(0,0,0);
    cam.vup      = Vec3(0,1,0);

    cam.defocusAngle = 10.0;
    cam.focalLength  = 3.4;

    cam.render(world);
}

//change the scene angle
void scene2() {
    HittableList world;

    auto checker = std::make_shared<CheckerTexture>(0.32, Color(.2, .3, .1), Color(.9, .9, .9));

    world.add(std::make_shared<Sphere>(Point3(0,-10, 0), 10, std::make_shared<Lambertian>(checker)));
    world.add(std::make_shared<Sphere>(Point3(0, 10, 0), 10, std::make_shared<Lambertian>(checker)));

    BaseCamera cam;

    cam.aspectRatio      = 16.0 / 9.0;
    cam.imageWidth       = 400;
    cam.samplesPerPixel  = 100;
    cam.maxDepth         = 50;

    cam.vfov             = 20;
    cam.lookfrom         = Point3(13,2,3);
    cam.lookat           = Point3(0,0,0);
    cam.vup              = Vec3(0,1,0);
    cam.background       = Color(0.70, 0.80, 1.00);
    
    cam.defocusAngle = 0;

    cam.render(world);
}

void earth(){ 
    auto earth_texture = std::make_shared<ImageTexture>("earthmap.jpg");
    auto earth_surface = std::make_shared<Lambertian>(earth_texture);
    auto globe         = std::make_shared<Sphere>(Point3(0,0,0), 2, earth_surface);

    BaseCamera cam;

    cam.aspectRatio      = 16.0 / 9.0;
    cam.imageWidth       = 400;
    cam.samplesPerPixel  = 100;
    cam.maxDepth         = 50;

    cam.vfov             = 20;
    cam.lookfrom         = Point3(0,0,12);
    cam.lookat           = Point3(0,0,0);
    cam.vup              = Vec3(0,1,0);
    cam.background       = Color(0.70, 0.80, 1.00);
    
    cam.defocusAngle = 0;

    cam.render(HittableList(globe));
}
/*
void Perlin() {
    HittableList world;

    auto pertext = std::make_shared<NoiseTexture>(1);
    world.add(std::make_shared<Sphere>(Point3(0,-1000,0), 1000, std::make_shared<Lambertian>(pertext)));
    world.add(std::make_shared<Sphere>(Point3(0,2,0), 2, std::  make_shared<Lambertian>(pertext)));

    CameraManager cam;
    cam.aspectRatio      = 16.0 / 9.0;
    cam.imageWidth       = 400;
    cam.samplesPerPixel  = 100;
    cam.maxDepth         = 50;

    cam.vfov     = 20;
    cam.lookfrom = Point3(13,2,3);
    cam.lookat   = Point3(0,0,0);
    cam.vup      = Vec3(0,1,0);

    cam.defocusAngle = 0;

    cam.render(world);

}
*/

void Quad_render() {
    HittableList world;

    // Materials
    auto left_red     = std::make_shared<Lambertian>(Color(1.0, 0.2, 0.2));
    auto back_green   = std::make_shared<Lambertian>(Color(0.2, 1.0, 0.2));
    auto right_blue   = std::make_shared<Lambertian>(Color(0.2, 0.2, 1.0));
    auto upper_orange = std::make_shared<Lambertian>(Color(1.0, 0.5, 0.0));
    auto lower_teal   = std::make_shared<Lambertian>(Color(0.2, 0.8, 0.8));

    // Quads
    world.add(std::make_shared<Quad>(Point3(-3,-2, 5), Vec3(0, 0,-4), Vec3(0, 4, 0), left_red));
    world.add(std::make_shared<Quad>(Point3(-2,-2, 0), Vec3(4, 0, 0), Vec3(0, 4, 0), back_green));
    world.add(std::make_shared<Quad>(Point3( 3,-2, 1), Vec3(0, 0, 4), Vec3(0, 4, 0), right_blue));
    world.add(std::make_shared<Quad>(Point3(-2, 3, 1), Vec3(4, 0, 0), Vec3(0, 0, 4), upper_orange));
    world.add(std::make_shared<Quad>(Point3(-2,-3, 5), Vec3(4, 0, 0), Vec3(0, 0,-4), lower_teal));

    BaseCamera cam;

    cam.aspectRatio      = 16.0 / 9.0;
    cam.imageWidth       = 400;
    cam.samplesPerPixel  = 100;
    cam.maxDepth         = 50;
    cam.background        = Color(0.70, 0.80, 1.00);
    
    cam.vfov     = 80;
    cam.lookfrom = Point3(0,0,9);
    cam.lookat   = Point3(0,0,0);
    cam.vup      = Vec3(0,1,0);

    cam.defocusAngle = 0;

    cam.render(world);
}
/*
void light () {
    HittableList world;

    auto pertext = std::make_shared<NoiseTexture>(0);
    world.add(std::make_shared<Sphere>(Point3(0,-1000,0), 1000, std:: make_shared<Lambertian>(pertext)));
    world.add(std::make_shared<Sphere>(Point3(0,2,0), 2,        std:: make_shared<Lambertian>(pertext)));

    auto difflight = std::make_shared<DiffuseLight>(Color(4,4,4));
    world.add(std::make_shared<Quad>  (Point3(3,1,-2), Vec3(2,0,0), Vec3(0,2,0), difflight));
    world.add(std::make_shared<Sphere>(Point3(0,7,0), 2, difflight));
    
    CameraManager cam;
    cam.aspectRatio      = 16.0 / 9.0;
    cam.imageWidth       = 400;
    cam.samplesPerPixel  = 100;
    cam.maxDepth         = 50;
    cam.background       = Color(0,0,0);

    cam.vfov             = 20;
    cam.lookfrom         = Point3(26,3,6);
    cam.lookat           = Point3(0,2,0);
    cam.vup              = Vec3(0,1,0);

    cam.defocusAngle     = 0;

    cam.render(world);
}
*/

void boxScene(){

        HittableList world;
    
        auto red    = std::make_shared<Lambertian>   (Color(.65, .05, .05));
        auto white  = std::make_shared<Lambertian>   (Color(.73, .73, .73));
        auto green  = std::make_shared<Lambertian>   (Color(.12, .45, .15));
        auto light  = std::make_shared<DiffuseLight> (Color(15, 15, 15));
        auto light2 = std::make_shared<DiffuseLight> (Color(3, 3, 3));
    
        world.add(std::make_shared<Quad>(Point3(555,0,0), Vec3(0,555,0), Vec3(0,0,555), green));
        world.add(std::make_shared<Quad>(Point3(555,0,0), Vec3(0,555,0), Vec3(0,0,555), green));
        world.add(std::make_shared<Quad>(Point3(0,0,0),   Vec3(0,555,0), Vec3(0,0,555), red));
        world.add(std::make_shared<Quad>(Point3(343, 554, 332), Vec3(-130,0,0), Vec3(0,0,-105), light));
        world.add(std::make_shared<Quad>(Point3(0,0,0), Vec3(555,0,0), Vec3(0,0,555), white));
        world.add(std::make_shared<Quad>(Point3(555,555,555), Vec3(-555,0,0), Vec3(0,0,-555), white));
        world.add(std::make_shared<Quad>(Point3(0,0,555), Vec3(555,0,0), Vec3(0,555,0), white));
    
        std::shared_ptr<Hittable> box1 = box(Point3(0,0,0), Point3(165,330,165), white);
        box1 = std::make_shared<rotateY>  (box1, 15);
        box1 = std::make_shared<Translate>(box1, Vec3(265,0,295));
        world.add(box1);
    
        std::shared_ptr<Hittable> box2 = box(Point3(0,0,0), Point3(165,165,165), white);
        box2 = std::make_shared<rotateY>(box2, -18);
        box2 = std::make_shared<Translate>(box2, Vec3(130,0,65));
        world.add(box2);
    
        std::shared_ptr<Hittable> box3 = box(Point3(0,0,0), Point3(30,30,30), light2);
        box3 = std::make_shared<rotateY>(box3, 30);
        box3 = std::make_shared<Translate>(box3, Vec3(0,0,10));
        world.add(box3);
        
        BaseCamera cam;
    
        cam.aspectRatio      = 16.0 / 9.0;
        cam.imageWidth       = 400;
        cam.samplesPerPixel  = 10;
        cam.maxDepth         = 100;
        cam.background        = Color(0,0,0);
    
        cam.vfov     = 40;
        cam.lookfrom = Point3(278, 278, -800);
        cam.lookat   = Point3(278, 278, 0);
        cam.vup      = Vec3(0,1,0);
    
        cam.defocusAngle = 0;
    
        cam.render(world);    
}

void boxScene2(){

        HittableList world;
        auto red    = std::make_shared<Lambertian>   (Color(.65, .05, .05));
        auto white  = std::make_shared<Lambertian>   (Color(.73, .73, .73));
        auto green  = std::make_shared<Lambertian>   (Color(.12, .45, .15));
        auto light  = std::make_shared<DiffuseLight> (Color( 15,  15,  15));
        auto light2 = std::make_shared<DiffuseLight> (Color(  1,   3,   3));
        auto light3 = std::make_shared<DiffuseLight> (Color(  3,   2,   1));
        world.add(std::make_shared<Quad>(Point3(555,0,0),       Vec3(0,555,0),  Vec3(0,0,555), white));
        world.add(std::make_shared<Quad>(Point3(0,0,0),         Vec3(0,555,0),  Vec3(0,0,555), white));
        world.add(std::make_shared<Quad>(Point3(343, 554, 332), Vec3(-130,0,0), Vec3(0,0,-105),light));
        world.add(std::make_shared<Quad>(Point3(0,0,0),         Vec3(555,0,0),  Vec3(0,0,555), green));
        world.add(std::make_shared<Quad>(Point3(555,555,555),   Vec3(-555,0,0), Vec3(0,0,-555),white));
        world.add(std::make_shared<Quad>(Point3(0,0,555),       Vec3(555,0,0),  Vec3(0,555,0), white));
    
        std::shared_ptr<Hittable> box1 = box(Point3(0,0,0), Point3(165,330,165), white);
        box1 = std::make_shared<rotateY>  (box1, 15);
        box1 = std::make_shared<Translate>(box1, Vec3(265,0,295));
        world.add(box1);
    
        std::shared_ptr<Hittable> box2 = box(Point3(0,0,0), Point3(165,165,165), light3);
        box2 = std::make_shared<rotateY>  (box2, -18);
        box2 = std::make_shared<Translate>(box2, Vec3(130,0,65));
        world.add(box2);
    
        std::shared_ptr<Hittable> box3 = box(Point3(0,0,0), Point3(30,30,30), light2);
        box3 = std::make_shared<rotateY>  (box3, 30);
        box3 = std::make_shared<Translate>(box3, Vec3(0,0,10));
        world.add(box3);
        
        BaseCamera cam;
    
        cam.aspectRatio      = 16.0 / 9.0;
        cam.imageWidth       = 400 ;
        cam.samplesPerPixel  = 150;
        cam.maxDepth         = 150;
        cam.background       = Color(0,0,0);
    
        cam.vfov     = 60;
        cam.lookfrom = Point3(278, 600, -800);
        cam.lookat   = Point3(278, 278, 0);
        cam.vup      = Vec3(0,1,0);
    
        cam.defocusAngle = 2;
    
        cam.render(world);    
}

void yard(){

        HittableList world;
        auto red    = std::make_shared<Lambertian>   (Color(.65, .05, .05));
        auto white  = std::make_shared<Lambertian>   (Color(.73, .73, .73));
        auto green  = std::make_shared<Lambertian>   (Color(.12, .45, .15));
        auto light  = std::make_shared<DiffuseLight> (Color( 15,  15,  15));
        auto light2 = std::make_shared<DiffuseLight> (Color(  1,   3,   3));
        auto light3 = std::make_shared<DiffuseLight> (Color(  4,   1,   1));

        world.add(std::make_shared<Quad>(Point3(0,0,0), Vec3(10000,0,0),  Vec3(0,0,10000), green));
        
        std::shared_ptr<Hittable> box1 = box(Point3(0,0,0), Point3(165,330,165), white);
        box1 = std::make_shared<rotateY>  (box1, 15);
        box1 = std::make_shared<Translate>(box1, Vec3(265,0,295));
        world.add(box1);
    
        std::shared_ptr<Hittable> box2 = box(Point3(0,0,0), Point3(165,165,165), light3);
        box2 = std::make_shared<rotateY>  (box2, -18);
        box2 = std::make_shared<Translate>(box2, Vec3(130,0,65));
        world.add(box2);
    
        std::shared_ptr<Hittable> box3 = box(Point3(0,0,0), Point3(30,30,30), light2);

        box3 = std::make_shared<rotateY>  (box3, 30);
        box3 = std::make_shared<Translate>(box3, Vec3(0,0,10));
        world.add(box3);
        std::shared_ptr<Hittable> box4 = box(Point3(-INF,INF,INF), Point3(INF,INF,INF), red);
        world.add(box4) ; 
        BaseCamera cam;
    
        cam.aspectRatio      = 16.0 / 9.0;
        cam.imageWidth       = 300 ;
        cam.samplesPerPixel  = 150;
        cam.maxDepth         = 150;
        cam.background       = Color(0,0,0.001);
    
        cam.vfov     = 40;
        cam.lookfrom = Point3(278, 600, -800);
        cam.lookat   = Point3(278, 278, 0);
        cam.vup      = Vec3(0,1,0);
    
        cam.defocusAngle = 0;
    
        cam.render(world);    
}

void smokeBoxScene(){

        HittableList world;
    
        auto red   = std::make_shared<Lambertian>   (Color(.65, .05, .05));
        auto white = std::make_shared<Lambertian>   (Color(.73, .73, .73));
        auto green = std::make_shared<Lambertian>   (Color(.12, .45, .15));
        auto light = std::make_shared<DiffuseLight> (Color(15, 15, 15));
    
        world.add(std::make_shared<Quad>(Point3(555,0,0), Vec3(0,555,0), Vec3(0,0,555), green));
        world.add(std::make_shared<Quad>(Point3(555,0,0), Vec3(0,555,0), Vec3(0,0,555), green));
        world.add(std::make_shared<Quad>(Point3(0,0,0),   Vec3(0,555,0), Vec3(0,0,555), red));
        world.add(std::make_shared<Quad>(Point3(343, 554, 332), Vec3(-130,0,0), Vec3(0,0,-105), light));
        world.add(std::make_shared<Quad>(Point3(0,0,0), Vec3(555,0,0), Vec3(0,0,555), white));
        world.add(std::make_shared<Quad>(Point3(555,555,555), Vec3(-555,0,0), Vec3(0,0,-555), white));
        world.add(std::make_shared<Quad>(Point3(0,0,555), Vec3(555,0,0), Vec3(0,555,0), white));
    
        std::shared_ptr<Hittable> box1 = box(Point3(0,0,0), Point3(165,330,165), white);
        box1 = std::make_shared<rotateY>  (box1, 15);
        box1 = std::make_shared<Translate>(box1, Vec3(265,0,295));
        world.add(box1);
    
        std::shared_ptr<Hittable> box2 = box(Point3(0,0,0), Point3(165,165,165), white);
        box2 = std::make_shared<rotateY>(box2, -18);
        box2 = std::make_shared<Translate>(box2, Vec3(130,0,65));
        world.add(box2);
    
        BaseCamera cam;
    
        cam.aspectRatio      = 16.0 / 9.0;
        cam.imageWidth       = 400;
        cam.samplesPerPixel  = 100;
        cam.maxDepth         = 50;
        cam.background        = Color(0,0,0);
    
        cam.vfov     = 40;
        cam.lookfrom = Point3(278, 278, -800);
        cam.lookat   = Point3(278, 278, 0);
        cam.vup      = Vec3(0,1,0);
    
        cam.defocusAngle = 0;
    
        world.add(std::make_shared<constantMedium>(box2, 0.01, Color(1,1,1)));
        world.add(std::make_shared<constantMedium>(box1, 0.01, Color(0,0,0)));

        cam.render(world);    
}

void circuit(){
    
    BaseCamera cam; 
    HittableList world; 
    world.add(std::make_shared<Sphere>(Point3(0,-100.5,-1), 100, std::make_shared<Metal>(Point3(1,1,1),1)));
    
    world.add(std::make_shared<Sphere>(Point3(0,0,-1), 0.5, std::make_shared<Metal>(Point3(1,1,1),1)));
    
    cam.background= Point3(0.5,0.7,0.9);
    cam.aspectRatio = 1;

    cam.imageWidth       = 256;
    cam.samplesPerPixel  = 20;
    cam.maxDepth         = 5;
        
    cam.lookfrom= Point3(0,0,0); 
    cam.lookat  = Point3(0,0,-1); 
    cam.focalLength = 100;
    cam.render(world);
}
