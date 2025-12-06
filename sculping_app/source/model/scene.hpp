#include <vector>
#include <memory>
#include "camera.hpp"
#include "light.hpp"
#include <model.hpp>
//class Scene {
//public:
//    Scene() = default;
//
//    void setCamera(std::shared_ptr<Camera> cam) {
//        camera = std::move(cam);
//    }
//
//    std::shared_ptr<Camera> getCamera() const {
//        return camera;
//    }
//
//    void addLight(std::shared_ptr<Light> light) {
//        lights.push_back(std::move(light));
//    }
//
//    const std::vector<std::shared_ptr<Light>>& getLights() const {
//        return lights;
//    }
//
//    Models& getModels() {
//        // return models;
//    }
//
//    const Models& getModels() const {
//        return models;
//    }
//
//private:
//    std::shared_ptr<Camera> camera;
//    std::vector<std::shared_ptr<Light>> lights;
//    //Models models;
//};
//