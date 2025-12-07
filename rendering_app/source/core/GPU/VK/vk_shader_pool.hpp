#ifndef Vk_SHADERMOUDULE_HPP
#define Vk_SHADERMOUDULE_HPP
#include <string>
#include <filesystem>
#include <unordered_map>
#include <vulkan/vulkan.h>
#include <cstdint>
#include "IShader.hpp"
#include "vk_context.hpp"

//spirv reflect -> compile and register dsl
namespace gpu
{
  class VkShaderPool : public IShader
  {
    public:
    VkShaderPool();
    ~VkShaderPool();
    VkShaderModule compile(std::string filePath, shaderc_shader_kind flag);
    //uintptr_t virtual getShader() override;
    std::uintptr_t getShader(const std::string& filePath ,shaderc_shader_kind flag) override;

    private:
    void compileShader(const std::string& shaderSource,
                       const std::string& outputFilePath,
                       shaderc_shader_kind flag) const;
    void updateBinPath(std::string filepath);

    private:
    VkShaderModule shader;
    std::unordered_map<std::string, VkShaderModule> shaderModules_;
    std::string shaderPath_;
    std::string binPath_;
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
  };
}


#endif
