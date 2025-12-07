#ifndef MYPROJECT_SHADER_H
#define MYPROJECT_SHADER_H
#include <string>
#include <cstdint>
#include <shaderc/shaderc.hpp>

namespace gpu
{
  class IShader
  {
    public:
    std::string name;
    std::string path;
    std::string shaderModule;
    std::uintptr_t shaderBin;
    std::vector<std::uintptr_t> descriptorLayouts;
    ~IShader() = default;
    //std::uintptr_t virtual getShader();
    virtual std::uintptr_t getShader(const std::string& filePath,
                                    shaderc_shader_kind flag) =0;
  };
}


#endif //MYPROJECT_SHADER_H
