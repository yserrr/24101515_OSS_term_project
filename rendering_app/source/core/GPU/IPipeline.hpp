//
// Created by dlwog on 25. 10. 22..
//

#ifndef MYPROJECT_IPIPELINE_H
#define MYPROJECT_IPIPELINE_H
#include <cstdint>
#include "util/cast.hpp"
#include "IShader.hpp"

namespace gpu
{
  class IPipelineProgram
  {
    public :
    std::uintptr_t pipelineHandle;
    std::uintptr_t pipelineLayout = 0;
    std::uintptr_t vertShaderModule = 0;
    std::uintptr_t fragShaderModule = 0;
    std::uintptr_t* pColorAttachmentFormat = nullptr;
    //VertexType vertexType = VertexType::ALL;
    std::uint32_t topology = 3;
    std::uint32_t cullMode = 0x00000002;
    std::uint32_t frontFace = 1;

    bool operator==(const IPipelineProgram& other) const
    {
      //todo: imple color attachment bind
      return pipelineLayout == other.pipelineLayout &&
        vertShaderModule == other.vertShaderModule &&
        fragShaderModule == other.fragShaderModule &&
        topology == other.topology &&
        cullMode == other.cullMode &&
        frontFace == other.frontFace;
    }
  };
}


#endif //MYPROJECT_PIPELINE_H
