//
// Created by dlwog on 25. 10. 30..
//

#ifndef MYPROJECT_CPU_CONTEXT_HPP
#define MYPROJECT_CPU_CONTEXT_HPP
#include <vector>
#include <memory>
struct Model;

namespace cpu
{
  class Renderer;
  class ResourceManager;
  class Context
  {
    public:
    Renderer* pRenderer;
    ResourceManager* pResourceManager;
    std::vector<Model*> drawHandle_;
    //user
    //io

    bool debug = true;

  };
  extern std::unique_ptr<Context> ctx__;
}

#endif //MYPROJECT_CONTEXT_HPP