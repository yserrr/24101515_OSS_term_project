# CG-Rendering Project

![Vulkan](https://img.shields.io/badge/Vulkan-GRAPHICS-orange)
![C++](https://img.shields.io/badge/C++-17-blue)

**Project Overview**  
Rendering 구조와 설계를 공부하고 실험하기 위해 시작한 프로젝트입니다.

---

## ⚙️ Requirements

- **C++ Compiler:** C++17 이상 지원
- **Dependencies:**
  - CMake
  - Vulkan
  - spdlog
  - shaderc
  - KTX
  - GLFW
  - Assimp
  - ImGui
  - Spirv-reflect

---

## 🗂 Project Structure

프로젝트 구조는 다음과 같습니다:

![Project Structure](img.png)


---



# Frame Graph 

# Dependency:
naive한 frame graph가 구현되어있습니다.
각 Frame Pass는 다음으로 graph를 build합니다. 
- Dynamic Rendering을 사용하여 pass가 아닌 frame image단위로 의존성을 추적합니다.
- READ -> WRITE 리소스의 경우, 자동으로 resource barrier를 삽입합니다. 
- last_writer가 존재한다면,Write ->Write에 맞춰서barrier를 삽입합니다. 
- write한 리소스는 명시적으로 RenderPass가 등록되어있지 않다면, 자동으로 No Clear로 pass가 삽입됩니다.
--- 


# use example :

render pass를 추가할 수 있습니다.
```bash
pass->read__.push_back(renderTargetFilm_->bloomingExtractAttachment_.get());
    pass->write__.push_back(renderTargetFilm_->bloomingBlurAttachment_.get());
    pass->execute = [this, pass](gpu::CommandBuffer cmd)
    {
      gpu::cmdBindDescriptorSets(cmd,
                                 VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 pipeline_->pipelineLayout_h,
                                 0,
                                 1,
                                 &gpu::ctx__->pDescriptorAllocator->descriptorSets
                                 [frameIndex_],
                                 0,
                                 nullptr);
      gpu::cmdBeginRendering(cmd, pass);
      renderTargetFilm_->updateFrameConstant();
      pushFrameConstant(cmd);
      gpu::cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_->bloomingBlurWritePipeline__);
      pipeline_->cmdSetPolygonMode(cmd, pipeline_->polygonMode);
      vkCmdSetDepthTestEnable(cmd, pipeline_->depthTest);
      gpu::cmdSetViewports(cmd,
                           0.0,
                           0.0,
                           (float)gpu::ctx__->pSwapChainContext->extent__.width,
                           (float)gpu::ctx__->pSwapChainContext->extent__.height
                          );
      gpu::cmdDrawQuad(cmd);
      gpu::cmdEndRendering(cmd);
    };
    uploadPasses_.push_back(pass);
  }
``` 
## 🚀 Project Results

<div align="center">

<img src="img_1.png" alt="Result 1" width="400"/>
<img src="img_2.png" alt="Result 2" width="400"/>
<img src="img_3.png" alt="Result 3" width="400"/>

Frame View : 
![img_4.png](img_4.png)
MRT:
![img_5.png](img_5.png)
</div>

## 📌 Notes

- Vulkan 기반 Rendering 구조 실험 중심 프로젝트
- 학습 목적이며, 구조 이해 및 실습 위주로 구현
- 향후 최적화 및 다양한 그래픽 기능 확장 계획

