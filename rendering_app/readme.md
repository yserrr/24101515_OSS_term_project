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

# Frame Graph  Dependency:
naive하게 frame graph가 구현되어있습니다.
- Dynamic Rendering을 사용하여 pass가 아닌 frame image단위로 의존성을 추적합니다.
- READ -> WRITE 리소스의 경우, 자동으로 resource barrier를 삽입합니다. 
- last_writer가 존재한다면,WW에 맞춰서barrier를 삽입합니다. 
- write한 리소스는 명시적으로 RenderPass가 등록되어있지 않다면, 자동으로 No Clear로 pass가 삽입됩니다.

--- 
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
