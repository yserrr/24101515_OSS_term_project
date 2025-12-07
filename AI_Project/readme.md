# AI Project

![Vulkan](https://img.shields.io/badge/Vulkan-Compute-orange)
![C++](https://img.shields.io/badge/C++-17-blue)


프로젝트는 GGML https://github.com/ggml-org/ggml  을 포크하여 시작했습니다

`v_` 접두사는 `ggml`에서 가져온 함수/변수임을 의미합니다.

프로젝트 대부분은 ggml을 참조하여 변형하면서 구현하고 있습니다. 

본 프로젝트의 주된 목표는 **개인 학습** 입니다.

부족한 부분이 있을 수 있으며, 일부 경우 기존보다 성능이 떨어질 수도 있습니다.  

---
## 🧠 Project Overview

AI Project는 **Vulkan Compute Shader**를 활용하여 AI 구조를 구현하는 프로젝트입니다.

- GGML의 Vulkan backend를 fork하여 시작하였습니다.
- CPU fallback 없이, 단일 벡엔드 비동기 구조를 목표로합니다.
- OP fallback 없는 구조 목표로 구현합니다. 
- Vulkan을 통해서 GPU 벤더와 무관하게 학습가능한 환경을 만드는 것이 목표입니다.
- 구체적 구현
  - source/compute
  - source/compute/vk
  - source/compute/vk_kernels 폴더에 구현됩니다.
  

Vulkan을 활용하여 AI 학습 연산 환경을 만드는 것을 목표로 합니다.

---

## 🔧 Build

프로젝트는 **CMake**를 사용하여 빌드 가능합니다:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

✅ 실행 예시
이후 test 폴더 내의 cpp파일은 실행가능하도록 생성됩니다. 
cmake build 이후 사용할 수 있습니다.

```
./test/compute    # 기본 Vulkan 연산 테스트
./test/mnist_train    # MNIST 데이터셋 학습 실행
```
코드 compute.cpp, mnist_train.cpp을 참고해주세요

## 🎯 Project Goal

- Vulkan 기반 AI 학습 환경 구축
- 단일 backend 구조로 모든 연산 처리
- OP fallback 없이 모든 연산을 Vulkan shader에서 처리
- MNIST 등 기본 데이터셋 학습 구조 구현 및 테스트
---
## 🚧 Current Progress
SINGLE VULKAN BACKEND
- nnist train 예제가 구현되었습니다.
- Vulkan backend를 단일 구조로 진행하고 있습니다. cpu 연산을 제거하여, vk에서 지원하지 않는 연산이 많습니다. 이를 구현하고 있습니다.
- 기존 vk에서 없는 연산을 추가하고 있습니다.

## ⚙️ Requirements
- C++ Compiler: C++17 
- Dependencies:
- CMake
- Vulkan
- Pybind(python vision dataset을 사용한다면 필요합니다.) 



## Run Example : 
[mnist_train.cpp](source/test/mnist_train.cpp)
```
v_vulkan: Found 1 Vulkan devices:
v_vulkan: 0 = NVIDIA GeForce RTX 2060 (NVIDIA) | uma: 0 | fp16: 1 | bf16: 1 | warp size: 32 | shared memory: 49152 | int
 dot: 1 | matrix cores: NV_coopmat2
=== GRAPH ===
n_nodes = 17
 -   0: [   500,    10,     1]             NONE x
 -   1: [   784,   500,     1]             NONE x
 -   2: [   500, 15000,     1]          MUL_MAT
.
.
.
n_leafs = 2
 -   0: [   784, 15000]     NONE           images
 -   1: [    10, 15000]     NONE   labels_one_hot
========================================
node name: COUNT_EQUAL
backword compute called
node name: ARGMAX
backword compute called
.
.
.

mat_mul called
 node name: NONE
backword compute called
node name: NONE
backword compute called
main: epoch 0001/0060:
train: [██▋     ] data=0015000/0045000 loss=2.35257±nan acc=11.91±0.26% t=00:00:00 ETA=00:00:00
train: [█████▎  ] data=0030000/0045000 loss=2.34510±0.00747 acc=17.43±0.22% t=00:00:00 ETA=00:00:00
train: [███████▉] data=004500[main.cpp](../rendering_app/main.cpp)0/0045000 loss=2.33782±0.00846 acc=24.70±0.20% t=00:00:00 ETA=00:00:00
val:   [████████] data=0060000/0045000 loss=2.30735±nan acc=45.43±0.41% t=00:00:00 ETA=00:00:00
main: epoch 0002/0060:
train: [██▋     ] data=0015000/0045000 loss=2.30679±nan acc=45.95±0.41% t=00:00:00 ETA=00:00:00
train: [█████▎  ] data=0030000/0045000 loss=2.29847±0.00833 acc=46.45±0.29% t=00:00:00 ETA=00:00:00
train: [███████▉] data=0045000/0045000 loss=2.28946±0.01021 acc=46.29±0.24% t=00:00:00 ETA=00:00:00
val:   [████████] data=0060000/0045000 loss=2.25014±nan acc=44.61±0.41% t=00:00:00 ETA=00:00:00
.
.
.

main: epoch 0060/0060:
train: [██▋     ] data=0015000/0045000 loss=0.24977±nan acc=94.49±0.19% t=00:00:00 ETA=00:00:00
train: [█████▎  ] data=0030000/0045000 loss=0.24977±nan acc=94.49±0.19% t=00:00:00 ETA=00:00:00
train: [█████▎  ] data=0030000/0045000 loss=0.25663±0.00686 acc=94.32±0.13% t=00:00:00 ETA=00:00:00
train: [███████▉] data=0045000/0045000 loss=0.25767±0.00410 acc=94.26±0.11% t=00:00:00 ETA=00:00:00
val:   [████████] data=0060000/0045000 loss=0.27239±nan acc=94.00±0.19% t=00:00:00 ETA=00:00:00
main: training took 00:00:08

```
## TODO: 
- 향후 목표:
    - MPI를 활용한 분산 학습 환경 구축
    - class-313 Computing Farm 만들기
    - CNN, GAN, Diffusion,Attension, MOE 등 고급 구조 설계 및 실험
    - 병렬 프로그래밍 및 CPU - GPU Overlapping 구조 실험
    - SPIR-V Jit compile 구조 만들어보기



  
