# FzbRenderer 1.0

<img src="./result/cover.png" alt="FzbRenderer_nvvk" style="width:100%; border-radius:6px; display:block;" />

## 项目简介

本项目基于 nvidia 的[nvpro_core2](https://github.com/nvpro-samples/nvpro_core2)进行开发，在[nvidia 的硬件光追教程](https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR)上进行修改和填充。
目前实现的功能有

- **xml 交互**：
  - 用户可以在 renderInfo/renderInfo.xml 中指定使用的渲染器以及相应参数
  - 用户可以在 resources/xxx/sceneInfo.xml 中指定 xxx 场景的信息，如相机参数、材质、光照、mesh 和实例(与[mitsuba 3](https://www.mitsuba-renderer.org/)相似)
- **可视化界面**：基于 nvpro_core2 带有的 ImGui 库
- **模型读取**：可以读取 gltf、glb 和 obj 文件
- **BSDF 材质**：diffuse、conductor、dielectric、roughConductor 和 roughDielectric
- **延时渲染器**: 传统的延迟渲染
- **路径追踪渲染器**: 传统的 PathTracing
  - 反射、折射
  - BSDF 重要性采样
  - NEE（均匀采样和球面矩形采样）

## Quick Start

### Prerequisites

- **[nvpro_core2](https://github.com/nvpro-samples/nvpro_core2)**：Vulkan helper classes and utilities
- **[Vulkan 1.4+](https://vulkan.lunarg.com/sdk/home)**: Compatible GPU and drivers
- **[Cmake](https://cmake.org/download/) 3.18+**
- **[pugixml](https://github.com/zeux/pugixml)**: xml parser
- **[assimp](https://github.com/assimp/assimp)**: obj loader

### 注意事项

如果没有指定地址，Cmake 编译时会自动下载 nvpro_core2 和 pugixml 到 thrid-party 文件夹下。
但是 vulkan SDK 和 assimp 需要手动下载并在 CMake 编译时指定地址。
推荐使用 VSCode 的 CMake Tool 插件。
编译文件将会在\_bin 文件夹下。

## 未来计划

- **SVO PathGuiding(正在进行)**
- **[Bounding Voxel Sampling PathGuiding](https://dl.acm.org/doi/10.1145/3658203)**
- **[ReSTIR](https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing)**
- **……**
