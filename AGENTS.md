# AGENTS.md

FzbRenderer 1.0 —— 基于 nvpro_core2 的 Vulkan 渲染器集合（光栅化 / 路径追踪 / Real-time Path Guiding / 降噪）。
**一个可执行文件，通过 `rendererInfo/rendererInfo.xml` 在十余个渲染器之间切换。**
本文件是给 AI 代理的项目地图与工作约定。动手前先读这里，不要靠猜。

## 目录地图

| 路径 | 作用 |
|---|---|
| `src/main.cpp` | 入口 |
| `src/common/Application/` | 应用框架：解析 `rendererInfo.xml`、开窗、主循环、ImGui |
| `src/common/Scene/` `Mesh/` `Material/` `Image/` `Buffer/` | 场景图与 GPU 资源 |
| `src/common/RayTracingManager/` `RasterizationManager/` `Semaphore/` | 加速结构、光栅化、同步 |
| `src/common/CUDA/` | CUDA 互操作（光注入、去噪等） |
| `src/common/Shader/` | Slang 编译、shader 结构体共享头 |
| `src/feature/` | 可复用特性：`ShadowMap` `TAA` `SVGF` `FLIP` `PathTracing` `LightInject` `SceneDivision` |
| `src/renderer/<Name>/` | 每个渲染器一个目录，含 `.cpp/.h`、`shaders/`、可选 `README.md` |
| `rendererInfo/rendererInfo.xml` | **唯一入口配置**：选渲染器、分辨率、场景、各渲染器参数 |
| `resources/<scene>/sceneInfo.xml` | 场景描述：相机、材质、灯光、mesh、实例 |
| `third_party/` | vendored：`nvpro_core2`、Slang。**只读** |

## 构建

生成器 **Visual Studio 17 2022**，C++17 + CUDA，构建目录 `build/`，产物 `_bin/{Debug,Release}/FzbRenderer_nvvk.exe`。

```powershell
cmake --build build --config Release --target FzbRenderer_nvvk   # 日常：增量构建
cmake --build build --config Debug                                # 需要调试/断点
cmake -S . -B build                                               # 重新配置（参数见 .vscode/settings.json）
```

依赖在**仓库外**，不要假设 `third_party/` 里都有：

- Vulkan SDK / assimp / pugixml / meshoptimizer：`C:/Users/fangzanbo/Desktop/render/third_party/...`
- TensorRT 10.16.1.11：`C:/Users/fangzanbo/Desktop/AI/TensorRT/...`
- Slang：仓库内 `third_party/Slang-windows-x86_64-2025.13.1`
- 权威参数清单：`.vscode/settings.json` 的 `cmake.configureArgs`

注意：

- **不要用 Rebuild Solution**（全量重编 + CUDA，非常慢）。工程已设 `DisableFastUpToDateCheck`，Build 就是增量。
- `src/CMakeLists.txt` 是历史遗留，根 `CMakeLists.txt` 直接 `GLOB_RECURSE src/*.*` 收集源码，改它没有效果。
- 新增文件不用改 CMake，重新配置即可。

## 运行与验证

exe 通过编译期宏 `TARGET_EXE_TO_SOURCE_DIRECTORY` 回到**源码目录**读取 `rendererInfo/` 和 `resources/`（见 `src/common/Application/Application.cpp:29,46`）。

推论：

- 改 `rendererInfo.xml`、换场景、**改 shader 都不需要重新编译**，直接跑 exe。
- 但 exe 必须保持在 `_bin/Release`（或保持到源码目录的相对位置），否则找不到 xml。

最小验证回路：

1. 改 `rendererInfo.xml` 的 `<renderer type="...">` 与 `<sceneXML path="...">`
2. 运行 exe，看 `_bin/Release/log_FzbRenderer_nvvk.txt`、stdout/stderr（**先看 `[SlangCompile]` 行**）
3. 渲染器内置的 `saveFrame` / 截图，对比 `_bin/Release/*.png`

- Validation layer 由 `nvvk::ValidationSettings` 配置（`Application.cpp`）。
- **Aftermath 默认关闭**（`USE_NSIGHT_AFTERMATH` 被注释）；需要 GPU 崩溃定位时打开，DLL 已在 `_bin`。
- 新产生的日志/截图放 `build/logs/` 或 `_bin/Release/`，**不要污染仓库根**。

## 新增一个渲染器：四处都要改

1. `src/renderer/<Name>/` 下实现类，继承 `FzbRenderer::Renderer`
2. `src/renderer/Renderer.cpp`：`#include` → `enum FzbRendererType` → `RendererTypeMap` → `createRenderer` 的 `switch`
3. `rendererInfo.xml` 里加 `<renderer type="<Name>">` 分支解析参数
4. CMake 不用改

`RendererTypeMap` 的字符串就是 xml 里 `type` 的合法取值（`Forward` `Deferred` `PathTracing` `PathTracing_soft` `VXPG` `LPV` `VRPG` `VRPG2` `PathTracing_KPCNNDenoising` `ReSTIR_GI` `StochasticLightcuts` `NeuralTextureCompression` `Mesh` `ZhiHuCode`）。

## Shader 管线（最容易踩的坑）

- **运行时编译**：`src/common/Shader/Shader.cpp` 用 `Application::slangCompiler.compileFile()` 从源码编译，结果缓存成同名 `.spv`（挨着 `.slang`）。
- 缓存以 **mtime 比较**失效：`.spv` 比 `.slang` 新就直接用缓存。
  → **改完 shader 没效果 / 行为还是旧的：先删掉对应 `.spv` 再跑。**
- 编译失败只 `LOGE` 打日志、不中断；程序会继续跑（可能用旧 spv 或空 module）。查问题时**必须看日志里的 `[SlangCompile]` 与 `Error compiling` 行**。
- C++ 与 Slang 共享结构体：`src/common/Shader/shaderio.h`、`shaderStructType.h` 以及各渲染器的 `*Shaderio.h`（如 `src/renderer/VRPG2/VRPG2Shaderio.h`）。
  **加字段必须两边同时改**，并注意 std430 / vec3 对齐。
- `src/common/Shader/nvvk/spv/*.slang.h` 是构建产物（`sky_simple`、`tonemapper`），**不要手改**。

## 编码坑（重要）

`src` 下约 **73 / 316** 个源文件是 **GBK** 编码（如 `src/main.cpp`、`src/renderer/Renderer.cpp`、`src/common/Scene/Scene.h`、`src/common/CUDA/*`、`src/feature/*` 多数文件）。

- 只读工具对这些文件会报 `invalid UTF-8`；改用 grep 定位行号，或按字节读取。
- **绝不要整文件重写这些文件**——中文注释会被写成乱码。只做最小范围的替换，或先确认编码再操作。
- 新增文件用 UTF-8。

## 版本控制现状（先看这条再动手）

- 当前分支 `RT_05`，工作区有未提交改动。
- **`src/renderer/VRPG2/` 和 `src/renderer/LPV/` 整个目录还没进 git（untracked）**，`_tools/` 下还有 30 多个手工 zip 快照。
  → 这意味着这些代码**没有 diff 基线、无法回退、无法 review**。在动 VRPG2 / LPV 之前，先让用户提交一次基线。
- `build/`、`_bin/`、`*.exe`、`*.dll` 已被 `.gitignore` 覆盖；`_tools/` 下的手工 zip 快照与 `.err/.out` 也已排除。
- `.gitignore` 只对**未跟踪**文件生效；已跟踪的 `*.bak`、`__pycache__/*.pyc`、`*.spv` 需要 `git rm --cached` 才会撤出索引。

## 工作协议（给 AI 代理）

1. **先定范围再动手**：先读 `Renderer.cpp` 注册表 + 目标渲染器目录 + 它的 `README.md`，列出本次要看的文件清单，再逐个处理。不要"通读全仓库"。
2. **每个改动单元都要可验证**：能在 `_bin/Release` 跑起来、能看日志/截图，再进入下一个改动。
3. **最小改动**：不顺手重构、不改命名、不改 public 接口、不调格式。
4. **不要新建 `.bak` / `.zip` 快照**。回退用 git；如果没基线，先提醒用户提交。
5. **输出审查结论的格式**：`文件:行` + 触发条件 + 为什么现有代码会失败 + 建议改法 + 置信度。先说结论，不要复述代码。
6. 不确定的结论标成"待确认"，不要当事实上报。

## 禁区

- 不改 `third_party/`（vendored 依赖）
- 不改 `_downloaded_resources/`、`build/` 里的历史日志
- 不做与当前任务无关的格式化 / 重命名 / 批量替换
- 不提交 `_bin/*.png`、`build/*.log`、`*.bak`、`*.zip`
