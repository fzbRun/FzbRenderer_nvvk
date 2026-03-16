#define NEE

#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "feature/LightInject/shaderio.h"
#include "feature/SceneDivision/RasterVoxelization/shaderio.h"

[[vk::push_constant]] ConstantBuffer<LightInjectPushConstant, ScalarDataLayout> pushConst;
[[vk::binding(StaticBindingPoints_LightInject::eVGB_LightInject)]] RWStructuredBuffer<VGBVoxelData, ScalarDataLayout> VGB;

