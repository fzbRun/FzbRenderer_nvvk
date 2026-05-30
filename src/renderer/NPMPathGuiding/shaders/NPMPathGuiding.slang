#include "common/Shader/shaderStructType.h"
#include "nvshaders/functions.h.slang"
#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/NPMPathGuiding/NPMPathGuidingShaderio.h"

[[vk::push_constant]] ConstantBuffer<NPMPathGuidingPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_NPMPG::eOutImage)]] RWTexture2D<float4> outImage;
[[vk::binding(StaticBindingPoints_NPMPG::eFlowerImage), vk::image_format("rgba8")]] Sampler2D<float4> flowerImage;
[[vk::binding(StaticBindingPoints_NPMPG::eInputTensor)]] RWStructuredBuffer<float> inputTensor;

[numthreads(16, 16, 1)]
[shader("compute")]
void computeMain_reversalColor(uint2 threadIndex: SV_DispatchThreadID) {
    if (threadIndex.x >= pushConst.screenSize.x || threadIndex.y >= pushConst.screenSize.y) return;

    if (pushConst.time == 0) {
        float4 color = flowerImage.SampleLevel((float2(threadIndex) + float2(0.5f)) / pushConst.screenSize, 0);
        float3 rgb = (color.xyz - float3(0.485, 0.456, 0.406)) / float3(0.229, 0.224, 0.225);

        uint rIndex = threadIndex.y * pushConst.screenSize.x + threadIndex.x;
        uint gIndex = rIndex + pushConst.screenSize.x * pushConst.screenSize.y;
        uint bIndex = gIndex + pushConst.screenSize.x * pushConst.screenSize.y;

        inputTensor[rIndex] = rgb.r;
        inputTensor[gIndex] = rgb.g;
        inputTensor[bIndex] = rgb.b;
    }
    else outImage[threadIndex] = flowerImage.SampleLevel((float2(threadIndex) + float2(0.5f)) / pushConst.screenSize, 0);
}