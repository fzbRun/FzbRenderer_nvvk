#include "common/Shader/shaderStructType.h"
#include "nvshaders/functions.h.slang"
#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/ZhiHuCode/ZhiHuCodeShaderio.h"

[[vk::push_constant]] ConstantBuffer<ZhiHuCodePushConstant> pushConst;
[[vk::binding(StaticBindingPoints_ZhiHuCode::eOutImage)]] RWTexture2D<float4> outImage;
[[vk::binding(StaticBindingPoints_ZhiHuCode::eFlowerImage), vk::image_format("rgba8")]] Sampler2D<float4> flowerImage;
[[vk::binding(StaticBindingPoints_ZhiHuCode::eInputBuffer)]] RWStructuredBuffer<float> inputBuffer;

[numthreads(16, 16, 1)]
[shader("compute")]
void computeMain_createInputBuffer(uint2 threadIndex: SV_DispatchThreadID) {
    if (threadIndex.x >= pushConst.screenSize.x || threadIndex.y >= pushConst.screenSize.y) return;

    float4 color = flowerImage.SampleLevel((float2(threadIndex) + float2(0.5f)) / pushConst.screenSize, 0);
    float3 rgb = (color.xyz - float3(0.485, 0.456, 0.406)) / float3(0.229, 0.224, 0.225);

    uint imageSize = pushConst.screenSize.x * pushConst.screenSize.y;
    uint rIndex = threadIndex.y * pushConst.screenSize.x + threadIndex.x;
    uint gIndex = rIndex + imageSize;
    uint bIndex = gIndex + imageSize;

    inputBuffer[rIndex] = rgb.r;
    inputBuffer[gIndex] = rgb.g;
    inputBuffer[bIndex] = rgb.b;
}