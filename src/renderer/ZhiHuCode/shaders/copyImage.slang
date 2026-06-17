#include "common/Shader/shaderStructType.h"
#include "nvshaders/functions.h.slang"
#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/ZhiHuCode/ZhiHuCodeShaderio.h"

[[vk::push_constant]] ConstantBuffer<ZhiHuCodePushConstant> pushConst;
[[vk::binding(StaticBindingPoints_ZhiHuCode::eOutImage)]] RWTexture2D<float4> outImage;
[[vk::binding(StaticBindingPoints_ZhiHuCode::eFlowerImage), vk::image_format("rgba8")]] Sampler2D<float4> flowerImage;

[numthreads(16, 16, 1)]
[shader("compute")]
void computeMain_copyImage(uint2 threadIndex: SV_DispatchThreadID) {
    if (threadIndex.x >= pushConst.screenSize.x || threadIndex.y >= pushConst.screenSize.y) return;

    uint2 flowerImageSize;
    flowerImage.GetDimensions(flowerImageSize.x, flowerImageSize.y);

    float2 uv = (float2(threadIndex) + 0.5f) / pushConst.screenSize;
    outImage[threadIndex] = flowerImage.SampleLevel(uv, 0);
}