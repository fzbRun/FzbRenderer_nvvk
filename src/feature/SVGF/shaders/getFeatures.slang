#include "feature/SVGF/SVGFShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<SVGFPushConstant> pushConst;
[[vk::binding(BindingPoints_SVGF::eAlbedoImage)]] RWTexture2D<float4> albedoImage;
[[vk::binding(BindingPoints_SVGF::eDepthImage)]] RWTexture2D<float> depthImage;
[[vk::binding(BindingPoints_SVGF::eNormalImage)]] RWTexture2D<float4> normalImage;
[[vk::binding(BindingPoints_SVGF::eVelocityImage)]] RWTexture2D<float4> velocityImage;
[[vk::binding(BindingPoints_SVGF::eVertexInfoImage)]] RWTexture2D<float4> vertexInfoImage;
[[vk::binding(BindingPoints_SVGF::eResultImage)]] RWTexture2D<float4> renderImage;

[[vk::binding(BindingPoints_SVGF::eDepthGradientImage)]] RWTexture2D<float4> depthGradientImage;
[[vk::binding(BindingPoints_SVGF::eIrradianceImage)]] RWTexture2D<float4> irradianceImage;

[[vk::binding(BindingPoints_SVGF::eMoment1Image)]] RWTexture2D<float4> moment1Image;
[[vk::binding(BindingPoints_SVGF::eMoment2Image)]] RWTexture2D<float4> moment2Image;

[[vk::binding(BindingPoints_SVGF::eVarianceImage)]] RWTexture2D<float4> varianceImage[2];

[[vk::binding(BindingPoints_SVGF::eHistoryDepthImage)]] RWTexture2D<float4> historyDepthImage;
[[vk::binding(BindingPoints_SVGF::eHistoryNormalImage)]] RWTexture2D<float4> historyNormalImage;
[[vk::binding(BindingPoints_SVGF::eHistoryVertexInfoImage)]] RWTexture2D<float4> historyVertexInfoImage;

groupshared float groupDepths[33][33];
void getGroupDepths(uint2 pixelIndex, uint2 groupThreadIndex) {
    int2 threadGroupStartIndex = pixelIndex - groupThreadIndex;
    bool2 lastThreadGroup = threadGroupStartIndex + 32 > pushConst.screenSize;
    int2 maxGroupThread = 32;
    if (lastThreadGroup.x) maxGroupThread.x = pushConst.screenSize.x - threadGroupStartIndex.x;
    if (lastThreadGroup.y) maxGroupThread.y = pushConst.screenSize.y - threadGroupStartIndex.y;

    float depth = depthImage[pixelIndex];
    groupDepths[groupThreadIndex.y][groupThreadIndex.x] = depth;

    if (groupThreadIndex.x == maxGroupThread.x - 1) {
        int2 rightPixelIndex = int2(pixelIndex.x + 1, pixelIndex.y);
        if (rightPixelIndex.x >= pushConst.screenSize.x) rightPixelIndex.x = pixelIndex.x - 1;
        groupDepths[groupThreadIndex.y][maxGroupThread.x] = depthImage[rightPixelIndex];
    }
    if (groupThreadIndex.y == maxGroupThread.y - 1) {
        int2 topPixelIndex = int2(pixelIndex.x, pixelIndex.y + 1);
        if (topPixelIndex.y >= pushConst.screenSize.y) topPixelIndex.y = pixelIndex.y - 1;
        groupDepths[maxGroupThread.y][groupThreadIndex.x] = depthImage[topPixelIndex];
    }
}
//float getClipDepth(float ndcDepth) {
//    float viewZ = (pushConst.nearPlane * pushConst.farPlane) / (pushConst.farPlane - ndcDepth * (pushConst.farPlane - pushConst.nearPlane));
//    return ndcDepth * viewZ;
//}
[numthreads(32, 32, 1)]
[shader("compute")]
void computeMain_getDepthGradient(int2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    getGroupDepths(threadIndex, groupThreadIndex);
    GroupMemoryBarrierWithGroupSync();
    if (any(threadIndex >= pushConst.screenSize)) return;

    float depth = groupDepths[groupThreadIndex.y][groupThreadIndex.x];
    if (depth > 1.0f - 1e-10f) return;

    float depthCenter = depth;
    float depthRight = groupDepths[groupThreadIndex.y][groupThreadIndex.x + 1];
    float depthTop = groupDepths[groupThreadIndex.y + 1][groupThreadIndex.x];

    float gradX = depthRight - depthCenter;
    float gradY = depthTop - depthCenter;

    depthGradientImage[threadIndex] = float4(gradX, gradY, 0.0f, 0.0f);
}
//--------------------------------------------------------------------------------------------------------------
[numthreads(32, 32, 1)]
[shader("compute")]
void computeMain_getEffectiveIrradiance(uint2 threadIndex: SV_DispatchThreadID) {
	if(any(threadIndex >= pushConst.screenSize)) return;

    float depth = depthImage[threadIndex];
    if (depth > 1.0f - 1e-10f) return;

    float3 irradiance = renderImage[threadIndex].xyz / max(albedoImage[threadIndex].xyz, 1e-6f);
    irradianceImage[threadIndex] = float4(irradiance, 1.0f);
}
//--------------------------------------------------------------------------------------------------------------
uint2 getClosestScreenPos(int2 screenPos) {
    float closestDepth = 1.0f;
    uint2 closestScreenPos = screenPos;

    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            uint2 newScreenPos = clamp(screenPos + int2(i, j), 0, pushConst.screenSize - 1);
            float depth = depthImage[newScreenPos];

            if (depth < closestDepth) {
                closestDepth = depth;
                closestScreenPos = newScreenPos;
            }
        }
    }

    return closestScreenPos;
}

[numthreads(32, 32, 1)]
void computeMain_varianceEstimate(uint2 threadIndex: SV_DispatchThreadID) {
    if (any(threadIndex >= pushConst.screenSize)) return;
    float depth = depthImage[threadIndex];
    if (depth > 1.0f - 1e-10f) return;

    float3 irradiance = irradianceImage[threadIndex].xyz;

    float3 newM1 = irradiance;
    float3 newM2 = irradiance * irradiance;
    float3 variance = max(0.0f, newM2 - newM1 * newM1);
    if (pushConst.frameIndex > 0) {
        uint2 closetScreenPos = getClosestScreenPos(threadIndex);
        float2 velocity = velocityImage[closetScreenPos].xy;

        float2 screenPos = (float2(threadIndex) + 0.5f) / pushConst.screenSize;
        float2 offsetScreenPos = screenPos - velocity;
        if (all(offsetScreenPos >= 0.0f) && all(offsetScreenPos <= 1.0f)) {
            uint2 offsetPixelIndex = uint2(offsetScreenPos * pushConst.screenSize);

            bool validHistory = true;

            float currentDepth = depthImage[threadIndex];
            float historyDepth = historyDepthImage[offsetPixelIndex].x;
            validHistory &= abs(currentDepth - historyDepth) < 0.01f;

            float3 currentNormal = normalImage[threadIndex].xyz;
            float3 historyNormal = historyNormalImage[offsetPixelIndex].xyz;
            validHistory &= dot(currentNormal, historyNormal) > 0.99f;

            float3 currentVertexInfo = vertexInfoImage[threadIndex].xyz;        //instanceIndex, meshIndex, materialIndex
            float3 historyVertexInfo = historyVertexInfoImage[offsetPixelIndex].xyz;
            validHistory &= all(currentVertexInfo == historyVertexInfo);

            float3 historyM1 = moment1Image[offsetPixelIndex].xyz;
            float3 historyM2 = moment2Image[offsetPixelIndex].xyz;

            float alpha = 0.05f;
            if (validHistory) {
                newM1 = lerp(historyM1, irradiance, alpha);
                newM2 = lerp(historyM2, irradiance * irradiance, alpha);
                variance = max(0.0f, newM2 - newM1 * newM1);
            }
        }
    }

    historyDepthImage[threadIndex].x = depthImage[threadIndex];
    historyNormalImage[threadIndex] = normalImage[threadIndex];
    historyVertexInfoImage[threadIndex] = vertexInfoImage[threadIndex];

    moment1Image[threadIndex] = float4(newM1, 1.0f);
    moment2Image[threadIndex] = float4(newM2, 1.0f);
    varianceImage[0][threadIndex] = float4(variance, 0.0f);
}