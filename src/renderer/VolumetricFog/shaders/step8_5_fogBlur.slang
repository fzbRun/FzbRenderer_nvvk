#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

#define COMPUTE_SHADER
#include "renderer/VolumetricFog/shaders/getVolumetricFog.slang"

[[vk::binding(StaticBindingPoints_VolumetricFog::eNormalImage)]] Texture2D<float4> normalImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eDepthImage)]] Texture2D<float> depthImage;

[[vk::binding(StaticBindingPoints_VolumetricFog::eRenderedFogResultImage)]] RWTexture2D<float4> renderedFogResultImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eDepthGradientImage)]] RWTexture2D<float4> depthGradientImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFilterImages)]] RWTexture2D<float4> filterImages[];

[[vk::binding(StaticBindingPoints_VolumetricFog::eRenderedImage)]] RWTexture2D<float4> renderedImage;

groupshared float groupDepths[34][34];
void getGroupDepths(uint2 pixelIndex, uint2 groupThreadIndex) {
    int2 threadGroupStartIndex = pixelIndex - groupThreadIndex;
    bool2 lastThreadGroup = threadGroupStartIndex + 32 > pushConst.screenSize;
    int2 maxGroupThread = 32;
    if (lastThreadGroup.x) maxGroupThread.x = pushConst.screenSize.x - threadGroupStartIndex.x;
    if (lastThreadGroup.y) maxGroupThread.y = pushConst.screenSize.y - threadGroupStartIndex.y;

    float depth = depthImage[pixelIndex];
    groupDepths[groupThreadIndex.y + 1][groupThreadIndex.x + 1] = depth;

    if (groupThreadIndex.x == 0) {
        int2 leftPixelIndex = int2(pixelIndex.x - 1, pixelIndex.y);
        if (leftPixelIndex.x < 0) leftPixelIndex.x = pixelIndex.x + 1;
        groupDepths[groupThreadIndex.y + 1][0] = depthImage[leftPixelIndex];
    }
    if (groupThreadIndex.x == maxGroupThread.x - 1) {
        int2 rightPixelIndex = int2(pixelIndex.x + 1, pixelIndex.y);
        if (rightPixelIndex.x >= pushConst.screenSize.x) rightPixelIndex.x = pixelIndex.x - 1;
        groupDepths[groupThreadIndex.y + 1][maxGroupThread.x + 1] = depthImage[rightPixelIndex];
    }
    if (groupThreadIndex.y == 0) {
        int2 bottomPixelIndex = int2(pixelIndex.x, pixelIndex.y - 1);
        if (bottomPixelIndex.y < 0) bottomPixelIndex.y = pixelIndex.y + 1;
        groupDepths[0][groupThreadIndex.x + 1] = depthImage[bottomPixelIndex];
    }
    if (groupThreadIndex.y == maxGroupThread.y - 1) {
        int2 topPixelIndex = int2(pixelIndex.x, pixelIndex.y + 1);
        if (topPixelIndex.y >= pushConst.screenSize.y) topPixelIndex.y = pixelIndex.y - 1;
        groupDepths[maxGroupThread.y + 1][groupThreadIndex.x + 1] = depthImage[topPixelIndex];
    }

    if (groupThreadIndex.x == 0 && groupThreadIndex.y == 0) {
        int2 cornerPixelIndex = int2(max(pixelIndex.x - 1, 0), max(pixelIndex.y - 1, 0));
        groupDepths[0][0] = depthImage[cornerPixelIndex];
    }
    if (groupThreadIndex.x == maxGroupThread.x - 1 && groupThreadIndex.y == 0) {
        int2 cornerPixelIndex = int2(min(pixelIndex.x + 1, pushConst.screenSize.x - 1), max(pixelIndex.y - 1, 0));
        groupDepths[0][maxGroupThread.x + 1] = depthImage[cornerPixelIndex];
    }
    if (groupThreadIndex.x == 0 && groupThreadIndex.y == maxGroupThread.y - 1) {
        int2 cornerPixelIndex = int2(max(pixelIndex.x - 1, 0), min(pixelIndex.y + 1, pushConst.screenSize.y - 1));
        groupDepths[maxGroupThread.y + 1][0] = depthImage[cornerPixelIndex];
    }
    if (groupThreadIndex.x == maxGroupThread.x - 1 && groupThreadIndex.y == maxGroupThread.y - 1) {
        int2 cornerPixelIndex = int2(min(pixelIndex.x + 1, pushConst.screenSize.x - 1), min(pixelIndex.y + 1, pushConst.screenSize.y - 1));
        groupDepths[maxGroupThread.y + 1][maxGroupThread.x + 1] = depthImage[cornerPixelIndex];
    }
}
[numthreads(32, 32, 1)]
[shader("compute")]
void computeMain_getDepthGradient(int2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    if (any(threadIndex >= pushConst.screenSize)) return;
    getGroupDepths(threadIndex, groupThreadIndex);
    GroupMemoryBarrierWithGroupSync();

    float depthLeft = groupDepths[groupThreadIndex.y + 1][groupThreadIndex.x];
    float depthRight = groupDepths[groupThreadIndex.y + 1][groupThreadIndex.x + 2];
    float depthDown = groupDepths[groupThreadIndex.y][groupThreadIndex.x + 1];
    float depthUp = groupDepths[groupThreadIndex.y + 2][groupThreadIndex.x + 1];
    float gradX = (depthRight - depthLeft) * 0.5f;
    float gradY = (depthUp - depthDown) * 0.5f;

    depthGradientImage[threadIndex] = float4(gradX, gradY, 0.0f, 0.0f);
}
//--------------------------------------------------------------------------------------------------------------
[numthreads(32, 32, 1)]
void computeMain_varianceEstimate(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    if (any(threadIndex >= pushConst.screenSize)) return;
}
//--------------------------------------------------------------------------------------------------------------
/*
groupshared float3 groupVariances[34][34];
void getGroupVariances(uint2 pixelIndex, uint2 groupThreadIndex, int varianceImageIndex) {
    int2 threadGroupStartIndex = pixelIndex - groupThreadIndex;
    bool2 lastThreadGroup = threadGroupStartIndex + 32 > pushConst.screenSize;
    int2 maxGroupThread = 32;
    if (lastThreadGroup.x) maxGroupThread.x = pushConst.screenSize.x - threadGroupStartIndex.x;
    if (lastThreadGroup.y) maxGroupThread.y = pushConst.screenSize.y - threadGroupStartIndex.y;

    float3 variance = fogVarianceImages[varianceImageIndex][pixelIndex].xyz;
    groupVariances[groupThreadIndex.y + 1][groupThreadIndex.x + 1] = variance;

    if (groupThreadIndex.x == 0) {
        int2 leftPixelIndex = int2(pixelIndex.x - 1, pixelIndex.y);
        if (leftPixelIndex.x < 0) leftPixelIndex.x = pixelIndex.x + 1;
        groupVariances[groupThreadIndex.y + 1][0] = fogVarianceImages[varianceImageIndex][leftPixelIndex].xyz;
    }
    if (groupThreadIndex.x == maxGroupThread.x - 1) {
        int2 rightPixelIndex = int2(pixelIndex.x + 1, pixelIndex.y);
        if (rightPixelIndex.x >= pushConst.screenSize.x) rightPixelIndex.x = pixelIndex.x - 1;
        groupVariances[groupThreadIndex.y + 1][maxGroupThread.x + 1] = fogVarianceImages[varianceImageIndex][rightPixelIndex].xyz;
    }
    if (groupThreadIndex.y == 0) {
        int2 bottomPixelIndex = int2(pixelIndex.x, pixelIndex.y - 1);
        if (bottomPixelIndex.y < 0) bottomPixelIndex.y = pixelIndex.y + 1;
        groupVariances[0][groupThreadIndex.x + 1] = fogVarianceImages[varianceImageIndex][bottomPixelIndex].xyz;
    }
    if (groupThreadIndex.y == maxGroupThread.y - 1) {
        int2 topPixelIndex = int2(pixelIndex.x, pixelIndex.y + 1);
        if (topPixelIndex.y >= pushConst.screenSize.y) topPixelIndex.y = pixelIndex.y - 1;
        groupVariances[maxGroupThread.y + 1][groupThreadIndex.x + 1] = fogVarianceImages[varianceImageIndex][topPixelIndex].xyz;
    }

    if (groupThreadIndex.x == 0 && groupThreadIndex.y == 0) {
        int2 cornerPixelIndex = int2(max(pixelIndex.x - 1, 0), max(pixelIndex.y - 1, 0));
        groupVariances[0][0] = fogVarianceImages[varianceImageIndex][cornerPixelIndex].xyz;
    }
    if (groupThreadIndex.x == maxGroupThread.x - 1 && groupThreadIndex.y == 0) {
        int2 cornerPixelIndex = int2(min(pixelIndex.x + 1, pushConst.screenSize.x - 1), max(pixelIndex.y - 1, 0));
        groupVariances[0][maxGroupThread.x + 1] = fogVarianceImages[varianceImageIndex][cornerPixelIndex].xyz;
    }
    if (groupThreadIndex.x == 0 && groupThreadIndex.y == maxGroupThread.y - 1) {
        int2 cornerPixelIndex = int2(max(pixelIndex.x - 1, 0), min(pixelIndex.y + 1, pushConst.screenSize.y - 1));
        groupVariances[maxGroupThread.y + 1][0] = fogVarianceImages[varianceImageIndex][cornerPixelIndex].xyz;
    }
    if (groupThreadIndex.x == maxGroupThread.x - 1 && groupThreadIndex.y == maxGroupThread.y - 1) {
        int2 cornerPixelIndex = int2(min(pixelIndex.x + 1, pushConst.screenSize.x - 1), min(pixelIndex.y + 1, pushConst.screenSize.y - 1));
        groupVariances[maxGroupThread.y + 1][maxGroupThread.x + 1] = fogVarianceImages[varianceImageIndex][cornerPixelIndex].xyz;
    }
}
static const float G3[3] = { 1.0f / 4.0f, 1.0f / 2.0f, 1.0f / 4.0f };
[numthreads(32, 32, 1)]
void computeMain_varianceConvolution(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    getGroupVariances(threadIndex, groupThreadIndex, 0);
    GroupMemoryBarrierWithGroupSync();
    if (any(threadIndex >= pushConst.screenSize)) return;
    float depth = depthImage[threadIndex];
    if (depth > 1.0f - 1e-10f) return;

    int2 center = groupThreadIndex + 1;

    float3 sum = 0.0f;
    float weightSum = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int2 sample = int2(center.x + dx, center.y + dy);
            float h = G3[dx + 1] * G3[dy + 1];
            sum += h * groupVariances[sample.y][sample.x];
            weightSum += h;
        }
    }

    fogVarianceImages[1][threadIndex] = float4(sum / weightSum, 1.0f);
}
*/
//--------------------------------------------------------------------------------------------------------------
#define SVGF_FILTER_K 5
static const float G5[SVGF_FILTER_K] = {
    1.0f / 16.0f,
    4.0f / 16.0f,
    6.0f / 16.0f,
    4.0f / 16.0f,
    1.0f / 16.0f
};
float getDepthWeight(uint2 uvCenter, uint2 uvNeighbor, float sigmaZ = 60.0f) {
    float zP = depthImage[uvCenter];
    float zQ = depthImage[uvNeighbor];

    float2 gradP = depthGradientImage[uvCenter].xy;
    int2 offset = int2(uvNeighbor) - int2(uvCenter);
    float gradProj = abs(gradP.x * float(offset.x) + gradP.y * float(offset.y));

    float eZ = abs(zP - zQ) / (sigmaZ * gradProj + 1e-6f);
    return exp(-eZ);
}
float getNormalWeight(uint2 uvCenter, uint2 uvNeighbor, float sigmaN = 128.0f) {
    float3 normalP = normalImage[uvCenter].xyz;
    float3 normalQ = normalImage[uvNeighbor].xyz;

    if (dot(normalP, normalP) < 1e-6f && dot(normalQ, normalQ) < 1e-6f) return 1.0f;

    float dp = dot(normalP, normalQ);
    dp = max(0.0f, dp);
    return pow(dp, sigmaN);
}
float getIrradianceWeight(float lumP, float lumQ, float lumVariance, float sigmaL = 1.0f) {
    float denom = sigmaL * sqrt(max(0.0f, lumVariance)) + 1e-6f;
    return exp(-abs(lumP - lumQ) / denom);
}
[numthreads(32, 32, 1)]
[shader("compute")]
void computeMain_blurFog_X(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    if (any(threadIndex >= pushConst.screenSize)) return;

    int filterIndex = pushConst.instanceIndex;
    int2 pixelIndex = threadIndex;
    float4 fogResult = filterIndex == 0 ? renderedFogResultImage[pixelIndex] : filterImages[0][pixelIndex];
    float lumP0 = fogResult.x;
    float lumP1 = fogResult.y;

    uint step = 1 << filterIndex;
    float weightSum0 = 0.0f, weightSum1 = 0.0f;
    float valueSum0 = 0.0f, valueSum1 = 0.0f;
    for (int i = 0; i < SVGF_FILTER_K; ++i) {
        int offsetX = (i - SVGF_FILTER_K / 2) * step;
        if (pixelIndex.x + offsetX < 0 || pixelIndex.x + offsetX > pushConst.screenSize.x - 1) continue;
        int2 offsetPixelIndex = int2(clamp(pixelIndex.x + offsetX, 0, pushConst.screenSize.x - 1), pixelIndex.y);

        float w0 = G5[i], w1 = G5[i]; float q0 = lumP0, q1 = lumP1;
        if (offsetX != 0) { // dont consider variance, so weight must = G5[i]
            float depthWeight = getDepthWeight(pixelIndex, offsetPixelIndex);
            float normalWeight = getNormalWeight(pixelIndex, offsetPixelIndex);

            float4 offsetFogResult = filterImages[0][offsetPixelIndex];
            float lumQ1 = luminance(offsetFogResult.y);
            float irradianceWeight1 = getIrradianceWeight(lumP1, lumQ1, 1.0f);

            //float lumQ1 = luminance(renderedFogAttenuationImage[offsetPixelIndex].xyz);
            //float irradianceWeight1 = getIrradianceWeight(lumP1, lumQ1, lumVariance);

            w0 *= depthWeight * normalWeight;
            w1 *= depthWeight * normalWeight * irradianceWeight1;
            if (filterIndex > 0) {
                float4 filterResult = filterImages[0][offsetPixelIndex];
                q0 = filterResult.x;
                q1 = filterResult.y;
            }
        }

        weightSum0 += w0;
        weightSum1 += w1;
        valueSum0 += w0 * q0;
        valueSum1 += w1 * q1;
    }

    valueSum0 /= max(weightSum0, 1e-6f);
    valueSum1 /= max(weightSum1, 1e-6f);

    filterImages[1][pixelIndex] = float4(valueSum0, valueSum1, 1.0f, 1.0f);
}
[numthreads(32, 32, 1)]
[shader("compute")]
void computeMain_blurFog_Y(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    uint2 screenSize = pushConst.screenSize;
    if (threadIndex.x >= screenSize.x || threadIndex.y >= screenSize.y) return;

    if (any(threadIndex >= pushConst.screenSize)) return;

    int2 pixelIndex = threadIndex;
    float4 fogResult = filterImages[1][pixelIndex]; //renderedFogResultImage[pixelIndex];
    float lumP0 = fogResult.x;
    float lumP1 = fogResult.y;

    int filterIndex = pushConst.instanceIndex;

    uint step = 1 << filterIndex;
    float weightSum0 = 0.0f, weightSum1 = 0.0f;
    float valueSum0 = 0.0f, valueSum1 = 0.0f;
    for (int i = 0; i < SVGF_FILTER_K; ++i) {
        int offsetY = (i - SVGF_FILTER_K / 2) * step;
        if (pixelIndex.y + offsetY < 0 || pixelIndex.y + offsetY > pushConst.screenSize.y - 1) continue;
        int2 offsetPixelIndex = int2(pixelIndex.x, clamp(pixelIndex.y + offsetY, 0, pushConst.screenSize.y - 1));

        float w0 = G5[i], w1 = G5[i];
        float q0 = lumP0, q1 = lumP1;
        if (offsetY != 0) {
            float depthWeight = getDepthWeight(pixelIndex, offsetPixelIndex);
            float normalWeight = getNormalWeight(pixelIndex, offsetPixelIndex);

            float4 offsetFogResult = filterImages[1][offsetPixelIndex];
            q0 = offsetFogResult.x;
            q1 = offsetFogResult.y;

            float lumQ1 = luminance(offsetFogResult.y);
            float irradianceWeight1 = getIrradianceWeight(lumP1, lumQ1, 1.0f);

            w0 *= depthWeight * normalWeight;
            w1 *= depthWeight * normalWeight * irradianceWeight1;
        }

        weightSum0 += w0;
        weightSum1 += w1;
        valueSum0 += w0 * q0;
        valueSum1 += w1 * q1;
    }

    valueSum0 /= max(weightSum0, 1e-6f);
    valueSum1 /= max(weightSum1, 1e-6f);

    if (filterIndex == pushConst.iteration - 1) renderedFogResultImage[pixelIndex] = float4(valueSum0, valueSum1, 1.0f, 1.0f);
    else filterImages[0][pixelIndex] = float4(valueSum0, valueSum1, 1.0f, 1.0f);
}
//--------------------------------------------------------------------------------------------------------------
[numthreads(32, 32, 1)]
[shader("compute")]
void computeMain_addFog(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    uint2 screenSize = pushConst.screenSize;
    if (threadIndex.x >= screenSize.x || threadIndex.y >= screenSize.y) return;
    float3 baseColor = renderedImage[threadIndex].xyz;
    float2 fogResult = renderedFogResultImage[threadIndex].xy;
    float3 resultColor = baseColor * fogResult.x + fogResult.y;
    //float3 resultColor = baseColor + renderedFogLImage[threadIndex].xyz;
    renderedImage[threadIndex].xyz = resultColor;
}