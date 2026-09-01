#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<FogBlurPushConstant> pushConst;

[[vk::binding(StaticBindingPoints_VolumetricFog::eNormalImage)]] Texture2D<float4> normalImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eDepthImage)]] Texture2D<float> depthImage;

[[vk::binding(StaticBindingPoints_VolumetricFog::eRenderedFogResultImage)]] RWTexture2D<float4> renderedFogResultImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eDepthGradientImage)]] RWTexture2D<float4> depthGradientImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFilterImages)]] RWTexture2D<float4> filterImages[];

[[vk::binding(StaticBindingPoints_VolumetricFog::eRenderedImage)]] RWTexture2D<float4> renderedImage;

float getLinearViewDepth(float rawDepth, float nearPlane, float farPlane){
    return (nearPlane * farPlane) / max(farPlane - rawDepth * (farPlane - nearPlane), 1e-6f);
}

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

    #ifdef LINEAR_DEPTH
    depthLeft = getLinearViewDepth(depthLeft, pushConst.cameraNearPlane, pushConst.cameraFarPlane);
    depthRight = getLinearViewDepth(depthRight, pushConst.cameraNearPlane, pushConst.cameraFarPlane);
    depthDown = getLinearViewDepth(depthDown, pushConst.cameraNearPlane, pushConst.cameraFarPlane);
    depthUp = getLinearViewDepth(depthUp, pushConst.cameraNearPlane, pushConst.cameraFarPlane);
    #endif

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
float getDepthWeight(float zP, float2 gradP, float zQ, uint2 uvCenter, uint2 uvNeighbor, float sigmaZ = 10.0f) {
    #ifdef LINEAR_DEPTH
    zP = getLinearViewDepth(zP, pushConst.cameraNearPlane, pushConst.cameraFarPlane);
    zQ = getLinearViewDepth(zQ, pushConst.cameraNearPlane, pushConst.cameraFarPlane);
    #endif

    int2 offset = int2(uvNeighbor) - int2(uvCenter);
    float gradProj = abs(gradP.x * float(offset.x) + gradP.y * float(offset.y));

    float eZ = abs(zP - zQ) / (sigmaZ * gradProj + 1e-6f);
    return exp(-eZ);
}
float getNormalWeight(float3 normalP, float3 normalQ, uint2 uvCenter, uint2 uvNeighbor, float sigmaN = 128.0f) {
    if (dot(normalP, normalP) < 1e-6f && dot(normalQ, normalQ) < 1e-6f) return 1.0f;

    float dp = dot(normalP, normalQ);
    dp = max(0.0f, dp);
    return pow(dp, sigmaN);
}
float getIrradianceWeight(float lumP, float lumQ, float lumVariance, float sigmaL = 16.0f) {
    float denom = sigmaL * sqrt(max(0.0f, lumVariance)) + 1e-6f;
    return exp(-abs(lumP - lumQ) / denom);
}

#define Group_NeighborFogInfo_Count BLUR_FOG_THREADGROUP_SIZE + (SVGF_FILTER_K / 2) * (1 << 4) * 2
groupshared float2 groupNeighborFogInfo[Group_NeighborFogInfo_Count];       // 2.5KB
groupshared float3 groupNeighborDepthInfo[Group_NeighborFogInfo_Count];     // 3.75KB
groupshared float3 groupNeighborNormalInfo[Group_NeighborFogInfo_Count];    // 3.75KB
void fillGroupNeighborFogInfo(uint2 pixelIndex, int groupThreadIndex, int threadGroupSize, RWTexture2D<float4> image, int filterIndex, bool isX) {
    uint haloSize = (SVGF_FILTER_K / 2) * (1 << filterIndex);
    if (all(pixelIndex < pushConst.screenSize)) {
        groupNeighborFogInfo[haloSize + groupThreadIndex] = image[pixelIndex].xy;
        groupNeighborDepthInfo[haloSize + groupThreadIndex].x = depthImage[pixelIndex];
        groupNeighborDepthInfo[haloSize + groupThreadIndex].yz = depthGradientImage[pixelIndex].xy;
        groupNeighborNormalInfo[haloSize + groupThreadIndex] = normalImage[pixelIndex].xyz;
    }

    int2 threadGroupStartPixel = pixelIndex - int2(isX, 1 - isX) * groupThreadIndex;
    int2 threadGroupEndPixel = threadGroupStartPixel + int2(isX, 1 - isX) * threadGroupSize;

    float2 haloFogInfo = float2(1.0f, 0.0f);
    float3 haloDepthInfo = float3(1.0f);
    float3 haloNormalInfo = float3(0.0f);
    // for (int i = groupThreadIndex; i < haloSize; i += BLUR_FOG_THREADGROUP_SIZE) {
    int i = groupThreadIndex;
    if (i < haloSize) {
        int2 haloPixelIndex = threadGroupStartPixel - int2(isX, 1 - isX) * (i + 1);

        if (all(haloPixelIndex >= 0) && all(haloPixelIndex < pushConst.screenSize)) {
            haloFogInfo = image[haloPixelIndex].xy;
            haloDepthInfo.x = depthImage[haloPixelIndex].x;
            haloDepthInfo.yz = depthGradientImage[haloPixelIndex].xy;
            haloNormalInfo = normalImage[haloPixelIndex].xyz;
        }
        groupNeighborFogInfo[haloSize - i - 1] = haloFogInfo;
        groupNeighborDepthInfo[haloSize - i - 1] = haloDepthInfo;
        groupNeighborNormalInfo[haloSize - i - 1] = haloNormalInfo;

        haloFogInfo = float2(1.0f, 0.0f);
        haloDepthInfo = float3(1.0f);
        haloNormalInfo = float3(0.0f);

        haloPixelIndex = threadGroupEndPixel + int2(isX, 1 - isX) * i;
        if (all(haloPixelIndex >= 0) && all(haloPixelIndex < pushConst.screenSize)) {
            haloFogInfo = image[haloPixelIndex].xy;
            haloDepthInfo.x = depthImage[haloPixelIndex].x;
            haloDepthInfo.yz = depthGradientImage[haloPixelIndex].xy;
            haloNormalInfo = normalImage[haloPixelIndex].xyz;
        }
        groupNeighborFogInfo[haloSize + BLUR_FOG_THREADGROUP_SIZE + i] = haloFogInfo;
        groupNeighborDepthInfo[haloSize + BLUR_FOG_THREADGROUP_SIZE + i] = haloDepthInfo;
        groupNeighborNormalInfo[haloSize + BLUR_FOG_THREADGROUP_SIZE + i] = haloNormalInfo;
    }
}

[numthreads(BLUR_FOG_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_blurFog_X(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID, uint threadGroupIndex: SV_GroupID) {
    int filterIndex = pushConst.filterIndex;
    uint threadGroupCount_row = (pushConst.screenSize.x + BLUR_FOG_THREADGROUP_SIZE - 1) / BLUR_FOG_THREADGROUP_SIZE;
    uint threadCount_row = threadGroupCount_row * BLUR_FOG_THREADGROUP_SIZE;
    int2 p_pixelIndex;
    p_pixelIndex.y = threadGroupIndex / threadGroupCount_row;
    p_pixelIndex.x = threadIndex - p_pixelIndex.y * threadCount_row;
    uint lastThreadCount_row = pushConst.screenSize.x - (threadGroupIndex % threadGroupCount_row) * BLUR_FOG_THREADGROUP_SIZE;
    uint threadGroupSize = lastThreadCount_row >= BLUR_FOG_THREADGROUP_SIZE ? BLUR_FOG_THREADGROUP_SIZE : lastThreadCount_row;

    if (filterIndex == 0) fillGroupNeighborFogInfo(p_pixelIndex, groupThreadIndex, threadGroupSize, renderedFogResultImage, filterIndex, true);
    else fillGroupNeighborFogInfo(p_pixelIndex, groupThreadIndex, threadGroupSize, filterImages[0], filterIndex, true);
    GroupMemoryBarrierWithGroupSync();
    if (any(p_pixelIndex >= pushConst.screenSize)) return;

    uint step = 1 << filterIndex;
    uint haloSize = (SVGF_FILTER_K / 2) * step;
    uint p_groupDataIndex = haloSize + groupThreadIndex;
    float2 p_fogResult = groupNeighborFogInfo[p_groupDataIndex];
    float p_A = p_fogResult.x;
    float p_L = p_fogResult.y;

    float3 depthInfo = groupNeighborDepthInfo[p_groupDataIndex];
    float p_Z = depthInfo.x;
    float2 p_gradZ = depthInfo.yz;
    float3 p_normal = groupNeighborNormalInfo[p_groupDataIndex];

    float weightSum_A = 0.0f, weightSum_L = 0.0f;
    float valueSum_A = 0.0f, valueSum_L = 0.0f;
    for (int i = 0; i < SVGF_FILTER_K; ++i) {
        int offsetX = (i - SVGF_FILTER_K / 2) * step;
        if (p_pixelIndex.x + offsetX < 0 || p_pixelIndex.x + offsetX > pushConst.screenSize.x - 1) continue;
        int2 q_pixelIndex = int2(p_pixelIndex.x + offsetX, p_pixelIndex.y);
        int q_groupDataIndex = p_groupDataIndex + offsetX;

        float w_A = G5[i], w_L = G5[i]; float q_A = p_A, q_L = p_L;
        if (offsetX != 0) { // dont consider variance, so weight must = G5[i]
            float2 q_fogResult = groupNeighborFogInfo[q_groupDataIndex];
            q_A = q_fogResult.x;
            q_L = q_fogResult.y;
            float q_Z = groupNeighborDepthInfo[q_groupDataIndex].x;
            float3 q_normal = groupNeighborNormalInfo[q_groupDataIndex];

            float depthWeight = getDepthWeight(p_Z, p_gradZ, q_Z, p_pixelIndex, q_pixelIndex);
            float normalWeight = getNormalWeight(p_normal, q_normal, p_pixelIndex, q_pixelIndex);
            float irradianceWeight1 = getIrradianceWeight(p_L, q_L, 1.0f);

            w_A *= depthWeight * normalWeight;
            w_L *= depthWeight * normalWeight * irradianceWeight1;
        }

        weightSum_A += w_A;
        weightSum_L += w_L;
        valueSum_A += w_A * q_A;
        valueSum_L += w_L * q_L;
    }

    valueSum_A /= max(weightSum_A, 1e-6f);
    valueSum_L /= max(weightSum_L, 1e-6f);

    filterImages[1][p_pixelIndex] = float4(valueSum_A, valueSum_L, 1.0f, 1.0f);
}
[numthreads(BLUR_FOG_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_blurFog_Y(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID, uint threadGroupIndex: SV_GroupID) {
    int filterIndex = pushConst.filterIndex;
    uint threadGroupCount_col = (pushConst.screenSize.y + BLUR_FOG_THREADGROUP_SIZE - 1) / BLUR_FOG_THREADGROUP_SIZE;
    uint threadCount_col = threadGroupCount_col * BLUR_FOG_THREADGROUP_SIZE;
    int2 p_pixelIndex;
    p_pixelIndex.x = threadGroupIndex / threadGroupCount_col;
    p_pixelIndex.y = threadIndex - p_pixelIndex.x * threadCount_col;
    uint lastThreadCount_col = pushConst.screenSize.y - (threadGroupIndex % threadGroupCount_col) * BLUR_FOG_THREADGROUP_SIZE;
    uint threadGroupSize = lastThreadCount_col >= BLUR_FOG_THREADGROUP_SIZE ? BLUR_FOG_THREADGROUP_SIZE : lastThreadCount_col;

    fillGroupNeighborFogInfo(p_pixelIndex, groupThreadIndex, threadGroupSize, filterImages[1], filterIndex, false);
    GroupMemoryBarrierWithGroupSync();
    if (any(p_pixelIndex >= pushConst.screenSize)) return;

    uint step = 1 << filterIndex;
    uint haloSize = (SVGF_FILTER_K / 2) * step;
    uint p_groupDataIndex = haloSize + groupThreadIndex;
    float2 p_fogResult = groupNeighborFogInfo[p_groupDataIndex];
    float p_A = p_fogResult.x;
    float p_L = p_fogResult.y;

    float3 depthInfo = groupNeighborDepthInfo[p_groupDataIndex];
    float p_Z = depthInfo.x;
    float2 p_gradZ = depthInfo.yz;
    float3 p_normal = groupNeighborNormalInfo[p_groupDataIndex];

    float weightSum_A = 0.0f, weightSum_L = 0.0f;
    float valueSum_A = 0.0f, valueSum_L = 0.0f;
    for (int i = 0; i < SVGF_FILTER_K; ++i) {
        int offsetY = (i - SVGF_FILTER_K / 2) * step;
        if (p_pixelIndex.y + offsetY < 0 || p_pixelIndex.y + offsetY > pushConst.screenSize.y - 1) continue;
        int2 q_pixelIndex = int2(p_pixelIndex.x, p_pixelIndex.y + offsetY);
        int q_groupDataIndex = p_groupDataIndex + offsetY;

        float w_A = G5[i], w_L = G5[i];
        float q_A = p_A, q_L = p_L;
        if (offsetY != 0) {
            float2 q_fogResult = groupNeighborFogInfo[q_groupDataIndex];
            q_A = q_fogResult.x;
            q_L = q_fogResult.y;
            float q_Z = groupNeighborDepthInfo[q_groupDataIndex].x;
            float3 q_normal = groupNeighborNormalInfo[q_groupDataIndex];

            float depthWeight = getDepthWeight(p_Z, p_gradZ, q_Z, p_pixelIndex, q_pixelIndex);
            float normalWeight = getNormalWeight(p_normal, q_normal, p_pixelIndex, q_pixelIndex);
            float irradianceWeight1 = getIrradianceWeight(p_L, q_L, 1.0f);

            w_A *= depthWeight * normalWeight;
            w_L *= depthWeight * normalWeight * irradianceWeight1;
        }

        weightSum_A += w_A;
        weightSum_L += w_L;
        valueSum_A += w_A * q_A;
        valueSum_L += w_L * q_L;
    }

    valueSum_A /= max(weightSum_A, 1e-6f);
    valueSum_L /= max(weightSum_L, 1e-6f);

    if (filterIndex == pushConst.fogBlurCount - 1) renderedFogResultImage[p_pixelIndex] = float4(valueSum_A, valueSum_L, 1.0f, 1.0f);
    else filterImages[0][p_pixelIndex] = float4(valueSum_A, valueSum_L, 1.0f, 1.0f);
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