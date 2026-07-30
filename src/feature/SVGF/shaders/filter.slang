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
[[vk::binding(BindingPoints_SVGF::eResultImage)]] RWTexture2D<float4> resultImage;

[[vk::binding(BindingPoints_SVGF::eDepthGradientImage)]] RWTexture2D<float4> depthGradientImage;
[[vk::binding(BindingPoints_SVGF::eIrradianceImage)]] RWTexture2D<float4> irradianceImage;

[[vk::binding(BindingPoints_SVGF::eMoment1Image)]] RWTexture2D<float4> moment1Image;
[[vk::binding(BindingPoints_SVGF::eVarianceImage)]] RWTexture2D<float4> varianceImage[2];
[[vk::binding(BindingPoints_SVGF::eFilterImages)]] RWTexture2D<float4> filterImage[2];

#define SVGF_FILTER_K 5
static const float G5[SVGF_FILTER_K] = {
    1.0f / 16.0f,
    4.0f / 16.0f,
    6.0f / 16.0f,
    4.0f / 16.0f,
    1.0f / 16.0f
};

//float getClipDepth(float ndcDepth) {
//    float viewZ = (pushConst.nearPlane * pushConst.farPlane) / (pushConst.farPlane - ndcDepth * (pushConst.farPlane - pushConst.nearPlane));
//    return ndcDepth * viewZ;
//}
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

    float dp = dot(normalP, normalQ);
    dp = max(0.0f, dp);
    return pow(dp, sigmaN);
}
float getIrradianceWeight(uint2 uvNeighbor, float lumP, float lumVariance, float sigmaL = 10.0f) {
    float lumQ = luminance(irradianceImage[uvNeighbor].xyz);
    float denom = sigmaL * sqrt(max(0.0f, lumVariance)) + 1e-6f;
    return exp(-abs(lumP - lumQ) / denom);
}

[numthreads(32, 32, 1)]
[shader("compute")]
void computeMain_Filter_X(uint2 threadIndex: SV_DispatchThreadID) {
    if (any(threadIndex >= pushConst.screenSize)) return;
    float depth = depthImage[threadIndex];
    if (depth > 1.0f - 1e-10f) return;

    int2 pixelIndex = threadIndex;
    float3 varGuideCenter = varianceImage[1][pixelIndex].xyz;
    float lumCenter = luminance(irradianceImage[pixelIndex].xyz);
    float lumVariance = luminance(varGuideCenter);

    uint step = 1 << pushConst.filterIndex;
    float weightSum = 0.0f;
    float3 irradianceSum = 0.0f, varianceSum = 0.0f;
    for (int i = 0; i < SVGF_FILTER_K; ++i) {
        int offsetX = (i - SVGF_FILTER_K / 2) * step;
        //offsetX = pixelIndex.x + offsetX < 0 || pixelIndex.x + offsetX > pushConst.screenSize.x ? -offsetX : offsetX;
        //int2 offsetPixelIndex = int2(pixelIndex.x + offsetX, pixelIndex.y);
        int2 offsetPixelIndex = int2(clamp(pixelIndex.x + offsetX, 0, pushConst.screenSize.x - 1), pixelIndex.y);
        float offsetPixelDepth = depthImage[offsetPixelIndex];
        if (offsetPixelDepth > 1.0f - 1e-10f) continue;

        float depthWeight = getDepthWeight(pixelIndex, offsetPixelIndex);
        float normalWeight = getNormalWeight(pixelIndex, offsetPixelIndex);
        float irradianceWeight = getIrradianceWeight(offsetPixelIndex, lumCenter, lumVariance);

        float weight = depthWeight * normalWeight * irradianceWeight * G5[i];
        weightSum += weight;
        irradianceSum += weight * (pushConst.filterIndex == 0 ? irradianceImage[offsetPixelIndex] : filterImage[0][offsetPixelIndex]).xyz;
        varianceSum += (weight * weight) * varianceImage[1][offsetPixelIndex].xyz;
    }

    irradianceSum /= max(weightSum, 1e-6f);
    varianceSum /= max(weightSum * weightSum, 1e-6f);

    filterImage[1][pixelIndex] = float4(irradianceSum, 1.0f);
    varianceImage[0][pixelIndex] = float4(varianceSum, 1.0f);
}

[numthreads(32, 32, 1)]
[shader("compute")]
void computeMain_Filter_Y(uint2 threadIndex: SV_DispatchThreadID) {
    if (any(threadIndex >= pushConst.screenSize)) return;
    float depth = depthImage[threadIndex];
    if (depth > 1.0f - 1e-10f) return;

    int2 pixelIndex = threadIndex;
    float3 varGuideCenter = varianceImage[1][pixelIndex].xyz;
    float lumCenter = luminance(irradianceImage[pixelIndex].xyz);
    float lumVariance = luminance(varGuideCenter);

    uint step = 1 << pushConst.filterIndex;
    float weightSum = 0.0f;
    float3 irradianceSum = 0.0f, varianceSum = 0.0f;
    for (int i = 0; i < SVGF_FILTER_K; ++i) {
        int offsetY = (i - SVGF_FILTER_K / 2) * step;
        // offsetY = pixelIndex.y + offsetY < 0 || pixelIndex.y + offsetY > pushConst.screenSize.y ? -offsetY : offsetY;
        // int2 offsetPixelIndex = int2(pixelIndex.x, pixelIndex.y + offsetY);
        int2 offsetPixelIndex = int2(pixelIndex.x, clamp(pixelIndex.y + offsetY, 0, pushConst.screenSize.y - 1));
        float offsetPixelDepth = depthImage[offsetPixelIndex];
        if (offsetPixelDepth > 1.0f - 1e-10f) continue;

        float depthWeight = getDepthWeight(pixelIndex, offsetPixelIndex);
        float normalWeight = getNormalWeight(pixelIndex, offsetPixelIndex);
        float irradianceWeight = getIrradianceWeight(offsetPixelIndex, lumCenter, lumVariance);

        float weight = depthWeight * normalWeight * irradianceWeight * G5[i];
        weightSum += weight;
        irradianceSum += weight * filterImage[1][offsetPixelIndex].xyz;
        varianceSum += (weight * weight) * varianceImage[1][offsetPixelIndex].xyz;
    }

    irradianceSum /= max(weightSum, 1e-6f);
    varianceSum /= max(weightSum * weightSum, 1e-6f);

    if (pushConst.filterIndex == SVGF_FILTER_COUNT - 1) {
        if (depthImage[threadIndex] < 1.0f - 1e-10f) {
            //float3 historayIrradiance = moment1Image[threadIndex].xyz;
            float3 color = irradianceSum * albedoImage[threadIndex].xyz;
            resultImage[pixelIndex] = float4(color, 1.0f);
        }
    } else {
        filterImage[0][pixelIndex] = float4(irradianceSum, 1.0f);
        varianceImage[0][pixelIndex] = float4(varianceSum, 1.0f);
    }
}
//------------------------------------------------------------------------------------------------------------------------------------
groupshared float3 groupVariances[34][34];
void getGroupVariances(uint2 pixelIndex, uint2 groupThreadIndex, int varianceImageIndex) {
    int2 threadGroupStartIndex = pixelIndex - groupThreadIndex;
    bool2 lastThreadGroup = threadGroupStartIndex + 32 > pushConst.screenSize;
    int2 maxGroupThread = 32;
    if (lastThreadGroup.x) maxGroupThread.x = pushConst.screenSize.x - threadGroupStartIndex.x;
    if (lastThreadGroup.y) maxGroupThread.y = pushConst.screenSize.y - threadGroupStartIndex.y;

    float3 variance = varianceImage[varianceImageIndex][pixelIndex].xyz;
    groupVariances[groupThreadIndex.y + 1][groupThreadIndex.x + 1] = variance;

    if (groupThreadIndex.x == 0) {
        int2 leftPixelIndex = int2(pixelIndex.x - 1, pixelIndex.y);
        if (leftPixelIndex.x < 0) leftPixelIndex.x = pixelIndex.x + 1;
        groupVariances[groupThreadIndex.y + 1][0] = varianceImage[varianceImageIndex][leftPixelIndex].xyz;
    }
    if (groupThreadIndex.x == maxGroupThread.x - 1) {
        int2 rightPixelIndex = int2(pixelIndex.x + 1, pixelIndex.y);
        if (rightPixelIndex.x >= pushConst.screenSize.x) rightPixelIndex.x = pixelIndex.x - 1;
        groupVariances[groupThreadIndex.y + 1][maxGroupThread.x + 1] = varianceImage[varianceImageIndex][rightPixelIndex].xyz;
    }
    if (groupThreadIndex.y == 0) {
        int2 bottomPixelIndex = int2(pixelIndex.x, pixelIndex.y - 1);
        if (bottomPixelIndex.y < 0) bottomPixelIndex.y = pixelIndex.y + 1;
        groupVariances[0][groupThreadIndex.x + 1] = varianceImage[varianceImageIndex][bottomPixelIndex].xyz;
    }
    if (groupThreadIndex.y == maxGroupThread.y - 1) {
        int2 topPixelIndex = int2(pixelIndex.x, pixelIndex.y + 1);
        if (topPixelIndex.y >= pushConst.screenSize.y) topPixelIndex.y = pixelIndex.y - 1;
        groupVariances[maxGroupThread.y + 1][groupThreadIndex.x + 1] = varianceImage[varianceImageIndex][topPixelIndex].xyz;
    }

    if (groupThreadIndex.x == 0 && groupThreadIndex.y == 0) {
        int2 cornerPixelIndex = int2(max(pixelIndex.x - 1, 0), max(pixelIndex.y - 1, 0));
        groupVariances[0][0] = varianceImage[varianceImageIndex][cornerPixelIndex].xyz;
    }
    if (groupThreadIndex.x == maxGroupThread.x - 1 && groupThreadIndex.y == 0) {
        int2 cornerPixelIndex = int2(min(pixelIndex.x + 1, pushConst.screenSize.x - 1), max(pixelIndex.y - 1, 0));
        groupVariances[0][maxGroupThread.x + 1] = varianceImage[varianceImageIndex][cornerPixelIndex].xyz;
    }
    if (groupThreadIndex.x == 0 && groupThreadIndex.y == maxGroupThread.y - 1) {
        int2 cornerPixelIndex = int2(max(pixelIndex.x - 1, 0), min(pixelIndex.y + 1, pushConst.screenSize.y - 1));
        groupVariances[maxGroupThread.y + 1][0] = varianceImage[varianceImageIndex][cornerPixelIndex].xyz;
    }
    if (groupThreadIndex.x == maxGroupThread.x - 1 && groupThreadIndex.y == maxGroupThread.y - 1) {
        int2 cornerPixelIndex = int2(min(pixelIndex.x + 1, pushConst.screenSize.x - 1), min(pixelIndex.y + 1, pushConst.screenSize.y - 1));
        groupVariances[maxGroupThread.y + 1][maxGroupThread.x + 1] = varianceImage[varianceImageIndex][cornerPixelIndex].xyz;
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

    varianceImage[1][threadIndex] = float4(sum / weightSum, 1.0f);
}