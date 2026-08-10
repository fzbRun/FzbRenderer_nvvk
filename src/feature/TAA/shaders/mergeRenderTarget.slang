#include "feature/TAA/TAAShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<TAAPushConstant> pushConst;
[[vk::binding(BindingPoints_TAA::eRendereTarget_Current)]] Sampler2D<float4> rendereTarget_Current;
[[vk::binding(BindingPoints_TAA::eRenderTarget_Last)]] Sampler2D<float4> rendereTarget_Last;
[[vk::binding(BindingPoints_TAA::eRenderTarget_Final)]] RWTexture2D<float4> rendereTarget_Final;
[[vk::binding(BindingPoints_TAA::eDepthImage)]] Texture2D<float> depthImage;
[[vk::binding(BindingPoints_TAA::eVelocityImage)]] Sampler2D<float4> velocityImage;

float3 RGB2YCoCgR(float3 rgbColor) {
    float3 YCoCgRColor;

    YCoCgRColor.y = rgbColor.r - rgbColor.b;
    float temp = rgbColor.b + YCoCgRColor.y / 2;
    YCoCgRColor.z = rgbColor.g - temp;
    YCoCgRColor.x = temp + YCoCgRColor.z / 2;

    return YCoCgRColor;
}
float3 YCoCgR2RGB(float3 YCoCgRColor) {
    float3 rgbColor;

    float temp = YCoCgRColor.x - YCoCgRColor.z / 2;
    rgbColor.g = YCoCgRColor.z + temp;
    rgbColor.b = temp - YCoCgRColor.y / 2;
    rgbColor.r = rgbColor.b + YCoCgRColor.y;

    return rgbColor;
}
float Luminance(float3 color){
    return 0.25 * color.r + 0.5 * color.g + 0.25 * color.b;
}
float3 ToneMap(float3 color){
    return color / (1 + Luminance(color));
}
float3 UnToneMap(float3 color){
    return color / (1 - Luminance(color));
}
uint2 getClosestScreenPos(int2 screenPos){
    float2 deltaRes = 1.0f / pushConst.screenSize;
    float closestDepth = 1.0f;
    uint2 closestScreenPos = screenPos;

    for (int i = -1; i <= 1; ++i){
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
float3 clipAABB(float3 nowColor, float3 preColor, int2 screenPos) {
    float3 aabbMin = nowColor, aabbMax = nowColor;
    float2 deltaRes = 1.0f / pushConst.screenSize;
    float3 m1 = float3(0), m2 = float3(0);

    for (int i = -1; i <= 1; ++i){
        for (int j = -1; j <= 1; ++j) {
            uint2 newScreenPos = clamp(screenPos + uint2(i, j), 0, pushConst.screenSize - 1);
            float2 uv = (float2(newScreenPos) + 0.5f) * deltaRes;
            float3 C = RGB2YCoCgR(ToneMap(rendereTarget_Current.SampleLevel(uv, 0).xyz));
            m1 += C;
            m2 += C * C;
        }
    }

    // Variance clip
    const int N = 9;
    const float VarianceClipGamma = 1.0f;
    float3 mu = m1 / N;
    float3 sigma = sqrt(abs(m2 / N - mu * mu));
    aabbMin = mu - VarianceClipGamma * sigma;
    aabbMax = mu + VarianceClipGamma * sigma;

    // clip to center
    float3 p_clip = 0.5 * (aabbMax + aabbMin);
    float3 e_clip = 0.5 * (aabbMax - aabbMin);

    float3 v_clip = preColor - p_clip;
    float3 v_unit = v_clip.xyz / max(e_clip, 1e-6f);
    float3 a_unit = abs(v_unit);
    float ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));

    if (ma_unit > 1.0)  return p_clip + v_clip / ma_unit;
    else return preColor;
}

[numthreads(32, 32, 1)]
[shader("compute")]
void computeMain_mergeRenderTarget(uint2 threadIndex: SV_DispatchThreadID) {
    if (any(threadIndex >= pushConst.screenSize)) return;

    float2 uv = ((float2)threadIndex + 0.5f) / pushConst.screenSize;
    uv -= pushConst.jitter;

    float4 result_Current = rendereTarget_Current.SampleLevel(uv, 0);
    if (pushConst.frameIndex == 0) {
        rendereTarget_Final[threadIndex] = result_Current;
        return;
    }

    uint2 closetScreenPos = getClosestScreenPos(threadIndex);
    float2 velocity = velocityImage.SampleLevel((float2(closetScreenPos) + 0.5f) / pushConst.screenSize, 0).xy;

    float2 screenPos = (float2(threadIndex) + 0.5f) / pushConst.screenSize;
    float2 offsetScreenPos = screenPos - velocity;
    if (any(offsetScreenPos < 0.0f) || any(offsetScreenPos > 1.0f)) {
        rendereTarget_Final[threadIndex] = result_Current;
        return;
    }
    float4 result_Last = rendereTarget_Last.SampleLevel(offsetScreenPos, 0);

    float3 color_Current = RGB2YCoCgR(ToneMap(result_Current.xyz));
    float3 color_Last = RGB2YCoCgR(ToneMap(result_Last.xyz));

    color_Last = clipAABB(color_Current, color_Last, threadIndex);

    color_Current = UnToneMap(YCoCgR2RGB(color_Current));
    color_Last = UnToneMap(YCoCgR2RGB(color_Last));

    float3 result = lerp(color_Last, color_Current, pushConst.mergeRatio);
    rendereTarget_Final[threadIndex] = float4(result, 1.0f);
}
