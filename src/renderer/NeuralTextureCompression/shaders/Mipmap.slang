#include "renderer/NeuralTextureCompression/NeuralTextureCompressionShaderio.h"

[[vk::push_constant]] ConstantBuffer<NeuralTextureCompressionPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_NTC::eSourceTextures)]] Sampler2D sourceTextures[];
[[vk::binding(StaticBindingPoints_NTC::eMipmapSourceTextures)]] Sampler2D mipmapSourceTextures[];
[[vk::binding(StaticBindingPoints_NTC::eRenderedTexture_Reduced)]] RWTexture2D<float4> rendereredTexture_Reduced;
[[vk::binding(StaticBindingPoints_NTC::eRenderedTexture_Full)]] RWTexture2D<float4> rendereredTexture_Full;
[[vk::binding(StaticBindingPoints_NTC::eRenderedTexture_Full_2)]] RWTexture2D<float4> rendereredTexture_Full_2;
[[vk::binding(StaticBindingPoints_NTC::eLossTexture)]] RWTexture2D<float4> lossTexture;
//-------------------------------------------------------------------------------------------
#define PI 3.14159265358979323846f

// Simple mathematical helper functions for clamp and square.
[Differentiable]
float clamp(float x, float x_min, float x_max){
    return max(min(x, x_max), x_min);
}

[Differentiable]
float sqr(float x){
    return x * x;
}

// Schlick Fresnel approximation for specular reflections.
[Differentiable]
float SchlickFresnel(float u){
    float m = clamp(1 - u, 0, 1);
    float m2 = m * m;
    return m2 * m2 * m; // pow(m,5)
}

[Differentiable]
float GTR1(float NdotH, float a){
    if (a >= 1)
        return 1 / PI;
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return (a2 - 1) / (PI * log(a2) * t);
}

[Differentiable]
float GTR2(float NdotH, float a){
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return a2 / (PI * t * t);
}

[Differentiable]
float smithG_GGX(float NdotV, float alphaG){
    float a = alphaG * alphaG;
    float b = NdotV * NdotV;
    return 1 / (NdotV + sqrt(a + b - a * b));
}

// Compute BRDF lighting for a given light drection (L), view direction (V) and
// normal vector (N), using given BRDF lighting properties.
[Differentiable]
float3 DisneyBRDF(float3 baseColor, float3 L, float3 V, float3 N, float roughness, float metallic, float specular){
    const float specularTint = 0;
    const float anisotropic = 0;
    const float sheenTint = .5;

    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    if (NdotL < 0 || NdotV < 0)
        return float3(0);

    float3 H = normalize(L + V);
    float NdotH = dot(N, H);
    float LdotH = dot(L, H);

    float3 Cdlin = baseColor;
    float Cdlum = .3f * Cdlin[0] + .6f * Cdlin[1] + .1f * Cdlin[2]; // luminance approx.

    float3 Ctint = Cdlum > 0 ? Cdlin / Cdlum : float3(1); // normalize lum. to isolate hue+sat
    float3 Cspec0 = lerp(specular * float3(.08f) *
        lerp(float3(1.f), Ctint, specularTint), Cdlin, metallic);
    float3 Csheen = lerp(float3(1), Ctint, sheenTint);

    // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
    // and lerp in diffuse retro-reflection based on roughness
    float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV);
    float Fd90 = 0.5f + 2 * LdotH * LdotH * roughness;
    float Fd = lerp(1.f, Fd90, FL) * lerp(1.f, Fd90, FV);

    // Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
    // 1.25 scale is used to (roughly) preserve albedo
    // Fss90 used to "flatten" retroreflection based on roughness
    float Fss90 = LdotH * LdotH * roughness;
    float Fss = lerp(1.f, Fss90, FL) * lerp(1.f, Fss90, FV);
    float ss = 1.25f * (Fss * (1 / (NdotL + NdotV) - .5f) + .5f);

    // specular
    float a = sqr(roughness);
    float Ds = GTR2(NdotH, a);
    float FH = SchlickFresnel(LdotH);
    float3 Fs = lerp(Cspec0, float3(1), FH);
    float Gs;
    Gs = smithG_GGX(NdotL, a);
    Gs *= smithG_GGX(NdotV, a);

    return ((1 / PI) * Fd * Cdlin) *
        (1 - metallic) + Gs * Fs * Ds;
}

[Differentiable]
public float3 eval_brdf(float3 albedo, float3 light_dir, float3 view_dir, float3 normal, float roughness, float metallic, float specular){
    float3 brdf = DisneyBRDF(albedo, light_dir, view_dir, normal, roughness, metallic, specular);
    brdf *= max(0, dot(normal, light_dir));
    return brdf;
}
//-------------------------------------------------------------------------------------------
float3 sampleDiffuseTexture(float2 texCoords, bool mipmap) {
    texCoords = clamp(texCoords, 0.0f, 1.0f);
    return mipmap ? mipmapSourceTextures[0].SampleLevel(texCoords, 0).xyz : sourceTextures[0].SampleLevel(texCoords, 0).xyz;
}
float3 sampleNormalTexture(float2 texCoords, bool mipmap) {
    texCoords = clamp(texCoords, 0.0f, 1.0f);
    float3 localNormal = mipmap ? mipmapSourceTextures[1].SampleLevel(texCoords, 0).xyz : sourceTextures[1].SampleLevel(texCoords, 0).xyz;
    return normalize(localNormal * 2.0f - 1.0f);

    //float3 normal = float3(0.0f, 0.0f, 1.0f);
    //float3 tangent = float3(1.0f, 0.0f, 0.0f);
    //float3 bitangent = float3(0.0f, 1.0f, 0.0f);
    //float3x3 TBN = float3x3(tangent, bitangent, normal);
    //return normalize(mul(localNormal, TBN));
}
float sampleRoughnessTexture(float2 texCoords, bool mipmap) {
    texCoords = clamp(texCoords, 0.0f, 1.0f);
    return mipmap ? mipmapSourceTextures[2].SampleLevel(texCoords, 0).x : sourceTextures[2].SampleLevel(texCoords, 0).x;
}
//-------------------------------------------------------------------------------------------
float3 downsample3(int2 pixel, Sampler2D sourceTexture){
    float3 res = 0;
    res += sourceTexture[pixel * 2 + int2(0, 0)].xyz;
    res += sourceTexture[pixel * 2 + int2(1, 0)].xyz;
    res += sourceTexture[pixel * 2 + int2(0, 1)].xyz;
    res += sourceTexture[pixel * 2 + int2(1, 1)].xyz;
    return res * 0.25;
}
float downsample1(int2 pixel, Sampler2D sourceTexture){
    float res = 0;
    res += sourceTexture[pixel * 2 + int2(0, 0)].x;
    res += sourceTexture[pixel * 2 + int2(1, 0)].x;
    res += sourceTexture[pixel * 2 + int2(0, 1)].x;
    res += sourceTexture[pixel * 2 + int2(1, 1)].x;
    return res * 0.25;
}
//-------------------------------------------------------------------------------------------
#define Light_Instensity 1.5f

[numthreads(16, 16, 1)]
[shader("compute")]
void computeMain_ReducedSizeMipmap(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    if (any(threadIndex >= pushConst.screenSize)) return;

    float2 texCoords = ((float2)threadIndex + 0.5f) / pushConst.screenSize;
    float3 albedo = sampleDiffuseTexture(texCoords, true);
    float3 normal = sampleNormalTexture(texCoords, true);
    float roughness = sampleRoughnessTexture(texCoords, true);

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    float3 ndcPos = float3(texCoords * 2.0f - 1.0f, 0.0f);
    float4 viewPos = mul(sceneInfo.projInvMatrix, float4(ndcPos, 1.0f));
    viewPos /= viewPos.w;

    float3 V = float3(0, 0, 1); //normalize(-viewPos.xyz);
    float3 L = normalize(float3(0.2, 0.2, 1));
    float3 brdf_eval = eval_brdf(albedo, L, V, normal, roughness, 0.0f, 1.0f);

    rendereredTexture_Reduced[threadIndex] = float4(brdf_eval * Light_Instensity, 1.0f);
}
//-------------------------------------------------------------------------------------------
[numthreads(16, 16, 1)]
[shader("compute")]
void computeMain_FullSizeMipmap(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    uint2 textureSize;
    sourceTextures[0].GetDimensions(textureSize.x, textureSize.y);
    if (any(threadIndex >= textureSize)) return;

    float2 texCoords = ((float2)threadIndex + 0.5f) / textureSize;
    float3 albedo = sampleDiffuseTexture(texCoords, false);
    float3 normal = sampleNormalTexture(texCoords, false);
    float roughness = sampleRoughnessTexture(texCoords, false);

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    float3 ndcPos = float3(texCoords * 2.0f - 1.0f, 0.0f);
    float4 viewPos = mul(sceneInfo.projInvMatrix, float4(ndcPos, 1.0f));
    viewPos /= viewPos.w;

    float3 V = float3(0, 0, 1); //normalize(-viewPos.xyz);
    float3 L = normalize(float3(0.2, 0.2, 1));
    float3 brdf_eval = eval_brdf(albedo, L, V, normal, roughness, 0.0f, 1.0f);

    rendereredTexture_Full[threadIndex] = float4(brdf_eval * Light_Instensity, 1.0f);
}
//-------------------------------------------------------------------------------------------
[numthreads(16, 16, 1)]
[shader("compute")]
void computeMain_getLossTexture(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    if (any(threadIndex >= pushConst.screenSize)) return;

    float3 reference = rendereredTexture_Full_2[threadIndex].rgb;
    float3 candidate = rendereredTexture_Reduced[threadIndex].rgb;
    float3 error = candidate - reference;

    lossTexture[threadIndex] = float4(error * error, 1.0f);
}
//------------------------------------------------------------------------------------------
float3 tonemap_aces_film(float3 x){
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}
float3 toSrgb(float3 c){
    float3 lo = c * 12.92f;
    float3 hi = 1.055f * pow(c, float3(1.0f / 2.4f)) - 0.055f;
    return lerp(lo, hi, float3(greaterThan(c, float3(0.0031308f))));
}

[[vk::binding(StaticBindingPoints_NTC::eTonemappedTexture)]] RWTexture2D<float4> tonemappedTexture;

// viewMode: 0 = reduced, 1 = full/ref, 2 = loss
[numthreads(16, 16, 1)]
[shader("compute")]
void computeMain_tonemapped(uint2 threadIndex: SV_DispatchThreadID)
{
    if (any(threadIndex >= pushConst.screenSize)) return;

    float3 color;
    if (pushConst.showTextureType == 2) color = lossTexture[threadIndex].rgb;
    else if (pushConst.showTextureType == 1) color = rendereredTexture_Full_2[threadIndex].rgb;
    else color = rendereredTexture_Reduced[threadIndex].rgb;

    if (pushConst.showTextureType != 2) color = tonemap_aces_film(color); // Ö»¶ÔäÖÈ¾Í¼×ö ACES

    tonemappedTexture[threadIndex] = float4(toSrgb(saturate(color)), 1.0f);
}