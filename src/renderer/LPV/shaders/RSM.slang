//#define LPV_DEBUG
#include "renderer/LPV/shaders/LPVCommon.slang"

bool LPV_RsmTexel(int2 texel, int lightIndex, int rsmResolution, out float3 flux, out float surfelArea, out LPVHit hit) {
    flux = float3(0.0, 0.0, 0.0);
    surfelArea = 0.0;
    hit.hit = false;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    LPVFrameInfo frame = frameInfo[0];
    Light light = sceneInfo.lights[lightIndex];
    if (light.type == int(LightType::Non)) return false;

    float2 uv = (float2(texel) + 0.5) / float(rsmResolution);
    float2 clip = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    float4x4 inverseLightViewProj = frame.lightViewProjInverse[lightIndex];
    float4 nearPoint = mul(float4(clip, 0.0, 1.0), inverseLightViewProj);
    float4 farPoint = mul(float4(clip, 1.0, 1.0), inverseLightViewProj);
    if (abs(nearPoint.w) < 1e-8 || abs(farPoint.w) < 1e-8) return false;
    nearPoint /= nearPoint.w;
    farPoint /= farPoint.w;

    float3 rayDirection = farPoint.xyz - nearPoint.xyz;
    if (dot(rayDirection, rayDirection) < 1e-12) return false;
    rayDirection = normalize(rayDirection);

    RayDesc ray = LPV_SpawnRay(nearPoint.xyz + rayDirection * 1e-3, rayDirection);
    hit = LPV_TraceRay(ray);
    if (!hit.hit) return false;

    float3 toLight;
    float3 radiance;
    float distanceToLight;
    if (!LPV_LightDirection(light, hit.position, toLight, radiance, distanceToLight)) return false;

    float cosLight = dot(hit.normal, toLight);
    if (cosLight <= 1e-3) return false;

    float3 albedo = LPV_AlbedoOfHit(hit);
    if (max(albedo.x, max(albedo.y, albedo.z)) <= 0.0) return false;

    /* Footprint of one RSM texel: an area perpendicular to the *light ray*, not to the surface.
       Orthographic light: constant texel area. Perspective light: the texel subtends a constant solid
       angle, so its footprint at the hit distance is d^2 * dOmega. */
    float texelFootprint = 0.0;
    if (light.type == int(LightType::Direction)) texelFootprint = frame.lightRsmParams[lightIndex].x;
    else texelFootprint = distanceToLight * distanceToLight * frame.lightRsmParams[lightIndex].y;
    if (texelFootprint <= 0.0) return false;

    float cosClamped = max(cosLight, 0.05);
    surfelArea = texelFootprint / cosClamped;
    flux = albedo * (radiance * cosClamped) * surfelArea * LPV_INV_PI;
    return true;
}
#ifndef LPV_DEBUG
[numthreads(LPV_RSM_THREADGROUP_X, LPV_RSM_THREADGROUP_Y, 1)]
[shader("compute")] 
void lpvRsmInjectMain(uint2 threadIndex: SV_DispatchThreadID){
    int rsmResolution = frameInfo[0].info.z;
    if (any(int2(threadIndex) >= int2(rsmResolution, rsmResolution))) return;

    float3 flux;
    float surfelArea;
    LPVHit hit;
    if (!LPV_RsmTexel(int2(threadIndex), pushConst.lightIndex, rsmResolution, flux, surfelArea, hit)) return;

    float4 lobe = LPV_SH_Lobe(hit.normal);
    int cascadeCount = LPV_CascadeCount();
    for (int cascade = 0; cascade < cascadeCount; ++cascade) {
        float cellSize = LPV_CascadeCellSize(cascade);
        if (LPV_ObjectSize(hit.instanceIndex) < cellSize) continue;

        float3 injectionPosition = hit.position + hit.normal * (0.5 * cellSize);
        int3 cell = int3(floor((injectionPosition - LPV_CascadeOrigin(cascade)) / cellSize));
        if (!LPV_CellInside(cell, LPV_GridResolution())) continue;

        LPV_AtomicAddVolumeSH(0, cascade, cell, LPV_MakeSH(flux.x * lobe, flux.y * lobe, flux.z * lobe));
        LPV_InjectBlocking(LPVGeometrySource::eLPVGeometryFromRsm, cascade, hit.position, hit.normal, surfelArea);
    }
}
#endif
//------------------------------------------------------------------------------------------------------------
#ifdef LPV_DEBUG
[numthreads(16, 16, 1)]
[shader("compute")]
void showRSMMain(uint2 threadIndex: SV_DispatchThreadID) {
    if (any(threadIndex >= pushConst.screenSize)) return;

    int rsm = max(frameInfo[0].info.z, 1);
    int2 texel = clamp(int2(float2(threadIndex) / float2(pushConst.screenSize) * float(rsm)), int2(0, 0), int2(rsm - 1, rsm - 1));
    float3 flux; float surfelArea; LPVHit rsmHit;
    LPV_RsmTexel(texel, pushConst.lightIndex, rsm, flux, surfelArea, rsmHit);
    outImage[threadIndex] = float4(flux, 1.0);
}
#endif