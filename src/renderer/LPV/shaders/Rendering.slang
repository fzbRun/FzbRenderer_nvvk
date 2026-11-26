#include "renderer/LPV/shaders/LPVCommon.slang"

bool LPV_WorldToCascade(int cascade, float3 position, out float3 uvw){
    float cellSize = LPV_CascadeCellSize(cascade);
    uvw = (position - LPV_CascadeOrigin(cascade)) / cellSize;
    float resolution = float(LPV_CascadeResolution(cascade));
    return all(uvw >= float3(-1.0, -1.0, -1.0)) && all(uvw <= float3(resolution + 1.0, resolution + 1.0, resolution + 1.0));
}
LPVSH LPV_SampleAccumShading(int cascade, float3 position){
    if ((pushConst.flags & LPVFlags::eLPVFlagNearestGather) != 0)
        return LPV_LoadAccumNearest(cascade, position);
    return LPV_SampleAccum(cascade, position);
}
LPVSH LPV_SampleAccumOccluded(int cascade, float3 position)
{
    const int    n      = LPV_CascadeResolution(cascade);
    const float  s      = LPV_CascadeCellSize(cascade);
    const float3 origin = LPV_CascadeOrigin(cascade);

    const bool  useOcclusion = (pushConst.flags & LPVFlags::eLPVFlagOcclusion) != 0;
    const float strength     = saturate(pushConst.gatherOcclusion);
    // Off: take the original path bit for bit, so the toggle is a pure comparison switch.
    if (!useOcclusion || strength <= 0.0)
        return LPV_SampleAccumShading(cascade, position);
    const float4 blocking    = useOcclusion ? LPV_SampleGvMerged(cascade, position) : float4(0.0);

    if ((pushConst.flags & LPVFlags::eLPVFlagNearestGather) != 0){
        const int3   cell = int3(clamp(floor((position - origin) / s), float3(0.0), float3(n - 1)));
        const float3 rel  = LPV_CellCenter(cascade, cell) - position;
        const float3 d    = dot(rel, rel) > 1e-10 ? normalize(rel) : float3(0.0);
        const float  a    = 1.0 - strength * saturate(dot(blocking, LPV_SH_Basis(d)));
        return LPV_MakeSH(LPV_LoadAccumSH(cascade, cell, 0) * a, LPV_LoadAccumSH(cascade, cell, 1) * a,
                          LPV_LoadAccumSH(cascade, cell, 2) * a);
    }

    const float3 g  = (position - origin) / s - 0.5;
    const int3   i0 = int3(floor(g));
    const int3   i1 = min(i0 + int3(1), int3(n - 1));
    const float3 t  = clamp(g - float3(i0), 0.0, 1.0);

    LPVSH result = LPV_MakeSH(float4(0.0), float4(0.0), float4(0.0));
    for (int dz = 0; dz < 2; ++dz){
        const float wz = dz == 1 ? t.z : 1.0 - t.z;
        for (int dy = 0; dy < 2; ++dy){
            const float wy = dy == 1 ? t.y : 1.0 - t.y;
            for (int dx = 0; dx < 2; ++dx){
                const float w = (dx == 1 ? t.x : 1.0 - t.x) * wy * wz;
                if (w <= 0.0) continue;

                const int3   cell = int3(dx == 1 ? i1.x : i0.x, dy == 1 ? i1.y : i0.y, dz == 1 ? i1.z : i0.z);
                const float3 d    = float3(dx == 1 ? 0.577350269 : -0.577350269,
                                           dy == 1 ? 0.577350269 : -0.577350269,
                                           dz == 1 ? 0.577350269 : -0.577350269);
                const float  a    = 1.0 - strength * saturate(dot(blocking, LPV_SH_Basis(d)));
                const float  wa   = w * a;

                result.r += wa * LPV_LoadAccumSH(cascade, cell, 0);
                result.g += wa * LPV_LoadAccumSH(cascade, cell, 1);
                result.b += wa * LPV_LoadAccumSH(cascade, cell, 2);
            }
        }
    }
    return result;
}
float3 LPV_IndirectFromCascade(int cascade, float3 position, float3 normal, float3 albedo){
    float cellSize = LPV_CascadeCellSize(cascade);

    LPVSH sh = LPV_SampleAccumOccluded(cascade, position);

    float dampening = 1.0;
    if ((pushConst.flags & LPVFlags::eLPVFlagDampening) != 0){
        float halfCell = 0.5 * cellSize;
        LPVSH shPlus = LPV_LoadAccumNearest(cascade, position + normal * halfCell);
        LPVSH shMinus = LPV_LoadAccumNearest(cascade, position - normal * halfCell);

        float4 c = sh.r + sh.g + sh.b;
        float4 d = (shPlus.r + shPlus.g + shPlus.b) - (shMinus.r + shMinus.g + shMinus.b);

        float leak = saturate(dot(d, c) / (dot(c, c) + 1e-6));
        float mag = saturate(2.0 * length(d) / (length(c) + 1e-6));
        dampening = saturate(1.0 - smoothstep(0.0, 1.0, leak) * mag);
    }

    // Solid angle of one cell seen from a point half a cell away: s^2 / (s/2)^2 = 4 (dimensionless).
    const float cellSolidAngle = 4.0;
    float3 irradiance = LPV_SH_EvalIrradiance(sh, normal) * cellSolidAngle * dampening;
    irradiance = max(irradiance, float3(0.0, 0.0, 0.0));

    return albedo * LPV_INV_PI * irradiance;
}
float LPV_GatherTransmittanceDebug(int cascade, float3 position)
{
    const int    n      = LPV_CascadeResolution(cascade);
    const float  s      = LPV_CascadeCellSize(cascade);
    const float3 origin = LPV_CascadeOrigin(cascade);

    const bool  useOcclusion = (pushConst.flags & LPVFlags::eLPVFlagOcclusion) != 0;
    const float strength     = saturate(pushConst.gatherOcclusion);
    if (!useOcclusion || strength <= 0.0)
        return 1.0;

    const float4 blocking = LPV_SampleGvMerged(cascade, position);
    const float3 g  = (position - origin) / s - 0.5;
    const int3   i0 = int3(floor(g));
    const int3   i1 = min(i0 + int3(1), int3(n - 1));
    const float3 t  = clamp(g - float3(i0), 0.0, 1.0);

    float sum = 0.0;
    float weights = 0.0;
    for (int dz = 0; dz < 2; ++dz)
    {
        const float wz = dz == 1 ? t.z : 1.0 - t.z;
        for (int dy = 0; dy < 2; ++dy)
        {
            const float wy = dy == 1 ? t.y : 1.0 - t.y;
            for (int dx = 0; dx < 2; ++dx)
            {
                const float w = (dx == 1 ? t.x : 1.0 - t.x) * wy * wz;
                if (w <= 0.0)
                    continue;
                const float3 d = float3(dx == 1 ? 0.577350269 : -0.577350269,
                                        dy == 1 ? 0.577350269 : -0.577350269,
                                        dz == 1 ? 0.577350269 : -0.577350269);
                sum += w * (1.0 - strength * saturate(dot(blocking, LPV_SH_Basis(d))));
                weights += w;
            }
        }
    }
    return sum / max(weights, 1e-6);
}
float3 LPV_IndirectLighting(float3 position, float3 normal, float3 albedo){
    int cascadeCount = LPV_CascadeCount();

    if ((pushConst.flags & LPVFlags::eLPVFlagDebugCascadeId) != 0)
    {
        int cascade = -1;
        float3 uvw;
        for (int candidate = 0; candidate < cascadeCount; ++candidate)
            if (LPV_WorldToCascade(candidate, position, uvw)) { cascade = candidate; break; }
        return cascade < 0 ? float3(1.0, 0.0, 1.0)
                           : float3(cascade == 0 ? 1.0 : 0.0, cascade == 1 ? 1.0 : 0.0, cascade >= 2 ? 1.0 : 0.0);
    }
    if ((pushConst.flags & LPVFlags::eLPVFlagDebugTransmittance) != 0)
        return float3(LPV_GatherTransmittanceDebug(cascadeCount - 1, position));
    if ((pushConst.flags & LPVFlags::eLPVFlagDebugSH) != 0)
    {
        LPVSH debugSh = LPV_SampleAccumOccluded(cascadeCount - 1, position);
        float value   = length(debugSh.r.xyz + debugSh.g.xyz + debugSh.b.xyz) * 20.0;
        return float3(saturate(value));
    }

    float3 sum = float3(0.0, 0.0, 0.0);
    for (int cascade = 0; cascade < cascadeCount; ++cascade)
        sum += LPV_IndirectFromCascade(cascade, position, normal, albedo);
    return sum / float(max(cascadeCount, 1));
}
float3 LPV_GlossyReflection(float3 position, float3 normal, float3 specular, float3 outgoing){
    float3 result = float3(0.0, 0.0, 0.0);
    if (max(specular.x, max(specular.y, specular.z)) <= 0.0) return result;

    int cascadeCount = LPV_CascadeCount();
    const int cascade = cascadeCount - 1;
    if (cascade < 0) return result;

    // Mirror the view vector about the normal to obtain the reflection direction.
    float3 reflectionDirection = 2.0 * dot(normal, outgoing) * normal - outgoing;
    if (dot(reflectionDirection, normal) <= 0.0) return result;

    float cellSize = LPV_CascadeCellSize(cascade);
    float3 sum = float3(0.0, 0.0, 0.0);
    const int stepCount = 4;
    for (int step = 1; step <= stepCount; ++step){
        float distance = float(step) * cellSize;
        float3 probePos = position + reflectionDirection * distance;
        LPVSH sh = LPV_SampleAccumOccluded(cascade, probePos);
        /* Intensity (not irradiance: no cosine convolution) of the light that arrives along the
           reflection path, i.e. travelling along -r. */
        float3 intensity = max(LPV_SH_EvalIntensity(sh, -reflectionDirection), float3(0.0, 0.0, 0.0));
        sum += intensity / (distance * distance);
    }

    /* Paper 7: "compute the incident light from the reflection direction r by marching through the
       LPV and averaging the intensity I(-r) of all cells, divided by the squared distance to the cell
       that we pass through; in practice 4 cells proved to be sufficient." */
    float3 radiance = sum / float(stepCount);
    return specular * radiance;
}
//--------------------------------------------------------------------------------------------------------
[numthreads(LPV_PIXEL_THREADGROUP_X, LPV_PIXEL_THREADGROUP_Y, 1)]
[shader("compute")] void lpvGatherMain(uint2 threadIndex: SV_DispatchThreadID){
    if (any(threadIndex >= pushConst.screenSize)) return;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    uint pixelIndex = threadIndex.y * pushConst.screenSize.x + threadIndex.x;
    LPVPixelData data = pixelData[pixelIndex];

    if (data.positionValid.w < 0.5){
        RayDesc cameraRay = LPV_CameraRay(threadIndex, float2(0.5, 0.5));
        float3 backgroundColor = sceneInfo.useSky == 1 ? evalSimpleSky(sceneInfo.skySimpleParam, cameraRay.Direction) : sceneInfo.backgroundColor;
        outImage[threadIndex] = float4(backgroundColor, 1.0);
        return;
    }

    float3 color = (pushConst.flags & int(LPVFlags::eLPVFlagNoDirect)) != 0 ? float3(0.0, 0.0, 0.0) : data.directColor.xyz;
    float3 indirect = LPV_IndirectLighting(data.positionValid.xyz, normalize(data.normal.xyz), data.albedo.xyz);
    color += indirect;
    color += LPV_GlossyReflection(data.positionValid.xyz, normalize(data.normal.xyz), data.specular.xyz, -LPV_CameraRay(threadIndex, float2(0.5, 0.5)).Direction);
    outImage[threadIndex] = float4(color, 1.0);
}