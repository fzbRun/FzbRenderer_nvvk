#include "renderer/LPV/shaders/LPVCommon.slang"

bool LPV_WorldToCascade(int cascade, float3 position, out float3 uvw)
{
    float cellSize = LPV_CascadeCellSize(cascade);
    uvw = (position - LPV_CascadeOrigin(cascade)) / cellSize;
    float resolution = float(LPV_GridResolution());
    // A one cell margin keeps the transition between two cascades continuous.
    return all(uvw >= float3(-1.0, -1.0, -1.0)) && all(uvw <= float3(resolution + 1.0, resolution + 1.0, resolution + 1.0));
}

// Indirect radiance of one cascade: trilinear SH lookup, irradiance with the half cell distance
// assumption and the derivative dampening of paper section 3.4.
float3 LPV_IndirectFromCascade(int cascade, float3 position, float3 normal, float3 albedo){
    float cellSize = LPV_CascadeCellSize(cascade);

    /* Paper 3.4: "we query the intensity by a tri-linearly interpolated lookup of the SH
       coefficients" - at the shading point itself, no offset. */
    LPVSH sh = LPV_SampleAccum(cascade, position);

    float dampening = 1.0;
    if ((pushConst.flags & LPVFlags::eLPVFlagDampening) != 0)
    {
        /* Paper 3.4: "we determine the tri-linearly interpolated SH coefficients c_l,m and the
           directional derivative in normal direction (computed via differencing). Whenever the
           derivative is large ... we dampen c_l,m before computing the lighting." The paper gives no
           closed form, this is the implementation of that sentence (unit strength, as described). */
        float halfCell = 0.5 * cellSize;
        LPVSH shPlus = LPV_LoadAccumNearest(cascade, position + normal * halfCell);
        LPVSH shMinus = LPV_LoadAccumNearest(cascade, position - normal * halfCell);

        float4 c = sh.r + sh.g + sh.b;
        float4 d = (shPlus.r + shPlus.g + shPlus.b) - (shMinus.r + shMinus.g + shMinus.b);

        float leak = saturate(dot(d, c) / (dot(c, c) + 1e-6));
        float mag = saturate(2.0 * length(d) / (length(c) + 1e-6));
        dampening = saturate(1.0 - smoothstep(0.0, 1.0, leak) * mag);
    }

    /* Paper 3.4: "we assume that the distance between the cell's center (where the intensity is
       assumed to be) and the surface to be lit is half the grid size s", so E = I * (2/s)^2.

       LPV_SH_EvalIrradiance() convolves with the cosine lobe along -normal, see the convention note in
       LPVCommon.slang: the field is indexed by the travel direction, so the lobe has to be evaluated
       against the surface normal. */
    float inverseDistanceSquared = 4.0 / (cellSize * cellSize);
    float3 irradiance = LPV_SH_EvalIrradiance(sh, normal) * inverseDistanceSquared * dampening;
    irradiance = max(irradiance, float3(0.0, 0.0, 0.0));

    return albedo * LPV_INV_PI * irradiance;
}

/* Indirect lighting with the cascades (paper section 4.1):

     "When using the LPVs for lighting, we look up the finest grid at the respective location. At the
      boundary of a fine grid we create a smooth transition to the next-coarser grid by interpolation
      between both levels."

   So exactly two levels take part: the finest grid containing the point, and the next coarser one,
   blended by the distance to the border of the finer grid (0 inside, 1 at the border). */
float3 LPV_IndirectLighting(float3 position, float3 normal, float3 albedo)
{
    int cascadeCount = LPV_CascadeCount();

    int cascade = -1;
    float3 uvw;
    for (int candidate = 0; candidate < cascadeCount; ++candidate)
    {
        if (LPV_WorldToCascade(candidate, position, uvw))
        {
            cascade = candidate;
            break;
        }
    }
    if (cascade < 0)
        return float3(0.0, 0.0, 0.0);

    float3 radiance = LPV_IndirectFromCascade(cascade, position, normal, albedo);
    if (cascade + 1 >= cascadeCount)
        return radiance;

    // Interpolation weight towards the next coarser grid: 0 one cell inside, 1 at the border.
    float resolution = float(LPV_GridResolution());
    float border = min(min(uvw.x, resolution - uvw.x),
                       min(min(uvw.y, resolution - uvw.y), min(uvw.z, resolution - uvw.z)));
    float blend = smoothstep(0.0, 1.0, 1.0 - saturate(border));

    float3 coarse = LPV_IndirectFromCascade(cascade + 1, position, normal, albedo);
    return lerp(radiance, coarse, blend);
}

/* Glossy / mirror indirect light (paper section 7): march a few cells along the reflection
   direction and average the intensity stored in the LPV, weighted by 1/d^2. This gives specular
   surfaces a plausible reflection of the propagated light instead of leaving them black, which is
   what a pure NEE + punctual light shading would do. */
float3 LPV_GlossyReflection(float3 position, float3 normal, float3 specular, float3 outgoing)
{
    float3 result = float3(0.0, 0.0, 0.0);
    if (max(specular.x, max(specular.y, specular.z)) <= 0.0)
        return result;

    int cascadeCount = LPV_CascadeCount();
    int cascade = -1;
    float3 uvw;
    for (int candidate = 0; candidate < cascadeCount; ++candidate)
    {
        if (LPV_WorldToCascade(candidate, position, uvw))
        {
            cascade = candidate;
            break;
        }
    }
    if (cascade < 0)
        return result;

    // Mirror the view vector about the normal to obtain the reflection direction.
    float3 reflectionDirection = 2.0 * dot(normal, outgoing) * normal - outgoing;
    if (dot(reflectionDirection, normal) <= 0.0)
        return result;

    float cellSize = LPV_CascadeCellSize(cascade);
    float3 sum = float3(0.0, 0.0, 0.0);
    const int stepCount = 4;
    for (int step = 1; step <= stepCount; ++step)
    {
        float distance = float(step) * cellSize;
        float3 probePos = position + reflectionDirection * distance;
        LPVSH sh = LPV_SampleAccum(cascade, probePos);
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

//------------------------------------------------------------------------------------------
// Pass 5: final gather
//------------------------------------------------------------------------------------------
[numthreads(LPV_PIXEL_THREADGROUP_X, LPV_PIXEL_THREADGROUP_Y, 1)]
[shader("compute")] void lpvGatherMain(uint2 threadIndex: SV_DispatchThreadID)
{
    if (any(threadIndex >= pushConst.screenSize)) return;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    uint pixelIndex = threadIndex.y * pushConst.screenSize.x + threadIndex.x;
    LPVPixelData data = pixelData[pixelIndex];

    if (data.positionValid.w < 0.5){
        RayDesc cameraRay = LPV_CameraRay(threadIndex, float2(0.5, 0.5));
        float3 backgroundColor = sceneInfo.useSky == 1
                                     ? evalSimpleSky(sceneInfo.skySimpleParam, cameraRay.Direction)
                                                       : sceneInfo.backgroundColor;
        outImage[threadIndex] = float4(backgroundColor, 1.0);
        return;
    }

    float3 color = (pushConst.flags & int(LPVFlags::eLPVFlagNoDirect)) != 0 ? float3(0.0, 0.0, 0.0) : data.directColor.xyz;
    float3 indirect = LPV_IndirectLighting(data.positionValid.xyz, normalize(data.normal.xyz), data.albedo.xyz);
    color += indirect;
    color += LPV_GlossyReflection(data.positionValid.xyz, normalize(data.normal.xyz), data.specular.xyz, -LPV_CameraRay(threadIndex, float2(0.5, 0.5)).Direction);
    outImage[threadIndex] = float4(color, 1.0);
}