#pragma once

//------------------------------------------------------------------------------------------
// Compute passes of the Light Propagation Volumes renderer. One module, one entry point per pass:
//
//   lpvGBufferMain    camera G-buffer (primary hit) + direct lighting + blocking of the camera surfels
//   lpvRsmInjectMain  virtual reflective shadow map: traces the light view, injects VPLs and blockers
//   lpvGvMergeMain    merges the camera and RSM geometry volumes with a maximum
//   lpvPropagateMain  one 6 face gather iteration (ping pong, accumulated)
//   lpvGatherMain     final gather: direct + indirect, trilinear lookup, dampening, cascade blend
//------------------------------------------------------------------------------------------

#include "renderer/LPV/shaders/LPVCommon.slang"

//------------------------------------------------------------------------------------------
// Direct lighting, evaluated from the primary hit. The BSDF convention of this project is
// getBSDF(material, incidence, normal, outgoing, TBN, isExt) where `incidence` points *from the
// surface towards the light* (the same convention as CallablePayload::incidence) and `outgoing`
// points towards the camera; the cosine is applied here.
//------------------------------------------------------------------------------------------
float3 LPV_DirectLighting(LPVHit hit, BSDFMaterial material, float3x3 TBN, bool isExt, float3 outgoing)
{
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    float3    result    = float3(0.0, 0.0, 0.0);

    int lightCount = min(sceneInfo.numLights, LIGHT_COUNT);
    for (int lightIndex = 0; lightIndex < lightCount; ++lightIndex)
    {
        Light light = sceneInfo.lights[lightIndex];
        float3 direction;
        float3 radiance;
        float  distanceToLight;
        if (!LPV_LightDirection(light, hit.position, direction, radiance, distanceToLight))
            continue;

        float cosine = dot(direction, hit.normal);
        if (cosine <= 0.0)
            continue;

        // Bit 6 of the flags drops the shadow ray (debug / correctness inspection only). It used to
        // be bit 5, which collides with debug view 4 and quietly unshadowed the Reference GI view.
        if ((pushConst.flags & int(LPVFlags::eLPVFlagNoShadow)) == 0
            && LPV_Occluded(hit.position, direction, distanceToLight))
            continue;

        float3 bsdf = getBSDF(material, direction, hit.normal, outgoing, TBN, isExt);
        result += radiance * bsdf * cosine;
    }
    return result;
}

// CallablePayload used only to reuse the project's material evaluation (textures, normal map).
void LPV_MakePayload(inout CallablePayload payload, LPVHit hit, float3 outgoing, uint seed, SceneInfo sceneInfo)
{
    payload.material              = sceneInfo.materials[hit.materialIndex];
    payload.hitPos                = hit.position;
    payload.hitNormal             = hit.normal;
    payload.outgoing              = outgoing;
    payload.isExt                 = hit.frontFacing;
    payload.randomSeed            = seed;
    payload.bsdf                  = float3(0.0, 0.0, 0.0);
    payload.pdf                   = 1.0;
    payload.incidence             = float3(0.0, 0.0, 0.0);
    payload.directionLightDir     = float3(0.0, 0.0, 0.0);
    payload.directionLightRadiance = float3(0.0, 0.0, 0.0);

    float3 tangent, bitangent;
    orthonormalBasis(payload.hitNormal, tangent, bitangent);
    payload.TBN = float3x3(tangent, bitangent, payload.hitNormal);

    if (hit.hasTexCoords)
        getTextueData(payload, hit.texCoords);
}

//------------------------------------------------------------------------------------------
// Pass 1: camera G-buffer
//------------------------------------------------------------------------------------------
[numthreads(LPV_PIXEL_THREADGROUP_X, LPV_PIXEL_THREADGROUP_Y, 1)]
[shader("compute")] void lpvGBufferMain(uint2 threadIndex : SV_DispatchThreadID)
{
    if (any(threadIndex >= pushConst.screenSize))
        return;

    SceneInfo sceneInfo  = pushConst.sceneInfoAddress[0];
    uint      pixelIndex = threadIndex.y * pushConst.screenSize.x + threadIndex.x;

    RayDesc cameraRay = LPV_CameraRay(threadIndex, float2(0.5, 0.5));
    LPVHit  hit       = LPV_TraceRay(cameraRay);

    LPVPixelData data;
    if (!hit.hit)
    {
        data.positionValid = float4(0.0, 0.0, 0.0, 0.0);
        data.normal        = float4(0.0, 1.0, 0.0, 0.0);
        data.albedo        = float4(0.0, 0.0, 0.0, 0.0);
        data.directColor   = float4(0.0, 0.0, 0.0, 0.0);
        data.specular      = float4(0.0, 0.0, 0.0, 0.0);
        pixelData[pixelIndex] = data;
        return;
    }

    float3 outgoing = -cameraRay.Direction;

    // Material evaluation, exactly like the other compute renderers of this project.
    CallablePayload payload;
    LPV_MakePayload(payload, hit, outgoing, 0u, sceneInfo);
    BSDFMaterial material = payload.material;
    float3       normal   = payload.hitNormal;

    // Direct lighting (shadow ray per punctual light) + emissive.
    float3 direct = material.emissive + LPV_DirectLighting(hit, material, payload.TBN, hit.frontFacing, outgoing);

    // Diffuse albedo for the indirect term: LPV models the diffuse bounce only, purely specular
    // BSDFs (conductor / dielectric) must not receive an extra diffuse lobe.
    float3 diffuseAlbedo = float3(0.0, 0.0, 0.0);
    if (material.type == MaterialType::Diffuse || material.type == MaterialType::RoughPlastic)
        diffuseAlbedo = material.albedo;

    // Specular reflectance, used by the glossy LPV lookup of the gather pass (paper section 7).
    float3 specular = float3(0.0, 0.0, 0.0);
    if (material.type == MaterialType::RoughPlastic)
        specular = material.albedo_specular;
    else if (material.type == MaterialType::Conductor || material.type == MaterialType::RoughConductor ||
             material.type == MaterialType::Dielectric || material.type == MaterialType::RoughDielectric)
        specular = material.albedo;

    data.positionValid = float4(hit.position, 1.0);
    data.normal        = float4(normal, 0.0);
    data.albedo        = float4(diffuseAlbedo, 0.0);
    data.directColor   = float4(direct, 0.0);
    data.specular      = float4(specular, 0.0);
    pixelData[pixelIndex] = data;

    /* Blocking potential of the camera surfel (paper section 3.2). One pixel covers an area
       perpendicular to the *view ray*; dividing by the observation cosine |n.v| undoes that
       projection and yields the true area of the surface patch, which is what the blocking potential
       needs (the projected footprint would be wrong by |n.v|). The clamp bounds the magnification at
       grazing angles (~87 degrees). */
    float  distance       = max(hit.distance, 1e-4);
    float3 viewDir        = normalize(hit.position - sceneInfo.cameraPosition);
    float  cosView        = max(abs(dot(normal, viewDir)), 0.05);
    float  pixelFootprint = distance * distance * LPV_PixelSolidAngle();
    float  surfelArea     = pixelFootprint / cosView;

    int cascadeCount = LPV_CascadeCount();
    for (int cascade = 0; cascade < cascadeCount; ++cascade)
        LPV_InjectBlocking(LPVGeometrySource::eLPVGeometryFromCamera, cascade, hit.position, normal, surfelArea);
}

//------------------------------------------------------------------------------------------
// Pass 2: virtual reflective shadow map
//
// Instead of rasterising an RSM, one ray per virtual RSM texel is traced from the light, which
// gives exactly the same VPL set (the surfaces visible from the light) without any raster state.
// The RSM is per construction "one layer" (no depth peeling); the paper uses the camera view to
// complete the geometry volume, which this renderer does in pass 1.
//------------------------------------------------------------------------------------------
/* One virtual RSM texel: trace the light ray and compute the reflected flux of the surface patch
   behind it (RSM paper, section 3.2). Returns false when the texel creates no VPL (miss, back facing,
   black material, degenerate footprint). Shared by the injection pass and the "RSM" debug view so
   that both always agree.

   Written with a single area so that the flux and the blocking potential use the same quantity:

       A_s  = texelFootprint / |n.l|          true area of the surface patch behind the texel
       E    = E_l * |n.l|                     irradiance received by that patch (radiance holds E_l)
       Phi  = rho * E * A_s = rho * E_l * texelFootprint

   The observation cosine therefore appears exactly once in E and once as 1/A_s, and cancels in the
   flux; using the projected footprint with the cosine-carrying E (or A_s with E_l) is the classic
   cos^2 / 1 error. The clamp is shared by both, so the cancellation is exact at grazing angles too.
   The emitted *direction* does not appear here at all: it is the clamped cosine lobe at injection.

   Section 3.1 stores the SH coefficients of the *intensity* distribution of the VPL, i.e. of
   I(w) = (Phi/pi) * max(0, n.w): the paper writes I_p(w) = Phi_p <n_p,w>_+, where Phi_p is the peak
   intensity that a VPL of total emitted flux pi*Phi_p carries, hence the 1/pi below. */
bool LPV_RsmTexel(int2 texel, int lightIndex, int rsmResolution, out float3 flux, out float surfelArea,
                  out LPVHit hit)
{
    flux       = float3(0.0, 0.0, 0.0);
    surfelArea = 0.0;
    hit.hit    = false;

    SceneInfo    sceneInfo = pushConst.sceneInfoAddress[0];
    LPVFrameInfo frame     = frameInfo[0];
    Light        light     = sceneInfo.lights[lightIndex];
    if (light.type == int(LightType::Non))
        return false;

    // Unproject the texel center onto the light near / far plane. For the orthographic case both
    // points share the texel position, for a perspective light they lie on the same ray.
    float2 uv   = (float2(texel) + 0.5) / float(rsmResolution);
    float2 clip = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    float4x4 inverseLightViewProj = frame.lightViewProjInverse[lightIndex];
    float4   nearPoint = mul(float4(clip, 0.0, 1.0), inverseLightViewProj);
    float4   farPoint  = mul(float4(clip, 1.0, 1.0), inverseLightViewProj);
    if (abs(nearPoint.w) < 1e-8 || abs(farPoint.w) < 1e-8)
        return false;
    nearPoint /= nearPoint.w;
    farPoint /= farPoint.w;

    float3 rayDirection = farPoint.xyz - nearPoint.xyz;
    if (dot(rayDirection, rayDirection) < 1e-12)
        return false;
    rayDirection = normalize(rayDirection);

    RayDesc ray = LPV_SpawnRay(nearPoint.xyz + rayDirection * 1e-3, rayDirection);
    hit = LPV_TraceRay(ray);
    if (!hit.hit)
        return false;

    float3 toLight;
    float3 radiance;
    float  distanceToLight;
    if (!LPV_LightDirection(light, hit.position, toLight, radiance, distanceToLight))
        return false;

    // Only front facing surfels emit: the RSM stores what the light actually sees.
    float cosLight = dot(hit.normal, toLight);
    if (cosLight <= 1e-3)
        return false;

    float3 albedo = LPV_AlbedoOfHit(hit);
    if (max(albedo.x, max(albedo.y, albedo.z)) <= 0.0)
        return false;

    /* Footprint of one RSM texel: an area perpendicular to the *light ray*, not to the surface.
       Orthographic light: constant texel area. Perspective light: the texel subtends a constant solid
       angle, so its footprint at the hit distance is d^2 * dOmega. */
    float texelFootprint = 0.0;
    if (light.type == int(LightType::Direction))
        texelFootprint = frame.lightRsmParams[lightIndex].x;
    else
        texelFootprint = distanceToLight * distanceToLight * frame.lightRsmParams[lightIndex].y;
    if (texelFootprint <= 0.0)
        return false;

    float cosClamped = max(cosLight, 0.05);
    surfelArea       = texelFootprint / cosClamped;
    flux             = albedo * (radiance * cosClamped) * surfelArea * LPV_INV_PI;
    return true;
}

[numthreads(LPV_RSM_THREADGROUP_X, LPV_RSM_THREADGROUP_Y, 1)]
[shader("compute")] void lpvRsmInjectMain(uint2 threadIndex : SV_DispatchThreadID)
{
    int rsmResolution = frameInfo[0].info.z;
    if (any(int2(threadIndex) >= int2(rsmResolution, rsmResolution)))
        return;

    float3 flux;
    float  surfelArea;
    LPVHit hit;
    if (!LPV_RsmTexel(int2(threadIndex), pushConst.lightIndex, rsmResolution, flux, surfelArea, hit))
        return;

    float4 lobe = LPV_SH_Lobe(hit.normal);

    int cascadeCount = LPV_CascadeCount();
    for (int cascade = 0; cascade < cascadeCount; ++cascade)
    {
        float cellSize = LPV_CascadeCellSize(cascade);
        /* Paper 4.1: "if the size of an object (determined using its bounding box) is smaller than
           the cell size of a grid, we do not create VPLs from this object by not rendering it to the
           RSM. This is to prevent aliasing artifacts." */
        if (LPV_ObjectSize(hit.instanceIndex) < cellSize)
            continue;

        /* Paper 3.1: "we virtually move each VPL by half the cell spacing in the direction of its
           normal, before determining the cell" - a fixed half cell, not a tunable. */
        float3 injectionPosition = hit.position + hit.normal * (0.5 * cellSize);
        int3   cell = int3(floor((injectionPosition - LPV_CascadeOrigin(cascade)) / cellSize));
        if (!LPV_CellInside(cell, LPV_GridResolution()))
            continue;

        LPV_AtomicAddVolumeSH(0, cascade, cell, LPV_MakeSH(flux.x * lobe, flux.y * lobe, flux.z * lobe));
        LPV_InjectBlocking(LPVGeometrySource::eLPVGeometryFromRsm, cascade, hit.position, hit.normal, surfelArea);
    }
}

//------------------------------------------------------------------------------------------
// Pass 3: merge the geometry volumes (component wise maximum, paper section 3.2)
//------------------------------------------------------------------------------------------
[numthreads(LPV_VOXEL_THREADGROUP, LPV_VOXEL_THREADGROUP, LPV_VOXEL_THREADGROUP)]
[shader("compute")] void lpvGvMergeMain(uint3 threadIndex : SV_DispatchThreadID)
{
    int resolution    = LPV_GridResolution();
    int cascadeCount  = LPV_CascadeCount();
    if (threadIndex.z >= uint(cascadeCount * resolution))
        return;

    int  cascade = int(threadIndex.z) / resolution;
    int3 cell    = int3(int(threadIndex.x), int(threadIndex.y), int(threadIndex.z) - cascade * resolution);
    if (!LPV_CellInside(cell, resolution))
        return;

    uint base = LPV_GvFloatIndex(cascade, cell, 0);
    for (int coefficient = 0; coefficient < LPV_GV_FLOATS; ++coefficient)
        gvMerged[base + coefficient] = max(gvCamera[base + coefficient], gvRsm[base + coefficient]);
}

//------------------------------------------------------------------------------------------
// Pass 4: one propagation iteration
//
// Gather formulation: for every cell and every face, the neighbouring cell's intensity is
// evaluated in the face direction, converted into the flux through that face (face solid angle
// 4*pi/6), attenuated by the geometry volume, reprojected into a new clamped cosine lobe and
// accumulated (paper section 3.3).
//------------------------------------------------------------------------------------------
int3 LPV_FaceAxis(int face)
{
    if (face == 0) return int3(1, 0, 0);
    if (face == 1) return int3(-1, 0, 0);
    if (face == 2) return int3(0, 1, 0);
    if (face == 3) return int3(0, -1, 0);
    if (face == 4) return int3(0, 0, 1);
    return int3(0, 0, -1);
}

[numthreads(LPV_VOXEL_THREADGROUP, LPV_VOXEL_THREADGROUP, LPV_VOXEL_THREADGROUP)]
[shader("compute")] void lpvPropagateMain(uint3 threadIndex : SV_DispatchThreadID)
{
    int resolution   = LPV_GridResolution();
    int cascadeCount = LPV_CascadeCount();
    if (threadIndex.z >= uint(cascadeCount * resolution))
        return;

    int  cascade = int(threadIndex.z) / resolution;
    int3 cell    = int3(int(threadIndex.x), int(threadIndex.y), int(threadIndex.z) - cascade * resolution);
    if (!LPV_CellInside(cell, resolution))
        return;

    int iteration = pushConst.iteration;
    int readVolume  = iteration & 1;
    int writeVolume = 1 - readVolume;

    float cellSize = LPV_CascadeCellSize(cascade);
    bool  useOcclusion = (pushConst.flags & LPVFlags::eLPVFlagOcclusion) != 0;

    float3 cellCenter = LPV_CellCenter(cascade, cell);
    LPVSH result = LPV_MakeSH(float4(0.0, 0.0, 0.0, 0.0), float4(0.0, 0.0, 0.0, 0.0), float4(0.0, 0.0, 0.0, 0.0));
    for (int offset = 0; offset < 6; ++offset) {
        int3 offset3 = LPV_FaceAxis(offset);
        int3 neighbour = cell - offset3;
        if (!LPV_CellInside(neighbour, resolution)) continue;

        float4 neighbourR = LPV_LoadVolumeSH(readVolume, cascade, neighbour, 0);
        float4 neighbourG = LPV_LoadVolumeSH(readVolume, cascade, neighbour, 1);
        float4 neighbourB = LPV_LoadVolumeSH(readVolume, cascade, neighbour, 2);

        float3 neighbourCenter = cellCenter - offset3 * cellSize;

        float4 blocking = 0.0f;
        if (useOcclusion && iteration > 0) {
            float3 faceCenter = cellCenter - 0.5 * cellSize * offset3;
            blocking = LPV_SampleGvMerged(cascade, faceCenter);
        }

        for (int face = 0; face < 6; ++face) {
            if (offset == face) continue;

            int3 axis = LPV_FaceAxis(face);
            float3 faceNormal = float3(axis.x, axis.y, axis.z);
            float3 faceCenter = cellCenter - 0.5 * cellSize * faceNormal;

            float3 propagateDir = normalize(faceCenter - neighbourCenter);

            float attenuation = 1.0;
            if (useOcclusion && iteration > 0) {
                float blockProb = saturate(dot(blocking, LPV_SH_Basis(-propagateDir)));
                attenuation = 1.0 - blockProb;
            }

            float4 basisValue = LPV_SH_Basis(propagateDir);
            float intensityR = max(0.0, dot(neighbourR, basisValue));
            float intensityG = max(0.0, dot(neighbourG, basisValue));
            float intensityB = max(0.0, dot(neighbourB, basisValue));

            float4 lobe = LPV_SH_Lobe(-faceNormal);
            float solidAngle = (offset / 2 == face / 2) ? LPV_DIRECT_FACE_SOLID_ANGLE : LPV_SIDE_FACE_SOLID_ANGLE;
            float scale = solidAngle * LPV_INV_PI * attenuation;
            result.r += (scale * intensityR) * lobe;
            result.g += (scale * intensityG) * lobe;
            result.b += (scale * intensityB) * lobe;
        }
    }

    LPV_StoreVolumeSH(writeVolume, cascade, cell, result);
    LPV_AddAccumSH(cascade, cell, result);
}

//------------------------------------------------------------------------------------------
// Final gather helpers
//------------------------------------------------------------------------------------------
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
float3 LPV_IndirectFromCascade(int cascade, float3 position, float3 normal, float3 albedo)
{
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
        LPVSH shPlus   = LPV_LoadAccumNearest(cascade, position + normal * halfCell);
        LPVSH shMinus  = LPV_LoadAccumNearest(cascade, position - normal * halfCell);

        float4 c = sh.r + sh.g + sh.b;
        float4 d = (shPlus.r + shPlus.g + shPlus.b) - (shMinus.r + shMinus.g + shMinus.b);

        float leak = saturate(dot(d, c) / (dot(c, c) + 1e-6));
        float mag  = saturate(2.0 * length(d) / (length(c) + 1e-6));
        dampening  = saturate(1.0 - smoothstep(0.0, 1.0, leak) * mag);
    }

    /* Paper 3.4: "we assume that the distance between the cell's center (where the intensity is
       assumed to be) and the surface to be lit is half the grid size s", so E = I * (2/s)^2.

       LPV_SH_EvalIrradiance() convolves with the cosine lobe along -normal, see the convention note in
       LPVCommon.slang: the field is indexed by the travel direction, so the lobe has to be evaluated
       against the surface normal. */
    float  inverseDistanceSquared = 4.0 / (cellSize * cellSize);
    float3 irradiance = LPV_SH_EvalIrradiance(sh, normal) * inverseDistanceSquared * dampening;
    irradiance        = max(irradiance, float3(0.0, 0.0, 0.0));

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

    int    cascade = -1;
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

    int    cascadeCount = LPV_CascadeCount();
    int    cascade      = -1;
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

    float  cellSize = LPV_CascadeCellSize(cascade);
    float3 sum      = float3(0.0, 0.0, 0.0);
    const int stepCount = 4;
    for (int step = 1; step <= stepCount; ++step)
    {
        float  distance = float(step) * cellSize;
        float3 probePos = position + reflectionDirection * distance;
        LPVSH  sh       = LPV_SampleAccum(cascade, probePos);
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

/* Unbiased single bounce reference estimator, used by the "Reference GI" debug view to validate
   and calibrate the LPV normalisation in the same frame / camera as the LPV result. Cosine
   weighted hemisphere sampling: with pdf = cos/pi the outgoing radiance is albedo * E[L_i]. */
float3 LPV_ReferenceIndirect(float3 position, float3 albedo, float3 normal, float3 outgoing, uint seed, int sampleCount)
{
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    float3    tangent, bitangent;
    orthonormalBasis(normal, tangent, bitangent);

    float3 sum = float3(0.0, 0.0, 0.0);
    for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
    {
        float2 xi = float2(rand(seed), rand(seed));
        float3 local = cosineSampleHemisphere(xi.x, xi.y);
        float3 direction = normalize(tangent * local.x + bitangent * local.y + normal * local.z);
        if (dot(direction, normal) <= 0.0)
            continue;

        RayDesc ray = LPV_SpawnRay(offsetRay(position, normal), direction);
        LPVHit  bounceHit = LPV_TraceRay(ray);
        if (!bounceHit.hit)
        {
            float3 background = sceneInfo.useSky == 1 ? evalSimpleSky(sceneInfo.skySimpleParam, direction) : sceneInfo.backgroundColor;
            sum += background;
            continue;
        }

        CallablePayload payload;
        LPV_MakePayload(payload, bounceHit, -direction, seed, sceneInfo);
        float3 direct = payload.material.emissive +
                        LPV_DirectLighting(bounceHit, payload.material, payload.TBN, bounceHit.frontFacing, -direction);
        sum += direct;
    }
    return albedo * sum / float(max(sampleCount, 1));
}
//------------------------------------------------------------------------------------------
// Pass 5: final gather
//------------------------------------------------------------------------------------------
[numthreads(LPV_PIXEL_THREADGROUP_X, LPV_PIXEL_THREADGROUP_Y, 1)]
[shader("compute")] void lpvGatherMain(uint2 threadIndex : SV_DispatchThreadID)
{
    if (any(threadIndex >= pushConst.screenSize))
        return;

    int debugView = (pushConst.flags >> 3) & 7;

    /* Debug view "RSM": the virtual RSM of the first light, one screen pixel per texel (nearest
       neighbour upscale). It is evaluated exactly like the injection pass (LPV_RsmTexel), so it shows
       the reflected flux that is actually injected - no separate RSM target, no stale frame. */
    if (debugView == LPVDebugView::eLPVDebugRsm)
    {
        int   rsm  = max(frameInfo[0].info.z, 1);
        int2  texel = clamp(int2(float2(threadIndex) / float2(pushConst.screenSize) * float(rsm)),
                            int2(0, 0), int2(rsm - 1, rsm - 1));
        float3 flux;
        float  surfelArea;
        LPVHit rsmHit;
        LPV_RsmTexel(texel, 0, rsm, flux, surfelArea, rsmHit);
        outImage[threadIndex] = float4(flux, 1.0);
        return;
    }

    SceneInfo    sceneInfo  = pushConst.sceneInfoAddress[0];
    uint         pixelIndex = threadIndex.y * pushConst.screenSize.x + threadIndex.x;
    LPVPixelData data       = pixelData[pixelIndex];

    if (data.positionValid.w < 0.5)
    {
        RayDesc cameraRay = LPV_CameraRay(threadIndex, float2(0.5, 0.5));
        float3 backgroundColor = sceneInfo.useSky == 1
                                     ? evalSimpleSky(sceneInfo.skySimpleParam, cameraRay.Direction)
                                     : sceneInfo.backgroundColor;
        outImage[threadIndex] = float4(backgroundColor, 1.0);
        return;
    }

    /* "Show direct light" toggle: with the flag set the direct term is dropped, so what remains is the
       LPV contribution (+ the glossy lookup), which is handy to judge the indirect term alone without
       switching to the "Indirect only" debug view (that one also disables the glossy term). */
    float3 color = (pushConst.flags & int(LPVFlags::eLPVFlagNoDirect)) != 0 ? float3(0.0, 0.0, 0.0)
                                                                           : data.directColor.xyz;

    if (debugView == LPVDebugView::eLPVDebugCascade)
    {
        // Debug view: cascade membership of the pixel.
        int    cascadeCount = LPV_CascadeCount();
        float3 debugColor   = float3(0.1, 0.1, 0.1);
        for (int candidate = 0; candidate < cascadeCount; ++candidate)
        {
            float3 uvw;
            if (LPV_WorldToCascade(candidate, data.positionValid.xyz, uvw))
            {
                debugColor = candidate == 0 ? float3(0.2, 1.0, 0.2)
                            : candidate == 1 ? float3(0.2, 0.6, 1.0)
                                             : float3(1.0, 0.5, 0.2);
                break;
            }
        }
        outImage[threadIndex] = float4(debugColor, 1.0);
        return;
    }
    if (debugView == LPVDebugView::eLPVDebugDirect)
    {
        outImage[threadIndex] = float4(color, 1.0);
        return;
    }

    float3 indirect = LPV_IndirectLighting(data.positionValid.xyz, normalize(data.normal.xyz), data.albedo.xyz);
    color += indirect;
    if (debugView == LPVDebugView::eLPVDebugIndirect)
    {
        outImage[threadIndex] = float4(indirect, 1.0);
        return;
    }
    if (debugView == LPVDebugView::eLPVDebugReference)
    {
        // Unbiased single bounce reference for the same pixel, see LPV_ReferenceIndirect.
        uint referenceSeed = xxhash32(uint3(threadIndex, uint(max(pushConst.frameIndex, 0))));
        float3 reference = LPV_ReferenceIndirect(data.positionValid.xyz, data.albedo.xyz, normalize(data.normal.xyz),
                                                -LPV_CameraRay(threadIndex, float2(0.5, 0.5)).Direction, referenceSeed, 8);
        outImage[threadIndex] = float4(reference, 1.0);
        return;
    }

    // Paper 7: glossy surfaces gather the propagated light along the reflection direction.
    color += LPV_GlossyReflection(data.positionValid.xyz, normalize(data.normal.xyz), data.specular.xyz,
                                  -LPV_CameraRay(threadIndex, float2(0.5, 0.5)).Direction);

    outImage[threadIndex] = float4(color, 1.0);
}
