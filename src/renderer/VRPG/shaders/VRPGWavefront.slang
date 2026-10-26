//------------------------------------------------------------------------------------------
// Experimental wavefront path tracer for VRPG.
//
// This file is intentionally independent from the stable VRPG.slang single-dispatch
// path tester. It shares only the scene/AS bindings (pathTracingCommon.slang) and the
// VRPG push constant block, and writes to the same outImage so the two paths can be
// compared from the UI toggle.
//
// Three compute entry points are dispatched by VRPGRenderer::wavefrontPathTracing():
//   wavefrontInitMain    - camera ray generation + first ray trace, fills pathStates
//   wavefrontBounceMain  - shades the pending hit and traces the next segment (1 per dispatch)
//   wavefrontResolveMain - temporally accumulates pathStates into outImage
//
// Notes / limitations (prototype):
//   * One path per pixel per frame (pushConst.spp is ignored).
//   * BSDF sampling + NEE only: no VRPG path guiding and no MIS.
//------------------------------------------------------------------------------------------
#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "renderer/VRPG/VRPGShaderio.h"
#include "renderer/VRPG/shaders/VRPGGuide.slang"

[[vk::binding(StaticBindingPoints_VRPGWavefront::ePathState, 0)]]
RWStructuredBuffer<WavefrontPathState, ScalarDataLayout> pathStates;

#define FZB_WAVEFRONT_RR_PROBABILITY 0.8f

//------------------------------------------------------------------------------------------
// Scene helpers
//------------------------------------------------------------------------------------------
float3 wavefrontRayMiss(float3 rayDirection)
{
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    if (sceneInfo.useSky == 1) return evalSimpleSky(sceneInfo.skySimpleParam, rayDirection);
    return sceneInfo.backgroundColor;
}

struct WavefrontHitInfo
{
    bool   hit;
    bool   frontFacing;    // geometric normal faced the incoming ray
    float3 hitPos;
    float3 hitNormal;      // shading normal (normal map already applied)
    float2 texCoords;
    bool   hasTexCoords;
    uint   materialIndex;
};

// Inline ray query identical in spirit to the stable VRPG traceRay(), but returns the
// shading data through a plain struct so the caller can persist it in pathStates.
WavefrontHitInfo traceWavefrontRay(RayDesc ray)
{
    WavefrontHitInfo result;
    result.hit          = false;
    result.frontFacing  = true;
    result.hitPos       = float3(0.0f);
    result.hitNormal    = float3(0.0f, 1.0f, 0.0f);
    result.texCoords    = float2(0.0f);
    result.hasTexCoords = false;
    result.materialIndex = 0;

#ifndef NDEBUG
    // A NaN/Inf ray can hang traversal, so reject it in debug builds.
    if (any(isnan(ray.Origin)) || any(isinf(ray.Origin)) ||
        any(isnan(ray.Direction)) || any(isinf(ray.Direction)))
        return result;
#endif

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, ray);

    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            q.CommitNonOpaqueTriangleHit();
    }

    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        return result;

    float2 barycentricCoords = q.CommittedTriangleBarycentrics();
    float4x3 worldToObject  = q.CommittedWorldToObject4x3();
    float4x3 objectToWorld  = q.CommittedObjectToWorld4x3();
    uint instanceIndex      = q.CommittedInstanceIndex();
    uint primitiveIndex     = q.CommittedPrimitiveIndex();

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Instance instance   = sceneInfo.instances[instanceIndex];
    Mesh mesh           = sceneInfo.meshes[instance.meshIndex];
    BSDFMaterial material = sceneInfo.materials[instance.materialIndex];

    const float3 barycentrics =
        float3(1.0 - barycentricCoords.x - barycentricCoords.y, barycentricCoords.x, barycentricCoords.y);

    float3 triPos[3] = q.CommittedGetIntersectionTriangleVertexPositions();
    const float3 localPosition = triPos[0] * barycentrics.x + triPos[1] * barycentrics.y + triPos[2] * barycentrics.z;
    result.hitPos = float3(mul(float4(localPosition, 1.0), objectToWorld));

    const float3 edge1 = triPos[1] - triPos[0];
    const float3 edge2 = triPos[2] - triPos[0];
    const float3 geoNormal = normalize(cross(edge1, edge2));
    result.hitNormal = normalize(mul(geoNormal, transpose(worldToObject)).xyz);

    float cosineON = dot(result.hitNormal, -ray.Direction);
    result.frontFacing = cosineON > 0.0f;
    if (cosineON <= 0.0f) result.hitNormal = -result.hitNormal;

    float3 tangent, bitangent;
    orthonormalBasis(result.hitNormal, tangent, bitangent);
    float3x3 TBN = float3x3(tangent, bitangent, result.hitNormal); // column matrix

    if (material.materialMapIndex.x >= 0 || material.materialMapIndex.y >= 0 || material.materialMapIndex.z >= 0) {
        int3 indices = getTriangleIndices(mesh.dataBuffer, mesh.triMesh, primitiveIndex);
        float2 texCoords = getTriangleAttribute<float2>(mesh.dataBuffer, mesh.triMesh.texCoords, indices, barycentrics);
        texCoords.y = 1.0f - texCoords.y;

        // Reuse the common helper so normal mapping and textured BSDF parameters behave
        // exactly like the stable path. getTextueData() no-ops for uv (0,0).
        CallablePayload texturePayload;
        texturePayload.material  = material;
        texturePayload.hitNormal = result.hitNormal;
        texturePayload.TBN       = TBN;
        getTextueData(texturePayload, texCoords);

        result.hitNormal    = texturePayload.hitNormal;
        result.texCoords    = texCoords;
        result.hasTexCoords = (texCoords.x != 0.0f || texCoords.y != 0.0f);
    }

    result.materialIndex = instance.materialIndex;
    result.hit = true;
    return result;
}

// Rebuild the callable payload the common BSDF/NEE code expects from the persisted state.
CallablePayload loadWavefrontPayload(WavefrontPathState state)
{
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];

    CallablePayload payload;
    payload.material = sceneInfo.materials[state.info.z];

    if (state.uvFlags.z > 0.5f)
        getBSDFTextureData(payload.material, state.uvFlags.xy);

    if (payload.material.type != MaterialType::Dielectric &&
        payload.material.type != MaterialType::RoughDielectric)
        payload.material.emissive *= state.uvFlags.w > 0.5f ? 1.0f : 0.0f;

    payload.hitPos    = state.hitPos.xyz;
    payload.hitNormal = normalize(state.hitNormal.xyz);
    payload.outgoing  = normalize(state.outgoing.xyz);

    float3 tangent, bitangent;
    orthonormalBasis(payload.hitNormal, tangent, bitangent);
    payload.TBN = float3x3(tangent, bitangent, payload.hitNormal);

    payload.randomSeed = state.info.w;
    payload.bsdf       = float3(0.0f);
    payload.pdf        = 1.0f;
    payload.isExt      = true;
    return payload;
}

//------------------------------------------------------------------------------------------
// Entry: camera ray + first trace
//------------------------------------------------------------------------------------------
[numthreads(FZB_WAVEFRONT_THREADGROUP_SIZE_X, FZB_WAVEFRONT_THREADGROUP_SIZE_Y, 1)]
[shader("compute")]
void wavefrontInitMain(uint2 threadIndex : SV_DispatchThreadID)
{
    const uint2 sceneSize = pushConst.sceneSize;
    if (threadIndex.x >= sceneSize.x || threadIndex.y >= sceneSize.y) return;
    const uint pathIndex = threadIndex.y * sceneSize.x + threadIndex.x;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];

    WavefrontPathState state;
    state.accumulatedRadiance = float4(0.0f);
    state.throughput          = float4(1.0f);
    state.rayOrigin           = float4(0.0f);
    state.rayDirection        = float4(0.0f);
    state.hitPos              = float4(0.0f);
    state.hitNormal           = float4(0.0f, 1.0f, 0.0f, 0.0f);
    state.outgoing            = float4(0.0f);
    state.uvFlags             = float4(0.0f);
    state.info                = uint4(0, 0, 0, 0);

    uint seed = xxhash32(uint3(threadIndex, pushConst.randomFrameIndex));

    float r1 = rand(seed);
    float r2 = rand(seed);
    float2 subpixelJitter = pushConst.frameIndex == 0 ? float2(0.5f, 0.5f) : float2(r1, r2);

    const float2 launchID   = float2(threadIndex);
    const float2 launchSize = float2(sceneSize);
    const float2 pixelCenter = launchID + subpixelJitter;
    const float2 clipCoords  = pixelCenter / launchSize * 2.0 - 1.0;
    float4 viewCoords = mul(float4(clipCoords, 1.0, 1.0), sceneInfo.projInvMatrix);
    viewCoords /= viewCoords.w;

    RayDesc ray;
    ray.Origin    = sceneInfo.cameraPosition;
    ray.Direction = normalize(mul(float4(normalize(viewCoords.xyz), 0.0), sceneInfo.viewInvMatrix).xyz);
    ray.TMin      = 0.001;
    ray.TMax      = INFINITE;

    state.rayOrigin    = float4(ray.Origin, 0.0f);
    state.rayDirection = float4(ray.Direction, 0.0f);
    state.info.w       = seed;

    WavefrontHitInfo hitInfo = traceWavefrontRay(ray);
    if (!hitInfo.hit)
    {
        state.accumulatedRadiance = float4(wavefrontRayMiss(ray.Direction), 0.0f);
        state.info = uint4(0, 0, 0, seed); // inactive: nothing left to shade
    }
    else
    {
        state.hitPos   = float4(hitInfo.hitPos, 0.0f);
        state.hitNormal = float4(hitInfo.hitNormal, 0.0f);
        state.outgoing = float4(-ray.Direction, 0.0f);
        state.uvFlags  = float4(hitInfo.texCoords, hitInfo.hasTexCoords ? 1.0f : 0.0f,
                                hitInfo.frontFacing ? 1.0f : 0.0f);
        state.info = uint4(1, 0, hitInfo.materialIndex, seed);
    }

    pathStates[pathIndex] = state;
}

//------------------------------------------------------------------------------------------
// Entry: shade the pending hit and trace the next segment
//------------------------------------------------------------------------------------------
[numthreads(FZB_WAVEFRONT_THREADGROUP_SIZE_X, FZB_WAVEFRONT_THREADGROUP_SIZE_Y, 1)]
[shader("compute")]
void wavefrontBounceMain(uint2 threadIndex : SV_DispatchThreadID)
{
    const uint2 sceneSize = pushConst.sceneSize;
    if (threadIndex.x >= sceneSize.x || threadIndex.y >= sceneSize.y) return;
    const uint pathIndex = threadIndex.y * sceneSize.x + threadIndex.x;

    WavefrontPathState state = pathStates[pathIndex];
    if (state.info.x == 0) return;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    CallablePayload payload = loadWavefrontPayload(state);
    float3 currentHitPos = payload.hitPos;

    // ------------------------------------------------ guide sample
    VRSampleData vrSampleData;
    bool getVrSample = getNodeAABB_E_VRPG(vrSampleData, payload, threadIndex);

    WavefrontHitInfo guideHit;
    guideHit.hit = false;
    if (getVrSample && vrSampleData.pdf > 0.0f) {
        RayDesc guideRay;
        guideRay.Origin = offsetRay(currentHitPos, payload.hitNormal);
        guideRay.Direction = vrSampleData.sampleDir;
        guideRay.TMin = 0.001f;
        guideRay.TMax = INFINITE;
        guideHit = traceWavefrontRay(guideRay);
    }

    // ------------------------------------------------ BSDF sample
    getDirectionLight(payload, sceneInfo);
    BSDF_SAMPLE(payload);
    state.accumulatedRadiance.xyz +=
        (payload.material.emissive + payload.directionLightRadiance) * state.throughput.xyz;

    float3 bsdfDir = payload.incidence;
    float3 bsdfCos = payload.bsdf * abs(dot(bsdfDir, payload.hitNormal));
    float bsdfPdf = payload.pdf;
    if (!(bsdfPdf >= 0.0f) || isinf(bsdfPdf)) bsdfPdf = 0.0f;

    RayDesc bsdfRay;
    bsdfRay.Origin = offsetRay(currentHitPos, payload.hitNormal);
    bsdfRay.Direction = bsdfDir;
    bsdfRay.TMin = 0.001f;
    bsdfRay.TMax = INFINITE;
    WavefrontHitInfo bsdfHit = traceWavefrontRay(bsdfRay);

    // ------------------------------------------------ MIS weights
    bool guideValid = getVrSample && guideHit.hit && vrSampleData.pdf > 0.0f;
    float3 guideBsdfCos = float3(0.0f);
    float guideBsdfPdf = 0.0f;
    float guideWeight = 0.0f;
    if (guideValid) {
        float cosine = abs(dot(payload.hitNormal, vrSampleData.sampleDir));
        guideBsdfCos = cosine * getBSDF(payload.material, vrSampleData.sampleDir,
                                        payload.hitNormal, payload.outgoing, payload.TBN, payload.isExt);
        guideBsdfPdf = getPdf(payload.material, vrSampleData.sampleDir,
                              payload.hitNormal, payload.outgoing, payload.TBN, payload.isExt);
        if (!(guideBsdfPdf >= 0.0f) || isinf(guideBsdfPdf)) guideBsdfPdf = 0.0f;
        guideWeight = vrSampleData.pdf / max(vrSampleData.pdf + guideBsdfPdf, 1e-16f);
    }

    float bsdfPgPdf = 0.0f;
    if (guideValid && bsdfHit.hit) {
        bsdfPgPdf = getPgPdf2(vrSampleData.wSum, vrSampleData.nodeLabel_G, vrSampleData.outgoingIndex,
                              bsdfHit.hitPos, bsdfHit.hitNormal, currentHitPos,
                              bsdfDir, bsdfCos);
        if (!(bsdfPgPdf >= 0.0f) || isinf(bsdfPgPdf)) bsdfPgPdf = 0.0f;
    }
    float bsdfWeight = bsdfPdf / max(bsdfPdf + bsdfPgPdf, 1e-16f);

    bool guideAvailable = guideValid && guideWeight > 0.0f && any(guideBsdfCos > 0.0f);
    bool bsdfAvailable = bsdfPdf > 0.0f && bsdfWeight > 0.0f && any(bsdfCos > 0.0f);
    if (!guideAvailable && !bsdfAvailable) {
        state.info.x = 0;
        pathStates[pathIndex] = state;
        return;
    }

    float selectGuideProbability = guideAvailable ? (bsdfAvailable ? 0.5f : 1.0f) : 0.0f;
    bool chooseGuide = rand(payload.randomSeed) < selectGuideProbability;

    WavefrontHitInfo selectedHit;
    float3 selectedDir;
    float3 selectedCos;
    float selectedPdf;
    if (chooseGuide) {
        selectedHit = guideHit;
        selectedDir = vrSampleData.sampleDir;
        selectedCos = guideBsdfCos;
        selectedPdf = vrSampleData.pdf;
        float w = guideWeight / max(selectGuideProbability, 1e-16f);
        state.throughput.xyz *= w * selectedCos / max(selectedPdf, 1e-16f);
    } else {
        selectedHit = bsdfHit;
        selectedDir = bsdfDir;
        selectedCos = bsdfCos;
        selectedPdf = bsdfPdf;
        float w = bsdfWeight / max(1.0f - selectGuideProbability, 1e-16f);
        state.throughput.xyz *= w * selectedCos / max(selectedPdf, 1e-16f);
    }

    uint nextBounce = state.info.y + 1;
    state.info.y = nextBounce;
    state.info.w = payload.randomSeed;

    if (any(isnan(state.throughput.xyz)) || any(isinf(state.throughput.xyz)) ||
        all(state.throughput.xyz <= 0.0f) || nextBounce >= uint(pushConst.maxDepth)) {
        state.info.x = 0;
        pathStates[pathIndex] = state;
        return;
    }

    if (rand(payload.randomSeed) >= FZB_WAVEFRONT_RR_PROBABILITY) {
        state.info.x = 0;
        state.info.w = payload.randomSeed;
        pathStates[pathIndex] = state;
        return;
    }
    state.throughput.xyz /= FZB_WAVEFRONT_RR_PROBABILITY;

    if (!selectedHit.hit) {
        state.accumulatedRadiance.xyz += wavefrontRayMiss(normalize(selectedDir)) * state.throughput.xyz;
        state.info.x = 0;
    } else {
        state.hitPos = float4(selectedHit.hitPos, 0.0f);
        state.hitNormal = float4(selectedHit.hitNormal, 0.0f);
        state.outgoing = float4(-normalize(selectedDir), 0.0f);
        state.uvFlags = float4(selectedHit.texCoords,
                               selectedHit.hasTexCoords ? 1.0f : 0.0f,
                               selectedHit.frontFacing ? 1.0f : 0.0f);
        state.info.z = selectedHit.materialIndex;
        state.info.x = 1;
    }

    state.info.w = payload.randomSeed;
    pathStates[pathIndex] = state;
}

//------------------------------------------------------------------------------------------
// Entry: accumulate into outImage
//------------------------------------------------------------------------------------------
[numthreads(FZB_WAVEFRONT_THREADGROUP_SIZE_X, FZB_WAVEFRONT_THREADGROUP_SIZE_Y, 1)]
[shader("compute")]
void wavefrontResolveMain(uint2 threadIndex : SV_DispatchThreadID)
{
    const uint2 sceneSize = pushConst.sceneSize;
    if (threadIndex.x >= sceneSize.x || threadIndex.y >= sceneSize.y) return;
    const uint pathIndex = threadIndex.y * sceneSize.x + threadIndex.x;

    WavefrontPathState state = pathStates[pathIndex];

    float3 radiance = state.accumulatedRadiance.xyz;
    if (any(isnan(radiance)) || any(isinf(radiance))) radiance = float3(0.0f);

    // Same temporal accumulation policy as the stable VRPG shader.
    if (pushConst.frameIndex >= pushConst.maxFrameCount && pushConst.maxFrameCount > 1)
    {
        float3 oldColor = outImage[int2(threadIndex)].xyz;
        outImage[int2(threadIndex)] = float4(lerp(oldColor, radiance, 0.5f), 1.0f);
    }
    else if (pushConst.maxFrameCount > 1 && pushConst.frameIndex > 0)
    {
        float a = 1.0f / float(pushConst.frameIndex + 1);
        float3 oldColor = outImage[int2(threadIndex)].xyz;
        outImage[int2(threadIndex)] = float4(lerp(oldColor, radiance, a), 1.0f);
    }
    else
    {
        outImage[int2(threadIndex)] = float4(radiance, 1.0f);
    }
}
