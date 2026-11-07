#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "renderer/VRPG/VRPGShaderio.h"
#include "renderer/VRPG/Octree/OctreeShaderio_VRPG.h"
#include "renderer/VRPG/shaders/VRPGCommon.slang"
#include "renderer/VRPG/shaders/VRPGGuide.slang"

#ifndef NDEBUG
[[vk::binding(StaticBindingPoints_VRPG::eDepthImage, 0)]] RWTexture2D<float> depthImage;
[[vk::binding(StaticBindingPoints_VRPG::ePGValueImage, 0)]] RWTexture2D<float4> pgValueImage;
[[vk::binding(StaticBindingPoints_VRPG::ePGValue2Image, 0)]] RWTexture2D<float4> pgValue2Image;
[[vk::binding(StaticBindingPoints_VRPG::ePGVarianceImage, 0)]] RWTexture2D<float4> pgVarianceImage;
#endif

struct VRPGGuideHit {
    bool hit;
    bool frontFacing;
    bool hasTexCoords;
    float3 hitPos;
    float3 hitNormal;
    float2 texCoords;
    int materialIndex;
};

VRPGGuideHit traceGuideRay(RayDesc ray)
{
    VRPGGuideHit result;
    result.hit = false;
    result.frontFacing = true;
    result.hasTexCoords = false;
    result.hitPos = float3(0.0f);
    result.hitNormal = float3(0.0f, 1.0f, 0.0f);
    result.texCoords = float2(0.0f);
    result.materialIndex = 0;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            q.CommitNonOpaqueTriangleHit();
    }
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        return result;

    float2 bary2 = q.CommittedTriangleBarycentrics();
    float3 bary = float3(1.0f - bary2.x - bary2.y, bary2.x, bary2.y);
    float4x3 worldToObject = q.CommittedWorldToObject4x3();
    float4x3 objectToWorld = q.CommittedObjectToWorld4x3();
    int instanceIndex = q.CommittedInstanceIndex();
    uint primitiveIndex = q.CommittedPrimitiveIndex();
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Instance instance = sceneInfo.instances[instanceIndex];
    Mesh mesh = sceneInfo.meshes[instance.meshIndex];
    BSDFMaterial material = sceneInfo.materials[instance.materialIndex];

    float3 triPos[3] = q.CommittedGetIntersectionTriangleVertexPositions();
    float3 localPos = triPos[0] * bary.x + triPos[1] * bary.y + triPos[2] * bary.z;
    result.hitPos = float3(mul(float4(localPos, 1.0), objectToWorld));
    float3 geoNormal = normalize(cross(triPos[1] - triPos[0], triPos[2] - triPos[0]));
    result.hitNormal = normalize(mul(geoNormal, transpose(worldToObject)).xyz);
    float cosineON = dot(result.hitNormal, -ray.Direction);
    result.frontFacing = cosineON > 0.0f;
    if (cosineON <= 0.0f) result.hitNormal = -result.hitNormal;

    if (material.materialMapIndex.x >= 0 || material.materialMapIndex.y >= 0 || material.materialMapIndex.z >= 0) {
        int3 indices = getTriangleIndices(mesh.dataBuffer, mesh.triMesh, primitiveIndex);
        result.texCoords = getTriangleAttribute<float2>(mesh.dataBuffer, mesh.triMesh.texCoords, indices, bary);
        result.texCoords.y = 1.0f - result.texCoords.y;
        result.hasTexCoords = true;
    }
    result.materialIndex = instance.materialIndex;
    result.hit = true;
    return result;
}

HitPayload_VRPG makeGuidePayload(VRPGGuideHit hit, float3 guideDir, uint seed, SceneInfo sceneInfo)
{
    HitPayload_VRPG payload;
    payload.material = sceneInfo.materials[hit.materialIndex];
    payload.hitPos = hit.hitPos;
    payload.hitNormal = hit.hitNormal;
    payload.outgoing = -guideDir;
    payload.incidence = guideDir;
    payload.randomSeed = seed;
    payload.bsdf = float3(0.0f);
    payload.pdf = 1.0f;
    payload.isExt = hit.frontFacing;
    orthonormalBasis(payload.hitNormal, payload.TBN[0], payload.TBN[1]);
    payload.TBN[2] = payload.hitNormal;
    if (hit.hasTexCoords)
        getTextueData(payload, hit.texCoords);
    if (payload.material.type != MaterialType::Dielectric && payload.material.type != MaterialType::RoughDielectric)
        payload.material.emissive *= hit.frontFacing ? 1.0f : 0.0f;
    return payload;
}

bool traceRay(RayDesc ray, inout HitPayload_VRPG payload)
{
    #ifndef NDEBUG
    if (any(isnan(ray.Origin)) || any(isinf(ray.Origin)) ||
        any(isnan(ray.Direction)) || any(isinf(ray.Direction)))
        return false;
    #endif
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            q.CommitNonOpaqueTriangleHit();
    }
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        float2 barycentricCoords = q.CommittedTriangleBarycentrics();
        float4x3 worldToObject = q.CommittedWorldToObject4x3();
        float4x3 objectToWorld = q.CommittedObjectToWorld4x3();
        int instanceIndex = q.CommittedInstanceIndex();
        uint primitiveIndex = q.CommittedPrimitiveIndex();
        SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
        Instance instance = sceneInfo.instances[instanceIndex];
        Mesh mesh = sceneInfo.meshes[instance.meshIndex];
        payload.material = sceneInfo.materials[instance.materialIndex];
        const float3 barycentrics = float3(1.0f - barycentricCoords.x - barycentricCoords.y, barycentricCoords.x, barycentricCoords.y);
        float3 triPos[3] = q.CommittedGetIntersectionTriangleVertexPositions();
        const float3 localPosition = triPos[0] * barycentrics.x + triPos[1] * barycentrics.y + triPos[2] * barycentrics.z;
        payload.hitPos = float3(mul(float4(localPosition, 1.0), objectToWorld));
        const float3 edge1 = triPos[1] - triPos[0];
        const float3 edge2 = triPos[2] - triPos[0];
        const float3 geoNormal = normalize(cross(edge1, edge2));
        payload.hitNormal = normalize(mul(geoNormal, transpose(worldToObject)).xyz);
        float cosineON = dot(payload.hitNormal, -ray.Direction);
        if (cosineON <= 0.0f) payload.hitNormal = -payload.hitNormal;
        if (payload.material.type != MaterialType::Dielectric && payload.material.type != MaterialType::RoughDielectric)
            payload.material.emissive *= cosineON > 0.0f ? 1.0f : 0.0f;
        float3 tangent, bitangent;
        orthonormalBasis(payload.hitNormal, tangent, bitangent);
        payload.TBN = float3x3(tangent, bitangent, payload.hitNormal);
        if (payload.material.materialMapIndex.x >= 0 || payload.material.materialMapIndex.y >= 0 || payload.material.materialMapIndex.z >= 0) {
            int3 indices = getTriangleIndices(mesh.dataBuffer, mesh.triMesh, primitiveIndex);
            float2 texCoords = getTriangleAttribute<float2>(mesh.dataBuffer, mesh.triMesh.texCoords, indices, barycentrics);
            texCoords.y = 1.0f - texCoords.y;
            getTextueData(payload, texCoords);
        }
        return true;
    }
    return false;
}
float3 RayMiss(float3 rayDirection) {
    SceneInfo* sceneInfo = pushConst.sceneInfoAddress;
    if (sceneInfo.useSky == 1) return evalSimpleSky(sceneInfo.skySimpleParam, rayDirection);
    else return sceneInfo.backgroundColor;
}
//---------------------------------------------PathGuiding-----------------------------------------
/*
[numthreads(FZB_PATHGUIDING_THREADGROUP_SIZE_X, FZB_PATHGUIDING_THREADGROUP_SIZE_Y, 1)]
[shader("compute")]
void computeMain_VRPG(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    threadIndex += pushConst.dispatchOffset;
    uint groupThreadIndex_linear = groupThreadIndex.y * FZB_PATHGUIDING_THREADGROUP_SIZE_X + groupThreadIndex.x;
    uint threadGroupThreadTotalCount = FZB_PATHGUIDING_THREADGROUP_SIZE_X * FZB_PATHGUIDING_THREADGROUP_SIZE_Y;

    float2 launchID = (float2)threadIndex;
    float2 launchSize = (float2)pushConst.sceneSize;
    if (threadIndex.x >= pushConst.sceneSize.x || threadIndex.y >= pushConst.sceneSize.y) return;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];

    const uint rayFlags = 0; // RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    RayDesc ray;
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    HitPayload_VRPG payload;
    payload.randomSeed = xxhash32(uint3(threadIndex, pushConst.randomFrameIndex));

    float3 accumulatedRadiance = float3(0.0f);
    float RR = 0.8f;

    #ifndef NDEBUG
    float3 pgValue = 0.0f;
    float3 pgValue2 = 0.0f;
    #endif

    float3 firstHitPos = float3(0.0f);
    bool firstHit = false;
    for (int sampleIndex = 0; sampleIndex < pushConst.spp; ++sampleIndex) {
        float r1 = rand(payload.randomSeed);
        float r2 = rand(payload.randomSeed);
        float2 subpixel_jitter = pushConst.frameIndex == 0 && sampleIndex == 0 ? float2(0.5f, 0.5f) : float2(r1, r2);

        const float2 pixelCenter = launchID + subpixel_jitter;
        const float2 clipCoords = pixelCenter / launchSize * 2.0 - 1.0;
        float4 viewCoords = mul(float4(clipCoords, 1.0, 1.0), sceneInfo.projInvMatrix);
        viewCoords /= viewCoords.w;

        ray.Origin = sceneInfo.cameraPosition;
        ray.Direction = normalize(mul(float4(normalize(viewCoords.xyz), 0.0), sceneInfo.viewInvMatrix).xyz);

        payload.bsdf = float3(1.0f);
        payload.pdf = 1.0f;
        payload.outgoing = -ray.Direction;
        payload.isExt = true;

        float3 bsdf_cosine_pdf = float3(1.0f);

        uint bounceDepth = 0; bool useBsdfSample = false, hit = true;
        while (bounceDepth < pushConst.maxDepth) {
            if (!useBsdfSample) hit = traceRay(ray, payload);
            useBsdfSample = false;

            #ifndef NDEBUG
            if (bounceDepth == 0 && sampleIndex == 0) {
                float3 hitPos = payload.hitPos + 0.01f * ray.Direction;
                float4 clipPos = mul(float4(hitPos, 1.0f), sceneInfo.viewProjMatrix);
                float depth = clipPos.z / clipPos.w; // any hitPoint must in our cameraView
                depthImage[int2(launchID)] = depth;
            } else if (bounceDepth == MISS_DEPTH) depthImage[int2(launchID)] = 1.0f;
            #endif

            if (!hit) {
                accumulatedRadiance += RayMiss(ray.Direction) * bsdf_cosine_pdf;
                bounceDepth = MISS_DEPTH;
            } else {
                //----------------------------------------------------------------------------------------------
                PGSampleData pgSampleData;
                bool useVRPG = getNodeAABB_E_VRPG(pgSampleData, payload, threadIndex);

                float3 pgSample_bsdf_cosine = 0.0f;
                float pgSample_bsdfPdf = 0.0f;
                float pgSample_weight = 0.0f;
                if (useVRPG) {
                    float cosine = abs(dot(payload.hitNormal, pgSampleData.sampleDir));
                    pgSample_bsdf_cosine = cosine * getBSDF(payload.material, pgSampleData.sampleDir, payload.hitNormal, payload.outgoing, payload.TBN, payload.isExt);
                    pgSample_bsdfPdf = getPdf(payload.material, pgSampleData.sampleDir, payload.hitNormal, payload.outgoing, payload.TBN, payload.isExt);
                    pgSample_weight = pgSampleData.pdf / max(pgSampleData.pdf + pgSample_bsdfPdf, 1e-16f);
                }
                //----------------------------------------------------------------------------------------------
                getDirectionLight(payload, sceneInfo);
                BSDF_SAMPLE(payload);
                accumulatedRadiance += (payload.material.emissive + payload.directionLightRadiance) * bsdf_cosine_pdf;

                float3 bsdfSampleDir = payload.incidence;
                float3 bsdfSample_bsdf_cosine = payload.bsdf * abs(dot(bsdfSampleDir, payload.hitNormal));
                float bsdfSample_bsdfPdf = payload.pdf;

                float3 hitNormal = payload.hitNormal;
                float3 hitPos = payload.hitPos;
                ray.Origin = offsetRay(hitPos, hitNormal);
                ray.Direction = bsdfSampleDir;
                hit = traceRay(ray, payload);       //after here, payload is changed to next hitPos's payload!!!!!!

                float bsdfSample_pgPdf = 0.0f;
                if (useVRPG && hit) {
                    #if (VRPG_SPATIAL_REUSE || VRPG_TEMPORAL_REUSE)
                        bsdfSample_pgPdf = getPgPdf_SPReuse(pgSampleData.wSum,
                            payload.hitPos, payload.hitNormal, hitPos,
                            bsdfSampleDir, bsdfSample_bsdf_cosine, threadIndex);
                    #else
                        bsdfSample_pgPdf = getPgPdf2(pgSampleData.wSum, pgSampleData.nodeLabel_G, pgSampleData.outgoingIndex,
                            payload.hitPos, payload.hitNormal, hitPos, bsdfSampleDir, bsdfSample_bsdf_cosine);
                    #endif
                }
                    
                float bsdfSample_weight = bsdfSample_bsdfPdf / max(bsdfSample_pgPdf + bsdfSample_bsdfPdf, 1e-16f);

                #ifndef NDEBUG
                if (bounceDepth > 0) {
                    float3 pgValue_temp = (payload.material.emissive + payload.directionLightRadiance) * bsdf_cosine_pdf;
                    pgValue += pgValue_temp;
                    pgValue2 += pgValue_temp * pgValue_temp;
                }
                #endif
                //----------------------------------------------------MIS------------------------------------------
                float pgSample_selectProbability = pgSample_weight / max(pgSample_weight + bsdfSample_weight, 1e-16f);
                float randomNumber = rand(payload.randomSeed);
                if (randomNumber <= pgSample_selectProbability) {
                    payload.incidence = pgSampleData.sampleDir;
                    bsdf_cosine_pdf *= pgSample_weight * pgSample_bsdf_cosine / pgSampleData.pdf / max(pgSample_selectProbability, 1e-16f);
                } else {
                    useBsdfSample = true;
                    payload.incidence = bsdfSampleDir;
                    bsdf_cosine_pdf *= bsdfSample_weight * bsdfSample_bsdf_cosine / bsdfSample_bsdfPdf / max(1.0f - pgSample_selectProbability, 1e-16f);
                }
                //----------------------------------------------------------------------------------------------
                bool end = any(bsdf_cosine_pdf < 0.0f) || all(bsdf_cosine_pdf == 0.0f) || any(isnan(bsdf_cosine_pdf)) || any(isinf(bsdf_cosine_pdf));
                if (end) break;

                randomNumber = rand(payload.randomSeed);
                if (randomNumber >= RR) break;
                bsdf_cosine_pdf /= RR;

                payload.outgoing = -payload.incidence;
                if (!useBsdfSample) {
                    ray.Direction = payload.incidence;
                    ray.Origin = offsetRay(hitPos, hitNormal);
                }

                ++bounceDepth;
            }
        }
    }
    accumulatedRadiance /= pushConst.spp;

    #ifndef NDEBUG
    uint n = max(pushConst.spp, 1);
    pgValue /= n;
    pgValue2 /= n;
    if (pushConst.frameIndex > 0) {
        pgValue = (pgValue + pgValueImage[threadIndex].xyz * (pushConst.frameIndex - 1)) / pushConst.frameIndex;
        pgValue2 = (pgValue2 + pgValue2Image[threadIndex].xyz * (pushConst.frameIndex - 1)) / pushConst.frameIndex;

        pgVarianceImage[threadIndex].x = luminance(pgValue2 - pgValue * pgValue);
    }
    pgValueImage[threadIndex].xyz = pgValue;
    pgValue2Image[threadIndex].xyz = pgValue2;
    #endif

    // When the maximum frame rate is reached, the original image will decay by ((maxframe-1)/maxframe)^k < 1, and the final image will turn black
    if (pushConst.maxFrameCount > 1 && pushConst.frameIndex > 0){
        float a = 1.0f / float(pushConst.frameIndex + 1);
        float3 old_color = outImage[int2(launchID)].xyz;
        outImage[int2(launchID)] = float4(lerp(old_color, accumulatedRadiance, a), 1.0f);
    }
    else outImage[int2(launchID)] = float4(accumulatedRadiance, 1.0f);
}
*/
[numthreads(FZB_PATHGUIDING_THREADGROUP_SIZE_X, FZB_PATHGUIDING_THREADGROUP_SIZE_Y, 1)]
[shader("compute")]
void computeMain_VRPG(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    threadIndex += pushConst.dispatchOffset;
    uint groupThreadIndex_linear = groupThreadIndex.y * FZB_PATHGUIDING_THREADGROUP_SIZE_X + groupThreadIndex.x;
    uint threadGroupThreadTotalCount = FZB_PATHGUIDING_THREADGROUP_SIZE_X * FZB_PATHGUIDING_THREADGROUP_SIZE_Y;

    float2 launchID = (float2)threadIndex;
    float2 launchSize = (float2)pushConst.sceneSize;
    if (threadIndex.x >= pushConst.sceneSize.x || threadIndex.y >= pushConst.sceneSize.y) return;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];

    const uint rayFlags = 0;
    RayDesc ray;
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    HitPayload_VRPG payload;
    HitPayload_VRPG vrPayload;

    uint pgSeed = xxhash32(uint3(threadIndex, pushConst.randomFrameIndex));

    float3 accumulatedRadiance = float3(0.0f);
    float RR = 0.8f;

    #ifndef NDEBUG
    float3 pgValue = 0.0f;
    float3 pgValue2 = 0.0f;
    #endif

    float3 firstHitPos = float3(0.0f);
    bool firstHit = false;
    for (int sampleIndex = 0; sampleIndex < pushConst.spp; ++sampleIndex) {
        float r1 = rand(pgSeed);
        float r2 = rand(pgSeed);
        float2 subpixel_jitter = pushConst.frameIndex == 0 && sampleIndex == 0 ? float2(0.5f, 0.5f) : float2(r1, r2);

        const float2 pixelCenter = launchID + subpixel_jitter;
        const float2 clipCoords = pixelCenter / launchSize * 2.0 - 1.0;
        float4 viewCoords = mul(float4(clipCoords, 1.0, 1.0), sceneInfo.projInvMatrix);
        viewCoords /= viewCoords.w;

        ray.Origin = sceneInfo.cameraPosition;
        ray.Direction = normalize(mul(float4(normalize(viewCoords.xyz), 0.0), sceneInfo.viewInvMatrix).xyz);

        payload.bsdf = float3(1.0f);
        payload.pdf = 1.0f;
        payload.outgoing = -ray.Direction;
        payload.isExt = true;
        payload.randomSeed = pgSeed;

        float3 bsdf_cosine_pdf = float3(1.0f);
        uint bounceDepth = 0;
        bool hit = traceRay(ray, payload);
        firstHitPos = payload.hitPos;
        firstHit = hit;


        while (bounceDepth < pushConst.maxDepth) {
            #ifndef NDEBUG
            if (bounceDepth == 0 && sampleIndex == 0) {
                if (hit) {
                    float3 hitPos = payload.hitPos + 0.01f * ray.Direction;
                    float4 clipPos = mul(float4(hitPos, 1.0f), sceneInfo.viewProjMatrix);
                    depthImage[int2(launchID)] = clipPos.z / clipPos.w;
                }
                else {
                    depthImage[int2(launchID)] = 1.0f;
                }
            }
            #endif

            if (!hit) {
                accumulatedRadiance += RayMiss(ray.Direction) * bsdf_cosine_pdf;
                break;
            }

            //-------------------------------------------------- PG candidate --------------------------------------------------
            payload.randomSeed = pgSeed;

            #if VRPG_PATH_SAMPLE_MODE == 0
            VRSampleData vrSampleData;
            #else
            VRSampleData vrSampleData = {};
            #endif
            bool getVrSample = getNodeAABB_E_VRPG(vrSampleData, payload, threadIndex);
            pgSeed = payload.randomSeed;

            float randomNumber = 0.0f;
            #if VRPG_PATH_SAMPLE_MODE == 1
            {
                HitPayload_VRPG bsdfPayload;
                // One-sample MIS mixture: choose guide or BSDF with probability
                // q, trace only that ray, then apply the balance-heuristic
                // weight for the chosen technique.
                getDirectionLight(payload, sceneInfo);
                accumulatedRadiance += (payload.material.emissive + payload.directionLightRadiance) * bsdf_cosine_pdf;

                float3 previousHitPos = payload.hitPos;
                bsdfPayload = payload;
                bsdfPayload.randomSeed = pgSeed;
                getDirectionLight(bsdfPayload, sceneInfo);
                BSDF_SAMPLE(bsdfPayload);

                float3 bsdfDir = bsdfPayload.incidence;
                float3 bsdfCos = bsdfPayload.bsdf * abs(dot(bsdfDir, bsdfPayload.hitNormal));
                float bsdfPdf = bsdfPayload.pdf;
                if (!(bsdfPdf >= 0.0f) || isinf(bsdfPdf)) bsdfPdf = 0.0f;

                float guidePdf = vrSampleData.pdf;
                if (!(guidePdf >= 0.0f) || isinf(guidePdf)) guidePdf = 0.0f;

                float3 guideDir = vrSampleData.sampleDir;
                float3 guideBsdfCos = float3(0.0f);
                float guideBsdfPdf = 0.0f;
                bool guideValid = getVrSample && guidePdf > 0.0f;
                if (guideValid) {
                    float cosine = abs(dot(payload.hitNormal, guideDir));
                    guideBsdfCos = cosine * getBSDF(payload.material, guideDir,
                                                     payload.hitNormal, payload.outgoing,
                                                     payload.TBN, payload.isExt);
                    if (any(isnan(guideBsdfCos)) || any(isinf(guideBsdfCos)))
                        guideBsdfCos = float3(0.0f);
                    if (!any(guideBsdfCos > 0.0f))
                        guideValid = false;

                    guideBsdfPdf = getPdf(payload.material, guideDir, payload.hitNormal,
                                          payload.outgoing, payload.TBN, payload.isExt);
                    if (!(guideBsdfPdf >= 0.0f) || isinf(guideBsdfPdf)) guideBsdfPdf = 0.0f;
                }

                bool bsdfValid = bsdfPdf > 0.0f && any(bsdfCos > 0.0f);
                float q = guideValid ? (bsdfValid ? 0.5f : 1.0f) : 0.0f;
                if (!(q >= 0.0f) || isinf(q)) q = 0.0f;
                q = clamp(q, 0.0f, 1.0f);
                bool chooseGuide = guideValid && rand(pgSeed) < q;

                if (chooseGuide) {
                    vrPayload = payload;
                    vrPayload.incidence = guideDir;
                    ray.Origin = offsetRay(payload.hitPos, payload.hitNormal);
                    ray.Direction = guideDir;
                    bool guideHit = traceRay(ray, vrPayload);
                    if (guideHit) {
                        float wg = guidePdf / max(guidePdf + guideBsdfPdf, 1e-16f);
                        payload = vrPayload;
                        payload.incidence = guideDir;
                        bsdf_cosine_pdf *= wg * guideBsdfCos / max(guidePdf * q, 1e-16f);
                        hit = true;
                    }
                    else {
                        payload.incidence = guideDir;
                        ray.Direction = guideDir;
                        bsdf_cosine_pdf *= 1.0f / max(q, 1e-16f);
                        hit = false;
                    }
                }
                else {
                    ray.Origin = offsetRay(payload.hitPos, payload.hitNormal);
                    ray.Direction = bsdfDir;
                    bool bsdfHit = traceRay(ray, bsdfPayload);
                    payload = bsdfPayload;
                    payload.incidence = bsdfDir;

                    float pgPdf = 0.0f;
                    if (bsdfHit) {
                        #if (VRPG_SPATIAL_REUSE || VRPG_TEMPORAL_REUSE)
                            pgPdf = getPgPdf_SPReuse(vrSampleData.wSum,
                                payload.hitPos, payload.hitNormal, previousHitPos,
                                bsdfDir, bsdfCos, threadIndex);
                        #else
                            pgPdf = getPgPdf2(vrSampleData.wSum, vrSampleData.nodeLabel_G,
                                vrSampleData.outgoingIndex, payload.hitPos, payload.hitNormal,
                                previousHitPos, bsdfDir, bsdfCos);
                        #endif
                        if (!(pgPdf >= 0.0f) || isinf(pgPdf)) pgPdf = 0.0f;
                    }

                    if (bsdfHit) {
                        float wb = bsdfPdf / max(bsdfPdf + pgPdf, 1e-16f);
                        bsdf_cosine_pdf *= wb * bsdfCos / max(bsdfPdf * (1.0f - q), 1e-16f);
                    }
                    else {
                        bsdf_cosine_pdf *= 1.0f / max(1.0f - q, 1e-16f);
                    }
                    hit = bsdfHit;
                }
            }
            #elif VRPG_PATH_SAMPLE_MODE == 2
            {
                HitPayload_VRPG bsdfPayload;
                // Single-ray mixture: use guide sampling when available,
                // otherwise fall back to BSDF sampling.
                getDirectionLight(payload, sceneInfo);
                accumulatedRadiance += (payload.material.emissive + payload.directionLightRadiance) * bsdf_cosine_pdf;

                bool oneHit = false;
                if (getVrSample && vrSampleData.pdf > 0.0f) {
                    vrPayload = payload;
                    vrPayload.incidence = vrSampleData.sampleDir;
                    ray.Origin = offsetRay(payload.hitPos, payload.hitNormal);
                    ray.Direction = vrSampleData.sampleDir;
                    oneHit = traceRay(ray, vrPayload);

                    if (oneHit) {
                        float3 guideBsdfCosine = abs(dot(payload.hitNormal, vrSampleData.sampleDir)) *
                            getBSDF(payload.material, vrSampleData.sampleDir, payload.hitNormal,
                                    payload.outgoing, payload.TBN, payload.isExt);
                        payload = vrPayload;
                        payload.incidence = vrSampleData.sampleDir;
                        bsdf_cosine_pdf *= guideBsdfCosine / max(vrSampleData.pdf, 1e-16f);
                        hit = true;
                    }
                    else {
                        payload.incidence = vrSampleData.sampleDir;
                        ray.Direction = vrSampleData.sampleDir;
                        hit = false;
                    }
                }
                else {
                    bsdfPayload = payload;
                    bsdfPayload.randomSeed = pgSeed;
                    getDirectionLight(bsdfPayload, sceneInfo);
                    BSDF_SAMPLE(bsdfPayload);
                    float3 bsdfDir = bsdfPayload.incidence;
                    float3 bsdfCos = bsdfPayload.bsdf * abs(dot(bsdfDir, bsdfPayload.hitNormal));
                    float bsdfPdf = bsdfPayload.pdf;
                    ray.Origin = offsetRay(payload.hitPos, payload.hitNormal);
                    ray.Direction = bsdfDir;
                    oneHit = traceRay(ray, bsdfPayload);
                    payload = bsdfPayload;
                    payload.incidence = bsdfDir;
                    bsdf_cosine_pdf *= bsdfCos / max(bsdfPdf, 1e-16f);
                    hit = oneHit;
                }
            }
            #else
            {
                VRPGGuideHit guideHit;
                bool vrHit = false;
                bool validSample = false;

                if (getVrSample) {
                    ray.Origin = offsetRay(payload.hitPos, payload.hitNormal);
                    ray.Direction = vrSampleData.sampleDir;
                    guideHit = traceGuideRay(ray);
                    vrHit = guideHit.hit;

                    if (vrHit) {
                        validSample = checkPointInAABB(guideHit.hitPos, vrSampleData.aabb);
                        validSample = validSample && (getNormalIndex(guideHit.hitNormal) == vrSampleData.normalIndex);
                        validSample = validSample && (vrSampleData.pdf > 0.0f);
                    }
                }

                //-------------------------------------------------- BSDF candidate -------------------------------------------------
                float3 previousHitPos = payload.hitPos;

                float3 pgSample_bsdf_cosine = float3(0.0f);
                float pgSample_bsdfPdf = 0.0f;
                float pgSample_weight = 0.0f;
                if (validSample) {
                    float cosine = abs(dot(payload.hitNormal, vrSampleData.sampleDir));
                    pgSample_bsdf_cosine = cosine * getBSDF(payload.material, vrSampleData.sampleDir,
                                                            payload.hitNormal, payload.outgoing,
                                                            payload.TBN, payload.isExt);
                    pgSample_bsdfPdf = getPdf(payload.material, vrSampleData.sampleDir,
                                              payload.hitNormal, payload.outgoing,
                                              payload.TBN, payload.isExt);
                    if (!(pgSample_bsdfPdf >= 0.0f) || isinf(pgSample_bsdfPdf)) pgSample_bsdfPdf = 0.0f;
                    pgSample_weight = vrSampleData.pdf / max(vrSampleData.pdf + pgSample_bsdfPdf, 1e-16f);
                }

                getDirectionLight(payload, sceneInfo);

                // BSDF_SAMPLE scales directionLightRadiance by the material
                // BRDF, so it must run before the direct-light contribution.
                BSDF_SAMPLE(payload);

                // Direct lighting at the current hit is independent of which outgoing direction is selected.
                accumulatedRadiance += (payload.material.emissive + payload.directionLightRadiance) * bsdf_cosine_pdf;
                //if (bounceDepth == 2) accumulatedRadiance += (payload.material.emissive + payload.directionLightRadiance) * bsdf_cosine_pdf * 20.0f;

                float3 bsdfSampleDir = payload.incidence;
                float3 bsdfSample_bsdf_cosine = payload.bsdf * abs(dot(bsdfSampleDir, payload.hitNormal));
                float bsdfSample_bsdfPdf = payload.pdf;
                if (!(bsdfSample_bsdfPdf >= 0.0f) || isinf(bsdfSample_bsdfPdf)) bsdfSample_bsdfPdf = 0.0f;

                ray.Origin = offsetRay(previousHitPos, payload.hitNormal);
                ray.Direction = bsdfSampleDir;
                bool bsdfHit = traceRay(ray, payload);
                pgSeed = payload.randomSeed;

                float bsdfSample_pgPdf = 0.0f;
                if (validSample && bsdfHit) {
                    #if (VRPG_SPATIAL_REUSE || VRPG_TEMPORAL_REUSE)
                        bsdfSample_pgPdf = getPgPdf_SPReuse(vrSampleData.wSum,
                            payload.hitPos, payload.hitNormal, previousHitPos,
                            bsdfSampleDir, bsdfSample_bsdf_cosine, threadIndex);
                    #else
                        bsdfSample_pgPdf = getPgPdf2(vrSampleData.wSum, vrSampleData.nodeLabel_G, vrSampleData.outgoingIndex,
                            payload.hitPos, payload.hitNormal, previousHitPos,
                            bsdfSampleDir, bsdfSample_bsdf_cosine);
                    #endif
                    if (!(bsdfSample_pgPdf >= 0.0f) || isinf(bsdfSample_pgPdf)) bsdfSample_pgPdf = 0.0f;
                }

                float bsdfSample_weight = bsdfSample_bsdfPdf / max(bsdfSample_pgPdf + bsdfSample_bsdfPdf, 1e-16f);                //-------------------------------------------------- choose sample -------------------------------------------------
                // A technique whose contribution or MIS weight is zero cannot contribute; do not waste a path on it.
                bool pgAvailable = validSample && (pgSample_weight > 0.0f) && any(pgSample_bsdf_cosine > 0.0f);
                bool bsdfAvailable = (bsdfSample_bsdfPdf > 0.0f) && (bsdfSample_weight > 0.0f) && any(bsdfSample_bsdf_cosine > 0.0f);
                if (!pgAvailable && !bsdfAvailable) break;

                float pgSample_selectProbability = pgAvailable ? (bsdfAvailable ? 0.5f : 1.0f) : 0.0f;
                randomNumber = rand(pgSeed);

                if (randomNumber < pgSample_selectProbability) {
                    payload = makeGuidePayload(guideHit, vrSampleData.sampleDir, pgSeed, sceneInfo);
                    bsdf_cosine_pdf *= pgSample_weight * pgSample_bsdf_cosine / max(vrSampleData.pdf, 1e-16f) / max(pgSample_selectProbability, 1e-16f);
                    hit = vrHit;
                }
                else {
                    // payload already contains the BSDF trace result.
                    bsdf_cosine_pdf *= bsdfSample_weight * bsdfSample_bsdf_cosine / max(bsdfSample_bsdfPdf, 1e-16f) / max(1.0f - pgSample_selectProbability, 1e-16f);
                    hit = bsdfHit;
                }
            }
            #endif
            if (!hit) {
                // Let the next iteration add the sky contribution with the selected direction.
                ray.Direction = payload.incidence;
            }

            bool end = any(bsdf_cosine_pdf < 0.0f) || all(bsdf_cosine_pdf == 0.0f) || any(isnan(bsdf_cosine_pdf)) || any(isinf(bsdf_cosine_pdf));
            if (end) break;

            randomNumber = rand(pgSeed);
            if (randomNumber >= RR) break;
            bsdf_cosine_pdf /= RR;

            payload.randomSeed = pgSeed;
            payload.outgoing = -payload.incidence;
            ++bounceDepth;
        }
    }
    #if VRPG_CAMERA_REPROJECTION
    if (firstHit && pushConst.frameIndex == 0 && pushConst.staticScene != 0) {
        float4 prevClip = mul(float4(firstHitPos, 1.0f), sceneInfo.prevViewProjMatrix);
        if (prevClip.w > 1e-6f) {
            float2 prevNdc = prevClip.xy / prevClip.w;
            int2 prevPixel = int2((prevNdc * 0.5f + 0.5f) * float2(pushConst.sceneSize));
            if (all(prevPixel >= int2(0)) && all(prevPixel < int2(pushConst.sceneSize))) {
                float3 history = outImage[prevPixel].xyz;
                accumulatedRadiance = 0.5f * (history + accumulatedRadiance);
            }
        }
    }
    #endif

    accumulatedRadiance /= pushConst.spp;

    #ifndef NDEBUG
    uint n = max(pushConst.spp, 1);
    pgValue /= n;
    pgValue2 /= n;
    if (pushConst.frameIndex > 0) {
        pgValue = (pgValue + pgValueImage[threadIndex].xyz * (pushConst.frameIndex - 1)) / pushConst.frameIndex;
        pgValue2 = (pgValue2 + pgValue2Image[threadIndex].xyz * (pushConst.frameIndex - 1)) / pushConst.frameIndex;

        pgVarianceImage[threadIndex].x = luminance(pgValue2 - pgValue * pgValue);
    }
    pgValueImage[threadIndex].xyz = pgValue;
    pgValue2Image[threadIndex].xyz = pgValue2;
    #endif

    if (pushConst.maxFrameCount == 2 && pushConst.maxFrameCount > 1) {
        float a = 0.5f;
        float3 old_color = outImage[int2(launchID)].xyz;
        outImage[int2(launchID)] = float4(lerp(old_color, accumulatedRadiance, a), 1.0f);
    }
    else if (pushConst.maxFrameCount > 1 && pushConst.frameIndex > 0) {
        float a = 1.0f / float(pushConst.frameIndex + 1);
        float3 old_color = outImage[int2(launchID)].xyz;
        outImage[int2(launchID)] = float4(lerp(old_color, accumulatedRadiance, a), 1.0f);
    }
    else outImage[int2(launchID)] = float4(accumulatedRadiance, 1.0f);
}
