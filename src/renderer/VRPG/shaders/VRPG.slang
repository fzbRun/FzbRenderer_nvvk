#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "renderer/VRPG/VRPGShaderio.h"
#include "renderer/VRPG/Octree/OctreeShaderio_VRPG.h"
#include "renderer/VRPG/shaders/VRPGCommon.slang"
#include "renderer/VRPG/shaders/VRPGGuide.slang"

#ifndef NDEBUG
[[vk::binding(StaticBindingPoints_VRPG::eDepthImage, 0)]] RWTexture2D<float> depthImage;
#endif

struct VRPGGuideHit {
    bool hit;
    bool frontFacing;
    bool hasTexCoords;
    float3 hitPos;
    float3 hitNormal;
    float2 texCoords;
    BSDFMaterial material;
};

float3 getDirectionLight2(VRPGGuideHit hit, float3 outgoing, bool isExt, SceneInfo sceneInfo) {
    float3x3 TBN;
    orthonormalBasis(hit.hitNormal, TBN[0], TBN[1]);
    TBN[2] = hit.hitNormal;

    bool isDielectric = hit.material.type == uint(MaterialType::Dielectric) ||
                        hit.material.type == uint(MaterialType::RoughDielectric);
    uint lightNum = min(sceneInfo.numLights, NEE_MAX_SAMPLE_NUM);
    for (int lightIndex = 0; lightIndex < lightNum; ++lightIndex) {
        Light light = sceneInfo.lights[lightIndex];
        if (light.type == uint(LightType::Direction)) {
            float3 sampleDir = -light.direction;
            float cosine = dot(sampleDir, hit.hitNormal);
            if (!isDielectric && cosine <= 0.001f) continue;

            const uint rayFlags = 0;
            RayDesc tempRay;
            tempRay.Direction = normalize(sampleDir);
            tempRay.Origin = hit.hitPos + 0.00001f * tempRay.Direction;
            tempRay.TMin = 0.001f;
            tempRay.TMax = INFINITE;
            HitTestPayload tempPayload;
            tempPayload.distance = MISS_DISTANCE;

            hitTestTraceRay(tempRay, tempPayload);
            if (tempPayload.distance != MISS_DISTANCE) continue;

            float3 bsdf = getBSDF(hit.material, sampleDir, hit.hitNormal, outgoing, TBN, isExt);
            return light.color * light.intensity * abs(cosine) * bsdf;
        }
    }

    return 0.0f;
}

VRPGGuideHit traceGuideRay(RayDesc ray)
{
    VRPGGuideHit result;
    result.hit = false;
    result.frontFacing = true;
    result.hasTexCoords = false;
    result.hitPos = float3(0.0f);
    result.hitNormal = float3(0.0f, 1.0f, 0.0f);
    result.texCoords = float2(0.0f);

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
    result.material = sceneInfo.materials[instance.materialIndex];

    float3 triPos[3] = q.CommittedGetIntersectionTriangleVertexPositions();
    float3 localPos = triPos[0] * bary.x + triPos[1] * bary.y + triPos[2] * bary.z;
    result.hitPos = float3(mul(float4(localPos, 1.0), objectToWorld));
    float3 geoNormal = normalize(cross(triPos[1] - triPos[0], triPos[2] - triPos[0]));
    result.hitNormal = normalize(mul(geoNormal, transpose(worldToObject)).xyz);
    float cosineON = dot(result.hitNormal, -ray.Direction);
    result.frontFacing = cosineON > 0.0f;
    if (cosineON <= 0.0f) result.hitNormal = -result.hitNormal;

    if (result.material.materialMapIndex.x >= 0 || result.material.materialMapIndex.y >= 0 || result.material.materialMapIndex.z >= 0) {
        int3 indices = getTriangleIndices(mesh.dataBuffer, mesh.triMesh, primitiveIndex);
        result.texCoords = getTriangleAttribute<float2>(mesh.dataBuffer, mesh.triMesh.texCoords, indices, bary);
        result.texCoords.y = 1.0f - result.texCoords.y;
        result.hasTexCoords = true;
        if (any(result.texCoords > 0.0f)){
            int normalMapIndex = result.material.materialMapIndex.x;
            if (normalMapIndex >= 0) {
                float3 tangent, bitangent;
                orthonormalBasis(result.hitNormal, tangent, bitangent);
                float3x3 TBN = float3x3(tangent, bitangent, result.hitNormal);

                float3 localNormal = textures[normalMapIndex].SampleLevel(result.texCoords, 0).xyz;
                result.hitNormal = mul(localNormal, TBN);
            }
            getBSDFTextureData(result.material, result.texCoords);
        }
    }
    if (result.material.type != MaterialType::Dielectric && result.material.type != MaterialType::RoughDielectric)
        result.material.emissive *= result.frontFacing ? 1.0f : 0.0f;

    result.hit = true;
    return result;
}

HitPayload_VRPG makeGuidePayload(VRPGGuideHit hit, float3 guideDir, uint seed, SceneInfo sceneInfo)
{
    HitPayload_VRPG payload;
    payload.material = hit.material;
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
    //if (hit.hasTexCoords)
    //    getTextueData(payload, hit.texCoords);
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
        payload.isExt = cosineON > 0.0f;
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

bool traceRay_test(RayDesc ray, inout HitPayload_VRPG payload)
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

        const int3 idx = getTriangleIndices(mesh.dataBuffer, mesh.triMesh, primitiveIndex);

        if (any(idx < 0) ||
    idx.x >= int(mesh.triMesh.positions.count) ||
    idx.y >= int(mesh.triMesh.positions.count) ||
            idx.z >= int(mesh.triMesh.positions.count))
        {
            return false;
        }

        BufferView posf = mesh.triMesh.positions;
        posf.byteStride = 4;

        const uint i0 = uint(idx.x) * 3u;
        const uint i1 = uint(idx.y) * 3u;
        const uint i2 = uint(idx.z) * 3u;

        const float3 v0 = float3(getAttribute<float>(mesh.dataBuffer, posf, i0 + 0u),
                                 getAttribute<float>(mesh.dataBuffer, posf, i0 + 1u),
                                 getAttribute<float>(mesh.dataBuffer, posf, i0 + 2u));
        const float3 v1 = float3(getAttribute<float>(mesh.dataBuffer, posf, i1 + 0u),
                                 getAttribute<float>(mesh.dataBuffer, posf, i1 + 1u),
                                 getAttribute<float>(mesh.dataBuffer, posf, i1 + 2u));
        const float3 v2 = float3(getAttribute<float>(mesh.dataBuffer, posf, i2 + 0u),
                                 getAttribute<float>(mesh.dataBuffer, posf, i2 + 1u),
                                 getAttribute<float>(mesh.dataBuffer, posf, i2 + 2u));

        const float3 localPosition = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
        payload.hitPos = float3(mul(float4(localPosition, 1.0), objectToWorld));

        const float3 geoNormal = normalize(cross(v1 - v0, v2 - v0));
        float3x3 W = float3x3(worldToObject[0].xyz, worldToObject[1].xyz, worldToObject[2].xyz);
        payload.hitNormal = normalize(mul(geoNormal, W));

        if (any(isnan(payload.hitNormal))) return false;
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

//------------------------------------------------------------------------------------------
// Path sampling techniques. The BSDF and the guide are sampled by dedicated helpers and
// VRPG_PATH_SAMPLE_MODE only decides how the two are combined:
//   0 (default) : two sample balance MIS (original behaviour: guide ray + BSDF ray, one sample is
//                 chosen and both contributions are divided by the selection probability)
//   1           : one sample 50/50 (only the chosen technique is traced)
//   2           : single ray, guide when available, BSDF otherwise (no MIS weights)
//   3           : BSDF only (reference for bias tests)
//   4           : guide only (reference for bias tests)
//------------------------------------------------------------------------------------------
struct VRPGTechniqueSample
{
    bool   valid;
    float3 dir;
    float3 bsdfCosine;
    float  bsdfPdf;
    float  guidePdf;
};

// Samples the BSDF technique. Returns the direct lighting term of the current vertex (it has to be
// captured before BSDF_SAMPLE, which folds the BRDF into directionLightRadiance).
VRPGTechniqueSample vrpgSampleBsdf(inout HitPayload_VRPG payload, SceneInfo sceneInfo){
    VRPGTechniqueSample s;
    s.valid = false;
    s.dir = float3(0.0f, 0.0f, 1.0f);
    s.bsdfCosine = float3(0.0f);
    s.bsdfPdf = 0.0f;
    s.guidePdf = 0.0f;

    BSDF_SAMPLE(payload);

    s.dir = payload.incidence;
    s.bsdfCosine = payload.bsdf * abs(dot(s.dir, payload.hitNormal));
    s.bsdfPdf = payload.pdf;
    if (!(s.bsdfPdf >= 0.0f) || isinf(s.bsdfPdf)) s.bsdfPdf = 0.0f;
    if (any(isnan(s.bsdfCosine)) || any(isinf(s.bsdfCosine))) s.bsdfCosine = float3(0.0f);
    s.valid = (s.bsdfPdf > 0.0f) && any(s.bsdfCosine > 0.0f);
    return s;
}

// Shading point counterpart of the same data for the guide direction.
VRPGTechniqueSample vrpgSampleGuide(HitPayload_VRPG payload, VRSampleData vrSampleData, bool getVrSample){
    VRPGTechniqueSample s;
    s.valid = false;
    s.dir = float3(0.0f, 0.0f, 1.0f);
    s.bsdfCosine = float3(0.0f);
    s.bsdfPdf = 0.0f;
    s.guidePdf = 0.0f;

    if (!getVrSample) return s;
    s.guidePdf = vrSampleData.pdf;
    if (!(s.guidePdf > 0.0f) || isinf(s.guidePdf)){
        s.guidePdf = 0.0f;
        return s;
    }

    s.dir = vrSampleData.sampleDir;
    const float cosine = abs(dot(payload.hitNormal, s.dir));
    s.bsdfCosine = cosine * getBSDF(payload.material, s.dir, payload.hitNormal, payload.outgoing, payload.TBN, payload.isExt);
    if (any(isnan(s.bsdfCosine)) || any(isinf(s.bsdfCosine))) s.bsdfCosine = float3(0.0f);

    s.bsdfPdf = getPdf(payload.material, s.dir, payload.hitNormal, payload.outgoing, payload.TBN, payload.isExt);
    if (!(s.bsdfPdf >= 0.0f) || isinf(s.bsdfPdf)) s.bsdfPdf = 0.0f;
    s.valid = any(s.bsdfCosine > 0.0f);
    return s;
}

// Guide density along an arbitrary direction (needs the payload of the vertex that direction hits).
float vrpgGuidePdfAt(HitPayload_VRPG nextPayload, float3 previousHitPos, VRSampleData vrSampleData,
                     float3 dir, float3 bsdfCosine, uint2 threadIndex)
{
    float pdf;
    #if (VRPG_SPATIAL_REUSE || VRPG_TEMPORAL_REUSE)
    pdf = getPgPdf_SPReuse(vrSampleData.wSum, nextPayload.hitPos, nextPayload.hitNormal,
                           previousHitPos, dir, bsdfCosine, threadIndex);
    #else
    pdf = getPgPdf2(vrSampleData.wSum, vrSampleData.nodeLabel_G, vrSampleData.outgoingIndex,
                    nextPayload.hitPos, nextPayload.hitNormal, previousHitPos, dir, bsdfCosine);
    #endif
    if (!(pdf >= 0.0f) || isinf(pdf)) pdf = 0.0f;
    return pdf;
}

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
            if(bounceDepth == 0) {
                getDirectionLight2(payload, sceneInfo);
                accumulatedRadiance += (payload.material.emissive + payload.directionLightRadiance) * bsdf_cosine_pdf;
            }
            //-------------------------------------------------- technique samples --------------------------------------------------
            // The two techniques are sampled independently; the mode below only decides how they
            // are combined. The direct lighting term is produced as a side effect of the BSDF
            // sample (getDirectionLight/BSDF_SAMPLE fold the BRDF into directionLightRadiance) and
            // is independent of the outgoing direction that ends up being used.
            float3 previousHitPos = payload.hitPos;

            VRSampleData vrSampleData = {};
            payload.randomSeed = pgSeed;
            bool getVrSample = getNodeAABB_E_VRPG(vrSampleData, payload, threadIndex);
            pgSeed = payload.randomSeed;

            HitPayload_VRPG bsdfPayload = payload;
            bsdfPayload.randomSeed = pgSeed;
            VRPGTechniqueSample bsdfSample = vrpgSampleBsdf(bsdfPayload, sceneInfo);
            pgSeed = bsdfPayload.randomSeed;

            VRPGTechniqueSample guideSample = vrpgSampleGuide(payload, vrSampleData, getVrSample);
            float randomNumber = 0.0f;

            #if VRPG_PATH_SAMPLE_MODE == 0
            //-------------------------------------------------- two sample balance MIS (default) --------------------------------------------------
            {
                VRPGGuideHit guideHit;
                bool vrHit = false;
                bool validSample = false;
                if (guideSample.valid){
                    ray.Origin = offsetRay(previousHitPos, payload.hitNormal);
                    ray.Direction = guideSample.dir;
                    guideHit = traceGuideRay(ray);
                    vrHit = guideHit.hit;
                    if (vrHit){
                        validSample = checkPointInAABB(guideHit.hitPos, vrSampleData.aabb);
                        validSample = validSample && (getNormalIndex(guideHit.hitNormal) == vrSampleData.normalIndex);
                        validSample = validSample && (guideSample.guidePdf > 0.0f);
                    }
                }
                const float pgSample_weight = validSample ? guideSample.guidePdf / max(guideSample.guidePdf + guideSample.bsdfPdf, 1e-16f) : 0.0f;
                const bool pgAvailable = validSample && (pgSample_weight > 0.0f) && any(guideSample.bsdfCosine > 0.0f);
                float3 bsdf_cosine_pdf_pgSample = bsdf_cosine_pdf * guideSample.bsdfCosine / max(guideSample.guidePdf, 1e-16f) * pgSample_weight;
                if (pgAvailable) {
                    float3 directionLight = getDirectionLight2(guideHit, -guideSample.dir, guideHit.frontFacing, sceneInfo);
                    accumulatedRadiance += (guideHit.material.emissive + directionLight) * bsdf_cosine_pdf_pgSample;
                }

                ray.Origin = offsetRay(previousHitPos, payload.hitNormal);
                ray.Direction = bsdfSample.dir;
                bool bsdfHit = traceRay(ray, payload);

                const float bsdfSample_pgPdf = (validSample && bsdfHit) ? vrpgGuidePdfAt(payload, previousHitPos, vrSampleData, bsdfSample.dir, bsdfSample.bsdfCosine, threadIndex) : 0.0f;
                const float bsdfSample_weight = bsdfSample.bsdfPdf / max(bsdfSample_pgPdf + bsdfSample.bsdfPdf, 1e-16f);
                const bool bsdfAvailable = (bsdfSample.bsdfPdf > 0.0f) && (bsdfSample_weight > 0.0f) && any(bsdfSample.bsdfCosine > 0.0f);
                float3 bsdf_cosine_pdf_bsdfSample = bsdf_cosine_pdf * bsdfSample.bsdfCosine / max(bsdfSample.bsdfPdf, 1e-16f) * bsdfSample_weight;
                if (bsdfAvailable) {
                    getDirectionLight2(payload, sceneInfo);
                    accumulatedRadiance += (payload.material.emissive + payload.directionLightRadiance) * bsdf_cosine_pdf_bsdfSample;
                }
                if (!pgAvailable && !bsdfAvailable) break;

                const float selectProbability = pgAvailable ? (bsdfAvailable ? 0.5f : 1.0f) : 0.0f;
                pgSeed = payload.randomSeed;
                randomNumber = rand(pgSeed);
                if (randomNumber < selectProbability){
                    payload = makeGuidePayload(guideHit, guideSample.dir, pgSeed, sceneInfo);
                    bsdf_cosine_pdf = bsdf_cosine_pdf_pgSample / max(selectProbability, 1e-16f);
                    hit = vrHit;
                }
                else {
                    payload.incidence = bsdfSample.dir;
                    bsdf_cosine_pdf = bsdf_cosine_pdf_bsdfSample / max(1.0f - selectProbability, 1e-16f);
                    hit = bsdfHit;
                }
            }
            #else
            //-------------------------------------------------- one sample 50/50 with balance weights (mode 1) --------------------------------------------------
            // Picks ONE technique with probability q and traces only that ray. The chosen
            // technique's contribution is divided by q (one sample MIS), so its expectation equals
            // the sum of the two weighted terms of mode 0.
            {
                const bool guideOk = guideSample.valid;
                const bool bsdfOk = bsdfSample.valid;
                if (!guideOk && !bsdfOk) break;

                const float q = guideOk ? (bsdfOk ? 0.5f : 1.0f) : 0.0f;
                const bool chooseGuide = guideOk && (rand(pgSeed) < q);

                if (chooseGuide)
                {
                    ray.Origin = offsetRay(previousHitPos, payload.hitNormal);
                    ray.Direction = guideSample.dir;
                    VRPGGuideHit guideHit = traceGuideRay(ray);

                    const bool hitInVoxel = guideHit.hit
                        && checkPointInAABB(guideHit.hitPos, vrSampleData.aabb)
                        && (getNormalIndex(guideHit.hitNormal) == vrSampleData.normalIndex)
                        && (guideSample.guidePdf > 0.0f);

                    const float wg = bsdfOk
                        ? guideSample.guidePdf / max(guideSample.guidePdf + guideSample.bsdfPdf, 1e-16f)
                        : 1.0f;
                    float3 sampleThroughput = bsdf_cosine_pdf * guideSample.bsdfCosine
                                            / max(guideSample.guidePdf, 1e-16f) * wg;
                    if (hitInVoxel)
                    {
                        float3 directionLight = getDirectionLight2(guideHit, -guideSample.dir, guideHit.frontFacing, sceneInfo);
                        accumulatedRadiance += (guideHit.material.emissive + directionLight)
                                             * sampleThroughput / max(q, 1e-16f);
                    }
                    payload = makeGuidePayload(guideHit, guideSample.dir, pgSeed, sceneInfo);
                    bsdf_cosine_pdf = sampleThroughput / max(q, 1e-16f);
                    hit = hitInVoxel;
                }
                else
                {
                    ray.Origin = offsetRay(previousHitPos, payload.hitNormal);
                    ray.Direction = bsdfSample.dir;
                    const bool bsdfHit = traceRay(ray, payload);
                    payload.incidence = bsdfSample.dir;   // traceRay leaves incidence untouched

                    const float bsdfSample_pgPdf = bsdfHit
                        ? vrpgGuidePdfAt(payload, previousHitPos, vrSampleData, bsdfSample.dir,
                                         bsdfSample.bsdfCosine, threadIndex)
                        : 0.0f;
                    const float wb = guideOk
                        ? bsdfSample.bsdfPdf / max(bsdfSample.bsdfPdf + bsdfSample_pgPdf, 1e-16f)
                        : 1.0f;
                    float3 sampleThroughput = bsdf_cosine_pdf * bsdfSample.bsdfCosine
                                            / max(bsdfSample.bsdfPdf, 1e-16f) * wb;
                    if (bsdfHit)
                    {
                        getDirectionLight2(payload, sceneInfo);
                        accumulatedRadiance += (payload.material.emissive + payload.directionLightRadiance)
                                             * sampleThroughput / max(1.0f - q, 1e-16f);
                    }
                    bsdf_cosine_pdf = sampleThroughput / max(1.0f - q, 1e-16f);
                    hit = bsdfHit;
                }
            }
            #endif
            if (!hit) ray.Direction = payload.incidence;

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

    accumulatedRadiance /= pushConst.spp;

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
