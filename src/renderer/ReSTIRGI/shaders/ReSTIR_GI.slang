#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "renderer/ReSTIRGI/ReSTIR_GIShaderio.h"

[[vk::push_constant]] ConstantBuffer<ReSTIR_GIConstant, ScalarDataLayout> pushConst;
[[vk::binding(StaticBindingPoints_ReSTIR_GI::eAreaLights)]] RWStructuredBuffer<AreaLight_ReSTIR_GI, ScalarDataLayout> areaLightsBuffer;
[[vk::binding(StaticBindingPoints_ReSTIR_GI::ePixelData)]] RWStructuredBuffer<PixelData_ReSTIR_GI, ScalarDataLayout> pixelDataBuffer;

//------------------------------------------SharedMemory----------------------------------------
groupshared SceneInfo groupSceneInfo;

#define SharedAreaLightCount 81
groupshared AreaLight_ReSTIR_GI sharedAreaLights[SharedAreaLightCount];

groupshared PixelData_ReSTIR_GI sharedPixelData[16][16];
//-----------------------------------------------HitFunction--------------------------------------------
bool traceRay(RayDesc ray, inout CallablePayload payload, inout int materialIndex){
    // Initialize inline ray query with no special flags
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, ray);

    // Process all potential intersections along the ray
    while (q.Proceed())
    {
        // For non-opaque triangles, force them to be opaque (simplified alpha handling)
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            q.CommitNonOpaqueTriangleHit(); // Accept all intersections as solid
    }

    // Check if ray hit any geometry
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        // Extract intersection data from the committed hit
        float2 barycentricCoords = q.CommittedTriangleBarycentrics(); // (u,v) coordinates on triangle
        float4x3 worldToObject = q.CommittedWorldToObject4x3();       // Transform matrix
        float4x3 objectToWorld = q.CommittedObjectToWorld4x3();       // Inverse transform matrix
        int instanceIndex = q.CommittedInstanceIndex();               // Instance index in scene
        uint primitiveIndex = q.CommittedPrimitiveIndex();

        SceneInfo sceneInfo = groupSceneInfo;
        Instance instance = sceneInfo.instances[instanceIndex]; // Instance data
        Mesh mesh = sceneInfo.meshes[instance.meshIndex];
        materialIndex = instance.materialIndex;
        payload.material = sceneInfo.materials[instance.materialIndex]; // Material properties

        // Convert built-in barycentric coordinates to full barycentric coordinates
        const float3 barycentrics =
            float3(1.0 - barycentricCoords.x - barycentricCoords.y, barycentricCoords.x, barycentricCoords.y);

        // Process intersection data into world-space shading information
        // HitState hitState = getHitStateWithPositionFetch(barycentrics);
        float3 triPos[3];
        triPos = q.CommittedGetIntersectionTriangleVertexPositions();
        const float3 localPosition = triPos[0] * barycentrics.x + triPos[1] * barycentrics.y + triPos[2] * barycentrics.z;
        payload.hitPos = float3(mul(float4(localPosition, 1.0), objectToWorld));

        const float3 edge1 = triPos[1] - triPos[0];
        const float3 edge2 = triPos[2] - triPos[0];
        const float3 geoNormal = normalize(cross(edge1, edge2));

        // payload.hitNormal = normalize(mul(worldToObject, geoNormal).xyz);
        payload.hitNormal = normalize(mul(geoNormal, transpose(worldToObject)).xyz);
        float cosineON = dot(payload.hitNormal, -ray.Direction);
        if (cosineON <= 0.0f) payload.hitNormal = -payload.hitNormal;
        if (payload.material.type != MaterialType::Dielectric && payload.material.type != MaterialType::RoughDielectric)
            payload.material.emissive *= cosineON > 0.0f ? 1.0f : 0.0f;

        float3 tangent, bitangent;
        orthonormalBasis(payload.hitNormal, tangent, bitangent);
        payload.TBN = float3x3(tangent, bitangent, payload.hitNormal); // column matrix

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
bool traceRay2(RayDesc ray, inout float3 hitPos, inout float3 emissive) {
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, ray);

    while (q.Proceed())
    {
        // For non-opaque triangles, force them to be opaque (simplified alpha handling)
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            q.CommitNonOpaqueTriangleHit(); // Accept all intersections as solid
    }

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        float2 barycentricCoords = q.CommittedTriangleBarycentrics(); // (u,v) coordinates on triangle
        float4x3 worldToObject = q.CommittedWorldToObject4x3();       // Transform matrix
        float4x3 objectToWorld = q.CommittedObjectToWorld4x3();       // Inverse transform matrix
        int instanceIndex = q.CommittedInstanceIndex();               // Instance index in scene
        uint primitiveIndex = q.CommittedPrimitiveIndex();

        Instance instance = groupSceneInfo.instances[instanceIndex]; // Instance data


        const float3 barycentrics =
            float3(1.0 - barycentricCoords.x - barycentricCoords.y, barycentricCoords.x, barycentricCoords.y);

        float3 triPos[3];
        triPos = q.CommittedGetIntersectionTriangleVertexPositions();
        const float3 localPosition = triPos[0] * barycentrics.x + triPos[1] * barycentrics.y + triPos[2] * barycentrics.z;
        hitPos = float3(mul(float4(localPosition, 1.0), objectToWorld));

        const float3 edge1 = triPos[1] - triPos[0];
        const float3 edge2 = triPos[2] - triPos[0];
        const float3 geoNormal = normalize(cross(edge1, edge2));

        float3 hitNormal = normalize(mul(geoNormal, transpose(worldToObject)).xyz);
        float cosineON = dot(hitNormal, -ray.Direction);
        if (cosineON <= 0.0f) hitNormal = -hitNormal;

        BSDFMaterial material = groupSceneInfo.materials[instance.materialIndex];
        if (material.type != MaterialType::Dielectric && material.type != MaterialType::RoughDielectric)
            material.emissive *= cosineON > 0.0f ? 1.0f : 0.0f;
        emissive = material.emissive;

        return true;
    }

    return false;
}
float3 RayMiss(float3 rayDirection) {
    SceneInfo sceneInfo = groupSceneInfo;;
    if (sceneInfo.useSky == 1) return evalSimpleSky(sceneInfo.skySimpleParam, rayDirection);
    else return sceneInfo.backgroundColor;
}
//-----------------------------------------------RIS--------------------------------------------
#define NEE_SAMPLE_COUNT 8
#define BSDF_SAMPLE_COUNT 8
void RIS(inout CallablePayload payload, inout Reservoir reservoir, inout float3 f, int areaLightCount) {
    reservoir.M = NEE_SAMPLE_COUNT + BSDF_SAMPLE_COUNT;
    reservoir.W = 0.0f;
    f = 0.0f;
    float p_hat_sample = 1.0f;

    const uint rayFlags = 0;
    RayDesc tempRay;
    tempRay.TMin = 0.001f;
    tempRay.TMax = INFINITE;

    float wSum = 0.0f;
    for (int sampleIndex = 0; sampleIndex < NEE_SAMPLE_COUNT; ++sampleIndex) {
        uint areaLightIndex = int(rand(payload.randomSeed) * areaLightCount);
        float p_nee = 1.0f / areaLightCount;

        AreaLight_ReSTIR_GI areaLight = sharedAreaLights[areaLightIndex];
        Quadrilateral_initialization quad = initQuadrilateral(payload.hitPos, areaLight.startPos, areaLight.edge1, areaLight.edge2);
        if (isnan(quad.S) || quad.S == 0.0f) continue;

        float randomU = rand(payload.randomSeed); float randomV = rand(payload.randomSeed);
        float3 sampleDir = normalize(sphericalRectangleSample(quad, randomU, randomV, p_nee));

        tempRay.Direction = sampleDir;
        tempRay.Origin = payload.hitPos + 0.00001f * tempRay.Direction;
        float3 pos, emissive;
        if (!traceRay2(tempRay, pos, emissive)) continue;
        if (!PointInRectangle(pos, areaLight.startPos, areaLight.edge1, areaLight.edge2)) continue;

        float cosine = abs(dot(sampleDir, payload.hitNormal));
        float3 bsdf = getBSDF(payload.material, sampleDir, payload.hitNormal, payload.outgoing, payload.TBN, payload.isExt);
        float3 f_candidate = emissive * bsdf * cosine;
        float p_hat_candidate = luminance(f_candidate);

        float p_bsdf = getPdf(payload.material, sampleDir, payload.hitNormal, payload.outgoing, payload.TBN, payload.isExt);
        float mi = p_nee / (BSDF_SAMPLE_COUNT * p_bsdf + NEE_SAMPLE_COUNT * p_nee);

        float Wi = 1.0f / p_nee;
        float wi = mi * p_hat_candidate * Wi;

        wSum += wi;
        if (rand(payload.randomSeed) <= wi / max(wSum, 1e-30f)) {
            reservoir.samplePos = pos;
            reservoir.sampleEmissive = emissive;
            f = f_candidate;
            p_hat_sample = p_hat_candidate;
        }
    }

    CallablePayload originalPayload = payload;
    for (int sampleIndex = 0; sampleIndex < BSDF_SAMPLE_COUNT; ++sampleIndex) {
        payload = originalPayload;
        BSDF_SAMPLE(payload);

        tempRay.Direction = payload.incidence;
        tempRay.Origin = payload.hitPos + 0.00001f * tempRay.Direction;
        float3 pos, emissive;
        if (!traceRay2(tempRay, pos, emissive)) continue;

        AreaLight_ReSTIR_GI areaLight; bool inAreaLight = false;
        for (int areaLightIndex = 0; areaLightIndex < areaLightCount; ++areaLightIndex) {
            areaLight = sharedAreaLights[areaLightIndex];
            if ((inAreaLight = PointInRectangle(pos, areaLight.startPos, areaLight.edge1, areaLight.edge2))) break;
        };
        if (!inAreaLight) continue;

        Quadrilateral_initialization quad = initQuadrilateral(payload.hitPos, areaLight.startPos, areaLight.edge1, areaLight.edge2);
        if (isnan(quad.S) || quad.S == 0.0f) continue;
        float p_nee = 1.0f / (areaLightCount * quad.S);

        float cosine = abs(dot(payload.incidence, payload.hitNormal));
        float3 f_candidate = emissive * payload.bsdf * cosine;
        float p_hat_candidate = luminance(f_candidate);

        float mi = payload.pdf / (BSDF_SAMPLE_COUNT * payload.pdf + NEE_SAMPLE_COUNT * p_nee);

        float Wi = 1.0f / payload.pdf;
        float wi = mi * p_hat_candidate * Wi;

        wSum += wi;
        if (rand(payload.randomSeed) <= wi / max(wSum, 1e-30f)) {
            reservoir.samplePos = pos;
            reservoir.sampleEmissive = emissive;
            f = f_candidate;
            p_hat_sample = p_hat_candidate;
        }
    }

    reservoir.W = wSum / max(p_hat_sample, 1e-30f);
    payload = originalPayload;
}
//---------------------------------------------ReSTIR_GI-----------------------------------------
[numthreads(16, 16, 1)]
[shader("compute")]
void computeMain_ReSTIR_GI(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    uint groupThreadIndex_linear = groupThreadIndex.y * 16 + groupThreadIndex.x;
    uint sharedAreaLightCount = min(pushConst.areaLightCount, SharedAreaLightCount);
    if (groupThreadIndex_linear < sharedAreaLightCount) sharedAreaLights[groupThreadIndex_linear] = areaLightsBuffer[groupThreadIndex_linear];
    if (groupThreadIndex_linear == 0) groupSceneInfo = pushConst.sceneInfoAddress[0];
    GroupMemoryBarrierWithGroupSync();

    float2 launchID = (float2)threadIndex;
    float2 launchSize = (float2)pushConst.sceneSize;
    if (any(threadIndex >= pushConst.sceneSize)) return;

    SceneInfo sceneInfo = groupSceneInfo;

    const uint rayFlags = 0; // RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    RayDesc ray;
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    CallablePayload payload;
    payload.randomSeed = xxhash32(uint3(threadIndex, pushConst.frameIndex));

    float r1 = rand(payload.randomSeed);
    float r2 = rand(payload.randomSeed);
    float2 subpixel_jitter = float2(0.5f, 0.5f);

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

    PixelData_ReSTIR_GI pixelData;

    float3 result = 0.0f;
    float3 bsdf_cosine_pdf = 1.0f;
    int materialIndex = -1;
    bool hit = traceRay(ray, payload, materialIndex);
    if (!hit) {
        result += RayMiss(ray.Direction) * bsdf_cosine_pdf;

        pixelData.materialIndex = -1;
        pixelData.reservoir.W = 0.0f;
        pixelData.emissive = result;
    }
    else {
        result += payload.material.emissive * bsdf_cosine_pdf;

        if (pushConst.mode == 0) {
            BSDF_SAMPLE(payload);
            bsdf_cosine_pdf *= payload.bsdf * abs(dot(payload.incidence, payload.hitNormal)) / payload.pdf;

            ray.Direction = payload.incidence;
            ray.Origin = payload.hitPos + 0.00001f * ray.Direction;

            float3 pos, emissive;
            if (traceRay2(ray, pos, emissive)) result += emissive * bsdf_cosine_pdf;
        } else {
            float3 f;
            RIS(payload, pixelData.reservoir, f, sharedAreaLightCount);
            result += f * pixelData.reservoir.W;

            pixelData.materialIndex = materialIndex;
            pixelData.hitPos = payload.hitPos;
            pixelData.hitNormal = payload.hitNormal;
            pixelData.outgoing = payload.outgoing;
            pixelData.emissive = payload.material.emissive;
            pixelData.p_hat = luminance(f);
        }
    }
    if (pushConst.mode == 2) {
        uint pixelIndexU = threadIndex.y * pushConst.sceneSize.x + threadIndex.x;
        pixelDataBuffer[pixelIndexU] = pixelData;
    } else {
        if (pushConst.maxFrameCount > 1 && pushConst.frameIndex > 0) {
            float a = 1.0f / float(pushConst.frameIndex + 1);
            float3 old_color = outImage[int2(launchID)].xyz;
            outImage[int2(launchID)] = float4(lerp(old_color, result, a), 1.0f);
        }
        else outImage[int2(launchID)] = float4(result, 1.0f);
    }
}

#define SpatialReuseCount 15
[numthreads(16, 16, 1)]
[shader("compute")]
void computeMain_ReSTIR_GI_SpatialReuse(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    uint pixelIndex_linear = threadIndex.y * pushConst.sceneSize.x + threadIndex.x;
    sharedPixelData[groupThreadIndex.y][groupThreadIndex.x] = pixelDataBuffer[pixelIndex_linear];
    GroupMemoryBarrierWithGroupSync();
    if (any(threadIndex >= pushConst.sceneSize)) return;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];

    PixelData_ReSTIR_GI pixelData_cur = sharedPixelData[groupThreadIndex.y][groupThreadIndex.x];
    if (pixelData_cur.materialIndex == -1) {
        float2 launchID = (float2)threadIndex;
        if (pushConst.maxFrameCount > 1 && pushConst.frameIndex > 0) {
            float a = 1.0f / float(pushConst.frameIndex + 1);
            float3 old_color = outImage[int2(launchID)].xyz;
            outImage[int2(launchID)] = float4(lerp(old_color, pixelData_cur.emissive, a), 1.0f);
        }
        else outImage[int2(launchID)] = float4(pixelData_cur.emissive, 1.0f);
        return;
    }

    BSDFMaterial material_cur = sceneInfo.materials[pixelData_cur.materialIndex];
    float3 tangent, bitangent;
    orthonormalBasis(pixelData_cur.hitNormal, tangent, bitangent);
    float3x3 TBN_cur = float3x3(tangent, bitangent, pixelData_cur.hitNormal);

    uint2 groupMinPixelXY = threadIndex - groupThreadIndex;
    uint2 groupMaxPixelXY = groupMinPixelXY + uint2(16, 16);

    int2 samplePixels[SpatialReuseCount + 1];
    samplePixels[SpatialReuseCount] = threadIndex;
    float p_select = 1.0f / (M_PI * pushConst.spatialReuseRadius * pushConst.spatialReuseRadius);
    uint randomSeed = xxhash32(uint3(threadIndex, pushConst.frameIndex));
    for (int i = 0; i < SpatialReuseCount; ++i) {
        float2 sampleXY = sample_circle_mean(randomSeed, pushConst.spatialReuseRadius);
        samplePixels[i] = int2(threadIndex) + int2(sampleXY);
    }

    float wSum = 0.0f;
    float sample_p_hat = 1.0f;
    float3 sample_f = 0.0f;
    for (int i = 0; i < SpatialReuseCount + 1; ++i) {
        int2 samplePixel = samplePixels[i];
        if (any(samplePixel < 0) || any(samplePixel >= pushConst.sceneSize)) continue;

        PixelData_ReSTIR_GI pixelData_reuse;
        if (all(samplePixel >= groupMinPixelXY) && all(samplePixel < groupMaxPixelXY))
            pixelData_reuse = sharedPixelData[samplePixel.y - groupMinPixelXY.y][samplePixel.x - groupMinPixelXY.x];
        else pixelData_reuse = pixelDataBuffer[samplePixel.y * pushConst.sceneSize.x + samplePixel.x];
        if (pixelData_reuse.reservoir.W == 0.0f) continue;

        float p_hat_sum = pixelData_reuse.p_hat;
        for (int j = 0; j < SpatialReuseCount + 1; ++j) {
            if (i == j) continue;
            int2 samplePixel = samplePixels[j];
            if (any(samplePixel < 0) || any(samplePixel >= pushConst.sceneSize)) continue;

            PixelData_ReSTIR_GI pixelData_reuse_2;
            if (all(samplePixel >= groupMinPixelXY) && all(samplePixel < groupMaxPixelXY))
                pixelData_reuse_2 = sharedPixelData[samplePixel.y - groupMinPixelXY.y][samplePixel.x - groupMinPixelXY.x];
            else pixelData_reuse_2 = pixelDataBuffer[samplePixel.y * pushConst.sceneSize.x + samplePixel.x];
            if (pixelData_reuse_2.reservoir.W == 0.0f) continue;

            BSDFMaterial material = sceneInfo.materials[pixelData_reuse_2.materialIndex];
            float3 sampleDir = normalize(pixelData_reuse.reservoir.samplePos - pixelData_reuse_2.hitPos);
            orthonormalBasis(pixelData_reuse_2.hitNormal, tangent, bitangent);
            float3x3 TBN = float3x3(tangent, bitangent, pixelData_reuse_2.hitNormal);

            float3 bsdf = getBSDF(material, sampleDir, pixelData_reuse_2.hitNormal, pixelData_reuse_2.outgoing, TBN, true);
            float cosine = abs(dot(sampleDir, pixelData_reuse_2.hitNormal));
            float3 f = pixelData_reuse.reservoir.sampleEmissive * bsdf * cosine;
            float p_hat_2 = luminance(f);
            p_hat_sum += p_hat_2;
        }
        float mi = pixelData_reuse.p_hat / p_hat_sum;

        float3 sampleDir = normalize(pixelData_reuse.reservoir.samplePos - pixelData_cur.hitPos);
        float3 bsdf = getBSDF(material_cur, sampleDir, pixelData_cur.hitNormal, pixelData_cur.outgoing, TBN_cur, true);
        float cosine = abs(dot(sampleDir, pixelData_cur.hitNormal));
        float3 f = pixelData_reuse.reservoir.sampleEmissive * bsdf * cosine;
        float p_hat = luminance(f);
        float wi = mi * p_hat * pixelData_reuse.reservoir.W;

        wSum += wi;
        if (rand(randomSeed) <= wi / max(wSum, 1e-30f)) {
            pixelData_cur.reservoir.samplePos = pixelData_reuse.reservoir.samplePos;
            pixelData_cur.reservoir.sampleEmissive = pixelData_reuse.reservoir.sampleEmissive;
            sample_p_hat = p_hat;
            sample_f = f;
        }
    }
    pixelData_cur.reservoir.W = wSum / sample_p_hat;
    float3 result = pixelData_cur.emissive + sample_f * pixelData_cur.reservoir.W;

    float2 launchID = (float2)threadIndex;
    if (pushConst.maxFrameCount > 1 && pushConst.frameIndex > 0) {
        float a = 1.0f / float(pushConst.frameIndex + 1);
        float3 old_color = outImage[int2(launchID)].xyz;
        outImage[int2(launchID)] = float4(lerp(old_color, result, a), 1.0f);
    }
    else outImage[int2(launchID)] = float4(result, 1.0f);
}