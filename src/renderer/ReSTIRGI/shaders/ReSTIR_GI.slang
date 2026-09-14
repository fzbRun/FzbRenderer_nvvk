#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "renderer/ReSTIRGI/ReSTIR_GIShaderio.h"

[[vk::push_constant]] ConstantBuffer<ReSTIR_GIConstant, ScalarDataLayout> pushConst;
[[vk::binding(StaticBindingPoints_ReSTIR_GI::eAreaLights)]] RWStructuredBuffer<AreaLight_ReSTIR_GI, ScalarDataLayout> areaLightsBuffer;
[[vk::binding(StaticBindingPoints_ReSTIR_GI::ePixelData)]] RWStructuredBuffer<PixelData_ReSTIR_GI, ScalarDataLayout> pixelDataBuffer;
[[vk::binding(StaticBindingPoints_ReSTIR_GI::eVelocityImage)]] RWTexture2D<float4> velocityImage;
[[vk::binding(StaticBindingPoints_ReSTIR_GI::ePixelData_lastFrame)]] RWStructuredBuffer<PixelData_ReSTIR_GI, ScalarDataLayout> pixelDataBuffer_lastFrame;

#define Max_Temporal_Confidence 20

//------------------------------------------SharedMemory----------------------------------------
groupshared SceneInfo groupSceneInfo;

#define SharedAreaLightCount 81
groupshared AreaLight_ReSTIR_GI sharedAreaLights[SharedAreaLightCount];

groupshared PixelData_ReSTIR_GI sharedPixelData[Spatial_Reuse_GroupSize][Spatial_Reuse_GroupSize];
//-----------------------------------------------CommonFunction----------------------------------------
bool getPixelData(int2 pixelIndex, int2 groupThreadIndex, int2 samplePixel, inout PixelData_ReSTIR_GI pixelData) {
    uint2 groupMinPixelXY = pixelIndex - groupThreadIndex;
    uint2 groupMaxPixelXY = min(groupMinPixelXY + uint2(Spatial_Reuse_GroupSize, Spatial_Reuse_GroupSize), pushConst.screenSize);

    if (all(samplePixel >= groupMinPixelXY) && all(samplePixel < groupMaxPixelXY))
        pixelData = sharedPixelData[samplePixel.y - groupMinPixelXY.y][samplePixel.x - groupMinPixelXY.x];
    else pixelData = pixelDataBuffer[samplePixel.y * pushConst.screenSize.x + samplePixel.x];
    if (pixelData.materialIndex == -1 || pixelData.reservoir.W == 0.0f) return false;

    return true;
}
//-----------------------------------------------HitFunction--------------------------------------------
bool traceRay(RayDesc ray, inout CallablePayload payload, inout int materialIndex, inout float2 texCoords){
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
            texCoords = getTriangleAttribute<float2>(mesh.dataBuffer, mesh.triMesh.texCoords, indices, barycentrics);
            texCoords.y = 1.0f - texCoords.y;
            getTextueData(payload, texCoords);
        }

        return true;
    }
    return false;
}
bool traceRay2(RayDesc ray, inout float3 hitPos, inout float3 emissive) {
    if(any(isnan(ray.Direction))) return false;

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

        BSDFMaterial material = groupSceneInfo.materials[instance.materialIndex];
        if (material.type != MaterialType::Dielectric && material.type != MaterialType::RoughDielectric)
            material.emissive *= cosineON > 0.0f ? 1.0f : 0.0f;
        emissive = material.emissive;

        return true;
    }

    return false;
}
bool traceRay3(RayDesc ray, inout float3 hitPos) {
    if (any(isnan(ray.Direction))) return false;

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
        float4x3 objectToWorld = q.CommittedObjectToWorld4x3();       // Inverse transform matrix
        int instanceIndex = q.CommittedInstanceIndex();               // Instance index in scene

        Instance instance = groupSceneInfo.instances[instanceIndex]; // Instance data

        const float3 barycentrics =
            float3(1.0 - barycentricCoords.x - barycentricCoords.y, barycentricCoords.x, barycentricCoords.y);

        float3 triPos[3];
        triPos = q.CommittedGetIntersectionTriangleVertexPositions();
        const float3 localPosition = triPos[0] * barycentrics.x + triPos[1] * barycentrics.y + triPos[2] * barycentrics.z;
        hitPos = float3(mul(float4(localPosition, 1.0), objectToWorld));

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
#define NEE_SAMPLE_COUNT 1
#define BSDF_SAMPLE_COUNT 1
void RIS(inout CallablePayload payload, inout Reservoir reservoir, inout float3 f, int areaLightCount) {
    reservoir.M = NEE_SAMPLE_COUNT + BSDF_SAMPLE_COUNT;
    reservoir.W = 0.0f;
    f = 0.0f;
    float p_hat_sample = 1.0f;

    RayDesc tempRay;
    tempRay.TMin = 0.001f;
    tempRay.TMax = INFINITE;

    float wSum = 0.0f;
    for (int sampleIndex = 0; sampleIndex < NEE_SAMPLE_COUNT; ++sampleIndex) {
        uint areaLightIndex = min(int(rand(payload.randomSeed) * areaLightCount), areaLightCount - 1);
        float p_nee = 1.0f / areaLightCount;

        AreaLight_ReSTIR_GI areaLight = sharedAreaLights[areaLightIndex];
        Quadrilateral_initialization quad = initQuadrilateral(payload.hitPos, areaLight.startPos, areaLight.edge1, areaLight.edge2);
        if (isnan(quad.S) || quad.S == 0.0f || abs(quad.z0) < 1e-6f) continue;

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
        float p_hat_candidate = dot(f_candidate, 1.0f);

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
        if (isnan(quad.S) || quad.S == 0.0f || abs(quad.z0) < 1e-6f) continue;
        float p_nee = 1.0f / (areaLightCount * quad.S);

        float cosine = abs(dot(payload.incidence, payload.hitNormal));
        float3 f_candidate = emissive * payload.bsdf * cosine;
        float p_hat_candidate = dot(f_candidate, 1.0f);

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
    float2 launchSize = (float2)pushConst.screenSize;
    if (any(threadIndex >= pushConst.screenSize)) return;

    SceneInfo sceneInfo = groupSceneInfo;

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
    int materialIndex = -1; float2 texCoords;
    bool hit = traceRay(ray, payload, materialIndex, texCoords);
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
            pixelData.texCoords = texCoords;
            pixelData.hitPos = payload.hitPos;
            pixelData.hitNormal = payload.hitNormal;
            pixelData.outgoing = payload.outgoing;
            pixelData.emissive = payload.material.emissive;
            pixelData.f = f;
        }
    }
    if (pushConst.mode >= 2) {
        uint pixelIndexU = threadIndex.y * pushConst.screenSize.x + threadIndex.x;
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

#define SpatialReuseCount 4
[numthreads(Spatial_Reuse_GroupSize, Spatial_Reuse_GroupSize, 1)]
[shader("compute")]
void computeMain_ReSTIR_GI_SpatialReuse(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    if (all(threadIndex < pushConst.screenSize)) {
        uint pixelIndex_linear = threadIndex.y * pushConst.screenSize.x + threadIndex.x;
        sharedPixelData[groupThreadIndex.y][groupThreadIndex.x] = pixelDataBuffer[pixelIndex_linear];
    }
    if (all(groupThreadIndex == 0)) groupSceneInfo = pushConst.sceneInfoAddress[0];
    GroupMemoryBarrierWithGroupSync();
    if (any(threadIndex >= pushConst.screenSize)) return;

    SceneInfo sceneInfo = groupSceneInfo;

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

    BSDFMaterial material_cur = groupSceneInfo.materials[pixelData_cur.materialIndex];
    getBSDFTextureData(material_cur, pixelData_cur.texCoords);
    float3 tangent, bitangent;
    orthonormalBasis(pixelData_cur.hitNormal, tangent, bitangent);
    float3x3 TBN_cur = float3x3(tangent, bitangent, pixelData_cur.hitNormal);

    int2 samplePixels[SpatialReuseCount];
    //samplePixels[SpatialReuseCount - 1] = threadIndex;
    float p_select = 1.0f / (pushConst.spatialReuseRadius * pushConst.spatialReuseRadius);
    uint randomSeed = xxhash32(uint3(threadIndex, pushConst.frameIndex));
    for (int i = 0; i < SpatialReuseCount; ++i) {
        //int sampleIndex = int(pushConst.spatialReuseRadius * pushConst.spatialReuseRadius * rand(randomSeed));
        //int2 sampleXY = int2(sampleIndex % pushConst.spatialReuseRadius, sampleIndex / pushConst.spatialReuseRadius) - pushConst.spatialReuseRadius / 2;
        //samplePixels[i] = int2(threadIndex) + sampleXY;

        int2 randNeighbor = (int2)sample_circle_mean(randomSeed, (float)pushConst.spatialReuseRadius);
        samplePixels[i] = int2(threadIndex) + randNeighbor;
    }

    RayDesc tempRay;
    tempRay.TMin = 0.001f;
    tempRay.TMax = INFINITE;

    pixelData_cur.reservoir.M = 0;

    int M_sum = 0;
    float wSum = 0.0f;
    float p_hat_y = 1.0f;
    float3 f_y = 0.0f;
    for (int i = 0; i < SpatialReuseCount; ++i) {
        int2 samplePixel = samplePixels[i];
        if (any(samplePixel < 0) || any(samplePixel >= pushConst.screenSize)) continue;

        PixelData_ReSTIR_GI pixelData_x_i;
        if (!getPixelData(threadIndex, groupThreadIndex, samplePixel, pixelData_x_i)) continue;

        //bool geometryChange = dot(pixelData_cur.hitNormal, pixelData_x_i.hitNormal) < 0.8f;
        //geometryChange |= distance(pixelData_cur.hitPos, pixelData_x_i.hitPos) > 0.1f;
        //if (geometryChange) continue;

        M_sum += pixelData_x_i.reservoir.M;

        float3 sampleDir = normalize(pixelData_x_i.reservoir.samplePos - pixelData_cur.hitPos);
        tempRay.Direction = sampleDir;
        tempRay.Origin = pixelData_cur.hitPos + 0.00001f * tempRay.Direction;
        float3 pos;
        if (!traceRay3(tempRay, pos)) continue;
        if (distance(pos, pixelData_x_i.reservoir.samplePos) > 0.01f) continue;

        float3 bsdf = getBSDF(material_cur, sampleDir, pixelData_cur.hitNormal, pixelData_cur.outgoing, TBN_cur, true);
        float cosine = abs(dot(sampleDir, pixelData_cur.hitNormal));
        float3 f_x_i = pixelData_x_i.reservoir.sampleEmissive * bsdf * cosine;
        float p_hat_x_i = dot(f_x_i, 1.0f);
        if (p_hat_x_i == 0.0f) continue;

        float p_x_i = dot(pixelData_x_i.f, 1.0f); // * (i == SpatialReuseCount - 1 ? 1.0f : p_select);
        float p_x_j_sum = p_x_i * pixelData_x_i.reservoir.M;
        for (int j = 0; j < SpatialReuseCount; ++j) {
            if (i == j) continue;
            int2 samplePixel = samplePixels[j];
            if (any(samplePixel < 0) || any(samplePixel >= pushConst.screenSize)) continue;

            PixelData_ReSTIR_GI pixelData_x_j;
            if (!getPixelData(threadIndex, groupThreadIndex, samplePixel, pixelData_x_j)) continue;

            BSDFMaterial material_x_j = sceneInfo.materials[pixelData_x_j.materialIndex];
            getBSDFTextureData(material_x_j, pixelData_x_j.texCoords);

            sampleDir = normalize(pixelData_x_i.reservoir.samplePos - pixelData_x_j.hitPos);
            tempRay.Direction = sampleDir;
            tempRay.Origin = pixelData_x_j.hitPos + 0.00001f * tempRay.Direction;
            if (!traceRay3(tempRay, pos)) continue;
            if (distance(pos, pixelData_x_i.reservoir.samplePos) > 0.01f) continue;

            orthonormalBasis(pixelData_x_j.hitNormal, tangent, bitangent);
            float3x3 TBN = float3x3(tangent, bitangent, pixelData_x_j.hitNormal);

            bsdf = getBSDF(material_x_j, sampleDir, pixelData_x_j.hitNormal, pixelData_x_j.outgoing, TBN, true);
            cosine = abs(dot(sampleDir, pixelData_x_j.hitNormal));
            float3 f_x_j = pixelData_x_i.reservoir.sampleEmissive * bsdf * cosine;
            float p_x_j = dot(f_x_j, 1.0f); // * (j == SpatialReuseCount - 1 ? 1.0f : p_select);
            p_x_j_sum += p_x_j * pixelData_x_j.reservoir.M;
        }
        float mi = p_x_i * pixelData_x_i.reservoir.M / max(p_x_j_sum, 1e-30f);
        float wi = mi * p_hat_x_i * pixelData_x_i.reservoir.W; // / (i == SpatialReuseCount - 1 ? 1.0f : p_select);

        wSum += wi;
        if (rand(randomSeed) <= wi / max(wSum, 1e-30f)) {
            p_hat_y = p_hat_x_i;
            f_y = f_x_i;

            pixelData_cur.reservoir.samplePos = pixelData_x_i.reservoir.samplePos;
            pixelData_cur.reservoir.sampleEmissive = pixelData_x_i.reservoir.sampleEmissive;
        }
    }
    float W_y = wSum / p_hat_y;
    pixelData_cur.reservoir.W = W_y;
    pixelData_cur.reservoir.M = M_sum;
    pixelData_cur.f = f_y;
    uint pixelIndexU = threadIndex.y * pushConst.screenSize.x + threadIndex.x;
    pixelDataBuffer[pixelIndexU] = pixelData_cur;

    float3 result = pixelData_cur.emissive + f_y * W_y;
    float2 launchID = (float2)threadIndex;
    if (pushConst.maxFrameCount > 1 && pushConst.frameIndex > 0) {
        float a = 1.0f / float(pushConst.frameIndex + 1);
        float3 old_color = outImage[int2(launchID)].xyz;
        outImage[int2(launchID)] = float4(lerp(old_color, result, a), 1.0f);
    }
    else outImage[int2(launchID)] = float4(result, 1.0f);
}

[numthreads(Spatial_Reuse_GroupSize, Spatial_Reuse_GroupSize, 1)]
[shader("compute")]
void computeMain_ReSTIR_GI_TemporalReuse(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    if (all(groupThreadIndex == 0)) groupSceneInfo = pushConst.sceneInfoAddress[0];
    GroupMemoryBarrierWithGroupSync();
    if (any(threadIndex >= pushConst.screenSize)) return;

    SceneInfo sceneInfo = groupSceneInfo;
    float2 launchID = (float2)threadIndex;
    float a = 1.0f / float(pushConst.frameIndex + 1);

    uint pixelIndex_linear = threadIndex.y * pushConst.screenSize.x + threadIndex.x;
    PixelData_ReSTIR_GI pixelData_cur = pixelDataBuffer[pixelIndex_linear];

    if (pixelData_cur.materialIndex == -1) return;

    float3 result = pixelData_cur.emissive + pixelData_cur.f * pixelData_cur.reservoir.W;
    if (pushConst.frameIndex == 0) return;

    float2 velocity = velocityImage[threadIndex].xy;
    float2 screenPos = (float2(threadIndex) + 0.5f) / pushConst.screenSize;
    float2 offsetScreenPos = screenPos - velocity;
    if (any(offsetScreenPos < 0.0f) || any(offsetScreenPos >= 1.0f)) return;

    int2 lastPixelIndex = int2(offsetScreenPos * pushConst.screenSize);
    uint lastPixelIndex_linear = lastPixelIndex.y * pushConst.screenSize.x + lastPixelIndex.x;
    PixelData_ReSTIR_GI pixelData_last = pixelDataBuffer_lastFrame[lastPixelIndex_linear];
    if (pixelData_last.materialIndex == -1 || pixelData_last.reservoir.W == 0.0f) return;

    bool geometryChange = dot(pixelData_cur.hitNormal, pixelData_last.hitNormal) < 0.9f;
    geometryChange |= distance(pixelData_cur.hitPos, pixelData_last.hitPos) > 0.01f;
    if (geometryChange) return;

    uint randomSeed = xxhash32(uint3(threadIndex, pushConst.frameIndex));

    RayDesc tempRay;
    tempRay.TMin = 0.001f;
    tempRay.TMax = INFINITE;

    BSDFMaterial material_cur = groupSceneInfo.materials[pixelData_cur.materialIndex];
    getBSDFTextureData(material_cur, pixelData_cur.texCoords);
    float3 tangent, bitangent;
    orthonormalBasis(pixelData_cur.hitNormal, tangent, bitangent);
    float3x3 TBN_cur = float3x3(tangent, bitangent, pixelData_cur.hitNormal);

    BSDFMaterial material_last = groupSceneInfo.materials[pixelData_last.materialIndex];
    getBSDFTextureData(material_last, pixelData_last.texCoords);
    orthonormalBasis(pixelData_last.hitNormal, tangent, bitangent);
    float3x3 TBN_last = float3x3(tangent, bitangent, pixelData_last.hitNormal);

    int M_cur = pixelData_cur.reservoir.M;
    int M_last = pixelData_last.reservoir.M;
    float3 samplePos = 0.0f, sampleEmissive = 0.0f;
    float wSum = 0.0f; float3 f_y = 0.0f;
    {
        //float3 f_curPixel = 0.0f;
        //float p_hat_curPixel = 0.0f;
        //float p_hat_lastPixel = dot(pixelData_last.f, 1.0f);
        //float mis_sum = M_last * p_hat_lastPixel;
        //
        //float3 sampleDir = normalize(pixelData_last.reservoir.samplePos - pixelData_cur.hitPos);
        //tempRay.Direction = sampleDir;
        //tempRay.Origin = pixelData_cur.hitPos + 0.00001f * tempRay.Direction;
        //float3 pos;
        //if (traceRay3(tempRay, pos)) {
        //    if (distance(pos, pixelData_last.reservoir.samplePos) <= 0.01f) {
        //        float3 bsdf = getBSDF(material_cur, sampleDir, pixelData_cur.hitNormal, pixelData_cur.outgoing, TBN_cur, true);
        //        float cosine = abs(dot(sampleDir, pixelData_cur.hitNormal));
        //        f_curPixel = pixelData_last.reservoir.sampleEmissive * bsdf * cosine;
        //        p_hat_curPixel = dot(f_curPixel, 1.0f);
        //        mis_sum += M_cur * p_hat_curPixel;
        //
        //        M_final += M_last;
        //    }
        //}

        float p_hat_curPixel = dot(pixelData_last.f, 1.0f);
        float mi = (float)M_last / (M_cur + M_last);    //M_last * p_hat_lastPixel / mis_sum;
        float wi = mi * p_hat_curPixel * pixelData_last.reservoir.W;
        wSum += wi;
        if (rand(randomSeed) <= wi / max(wSum, 1e-30f)) {
            samplePos = pixelData_last.reservoir.samplePos;
            sampleEmissive = pixelData_last.reservoir.sampleEmissive;

            f_y = pixelData_last.f;
        }
    }
    if (pixelData_cur.reservoir.W > 0.0f){
        //float p_hat_curPixel = dot(pixelData_cur.f, 1.0f);
        //float p_hat_lastPixel = 0.0f;
        //float mis_sum = M_cur * p_hat_curPixel;
        //
        //float3 sampleDir = normalize(pixelData_cur.reservoir.samplePos - pixelData_last.hitPos);
        //tempRay.Direction = sampleDir;
        //tempRay.Origin = pixelData_last.hitPos + 0.00001f * tempRay.Direction;
        //float3 pos;
        //if (traceRay3(tempRay, pos)) {
        //    if (distance(pos, pixelData_cur.reservoir.samplePos) <= 0.01f) {
        //        float3 bsdf = getBSDF(material_last, sampleDir, pixelData_last.hitNormal, pixelData_last.outgoing, TBN_last, true);
        //        float cosine = abs(dot(sampleDir, pixelData_last.hitNormal));
        //        float3 f_lastPixel = pixelData_cur.reservoir.sampleEmissive * bsdf * cosine;
        //        p_hat_lastPixel = dot(f_lastPixel, 1.0f);
        //        mis_sum += M_last * p_hat_lastPixel;
        //    }
        //}

        float p_hat_curPixel = dot(pixelData_cur.f, 1.0f);
        float mi = (float)M_cur / (M_cur + M_last);    //M_cur * p_hat_curPixel / mis_sum;
        float wi = mi * p_hat_curPixel * pixelData_cur.reservoir.W;
        wSum += wi;
        if (rand(randomSeed) <= wi / max(wSum, 1e-30f)) {
            samplePos = pixelData_cur.reservoir.samplePos;
            sampleEmissive = pixelData_cur.reservoir.sampleEmissive;

            f_y = pixelData_cur.f;
        }
    }

    //if (pushConst.frameIndex == 0 && all(f_y == 0.0f)) printf("wSum: %f\n", wSum);

    pixelData_cur.reservoir.samplePos = samplePos;
    pixelData_cur.reservoir.sampleEmissive = sampleEmissive;
    pixelData_cur.reservoir.W = wSum / dot(f_y, 1.0f);
    pixelData_cur.reservoir.M = min(M_cur + M_last, Max_Temporal_Confidence);
    pixelData_cur.f = f_y;
    uint pixelIndexU = threadIndex.y * pushConst.screenSize.x + threadIndex.x;
    pixelDataBuffer[pixelIndexU] = pixelData_cur;
}