#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "renderer/PathTracing_KPCNNDenoising/PathTracing_KPCNNDenosingShaderio.h"

[[vk::push_constant]] ConstantBuffer<KPCNN_DenoisingPTPushConstant, ScalarDataLayout> pushConst;
[[vk::binding(StaticBindingPoints_KPCNNPT::eInputBuffer_irradiance)]] RWStructuredBuffer<float3> InputBuffer_irradiance;
[[vk::binding(StaticBindingPoints_KPCNNPT::eInputBuffer_irradianceVariance)]] RWStructuredBuffer<float> InputBuffer_irradianceVariance;
[[vk::binding(StaticBindingPoints_KPCNNPT::eInputBuffer_gradIrradiance)]] RWStructuredBuffer<float3> InputBuffer_gradIrradiance;

[[vk::binding(StaticBindingPoints_KPCNNPT::eInputBuffer_normalVariance_diff)]] RWStructuredBuffer<float> InputBuffer_normalVariance_diff;
[[vk::binding(StaticBindingPoints_KPCNNPT::eInputBuffer_gradNormal_diff)]] RWStructuredBuffer<float3> InputBuffer_gradNormal_diff;

[[vk::binding(StaticBindingPoints_KPCNNPT::eInputBuffer_depthVariance_diff)]] RWStructuredBuffer<float> InputBuffer_depthVariance_diff;
[[vk::binding(StaticBindingPoints_KPCNNPT::eInputBuffer_gradDepth_diff)]] RWStructuredBuffer<float> InputBuffer_gradDepth_diff;

[[vk::binding(StaticBindingPoints_KPCNNPT::eInputBuffer_albedoVariance_diff)]] RWStructuredBuffer<float> InputBuffer_albedoVariance_diff;
[[vk::binding(StaticBindingPoints_KPCNNPT::eInputBuffer_gradAlbedo_diff)]] RWStructuredBuffer<float3> InputBuffer_gradAlbedo_diff;

[[vk::binding(StaticBindingPoints_KPCNNPT::eInputBuffer_spec)]] RWStructuredBuffer<float3> InputBuffer_spec;
[[vk::binding(StaticBindingPoints_KPCNNPT::eInputBuffer_specVariance)]] RWStructuredBuffer<float> InputBuffer_specVariance;
[[vk::binding(StaticBindingPoints_KPCNNPT::eInputBuffer_gradSpec)]] RWStructuredBuffer<float3> InputBuffer_gradSpec;

[[vk::binding(StaticBindingPoints_KPCNNPT::eInputBuffer_normalVariance_spec)]] RWStructuredBuffer<float> InputBuffer_normalVariance_spec;
[[vk::binding(StaticBindingPoints_KPCNNPT::eInputBuffer_gradNormal_spec)]] RWStructuredBuffer<float3> InputBuffer_gradNormal_spec;

[[vk::binding(StaticBindingPoints_KPCNNPT::eInputBuffer_depthVariance_spec)]] RWStructuredBuffer<float> InputBuffer_depthVariance_spec;
[[vk::binding(StaticBindingPoints_KPCNNPT::eInputBuffer_gradDepth_spec)]] RWStructuredBuffer<float> InputBuffer_gradDepth_spec;

[[vk::binding(StaticBindingPoints_KPCNNPT::eInputBuffer_albedoVariance_spec)]] RWStructuredBuffer<float> InputBuffer_albedoVariance_spec;
[[vk::binding(StaticBindingPoints_KPCNNPT::eInputBuffer_gradAlbedo_spec)]] RWStructuredBuffer<float3> InputBuffer_gradAlbedo_spec;

[[vk::binding(StaticBindingPoints_KPCNNPT::eNormalBuffer)]] RWStructuredBuffer<float3> NormalBuffer;
[[vk::binding(StaticBindingPoints_KPCNNPT::eDepthBuffer)]] RWStructuredBuffer<float> DepthBuffer;
[[vk::binding(StaticBindingPoints_KPCNNPT::eMaxDepthBuffer)]] RWStructuredBuffer<int> MaxDepthBuffer;
[[vk::binding(StaticBindingPoints_KPCNNPT::eAlbedoBuffer)]] RWStructuredBuffer<float3> AlbedoBuffer;

#ifndef NDEBUG
[[vk::binding(StaticBindingPoints_KPCNNPT::eColorDebugImage, 0)]] RWTexture2D<float4> ColorDebugImage;
[[vk::binding(StaticBindingPoints_KPCNNPT::eDiffuseDebugImage, 0)]] RWTexture2D<float4> DiffuseDebugImage;
[[vk::binding(StaticBindingPoints_KPCNNPT::eSpecularDebugImage, 0)]] RWTexture2D<float4> SpecularDebugImage;
[[vk::binding(StaticBindingPoints_KPCNNPT::eIrradianceDebugImage, 0)]] RWTexture2D<float4> IrradianceDebugImage;
[[vk::binding(StaticBindingPoints_KPCNNPT::eNormalDebugImage, 0)]] RWTexture2D<float4> NormalDebugImage;
[[vk::binding(StaticBindingPoints_KPCNNPT::eDepthDebugImage, 0)]] RWTexture2D<float4> DepthDebugImage;
[[vk::binding(StaticBindingPoints_KPCNNPT::eAlebdoDebugImage, 0)]] RWTexture2D<float4> AlbedoDebugImage;

[[vk::binding(StaticBindingPoints_KPCNNPT::eIrradianceVarianceDebugImage, 0)]] RWTexture2D<float4> IrradianceVarianceDebugImage;
[[vk::binding(StaticBindingPoints_KPCNNPT::eSpecularVariancDebugImage, 0)]] RWTexture2D<float4> SpecularVarianceDebugImage;
[[vk::binding(StaticBindingPoints_KPCNNPT::eNormalVarianceDebugImage, 0)]] RWTexture2D<float4> NormalVarianceDebugImage;
[[vk::binding(StaticBindingPoints_KPCNNPT::eDepthVarianceDebugImage, 0)]] RWTexture2D<float4> DepthVarianceDebugImage;
[[vk::binding(StaticBindingPoints_KPCNNPT::eAlbedoVarianceDebugImage, 0)]] RWTexture2D<float4> AlbedoVarianceDebugImage;
#endif
//---------------------------------------------PathTracing-----------------------------------------
typedef CallablePayload HitPayload_KPCNNPT;

bool traceRay(RayDesc ray, inout HitPayload_KPCNNPT payload)
{
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

        SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
        Instance instance = sceneInfo.instances[instanceIndex]; // Instance data
        Mesh mesh = sceneInfo.meshes[instance.meshIndex];
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
float3 RayMiss(float3 rayDirection) {
    SceneInfo *sceneInfo = pushConst.sceneInfoAddress;
    if (sceneInfo.useSky == 1) return evalSimpleSky(sceneInfo.skySimpleParam, rayDirection);
    else return sceneInfo.backgroundColor;
}

groupshared float groupWarpMaxDepth[(PATHTRACING_BLOCKSIZE_KPCNN * PATHTRACING_BLOCKSIZE_KPCNN + 31) / 32];

[numthreads(PATHTRACING_BLOCKSIZE_KPCNN, PATHTRACING_BLOCKSIZE_KPCNN, 1)]
[shader("compute")]
void computeMain_PathTracing(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    float2 launchID = (float2)threadIndex;
    float2 launchSize = (float2)pushConst.screenSize;

    uint groupThreadIndexLinear = groupThreadIndex.y * PATHTRACING_BLOCKSIZE_KPCNN + groupThreadIndex.x;
    if (groupThreadIndexLinear < (PATHTRACING_BLOCKSIZE_KPCNN * PATHTRACING_BLOCKSIZE_KPCNN + 31) / 32) groupWarpMaxDepth[groupThreadIndexLinear] = 0.0f;
    //don't need sync, becouse sync will be done in later
    if (threadIndex.x >= pushConst.screenSize.x || threadIndex.y >= pushConst.screenSize.y) return;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];

    const uint rayFlags = 0; // RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    RayDesc ray;
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    HitPayload_KPCNNPT payload;
    payload.randomSeed = xxhash32(uint3(threadIndex, pushConst.frameIndex));

    float RR = 0.8f;

    float3 accNormal = float3(0.0f);
    float3 accNormal2 = float3(0.0f);
    float accDepth = 0.0f;
    float accDepth2 = 0.0f;
    float3 accAlbedo = float3(0.0f);
    float3 accAlbedo2 = float3(0.0f);

    float3 accDiffuse = float3(0.0f);
    float3 accDiffuse2 = float3(0.0f);
    float3 accSpecular = float3(0.0f);
    float3 accSpecular2 = float3(0.0f);

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

        float3 diffuse = float3(0.0f), specular = float3(0.0f);
        bool isDiffuse = false;
        bool notHitDiffuse = true;

        float3 bsdf_cosine_pdf = float3(1.0f);
        for (uint bounceDepth = 0; bounceDepth < pushConst.maxBounceCount; ++bounceDepth) {
            bool hit = traceRay(ray, payload);

            if (!hit) {
                float3 missColor = RayMiss(ray.Direction) * bsdf_cosine_pdf;
                if (isDiffuse) diffuse += missColor;
                else specular += missColor;

                bounceDepth = MISS_DEPTH;
                break;
            }

            if (notHitDiffuse) {
                if (payload.material.roughness > 0.1f) { // diffuse
                    notHitDiffuse = false;
                    isDiffuse = bounceDepth == 0;
            
                    accNormal += payload.hitNormal;
                    accNormal2 += payload.hitNormal * payload.hitNormal;
            
                    float depth = length(payload.hitPos - sceneInfo.cameraPosition);
                    accDepth += depth;
                    accDepth2 += depth * depth;
            
                    accAlbedo += payload.material.albedo;
                    accAlbedo2 += payload.material.albedo * payload.material.albedo;
                }
            }

            if (isDiffuse) diffuse += payload.material.emissive * bsdf_cosine_pdf;
            else specular += payload.material.emissive * bsdf_cosine_pdf;

            BSDF_SAMPLE(payload);
            float cosine = abs(dot(payload.hitNormal, payload.incidence));
            bsdf_cosine_pdf *= payload.bsdf * cosine / payload.pdf;

            bool end = any(bsdf_cosine_pdf <= 0.0f) || all(bsdf_cosine_pdf <= 0.0f) || any(isnan(bsdf_cosine_pdf)) || any(isinf(bsdf_cosine_pdf));
            if (end) break;

            if (rand(payload.randomSeed) >= RR) break;
            bsdf_cosine_pdf /= RR;

            ray.Origin = payload.hitPos + 0.001f * payload.incidence;
            ray.Direction = payload.incidence;
        }

        if (isDiffuse) {
            accDiffuse += diffuse;
            accDiffuse2 += diffuse * diffuse;
        }else{
            accSpecular += specular;
            accSpecular2 += specular * specular;
        }
    }

    uint pixelIndex = threadIndex.y * pushConst.screenSize.x + threadIndex.x;
    //-------------------------------irradiance-------------------------------------
    float3 meanAlbedo = accAlbedo / pushConst.spp;
    float3 meanDiffuse = accDiffuse / pushConst.spp;

    float3 irradiance = meanDiffuse / (meanAlbedo + eps_kpcnn);
    InputBuffer_irradiance[pixelIndex] = irradiance;

    /*
    Because the variance of the training data set is single channel, we can only get the lightness first and then divide it
    by the square of the mean before the training. Therefore, we process it according to the scheme before the training
    */
    // float3 irradianceVariance = (accDiffuse2 / pushConst.spp - meanDiffuse * meanDiffuse) / (meanAlbedo * meanAlbedo);
    float3 irradianceVariance = accDiffuse2 / pushConst.spp - meanDiffuse * meanDiffuse;
    float irradianceVarianceS = irradianceVariance.x * 0.2126f + irradianceVariance.y * 0.7152f + irradianceVariance.z * 0.0722f;

    //if(pushConst.frameIndex == 1 && any(accDiffuse > 0.0f)) printf("X: %d Y: %d,  accDiffuse: %f %f %f, accDiffuse2: %f %f %f, V: %f\n",
    //    threadIndex.x, threadIndex.y, 
    //    accDiffuse.x, accDiffuse.y, accDiffuse.z,
    //    accDiffuse2.x, accDiffuse2.y, accDiffuse2.z,
    //    irradianceVarianceS
    //);

    float meanAlbedoS = meanAlbedo.x * 0.2126f + meanAlbedo.y * 0.7152f + meanAlbedo.z * 0.0722f + eps_kpcnn;
    irradianceVarianceS /= (meanAlbedoS * meanAlbedoS);
    InputBuffer_irradianceVariance[pixelIndex] = irradianceVarianceS;
    //-------------------------------specular-------------------------------------
    float3 meanSpecular = max(accSpecular / pushConst.spp, 0.0f);
    InputBuffer_spec[pixelIndex] = log(meanSpecular + 1);

    float3 specularVariance = accSpecular2 / pushConst.spp - meanSpecular * meanSpecular;

    // float3 logSpecularVariance = specularVariance / (meanSpecular * meanSpecular);
    float specularVarianceS = specularVariance.x * 0.2126f + specularVariance.y * 0.7152f + specularVariance.z * 0.0722f;
    float meanSpecularS = meanSpecular.x * 0.2126f + meanSpecular.y * 0.7152f + meanSpecular.z * 0.0722f + eps_kpcnn;
    float logSpecularVariance = specularVarianceS / (meanSpecularS * meanSpecularS);

    InputBuffer_specVariance[pixelIndex] = logSpecularVariance;
    //-------------------------------normal-------------------------------------
    float3 meanNormal = accNormal / pushConst.spp;
    float3 normalVariance = accNormal2 / pushConst.spp - meanNormal * meanNormal;
    float normalVarianceS = normalVariance.x * 0.2126f + normalVariance.y * 0.7152f + normalVariance.z * 0.0722f;

    InputBuffer_normalVariance_diff[pixelIndex] = normalVarianceS;
    InputBuffer_normalVariance_spec[pixelIndex] = normalVarianceS;

    NormalBuffer[pixelIndex] = meanNormal;
    //-------------------------------depth-------------------------------------
    float meanDepth = accDepth / pushConst.spp;
    float depthVariance = accDepth2 / pushConst.spp - meanDepth * meanDepth;

    InputBuffer_depthVariance_diff[pixelIndex] = depthVariance;
    InputBuffer_depthVariance_spec[pixelIndex] = depthVariance;

    DepthBuffer[pixelIndex] = meanDepth;

    uint warpIndex = groupThreadIndexLinear / 32;
    uint warpLane = groupThreadIndexLinear & 31;
    float warpMaxDepth = WaveActiveMax(meanDepth);
    if (warpLane == 0) groupWarpMaxDepth[warpIndex] = warpMaxDepth;
    GroupMemoryBarrierWithGroupSync();
    if (warpIndex == 0 && warpLane < (PATHTRACING_BLOCKSIZE_KPCNN * PATHTRACING_BLOCKSIZE_KPCNN + 31) / 32) {
        warpMaxDepth = groupWarpMaxDepth[warpLane];
        float groupMaxDepth = WaveActiveMax(warpMaxDepth);
        if (warpLane == 0) {
            int groupMaxDepthU = FloatToOrderedInt(groupMaxDepth);
            InterlockedMax<int>(MaxDepthBuffer[0], groupMaxDepthU);
        }
    }
    //-------------------------------albedo-------------------------------------
    float3 albedoVariance = accAlbedo2 / pushConst.spp - meanAlbedo * meanAlbedo;
    float albedoVarianceS = albedoVariance.x * 0.2126f + albedoVariance.y * 0.7152f + albedoVariance.z * 0.0722f;

    InputBuffer_albedoVariance_diff[pixelIndex] = albedoVarianceS;

    InputBuffer_albedoVariance_spec[pixelIndex] = albedoVarianceS;

    AlbedoBuffer[pixelIndex] = meanAlbedo;
    //if (pushConst.frameIndex == 1) printf("albedo: %f %f %f\n", meanAlbedo.x, meanAlbedo.y, meanAlbedo.z);

    #ifndef NDEBUG
    DiffuseDebugImage[threadIndex] = float4(meanDiffuse, 1.0f);
    SpecularDebugImage[threadIndex] = float4(meanSpecular, 1.0f);
    float3 color = meanDiffuse + meanSpecular;
    ColorDebugImage[threadIndex]= float4(color, 1.0f);

    IrradianceDebugImage[threadIndex] = float4(irradiance, 1.0f);
    NormalDebugImage[threadIndex] = float4(meanNormal, 1.0f);
    // DepthDebugImage[threadIndex] = meanDepth / 100.0f; // far plane is 100
    AlbedoDebugImage[threadIndex] = float4(meanAlbedo, 1.0f);

    IrradianceVarianceDebugImage[threadIndex] = float4(float3(irradianceVarianceS), 1.0f);
    SpecularVarianceDebugImage[threadIndex] = float4(float3(logSpecularVariance), 1.0f);
    NormalVarianceDebugImage[threadIndex] = float4(float3(normalVarianceS), 1.0f);
    //DepthVarianceDebugImage[threadIndex] = float4(float3(depthVariance), 1.0f);
    AlbedoVarianceDebugImage[threadIndex] = float4(float3(albedoVarianceS), 1.0f);
    #endif
}

groupshared float groupMaxDepth;
groupshared float3 groupNormals[33][33]; // 13KB
groupshared float groupDepths[33][33];
groupshared float3 groupAlbedos[33][33];

[numthreads(32, 32, 1)]
[shader("compute")]
void computeMain_createGradBuffers(uint2 threadIndex: SV_DispatchThreadID, uint2 threadGroupIndex : SV_GroupID ,uint2 groupThreadIndex: SV_GroupThreadID) {
    uint pixelIndex = threadIndex.y * pushConst.screenSize.x + threadIndex.x;
    groupNormals[groupThreadIndex.y + 1][groupThreadIndex.x + 1] = NormalBuffer[pixelIndex];
    groupDepths[groupThreadIndex.y + 1][groupThreadIndex.x + 1] = DepthBuffer[pixelIndex];
    groupAlbedos[groupThreadIndex.y + 1][groupThreadIndex.x + 1] = AlbedoBuffer[pixelIndex];
    if (groupThreadIndex.x == 0 && groupThreadIndex.y == 0) groupMaxDepth = OrderedIntToFloat(MaxDepthBuffer[0]);
    GroupMemoryBarrierWithGroupSync();

    uint groupThreadIndexLinear = groupThreadIndex.y * 32 + groupThreadIndex.x;
    uint warpLane = groupThreadIndexLinear & 31;
    uint warpIndex = groupThreadIndexLinear / 32;

    uint2 groupPixelStartIndex = threadGroupIndex * 32;
    if (warpIndex == 0) {
        float3 leftNormal = 0.0f;
        float leftDepth = 0.0f;
        float3 leftAlbedo = 0.0f;

        if (groupPixelStartIndex.x == 0) {
            leftNormal = groupNormals[warpLane + 1][1]; ;
            leftDepth = groupDepths[warpLane + 1][1];
            leftAlbedo = groupAlbedos[warpLane + 1][1];
        } else {
            uint leftPixelIndex = (groupPixelStartIndex.y + warpLane) * pushConst.screenSize.x + (groupPixelStartIndex.x - 1);

            leftNormal = NormalBuffer[leftPixelIndex];
            leftDepth = DepthBuffer[leftPixelIndex];
            leftAlbedo = AlbedoBuffer[leftPixelIndex];
        }

        groupNormals[warpLane + 1][0] = leftNormal;
        groupDepths[warpLane + 1][0] = leftDepth;
        groupAlbedos[warpLane + 1][0] = leftAlbedo;
    }
    if (warpIndex == 1) {
        float3 upNormal = groupNormals[1][warpLane + 1];
        float upDepth = groupDepths[1][warpLane + 1];
        float3 upAlbedo = groupAlbedos[1][warpLane + 1];

        if (groupPixelStartIndex.y == 0) {
            upNormal = groupNormals[1][warpLane + 1]; ;
            upDepth = groupDepths[1][warpLane + 1];
            upAlbedo = groupAlbedos[1][warpLane + 1];
        } else {
            uint upPixelIndex = (groupPixelStartIndex.y - 1) * pushConst.screenSize.x + groupPixelStartIndex.x + warpLane;

            upNormal = NormalBuffer[upPixelIndex];
            upDepth = DepthBuffer[upPixelIndex];
            upAlbedo = AlbedoBuffer[upPixelIndex];
        }

        groupNormals[0][warpLane + 1] = upNormal;
        groupDepths[0][warpLane + 1] = upDepth;
        groupAlbedos[0][warpLane + 1] = upAlbedo;
    }
    GroupMemoryBarrierWithGroupSync();

    if (threadIndex.x >= pushConst.screenSize.x || threadIndex.y >= pushConst.screenSize.y) return;

    // create dx
    float3 normal = groupNormals[groupThreadIndex.y + 1][groupThreadIndex.x + 1];
    float3 leftNormal = warpLane == 0 ? groupNormals[groupThreadIndex.y + 1][groupThreadIndex.x] : WaveReadLaneAt(normal, warpLane - 1);
    float3 normalDX = 0.0f; //normal - leftNormal;
    InputBuffer_gradNormal_diff[pixelIndex] = normalDX;
    InputBuffer_gradNormal_spec[pixelIndex] = normalDX;

    float maxDepth = max(groupMaxDepth, 1e-30f);
    float depth = groupDepths[groupThreadIndex.y + 1][groupThreadIndex.x + 1] / maxDepth;
    float depthVariance = InputBuffer_depthVariance_diff[pixelIndex];
    InputBuffer_depthVariance_diff[pixelIndex] = depthVariance / (maxDepth * maxDepth);
    InputBuffer_depthVariance_spec[pixelIndex] = depthVariance / (maxDepth * maxDepth);

    float leftDepth = (warpLane == 0 ? groupDepths[groupThreadIndex.y + 1][groupThreadIndex.x] : WaveReadLaneAt(depth, warpLane - 1)) / maxDepth;
    float depthDX = 0.0f;   //depth - leftDepth;
    InputBuffer_gradDepth_diff[pixelIndex] = depthDX;
    InputBuffer_gradDepth_spec[pixelIndex] = depthDX;

    float3 albedo = groupAlbedos[groupThreadIndex.y + 1][groupThreadIndex.x + 1];
    float3 leftAlbedo = warpLane == 0 ? groupAlbedos[groupThreadIndex.y + 1][groupThreadIndex.x] : WaveReadLaneAt(albedo, warpLane - 1);
    float3 albedoDX = 0.0f; //albedo - leftAlbedo;
    InputBuffer_gradAlbedo_diff[pixelIndex] = albedoDX;
    InputBuffer_gradAlbedo_spec[pixelIndex] = albedoDX;

    // create dy
    uint imageSize = pushConst.screenSize.x * pushConst.screenSize.y;

    float3 upNormal = groupNormals[groupThreadIndex.y][groupThreadIndex.x + 1];
    float3 normalDY = 0.0f; //normal - upNormal;
    InputBuffer_gradNormal_diff[pixelIndex + imageSize] = normalDY;
    InputBuffer_gradNormal_spec[pixelIndex + imageSize] = normalDY;

    float upDepth = groupDepths[groupThreadIndex.y][groupThreadIndex.x + 1] / maxDepth;
    float depthDY = 0.0f;   //depth - upDepth;
    InputBuffer_gradDepth_diff[pixelIndex + imageSize] = depthDY;
    InputBuffer_gradDepth_spec[pixelIndex + imageSize] = depthDY;

    float3 upAlbedo = groupAlbedos[groupThreadIndex.y][groupThreadIndex.x + 1];
    float3 albedoDY = 0.0f; //albedo - upAlbedo;
    InputBuffer_gradAlbedo_diff[pixelIndex + imageSize] = albedoDY;
    InputBuffer_gradAlbedo_spec[pixelIndex + imageSize] = albedoDY;

    // Due to the limitation of shared memory, we reuse them
    GroupMemoryBarrierWithGroupSync();

    groupNormals[groupThreadIndex.y + 1][groupThreadIndex.x + 1] = InputBuffer_irradiance[pixelIndex];
    groupAlbedos[groupThreadIndex.y + 1][groupThreadIndex.x + 1] = InputBuffer_spec[pixelIndex];
    GroupMemoryBarrierWithGroupSync();

    if (warpIndex == 0) {
        float3 leftDiff = 0.0f;
        float3 leftSpec = 0.0f;

        if (groupPixelStartIndex.x == 0) {
            leftDiff = groupNormals[warpLane + 1][1];
            leftSpec = groupAlbedos[warpLane + 1][1];
        } else {
            uint leftPixelIndex = (groupPixelStartIndex.y + warpLane) * pushConst.screenSize.x + (groupPixelStartIndex.x - 1);

            leftDiff = InputBuffer_irradiance[leftPixelIndex];
            leftSpec = InputBuffer_spec[leftPixelIndex];
        }

        groupNormals[warpLane + 1][0] = leftDiff;
        groupAlbedos[warpLane + 1][0] = leftSpec;
    }
    if (warpIndex == 1) {
        float3 upDiff = groupNormals[1][warpLane + 1];
        float3 upSpec = groupAlbedos[1][warpLane + 1];

        if (groupPixelStartIndex.y == 0) {
            upDiff = groupNormals[1][warpLane + 1];
            upSpec = groupAlbedos[1][warpLane + 1];
        } else {
            uint upPixelIndex = (groupPixelStartIndex.y - 1) * pushConst.screenSize.x + groupPixelStartIndex.x + warpLane;

            upDiff = InputBuffer_irradiance[upPixelIndex];
            upSpec = InputBuffer_spec[upPixelIndex];
        }

        groupNormals[0][warpLane + 1] = upDiff;
        groupAlbedos[0][warpLane + 1] = upSpec;
    }
    GroupMemoryBarrierWithGroupSync();

    // create dx
    float3 irradiance = groupNormals[groupThreadIndex.y + 1][groupThreadIndex.x + 1];
    float3 leftIrradiance = warpLane == 0 ? groupNormals[groupThreadIndex.y + 1][groupThreadIndex.x] : WaveReadLaneAt(irradiance, warpLane - 1);
    float3 irradianceDX = 0.0f; //irradiance - leftIrradiance;
    InputBuffer_gradIrradiance[pixelIndex] = irradianceDX;

    float3 specular = groupAlbedos[groupThreadIndex.y + 1][groupThreadIndex.x + 1];
    float3 leftSpecular = warpLane == 0 ? groupAlbedos[groupThreadIndex.y + 1][groupThreadIndex.x] : WaveReadLaneAt(specular, warpLane - 1);
    float3 specularDX = 0.0f;   //specular - leftSpecular;
    InputBuffer_gradSpec[pixelIndex] = specularDX;

    // create dy
    float3 upIrradiance = groupNormals[groupThreadIndex.y][groupThreadIndex.x + 1];
    float3 irradianceDY = 0.0f; //irradiance - upIrradiance;
    InputBuffer_gradIrradiance[pixelIndex + imageSize] = irradianceDY;

    float3 upSpecular = groupAlbedos[groupThreadIndex.y][groupThreadIndex.x + 1];
    float3 specularDY = 0.0f;   //specular - upSpecular;
    InputBuffer_gradSpec[pixelIndex + imageSize] = specularDY;

    #ifndef NDEBUG
    if (pushConst.frameIndex == 1 && threadIndex.x == 0 && threadIndex.y == 0) printf("maxDepth: %f\n", maxDepth);
    DepthDebugImage[threadIndex] = depth;
    DepthVarianceDebugImage[threadIndex] = float4(float3(depthVariance / (maxDepth * maxDepth)), 1.0f);
    #endif
}