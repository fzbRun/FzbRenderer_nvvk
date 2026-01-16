#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "renderer/SVOPathGuidingRenderer/hard/shaderio.h"

[[vk::push_constant]] ConstantBuffer<SVOPathGuidingPushConstant, ScalarDataLayout> pushConst;
#ifndef NDEBUG
[[vk::binding(StaticBindingPoints_SVOPG::eDepthImage_SVOPG, 0)]] RWTexture2D<float> depthImage; // VK_FORMAT_D32_SFLOAT
#endif
//-------------------------------------------------Ö÷Shader-------------------------------------------------
[shader("raygeneration")]
void raygenMain()
{
    float2 launchID = (float2)DispatchRaysIndex().xy;
    float2 launchSize = (float2)DispatchRaysDimensions().xy;

    SceneInfo *sceneInfo = pushConst.sceneInfoAddress;

    uint seed = xxhash32(uint3(uint2(launchID.xy), pushConst.frameIndex));
    float r1 = rand(seed);
    float r2 = rand(seed);
    float2 subpixel_jitter = pushConst.frameIndex == 0 ? float2(0.5f, 0.5f) : float2(r1, r2);

    const uint rayFlags = 0; // RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    const float2 pixelCenter = launchID + subpixel_jitter;
    const float2 clipCoords = launchID / launchSize * 2.0 - 1.0;
    const float4 viewCoords = mul(float4(clipCoords, 1.0, 1.0), sceneInfo.projInvMatrix);

    RayDesc ray;
    ray.Origin = sceneInfo.cameraPosition; // mul(float4(0.0, 0.0, 0.0, 1.0), sceneInfo.viewInvMatrix).xyz;
    ray.Direction = normalize(mul(float4(normalize(viewCoords.xyz), 0.0), sceneInfo.viewInvMatrix).xyz);
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    // Initial state
    HitPayload payload;
    payload.randomSeed = seed;
    payload.radiance = float3(0, 0, 0);
    payload.bsdf_cosine = float3(1.0f);
    payload.pdf = 1.0f;
    payload.bounceDepth = 0;
    payload.rayOrigin = ray.Origin;
    payload.rayDirection = ray.Direction;
    payload.isExt = true;

    float3 accumulatedRadiance = float3(0, 0, 0);
    float RR = 0.8f;

    // Iterative reflection loop
    while (payload.bounceDepth < pushConst.maxDepth && (payload.bsdf_cosine.x + payload.bsdf_cosine.y + payload.bsdf_cosine.z > 0.0f))
    {
#ifdef PathTracingMotionBlur
        TraceMotionRay(topLevelAS, rayFlags, 0xff, 0, 0, 0, ray, pushConst.time, payload);
#else
        TraceRay(topLevelAS, rayFlags, 0xff, 0, 0, 0, ray, payload);
#endif
        accumulatedRadiance += payload.radiance;

        float randomNumber = rand(payload.randomSeed);
        if (randomNumber >= RR) break;
        payload.pdf *= RR;

#ifndef NDEBUG
        if (payload.bounceDepth == 1) {
            float3 hitPos = payload.rayOrigin + 0.01f * ray.Direction;
            float4 clipPos = mul(float4(hitPos, 1.0f), sceneInfo.viewProjMatrix);
            float depth = clipPos.z / clipPos.w; // any hitPoint must in our cameraView
            depthImage[int2(launchID)] = depth;
        } else if (payload.bounceDepth == MISS_DEPTH) depthImage[int2(launchID)] = 1.0f;
#endif

        ray.Origin = payload.rayOrigin;
        ray.Direction = payload.rayDirection;
    }

    // When the maximum frame rate is reached, the original image will decay by ((maxframe-1)/maxframe)^k < 1, and the final image will turn black
    if (pushConst.frameIndex > 0)
    {
        float a = 1.0f / float(pushConst.frameIndex + 1);
        float3 old_color = outImage[int2(launchID)].xyz;
        outImage[int2(launchID)] = float4(lerp(old_color, accumulatedRadiance, a), 1.0f);
    }
    else outImage[int2(launchID)] = float4(accumulatedRadiance, 1.0f);
}

[shader("closesthit")]
void rayClosestHitMain(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);
    uint instanceID = InstanceIndex();
    uint meshID = InstanceID();
    uint triID = PrimitiveIndex();

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Instance instance = sceneInfo.instances[instanceID];
    Mesh mesh = sceneInfo.meshes[instance.meshIndex];
    BSDFMaterial material = sceneInfo.materials[instance.materialIndex];

    HitState hitState = getHitStateWithPositionFetch(barycentrics);

    float cosineON = dot(hitState.normal, -payload.rayDirection);
    if (cosineON < 0) hitState.normal = -hitState.normal;

    CallablePayload callablePayload;
    callablePayload.material = material;
    callablePayload.TBN = hitState.TBN;
    callablePayload.randomSeed = payload.randomSeed;
    callablePayload.bsdf = float3(0.0f);
    callablePayload.pdf = 1.0f;
    callablePayload.hitPos = hitState.pos;
    callablePayload.hitNormal = hitState.normal;
    callablePayload.outgoing = -payload.rayDirection;
    callablePayload.isExt = payload.isExt;

    if (material.materialMapIndex.x >= 0 || material.materialMapIndex.y >= 0 || material.materialMapIndex.z >= 0) {
        int3 indices = getTriangleIndices(mesh.dataBuffer, mesh.triMesh, triID);
        float2 texCoords = getTriangleAttribute<float2>(mesh.dataBuffer, mesh.triMesh.texCoords, indices, barycentrics);
        texCoords.y = 1.0f - texCoords.y;
        getTextueData(callablePayload, texCoords);
    }

    CallShader<CallablePayload>(uint(material.type), callablePayload);

    payload.randomSeed = callablePayload.randomSeed;
    payload.radiance = material.emissive * payload.bsdf_cosine / payload.pdf;
    payload.rayDirection = callablePayload.incidence;
    payload.rayOrigin = offsetRay(hitState.pos, hitState.normal); // hitState.pos + 0.001f * payload.rayDirection;
    payload.bsdf_cosine *= callablePayload.bsdf * abs(dot(hitState.normal, payload.rayDirection));
    payload.pdf *= callablePayload.pdf;
    payload.isExt = callablePayload.isExt;

    payload.bounceDepth++;
}

[shader("miss")]
void rayMissMain(inout HitPayload payload)
{
    SceneInfo *sceneInfo = pushConst.sceneInfoAddress;
    if (sceneInfo.useSky == 1)
    {
        float3 skyColor = evalSimpleSky(sceneInfo.skySimpleParam, WorldRayDirection());
        payload.radiance += skyColor * payload.bsdf_cosine / payload.pdf;
    }
    else payload.radiance += sceneInfo.backgroundColor * payload.bsdf_cosine / payload.pdf;

    payload.bounceDepth = MISS_DEPTH; // Stop
}

//-------------------------------------------------callableShader--------------------------------------------------
[shader("callable")]
void diffuseMaterialMain(inout CallablePayload payload) {
    PopulateCallablePayload_DiffuseMaterial(payload);
}

[shader("callable")]
void conductorMaterialMain(inout CallablePayload payload) {
    payload.bsdf = 1.0f;
    payload.pdf = 1.0f;
    payload.incidence = reflect(-payload.outgoing, payload.hitNormal);
}

[shader("callable")]
void dielectricMaterialMain(inout CallablePayload payload) {
    float eta = payload.material.eta.x;
    if (!payload.isExt) eta = 1.0f / eta;

    float3 F;
    PopulateCallablePayload_DielectricMaterial(payload, F, eta);
}

[shader("callable")]
void roughConductorMaterialMain(inout CallablePayload payload) {
    float3 outgoing_tangentSpace;
    PopulateCallablePayload_RoughConductorMaterial(payload, outgoing_tangentSpace);
}

[shader("callable")]
void roughDielectricMaterialMain(inout CallablePayload payload) {
    float3 outgoing_tangentSpace;
    float eta;
    PopulateCallablePayload_RoughDielectricMaterial(payload, outgoing_tangentSpace, eta);
}