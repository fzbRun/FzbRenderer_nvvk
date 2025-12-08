#include "nvshaders/constants.h.slang"
#include "nvshaders/random.h.slang"
#include "nvshaders/sky_functions.h.slang"
#include "nvvk/Slang/pbr.h.slang"
#include "nvvk/shaderio.h"
#include "shaderStructType.h"

// clang-format off
[[vk::push_constant]]                           ConstantBuffer<PushConstant> pushConst;
[[vk::binding(BindingPoints::eTextures, 0)]]    Sampler2D textures[];
[[vk::binding(BindingPoints::eTlas, 1)]]        RaytracingAccelerationStructure topLevelAS;
[[vk::binding(BindingPoints::eOutImage, 1)]]    RWTexture2D<float4> outImage;
// clang-format on

#define MISS_DEPTH 1000

// Raytracing Payload
struct HitPayload {
    uint randomSeed;

    float3 radiance;
    float3 bsdf_cosine; // 累积的bsdf
    float pdf;          // 累积的pdf

    float3 rayOrigin;
    float3 rayDirection;

    uint bounceDepth;
    uint meshIndex;
};
struct HitState {
    float3 pos;
    float3 normal;
    float3x3 TBN;
};
struct CallablePayload {
    BSDFMaterial material;
    float3x3 TBN;
    uint randomSeed;

    float3 radiance;
    float3 bsdf; // 新采样向量后的bsdf
    float pdf;   // 新采样向量的pdf

    float3 hitPos = float3(0.0f);
    float3 hitNormal = float3(0.0f);

    float3 outgoing;  // 出射方向
    float3 incidence; // 采样方向
};

__generic<T : IFloat> T getAttribute(uint8_t* dataBufferAddress, BufferView bufferView, uint attributeIndex)
{
  if(bufferView.count > 0)
  {
    T* ptr = (T*)(dataBufferAddress + bufferView.offset + attributeIndex * bufferView.byteStride);
    return ptr[0];
  }

  return T(1);  // Error case
}
int3 getTriangleIndices(uint8_t* dataBufferAddress, const TriangleMesh mesh, int primitiveID)
{
  if(mesh.indices.byteStride == sizeof(int16_t))
  {
    int16_t3* indices = (int16_t3*)(dataBufferAddress + mesh.indices.offset);
    return indices[primitiveID];
  }
  else if(mesh.indices.byteStride == sizeof(int32_t))
  {
    int3* indices = (int3*)(dataBufferAddress + mesh.indices.offset);
    return indices[primitiveID];
  }

  return int3(-1);  // Error case
}
__generic<T : IFloat> T getTriangleAttribute(uint8_t* dataBufferAddress, BufferView bufferView, uint3 attributeIndex, float3 barycentrics)
{
  T attr0 = getAttribute<T>(dataBufferAddress, bufferView, attributeIndex.x);
  T attr1 = getAttribute<T>(dataBufferAddress, bufferView, attributeIndex.y);
  T attr2 = getAttribute<T>(dataBufferAddress, bufferView, attributeIndex.z);
  return T(barycentrics.x) * attr0 + T(barycentrics.y) * attr1 + T(barycentrics.z) * attr2;
}
HitState getHitStateWithPositionFetch(const float3 barycentrics) {
    HitState hitState;

    const float3 pos0 = HitTriangleVertexPosition(0);
    const float3 pos1 = HitTriangleVertexPosition(1);
    const float3 pos2 = HitTriangleVertexPosition(2);
    const float3 localPosition = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
    hitState.pos = float3(mul(float4(localPosition, 1.0), ObjectToWorld4x3()));

    const float3 edge1 = pos1 - pos0;
    const float3 edge2 = pos2 - pos0;
    const float3 geoNormal = normalize(cross(edge1, edge2));

    hitState.normal = normalize(mul(WorldToObject4x3(), geoNormal).xyz);
    float3 tangent = normalize(edge1);
    float3 bitangent = cross(hitState.normal, tangent);
    tangent = cross(bitangent, hitState.normal);
    hitState.TBN = float3x3(tangent, bitangent, hitState.normal);

    return hitState;
}

[shader("raygeneration")]
void raygenMain()
{
    float2 launchID   = (float2)DispatchRaysIndex().xy;
    float2 launchSize = (float2)DispatchRaysDimensions().xy;

    SceneInfo* sceneInfo = pushConst.sceneInfoAddress;

    uint seed = xxhash32(uint3(uint2(launchID.xy), pushConst.frameIndex));
    float r1 = rand(seed);
    float r2 = rand(seed);
    float2 subpixel_jitter = pushConst.frameIndex == 0 ? float2(0.5f, 0.5f) : float2(r1, r2);

    const uint rayFlags = RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    const float2 pixelCenter = launchID + subpixel_jitter;
    const float2 clipCoords = launchID / launchSize * 2.0 - 1.0;
    const float4 viewCoords = mul(float4(clipCoords, 1.0, 1.0), sceneInfo.projInvMatrix);

    RayDesc ray;
    ray.Origin = mul(float4(0.0, 0.0, 0.0, 1.0), sceneInfo.viewInvMatrix).xyz;
    ray.Direction = mul(float4(normalize(viewCoords.xyz), 0.0), sceneInfo.viewInvMatrix).xyz;
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    // Initial state
    HitPayload payload;
    payload.randomSeed = seed;
    payload.radiance  = float3(0, 0, 0);
    payload.bsdf_cosine = float3(1.0f);
    payload.pdf = 1.0f;
    payload.bounceDepth = 0;
    payload.rayOrigin = ray.Origin;
    payload.rayDirection = ray.Direction;

    float3 accumulatedRadiance = float3(0, 0, 0);

    // Iterative reflection loop
    while (payload.bounceDepth < pushConst.maxDepth && (payload.bsdf_cosine.x + payload.bsdf_cosine.y + payload.bsdf_cosine.z > 0.0f))
    {
        TraceRay(topLevelAS, rayFlags, 0xff, 0, 0, 0, ray, payload);
        accumulatedRadiance += payload.radiance;

        ray.Origin = payload.rayOrigin;
        ray.Direction = payload.rayDirection;
    }

    float3 color = accumulatedRadiance;

    if (pushConst.frameIndex > 0)
    {
        float a = 1.0f / float(pushConst.frameIndex + 1);
        float3 old_color = outImage[int2(launchID)].xyz;
        outImage[int2(launchID)] = float4(lerp(old_color, accumulatedRadiance, a), 1.f);
    }
    else outImage[int2(launchID)] = float4(accumulatedRadiance, 1.f);
}

[shader("closesthit")]
void rayClosestHitMain(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);
    uint   instanceID   = InstanceIndex();
    uint   meshID       = InstanceID();
    uint   triID        = PrimitiveIndex();

    SceneInfo         sceneInfo = pushConst.sceneInfoAddress[0];
    GltfInstance          instance  = sceneInfo.instances[instanceID];
    GltfMesh              mesh      = sceneInfo.meshes[instance.meshIndex];
    BSDFMaterial material  = sceneInfo.materials[instance.materialIndex];

    HitState hitState = getHitStateWithPositionFetch(barycentrics);

    CallablePayload callablePayload;
    callablePayload.material = material;
    callablePayload.TBN = hitState.TBN;
    callablePayload.randomSeed = payload.randomSeed;
    callablePayload.radiance = float3(0.0f);
    callablePayload.bsdf = float3(0.0f);
    callablePayload.pdf = 0.0f;
    callablePayload.hitPos = hitState.pos;
    callablePayload.hitNormal = hitState.normal;
    callablePayload.outgoing = -payload.rayDirection;
    CallShader<CallablePayload>(uint(material.type), callablePayload);
    
    payload.randomSeed = callablePayload.randomSeed;
    payload.radiance += (callablePayload.radiance + material.emissive) * payload.bsdf_cosine / payload.pdf;
    payload.rayDirection = callablePayload.incidence;
    payload.rayOrigin = hitState.pos + 0.001f * payload.rayDirection;
    payload.bsdf_cosine *= callablePayload.bsdf * dot(hitState.normal, payload.rayDirection);
    payload.pdf *= callablePayload.pdf;

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
/*
callableShader会进行如下操作
1. 计算BSDF
2. 计算pdf
3. 计算下一次采样方向
4. NEE(如果需要)
*/
[shader("callable")]
void diffuseMaterialMain(inout CallablePayload payload) {
    float randomNumber1 = rand(payload.randomSeed);
    float randomNumber2 = rand(payload.randomSeed);
    float cosTheta = sqrt(randomNumber1);
    float sinTheta = sqrt(1.0f - randomNumber1);
    float phi = M_TWO_PI * randomNumber2;
    payload.incidence.x = sinTheta * cos(phi);
    payload.incidence.y = sinTheta * sin(phi);
    payload.incidence.z = cosTheta;
    float3 localDirection = payload.incidence;
    payload.incidence = mul(payload.incidence, payload.TBN);
    payload.incidence = normalize(payload.incidence);

    float cosine = dot(payload.incidence, payload.hitNormal);
    payload.pdf = M_1_PI * cosine;
    payload.bsdf = payload.material.albedo * M_1_PI;
}

[shader("callable")]
void conductorMaterialMain(inout CallablePayload payload) {
    payload.bsdf = 1.0f;
    payload.pdf = 1.0f;
    payload.incidence = reflect(-payload.outgoing, payload.hitNormal);
}