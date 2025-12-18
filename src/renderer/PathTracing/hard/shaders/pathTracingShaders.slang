/*
slang大括号的左右(包括开始大括号的下一行开始，以及结束大括号的上行结尾)不能有注释，或者说中文注释
*/

#include "nvshaders/constants.h.slang"
#include "nvshaders/random.h.slang"
#include "nvshaders/sky_functions.h.slang"
#include "nvvk/Slang/pbr.h.slang"
#include "nvshaders/ray_utils.h.slang"
#include "shaderStructType.h"

// clang-format off
//ScalarDataLayout
[[vk::push_constant]]                           ConstantBuffer<PushConstant, ScalarDataLayout> pushConst;
[[vk::binding(BindingPoints::eTextures, 0)]]    Sampler2D textures[];
[[vk::binding(BindingPoints::eTlas, 1)]]        RaytracingAccelerationStructure topLevelAS;
[[vk::binding(BindingPoints::eOutImage, 1)]]    RWTexture2D<float4> outImage;
// clang-format on

#define MISS_DEPTH 1000
#define MISS_DISTANCE 10000000.0f
#define NEE_MAX_SAMPLE_NUM 2

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

    bool isExt;
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

    float3 bsdf; // 新采样向量后的bsdf
    float pdf;   // 新采样向量的pdf

    float3 hitPos = float3(0.0f);
    float3 hitNormal = float3(0.0f);

    float3 outgoing;  // 出射方向
    float3 incidence; // 采样方向

    bool isExt;

    // NEE
    uint lightNum = NEE_MAX_SAMPLE_NUM;
    float3 sampleDir_nee[NEE_MAX_SAMPLE_NUM];
    float3 radiance_nee[NEE_MAX_SAMPLE_NUM];
    float3 radianceSum_nee;
};


struct NEEPayload {
    float distance;
};
struct Quadrilateral {
    float3 startPos;
    float3 normal;
    float3 edge1;
    float3 edge2;
};

//--------------------------------------------------获取顶点属性-------------------------------------------------
__generic<T : IFloat> T getAttribute(uint8_t* dataBufferAddress, BufferView bufferView, uint attributeIndex)
{
  if(bufferView.count > 0)
  {
    T* ptr = (T*)(dataBufferAddress + bufferView.offset + attributeIndex * bufferView.byteStride);
    return ptr[0];
  }

  return T(0);  // Error case
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
    float3 tangent, bitangent;
    orthonormalBasis(hitState.normal, tangent, bitangent);
    hitState.TBN = float3x3(tangent, bitangent, hitState.normal);

    return hitState;
}

//-------------------------------------------------获取纹理数据------------------------------------------------
void getTextueData(inout CallablePayload payload, float2 texCoords) {
    if (texCoords.x == 0.0f && texCoords.y == 0.0f) return;

    int normalMapIndex = payload.material.materialMapIndex.x;
    if (normalMapIndex >= 0) {
        float3 localNormal = textures[normalMapIndex].SampleLevel(texCoords, 0).xyz;
        payload.hitNormal = mul(localNormal, payload.TBN);
        float3 tangent, bitangent;
        orthonormalBasis(payload.hitNormal, tangent, bitangent);
        payload.TBN = float3x3(tangent, bitangent, payload.hitNormal);
    }

    int albedoMapIndex = payload.material.materialMapIndex.y;
    if (albedoMapIndex == int(AlbedoMapType::Checkerboard)) {
    }
    else if (albedoMapIndex >= 0) payload.material.albedo *= textures[albedoMapIndex].SampleLevel(texCoords, 0).xyz;

    int bsdfParamMapIndex = payload.material.materialMapIndex.z;
    if (bsdfParamMapIndex >= 0) {
        float4 bsdfParam = textures[bsdfParamMapIndex].SampleLevel(texCoords, 0);
        payload.material.eta = bsdfParam.xyz;
        payload.material.roughness = bsdfParam.w;
    }
}

//-------------------------------------------------直接光采样--------------------------------------------------
bool hitTest(float distance, float3 sampleDir, float3 hitPos, float3 hitNormal) {
    const uint rayFlags = 0;
    RayDesc temRay;
    temRay.Direction = normalize(sampleDir);
    temRay.Origin = hitPos + 0.001f * temRay.Direction;    //offsetRay(hitPos, hitNormal);
    temRay.TMin = 0.001f;
    temRay.TMax = distance + 0.001f;
    NEEPayload tempPayload;
    tempPayload.distance = MISS_DISTANCE;
    TraceRay(topLevelAS, rayFlags, 0xff, pushConst.NEEShaderIndex, 0, 0, temRay, tempPayload);

    if (tempPayload.distance + 0.001f < distance) return false;
    return true;
}
[shader("closesthit")]
void NEEClosestHitMain(inout NEEPayload payload) {
    payload.distance = RayTCurrent();
}
float3 sphereRectangleSample(Quadrilateral quad, float3 hitPos, float u, float v, inout float pdf) {
    float exl = length(quad.edge1);
    float eyl = length(quad.edge2);
    float3 axisX = quad.edge1 / exl;
    float3 axisY = quad.edge2 / eyl;
    float3 axisZ = quad.normal;

    float3 d = quad.startPos - hitPos;
    float z0 = dot(d, axisZ);
    if (z0 > 0) {
        axisZ *= -1.0f;
        z0 *= -1.0f;
    }
    float z0sq = z0 * z0;

    float x0 = dot(d, axisX);
    float y0 = dot(d, axisY);
    float x1 = x0 + exl;
    float y1 = y0 + eyl;
    float y0sq = y0 * y0;
    float y1sq = y1 * y1;

    float3 v00 = float3(x0, y0, z0);
    float3 v01 = float3(x0, y1, z0);
    float3 v10 = float3(x1, y0, z0);
    float3 v11 = float3(x1, y1, z0);

    float3 n0 = float3(0.0f, z0, -y0); // normalize(cross(v00, v10));
    n0.z /= sqrt(z0sq + y0sq);         // 后续只有z需要被用到，x，y会乘以0，不用管
    float3 n1 = float3(-z0, 0.0f, x1); // normalize(cross(v10, v11));
    n1.z /= sqrt(z0sq + x1 * x1);
    float3 n2 = float3(0.0f, -z0, y1); // normalize(cross(v11, v01));
    n2.z /= sqrt(z0sq + y1sq);
    float3 n3 = float3(z0, 0.0f, -x0); // normalize(cross(v01, v00));
    n3.z /= sqrt(z0sq + x0 * x0);

    float g0 = acos(-n0.z * n1.z); // acos(-dot(n0, n1));
    float g1 = acos(-n1.z * n2.z); // acos(-dot(n1, n2));
    float g2 = acos(-n2.z * n3.z); // acos(-dot(n2, n3));
    float g3 = acos(-n3.z * n0.z); // acos(-dot(n3, n0));

    float b0 = n0.z;
    float b1 = n2.z;
    float b0sq = b0 * b0;
    float k = 2 * M_PI - g2 - g3;
    float S = g0 + g1 - k;

    pdf = 1.0f / S;

    float au = u * S + k;
    float fu = (cos(au) * b0 - b1) / sin(au);
    float cu = 1.0f / sqrt(fu * fu + b0sq) * (fu > 0.0f ? 1.0f : -1.0f);
    cu = clamp(cu, -1.0f, 1.0f);
    float xu = -(cu * z0) / sqrt(1.0f - cu * cu);
    xu = clamp(xu, x0, x1);

    float dl = sqrt(xu * xu + z0sq);
    float h0 = y0 / sqrt(dl * dl + y0sq);
    float h1 = y1 / sqrt(dl * dl + y1sq);
    float hv = h0 + v * (h1 - h0);
    float hv2 = hv * hv;
    float yv = hv2 < 1 - 1e-6 ? (hv * dl) / sqrt(1.0f - hv2) : y1;

    return xu * axisX + yv * axisY + z0 * axisZ; // 未归一化
}

void initNEE(inout CallablePayload payload) {
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    payload.lightNum = min(sceneInfo.numLights, NEE_MAX_SAMPLE_NUM);
    for (int lightIndex = 0; lightIndex < payload.lightNum; ++lightIndex) {
        payload.radiance_nee[lightIndex] = float3(0.0f);
        Light light = sceneInfo.lights[lightIndex];
        float3 flux = light.color * light.intensity;
        float3 sampleDir = float3(1.0f);
        if (light.type == uint(LightType::Point)) {
            sampleDir = light.pos - payload.hitPos;
            float distance = length(sampleDir);
            if (!hitTest(distance, sampleDir, payload.hitPos, payload.hitNormal)) continue; // 遮挡

            payload.radiance_nee[lightIndex] = flux / (4.0f * M_PI * distance * distance);
        }
        else if (light.type == uint(LightType::Spot)) {}
        else if (light.type == uint(LightType::Directional)) {}
        else if (light.type == uint(LightType::Area)) {
            float randomNumber1 = rand(payload.randomSeed);
            float randomNumber2 = rand(payload.randomSeed);
 
            if (light.SphericalRectangleSample == 1) {
                Quadrilateral quad;
                quad.startPos = light.pos;
                quad.normal = light.direction;
                quad.edge1 = light.edge1;
                quad.edge2 = light.edge2;
                float pdf = 1.0f;
                sampleDir = sphereRectangleSample(quad, payload.hitPos, randomNumber1, randomNumber2, pdf);

                if (dot(-sampleDir, light.direction) <= 0.01f) continue;

                float distance = length(sampleDir);
                if (!hitTest(distance, sampleDir, payload.hitPos, payload.hitNormal)) continue; // 遮挡

                payload.radiance_nee[lightIndex] = flux / (M_TWO_PI * length(light.edge1) * length(light.edge2) * pdf);
            } else {
                float3 samplePos = light.pos + light.edge1 * randomNumber1 + light.edge2 * randomNumber2;
                sampleDir = samplePos - payload.hitPos;

                bool judgment1 = dot(-sampleDir, light.direction) <= 0.01f;
                bool judgment2 = dot(sampleDir, payload.hitNormal) <= 0.01f &&
                                 (payload.material.type != uint(MaterialType::Deielectric) ||
                                 payload.material.type != uint(MaterialType::RoughDeielectric));
                if (judgment1 || judgment2) continue;

                float distance = length(sampleDir);
                if (!(hitTest(distance, sampleDir, payload.hitPos, payload.hitNormal))) continue; // 遮挡

                // 分母有一个cos和面积，雅可比有一个cos，pdf有一个面积，抵消了
                distance = max(distance, 0.001f);
                payload.radiance_nee[lightIndex] = flux / (M_TWO_PI * distance * distance);
            }
        }
        payload.sampleDir_nee[lightIndex] = normalize(sampleDir);
    }
}

//-------------------------------------------------主Shader-------------------------------------------------
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

    const uint rayFlags = 0;    //RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
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
    payload.isExt = true;

    float3 accumulatedRadiance = float3(0, 0, 0);
    float RR = 0.8f;

    // Iterative reflection loop
    while (payload.bounceDepth < pushConst.maxDepth && (payload.bsdf_cosine.x + payload.bsdf_cosine.y + payload.bsdf_cosine.z > 0.0f))
    {
        TraceRay(topLevelAS, rayFlags, 0xff, 0, 0, 0, ray, payload);
        accumulatedRadiance += payload.radiance;

        float randomNumber = rand(payload.randomSeed);
        if (randomNumber >= RR) break;
        payload.pdf *= RR;

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
    Instance          instance  = sceneInfo.instances[instanceID];
    Mesh              mesh      = sceneInfo.meshes[instance.meshIndex];
    BSDFMaterial material  = sceneInfo.materials[instance.materialIndex];

    HitState hitState = getHitStateWithPositionFetch(barycentrics);

    //int3 indices = getTriangleIndices(mesh.dataBuffer, mesh.triMesh, triID);
    // float3 pos = getTriangleAttribute<float3>(mesh.dataBuffer, mesh.triMesh.positions, indices, barycentrics);
    //float3 normal = getTriangleAttribute<float3>(mesh.dataBuffer, mesh.triMesh.normals, indices, barycentrics);
    // hitState.pos = float3(mul(float4(pos, 1.0), ObjectToWorld4x3()));
    // hitState.normal = normalize(mul(WorldToObject4x3(), normal).xyz);
    // float3 tangent, bitangent;
    // orthonormalBasis(hitState.normal, tangent, bitangent);
    // hitState.TBN = float3x3(tangent, bitangent, hitState.normal);

    float cosineON = dot(hitState.normal, -payload.rayDirection);
    float3 emissive = material.emissive;
    if (material.type == uint(MaterialType::Deielectric) || material.type == uint(MaterialType::RoughDeielectric)) {
        if (cosineON < 0) hitState.normal = -hitState.normal;
        cosineON = abs(cosineON);
    } else emissive *= max(cosineON, 0.0f);

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
    
    callablePayload.radianceSum_nee = float3(0.0f);
    if (pushConst.NEEShaderIndex > 0) initNEE(callablePayload);

    CallShader<CallablePayload>(uint(material.type), callablePayload);
        
    payload.randomSeed = callablePayload.randomSeed;
    payload.radiance += (callablePayload.radianceSum_nee + emissive) * payload.bsdf_cosine / payload.pdf;
    payload.rayDirection = callablePayload.incidence;
    payload.rayOrigin = offsetRay(hitState.pos, hitState.normal); // hitState.pos + 0.001f * payload.rayDirection;
    payload.bsdf_cosine *= callablePayload.bsdf * cosineON;
    payload.pdf *= callablePayload.pdf;
    payload.isExt = callablePayload.isExt;

    payload.bounceDepth++;
}

[shader("miss")]
void rayMissMain(inout HitPayload payload)
{
    SceneInfo* sceneInfo = pushConst.sceneInfoAddress;
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
    payload.bsdf = payload.material.albedo * payload.pdf;

    if (pushConst.NEEShaderIndex > 0) {
        for (int i = 0; i < payload.lightNum; ++i) {
            if(payload.radiance_nee[i].x + payload.radiance_nee[i].y + payload.radiance_nee[i].z < 0.01f) continue;

            float cosineON = dot(payload.outgoing, payload.hitNormal);
            float3 bsdf = M_1_PI * cosineON * payload.material.albedo;
            payload.radianceSum_nee += payload.radiance_nee[i] * bsdf * abs(dot(payload.sampleDir_nee[i], payload.hitNormal));
        }
    }
}

[shader("callable")]
void conductorMaterialMain(inout CallablePayload payload) {
    payload.bsdf = 1.0f;
    payload.pdf = 1.0f;
    payload.incidence = reflect(-payload.outgoing, payload.hitNormal);

    if (pushConst.NEEShaderIndex > 0) {
        for (int i = 0; i < payload.lightNum; ++i) {
            if(payload.radiance_nee[i].x + payload.radiance_nee[i].y + payload.radiance_nee[i].z < 0.01f) continue;

            float3 sampleDir = payload.sampleDir_nee[i];
            float3 h = normalize(sampleDir + payload.outgoing);
            float3 bsdf = float3(1.0f);
            if (abs(dot(h, payload.hitNormal)) < 0.95f) bsdf = float3(0.0f);

            payload.radianceSum_nee += payload.radiance_nee[i] * bsdf * abs(dot(sampleDir, payload.hitNormal));
        }
    }
}

[shader("callable")]
void dielectricMaterialMain(inout CallablePayload payload) {
    payload.bsdf = 1.0f;

    float cosineOH = dot(payload.outgoing, payload.hitNormal);
    float3 F = schlickFresnel(payload.material.albedo, float3(1.0F), cosineOH);
    float F_oneChanel = 0.299 * F.x + 0.587 * F.y + 0.114 * F.z;

    float randomNumber = rand(payload.randomSeed);
    if (randomNumber < F_oneChanel) {
        payload.incidence = reflect(-payload.outgoing, payload.hitNormal);
        payload.pdf = F_oneChanel;
    }
    else {
        float eta = payload.material.eta.x;
        if (!payload.isExt) eta = 1.0f / eta;

        bool totalInternalReflection = false;
        payload.incidence = refract(payload.outgoing, payload.hitNormal, eta, cosineOH, totalInternalReflection);
        payload.pdf = 1.0f - F_oneChanel;
        if(!totalInternalReflection) payload.isExt = !payload.isExt; 
    }

    if (pushConst.NEEShaderIndex > 0) {
        for (int i = 0; i < payload.lightNum; ++i) {
            if(payload.radiance_nee[i].x + payload.radiance_nee[i].y + payload.radiance_nee[i].z < 0.01f) continue;

            float3 sampleDir = payload.sampleDir_nee[i];
            float3 bsdf = float3(1.0f);
            if (dot(sampleDir, payload.hitNormal) > 0.0f) {
                float3 h = normalize(sampleDir + payload.outgoing);
                if (abs(dot(h, payload.hitNormal)) < 0.99f) bsdf = F;
            } else {
                float eta = payload.material.eta.x;
                if (!payload.isExt) eta = 1.0f / eta; // 所以NEE要在找到incidence之前

                float3 h = normalize(eta * payload.outgoing + sampleDir);
                if (abs(dot(h, payload.hitNormal)) < 0.99f) bsdf = 1.0f - F;
            }

            payload.radianceSum_nee += payload.radiance_nee[i] * bsdf * abs(dot(sampleDir, payload.hitNormal));
        }
    }
}

[shader("callable")]
void roughConductorMaterialMain(inout CallablePayload payload) {
    float randomNumber1 = rand(payload.randomSeed);
    float randomNumber2 = rand(payload.randomSeed);
    float roughness = payload.material.roughness;
    float3x3 transposeTBN = transpose(payload.TBN);

    float3 outgoing_tangentSpace = mul(payload.outgoing, transposeTBN);
    float3 h = hvd_ggx_sample_vndf(outgoing_tangentSpace, float2(roughness), float2(randomNumber1, randomNumber2));

    float3 incidence_tangentSpace = reflect(-outgoing_tangentSpace, h);
    payload.incidence = mul(incidence_tangentSpace, payload.TBN);

    float D_costhetaNH = hvd_ggx_eval(float2(1.0f / roughness), h); // h需要在切线空间中
    float D = D_costhetaNH / h.z;   //max(h.z, 1e-6f);
    float cosineOH = abs(dot(outgoing_tangentSpace, h));
    payload.pdf = D_costhetaNH / max(4.0f * cosineOH, 0.00001f);

    float G1_outgoing, G1_incidence;
    float G2 = ggx_smith_shadow_mask(
        G1_outgoing, 
        G1_incidence,
        outgoing_tangentSpace,
        incidence_tangentSpace,
        float2(roughness)
    );
    float3 F = schlickFresnel(payload.material.albedo, float3(1.0F), cosineOH);
    payload.bsdf = (D * G2 * F) / (4.0f * abs(outgoing_tangentSpace.z) * abs(incidence_tangentSpace.z) + 1e-6f);

    if (pushConst.NEEShaderIndex > 0) {
        for (int i = 0; i < payload.lightNum; ++i) {
            if(payload.radiance_nee[i].x + payload.radiance_nee[i].y + payload.radiance_nee[i].z < 0.01f) continue;

            float3 sampleDir = payload.sampleDir_nee[i];

            h = normalize(sampleDir + payload.outgoing);
            h = mul(h, transposeTBN); // h需要在切线空间中
            //因为TBN可能不准，导致出现细微的小于0的z，导致出错。上面的h不用这样是因为hvd_ggx_sample_vndf必然生成z大于0的h
            if (h.z <= 0.001f) continue; 
            D_costhetaNH = hvd_ggx_eval(float2(1.0f / roughness), h);
            D = D_costhetaNH / h.z;

            incidence_tangentSpace = mul(sampleDir, transposeTBN);

            G2 = ggx_smith_shadow_mask(
                G1_outgoing,
                G1_incidence,
                outgoing_tangentSpace,
                incidence_tangentSpace,
                float2(roughness)
            );

            float3 bsdf = (D * G2 * F) / (4.0f * abs(outgoing_tangentSpace.z) * abs(incidence_tangentSpace.z) + 1e-6f);
            payload.radianceSum_nee += payload.radiance_nee[i] * bsdf * abs(dot(sampleDir, payload.hitNormal));
        }
    }
}

[shader("callable")]
void roughDielectricMaterialMain(inout CallablePayload payload) {
    float randomNumber1 = rand(payload.randomSeed);
    float randomNumber2 = rand(payload.randomSeed);
    float roughness = payload.material.roughness;
    float3x3 transposeTBN = transpose(payload.TBN);

    float3 outgoing_tangentSpace = mul(payload.outgoing, transposeTBN);
    float3 h = hvd_ggx_sample_vndf(outgoing_tangentSpace, float2(roughness), float2(randomNumber1, randomNumber2));
    float cosineOH = abs(dot(outgoing_tangentSpace, h));

    float D_costhetaNH = hvd_ggx_eval(float2(1.0f / roughness), h); // h需要在切线空间中
    float D = D_costhetaNH / max(h.z, 1e-6f);

    float3 F = schlickFresnel(payload.material.albedo, float3(1.0F), cosineOH);
    float F_oneChanel = 0.299 * F.x + 0.587 * F.y + 0.114 * F.z;

    float eta = payload.material.eta.x;
    if (!payload.isExt) eta = 1.0f / eta;

    float randomNumber = rand(payload.randomSeed);
    float3 incidence_tangentSpace;
    bool refraction = false;
    if (randomNumber < F_oneChanel) incidence_tangentSpace = reflect(-outgoing_tangentSpace, h);
    else {
        bool totalInternalReflection = false;
        incidence_tangentSpace = refract(outgoing_tangentSpace, h, eta, cosineOH, totalInternalReflection);

        if (!totalInternalReflection) {
            payload.isExt = !payload.isExt;
            refraction = true;
        }
    }
    payload.incidence = mul(incidence_tangentSpace, payload.TBN);

    float G1_outgoing, G1_incidence;
    float G2 = ggx_smith_shadow_mask(
        G1_outgoing, 
        G1_incidence,
        outgoing_tangentSpace,
        incidence_tangentSpace,
        float2(roughness)
    );

    if (refraction) {
        float weight = dot(incidence_tangentSpace, h) / eta + cosineOH;
        payload.pdf = (1.0f - F_oneChanel) * D_costhetaNH * cosineOH / (weight * weight);

        payload.bsdf = D * G2 * (1.0f - F) * abs(dot(incidence_tangentSpace, h)) * cosineOH
                        / (weight * weight * abs(outgoing_tangentSpace.z) * abs(incidence_tangentSpace.z) + 1e-6f);
    }else {
        payload.pdf = F_oneChanel * D_costhetaNH / max(4.0f * cosineOH, 0.00001f);

        payload.bsdf = D * G2 * F / (4.0f * abs(outgoing_tangentSpace.z) * abs(incidence_tangentSpace.z) + 1e-6f);
    }

    if (pushConst.NEEShaderIndex > 0) {
        for (int i = 0; i < payload.lightNum; ++i) {
            if(payload.radiance_nee[i].x + payload.radiance_nee[i].y + payload.radiance_nee[i].z < 0.01f) continue;

            float3 sampleDir = payload.sampleDir_nee[i];

            refraction = dot(sampleDir, payload.hitNormal) < 0.0f;
            if (refraction) h = normalize(sampleDir + payload.outgoing * eta);
            else h = normalize(sampleDir + payload.outgoing);

            //这里与roughConductor不同，h需要与法线同侧
            h = mul(h, transposeTBN);                      // h需要在切线空间中
            if(h.z < 0.0f) h = -h;
            D_costhetaNH = hvd_ggx_eval(float2(1.0f / roughness), h);
            D = D_costhetaNH / max(h.z, 1e-6f);

            h = faceforward(h, outgoing_tangentSpace, -h); // dot(I, nerf) < 0 return n;else return -n

            incidence_tangentSpace = mul(sampleDir, transposeTBN);

            G2 = ggx_smith_shadow_mask(
                G1_outgoing,
                G1_incidence,
                outgoing_tangentSpace,
                incidence_tangentSpace,
                float2(roughness)
            );

            float3 bsdf = float3(1.0f);
            if (refraction) {
                float weight = dot(incidence_tangentSpace, h) / eta + cosineOH;
                payload.bsdf = D * G2 * (1.0f - F) * abs(dot(incidence_tangentSpace, h)) * cosineOH
                        / (weight * weight * abs(outgoing_tangentSpace.z) * abs(incidence_tangentSpace.z) + 1e-6f);
            } else
                bsdf = (D * G2 * F) / (4.0f * abs(outgoing_tangentSpace.z) * abs(incidence_tangentSpace.z) + 1e-6f);

            payload.radianceSum_nee += payload.radiance_nee[i] * bsdf * abs(dot(sampleDir, payload.hitNormal));
        }
    }
}