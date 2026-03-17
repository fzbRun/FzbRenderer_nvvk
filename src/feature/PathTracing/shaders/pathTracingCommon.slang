#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include "common/Shader/Slang/commonFunction.slang"
#include "nvshaders/constants.h.slang"
#include "nvshaders/random.h.slang"
#include "nvshaders/ray_utils.h.slang"
#include "nvshaders/sky_functions.h.slang"
#include "feature/PathTracing/shaderio.h"

[[vk::binding(StaticSetBindingPoints_PT::eTextures_PT, 0)]] Sampler2D textures[];
[[vk::binding(StaticSetBindingPoints_PT::eOutImage_PT, 0)]] RWTexture2D<float4> outImage;
[[vk::binding(DynamicSetBindingPoints_PT::eTlas_PT, 1)]] RaytracingAccelerationStructure topLevelAS;

#define MISS_DEPTH 1000
#define MISS_DISTANCE 10000000.0f
#define NEE_MAX_SAMPLE_NUM 1

// Raytracing Payload
struct HitPayload {
    uint randomSeed;

    float3 radiance;
    float3 bsdf_cosine; // 累积的bsdf
    float pdf;          // 累积的pdf

    float3 rayOrigin;
    float3 rayDirection;

    uint bounceDepth;

    bool isExt;
};
struct HitState {
    float3 pos;
    float3 normal;
    float3x3 TBN;
    float cosineON;
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

#ifdef NEE
    float3 sampleDir_nee[NEE_MAX_SAMPLE_NUM];
    float3 radiance_nee[NEE_MAX_SAMPLE_NUM];
    float radiancePdf_nee[NEE_MAX_SAMPLE_NUM];
    float3 radianceSum_nee;
    float3 rayOrigin;
#endif
};

struct NEEPayload {
    float distance;
};

//--------------------------------------------------获取顶点属性-------------------------------------------------
__generic<T : IFloat> T getAttribute(uint8_t *dataBufferAddress, BufferView bufferView, uint attributeIndex)
{
    if (bufferView.count > 0)
    {
        T *ptr = (T *)(dataBufferAddress + bufferView.offset + attributeIndex * bufferView.byteStride);
        return ptr[0];
    }

    return T(0); // Error case
}
int3 getTriangleIndices(uint8_t *dataBufferAddress, const TriangleMesh mesh, int primitiveID)
{
    if (mesh.indices.byteStride == sizeof(int16_t))
    {
        int16_t3 *indices = (int16_t3 *)(dataBufferAddress + mesh.indices.offset);
        return indices[primitiveID];
    }
    else if (mesh.indices.byteStride == sizeof(int32_t))
    {
        int3 *indices = (int3 *)(dataBufferAddress + mesh.indices.offset);
        return indices[primitiveID];
    }

    return int3(-1); // Error case
}
__generic<T : IFloat> T getTriangleAttribute(uint8_t *dataBufferAddress, BufferView bufferView, uint3 attributeIndex, float3 barycentrics)
{
    T attr0 = getAttribute<T>(dataBufferAddress, bufferView, attributeIndex.x);
    T attr1 = getAttribute<T>(dataBufferAddress, bufferView, attributeIndex.y);
    T attr2 = getAttribute<T>(dataBufferAddress, bufferView, attributeIndex.z);
    return T(barycentrics.x) * attr0 + T(barycentrics.y) * attr1 + T(barycentrics.z) * attr2;
}
HitState getHitStateWithPositionFetch(const float3 barycentrics, float3 rayDirection) {
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
    hitState.cosineON = dot(hitState.normal, -rayDirection);
    if (hitState.cosineON < 0) hitState.normal = -hitState.normal;

    float3 tangent, bitangent;
    orthonormalBasis(hitState.normal, tangent, bitangent);
    hitState.TBN = float3x3(tangent, bitangent, hitState.normal); // column matrix

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
        float2 uv = float2(texCoords.x * 20.0f, texCoords.y * 80.0f) * 2.0f;
        float2 checker = floor(uv) % 2.0f;
        float pattern = (checker.x + checker.y) % 2.0f;
        payload.material.albedo *= lerp(float3(0.8f), float3(0.2f), pattern);
    }
    else if (albedoMapIndex >= 0) payload.material.albedo *= textures[albedoMapIndex].SampleLevel(texCoords, 0).xyz;

    int bsdfParamMapIndex = payload.material.materialMapIndex.z;
    if (bsdfParamMapIndex >= 0) {
        float4 bsdfParam = textures[bsdfParamMapIndex].SampleLevel(texCoords, 0);
        payload.material.eta = bsdfParam.xyz;
        payload.material.roughness = bsdfParam.w;
    }
}
//---------------------------------------------closethit variable setting------------------------------------
void initCallableShader(inout CallablePayload callablePayload, 
    HitPayload payload, float3 barycentrics, SceneInfo sceneInfo) 
{
    uint instanceID = InstanceIndex();
    uint meshID = InstanceID();
    uint triID = PrimitiveIndex();

    Instance instance = sceneInfo.instances[instanceID];
    Mesh mesh = sceneInfo.meshes[instance.meshIndex];
    BSDFMaterial material = sceneInfo.materials[instance.materialIndex];

    HitState hitState = getHitStateWithPositionFetch(barycentrics, payload.rayDirection);

    // int3 indices = getTriangleIndices(mesh.dataBuffer, mesh.triMesh, triID);
    // float3 pos = getTriangleAttribute<float3>(mesh.dataBuffer, mesh.triMesh.positions, indices, barycentrics);
    // float3 normal = getTriangleAttribute<float3>(mesh.dataBuffer, mesh.triMesh.normals, indices, barycentrics);
    // hitState.pos = float3(mul(float4(pos, 1.0), ObjectToWorld4x3()));
    // hitState.normal = normalize(mul(WorldToObject4x3(), normal).xyz);
    // float3 tangent, bitangent;
    // orthonormalBasis(hitState.normal, tangent, bitangent);
    // hitState.TBN = float3x3(tangent, bitangent, hitState.normal);

    if (material.type != MaterialType::Dielectric && material.type != MaterialType::RoughDielectric)
        material.emissive *= max(hitState.cosineON, 0.0f);

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
}
//-------------------------------------------------直接光采样--------------------------------------------------
#ifdef NEE
bool hitTest(float distance, float3 sampleDir, float3 hitPos, float3 hitNormal, int NEEShaderIndex, float time) {
    const uint rayFlags = 0;
    RayDesc tempRay;
    tempRay.Direction = normalize(sampleDir);
    tempRay.Origin = hitPos + 0.001f * tempRay.Direction; // isDielectric ? hitPos + 0.001f * temRay.Direction : offsetRay(hitPos, hitNormal);
    tempRay.TMin = 0.001f;
    tempRay.TMax = distance + 0.001f;
    NEEPayload tempPayload;
    tempPayload.distance = MISS_DISTANCE;
#ifdef PathTracingMotionBlur
    TraceMotionRay(topLevelAS, rayFlags, 0xff, NEEShaderIndex, 0, 0, tempRay, time, tempPayload);
#else
    TraceRay(topLevelAS, rayFlags, 0xff, NEEShaderIndex, 0, 0, tempRay, tempPayload);
#endif

    if (tempPayload.distance + 0.002f < distance) return false;
    return true;
}
[shader("closesthit")]
void NEEClosestHitMain(inout NEEPayload payload) {
    payload.distance = RayTCurrent();
}


void initNEE(inout CallablePayload payload, SceneInfo sceneInfo, int NEEShaderIndex, float time) {
    bool isDielectric = payload.material.type == uint(MaterialType::Dielectric) ||
                        payload.material.type == uint(MaterialType::RoughDielectric);

    uint lightNum = min(sceneInfo.numLights, NEE_MAX_SAMPLE_NUM);
    for (int lightIndex = 0; lightIndex < lightNum; ++lightIndex) {
        payload.radiance_nee[lightIndex] = float3(0.0f);
        Light light = sceneInfo.lights[lightIndex];
        float3 flux = light.color * light.intensity;
        float3 sampleDir = float3(1.0f);
        if (light.type == uint(LightType::Point)) {
            sampleDir = light.pos - payload.hitPos;
            float distance = length(sampleDir);

            if (!hitTest(distance, sampleDir, payload.hitPos, payload.hitNormal, NEEShaderIndex, time)) continue; // 遮挡

            payload.radiance_nee[lightIndex] = flux;    // / (4.0f * M_PI * distance * distance);
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

                if (dot(-sampleDir, light.direction) <= 0.0f) continue;
                if (!isDielectric && dot(sampleDir, payload.hitNormal) <= 0.0f) continue;

                float distance = length(sampleDir);
                if (!hitTest(distance, sampleDir, payload.hitPos, payload.hitNormal, NEEShaderIndex, time)) continue; // 遮挡

                // L = dPhi / (dA * cosTheta * domega) * (bsdf * cosTheta2) / pdf
                payload.radiance_nee[lightIndex] = flux / pdf;  //flux / (M_TWO_PI * length(light.edge1) * length(light.edge2) * pdf);
                payload.radiancePdf_nee[lightIndex] = pdf;
            } else {
                float3 samplePos = light.pos + light.edge1 * randomNumber1 + light.edge2 * randomNumber2;
                sampleDir = samplePos - payload.hitPos;

                if (dot(-sampleDir, light.direction) <= 0.0f) continue;
                if (!isDielectric && dot(sampleDir, payload.hitNormal) <= 0.0f) continue;

                float distance = length(sampleDir);
                if (!(hitTest(distance, sampleDir, payload.hitPos, payload.hitNormal, NEEShaderIndex, time))) continue; // 遮挡

                // L = dPhi / (dA * cosTheta1 * domega) * (cosTheta1 / distance * distance) * (bsdf * cosTheta2) / pdf
                // 分母有一个cos和面积，雅可比有一个cos，pdf有一个面积，抵消了
                distance = max(distance, 0.001f);
                payload.radiance_nee[lightIndex] = flux * dot(normalize(sampleDir), light.direction) / (distance * distance * length(light.edge1) * length(light.edge2)); //flux / (M_TWO_PI * distance * distance);
                payload.radiancePdf_nee[lightIndex] = 1.0f / (sqrt(dot(light.edge1, light.edge1) * dot(light.edge2, light.edge2))); // 1 / area
            }
        }
        payload.sampleDir_nee[lightIndex] = normalize(sampleDir);
    }
}

#endif

bool isPosInDirectionLight(float3 pos, Light light) {
    const float EPSILON = 0.001f;
    if (light.type == uint(LightType::Point) || light.type == uint(LightType::Spot)) {
        if (dot(pos - light.pos, pos - light.pos) < EPSILON) return true;
    }
    else if (light.type == uint(LightType::Directional)) {} // directional light's judge in missShader
    else if (light.type == uint(LightType::Area)) {
        float3 w = pos - light.pos;
        float3 e1 = light.edge1;
        float3 e2 = light.edge2;

        float dist = dot(w, light.direction);
        if (abs(dist) > EPSILON) return false;

        float dot_e1e1 = dot(e1, e1);
        float dot_e1e2 = dot(e1, e2);
        float dot_e2e2 = dot(e2, e2);
        float dot_we1 = dot(w, e1);
        float dot_we2 = dot(w, e2);

        float det = dot_e1e1 * dot_e2e2 - dot_e1e2 * dot_e1e2;
        if (det < EPSILON) return false;

        float invDet = 1.0f / det;
        float u = (dot_e2e2 * dot_we1 - dot_e1e2 * dot_we2) * invDet;
        float v = (dot_e1e1 * dot_we2 - dot_e1e2 * dot_we1) * invDet;

        if ((u >= -EPSILON) & (u <= 1.0f + EPSILON) & (v >= -EPSILON) & (v <= 1.0f + EPSILON)) return true;
    }
    return false;
}

bool hitRectangle(inout float3 hitPos, float3 rayOrigin, float3 rayDirection, float3 rectanglePos, float3 E1, float3 E2) {
    float3 S = rayOrigin - rectanglePos;
    float3 S1 = cross(rayDirection, E2);
    float3 S2 = cross(S, E1);
    float3 tbb = 1 / dot(S1, E1) * float3(dot(S2, E2), dot(S1, S), dot(S2, rayDirection));
    if (tbb.x > 0 && (1.0f - tbb.y - tbb.z) > 0 && tbb.y > 0 && tbb.z > 0) {
        hitPos = tbb.x * rayDirection + rayOrigin;
        return true;
    }

    rectanglePos += E1 + E2;
    E1 = -E1;
    E2 = -E2;

    S = rayOrigin - rectanglePos;
    S1 = -S1;   //cross(rayDirection, E2);
    S2 = cross(S, E1);
    tbb = 1 / dot(S1, E1) * float3(dot(S2, E2), dot(S1, S), dot(S2, rayDirection));
    if (tbb.x > 0 && (1.0f - tbb.y - tbb.z) > 0 && tbb.y > 0 && tbb.z > 0) {
        hitPos = tbb.x * rayDirection + rayOrigin;
        return true;
    }

    return false;
}
float getPdf_directionLightSample(float3 pos, float3 sampleDir, Light light) {
    if (light.type == LightType::Area) {
        if (light.SphericalRectangleSample == 1) {
            float exl = length(light.edge1);
            float eyl = length(light.edge2);
            float3 axisX = light.edge1 / exl;
            float3 axisY = light.edge2 / eyl;
            float3 axisZ = light.direction;

            float3 d = light.pos - pos;
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

            return 1.0f / S;
        } else {
            float3 hitPos;
            if (!hitRectangle(hitPos, pos, sampleDir, light.pos, light.edge1, light.edge2)) return 0.0f;

            float area = sqrt(dot(light.edge1, light.edge1) * dot(light.edge2, light.edge2));
            float cosine = dot(light.direction, -sampleDir);
            float distanceSq = dot(pos - hitPos, pos - hitPos);
            return cosine / (area * distanceSq);
        }
    }
    else return 1;
}

//------------------------------------------Material BSDF---------------------------------------------------
float pdf_DiffuseMaterial(float3 incidence, float3 normal) {
    float cosine = dot(incidence, normal);
    return M_1_PI * cosine;
}
void PopulateCallablePayload_DiffuseMaterial(inout CallablePayload payload) {
    float randomNumber1 = rand(payload.randomSeed);
    float randomNumber2 = rand(payload.randomSeed);
    float cosTheta = sqrt(randomNumber1);
    float sinTheta = sqrt(1.0f - randomNumber1);
    float phi = M_TWO_PI * randomNumber2;
    payload.incidence.x = sinTheta * cos(phi);
    payload.incidence.y = sinTheta * sin(phi);
    payload.incidence.z = cosTheta;
    payload.incidence = mul(payload.incidence, payload.TBN);
    payload.incidence = normalize(payload.incidence);

    payload.pdf = M_1_PI * dot(payload.incidence, payload.hitNormal);
    payload.bsdf = payload.material.albedo * M_1_PI;
}

void PopulateCallablePayload_ConductorMaterial(inout CallablePayload payload) {
    payload.bsdf = payload.material.albedo;
    payload.pdf = 1.0f;
    payload.incidence = reflect(-payload.outgoing, payload.hitNormal);
}

float pdf_DielectricMaterial(float cosineOH, float3 F0, inout uint randomSeed, float eta) {
    float3 F = schlickFresnel(F0, float3(1.0F), cosineOH);
    float F_oneChanel = 0.299 * F.x + 0.587 * F.y + 0.114 * F.z;

    const float refraction = eta * eta * (1.0f - cosineOH * cosineOH);
    bool totalInternalReflection = (1.0f <= refraction);
    if (totalInternalReflection) return 1.0f;

    float randomNumber = rand(randomSeed);
    if (randomNumber < F_oneChanel) return F_oneChanel;
    else return 1.0f - F_oneChanel;
}
void PopulateCallablePayload_DielectricMaterial(inout CallablePayload payload) {
    payload.bsdf = 1.0f;

    float cosineOH = dot(payload.outgoing, payload.hitNormal);
    if (cosineOH == 0.0f) {
        payload.bsdf = float3(0.0f);
        payload.pdf = 1.0f;
        return;
    }
    float3 F = schlickFresnel(payload.material.albedo, float3(1.0F), cosineOH);
    float F_oneChanel = 0.299 * F.x + 0.587 * F.y + 0.114 * F.z;

    float eta = payload.material.eta.x;
    if (!payload.isExt) eta = 1.0f / eta;

    // total internal reflection
    const float refraction = eta * eta * (1.0f - cosineOH * cosineOH);
    bool totalInternalReflection = (1.0f <= refraction);
    if (totalInternalReflection) {
        payload.incidence = reflect(-payload.outgoing, payload.hitNormal);
        payload.pdf = 1;
        payload.bsdf = F;
    } else {
        float randomNumber = rand(payload.randomSeed);
        if (randomNumber < F_oneChanel) {
            payload.incidence = reflect(-payload.outgoing, payload.hitNormal);
            payload.pdf = F_oneChanel;
            payload.bsdf = F;
        }
        else {
            // refract(payload.outgoing, payload.hitNormal, eta, cosineOH, totalInternalReflection);
            payload.incidence = normalize((-payload.outgoing * eta + payload.hitNormal * (eta * cosineOH - sqrt(1.0f - refraction))));
            payload.pdf = 1.0f - F_oneChanel;
            payload.bsdf = (1.0f - F) * eta * eta; // radiance from incidence to outgoing
            payload.isExt = !payload.isExt;
        }
    }
}

float pdf_RoughConductorMaterial(float3 h_tangentSpace, float roughness, float cosineOH) {
    float D_costhetaNH = hvd_ggx_eval(float2(1.0f / roughness), h_tangentSpace); // h需要在切线空间中
    float D = D_costhetaNH / h_tangentSpace.z;                                   // max(h.z, 1e-6f);
    return max(D_costhetaNH / (4.0f * cosineOH), 0.001f);
}
void PopulateCallablePayload_RoughConductorMaterial(inout CallablePayload payload) {
    float randomNumber1 = rand(payload.randomSeed);
    float randomNumber2 = rand(payload.randomSeed);
    float roughness = payload.material.roughness;

    float3 outgoing_tangentSpace = mul(payload.TBN, payload.outgoing);
    float3 h_tangentSpace = hvd_ggx_sample_vndf(outgoing_tangentSpace, float2(roughness), float2(randomNumber1, randomNumber2));
    if (h_tangentSpace.z == 0.0f) {
        payload.bsdf = float3(0.0f);
        payload.pdf = 1.0f;
        return;
    }
    float cosineOH = dot(outgoing_tangentSpace, h_tangentSpace);
    if (cosineOH <= 0.0f) {
        payload.bsdf = float3(0.0f);
        payload.pdf = 1.0f;
        return;
    }

    float3 incidence_tangentSpace = reflect(-outgoing_tangentSpace, h_tangentSpace);
    if (incidence_tangentSpace.z <= 0.0f) {
        payload.bsdf = float3(0.0f);
        payload.pdf = 1.0f;
        return;
    }
    // If h is particularly biased towards n, the incident vector may appear in the negative normal hemisphere
    payload.incidence = mul(incidence_tangentSpace, payload.TBN);

    float D_costhetaNH = hvd_ggx_eval(float2(1.0f / roughness), h_tangentSpace); // h需要在切线空间中
    float D = D_costhetaNH / h_tangentSpace.z;                                   // max(h.z, 1e-6f);

    payload.pdf = D_costhetaNH / (4.0f * cosineOH);

    float G1_outgoing, G1_incidence;
    float G2 = ggx_smith_shadow_mask(
        G1_outgoing, 
        G1_incidence,
        outgoing_tangentSpace,
        incidence_tangentSpace,
                                     float2(roughness)
    );
    if (G2 <= 0.0f) {
        payload.bsdf = float3(0.0f);
        payload.pdf = 1.0f;
        return;
    }
    float3 F = schlickFresnel(payload.material.albedo, float3(1.0F), cosineOH);
    payload.bsdf = (D * G2 * F) / (4.0f * outgoing_tangentSpace.z * incidence_tangentSpace.z);
}

float pdf_RoughDielectricMaterial(
    float3 h_tangentSpace, float cosIH,
    float roughness, float cosineOH, float3 F0, inout uint randomSeed, float eta
) {
    float pdf = 1.0f;    

    float3 F = schlickFresnel(F0, float3(1.0F), cosineOH);
    float F_oneChanel = 0.299 * F.x + 0.587 * F.y + 0.114 * F.z;

    const float refraction = eta * eta * (1.0f - cosineOH * cosineOH);
    bool totalInternalReflection = (1.0f <= refraction);
    bool isRefraction = false;
    if (!totalInternalReflection) {
        float randomNumber = rand(randomSeed);
        if (randomNumber < F_oneChanel) pdf = F_oneChanel;
        else {
            pdf = 1.0f - F_oneChanel;
            isRefraction = true;
        }
    }

    float D_costhetaNH = hvd_ggx_eval(float2(1.0f / roughness), h_tangentSpace); // h需要在切线空间中
    float D = D_costhetaNH / h_tangentSpace.z;                                   // max(h.z, 1e-6f);

    if (isRefraction) {
        float weight = cosIH / eta + cosineOH;
        pdf *= max(D_costhetaNH * cosineOH / (weight * weight), 0.001f);
    } else pdf *= max(D_costhetaNH / (4.0f * cosineOH), 0.001f);

    return pdf;
}
void PopulateCallablePayload_RoughDielectricMaterial(inout CallablePayload payload) {
    float randomNumber1 = rand(payload.randomSeed);
    float randomNumber2 = rand(payload.randomSeed);
    float roughness = payload.material.roughness;

    float eta = payload.material.eta.x;
    if (!payload.isExt) eta = 1.0f / eta;

    float3 outgoing_tangentSpace = mul(payload.TBN, payload.outgoing);
    float3 h_tangentSpace = hvd_ggx_sample_vndf(outgoing_tangentSpace, float2(roughness), float2(randomNumber1, randomNumber2));
    if (h_tangentSpace.z == 0.0f) {
        payload.bsdf = float3(0.0f);
        payload.pdf = 1.0f;
        return;
    }
    float cosineOH = dot(outgoing_tangentSpace, h_tangentSpace);
    if (cosineOH <= 0.0f) {
        payload.bsdf = float3(0.0f);
        payload.pdf = 1.0f;
        return;
    }

    float3 F = schlickFresnel(payload.material.albedo, float3(1.0F), cosineOH);
    float F_oneChanel = 0.299 * F.x + 0.587 * F.y + 0.114 * F.z;

    float randomNumber = rand(payload.randomSeed);
    float3 incidence_tangentSpace;
    bool refraction = false;
    if (randomNumber < F_oneChanel) incidence_tangentSpace = reflect(-outgoing_tangentSpace, h_tangentSpace);
    else {
        bool totalInternalReflection = false;
        incidence_tangentSpace = refract(outgoing_tangentSpace, h_tangentSpace, eta, cosineOH, totalInternalReflection);

        refraction = !totalInternalReflection;
    }
    if (incidence_tangentSpace.z == 0.0f) {
        payload.bsdf = float3(0.0f);
        payload.pdf = 1.0f;
        return;
    }
    payload.incidence = mul(incidence_tangentSpace, payload.TBN);

    float D_costhetaNH = hvd_ggx_eval(float2(1.0f / roughness), h_tangentSpace); // h需要在切线空间中
    float D = D_costhetaNH / h_tangentSpace.z;

    float G1_outgoing, G1_incidence;
    float G2 = ggx_smith_shadow_mask(
        G1_outgoing, 
        G1_incidence,
        outgoing_tangentSpace,
        incidence_tangentSpace,
        float2(roughness)
    );
    if (G2 <= 0.0f) {
        payload.bsdf = float3(0.0f);
        payload.pdf = 1.0f;
        return;
    }

    if (refraction) {
        float weight = dot(incidence_tangentSpace, h_tangentSpace) / eta + cosineOH;
        payload.pdf = (1.0f - F_oneChanel) * D_costhetaNH * cosineOH / (weight * weight);

        payload.bsdf = D * G2 * (1.0f - F) * abs(dot(incidence_tangentSpace, h_tangentSpace)) * cosineOH
                        / abs(weight * weight * outgoing_tangentSpace.z * incidence_tangentSpace.z);
        payload.bsdf *= eta * eta; // radiance from incidence to outgoing

        payload.isExt = !payload.isExt;
    } else {
        payload.pdf = F_oneChanel * D_costhetaNH / (4.0f * cosineOH);
        payload.bsdf = D * G2 * F / abs(4.0f * outgoing_tangentSpace.z * incidence_tangentSpace.z);
    }
}

float3 getBSDF(BSDFMaterial material, float3 incidence, float3 normal, float3 outgoing, float3x3 TBN) {
    float cosineON = dot(outgoing, normal);
    float3 bsdf = float3(0.0f);
    if (material.type == MaterialType::Diffuse) bsdf = M_1_PI * material.albedo;
    else if (material.type == MaterialType::Conductor) {
        float3 h = normalize(incidence + outgoing);
        bsdf = cosineON > 0.0f && dot(h, normal) < 0.01f ? material.albedo : float3(0.0f);
    }
    else if (material.type == MaterialType::Dielectric) {
        float eta = material.eta.x;
        if (cosineON < 0) {
            cosineON = -cosineON;
            normal = -normal;
            eta = 1.0f / eta;
        }
        float3 F = schlickFresnel(material.albedo, float3(1.0F), cosineON); // cosineON = cosineOH
        float F_oneChanel = 0.299 * F.x + 0.587 * F.y + 0.114 * F.z;

        if (dot(incidence, normal) > 0.0f) {
            float3 h = normalize(incidence + outgoing);
            if (abs(dot(h, normal)) > 0.99f) bsdf = F;
        } else {
            float3 h = normalize(eta * outgoing + incidence);
            if (abs(dot(h, normal)) > 0.99f) bsdf = 1.0f - F;
        }
    } else if (material.type == MaterialType::RoughConductor) {
        if (cosineON <= 0.0f) return bsdf;
        float3 h = normalize(incidence + outgoing);
        float3 h_tangentSpace = mul(TBN, h);
        if (h_tangentSpace.z == 0.0f) return bsdf;

        float3 outgoing_tangentSpace = mul(TBN, outgoing);
        float cosineOH = dot(outgoing_tangentSpace, h_tangentSpace);
        if (cosineOH <= 0) return bsdf;

        float D_costhetaNH = hvd_ggx_eval(float2(1.0f / material.roughness), h_tangentSpace);
        float D = D_costhetaNH / h_tangentSpace.z;

        float3 F = schlickFresnel(material.albedo, float3(1.0F), cosineOH);

        float3 incidence_tangentSpace = mul(TBN, incidence);
        if (incidence_tangentSpace.z <= 0) return bsdf;

        float G1_outgoing, G1_incidence;
        float G2 = ggx_smith_shadow_mask(
                G1_outgoing,
                G1_incidence,
                outgoing_tangentSpace,
                incidence_tangentSpace,
                                         float2(material.roughness));
        if (G2 <= 0.0f) return bsdf;

        bsdf = (D * G2 * F) / (4.0f * abs(outgoing_tangentSpace.z) * abs(incidence_tangentSpace.z) + 0.001f);
    } else if (material.type == MaterialType::RoughDielectric) {
        float eta = material.eta.x;
        if (cosineON < 0) {
            cosineON = -cosineON;
            normal = -normal;
            eta = 1.0f / eta;
        }

        bool refraction = dot(incidence, normal) < 0.0f;
        float3 h;
        if (refraction) h = normalize(incidence + normal * eta);
        else h = normalize(incidence + outgoing);

        float3 h_tangentSpace = mul(TBN, h);
        if (h_tangentSpace.z <= 0.0f) return bsdf;

        float3 outgoing_tangentSpace = mul(TBN, outgoing);
        float cosineOH = dot(outgoing_tangentSpace, h_tangentSpace);
        if (cosineOH <= 0.0f) return bsdf;

        float D_costhetaNH = hvd_ggx_eval(float2(1.0f / material.roughness), h_tangentSpace);
        float D = D_costhetaNH / max(h_tangentSpace.z, 0.001f);

        float3 F = schlickFresnel(material.albedo, float3(1.0F), cosineOH);

        h_tangentSpace = faceforward(h_tangentSpace, outgoing_tangentSpace, -h_tangentSpace); // dot(I, nerf) < 0 return n;else return -n
        float3 incidence_tangentSpace = mul(TBN, incidence);

        float G1_outgoing, G1_incidence;
        float G2 = ggx_smith_shadow_mask(
                G1_outgoing,
                G1_incidence,
                outgoing_tangentSpace,
                incidence_tangentSpace,
                                         float2(material.roughness));
        if (G2 <= 0.0f) return bsdf;
        if (refraction) {
            float weight = dot(incidence_tangentSpace, h_tangentSpace) / eta + cosineOH;
            bsdf = D * G2 * (1.0f - F) * abs(dot(incidence_tangentSpace, h_tangentSpace)) * cosineOH
                        / (weight * weight * abs(outgoing_tangentSpace.z) * abs(incidence_tangentSpace.z) + 0.001f);
        } else
            bsdf = (D * G2 * F) / (4.0f * abs(outgoing_tangentSpace.z) * abs(incidence_tangentSpace.z) + 0.001f);
    }
    return bsdf;
}