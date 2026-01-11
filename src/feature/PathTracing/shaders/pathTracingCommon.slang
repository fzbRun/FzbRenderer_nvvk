#include "common/Shader/nvvk/Slang/pbr.h.slang"
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
//-------------------------------------------------直接光采样--------------------------------------------------
bool hitTest(float distance, float3 sampleDir, float3 hitPos, float3 hitNormal, int NEEShaderIndex, float time) {
    const uint rayFlags = 0;
    RayDesc tempRay;
    tempRay.Direction = normalize(sampleDir);
    tempRay.Origin = hitPos + 0.001f * tempRay.Direction; // isDielectric ? hitPos + 0.001f * temRay.Direction : offsetRay(hitPos, hitNormal);
    tempRay.TMin = 0.001f;
    tempRay.TMax = distance + 0.001f;
    NEEPayload tempPayload;
    tempPayload.distance = MISS_DISTANCE;
    // TraceRay(topLevelAS, rayFlags, 0xff, pushConst.NEEShaderIndex, 0, 0, temRay, tempPayload);
    TraceMotionRay(topLevelAS, rayFlags, 0xff, NEEShaderIndex, 0, 0, tempRay, time, tempPayload);

    if (tempPayload.distance + 0.002f < distance) return false;
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

void initNEE(inout CallablePayload payload, SceneInfo sceneInfo, int NEEShaderIndex, float time) {
    bool isDielectric = payload.material.type == uint(MaterialType::Deielectric) ||
                        payload.material.type == uint(MaterialType::RoughDeielectric);

    payload.lightNum = min(sceneInfo.numLights, NEE_MAX_SAMPLE_NUM);
    for (int lightIndex = 0; lightIndex < payload.lightNum; ++lightIndex) {
        payload.radiance_nee[lightIndex] = float3(0.0f);
        Light light = sceneInfo.lights[lightIndex];
        float3 flux = light.color * light.intensity;
        float3 sampleDir = float3(1.0f);
        if (light.type == uint(LightType::Point)) {
            sampleDir = light.pos - payload.hitPos;
            float distance = length(sampleDir);

            if (!hitTest(distance, sampleDir, payload.hitPos, payload.hitNormal, NEEShaderIndex, time)) continue; // 遮挡

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

                if (dot(-sampleDir, light.direction) <= 0.0f) continue;
                if (!isDielectric && dot(sampleDir, payload.hitNormal) <= 0.0f) continue;

                float distance = length(sampleDir);
                if (!hitTest(distance, sampleDir, payload.hitPos, payload.hitNormal, NEEShaderIndex, time)) continue; // 遮挡

                payload.radiance_nee[lightIndex] = flux / (M_TWO_PI * length(light.edge1) * length(light.edge2) * pdf);
            } else {
                float3 samplePos = light.pos + light.edge1 * randomNumber1 + light.edge2 * randomNumber2;
                sampleDir = samplePos - payload.hitPos;

                if (dot(-sampleDir, light.direction) <= 0.0f) continue;
                if (!isDielectric && dot(sampleDir, payload.hitNormal) <= 0.0f) continue;

                float distance = length(sampleDir);
                if (!(hitTest(distance, sampleDir, payload.hitPos, payload.hitNormal, NEEShaderIndex, time))) continue; // 遮挡

                // 分母有一个cos和面积，雅可比有一个cos，pdf有一个面积，抵消了
                distance = max(distance, 0.001f);
                payload.radiance_nee[lightIndex] = flux / (M_TWO_PI * distance * distance);
            }
        }
        payload.sampleDir_nee[lightIndex] = normalize(sampleDir);
    }
}

bool isPosInDirectionLight(float3 pos, SceneInfo sceneInfo) {
    int lightNum = min(sceneInfo.numLights, NEE_MAX_SAMPLE_NUM);
    const float EPSILON = 0.001f;
    for (int lightIndex = 0; lightIndex < lightNum; ++lightIndex) {
        Light light = sceneInfo.lights[lightIndex];
        if (light.type == uint(LightType::Point) || light.type == uint(LightType::Spot)) {
            if (dot(pos - light.pos, pos - light.pos) < EPSILON) return true;
        }
        else if (light.type == uint(LightType::Directional)) {} //directional light's judge in missShader
        else if (light.type == uint(LightType::Area)) {
            float3 w = pos - light.pos;
            float3 e1 = light.edge1;
            float3 e2 = light.edge2;

            float dist = dot(w, light.direction);
            if (abs(dist) > EPSILON) continue;

            float dot_e1e1 = dot(e1, e1);
            float dot_e1e2 = dot(e1, e2);
            float dot_e2e2 = dot(e2, e2);
            float dot_we1 = dot(w, e1);
            float dot_we2 = dot(w, e2);

            float det = dot_e1e1 * dot_e2e2 - dot_e1e2 * dot_e1e2;
            if (det < EPSILON) continue;

            float invDet = 1.0f / det;
            float u = (dot_e2e2 * dot_we1 - dot_e1e2 * dot_we2) * invDet;
            float v = (dot_e1e1 * dot_we2 - dot_e1e2 * dot_we1) * invDet;

            if ((u >= -EPSILON) & (u <= 1.0f + EPSILON) & (v >= -EPSILON) & (v <= 1.0f + EPSILON)) return true;
        }
    }
    return false;
}

float getPdf_directionLight(float3 sampleDir, Light light) {
    if (light.type == LightType::Area) {

    }
    else return 1;
}

//------------------------------------------Material BSDF---------------------------------------------------
float3 sampleIncidence_DiffuseMaterial(inout uint randomSeed, float3x3 TBN) {
    float randomNumber1 = rand(randomSeed);
    float randomNumber2 = rand(randomSeed);
    float cosTheta = sqrt(randomNumber1);
    float sinTheta = sqrt(1.0f - randomNumber1);
    float phi = M_TWO_PI * randomNumber2;
    float3 incidence;
    incidence.x = sinTheta * cos(phi);
    incidence.y = sinTheta * sin(phi);
    incidence.z = cosTheta;
    incidence = mul(incidence, TBN);
    incidence = normalize(incidence);
    return incidence;
}

void PopulateCallablePayload_DielectricMaterial(inout CallablePayload payload, inout float3 F) {
    payload.bsdf = 1.0f;

    float cosineOH = dot(payload.outgoing, payload.hitNormal);
    F = schlickFresnel(payload.material.albedo, float3(1.0F), cosineOH);
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
        if (!totalInternalReflection) payload.isExt = !payload.isExt;
    }
}

void PopulateCallablePayload_RoughConductorMaterial(inout CallablePayload payload, inout float3 outgoing_tangentSpace) {
    float randomNumber1 = rand(payload.randomSeed);
    float randomNumber2 = rand(payload.randomSeed);
    float roughness = payload.material.roughness;

    outgoing_tangentSpace = mul(payload.TBN, payload.outgoing);
    float3 h_tangentSpace = hvd_ggx_sample_vndf(outgoing_tangentSpace, float2(roughness), float2(randomNumber1, randomNumber2));

    float3 incidence_tangentSpace = reflect(-outgoing_tangentSpace, h_tangentSpace);
    // If h is particularly biased towards n, the incident vector may appear in the negative normal hemisphere
    payload.incidence = mul(incidence_tangentSpace, payload.TBN);

    float D_costhetaNH = hvd_ggx_eval(float2(1.0f / roughness), h_tangentSpace); // h需要在切线空间中
    float D = D_costhetaNH / h_tangentSpace.z;                                   // max(h.z, 1e-6f);
    float cosineOH = max(dot(outgoing_tangentSpace, h_tangentSpace), 0.001f);
    payload.pdf = max(D_costhetaNH / (4.0f * cosineOH), 0.001f);

    float G1_outgoing, G1_incidence;
    float G2 = ggx_smith_shadow_mask(
        G1_outgoing, 
        G1_incidence,
        outgoing_tangentSpace,
        incidence_tangentSpace,
                                     float2(roughness)
    );
    float3 F = schlickFresnel(payload.material.albedo, float3(1.0F), cosineOH);
    payload.bsdf = (D * G2 * F) / (4.0f * abs(outgoing_tangentSpace.z) * abs(incidence_tangentSpace.z) + 0.001f);
}

void PopulateCallablePayload_RoughDielectricMaterial(inout CallablePayload payload, inout float3 outgoing_tangentSpace, inout float eta) {
    float randomNumber1 = rand(payload.randomSeed);
    float randomNumber2 = rand(payload.randomSeed);
    float roughness = payload.material.roughness;

    outgoing_tangentSpace = mul(payload.TBN, payload.outgoing);
    float3 h_tangentSpace = hvd_ggx_sample_vndf(outgoing_tangentSpace, float2(roughness), float2(randomNumber1, randomNumber2));
    float cosineOH = max(dot(outgoing_tangentSpace, h_tangentSpace), 0.001f);

    float D_costhetaNH = hvd_ggx_eval(float2(1.0f / roughness), h_tangentSpace); // h需要在切线空间中
    float D = D_costhetaNH / max(h_tangentSpace.z, 0.001f);

    float3 F = schlickFresnel(payload.material.albedo, float3(1.0F), cosineOH);
    float F_oneChanel = 0.299 * F.x + 0.587 * F.y + 0.114 * F.z;

    eta = payload.material.eta.x;
    if (!payload.isExt) eta = 1.0f / eta;

    float randomNumber = rand(payload.randomSeed);
    float3 incidence_tangentSpace;
    bool refraction = false;
    if (randomNumber < F_oneChanel) incidence_tangentSpace = reflect(-outgoing_tangentSpace, h_tangentSpace);
    else {
        bool totalInternalReflection = false;
        incidence_tangentSpace = refract(outgoing_tangentSpace, h_tangentSpace, eta, cosineOH, totalInternalReflection);

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
        float weight = dot(incidence_tangentSpace, h_tangentSpace) / eta + cosineOH;
        payload.pdf = max((1.0f - F_oneChanel) * D_costhetaNH * cosineOH / (weight * weight), 0.001f);

        payload.bsdf = D * G2 * (1.0f - F) * abs(dot(incidence_tangentSpace, h_tangentSpace)) * cosineOH
                        / (weight * weight * abs(outgoing_tangentSpace.z) * abs(incidence_tangentSpace.z) + 0.001f);
    } else {
        // payload.pdf = F_oneChanel * D_costhetaNH / max(4.0f * cosineOH, 0.001f);
        payload.pdf = max(F_oneChanel * D_costhetaNH / (4.0f * cosineOH), 0.001f);

        payload.bsdf = D * G2 * F / (4.0f * abs(outgoing_tangentSpace.z) * abs(incidence_tangentSpace.z) + 0.001f);
    }
}