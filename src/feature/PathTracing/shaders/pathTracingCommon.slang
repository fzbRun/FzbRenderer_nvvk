#ifndef FZBRENDERER_PATHTRACING_COMMON_SLANG
#define FZBRENDERER_PATHTRACING_COMMON_SLANG

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

    float3 directionLightDir;
    float3 directionLightRadiance;

#ifdef NEE
    float3 sampleDir_nee[NEE_MAX_SAMPLE_NUM];
    float3 radiance_nee[NEE_MAX_SAMPLE_NUM];
    float radiancePdf_nee[NEE_MAX_SAMPLE_NUM];
    float3 radianceSum_nee;
    float3 rayOrigin;
#endif
};

struct HitTestPayload {
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

    //hitState.normal = normalize(mul(WorldToObject4x3(), geoNormal).xyz);
    hitState.normal = normalize(mul(geoNormal, transpose(WorldToObject4x3())).xyz);
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
        float2 uv = float2(texCoords.x * 8.0f, texCoords.y * 32.0f) * 2.0f;
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
        material.emissive *= hitState.cosineON > 0.0f ? 1.0f : 0.0f;

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
bool hitTest(float distance, float3 sampleDir, float3 hitPos, float3 hitNormal, int HitTestShaderIndex, float time) {
    const uint rayFlags = 0;
    RayDesc tempRay;
    tempRay.Direction = normalize(sampleDir);
    tempRay.Origin = hitPos + 0.001f * tempRay.Direction; // isDielectric ? hitPos + 0.001f * temRay.Direction : offsetRay(hitPos, hitNormal);
    tempRay.TMin = 0.001f;
    tempRay.TMax = distance + 0.001f;
    HitTestPayload tempPayload;
    tempPayload.distance = MISS_DISTANCE;
    #ifdef PathTracingMotionBlur
    TraceMotionRay(topLevelAS, rayFlags, 0xff, HitTestShaderIndex, 0, 0, tempRay, time, tempPayload);
    #else
    TraceRay(topLevelAS, rayFlags, 0xff, HitTestShaderIndex, 0, 0, tempRay, tempPayload);
    #endif

    if (tempPayload.distance + 0.002f < distance) return false;
    return true;
}
[shader("closesthit")]
void HitTestClosestHitMain(inout HitTestPayload payload) {
    payload.distance = RayTCurrent();
}

void getDirectionalLight(inout CallablePayload payload, SceneInfo sceneInfo, int HitTestShaderIndex) {
    payload.directionLightRadiance = 0.0f;

    bool isDielectric = payload.material.type == uint(MaterialType::Dielectric) ||
                        payload.material.type == uint(MaterialType::RoughDielectric);
    uint lightNum = min(sceneInfo.numLights, NEE_MAX_SAMPLE_NUM);
    for (int lightIndex = 0; lightIndex < lightNum; ++lightIndex) {
        Light light = sceneInfo.lights[lightIndex];
        if (light.type == uint(LightType::Direction)) {
            float3 sampleDir = -light.direction;
            float cosine = dot(sampleDir, payload.hitNormal);
            if (!isDielectric && cosine <= 0.001f) continue;

            const uint rayFlags = 0;
            RayDesc tempRay;
            tempRay.Direction = normalize(sampleDir);
            tempRay.Origin = payload.hitPos + 0.001f * tempRay.Direction;
            tempRay.TMin = 0.001f;
            tempRay.TMax = INFINITE;
            HitTestPayload tempPayload;
            tempPayload.distance = MISS_DISTANCE;

            #ifdef PathTracingMotionBlur
            TraceMotionRay(topLevelAS, rayFlags, 0xff, HitTestShaderIndex, 0, 0, tempRay, time, tempPayload);
            #else
            TraceRay(topLevelAS, rayFlags, 0xff, HitTestShaderIndex, 0, 0, tempRay, tempPayload);
            #endif

            if (tempPayload.distance != MISS_DISTANCE) continue;
            payload.directionLightDir = light.direction;
            payload.directionLightRadiance = light.color * light.intensity * cosine;
        }
    }
}

#ifdef NEE
void initNEE(inout CallablePayload payload, SceneInfo sceneInfo, int HitTestShaderIndex, float time) {
    bool isDielectric = payload.material.type == uint(MaterialType::Dielectric) ||
                        payload.material.type == uint(MaterialType::RoughDielectric);

    uint lightNum = min(sceneInfo.numLights, NEE_MAX_SAMPLE_NUM);
    for (int lightIndex = 0; lightIndex < lightNum; ++lightIndex) {
        payload.radiance_nee[lightIndex] = float3(0.0f);
        Light light = sceneInfo.lights[lightIndex];
        float3 radiance = light.color * light.intensity;
        float3 sampleDir = float3(1.0f);
        if (light.type == uint(LightType::Area)) {
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

                if (isnan(pdf) || isinf(pdf)) continue;
                if (dot(-sampleDir, light.direction) <= 0.001f) continue;
                if (!isDielectric && dot(sampleDir, payload.hitNormal) <= 0.001f) continue;

                float distance = length(sampleDir);
                if (!hitTest(distance, sampleDir, payload.hitPos, payload.hitNormal, HitTestShaderIndex, time)) continue; // 遮挡

                // L = dPhi / (dA * cosTheta * domega) * (bsdf * cosTheta2) / pdf
                payload.radiance_nee[lightIndex] = radiance; // / pdf;  //flux / (M_TWO_PI * length(light.edge1) * length(light.edge2) * pdf);
                payload.radiancePdf_nee[lightIndex] = pdf;
            } else {
                float3 samplePos = light.pos + light.edge1 * randomNumber1 + light.edge2 * randomNumber2;
                sampleDir = samplePos - payload.hitPos;

                if (dot(-sampleDir, light.direction) <= 0.0f) continue;
                if (!isDielectric && dot(sampleDir, payload.hitNormal) <= 0.0f) continue;

                float distance = length(sampleDir);
                if (!(hitTest(distance, sampleDir, payload.hitPos, payload.hitNormal, HitTestShaderIndex, time))) continue; // 遮挡

                // L = dPhi / (dA * cosTheta1 * domega) * (cosTheta1 / distance * distance) * (bsdf * cosTheta2) / pdf
                // 分母有一个cos和面积，雅可比有一个cos，pdf有一个面积，抵消了
                distance = max(distance, 0.001f);
                payload.radiance_nee[lightIndex] = radiance * dot(normalize(sampleDir), light.direction) / (distance * distance * length(light.edge1) * length(light.edge2)); // flux / (M_TWO_PI * distance * distance);
                payload.radiancePdf_nee[lightIndex] = 1.0f / (sqrt(dot(light.edge1, light.edge1) * dot(light.edge2, light.edge2))); // 1 / area
            }
        }
        else if (light.type == uint(LightType::Direction)) {
            sampleDir = -light.direction;
            if (!isDielectric && dot(sampleDir, payload.hitNormal) <= 0.001f) continue;

            const uint rayFlags = 0;
            RayDesc tempRay;
            tempRay.Direction = normalize(sampleDir);
            tempRay.Origin = payload.hitPos + 0.001f * tempRay.Direction;
            tempRay.TMin = 0.001f;
            tempRay.TMax = INFINITE;
            HitTestPayload tempPayload;
            tempPayload.distance = MISS_DISTANCE;

            #ifdef PathTracingMotionBlur
            TraceMotionRay(topLevelAS, rayFlags, 0xff, HitTestShaderIndex, 0, 0, tempRay, time, tempPayload);
            #else
            TraceRay(topLevelAS, rayFlags, 0xff, HitTestShaderIndex, 0, 0, tempRay, tempPayload);
            #endif

            if (tempPayload.distance != MISS_DISTANCE) continue;
            payload.radiance_nee[lightIndex] = radiance;
            payload.radiancePdf_nee[lightIndex] = 1.0f;
        }
        else if (light.type == uint(LightType::Point)) {
            sampleDir = light.pos - payload.hitPos;
            float distance = length(sampleDir);

            if (!hitTest(distance, sampleDir, payload.hitPos, payload.hitNormal, HitTestShaderIndex, time)) continue; // 遮挡

            payload.radiance_nee[lightIndex] = radiance;    // / (4.0f * M_PI * distance * distance);
        }
        else if (light.type == uint(LightType::Spot)) {}
        payload.sampleDir_nee[lightIndex] = normalize(sampleDir);
    }
}
#endif

bool isPosInDirectionLight(float3 pos, Light light) {
    const float EPSILON = 0.001f;
    if (light.type == uint(LightType::Point) || light.type == uint(LightType::Spot)) {
        if (dot(pos - light.pos, pos - light.pos) < EPSILON) return true;
    }
    else if (light.type == uint(LightType::Direction)) {} // directional light's judge in missShader
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
float GTR2_NDF(float cosineOH, float roughness) {
    const float alpha = roughness * roughness;
    const float a2 = alpha * alpha;
    const float t = 1 + (a2 - 1) * cosineOH * cosineOH;
    return a2 / (M_PI * t * t);
}
float IsotropicGGX_Masking(float3 v_local, float roughness) {
    const float alpha = roughness * roughness;
    const float a2 = alpha * alpha;
    const float3 v2 = v_local * v_local;
    const float Lambda = (-1 + sqrt(1 + (v2.x * a2 + v2.y * a2) / v2.z)) / 2;
    return 1 / (1 + Lambda);
}
float FresnelDielectric(float n_dot_i, float n_dot_t, float eta) {
    const float rs = (n_dot_i - eta * n_dot_t) / (n_dot_i + eta * n_dot_t);
    const float rp = (eta * n_dot_i - n_dot_t) / (eta * n_dot_i + n_dot_t);
    const float F = (rs * rs + rp * rp) / 2;
    return F;
}
float FresnelDielectric(float cosineOH, float eta) {
    const float n_dot_t_sq = 1 - (1 - cosineOH * cosineOH) / (eta * eta);
    if (n_dot_t_sq < 0) {
        // total internal reflection
        return 1.f;
    }
    float n_dot_t = sqrt(n_dot_t_sq);
    return FresnelDielectric(abs(cosineOH), n_dot_t, eta);
}
float3 SampleVisibleNormals(float3 local_o, float alpha, float2 rnd_param) {
    // The incoming direction is in the "ellipsodial configuration" in Heitz's paper
    float negative = 1;
    if (local_o.z < 0) {
        // Ensure the input is on top of the surface.
        local_o = -local_o;
        negative = -1;
    }

    // Transform the incoming direction to the "hemisphere configuration".
    float3 hemi_dir_in = normalize(float3(alpha * local_o.x, alpha * local_o.y, local_o.z));

    // Parameterization of the projected area of a hemisphere.
    // First, sample a disk.
    float r = sqrt(rnd_param.x);
    float phi = 2 * M_PI * rnd_param.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    // Vertically scale the position of a sample to account for the projection.
    float s = (1 + hemi_dir_in.z) / 2;
    t2 = (1 - s) * sqrt(1 - t1 * t1) + s * t2;
    // Point in the disk space
    float3 disk_N = float3(t1, t2, sqrt(max(0, 1 - t1 * t1 - t2 * t2)));

    // Reprojection onto hemisphere -- we get our sampled normal in hemisphere space.
    //float3x3 hemi_frame = createFrame(hemi_dir_in);
    //float3 hemi_N = mul(disk_N, TBN);   //to_world(hemi_frame, disk_N);

    // Transforming the normal back to the ellipsoid configuration
    return negative * normalize(float3(alpha * disk_N.x, alpha * disk_N.y, max(0, disk_N.z)));
}

float3 getBSDF_Diffuse(float3 albedo, float3 incidence, float3 normal) {
    float cosineIN = dot(incidence, normal);
    if (cosineIN <= 0.0f) return float3(0.0f);
    return M_1_PI * albedo;
}
float3 getBSDF_Conductor(float3 albedo, float3 incidence, float3 normal, float3 outgoing) {
    float cosineIN = dot(incidence, normal);
    float cosineON = dot(outgoing, normal);
    if (cosineIN <= 0.0f || cosineON <= 0.0f) return float3(0.0f);

    float3 reflection = 2.0f * cosineIN * normal - incidence;
    if (dot(outgoing, reflection) < 1.0f - 1e-4f) return float3(0.0f);

    float3 F = albedo + (1.0f - albedo) * pow(1.0f - cosineIN, 5.0f);
    return F / cosineON;
}
float3 getBSDF_Dielectric(float eta, float3 albedo, float3 incidence, float3 normal, float3 outgoing) {
    float cosineIN = dot(incidence, normal);
    if (cosineIN < 0.0f) {
        normal = -normal;
        cosineIN = -cosineIN;
    }
    float cosineON = dot(outgoing, normal);
    bool isRefraction = cosineIN * cosineON < 0.0f;
    cosineON = abs(cosineON) + 1e-10f;

    float SnellFactor = eta * eta * (1.0f - cosineIN * cosineIN);
    bool totalReflection = SnellFactor >= 1.0f;

    float3 F = schlickFresnel(albedo, float3(1.0F), cosineIN);

    if (isRefraction) {
        if (totalReflection) return float3(0.0f);

        float3 refraction = normalize((-incidence * eta + normal * (eta * cosineIN - sqrt(1.0f - SnellFactor))));
        if (dot(outgoing, refraction) < 1.0f - 1e-4f) return float3(0.0f);

        return (1.0f - F) / (eta * eta) / cosineON;
    } else {
        float3 reflection = 2.0f * cosineIN * normal - incidence;
        if (dot(outgoing, reflection) < 1.0f - 1e-4f) return float3(0.0f);

        return (totalReflection? 1.0f : F) / cosineON;
    }
}
float3 getBSDF_RoughConductor(float3 albedo, float roughness, float3 incidence, float3 normal, float3 outgoing, float3x3 TBN) {
    float cosineIN = dot(incidence, normal);
    float cosineON = dot(outgoing, normal);
    if (cosineIN <= 0.0f || cosineON <= 0.0f) return float3(0.0f);

    float3 h = normalize(incidence + outgoing);
    float3 h_tangentSpace = mul(TBN, h);
    if (h_tangentSpace.z == 0.0f) return float3(0.0f);

    float3 incidence_tangentSpace = mul(TBN, incidence);
    float3 outgoing_tangentSpace = mul(TBN, outgoing);

    float cosineOH = dot(h_tangentSpace, outgoing_tangentSpace);

    float D_costhetaNH = hvd_ggx_eval(float2(1.0f / roughness), h_tangentSpace);
    float D = D_costhetaNH / h_tangentSpace.z;

    float3 F = schlickFresnel(albedo, float3(1.0F), cosineOH);

    float G1_outgoing, G1_incidence;
    float G2 = ggx_smith_shadow_mask(
                G1_outgoing,
                G1_incidence,
                outgoing_tangentSpace,
                incidence_tangentSpace,
                float2(roughness));
    if (G2 <= 0.0f) return float3(0.0f);

    return (D * G2 * F) / (4.0f * cosineIN * cosineON + 1e-10f);
}
float3 getBSDF_RoughDielectric(float eta, float3 albedo, float roughness, float3 incidence, float3 normal, float3 outgoing, float3x3 TBN) {
    float cosineIN = dot(incidence, normal);
    if (cosineIN < 0.0f) {
        normal = -normal;
        orthonormalBasis(normal, TBN[0], TBN[1]);
        TBN[2] = normal;
    }

    float3 incidence_tangentSpace = mul(TBN, incidence);
    float3 outgoing_tangentSpace = mul(TBN, outgoing);
    float cosineON = outgoing_tangentSpace.z;

    bool isRefraction = cosineIN * cosineON < 0.0f;
    float3 h;
    if (isRefraction) h = normalize(incidence * eta + outgoing);
    else h = normalize(incidence + outgoing);

    float cosineIH = dot(incidence, h);
    float cosineOH = dot(outgoing, h);

    float SnellFactor = eta * eta * (1.0f - cosineIN * cosineIN);
    bool totalReflection = SnellFactor >= 1.0f;
    if (isRefraction && totalReflection) return float3(0.0f);

    float3 h_tangentSpace = mul(TBN, h);
    if (h_tangentSpace.z <= 0.0f) return float3(0.0f);

    float D_costhetaNH = hvd_ggx_eval(float2(1.0f / roughness), h_tangentSpace);
    float D = D_costhetaNH / h_tangentSpace.z;

    float3 F = schlickFresnel(albedo, float3(1.0F), cosineIH);

    float G1_outgoing, G1_incidence;
    float G2 = ggx_smith_shadow_mask(
                G1_outgoing,
                G1_incidence,
                outgoing_tangentSpace,
                incidence_tangentSpace,
                                     float2(roughness));
    if (G2 <= 0.0f) return float3(0.0f);
    if (isRefraction) {
        float weight = cosineIH * eta + cosineOH;
        return D * G2 * (1.0f - F) * abs(cosineIH * cosineOH)
                        / (weight * weight * abs(cosineON * cosineIN) + 1e-10f) / (eta * eta);
    } else return (D * G2 * (totalReflection ? 1.0f : F)) / (4.0f * abs(cosineON * cosineIN) + 1e-10f);
}
float3 getBSDF_RoughtPlastic(float eta, float3 albedo_diffuse, float3 albedo_specular, float roughness, float3 incidence, float3 normal, float3 outgoing, float3x3 TBN) {
    float cosineIN = dot(incidence, normal);
    float cosineON = dot(outgoing, normal);
    if (cosineIN <= 0.0f || cosineON <= 0.0f) return float3(1.0f);

    float3 h = normalize(incidence + outgoing);
    float3 h_tangentSpace = mul(TBN, h);
    if (h_tangentSpace.z == 0.0f) return float3(0.0f);

    float3 incidence_tangentSpace = mul(TBN, incidence);
    float3 outgoing_tangentSpace = mul(TBN, outgoing);

    float cosineOH = dot(h_tangentSpace, outgoing_tangentSpace);
    float cosineIH = dot(h_tangentSpace, incidence_tangentSpace);

    float D = GTR2_NDF(h_tangentSpace.z, roughness);

    float3 F_i = FresnelDielectric(cosineIH, eta);

    float G1_outgoing = IsotropicGGX_Masking(outgoing_tangentSpace, roughness);
    float G1_incidence = IsotropicGGX_Masking(incidence_tangentSpace, roughness);
    float G2 = G1_outgoing * G1_incidence;
    if (G2 <= 0.0f) return float3(0.0f);

    float3 specular_contib = albedo_specular * (D * G2 * F_i) / (4.0f * cosineIN * cosineON + 1e-10f);

    float3 F_o = FresnelDielectric(cosineOH, eta);
    float3 diffuse_contrib = albedo_diffuse * (1.0f - F_o) * (1.0f - F_i) / M_PI;

    return (specular_contib + diffuse_contrib) * cosineON;
}
float3 getBSDF(BSDFMaterial material, float3 incidence, float3 normal, float3 outgoing, float3x3 TBN, bool isExt) {
    float3 bsdf = float3(0.0f);
    if (material.type == MaterialType::Diffuse) bsdf = getBSDF_Diffuse(material.albedo, incidence, normal);
    else if (material.type == MaterialType::RoughConductor)
        bsdf = getBSDF_RoughConductor(material.albedo, material.roughness, incidence, normal, outgoing, TBN);
    else if (material.type == MaterialType::RoughPlastic)
        bsdf = getBSDF_RoughtPlastic(material.eta.x, material.albedo, material.albedo_specular, material.roughness, incidence, normal, outgoing, TBN);
    else if (material.type == MaterialType::Conductor) bsdf = getBSDF_Conductor(material.albedo, incidence, normal, outgoing);
    else if (material.type == MaterialType::Dielectric) {
        float eta = material.eta.x;
        if (!isExt) eta = 1.0f / eta;
        bsdf = getBSDF_Dielectric(eta, material.albedo, incidence, normal, outgoing);
    }
    else if (material.type == MaterialType::RoughDielectric) {
        float eta = material.eta.x;
        if (!isExt) eta = 1.0f / eta;
        bsdf = getBSDF_RoughDielectric(eta, material.albedo, material.roughness, incidence, normal, outgoing, TBN);
    }
    return bsdf;
}

float getPdf_Diffuse(float3 incidence, float3 normal) {
    float cosineIN = dot(incidence, normal);
    return M_1_PI * max(cosineIN, 1e-10f);
}
float getPdf_Conductor(float3 incidence, float3 normal, float3 outgoing) {
    float cosineIN = dot(incidence, normal);
    float cosineON = dot(outgoing, normal);
    if (cosineIN <= 0.0f || cosineON <= 0.0f) return 0.0f;

    float3 reflection = 2.0f * cosineIN * normal - incidence;
    if (dot(outgoing, reflection) < 1.0f - 1e-4f) return 0.0f;

    return 1.0f;
}
float getPdf_Dielectric(float eta, float3 albedo, float3 incidence, float3 normal, float3 outgoing) {
    float cosineIN = dot(incidence, normal);
    if (cosineIN < 0.0f) {
        normal = -normal;
        cosineIN = -cosineIN;
    }
    float cosintON = dot(outgoing, normal);
    bool isRefraction = cosineIN * cosintON < 0.0f;

    float SnellFactor = eta * eta * (1.0f - cosineIN * cosineIN);
    bool totalReflection = SnellFactor >= 1.0f;

    float3 F = schlickFresnel(albedo, float3(1.0F), cosineIN);
    float F_oneChanel = 0.299 * F.x + 0.587 * F.y + 0.114 * F.z;

    if (isRefraction) {
        if (totalReflection) return 0.0f;

        float3 refraction = normalize((-incidence * eta + normal * (eta * cosineIN - sqrt(1.0f - SnellFactor))));
        if (dot(outgoing, refraction) < 1.0f - 1e-4f) return 0.0f;

        return 1.0f - F_oneChanel;
    } else {
        float3 reflection = 2.0f * cosineIN * normal - incidence;
        if (dot(outgoing, reflection) < 1.0f - 1e-4f) return 0.0f;

        return totalReflection ? 1.0f : F_oneChanel;
    }
}
float getPdf_RoughConductor(float roughness, float3 incidence, float3 normal, float3 outgoing, float3x3 TBN) {
    float cosineIN = dot(incidence, normal);
    float cosineON = dot(outgoing, normal);
    if (cosineIN <= 0.0f || cosineON <= 0.0f) return 0.0f;

    float3 h = normalize(incidence + outgoing);
    float3 h_tangentSpace = mul(TBN, h);
    if (h_tangentSpace.z == 0.0f) return 0.0f;

    float cosineOH = dot(h, outgoing);

    float D_costhetaNH = hvd_ggx_eval(float2(1.0f / roughness), h_tangentSpace); // h需要在切线空间中
    float D = D_costhetaNH / h_tangentSpace.z;                                   // max(h.z, 1e-6f);

    return D_costhetaNH / (abs(4.0f * cosineOH) + 1e-10f);
}
float getPdf_RoughDielectric(float eta, float3 albedo, float roughness, float3 incidence, float3 normal, float3 outgoing, float3x3 TBN) {
    float cosineIN = dot(incidence, normal);
    if (cosineIN < 0.0f) {
        normal = -normal;
        orthonormalBasis(normal, TBN[0], TBN[1]);
        TBN[2] = normal;
    }

    float3 incidence_tangentSpace = mul(TBN, incidence);
    float3 outgoing_tangentSpace = mul(TBN, outgoing);
    float cosineON = outgoing_tangentSpace.z;

    bool isRefraction = cosineIN * cosineON < 0.0f;
    float3 h;
    if (isRefraction) h = normalize(incidence * eta + outgoing);
    else h = normalize(incidence + outgoing);

    float cosineIH = dot(incidence, h);
    float cosineOH = dot(outgoing, h);

    float SnellFactor = eta * eta * (1.0f - cosineIN * cosineIN);
    bool totalReflection = SnellFactor >= 1.0f;
    if (isRefraction && totalReflection) return 0.0f;
    else if (!isRefraction && totalReflection) return 1.0f;

    float3 h_tangentSpace = mul(TBN, h);
    if (h_tangentSpace.z <= 0.0f) return 0.0f;

    float D_costhetaNH = hvd_ggx_eval(float2(1.0f / roughness), h_tangentSpace);
    float D = D_costhetaNH / h_tangentSpace.z;

    float3 F = schlickFresnel(albedo, float3(1.0F), cosineIH);
    float F_oneChanel = 0.299 * F.x + 0.587 * F.y + 0.114 * F.z;

    if (isRefraction) {
        float weight = cosineIH * eta + cosineOH;
        return (1.0f - F_oneChanel) * D_costhetaNH * cosineIH / (weight * weight + 1e-10f);
    } else return F_oneChanel * D_costhetaNH / (abs(4.0f * cosineOH) + 1e-10f);
}
float getPdf_RoughPlastic(float3 albedo_diffuse, float3 albedo_specular, float roughness, float3 incidence, float3 normal, float3 outgoing, float3x3 TBN) {
    float cosineIN = dot(incidence, normal);
    float cosineON = dot(outgoing, normal);
    if (cosineIN <= 0.0f || cosineON <= 0.0f) return 0.0f;

    float3 h = normalize(incidence + outgoing);
    float3 h_tangentSpace = mul(TBN, h);
    if (h_tangentSpace.z == 0.0f) return 0.0f;

    float lS = 0.299 * albedo_specular.x + 0.587 * albedo_specular.y + 0.114 * albedo_specular.z;
    float lR = 0.299 * albedo_diffuse.x + 0.587 * albedo_diffuse.y + 0.114 * albedo_diffuse.z;
    float specular_prob = lS / (lS + lR);
    float diffuse_prob = 1.0f - specular_prob;

    float3 outgoing_tangentSpace = mul(TBN, outgoing);
    float G = IsotropicGGX_Masking(outgoing_tangentSpace, roughness);
    float D = GTR2_NDF(h_tangentSpace.z, roughness);

    specular_prob *= (G * D) / (4.0f * cosineON);
    diffuse_prob *= cosineIN / M_PI;
    return specular_prob + diffuse_prob;
}
float getPdf(BSDFMaterial material, float3 incidence, float3 normal, float3 outgoing, float3x3 TBN, bool isExt) {
    float pdf = 1.0f;
    if (material.type == MaterialType::Diffuse) pdf = getPdf_Diffuse(incidence, normal);
    else if (material.type == MaterialType::RoughConductor)
        pdf = getPdf_RoughConductor(material.roughness, incidence, normal, outgoing, TBN);
    else if (material.type == MaterialType::RoughPlastic)
        pdf = getPdf_RoughPlastic(material.albedo, material.albedo_specular, material.roughness, incidence, normal, outgoing, TBN);
    else if (material.type == MaterialType::Conductor) pdf = getPdf_Conductor(incidence, normal, outgoing);
    else if (material.type == MaterialType::Dielectric) {
        float eta = material.eta.x;
        if (!isExt) eta = 1.0f / eta;
        pdf = getPdf_Dielectric(eta, material.albedo, incidence, normal, outgoing);
    }
    else if (material.type == MaterialType::RoughDielectric) {
        float eta = material.eta.x;
        if (!isExt) eta = 1.0f / eta;
        pdf = getPdf_RoughDielectric(eta, material.albedo, material.roughness, incidence, normal, outgoing, TBN);
    }
    return pdf;
}

void PopulateCallablePayload_DiffuseMaterial(inout CallablePayload payload) {
    float cosineON = dot(payload.outgoing, payload.hitNormal);
    if (cosineON <= 0.0f) {
        payload.bsdf = float3(0.0f);
        payload.pdf = 1.0f;
        return;
    }

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

    payload.directionLightRadiance *= payload.material.albedo * M_1_PI;
}
void PopulateCallablePayload_ConductorMaterial(inout CallablePayload payload) {
    float cosineON = dot(payload.outgoing, payload.hitNormal);
    if (cosineON <= 0.0f) {
        payload.bsdf = float3(0.0f);
        payload.pdf = 1.0f;
        return;
    }

    payload.incidence = reflect(-payload.outgoing, payload.hitNormal);
    float3 F = payload.material.albedo + (1.0f - payload.material.albedo) * pow(1.0f - cosineON, 5.0f);
    payload.bsdf = F / cosineON;
    payload.pdf = 1.0f;

    if (abs(dot(-payload.directionLightDir, payload.incidence)) > 0.001f)
        payload.directionLightRadiance = 0.0f;
}
void PopulateCallablePayload_DielectricMaterial(inout CallablePayload payload) {
    float cosineON = dot(payload.outgoing, payload.hitNormal);
    if (cosineON == 0.0f) {
        payload.bsdf = float3(0.0f);
        payload.pdf = 1.0f;
        return;
    }

    float3 F = schlickFresnel(payload.material.albedo, float3(1.0F), cosineON);
    float F_oneChanel = 0.299 * F.x + 0.587 * F.y + 0.114 * F.z;

    float eta = payload.material.eta.x;
    if (!payload.isExt) eta = 1.0f / eta;

    // total internal reflection
    bool totalInternalReflection;
    float3 refraction = refract(payload.outgoing, payload.hitNormal, eta, cosineON, totalInternalReflection);
    float3 reflection = reflect(-payload.outgoing, payload.hitNormal);
    if (totalInternalReflection) {
        payload.incidence = reflection;
        payload.pdf = 1.0f;
        payload.bsdf = float3(1.0f) / cosineON;
    } else {
        float randomNumber = rand(payload.randomSeed);
        if (randomNumber < F_oneChanel) {
            payload.incidence = reflection;
            payload.pdf = F_oneChanel;
            payload.bsdf = F / cosineON;
        }
        else {
            // refract(payload.outgoing, payload.hitNormal, eta, cosineOH, totalInternalReflection);
            payload.incidence = refraction;
            payload.pdf = 1.0f - F_oneChanel;
            payload.bsdf = (1.0f - F) * (eta * eta) / abs(dot(refraction, payload.hitNormal)); // radiance from incidence to outgoing
            payload.isExt = !payload.isExt;
        }
    }

    if (any(payload.directionLightRadiance > 0.001f)) {
        if (abs(dot(-payload.directionLightDir, refraction)) <= 0.001f)
            payload.directionLightRadiance *= (1.0f - F) * (eta * eta) / abs(dot(refraction, payload.hitNormal));
        else if (abs(dot(-payload.directionLightDir, reflection)) <= 0.001f) {
            if (totalInternalReflection) payload.directionLightRadiance /= cosineON;
            else payload.directionLightRadiance *= F / cosineON;
        }
    }
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

    if (any(payload.directionLightRadiance > 0.001f)) {
        payload.directionLightRadiance *= 
            getBSDF_RoughConductor(payload.material.albedo, payload.material.roughness, -payload.directionLightDir, payload.hitNormal, payload.outgoing, payload.TBN);
    }
}
void PopulateCallablePayload_RoughDielectricMaterial(inout CallablePayload payload) {
    float randomNumber1 = rand(payload.randomSeed);
    float randomNumber2 = rand(payload.randomSeed);
    float roughness = payload.material.roughness;

    float eta = payload.material.eta.x;
    if (!payload.isExt) eta = 1.0f / eta;

    float3 outgoing_tangentSpace = mul(payload.TBN, payload.outgoing);
    float3 h_tangentSpace = hvd_ggx_sample_vndf(outgoing_tangentSpace, float2(roughness), float2(randomNumber1, randomNumber2));
    if (h_tangentSpace.z == 0.0f || outgoing_tangentSpace.z == 0.0f) {
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

    float3 incidence_tangentSpace;
    bool refraction = false;

    float SnellFactor = eta * eta * (1.0f - cosineOH * cosineOH);
    bool totalReflection = SnellFactor >= 1.0f;
    if (totalReflection) {
        incidence_tangentSpace = reflect(-outgoing_tangentSpace, h_tangentSpace);
    } else {
        float randomNumber = rand(payload.randomSeed);
        if (randomNumber < F_oneChanel) incidence_tangentSpace = reflect(-outgoing_tangentSpace, h_tangentSpace);
        else {
            incidence_tangentSpace = normalize( (-outgoing_tangentSpace * eta + h_tangentSpace * (eta * cosineOH - sqrt(1.0f - SnellFactor)))); ;
            refraction = true;
        }
    }
    if (!refraction && incidence_tangentSpace.z <= 0.0f) {
        payload.bsdf = float3(0.0f);
        payload.pdf = 1.0f;
        return;
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
        payload.pdf = (1.0f - F_oneChanel) * D_costhetaNH * cosineOH / (weight * weight + 1e-10f);

        payload.bsdf = D * G2 * (1.0f - F) * abs(dot(incidence_tangentSpace, h_tangentSpace)) * cosineOH
                        / (abs(weight * weight * outgoing_tangentSpace.z * incidence_tangentSpace.z) + 1e-10f);
        payload.bsdf *= eta * eta; // radiance from incidence to outgoing

        payload.isExt = !payload.isExt;
    } else {
        payload.pdf = (totalReflection ? 1.0f : F_oneChanel) * D_costhetaNH / (4.0f * cosineOH + 1e-10f);
        payload.bsdf = D * G2 * (totalReflection ? 1.0f : F) / (abs(4.0f * outgoing_tangentSpace.z * incidence_tangentSpace.z) + 1e-10f);
    }

    if (any(payload.directionLightRadiance > 0.001f)) {
        payload.directionLightRadiance *=
            getBSDF_RoughDielectric(payload.material.eta.x, payload.material.albedo, payload.material.roughness, -payload.directionLightDir, payload.hitNormal, payload.outgoing, payload.TBN);
    }
}

void PopulateCallablePayload_RoughPlasticMaterial(inout CallablePayload payload) {
    float roughness = payload.material.roughness;
    float alpha = roughness * roughness;

    float lS = 0.299 * payload.material.albedo_specular.x + 0.587 * payload.material.albedo_specular.y + 0.114 * payload.material.albedo_specular.z;
    float lR = 0.299 * payload.material.albedo.x + 0.587 * payload.material.albedo.y + 0.114 * payload.material.albedo.z;

    float specular_prob = lS / (lS + lR);
    float randomNumber = rand(payload.randomSeed);
    float randomNumber1 = rand(payload.randomSeed);
    float randomNumber2 = rand(payload.randomSeed);
    float3 incidence_tangentSpace, h_tangentSpace;
    float3 outgoing_tangentSpace = mul(payload.TBN, payload.outgoing);
    if (randomNumber <= specular_prob) {
        h_tangentSpace = SampleVisibleNormals( outgoing_tangentSpace, alpha, float2(randomNumber1, randomNumber2));    //hvd_ggx_sample_vndf(outgoing_tangentSpace, float2(roughness), float2(randomNumber1, randomNumber2));
        if (h_tangentSpace.z == 0.0f) {
            payload.bsdf = float3(0.0f);
            payload.pdf = 1.0f;
            return;
        }
        incidence_tangentSpace = reflect(-outgoing_tangentSpace, h_tangentSpace);
    } else {
        float cosTheta = sqrt(randomNumber1);
        float sinTheta = sqrt(1.0f - randomNumber1);
        float phi = M_TWO_PI * randomNumber2;
        incidence_tangentSpace.x = sinTheta * cos(phi);
        incidence_tangentSpace.y = sinTheta * sin(phi);
        incidence_tangentSpace.z = cosTheta;

        h_tangentSpace = normalize(outgoing_tangentSpace + incidence_tangentSpace);
    }
    float cosineOH = dot(outgoing_tangentSpace, h_tangentSpace);
    if (cosineOH <= 0.0f) {
        payload.bsdf = float3(0.0f);
        payload.pdf = 1.0f;
        return;
    }
    if (incidence_tangentSpace.z <= 0.0f) {
        payload.bsdf = float3(0.0f);
        payload.pdf = 1.0f;
        return;
    }
    payload.incidence = mul(incidence_tangentSpace, payload.TBN);
    payload.incidence = normalize(payload.incidence);

    float G1_outgoing = IsotropicGGX_Masking(outgoing_tangentSpace, roughness);
    float G1_incidence = IsotropicGGX_Masking(incidence_tangentSpace, roughness);
    float G2 = G1_outgoing * G1_incidence;
    //float G1_outgoing, G1_incidence;
    //float G2 = ggx_smith_shadow_mask(
    //            G1_outgoing,
    //            G1_incidence,
    //            outgoing_tangentSpace,
    //            incidence_tangentSpace,
    //            float2(roughness));
    if (G2 <= 0.0f) {
        payload.bsdf = float3(0.0f);
        payload.pdf = 1.0f;
        return;
    }
    // float D_costhetaNH = hvd_ggx_eval(float2(1.0f / roughness), h_tangentSpace);
    // float D = D_costhetaNH / h_tangentSpace.z;
    float D = GTR2_NDF(h_tangentSpace.z, roughness);

    float diffuse_prob = (1.0f - specular_prob) * incidence_tangentSpace.z / M_PI;
    specular_prob *= (G1_outgoing * D) / (4.0f * outgoing_tangentSpace.z);
    payload.pdf = specular_prob + diffuse_prob;

    float cosineIH = dot(incidence_tangentSpace, h_tangentSpace);
    float3 F_i = FresnelDielectric(cosineIH, payload.material.eta.x);   //schlickFresnel(payload.material.albedo_specular, float3(1.0F), cosineIH);

    float3 specular_contib = payload.material.albedo_specular * (D * G2 * F_i) / (4.0f * incidence_tangentSpace.z * outgoing_tangentSpace.z + 1e-10f);

    float3 F_o = FresnelDielectric( cosineOH, payload.material.eta.x);  //schlickFresnel(payload.material.albedo_specular, float3(1.0F), cosineOH);
    float3 diffuse_contrib = payload.material.albedo * (1.0f - F_o) * (1.0f - F_i) / M_PI;
    payload.bsdf = (specular_contib + diffuse_contrib) * incidence_tangentSpace.z;

    if (any(payload.directionLightRadiance > 0.001f)) {
        payload.directionLightRadiance *=
            getBSDF_RoughtPlastic(payload.material.eta.x, payload.material.albedo, payload.material.albedo_specular, payload.material.roughness, -payload.directionLightDir, payload.hitNormal, payload.outgoing, payload.TBN);
    }
}
//------------------------------------------BSDF Sample---------------------------------------------------
void BSDF_SAMPLE(inout CallablePayload payload) {
    if (payload.material.type == MaterialType::Diffuse) PopulateCallablePayload_DiffuseMaterial(payload);
    else if (payload.material.type == MaterialType::RoughConductor) PopulateCallablePayload_RoughConductorMaterial(payload);
    else if (payload.material.type == MaterialType::Conductor) PopulateCallablePayload_ConductorMaterial(payload);
    else if (payload.material.type == MaterialType::Dielectric) PopulateCallablePayload_DielectricMaterial(payload);
    else if (payload.material.type == MaterialType::RoughDielectric) PopulateCallablePayload_RoughDielectricMaterial(payload);
    else if (payload.material.type == MaterialType::RoughPlastic) PopulateCallablePayload_RoughPlasticMaterial(payload);
}

#endif