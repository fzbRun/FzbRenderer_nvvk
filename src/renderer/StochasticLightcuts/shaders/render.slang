#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "renderer/StochasticLightcuts/StochasticLightcutsShaderio.h"

[[vk::push_constant]] ConstantBuffer<StochasticLightcutsPushConstant> pushConst;

//------------------------------------------------------------------------------------------
groupshared SceneInfo groupSceneInfo;

#define MAX_LIGHTCUT_COUNT 32

//1 + 8 + 64 + 512 = 585
#define SHARED_LIGHTCUT_NODE_COUNT 585
groupshared LightNode groupLightNodes[SHARED_LIGHTCUT_NODE_COUNT];
//------------------------------------------------------------------------------------------
bool traceRay(RayDesc ray, inout CallablePayload payload)
{
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

        SceneInfo sceneInfo = groupSceneInfo;
        Instance instance = sceneInfo.instances[instanceIndex];
        Mesh mesh = sceneInfo.meshes[instance.meshIndex];
        payload.material = sceneInfo.materials[instance.materialIndex];

        const float3 barycentrics =
            float3(1.0 - barycentricCoords.x - barycentricCoords.y, barycentricCoords.x, barycentricCoords.y);

        float3 triPos[3];
        triPos = q.CommittedGetIntersectionTriangleVertexPositions();
        const float3 localPosition = triPos[0] * barycentrics.x + triPos[1] * barycentrics.y + triPos[2] * barycentrics.z;
        payload.hitPos = float3(mul(float4(localPosition, 1.0), objectToWorld));

        const float3 edge1 = triPos[1] - triPos[0];
        const float3 edge2 = triPos[2] - triPos[0];
        const float3 geoNormal = normalize(cross(edge1, edge2));

        payload.hitNormal = normalize(mul(geoNormal, transpose(worldToObject)).xyz);
        float cosineON = dot(payload.hitNormal, -ray.Direction);
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

bool traceRay2(RayDesc ray, inout float3 hitPos, inout float3 emissive) {
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

float3 RayMiss(float3 rayDirection) {
    SceneInfo sceneInfo = groupSceneInfo;
    if (sceneInfo.useSky == 1) return evalSimpleSky(sceneInfo.skySimpleParam, rayDirection);
    else return sceneInfo.backgroundColor;
}
//------------------------------------------------------------------------------------------
#define LIGHTCUT_RELATIVE_ERROR 0.02f
struct LightLeafNode {
    AABB aabb;
    float3 normal;
    float3 color;
    float pdf;
};

float MaxCosToAABB(float3 posX, float3 normalY, float3 minimum, float3 maximum) {
    if (all(normalY == 0.0f)) return 1.0f;
    normalY = normalize(normalY);

    float3 up = abs(normalY.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 tangentX = normalize(cross(up, normalY));
    float3 tangentY = cross(normalY, tangentX);

    float minX = 1e30, maxX = -1e30;
    float minY = 1e30, maxY = -1e30;
    float minZ = 1e30, maxZ = -1e30;

    [unroll]
    for (int i = 0; i < 8; ++i){
        float3 corner = float3(
            (i & 1) == 1 ? maximum.x : minimum.x,
            (i & 2) == 2 ? maximum.y : minimum.y,
            (i & 4) == 4 ? maximum.z : minimum.z
        );
        float3 rel = corner - posX;
        float x = dot(rel, tangentX);
        float y = dot(rel, tangentY);
        float z = dot(rel, normalY);

        minX = min(minX, x); maxX = max(maxX, x);
        minY = min(minY, y); maxY = max(maxY, y);
        minZ = min(minZ, z); maxZ = max(maxZ, z);
    }

    float minAbsX = (minX <= 0 && maxX >= 0) ? 0 : min(abs(minX), abs(maxX));
    float maxAbsX = max(abs(minX), abs(maxX));
    float minAbsY = (minY <= 0 && maxY >= 0) ? 0 : min(abs(minY), abs(maxY));
    float maxAbsY = max(abs(minY), abs(maxY));

    float denomSq = (maxZ >= 0) ? minAbsX * minAbsX + minAbsY * minAbsY + maxZ * maxZ
                                : maxAbsX * maxAbsX + maxAbsY * maxAbsY + maxZ * maxZ;

    if (denomSq <= 1e-20f) return 0.0;
    float bound = maxZ / sqrt(denomSq);
    return max(0.0, bound);
}

float3 getLightcutMUpperBound(LightNode lightNode, BSDFMaterial material, float3 pos, float3 normal, float3 outgoing) {
    float3 maxCosine = MaxCosToAABB(pos, lightNode.normal, lightNode.minimum, lightNode.maximum);

    float3 tangent, bitangent;
    orthonormalBasis(normal, tangent, bitangent);
    float3x3 TBN = float3x3(tangent, bitangent, normal);

    float3 maxBSDF_Cosine = -1.0f;
    for (int i = 0; i < 9; ++i) {
        float3 lightPos = lightNode.minimum + float3(
            (i & 1) == 1 ? lightNode.maximum.x : lightNode.minimum.x,
            (i & 2) == 2 ? lightNode.maximum.y : lightNode.minimum.y,
            (i & 4) == 4 ? lightNode.maximum.z : lightNode.minimum.z
        );
        if(i == 8) lightPos = (lightNode.maximum + lightNode.minimum) * 0.5f;
        float3 incidence = normalize(lightPos - pos);
        float3 brdf = getBSDF(material, incidence, normal, outgoing, TBN, true);
        maxBSDF_Cosine = max(maxBSDF_Cosine, brdf * maxCosine);
    }

    return maxBSDF_Cosine;
}
float getLightGUpperBound(LightNode lightNode, float3 pos) {
    float3 minimum = pos - lightNode.maximum;
    float3 maximum = pos - lightNode.minimum;

    float cosThetaMax = MaxCosToAABB(float3(0, 0, 0), lightNode.normal, minimum, maximum);
    cosThetaMax = clamp(cosThetaMax, -1.0, 1.0);

    float thetaMin = acos(cosThetaMax);
    if (thetaMin <= lightNode.halfAngle) return 1.0;
    else return max(0.0, cos(thetaMin - lightNode.halfAngle));
}

LightNode getLightNode(uint nodeIndex, int sharedLightcutNodeCount) {
    return nodeIndex < sharedLightcutNodeCount ? groupLightNodes[nodeIndex] : pushConst.lightTreeAddress[nodeIndex];
}
int getNodeLayer(int nodeIndex) {
    int layer = 0;
    int nodeCountSum = 1;
    for (; layer < pushConst.lightTreeDepth; ++layer) {
        if (nodeIndex < nodeCountSum) return layer;
        nodeCountSum += (1 << (3 * layer + 3));
    }
    return pushConst.lightTreeDepth;
}
LightLeafNode getLightLeafNode(uint subTreeNodeIndex, int sharedLightcutNodeCount, float3 pos, inout uint randomSeed) {
    LightLeafNode leafNode;
    leafNode.color = 0.0f;
    leafNode.pdf = 1.0f;

    LightNode node = getLightNode(subTreeNodeIndex, sharedLightcutNodeCount);
    if (all(node.color == 0.0f)) return leafNode;

    int layer = getNodeLayer(subTreeNodeIndex);
    int nodeIndex = subTreeNodeIndex;
    leafNode.pdf = 1.0f;
    for (; layer < pushConst.lightTreeDepth; ++layer) {
        float weightSum = 0.0f;
        float sampleWeight = 0.0f;
        int sampleSubNodeIndex = nodeIndex * 8 + 1;
        for (int blockLane = 0; blockLane < 8; ++blockLane) {
            int subNodeIndex = nodeIndex * 8 + blockLane + 1;
            node = getLightNode(subNodeIndex, sharedLightcutNodeCount);

            float M = 1.0f; //getLightcutMUpperBound(node, );
            float G = getLightGUpperBound(node, pos);
            float diagonalLength = length(node.maximum - node.minimum);
            float3 center = (node.maximum + node.minimum) * 0.5f;
            float d = max(length(pos - center), diagonalLength + 1e-3f);
            float weight = M * G * dot(node.color, 1.0f) / (d * d);

            //reservoir sampling
            weightSum += weight;
            if (weight / max(weightSum, 1e-10f) >= rand(randomSeed)) {
                sampleSubNodeIndex = subNodeIndex;
                leafNode.aabb = { node.minimum, node.maximum };
                leafNode.normal = node.normal;
                leafNode.color = node.color;
                sampleWeight = weight;
            }
        }
        nodeIndex = sampleSubNodeIndex;
        if (weightSum <= 0.0f) { leafNode.color = 0.0f; break; }
        leafNode.pdf *= sampleWeight / max(weightSum, 1e-10f);
    }
    return leafNode;
}

struct LightCutNode {
    uint nodeIndex;
    float error;
    float I;
};
float3 stochasticLightcutsSample(inout CallablePayload payload, int sharedLightcutNodeCount, int2 threadIndex) {
    LightCutNode cutNodes[MAX_LIGHTCUT_COUNT];
    if (all(groupLightNodes[0].color == 0.0f)) return 0.0f;
    cutNodes[0].nodeIndex = 0;

    uint cutNodeCount = 1;
    int maxErrorCutNodeIndex = 0;
    int maxErrorNodeIndex = 0;
    float cutNodeISum = 0.0f;

    int iteration = 20;
    while (iteration > 0) {
        bool firstHasISubNode = true;
        for (int i = 0; i < 8; ++i) {
            int subNodeIndex = maxErrorNodeIndex * 8 + i + 1;
            LightNode subNode = getLightNode(subNodeIndex, sharedLightcutNodeCount);
            if (all(subNode.color == 0.0f)) continue;

            int cutNodeIndex = maxErrorCutNodeIndex;
            if (!firstHasISubNode) {
                cutNodeIndex = cutNodeCount;
                ++cutNodeCount;
            } 
            firstHasISubNode = false;

            float I = dot(subNode.color, 1.0f);
            float3 MUpperBound = getLightcutMUpperBound(subNode, payload.material, payload.hitPos, payload.hitNormal, payload.outgoing);
            float GUpperBound = getLightGUpperBound(subNode, payload.hitPos);
            float diagonalLength = length(subNode.maximum - subNode.minimum);
            float3 center = (subNode.maximum + subNode.minimum) * 0.5f;
            float d = max(length(payload.hitPos - center), diagonalLength + 1e-3f);

            cutNodes[cutNodeIndex].nodeIndex = subNodeIndex;
            cutNodes[cutNodeIndex].error = I * dot(MUpperBound, 1.0f) * GUpperBound / (d * d);
            cutNodes[cutNodeIndex].I = I;

            cutNodeISum += I;
        }

        float errorThreshold = LIGHTCUT_RELATIVE_ERROR * cutNodeISum;
        float maxError = -3.402823466e+38F;
        int leafNodeStartPos = ((1 << (3 * pushConst.lightTreeDepth)) - 1) / 7;
        bool canContinue = false;
        for (int i = 0; i < cutNodeCount; ++i) {
            if (cutNodes[i].nodeIndex >= leafNodeStartPos) continue;
            if (maxError < cutNodes[i].error) {
                maxError = cutNodes[i].error;
                maxErrorCutNodeIndex = i;
                canContinue = true;
            }
        }
        if (maxError <= errorThreshold || !canContinue) break;
        if (cutNodeCount + 7 > MAX_LIGHTCUT_COUNT) break;
        cutNodeISum -= cutNodes[maxErrorCutNodeIndex].I;
        maxErrorNodeIndex = cutNodes[maxErrorCutNodeIndex].nodeIndex;

        --iteration;
    }

    RayDesc ray;
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    float3 colorSum = 0.0f;
    for (int i = 0; i < cutNodeCount; ++i) {
        LightLeafNode leafNode = getLightLeafNode(cutNodes[i].nodeIndex, sharedLightcutNodeCount, payload.hitPos, payload.randomSeed);
        float3 sampleDir;
        if (!SphericalRectangleSample(leafNode.aabb, payload.hitPos, payload.randomSeed, sampleDir, leafNode.pdf)) continue;

        ray.Direction = sampleDir;
        ray.Origin = payload.hitPos + 0.01f * ray.Direction;
        float3 hitPos, emissive;
        if (!traceRay2(ray, hitPos, emissive)) continue;
        if (!checkPointInAABB(hitPos, leafNode.aabb)) continue;

        float3 incidence = sampleDir;
        float3 bsdf = getBSDF(payload.material, incidence, payload.hitNormal, payload.outgoing, payload.TBN, payload.isExt);
        float cosineIN = abs(dot(payload.hitNormal, incidence));
        colorSum += emissive * bsdf * cosineIN / max(leafNode.pdf, 1e-6f);
    }

    //bool printfAble = pushConst.frameIndex == 0 && all(threadIndex >= pushConst.screenSize / 2 - 5) && all(threadIndex <= pushConst.screenSize / 2 + 5);
    //if (printfAble) {
    //    printf("%d %d %d %d\n", cutNodeCount, cutNodes[0].nodeIndex, cutNodes[1].nodeIndex, cutNodes[2].nodeIndex);
    //    colorSum = float3(100.0f, 0.0f, 0.0f);
    //}

    return colorSum;
}

//------------------------------------------------------------------------------------------
[numthreads(16, 16, 1)]
[shader("compute")]
void computeMain_render(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    uint groupThreadIndex_linear = groupThreadIndex.y * 16 + groupThreadIndex.x;
    uint sharedLightcutNodeCount = min(SHARED_LIGHTCUT_NODE_COUNT, pushConst.lightNodeTotalCount);
    for (int i = groupThreadIndex_linear; i < sharedLightcutNodeCount; i += 16 * 16) 
        groupLightNodes[i] = pushConst.lightTreeAddress[i];
    if (groupThreadIndex_linear == 0) groupSceneInfo = pushConst.sceneInfoAddress[0];
    GroupMemoryBarrierWithGroupSync();

    if (any(threadIndex >= pushConst.screenSize)) return;

    float2 launchID = (float2)threadIndex;
    float2 launchSize = (float2)pushConst.screenSize;

    SceneInfo sceneInfo = groupSceneInfo;

    RayDesc ray;
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    CallablePayload payload;
    payload.randomSeed = xxhash32(uint3(threadIndex.x, threadIndex.y, (uint)pushConst.frameIndex));

    const float2 pixelCenter = launchID + float2(0.5f, 0.5f);
    const float2 clipCoords = pixelCenter / launchSize * 2.0 - 1.0;
    float4 viewCoords = mul(float4(clipCoords, 1.0, 1.0), sceneInfo.projInvMatrix);
    viewCoords /= viewCoords.w;

    ray.Origin = sceneInfo.cameraPosition;
    ray.Direction = normalize(mul(float4(normalize(viewCoords.xyz), 0.0), sceneInfo.viewInvMatrix).xyz);

    payload.bsdf = float3(1.0f);
    payload.pdf = 1.0f;
    payload.outgoing = -ray.Direction;
    payload.isExt = true;

    float3 result = 0.0f;
    bool hit = traceRay(ray, payload);
    if (!hit) result += RayMiss(ray.Direction);
    else {
        result += payload.material.emissive;

        if (pushConst.mode == 0) {
            BSDF_SAMPLE(payload);
            float cosine = abs(dot(payload.hitNormal, payload.incidence));
            float3 bsdf_cosine_pdf = payload.bsdf * cosine / payload.pdf;

            ray.Direction = payload.incidence;
            ray.Origin = payload.hitPos + 0.01f * ray.Direction;
            hit = traceRay(ray, payload);

            result += hit ? bsdf_cosine_pdf * payload.material.emissive : RayMiss(ray.Direction);
        } else result += stochasticLightcutsSample(payload, (int)sharedLightcutNodeCount, threadIndex);
    }

    if (pushConst.maxFrameCount > 1 && pushConst.frameIndex > 0) {
        float a = 1.0f / float(pushConst.frameIndex + 1);
        float3 old_color = outImage[int2(launchID)].xyz;
        outImage[int2(launchID)] = float4(lerp(old_color, result, a), 1.0f);
    }
    else outImage[int2(launchID)] = float4(result, 1.0f);
}
