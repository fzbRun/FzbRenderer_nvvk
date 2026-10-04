#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/StochasticLightcuts/StochasticLightcutsShaderio.h"


[[vk::push_constant]] ConstantBuffer<StochasticLightcutsPushConstant> pushConst;
//------------------------------------------------------------------------------------------
groupshared SceneInfo groupSceneInfo;

#define SharedAreaLightCount 81
groupshared AreaLight_StochasticLightcuts sharedAreaLights[SharedAreaLightCount];

float AreaLightAreaInAABB(AreaLight_StochasticLightcuts light, AABB box){
    float3 e1 = light.edge1, e2 = light.edge2, s = light.startPos;
    float scale = length(cross(e1, e2));
    if (scale < 1e-8) return 0.0;
    float3 mn = box.minimum, mx = box.maximum;
    float2 poly[10], tmp[10];
    poly[0] = float2(0, 0); poly[1] = float2(1, 0); poly[2] = float2(1, 1); poly[3] = float2(0, 1);
    int n = 4;
    [unroll]
    for (int k = 0; k < 6; ++k)
    {
        int axis = k >> 1;
        float sign = (k & 1) == 1 ? -1.0 : 1.0;
        float e1a = (axis == 0) ? e1.x : (axis == 1) ? e1.y : e1.z;
        float e2a = (axis == 0) ? e2.x : (axis == 1) ? e2.y : e2.z;
        float sa = (axis == 0) ? s.x : (axis == 1) ? s.y : s.z;
        float ba = (k & 1) == 1 ? ((axis == 0)   ? mx.x : (axis == 1) ? mx.y : mx.z)
                           : ((axis == 0)   ? mn.x : (axis == 1) ? mn.y : mn.z);
        float a = sign * e1a, b = sign * e2a, c = sign * (sa - ba);
        int m = 0;
        for (int i = 0; i < n; ++i)
        {
            float2 p = poly[i], q = poly[(i + 1) % n];
            float fp = a * p.x + b * p.y + c, fq = a * q.x + b * q.y + c;
            bool pin = fp >= 0.0, qin = fq >= 0.0;
            if (pin) tmp[m++] = p;
            if (pin != qin) tmp[m++] = p + (fp / (fp - fq)) * (q - p);
        }
        n = m;
        if (n == 0) return 0.0;
        for (int i = 0; i < n; ++i) poly[i] = tmp[i];
    }
    if (n < 3) return 0.0;
    float area2 = 0.0;
    for (int i = 0; i < n; ++i)
    {
        float2 p = poly[i], q = poly[(i + 1) % n];
        area2 += p.x * q.y - p.y * q.x;
    }
    return abs(area2) * 0.5 * scale;
}

int getVoxelLinearIndex(int3 voxelIndex) {
    int halfSize = Scene_Resolution / 2;
    int subTreeNodeCount = halfSize * halfSize * halfSize;
    int voxelLinearIndex = 0;
    while (halfSize > 0) {
        voxelLinearIndex += int(voxelIndex.x >= halfSize) * subTreeNodeCount;
        voxelLinearIndex += int(voxelIndex.y >= halfSize) * 2 * subTreeNodeCount;
        voxelLinearIndex += int(voxelIndex.z >= halfSize) * 4 * subTreeNodeCount;

        voxelIndex = voxelIndex & (halfSize - 1);
        halfSize /= 2;
        subTreeNodeCount = halfSize * halfSize * halfSize;
    }
    return voxelLinearIndex;
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_injectLightInfo(uint3 threadIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    uint groupThreadIndex_linear = groupThreadIndex.z * 16 + groupThreadIndex.y * 4 + groupThreadIndex.x;
    uint sharedAreaLightCount = min(pushConst.areaLightCount, SharedAreaLightCount);
    for(int i = groupThreadIndex_linear; i < sharedAreaLightCount; i += 64) 
        sharedAreaLights[i] = pushConst.areaLightsAddress[i];
    if (groupThreadIndex_linear == 0) groupSceneInfo = pushConst.sceneInfoAddress[0];
    GroupMemoryBarrierWithGroupSync();

    float3 voxelSize = pushConst.sceneSize / Scene_Resolution;
    AABB voxelAABB;
    voxelAABB.minimum = pushConst.sceneStartPos + threadIndex * voxelSize;
    voxelAABB.maximum = voxelAABB.minimum + voxelSize;

    LightNode lightNode;
    {
        lightNode.normal = 0.0f;
        lightNode.color = 0.0f;
        lightNode.minimum.x = 3.402823466e+38F;
        lightNode.minimum.y = 3.402823466e+38F;
        lightNode.minimum.z = 3.402823466e+38F;
        lightNode.maximum.x = -3.402823466e+38F;
        lightNode.maximum.y = -3.402823466e+38F;
        lightNode.maximum.z = -3.402823466e+38F;
        lightNode.halfAngle = 0.0f;
    }
    float3 normalWeight = 0.0f;
    for (int i = 0; i < sharedAreaLightCount; ++i) {
        AreaLight_StochasticLightcuts areaLight = sharedAreaLights[i];
        float intersectArea = AreaLightAreaInAABB(areaLight, voxelAABB);
        if(intersectArea == 0.0f) continue;

        if (all(lightNode.color == 0.0f)) {
            lightNode.minimum = max(areaLight.startPos, voxelAABB.minimum);
            lightNode.maximum = min(areaLight.startPos + areaLight.edge1 + areaLight.edge2, voxelAABB.maximum);
        } else {
            lightNode.minimum = min(lightNode.minimum, max(areaLight.startPos, voxelAABB.minimum));
            lightNode.maximum = max(lightNode.maximum, min(areaLight.startPos + areaLight.edge1 + areaLight.edge2, voxelAABB.maximum));
        }

        normalWeight += intersectArea;
        lightNode.normal += intersectArea * areaLight.normal;
        
        lightNode.color += intersectArea * areaLight.color;
    }
    lightNode.normal /= max(normalWeight, 1e-10f);

    uint leafLightcutStartIndex = ((1 << (3 * pushConst.lightTreeDepth)) - 1) / 7;
    int leafNodeLinearIndex = getVoxelLinearIndex(threadIndex);    //threadIndex.z * Scene_Resolution * Scene_Resolution + threadIndex.y * Scene_Resolution + threadIndex.x;
    pushConst.lightTreeAddress[leafLightcutStartIndex + leafNodeLinearIndex] = lightNode;
}
//------------------------------------------------------------------------------------------
[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_CreateLightcuts_1(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (threadIndex >= pushConst.lightNodeTotalCount) return;

    LightNode lightNode;
    {
        lightNode.normal = 0.0f;
        lightNode.halfAngle = 0.0f;
        lightNode.minimum.x = 3.402823466e+38F;
        lightNode.minimum.y = 3.402823466e+38F;
        lightNode.minimum.z = 3.402823466e+38F;
        lightNode.maximum.x = -3.402823466e+38F;
        lightNode.maximum.y = -3.402823466e+38F;
        lightNode.maximum.z = -3.402823466e+38F;
        lightNode.color = 0.0f;
    }

    uint leafLightNodeCount = Scene_Resolution * Scene_Resolution * Scene_Resolution;
    uint midLightNodeCount = pushConst.lightNodeTotalCount - leafLightNodeCount;
    if (threadIndex >= leafLightNodeCount) {
        pushConst.lightTreeAddress[threadIndex - leafLightNodeCount] = lightNode;
        return;
    }
    lightNode = pushConst.lightTreeAddress[midLightNodeCount + threadIndex];
    bool hasData = any(lightNode.color > 0.0f);

    {
        uint warpLane = groupThreadIndex & 31;
        uint blockIndex = threadIndex / 8;
        uint blockLane = threadIndex & 7;
        uint blockFirstWarpLane = warpLane / 8 * 8;
        uint blockNodeMask = 0xff << blockFirstWarpLane;

        uint warpHasDataMask = WaveActiveBallot(hasData).x;
        uint blockHasDataNodeCount = countbits(warpHasDataMask & blockNodeMask);
        bool blockHasData = blockHasDataNodeCount > 0;
        if (blockLane == 0) pushConst.hasLightBlockArrayAddress[blockIndex] = (uint)blockHasData;
    }
}

groupshared uint groupHasLightBlockCount;
groupshared uint globalHasLightBlockOffset;
[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_CreateLightcuts_2(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (threadIndex >= pushConst.curLayerBlockCount) return;
    if (groupThreadIndex == 0) {
        groupHasLightBlockCount = 0;
        globalHasLightBlockOffset = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    uint warpIndex = groupThreadIndex / 32;
    uint warpLane = groupThreadIndex & 31;

    bool hasData = pushConst.hasLightBlockArrayAddress[threadIndex] == 1;
    uint warpHasDataBlockCount = WaveActiveCountBits(hasData);

    uint groupHasDataBlockOffset = 0;
    if (warpLane == 0) InterlockedAdd(groupHasLightBlockCount, warpHasDataBlockCount, groupHasDataBlockOffset);
    GroupMemoryBarrierWithGroupSync();

    if (groupThreadIndex == 0) InterlockedAdd(pushConst.hasLightBlockCountAddress[0], groupHasLightBlockCount, globalHasLightBlockOffset);
    GroupMemoryBarrierWithGroupSync();

    groupHasDataBlockOffset = WaveReadLaneAt(groupHasDataBlockOffset, 0);
    if (hasData) {
        uint globalHasDataBlockOffset = globalHasLightBlockOffset + groupHasDataBlockOffset + WavePrefixCountBits(hasData); ;
        pushConst.hasLightBlockIndexAddress[globalHasDataBlockOffset] = threadIndex;
    }

    if (threadIndex < pushConst.curLayerBlockCount / 8) pushConst.hasLightBlockArrayAddress[threadIndex] = false;
}

[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain_CreateLightcuts_3(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    DispatchIndirectCommand cmd;
    cmd.x = pushConst.hasLightBlockCountAddress[0] * 8;
    cmd.y = 1; cmd.z = 1;
    pushConst.cmd[0] = cmd;
}

groupshared uint globalHasLightBlockCount;

void atomicMergeAABB(inout AABB aabb) {
    uint laneIndex = WaveGetLaneIndex();

    for (int offset = 4; offset > 0; offset /= 2) {
        uint targetLane = laneIndex + offset;
        float other_val = WaveReadLaneAt(aabb.minimum.x, targetLane);
        aabb.minimum.x = min(aabb.minimum.x, other_val);
    }
    // aabb.minimum.x = WaveReadLaneAt(aabb.minimum.x, blockFirstWarpLane);

    for (int offset = 4; offset > 0; offset /= 2) {
        uint targetLane = laneIndex + offset;
        float other_val = WaveReadLaneAt(aabb.minimum.y, targetLane);
        aabb.minimum.y = min(aabb.minimum.y, other_val);
    }

    for (int offset = 4; offset > 0; offset /= 2) {
        uint targetLane = laneIndex + offset;
        float other_val = WaveReadLaneAt(aabb.minimum.z, targetLane);
        aabb.minimum.z = min(aabb.minimum.z, other_val);
    }

    for (int offset = 4; offset > 0; offset /= 2) {
        uint targetLane = laneIndex + offset;
        float other_val = WaveReadLaneAt(aabb.maximum.x, targetLane);
        aabb.maximum.x = max(aabb.maximum.x, other_val);
    }

    for (int offset = 4; offset > 0; offset /= 2) {
        uint targetLane = laneIndex + offset;
        float other_val = WaveReadLaneAt(aabb.maximum.y, targetLane);
        aabb.maximum.y = max(aabb.maximum.y, other_val);
    }

    for (int offset = 4; offset > 0; offset /= 2) {
        uint targetLane = laneIndex + offset;
        float other_val = WaveReadLaneAt(aabb.maximum.z, targetLane);
        aabb.maximum.z = max(aabb.maximum.z, other_val);
    }
}

[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_CreateLightcuts_4(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) globalHasLightBlockCount = pushConst.hasLightBlockCountAddress[0];
    GroupMemoryBarrierWithGroupSync();
    if (threadIndex >= globalHasLightBlockCount * 8) return;

    uint warpIndex = groupThreadIndex / 32;
    uint warpLane = groupThreadIndex & 31;

    uint blockLane = threadIndex & 7;
    uint blockFirstWarpLane = (warpLane / 8) * 8;
    uint blockNodeMask = 0xff << blockFirstWarpLane;

    uint currentBlockLabel = threadIndex / 8;
    uint nodeIndex = pushConst.hasLightBlockIndexAddress[currentBlockLabel] * 8 + blockLane + pushConst.curLayerNodeStartIndex;
    uint fatherNodeIndex = (nodeIndex - pushConst.curLayerNodeStartIndex) / 8 + pushConst.fatherLayerNodeStartIndex;

    LightNode lightNode = pushConst.lightTreeAddress[nodeIndex];
    bool hasData = any(lightNode.color > 0.0f);
    uint32_t warpHasDataMask = WaveActiveBallot(hasData).x;
    uint32_t blockHasDataNodeCount = countbits(warpHasDataMask & blockNodeMask);
    if (blockHasDataNodeCount == 1) {
        if (hasData) pushConst.lightTreeAddress[fatherNodeIndex] = lightNode;
    } else {
        float3 colorSum = lightNode.color;
        for (int offset = 4; offset > 0; offset /= 2) {
            uint targetLane = warpLane + offset;
            colorSum.x += WaveReadLaneAt(colorSum.x, targetLane);
            colorSum.y += WaveReadLaneAt(colorSum.y, targetLane);
            colorSum.z += WaveReadLaneAt(colorSum.z, targetLane);
        }

        AABB mergeAABB = { lightNode.minimum, lightNode.maximum };
        atomicMergeAABB(mergeAABB);

        float weightI = dot(lightNode.color, 1.0f);
        float3 mergeNormal = lightNode.normal * weightI;
        for (int offset = 4; offset > 0; offset /= 2) {
            uint targetLane = warpLane + offset;
            mergeNormal.x += WaveReadLaneAt(mergeNormal.x, targetLane);
            mergeNormal.y += WaveReadLaneAt(mergeNormal.y, targetLane);
            mergeNormal.z += WaveReadLaneAt(mergeNormal.z, targetLane);
            weightI += WaveReadLaneAt(weightI, targetLane);
        }
        mergeNormal.x = WaveReadLaneAt(mergeNormal.x, blockFirstWarpLane);
        mergeNormal.y = WaveReadLaneAt(mergeNormal.y, blockFirstWarpLane);
        mergeNormal.z = WaveReadLaneAt(mergeNormal.z, blockFirstWarpLane);
        weightI = WaveReadLaneAt(weightI, blockFirstWarpLane);
        float halfAngle = M_PI;
        if (any(abs(mergeNormal) > 1e-3f)) {
            mergeNormal = normalize(mergeNormal / weightI);
            float cosine = dot(lightNode.normal, mergeNormal);
            float maxCosine = cosine;
            for (int offset = 4; offset > 0; offset /= 2) {
                uint targetLane = warpLane + offset;
                maxCosine = max(WaveReadLaneAt(maxCosine, targetLane), maxCosine);
            }
            maxCosine = WaveReadLaneAt(maxCosine, blockFirstWarpLane);
            halfAngle = acos(maxCosine);
        }
        else mergeNormal = float3(0.0f, 1.0f, 0.0f);

        if (blockLane == 0) {
            lightNode.normal = mergeNormal;
            lightNode.color = colorSum;
            lightNode.minimum = mergeAABB.minimum;
            lightNode.maximum = mergeAABB.maximum;
            lightNode.halfAngle = halfAngle;
            pushConst.lightTreeAddress[fatherNodeIndex] = lightNode;
        }
    }

    if (blockLane == 0 && blockHasDataNodeCount > 0 && pushConst.curLayer > 1) {
        uint fatherBlockIndex = (fatherNodeIndex - pushConst.fatherLayerNodeStartIndex) / 8;
        pushConst.hasLightBlockArrayAddress[fatherBlockIndex] = true;
    }
}