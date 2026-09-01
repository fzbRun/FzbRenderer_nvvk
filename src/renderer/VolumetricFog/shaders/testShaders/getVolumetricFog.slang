#ifndef GET_VOLUMETRIC_FOG_SLANG
#define GET_VOLUMETRIC_FOG_SLANG

#include "renderer/VolumetricFog/shaders/shaderSource.slang"

void getHeightFogScatter(
    float3 startPos, float3 endPos, inout uint randomSeed, uint sampleCount,
                         VolumetricFogInfo fogInfo, GlobalInfo_VolumetricFog globalInfo,
    inout float L, inout float attenuation, inout float attenuation2
) {
    #ifdef COMPUTE_SHADER
    HeightFogInfo heightFogInfo = groupHeightFogInfos[fogInfo.typeFogIndex];
    #else
    HeightFogInfo heightFogInfo = heightFogInfoBuffer[fogInfo.typeFogIndex];
    #endif

    float3 fogStartPos = fogInfo.aabb.minimum;
    float3 gridMin = fogStartPos;
    float3 gridMax = fogInfo.aabb.maximum;

    float heightScale = heightFogInfo.heightScale;
    float absorption = heightFogInfo.absorption;
    float scattering = heightFogInfo.scattering;
    float phase = heightFogInfo.phase;
    float phase2 = phase * phase;
    float3 color = heightFogInfo.color;

    float3 rayDir = endPos - startPos;
    float maxDist = length(rayDir);
    if (maxDist < 1e-8) return;
    float3 dir = rayDir / maxDist;

    float tEnter, tExit;
    if (!isHit(startPos, endPos, gridMin, gridMax, tEnter, tExit)) return;
    float tTotal = tExit - tEnter;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    float E = 1.0f;
    float tPart = tTotal / sampleCount;
    for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
        float tPartStart = tEnter + sampleIndex * tPart;
        float tSample = tPartStart + tPart * (pushConst.randomStepping == 1 ? rand(randomSeed) : 0.5f);

        float3 samplePoint = startPos + dir * tSample;

        float fogHeight = max(samplePoint.y - fogStartPos.y, 0.0f);
        float extinctionCoeff = (absorption + scattering) * exp(-heightScale * fogHeight);
        extinctionCoeff = max(extinctionCoeff, 1e-30f);

        float3 indirectL; float a;
        getFogScatter(samplePoint, dir, tPart, sceneInfo, globalInfo, extinctionCoeff, scattering, phase, color, heightFogInfo.ambientIntensity, E, a, indirectL);

        // Soft boundary blending based on distance to fluid AABB
        float fluidBlend = 0.0f;
        for (int i = 0; i < pushConst.fluidFogCount; ++i) {
            if (globalInfo.fluidStartUp[i] == 0) continue;
            AABB fluidAABB = globalInfo.fluidAABB[i];

            float3 fluidAABBHalfSize = (fluidAABB.maximum - fluidAABB.minimum) * 0.5f;
            AABB mergeAABB = { fluidAABB.minimum - 0.5f * fluidAABBHalfSize, fluidAABB.maximum + 0.5f * fluidAABBHalfSize };
            if (globalInfo.fluidStartUp[0] == 1 && checkPointInAABB(samplePoint, mergeAABB)) {
                float3 d = min(abs(samplePoint - mergeAABB.minimum), abs(mergeAABB.maximum - samplePoint));
                float3 fluidGridCenter = (mergeAABB.minimum + mergeAABB.maximum) * 0.5f;
                float3 maxD = min(fluidGridCenter - mergeAABB.minimum, mergeAABB.maximum - fluidGridCenter);
                fluidBlend = min(min(d.x, d.y), d.z) / min(min(maxD.x, maxD.y), maxD.z);

                break;
            }
        }
        
        L += luminance(indirectL) * (1.0f - fluidBlend);
        E *= lerp(a, 1.0f, fluidBlend);

        if (all(E <= 1e-3f)) break;
    }
    attenuation *= E;
}

void getFluidFogScatter(
    float3 startPos, float3 endPos, inout uint randomSeed,
    VolumetricFogInfo fogInfo, GlobalInfo_VolumetricFog globalInfo,
    inout float L, inout float attenuation
) {
    #ifdef COMPUTE_SHADER
        FluidFogInfo fluidFogInfo = groupFluidFogInfos[fogInfo.typeFogIndex];
    #else
        FluidFogInfo fluidFogInfo = fluidFogInfoBuffer[fogInfo.typeFogIndex];
    #endif
    if(fluidFogInfo.startUp == 0) return;

    float3 fogStartPos = fogInfo.aabb.minimum;
    float3 voxelSize = fluidFogInfo.voxelSize;
    int3 gridSize = fluidFogInfo.gridSize;

    float3 gridMin = fogStartPos;
    float3 gridMax = fogStartPos + float3(gridSize) * voxelSize;
    float3 gridCenter = (gridMin + gridMax) * 0.5f;

    float3 rayDir = endPos - startPos;
    float maxDist = length(rayDir);
    if (maxDist < 1e-8) return;
    float3 dir = rayDir / maxDist;

    float tEnter, tExit;
    if (!isHit(startPos, endPos, gridMin, gridMax, tEnter, tExit)) return;
    float tTotal = tExit - tEnter;

    float eps = 1e-5 * min(min(voxelSize.x, voxelSize.y), voxelSize.z);
    float tCurrent = tEnter + eps;
    float3 pos = startPos + dir * tCurrent;

    float3 invDir; int3 voxel; int3 step; float3 tDelta; float3 voxelBorder; float3 tMax;
    {
        invDir.x = (abs(dir.x) > 1e-8) ? rcp(dir.x) : 1e30;
        invDir.y = (abs(dir.y) > 1e-8) ? rcp(dir.y) : 1e30;
        invDir.z = (abs(dir.z) > 1e-8) ? rcp(dir.z) : 1e30;

        float3 voxelF = (pos - fogStartPos) / voxelSize;
        voxel = int3(floor(voxelF));
        voxel = clamp(voxel, int3(0, 0, 0), gridSize - 1);

        step.x = (dir.x > 0.0) ? 1 : ((dir.x < 0.0) ? -1 : 0);
        step.y = (dir.y > 0.0) ? 1 : ((dir.y < 0.0) ? -1 : 0);
        step.z = (dir.z > 0.0) ? 1 : ((dir.z < 0.0) ? -1 : 0);

        tDelta.x = (step.x != 0) ? voxelSize.x * abs(invDir.x) : 1e30;
        tDelta.y = (step.y != 0) ? voxelSize.y * abs(invDir.y) : 1e30;
        tDelta.z = (step.z != 0) ? voxelSize.z * abs(invDir.z) : 1e30;

        voxelBorder.x = (step.x > 0) ? (fogStartPos.x + (voxel.x + 1) * voxelSize.x) : (fogStartPos.x + voxel.x * voxelSize.x);
        voxelBorder.y = (step.y > 0) ? (fogStartPos.y + (voxel.y + 1) * voxelSize.y) : (fogStartPos.y + voxel.y * voxelSize.y);
        voxelBorder.z = (step.z > 0) ? (fogStartPos.z + (voxel.z + 1) * voxelSize.z) : (fogStartPos.z + voxel.z * voxelSize.z);

        tMax.x = (step.x != 0) ? (voxelBorder.x - startPos.x) * invDir.x : 1e30;
        tMax.y = (step.y != 0) ? (voxelBorder.y - startPos.y) * invDir.y : 1e30;
        tMax.z = (step.z != 0) ? (voxelBorder.z - startPos.z) * invDir.z : 1e30;

        if (step.x != 0 && tMax.x < tCurrent) tMax.x += tDelta.x;
        if (step.y != 0 && tMax.y < tCurrent) tMax.y += tDelta.y;
        if (step.z != 0 && tMax.z < tCurrent) tMax.z += tDelta.z;
    }

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    float E = 1.0f;
    float gridLength = length((gridMax - gridMin));

    int maxIter = gridSize.x + gridSize.y + gridSize.z;
    for (int i = 0; i < maxIter; i++) {
        if (any(voxel < 0) || any(voxel >= gridSize)) break;

        float tNext = min(min(tMax.x, tMax.y), tMax.z);
        tNext = min(tNext, tExit);
        float dt = tNext - tCurrent;

        if (dt > 0.0) {
            float tSample = tCurrent + dt * rand(randomSeed);
            float3 samplePoint = startPos + dir * tSample;

            float3 uvw = (samplePoint - fogStartPos) / (gridSize * voxelSize);
            float4 voxelInfo = FluidFogVoxelInfoImages_sample[fogInfo.typeFogIndex].SampleLevel(uvw, 0);
            float extinctionCoeff = voxelInfo.x;
            extinctionCoeff = max(extinctionCoeff, 1e-30f);
            float scattering = voxelInfo.y;
            float phase = voxelInfo.z;
            float phase2 = phase * phase;
            float3 color = voxelInfo.w; // UnpackFloatToFloat3(voxelInfo.w);

            float3 fluidL; float a;
            getFogScatter(samplePoint, dir, dt, sceneInfo, globalInfo, extinctionCoeff, scattering, phase, color, 0.0f, E, a, fluidL);

            float3 fluidAABBHalfSize = (fogInfo.aabb.maximum - fogInfo.aabb.minimum) * 0.5f;
            AABB mergeAABB = { fogInfo.aabb.minimum - 0.5f * fluidAABBHalfSize, fogInfo.aabb.maximum + 0.5f * fluidAABBHalfSize };
            float3 d = min(samplePoint - mergeAABB.minimum, mergeAABB.maximum - samplePoint);
            float3 maxD = min(gridCenter - mergeAABB.minimum, mergeAABB.maximum - gridCenter);
            float fluidBlend = min(min(d.x, d.y), d.z) / min(min(maxD.x, maxD.y), maxD.z);

            E *= lerp(1.0f, a, fluidBlend);
            L += lerp(0.0f, luminance(fluidL), fluidBlend);
            if (all(E <= 0.0f)) break;
        }

        tCurrent = tNext;
        if (tCurrent >= tExit - 1e-6) break;

        float axisEpsilon = 1e-6f;
        if (tMax.x <= tNext + axisEpsilon) {
            voxel.x += step.x;
            tMax.x += tDelta.x;
        }
        if (tMax.y <= tNext + axisEpsilon) {
            voxel.y += step.y;
            tMax.y += tDelta.y;
        }
        if (tMax.z <= tNext + axisEpsilon) {
            voxel.z += step.z;
            tMax.z += tDelta.z;
        }
    }
    attenuation *= E;
}

void getGridFogScatter(
    float3 startPos, float3 endPos, inout uint randomSeed, uint sampleCount,
    VolumetricFogInfo fogInfo, GlobalInfo_VolumetricFog globalInfo,
    inout float L, inout float attenuation, inout float attenuation2
) {
    #ifdef COMPUTE_SHADER
    GridFogInfo gridFogInfo = gridFogInfoBuffer[fogInfo.typeFogIndex];
    #else
    GridFogInfo gridFogInfo = gridFogInfoBuffer[fogInfo.typeFogIndex];
    #endif

    float3 fogStartPos = fogInfo.aabb.minimum;
    float3 voxelSize = gridFogInfo.voxelSize;
    int3 gridSize = gridFogInfo.gridSize;

    float3 gridMin = fogStartPos;
    float3 gridMax = fogStartPos + float3(gridSize) * voxelSize;
    float3 gridCenter = (gridMin + gridMax) * 0.5f;

    float3 rayDir = endPos - startPos;
    float maxDist = length(rayDir);
    if (maxDist < 1e-8) return;
    float3 dir = rayDir / maxDist;

    float tEnter, tExit;
    if (!isHit(startPos, endPos, gridMin, gridMax, tEnter, tExit)) return;
    float tTotal = tExit - tEnter;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    float E = 1.0f;
    float gridLength = length((gridMax - gridMin));

    //float scattering = gridFogInfo.scattering;
    //float phase = gridFogInfo.phase;
    //float phase2 = phase * phase;
    //float3 color = gridFogInfo.color;

    const float distancePower = 2.0f;
    for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
        float u0 = pow(float(sampleIndex) / float(sampleCount), distancePower);
        float u1 = pow(float(sampleIndex + 1) / float(sampleCount), distancePower);
        float tPart = (u1 - u0) * tTotal;
        float tPartStart = tEnter + u0 * tTotal;
        float tSample = tPartStart + tPart * (pushConst.randomStepping == 1 ? rand(randomSeed) : 0.5f);

        float3 samplePoint = startPos + dir * tSample;
        float fluidBlend = 0.0f;
        for (int i = 0; i < pushConst.fluidFogCount; ++i) {
            if (globalInfo.fluidStartUp[i] == 0) continue;
            AABB fluidAABB = globalInfo.fluidAABB[i];

            float3 fluidAABBHalfSize = (fluidAABB.maximum - fluidAABB.minimum) * 0.5f;
            AABB mergeAABB = { fluidAABB.minimum - 0.5f * fluidAABBHalfSize, fluidAABB.maximum + 0.5f * fluidAABBHalfSize };
            if (globalInfo.fluidStartUp[0] == 1 && checkPointInAABB(samplePoint, mergeAABB)) {
                float3 d = min(abs(samplePoint - mergeAABB.minimum), abs(mergeAABB.maximum - samplePoint));
                float3 fluidGridCenter = (mergeAABB.minimum + mergeAABB.maximum) * 0.5f;
                float3 maxD = min(fluidGridCenter - mergeAABB.minimum, mergeAABB.maximum - fluidGridCenter);
                fluidBlend = min(min(d.x, d.y), d.z) / min(min(maxD.x, maxD.y), maxD.z);

                break;
            }
        }

        float3 uvw = (samplePoint - fogInfo.aabb.minimum) / (fogInfo.aabb.maximum - fogInfo.aabb.minimum);
        float4 voxelInfo = gridFogImages[fogInfo.typeFogIndex].SampleLevel(uvw, 0);

        float extinctionCoeff = voxelInfo.x;
        float scattering = voxelInfo.y;
        float phase = voxelInfo.z;
        float3 color = voxelInfo.w;

        float3 indirectL; float a;
        getFogScatter(samplePoint, dir, tPart, sceneInfo, globalInfo, extinctionCoeff, scattering, phase, color, gridFogInfo.ambientIntensity, E, a, indirectL);

        L += luminance(indirectL) * (1.0f - fluidBlend);

        E *= lerp(a, 1.0f, fluidBlend);
        if (all(E <= 1e-3f)) break;
    }
    attenuation *= E;
}
/*
bool getCameraFrustumFogData(float3 worldPos, inout float3 hitWorldPos, inout float3 uvw) {
    SceneInfo sceneInfo;
    #ifdef COMPUTE_SHADER
    sceneInfo = groupSceneInfo;
    #else
    sceneInfo = pushConst.sceneInfoAddress[0];
    #endif
    float3 cameraPos = sceneInfo.cameraPosition;
    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;

    uint3 gridSize = pushConst.frustumGridSize;
    float nearPlane = pushConst.cameraNearPlane;
    float farPlane  = pushConst.cameraFarPlane;
    float tanHalfFov = pushConst.tanCameraFov_2;
    float aspect = pushConst.aspectRatio;

    float3 toPoint = worldPos - cameraPos;
    float viewZ = dot(toPoint, cameraForward);
    float viewX = dot(toPoint, cameraRight);
    float viewY = dot(toPoint, cameraUp);

    float viewZClamped = clamp(viewZ, nearPlane, farPlane);
    float tLinear = (viewZClamped - nearPlane) / (farPlane - nearPlane);
    float curUvwZ = pow(tLinear, 1.0f / pushConst.compressionPrecision);

    float voxelZ = curUvwZ * float(gridSize.z);

    float entryVoxelZ = voxelZ;
    if(pushConst.forwardSampleCount > 0){
        float P = float(pushConst.compressionPrecision);
        float N = float(gridSize.z);

        float maxRaymarchDist = float(pushConst.forwardSampleCount);
        if (P > 1.001f) {
            float proposedEntryZ = voxelZ / P;

            float entryDepth_min = max(nearPlane, viewZClamped - maxRaymarchDist);
            float tEntry_min = pow((entryDepth_min - nearPlane) / (farPlane - nearPlane), 1.0f / P);
            float entryVoxelZ_min = tEntry_min * N;

            entryVoxelZ = max(proposedEntryZ, entryVoxelZ_min);
            entryVoxelZ = clamp(entryVoxelZ, 0.0f, voxelZ);
            //if (pushConst.frameIndex == 0 && voxelZ > 30.0f && voxelZ < 40.0f) printf("voxelZ: %f, entryVoxelZ: %f\n", voxelZ, entryVoxelZ);
        }
    }
    float entryUvwZ = entryVoxelZ / float(gridSize.z);

    float entryTLinear = pow(entryUvwZ, pushConst.compressionPrecision);
    float entryDepth = lerp(nearPlane, farPlane, entryTLinear);
    entryDepth = max(entryDepth, nearPlane);

    float halfHeight = entryDepth * tanHalfFov;
    float halfWidth  = halfHeight * aspect;

    float scale = entryDepth / max(viewZ, 1e-8f);
    float entryViewX = viewX * scale;
    float entryViewY = viewY * scale;

    uvw.x = (entryViewX / halfWidth  + 1.0f) * 0.5f;
    uvw.y = (entryViewY / halfHeight + 1.0f) * 0.5f;
    uvw.z = entryUvwZ;
    uvw = clamp(uvw, 0.0f, 1.0f - 1e-6f);

    hitWorldPos = cameraPos + cameraRight * entryViewX + cameraUp * entryViewY + cameraForward * entryDepth;

    return true;
}
*/
/*
void getVolumetricFogScatter(
    float3 startPos, float3 endPos, inout uint randomSeed, uint sampleCount,
    inout float L, inout float attenuation, inout float attenuation2
) {
    #ifdef COMPUTE_SHADER
    GlobalInfo_VolumetricFog globalInfo = groupGlobalInfo;
    SceneInfo sceneInfo = groupSceneInfo;
    #else
    GlobalInfo_VolumetricFog globalInfo = globalInfoBuffer[0];
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    #endif

    VolumetricFogInfo fogInfo;
    for (int i = 0; i < globalInfo.visibleVolumetricFogCount; ++i) {
        uint fogIndex = globalInfo.visibleVolumetricFogIndices[i];

        #ifdef COMPUTE_SHADER
        fogInfo = groupVolumetricFogInfos[fogIndex];
        #else
        fogInfo = fogInfosBuffer[fogIndex];
        #endif

        uint typeFogIndex = fogInfo.typeFogIndex;
        if (fogInfo.type == VolumetricFogType::Height) {
            getHeightFogScatter(startPos, endPos, randomSeed, sampleCount, fogInfo, globalInfo, L, attenuation, attenuation2);
        } else if (fogInfo.type == VolumetricFogType::Fluid) {
            getFluidFogScatter(startPos, endPos, randomSeed, fogInfo, globalInfo, L, attenuation);
        }
        else if (fogInfo.type == VolumetricFogType::Grid) {
            GridFogInfo gridFogInfo = gridFogInfoBuffer[typeFogIndex];
            getGridFogScatter(startPos, endPos, randomSeed, sampleCount, fogInfo, globalInfo, L, attenuation, attenuation2);
        }
    }

    attenuation2 *= getAttenuation2(endPos, globalInfo, sceneInfo);
}
*/
void getVolumetricFogScatter(
    float3 startPos, float3 endPos, inout uint randomSeed, uint sampleCount, float stepPower, bool useGlobalHeight,
    float spatialDither, bool advanceTemporal,
    inout float L, inout float attenuation, inout float attenuation2
) {
    #ifdef COMPUTE_SHADER
    GlobalInfo_VolumetricFog globalInfo = groupGlobalInfo;
    SceneInfo sceneInfo = groupSceneInfo;
    #else
    GlobalInfo_VolumetricFog globalInfo = globalInfoBuffer[0];
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    #endif

    float3 dir = normalize(endPos - startPos);
    float tTotal = length(endPos - startPos);
    // startPos..endPos 就是一个体素板的厚度，在这么短的一段里再做 pow(i/N, 2) 的非均匀
    // 分布没有物理依据：它把所有采样点都挤到板的近端，等于系统性地把雾往前拉半个体素，
    // 而且每个体素的偏移量随深度变化，体素之间就对不齐 —— 相机移动时表现为一层层的错位。
    // 改成均匀分层，层内偏移 = 逐单元有序抖动（打破空间相关性）+ 逐帧低差异推进（时域超采样）。
    // spatialDither 这一项是必须的：没有它，全屏/全网格所有单元的采样点落在完全相同的
    // 相对位置上，雾的量化边界处处对齐 —— fogAcc开时就是块状伪影，关时就是整片雾同步跳动。
    float layerOffset = getMarchLayerOffset(globalInfo.temporalFrameIndex, spatialDither, advanceTemporal);

    float tPartStart = 0;
    float tPartEnd = pow(1.0f / sampleCount, stepPower) * tTotal;
    for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
        //float tPart = tTotal / (float)sampleCount;
        //float tStart = tPart * (float)sampleIndex;
        //float3 samplePoint = startPos + dir * (tStart + tPart * layerOffset);

        tPartStart = tPartEnd;
        tPartEnd = pow(float(sampleIndex + 1) / sampleCount, stepPower) * tTotal;
        float tPart = tPartEnd - tPartStart;
        float3 samplePoint = startPos + dir * (tPartStart + tPart * layerOffset);
        

        float envExtinction = 0.0f;
        float envScattering = 0.0f;
        float envPhase = 0.0f;
        float3 envColor = 0.0f;

        if (useGlobalHeight && globalInfo.useGlobalHeightFog == 1 && samplePoint.y >= globalInfo.globalHeightFogY.x && samplePoint.y <= globalInfo.globalHeightFogY.y) {
            HeightFogInfo globalHeightFogInfo = globalInfo.globalHeightFogInfo;
            float heightAttenuation = exp(-globalHeightFogInfo.heightScale * (samplePoint.y - globalInfo.globalHeightFogY.x));
            float extinction = (globalHeightFogInfo.absorption + globalHeightFogInfo.scattering) * heightAttenuation;
            envExtinction = extinction;
            envScattering = extinction * globalHeightFogInfo.scattering;
            envPhase = extinction * globalHeightFogInfo.phase;
            envColor = extinction * globalHeightFogInfo.color;
        }

        for (int i = 0; i < globalInfo.visibleVolumetricFogCount; ++i) {
            uint fogIndex = globalInfo.visibleVolumetricFogIndices[i];
            VolumetricFogInfo fogInfo = getVolumetricFog(fogIndex);
            if (!checkPointInAABB(samplePoint, fogInfo.aabb)) continue;

            float3 fogStartPos = fogInfo.aabb.minimum;
            float extinction = 0.0f, scattering = 0.0f, phase = 0.0f; float3 color = 0.0f;
            if (fogInfo.type == VolumetricFogType::Height) {
                HeightFogInfo heightFogInfo = getHeightFog(fogInfo.typeFogIndex);

                float heightScale = heightFogInfo.heightScale;
                float height = max(samplePoint.y - fogStartPos.y, 0.0f);
                float heightAttenuation = exp(-heightScale * height);
                scattering = heightFogInfo.scattering;
                extinction = (heightFogInfo.absorption + scattering) * heightAttenuation;
                phase = heightFogInfo.phase;
                color = heightFogInfo.color * (1.0f + heightFogInfo.ambientIntensity) * 2.0f;
            } else if (fogInfo.type == VolumetricFogType::Fluid && globalInfo.fluidStartUp[fogInfo.typeFogIndex] == 1) {
                FluidFogInfo fluidFogInfo = getFluidFog(fogInfo.typeFogIndex);
                float3 uvw = (samplePoint - fogStartPos) / (fluidFogInfo.gridSize * fluidFogInfo.voxelSize);
                uvw = clamp(uvw, 0.0f, 1.0f - 1e-6f);
                float4 voxelInfo = FluidFogVoxelInfoImages_sample[fogInfo.typeFogIndex].SampleLevel(uvw, 0);

                // if in fluid, the voxelInfo has include the envionment fog info
                envExtinction = max(voxelInfo.x, 1e-30f);
                envScattering = envExtinction * voxelInfo.y;
                envPhase = envExtinction * voxelInfo.z;
                envColor = envExtinction * voxelInfo.w; // UnpackFloatToFloat3(voxelInfo.w);
                break;
            }
            else if (fogInfo.type == VolumetricFogType::Grid) {
                GridFogInfo gridFogInfo = getGridFog(fogInfo.typeFogIndex);

                float3 uvw = (samplePoint - fogInfo.aabb.minimum) / (fogInfo.aabb.maximum - fogInfo.aabb.minimum);
                float4 voxelInfo = gridFogImages[fogInfo.typeFogIndex].SampleLevel(uvw, 0);

                extinction = voxelInfo.x;
                scattering = voxelInfo.y;
                phase = voxelInfo.z;
                color = voxelInfo.w * (1.0f + gridFogInfo.ambientIntensity);
            }

            envExtinction += extinction;
            envScattering += extinction * scattering;
            envPhase += extinction * phase;
            envColor += extinction * color;
        }

        envScattering /= max(envExtinction, 1e-30f);
        envPhase /= max(envExtinction, 1e-30f);
        envColor /= max(envExtinction, 1e-30f);

        if (envExtinction <= 1e-30f) continue;

        float3 indirectL; float a;
        getFogScatter(samplePoint, dir, tPart, sceneInfo, globalInfo, envExtinction, envScattering, envPhase, envColor, 0.0f, attenuation, a, indirectL);

        attenuation *= a;
        L += luminance(indirectL);
        if (all(attenuation <= 1e-3f)) break;
    }

    attenuation2 *= getAttenuation2(endPos, globalInfo, sceneInfo);
}
void getVolumetricFogScatter_Frustum(
    AABB startAABB, AABB endAABB, inout uint randomSeed, uint sampleCount,
    inout float L, inout float attenuation, inout float attenuation2
){
}
//------------------------------------------------------------------------------------------------------------------------------------------------
// jitterStrength 的单位是“深度切片”，不再是世界空间距离。
// sliceDither 由调用者算好（屏幕空间IGN），传0表示不抖动。
float4 getFrustumImageInfo(float3 worldPos, bool linear, float sliceDither, Sampler3D<float4> image, inout uint randomSeed) {
    #ifdef COMPUTE_SHADER
    SceneInfo sceneInfo = groupSceneInfo;
    #else
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    #endif

    float nearPlane = pushConst.cameraNearPlane;
    float farPlane = pushConst.cameraFarPlane;
    float3 compressionParams = max(pushConst.compressionParams, 1e-4f);
    float tanHalfFov = pushConst.tanCameraFov_2;
    float aspect = pushConst.aspectRatio;
    uint3 gridSize = pushConst.frustumGridSize;
    float3 gridSizeF = float3(gridSize);

    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;

    float3 toPoint = worldPos - sceneInfo.cameraPosition;
    float viewZ = dot(toPoint, cameraForward);
    float viewX = dot(toPoint, cameraRight);
    float viewY = dot(toPoint, cameraUp);
    if (viewZ <= 0.0f) return float4(0, 0, 0, 0);

    float clampedZ = clamp(viewZ, nearPlane, farPlane);
    float scale = clampedZ / viewZ;
    viewX *= scale; viewY *= scale; viewZ = clampedZ;

    float halfHeight = viewZ * tanHalfFov;
    float halfWidth = halfHeight * aspect;
    float u = saturate(viewX / halfWidth * 0.5f + 0.5f);
    float v = saturate(viewY / halfHeight * 0.5f + 0.5f);
#ifdef Interpolation_Manual
    uint voxelZ = clamp(int(frustumInvMap(viewZ - nearPlane, compressionParams, (farPlane - nearPlane), gridSize.z) / (farPlane - nearPlane) * gridSize.z), 0, gridSize.z - 1);

    float depth = viewZ - nearPlane;
    float depth0 = frustumMap(float(voxelZ) / gridSizeF.z * (farPlane - nearPlane), compressionParams, (farPlane - nearPlane), gridSize.z);
    float depth1 = frustumMap(float(voxelZ + 1) / gridSizeF.z * (farPlane - nearPlane), compressionParams, (farPlane - nearPlane), gridSize.z);
    float centerZ = (depth0 + depth1) * 0.5f;
    if (!linear || depth == centerZ) {
        float w = (float(voxelZ) + 0.5f) / gridSize.z;
        return image.SampleLevel(float3(u, v, w), 0);
    }

    int z0 = voxelZ, z1 = voxelZ;
    float blendZ = 0.0f;
    if (depth < centerZ && voxelZ > 0) {
        z0 = voxelZ - 1; z1 = voxelZ;
        float dPrev = frustumMap(float(z0) / gridSizeF.z * (farPlane - nearPlane), compressionParams, (farPlane - nearPlane), gridSize.z);
        float centerPrev = (dPrev + depth0) * 0.5f;
        blendZ = saturate((depth - centerPrev) / (centerZ - centerPrev));
    } else if (depth > centerZ && voxelZ < int(gridSize.z) - 1) {
        z1 = voxelZ + 1;
        float dNext = frustumMap(float(z1 + 1) / gridSizeF.z * (farPlane - nearPlane), compressionParams, (farPlane - nearPlane), gridSize.z);
        float centerNext = (depth1 + dNext) * 0.5f;
        blendZ = saturate((depth - centerZ) / (centerNext - centerZ));
    }
    float w = (float(z0) + 0.5f) / gridSize.z;
    float4 s0 = image.SampleLevel(float3(u, v, w), 0);

    w = (float(z1) + 0.5f) / gridSize.z;
    float4 s1 = image.SampleLevel(float3(u, v, w), 0);

    //return blendZ;
    return lerp(s0, s1, blendZ);
#else
    // 先换算到切片空间，抖动在切片空间加，最后再归一化到[0,1]。
    // 这样抖动幅度在近处和远处都恰好是“一个体素”，不会像世界空间抖动那样
    // 近处过强（跨很多体素）远处过弱（完全不起作用）。
    float zSlice = frustumInvMap(viewZ - nearPlane, compressionParams, farPlane - nearPlane, gridSize.z) / (farPlane - nearPlane) * gridSizeF.z;
    zSlice += sliceDither;
    float w = saturate(zSlice / gridSizeF.z);
    float3 uvw = float3(u, v, w);

    if (linear) return image.SampleLevel(uvw, 0);
    else return image[uint3(uvw * gridSizeF)];
#endif
}

// ditherPos: 屏幕空间像素坐标，用于IGN有序抖动。传0则不抖动。
void getPreAccFog(
    float3 startPos, float3 endPos, inout uint randomSeed, float2 ditherPos,
    inout float L, inout float attenuation, inout float attenuation2
) {
    #ifdef COMPUTE_SHADER
    GlobalInfo_VolumetricFog globalInfo = groupGlobalInfo;
    SceneInfo sceneInfo = groupSceneInfo;
    #else
    GlobalInfo_VolumetricFog globalInfo = globalInfoBuffer[0];
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    #endif
    float3 dir = normalize(endPos - startPos);

    float3 cameraPos = sceneInfo.cameraPosition;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;
    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;

    float nearPlane = pushConst.cameraNearPlane;
    float farPlane = pushConst.cameraFarPlane;
    float tanHalfFov = pushConst.tanCameraFov_2;
    float aspect = pushConst.aspectRatio;
    float3 compressionParams = pushConst.compressionParams;

    float tEnter, tExit;
    if (!IntersectFrustum(startPos, endPos, cameraPos, cameraForward, cameraRight, cameraUp, tanHalfFov, aspect, nearPlane, farPlane, tEnter, tExit)) return;
    endPos = startPos + tExit * dir;
    startPos = startPos + tEnter * dir;

    // 透射率和入射光必须来自同一个体素样本，否则两者在体素边界处的跳变位置不一致，
    // 会出现“亮边/暗边”的描边式锯齿。原来的实现用两个不同强度、且随机数状态已经推进过的
    // 抖动分别采样同一张图，等于人为把两者解耦了。
    float sliceDither = getFogSliceDither(ditherPos, globalInfo.temporalFrameIndex, pushConst.jitterStrength0);
    float4 fogInfo = getFrustumImageInfo(endPos, true, sliceDither, volumetricFogAccResultImage_sample, randomSeed);

    // 远平面保护：视锥网格在远平面处被硬截断，超出的部分没有数据。
    // 参考引擎用FarClipProtectRange把最后20%的雾效渐隐掉，避免远处出现一条硬边。
    float viewZ = dot(endPos - cameraPos, cameraForward);
    float fadeStart = farPlane * 0.8f;
    float farClipProtect = 1.0f - saturate((viewZ - fadeStart) / max(farPlane - fadeStart, 1e-4f));

    attenuation *= lerp(1.0f, fogInfo.x, farClipProtect);
    L += fogInfo.y * farClipProtect;
    attenuation2 *= getAttenuation2(endPos, globalInfo, sceneInfo);
}
//------------------------------------------------------------------------------------------------------------------------------------------------
#ifdef USE_ENVFOG
void getEnvVolumetricFogScatter(
    float3 startPos, float3 endPos, inout uint randomSeed, uint sampleCount,
    float spatialDither, bool advanceTemporal,
    inout float L, inout float attenuation, inout float attenuation2
) {
    #ifdef COMPUTE_SHADER
    GlobalInfo_VolumetricFog globalInfo = groupGlobalInfo;
    SceneInfo sceneInfo = groupSceneInfo;
    #else
    GlobalInfo_VolumetricFog globalInfo = globalInfoBuffer[0];
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    #endif
    float3 dir = normalize(endPos - startPos);

    float3 cameraPos = sceneInfo.cameraPosition;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;
    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;

    float nearPlane = pushConst.cameraNearPlane;
    float farPlane = pushConst.cameraFarPlane;
    float tanHalfFov = pushConst.tanCameraFov_2;
    float aspect = pushConst.aspectRatio;
    float3 compressionPrecision = pushConst.compressionParams;

#ifdef Uniform_EnvFog_Grid
    float3 gridMin = globalInfo.envStartPos;
    float3 gridMax = globalInfo.envStartPos + float3(globalInfo.envGridSize) * globalInfo.envVoxelSize;

    float tGridEnter, tGridExit;
    if (!isHit(startPos, endPos, gridMin, gridMax, tGridEnter, tGridExit)) return;
    endPos = startPos + tGridExit * dir;
    startPos = startPos + tGridEnter * dir;

    float tTotal = tGridExit - tGridEnter;
    float3 E = 1.0f;
    float tPart = tTotal / pushConst.sampleCount;
    float layerOffsetUniform = getMarchLayerOffset(globalInfo.temporalFrameIndex, spatialDither, advanceTemporal);
    for (int sampleIndex = 0; sampleIndex < pushConst.sampleCount; ++sampleIndex) {
        float tSample = tPart * ((float)sampleIndex + layerOffsetUniform);
        float3 samplePoint = startPos + dir * tSample;

        float3 uvw = (samplePoint - gridMin + 0.5f) / (gridMax - gridMin);
        float4 voxelInfo = envVolumetricFogInfoImage_sample.SampleLevel(uvw, 0);
        float extinctionCoeff = max(voxelInfo.x, 1e-30f);
        float scattering = voxelInfo.y;
        float phase = voxelInfo.z;
        float phase2 = phase * phase;
        float3 color = voxelInfo.w;

        Light light = sceneInfo.lights[0];
        float3 lightColor = light.color * light.intensity;
        float3 lightDirection; float3 lightPos;
        if (light.type == LightType::Direction) {
            lightDirection = -light.direction;
            lightPos = samplePoint + 1000.0f * lightDirection;
        } else if (light.type == LightType::Point) {
            lightDirection = normalize(light.pos - samplePoint);
            lightPos = light.pos;
        }

        float fp = (1.0f - phase2) / (4.0f * M_PI * pow((1 + phase2 - 2.0f * phase * dot(-dir, lightDirection)), 1.5f));

        float3 activeSunColor = lightColor * getAttenuation2(samplePoint, globalInfo, sceneInfo);
        activeSunColor *= getShadow(float4(samplePoint, 1.0f));

        float x = extinctionCoeff * tPart;
        float factor = (x < 1e-4) ? tPart * (1.0 - 0.5 * x) : (1.0 - exp(-x)) / extinctionCoeff;
        L += activeSunColor * fp * scattering * color * E * factor;

        E *= exp(-extinctionCoeff * tPart);
        if (all(E <= 1e-3f)) break;
    }
    attenuation *= E;
#else
    float tEnter, tExit;
    if(!IntersectFrustum(startPos, endPos, cameraPos, cameraForward, cameraRight, cameraUp, tanHalfFov, aspect, nearPlane, farPlane, tEnter, tExit)) return;
    endPos = startPos + tExit * dir;
    startPos = startPos + tEnter * dir;

    // 同 getVolumetricFogScatter：均匀分层 + 逐单元有序抖动 + 逐帧低差异推进，
    // 取代 pow(i/N, 2) 的非均匀分布和逐样本白噪声。
    float tTotal = tExit - tEnter;
    float layerOffset = getMarchLayerOffset(globalInfo.temporalFrameIndex, spatialDither, advanceTemporal);
    for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
        float tPart = tTotal / (float)sampleCount;
        float tSample = tPart * ((float)sampleIndex + layerOffset);
        float3 samplePoint = startPos + dir * tSample;

        // 这里的samplePoint本身已经是抖动过的光线步进点，再叠加切片抖动只会引入额外噪声
        float4 voxelInfo = getFrustumImageInfo(samplePoint, true, 0.0f, envVolumetricFogInfoImage_sample, randomSeed);
        float extinction = max(voxelInfo.x, 1e-30f);
        float scattering = voxelInfo.y;
        float phase = voxelInfo.z;
        float3 color = voxelInfo.w;

        if (extinction <= 1e-30f) continue;

        float3 indirectL; float a;
        getFogScatter(samplePoint, dir, tPart, sceneInfo, globalInfo, extinction, scattering, phase, color, 0.0f, attenuation, a, indirectL);

        attenuation *= a;
        L += luminance(indirectL);
        if (all(attenuation <= 1e-3f)) break;
    }
#endif

    attenuation2 *= getAttenuation2(endPos, globalInfo, sceneInfo);
}

/*
void getEnvVolumetricFogScatter_DDA(
    float3 startPos, float3 endPos, inout uint randomSeed, bool rayMarching,
    inout float3 L, inout float3 attenuation, inout float3 attenuation2
) {
    #ifdef COMPUTE_SHADER
    GlobalInfo_VolumetricFog globalInfo = groupGlobalInfo;
    SceneInfo sceneInfo = groupSceneInfo;
    #else
    GlobalInfo_VolumetricFog globalInfo = globalInfoBuffer[0];
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    #endif
    float3 dir = normalize(endPos - startPos);

    float3 cameraPos = sceneInfo.cameraPosition;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;
    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;

    float nearPlane = pushConst.cameraNearPlane;
    float farPlane = pushConst.cameraFarPlane;
    float tanHalfFov = pushConst.tanCameraFov_2;
    float aspect = pushConst.aspectRatio;
    float3 compressionParams = pushConst.compressionParams;

    float3 hitWorldPos = startPos;
    if (pushConst.useAccFog == 1) {
        hitWorldPos = getCameraFrustumInteralPos(endPos, cameraPos, sceneInfo.viewInvMatrix, pushConst.frustumGridSize, nearPlane,
            farPlane, tanHalfFov, aspect, pushConst.forwardSampleCount);
        attenuation *= getFrustumImageInfo(hitWorldPos, true, true, volumetricFogAttenuationImage_sample, randomSeed).xyz;
        L += getFrustumImageInfo(hitWorldPos, true, true, volumetricFogLImage_sample, randomSeed).xyz;
    }
    if (length(endPos - hitWorldPos) <= 1e-5f || !rayMarching) return;
    startPos = hitWorldPos;
#ifdef Uniform_EnvFog_Grid
    float3 gridMin = globalInfo.envStartPos;
    float3 gridMax = globalInfo.envStartPos + float3(globalInfo.envGridSize) * globalInfo.envVoxelSize;

    float tGridEnter, tGridExit;
    if (!isHit(startPos, endPos, gridMin, gridMax, tGridEnter, tGridExit)) return;
    endPos = startPos + tGridExit * dir;
    startPos = startPos + tGridEnter * dir;

    {
        float tLightEnter, tLightExit;
        float3 mainLightDir = sceneInfo.lights[0].direction;
        float3 lightStart = endPos - mainLightDir * 1000.0f;
        float mainLightDt = 1e-6f;
        if (isHit(lightStart, endPos, gridMin, gridMax, tLightEnter, tLightExit))
            mainLightDt = tLightExit - tLightEnter;
        float attenuationLength = max(globalInfo.envFogLightAttenuationLength, 1e-6f);
        float attenuationRatio = max(mainLightDt / attenuationLength, 0.0f);
        float lightAttenuation = pow(max(globalInfo.envFogLightAttenuationEstimator, 1e-6f), attenuationRatio);
        attenuation2 *= lightAttenuation * pushConst.lightAttenuationStrength;
    }

    float3 uvwStart = (startPos - gridMin) / (gridMax - gridMin);
    float3 uvwEnd = (endPos - gridMin) / (gridMax - gridMin);
    float3 uvwDir = uvwEnd - uvwStart;
    float uvwDist = length(uvwDir);
    if (uvwDist < 1e-8) return;

    int3 gridSize = int3(globalInfo.envGridSize);
    float3 voxelSize = 1.0 / float3(gridSize);

    int3 voxel = int3(floor(uvwStart * float3(gridSize)));
    voxel = clamp(voxel, int3(0, 0, 0), gridSize - 1);

    int3 step;
    step.x = (uvwDir.x > 0.0) ? 1 : ((uvwDir.x < 0.0) ? -1 : 0);
    step.y = (uvwDir.y > 0.0) ? 1 : ((uvwDir.y < 0.0) ? -1 : 0);
    step.z = (uvwDir.z > 0.0) ? 1 : ((uvwDir.z < 0.0) ? -1 : 0);

    float3 tDelta;
    tDelta.x = (step.x != 0) ? voxelSize.x / abs(uvwDir.x) : 1e30;
    tDelta.y = (step.y != 0) ? voxelSize.y / abs(uvwDir.y) : 1e30;
    tDelta.z = (step.z != 0) ? voxelSize.z / abs(uvwDir.z) : 1e30;

    float3 voxelBorder;
    voxelBorder.x = (step.x > 0) ? (voxel.x + 1) * voxelSize.x : voxel.x * voxelSize.x;
    voxelBorder.y = (step.y > 0) ? (voxel.y + 1) * voxelSize.y : voxel.y * voxelSize.y;
    voxelBorder.z = (step.z > 0) ? (voxel.z + 1) * voxelSize.z : voxel.z * voxelSize.z;

    float3 tMax;
    tMax.x = (step.x != 0) ? (voxelBorder.x - uvwStart.x) / uvwDir.x : 1e30;
    tMax.y = (step.y != 0) ? (voxelBorder.y - uvwStart.y) / uvwDir.y : 1e30;
    tMax.z = (step.z != 0) ? (voxelBorder.z - uvwStart.z) / uvwDir.z : 1e30;

    if (step.x != 0 && tMax.x < 0.0) tMax.x += tDelta.x;
    if (step.y != 0 && tMax.y < 0.0) tMax.y += tDelta.y;
    if (step.z != 0 && tMax.z < 0.0) tMax.z += tDelta.z;

    float tCurrent = 0.0;

    int maxIter = gridSize.x + gridSize.y + gridSize.z;
    for (int i = 0; i < maxIter; i++) {
        if (any(voxel < 0) || any(voxel >= gridSize)) break;

        float tNext = min(min(tMax.x, tMax.y), tMax.z);
        tNext = min(tNext, 1.0);
        float tSegUVW = tNext - tCurrent;

        if (tSegUVW > 0.0) {
            float3 uvwEntry = uvwStart + tCurrent * uvwDir;
            float3 uvwExit = uvwStart + tNext * uvwDir;

            float jitter = pushConst.randomStepping == 1 ? rand(randomSeed) : 0.5;
            float3 uvwJitter = uvwEntry + jitter * (uvwExit - uvwEntry);

            float3 worldEntry = gridMin + uvwEntry * (gridMax - gridMin);
            float3 worldExit = gridMin + uvwExit * (gridMax - gridMin);
            float3 worldJitter = gridMin + uvwJitter * (gridMax - gridMin);
            float tJitter = dot(worldJitter - startPos, dir);
            float3 samplePoint = startPos + dir * tJitter;

            float tWorldEntry = dot(worldEntry - startPos, dir);
            float tWorldExit = dot(worldExit - startPos, dir);
            float worldDt = max(tWorldExit - tWorldEntry, 1e-8f);

            float3 uvwSample = (samplePoint - gridMin + 0.5f) / (gridMax - gridMin);
            uint3 voxelIndex = clamp(uint3(uvwSample * gridSize), 0, gridSize - 1);
            float4 voxelInfo = envVolumetricFogInfoImage_sample.SampleLevel(uvwSample, 0);

            float extinctionCoeff = max(voxelInfo.x, 1e-30f);
            float scattering = voxelInfo.y;
            float phase = voxelInfo.z;
            float phase2 = phase * phase;
            float3 color = voxelInfo.w;

            for (int lightIndex = 0; lightIndex < sceneInfo.numLights; ++lightIndex) {
                Light light = sceneInfo.lights[lightIndex];
                float3 lightColor = light.color * light.intensity;
                float3 lightDirection; float3 lightPos;
                if (light.type == LightType::Direction) {
                    lightDirection = -light.direction;
                    lightPos = samplePoint + 1000.0f * lightDirection;
                } else if (light.type == LightType::Point) {
                    lightDirection = normalize(light.pos - samplePoint);
                    lightPos = light.pos;
                }

                float fp = (1.0f - phase2) / (4.0f * M_PI * pow((1 + phase2 - 2.0f * phase * dot(-dir, lightDirection)), 1.5f));

                float tLightEnter, tLightExit;
                float lightDt = 1e-6f;
                if (isHit(lightPos, samplePoint, gridMin, gridMax, tLightEnter, tLightExit))
                    lightDt = tLightExit - tLightEnter;

                float attenuationLength = max(globalInfo.envFogLightAttenuationLength, 1e-6f);
                float attenuationRatio = max(lightDt / attenuationLength, 0.0f);
                float lightAttenuation = pow(max(globalInfo.envFogLightAttenuationEstimator, 1e-6f), attenuationRatio);
                float3 activeSunColor = lightColor * lightAttenuation * pushConst.lightAttenuationStrength;
                if (lightIndex == 0) activeSunColor *= getShadow(float4(samplePoint, 1.0f));

                float x = extinctionCoeff * worldDt;
                float factor = (x < 1e-4) ? worldDt * (1.0 - 0.5 * x) : (1.0 - exp(-x)) / extinctionCoeff;
                L += activeSunColor * fp * scattering * color * attenuation * factor;
            }

            attenuation *= exp(-extinctionCoeff * worldDt);
            if (all(attenuation <= 1e-3f)) break;
        }

        tCurrent = tNext;
        if (tCurrent >= 1.0 - 1e-6) break;

        float axisEpsilon = 1e-6f;
        if (tMax.x <= tNext + axisEpsilon) {
            voxel.x += step.x;
            tMax.x += tDelta.x;
        }
        if (tMax.y <= tNext + axisEpsilon) {
            voxel.y += step.y;
            tMax.y += tDelta.y;
        }
        if (tMax.z <= tNext + axisEpsilon) {
            voxel.z += step.z;
            tMax.z += tDelta.z;
        }
    }
#else
    float tEntry, tExit;
    if (!IntersectFrustum(startPos, endPos, cameraPos, cameraForward, cameraRight, cameraUp, tanHalfFov, aspect, nearPlane, farPlane, tEntry, tExit)) return;
    float tTotal = tExit - tEntry;
    startPos = startPos + tEntry * dir;
    endPos = startPos + tTotal * dir;

    attenuation2 *= getAttenuation2(endPos, globalInfo, sceneInfo);

    float3 toStart = startPos - cameraPos;
    float depthS = dot(toStart, cameraForward);
    float vxS = dot(toStart, cameraRight);
    float vyS = dot(toStart, cameraUp);
    float halfWS = depthS * tanHalfFov * aspect;
    float halfHS = depthS * tanHalfFov;
    float3 uvwStart;
    uvwStart.z = frustumInvMap(depthS - nearPlane, compressionParams, farPlane - nearPlane) / (farPlane - nearPlane);
    uvwStart.x = saturate((vxS / max(halfWS, 1e-8f) + 1.0) * 0.5);
    uvwStart.y = saturate((vyS / max(halfHS, 1e-8f) + 1.0) * 0.5);

    float3 toEnd = endPos - cameraPos;
    float depthE = dot(toEnd, cameraForward);
    float vxE = dot(toEnd, cameraRight);
    float vyE = dot(toEnd, cameraUp);
    float halfWE = depthE * tanHalfFov * aspect;
    float halfHE = depthE * tanHalfFov;
    float3 uvwEnd;
    uvwEnd.z = frustumInvMap(depthE - nearPlane, compressionParams, farPlane - nearPlane) / (farPlane - nearPlane);
    uvwEnd.x = saturate((vxE / max(halfWE, 1e-8f) + 1.0) * 0.5);
    uvwEnd.y = saturate((vyE / max(halfHE, 1e-8f) + 1.0) * 0.5);

    float3 uvwDir = uvwEnd - uvwStart;
    float uvwDist = length(uvwDir);
    if (uvwDist < 1e-8) return;

    int3 gridSize = int3(pushConst.frustumGridSize);
    float3 voxelSize = 1.0 / float3(gridSize);

    int3 voxel = int3(floor(uvwStart * float3(gridSize)));
    voxel = clamp(voxel, int3(0, 0, 0), gridSize - 1);

    int3 step;
    step.x = (uvwDir.x > 0.0) ? 1 : ((uvwDir.x < 0.0) ? -1 : 0);
    step.y = (uvwDir.y > 0.0) ? 1 : ((uvwDir.y < 0.0) ? -1 : 0);
    step.z = (uvwDir.z > 0.0) ? 1 : ((uvwDir.z < 0.0) ? -1 : 0);

    float3 tDelta;
    tDelta.x = (step.x != 0) ? voxelSize.x / abs(uvwDir.x) : 1e30;
    tDelta.y = (step.y != 0) ? voxelSize.y / abs(uvwDir.y) : 1e30;
    tDelta.z = (step.z != 0) ? voxelSize.z / abs(uvwDir.z) : 1e30;

    float3 voxelBorder;
    voxelBorder.x = (step.x > 0) ? (voxel.x + 1) * voxelSize.x : voxel.x * voxelSize.x;
    voxelBorder.y = (step.y > 0) ? (voxel.y + 1) * voxelSize.y : voxel.y * voxelSize.y;
    voxelBorder.z = (step.z > 0) ? (voxel.z + 1) * voxelSize.z : voxel.z * voxelSize.z;

    float3 tMax;
    tMax.x = (step.x != 0) ? (voxelBorder.x - uvwStart.x) / uvwDir.x : 1e30;
    tMax.y = (step.y != 0) ? (voxelBorder.y - uvwStart.y) / uvwDir.y : 1e30;
    tMax.z = (step.z != 0) ? (voxelBorder.z - uvwStart.z) / uvwDir.z : 1e30;

    if (step.x != 0 && tMax.x < 0.0) tMax.x += tDelta.x;
    if (step.y != 0 && tMax.y < 0.0) tMax.y += tDelta.y;
    if (step.z != 0 && tMax.z < 0.0) tMax.z += tDelta.z;

    float3 E = 1.0f;
    float tCurrent = 0.0;
    int maxIter = gridSize.x + gridSize.y + gridSize.z;
    for (int i = 0; i < maxIter; i++) {
        if (any(voxel < 0) || any(voxel >= gridSize)) break;

        float tNext = min(min(tMax.x, tMax.y), tMax.z);
        tNext = min(tNext, 1.0);
        float tSegUVW = tNext - tCurrent;

        if (tSegUVW > 0.0) {
            float3 uvwEntry = uvwStart + tCurrent * uvwDir;
            float3 uvwExit = uvwStart + tNext * uvwDir;

            float jitter = pushConst.randomStepping == 1 ? rand(randomSeed) : 0.5;
            float3 uvwJitter = uvwEntry + jitter * (uvwExit - uvwEntry);

            float dEtry = frustumMap(uvwEntry.z * (farPlane - nearPlane), compressionParams, farPlane - nearPlane) + nearPlane;
            float hhEtry = dEtry * tanHalfFov;
            float3 worldEntry = cameraPos
                + cameraRight * ((uvwEntry.x * 2.0 - 1.0) * hhEtry * aspect)
                + cameraUp * ((uvwEntry.y * 2.0 - 1.0) * hhEtry)
                + cameraForward * dEtry;

            float dExit = frustumMap(uvwExit.z * (farPlane - nearPlane), compressionParams, farPlane - nearPlane) + nearPlane;
            float hhExit = dExit * tanHalfFov;
            float3 worldExit = cameraPos
                + cameraRight * ((uvwExit.x * 2.0 - 1.0) * hhExit * aspect)
                + cameraUp * ((uvwExit.y * 2.0 - 1.0) * hhExit)
                + cameraForward * dExit;

            float dJit = frustumMap(uvwJitter.z * (farPlane - nearPlane), compressionParams, farPlane - nearPlane) + nearPlane;
            float hhJit = dJit * tanHalfFov;
            float3 worldJitter = cameraPos
                + cameraRight * ((uvwJitter.x * 2.0 - 1.0) * hhJit * aspect)
                + cameraUp * ((uvwJitter.y * 2.0 - 1.0) * hhJit)
                + cameraForward * dJit;
            float tJitter = dot(worldJitter - startPos, dir);
            float3 samplePoint = startPos + dir * tJitter;

            float tWorldEntry = dot(worldEntry - startPos, dir);
            float tWorldExit = dot(worldExit - startPos, dir);
            float worldDt = max(tWorldExit - tWorldEntry, 1e-8f);

            float4 voxelInfo = getFrustumImageInfo(samplePoint, true, true, envVolumetricFogInfoImage_sample, randomSeed);

            float extinctionCoeff = max(voxelInfo.x, 1e-30f);
            float scattering = voxelInfo.y;
            float phase = voxelInfo.z;
            float phase2 = phase * phase;
            float3 color = voxelInfo.w;

            Light light = sceneInfo.lights[0];
            float3 lightColor = light.color * light.intensity;
            float3 lightDirection = -light.direction; float3 lightPos = samplePoint + 1000.0f * lightDirection;

            float fp = (1.0f - phase2) / (4.0f * M_PI * pow((1 + phase2 - 2.0f * phase * dot(-dir, lightDirection)), 1.5f));

            float3 activeSunColor = lightColor * getAttenuation2(samplePoint, globalInfo, sceneInfo);
            activeSunColor *= getShadow(float4(samplePoint, 1.0f));

            L += activeSunColor * fp * scattering * color * E / extinctionCoeff * (1.0f - exp(-extinctionCoeff * worldDt));

            E *= exp(-extinctionCoeff * worldDt);
            if (all(E <= 1e-3f)) break;
        }

        tCurrent = tNext;
        if (tCurrent >= 1.0 - 1e-6) break;

        if (tMax.x < tMax.y) {
            if (tMax.x < tMax.z) { voxel.x += step.x; tMax.x += tDelta.x; }
            else { voxel.z += step.z; tMax.z += tDelta.z; }
        } else {
            if (tMax.y < tMax.z) { voxel.y += step.y; tMax.y += tDelta.y; }
            else { voxel.z += step.z; tMax.z += tDelta.z; }
        }
    }
    attenuation *= E;
#endif
}
*/
#endif
#endif
