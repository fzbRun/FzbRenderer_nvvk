#ifndef LPV_DEBUG
#define LPV_DEBUG
#endif
#include "renderer/LPV/shaders/LPVCommon.slang"

[numthreads(16, 16, 1)]
[shader("compute")]
void createDepthImageMain(uint2 threadIndex: SV_DispatchThreadID) {
    if (any(threadIndex >= pushConst.screenSize)) return;

    uint pixelIndex = threadIndex.y * pushConst.screenSize.x + threadIndex.x;
    LPVPixelData pixelData = pixelData[pixelIndex];

    if (pixelData.positionValid.w == 0.0f) {
        depthImage[threadIndex] = 1.0f;
        return;
    }
    float4 pos = pixelData.positionValid;
    pos = mul(pos, pushConst.sceneInfoAddress.viewMatrix);
    //pos.z -= 2.0f;

    float4 ndcPos = mul(pos, pushConst.sceneInfoAddress.projMatrix);
    ndcPos /= ndcPos.w;
    depthImage[threadIndex] = ndcPos.z;
}