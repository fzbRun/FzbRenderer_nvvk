#include "renderer/LPV/shaders/LPVCommon.slang"

//-----------------------------------------------------------------------------------------
// Pass 3: merge the geometry volumes (component wise maximum, paper section 3.2)
//------------------------------------------------------------------------------------------
[numthreads(LPV_VOXEL_THREADGROUP, LPV_VOXEL_THREADGROUP, LPV_VOXEL_THREADGROUP)]
[shader("compute")] void lpvGvMergeMain(uint3 threadIndex : SV_DispatchThreadID){
    int resolution = LPV_GvResolution(pushConst.cascadeIndex);
    int3 cell = int3(threadIndex);
    if (!LPV_CellInside(cell, resolution)) return;

    int cascade = pushConst.cascadeIndex;
    uint base = LPV_GvFloatIndex(cascade, cell, 0);
    for (int coefficient = 0; coefficient < LPV_GV_FLOATS; ++coefficient)
        gvMerged[cascade][base + coefficient] = max(gvCamera[cascade][base + coefficient], gvRsm[cascade][base + coefficient]);
}
