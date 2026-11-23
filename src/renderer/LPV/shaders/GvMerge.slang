#include "renderer/LPV/shaders/LPVCommon.slang"

//-----------------------------------------------------------------------------------------
// Pass 3: merge the geometry volumes (component wise maximum, paper section 3.2)
//------------------------------------------------------------------------------------------
[numthreads(LPV_VOXEL_THREADGROUP, LPV_VOXEL_THREADGROUP, LPV_VOXEL_THREADGROUP)]
[shader("compute")] void lpvGvMergeMain(uint3 threadIndex : SV_DispatchThreadID){
    int resolution    = LPV_GridResolution();
    int cascadeCount  = LPV_CascadeCount();
    if (threadIndex.z >= uint(cascadeCount * resolution)) return;

    int  cascade = int(threadIndex.z) / resolution;
    int3 cell    = int3(int(threadIndex.x), int(threadIndex.y), int(threadIndex.z) - cascade * resolution);
    if (!LPV_CellInside(cell, resolution))  return;

    uint base = LPV_GvFloatIndex(cascade, cell, 0);
    for (int coefficient = 0; coefficient < LPV_GV_FLOATS; ++coefficient)
        gvMerged[base + coefficient] = max(gvCamera[base + coefficient], gvRsm[base + coefficient]);
}