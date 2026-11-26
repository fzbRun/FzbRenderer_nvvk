#ifndef LPV_DEBUG
#define LPV_DEBUG
#endif
#include "renderer/LPV/shaders/LPVCommon.slang"

int3 LPV_FaceAxis(int face) {
    static const int3 a[6] = {
        int3(1, 0, 0),  int3(-1, 0, 0),
        int3(0, 1, 0), int3(0, -1, 0),
        int3(0, 0, 1),  int3(0, 0, -1)
    };
    return a[uint(face) < 6u ? face : 5];
}

[numthreads(1024, 1, 1)]
[shader("compute")] void printfSHMain(uint threadIndex: SV_DispatchThreadID) {
    int3 range = pushConst.showGridVoxelRangeMax - pushConst.showGridVoxelRangeMin + 1;
    int showVoxelCount = range.x * range.y * range.z;
    if(threadIndex >= showVoxelCount) return;

    int3 voxelIndex;
    voxelIndex.z = threadIndex / (range.x * range.y);
    voxelIndex.y = (threadIndex - voxelIndex.z * range.x * range.y) / range.x;
    voxelIndex.x = threadIndex % range.x;
    voxelIndex += pushConst.showGridVoxelRangeMin;

    int cascade = pushConst.cascadeIndex;
    if (pushConst.showSHType == 0 || pushConst.showSHType == 1 || pushConst.showSHType == 2) {
        float3 voxelCenterPos = LPV_CellCenter(cascade, voxelIndex);

        float blockProb[6];
        for (int face = 0; face < 6; ++face) {
            float3 normal = float3(LPV_FaceAxis(face));
            float3 pos = voxelCenterPos + normal * 0.5f * LPV_CascadeCellSize(cascade);

            float4 cBlocking;
            if (pushConst.showSHType == 2) cBlocking = LPV_LoadGvMerged(cascade, voxelIndex);
            else cBlocking = LPV_LoadGvAny(pushConst.gvSource, cascade, voxelIndex);
            blockProb[face] = saturate(dot(cBlocking, LPV_SH_Basis(normal)));
        }

        printf("\nVoxelIndex: %d %d %d    Gv SH:\n +X: %f, -X: %f, +Y: %f, -Y: %f, +Z: %f, -Z: %f\n", 
            voxelIndex.x, voxelIndex.y, voxelIndex.z,
            blockProb[0], blockProb[1], blockProb[2], blockProb[3], blockProb[4], blockProb[5]);
    } else if (pushConst.showSHType == 3 || pushConst.showSHType == 4) {
        int sourceIndex = pushConst.iterationCount & 1;

        float4 cR; float4 cG; float4 cB;
        if (pushConst.showSHType == 3) {
            cR = LPV_LoadVolumeSH(sourceIndex, cascade, voxelIndex, 0);
            cG = LPV_LoadVolumeSH(sourceIndex, cascade, voxelIndex, 1);
            cB = LPV_LoadVolumeSH(sourceIndex, cascade, voxelIndex, 2);
        } else {
            cR = LPV_LoadAccumSH(cascade, voxelIndex, 0);
            cG = LPV_LoadAccumSH(cascade, voxelIndex, 1);
            cB = LPV_LoadAccumSH(cascade, voxelIndex, 2);
        }

        float3 radiance[6];
        for (int face = 0; face < 6; ++face) {
            float3 normal = float3(LPV_FaceAxis(face));
            float4 basisSH = LPV_SH_Basis(normal);
            radiance[face].x = max(0.0, dot(cR, basisSH));
            radiance[face].y = max(0.0, dot(cG, basisSH));
            radiance[face].z = max(0.0, dot(cB, basisSH));

            radiance[face] = abs(radiance[face]);
        }

        printf("\nVoxelIndex: %d %d %d    Gv SH:\n +X: %f %f %f, -X: %f %f %f, +Y: %f %f %f, -Y: %f %f %f, +Z: %f %f  %f, -Z: %f %f %f\n",
            voxelIndex.x, voxelIndex.y, voxelIndex.z,
            radiance[0].x, radiance[0].y, radiance[0].z,
            radiance[1].x, radiance[1].y, radiance[1].z,
            radiance[2].x, radiance[2].y, radiance[2].z,
            radiance[3].x, radiance[3].y, radiance[3].z,
            radiance[4].x, radiance[4].y, radiance[4].z,
            radiance[5].x, radiance[5].y, radiance[5].z,
        );
    }
}