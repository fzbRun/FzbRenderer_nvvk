#include "renderer/LPV/shaders/LPVCommon.slang"

int3 LPV_FaceAxis(int face) {
    static const int3 a[6] = {
        int3(1, 0, 0),  int3(-1, 0, 0),
        int3(0, 1, 0), int3(0, -1, 0),
        int3(0, 0, 1),  int3(0, 0, -1)
    };
    return a[uint(face) < 6u ? face : 5];
}

[numthreads(LPV_VOXEL_THREADGROUP, LPV_VOXEL_THREADGROUP, LPV_VOXEL_THREADGROUP)]
[shader("compute")] void lpvPropagateMain(uint3 threadIndex: SV_DispatchThreadID){
    int resolution = LPV_GridResolution();
    int cascadeCount = LPV_CascadeCount();
    if (threadIndex.z >= uint(cascadeCount * resolution)) return;

    int cascade = int(threadIndex.z) / resolution;
    int3 cell = int3(int(threadIndex.x), int(threadIndex.y), int(threadIndex.z) - cascade * resolution);
    if (!LPV_CellInside(cell, resolution)) return;

    int iteration = pushConst.iteration;
    int readVolume = iteration & 1;
    int writeVolume = 1 - readVolume;

    float cellSize = LPV_CascadeCellSize(cascade);
    bool useOcclusion = (pushConst.flags & LPVFlags::eLPVFlagOcclusion) != 0;

    float3 cellCenter = LPV_CellCenter(cascade, cell);
    LPVSH result = LPV_MakeSH(float4(0.0, 0.0, 0.0, 0.0), float4(0.0, 0.0, 0.0, 0.0), float4(0.0, 0.0, 0.0, 0.0));
    for (int offset = 0; offset < 6; ++offset) {
        int3 offset3 = LPV_FaceAxis(offset);
        int3 neighbour = cell - offset3;
        if (!LPV_CellInside(neighbour, resolution)) continue;

        float4 neighbourR = LPV_LoadVolumeSH(readVolume, cascade, neighbour, 0);
        float4 neighbourG = LPV_LoadVolumeSH(readVolume, cascade, neighbour, 1);
        float4 neighbourB = LPV_LoadVolumeSH(readVolume, cascade, neighbour, 2);

        float3 neighbourCenter = cellCenter - offset3 * cellSize;

        float4 blocking = 0.0f;
        if (useOcclusion && iteration > 0) {
            float3 faceCenter = cellCenter - 0.5 * cellSize * offset3;
            blocking = LPV_SampleGvMerged(cascade, faceCenter);
        }

        for (int face = 0; face < 6; ++face) {
            if (offset == face) continue;

            int3 axis = LPV_FaceAxis(face);
            float3 faceNormal = float3(axis.x, axis.y, axis.z);
            float3 faceCenter = cellCenter - 0.5 * cellSize * faceNormal;

            float3 propagateDir = normalize(faceCenter - neighbourCenter);

            float attenuation = 1.0;
            if (useOcclusion && iteration > 0) {
                float blockProb = saturate(dot(blocking, LPV_SH_Basis(-propagateDir)));
                attenuation = 1.0 - blockProb;
            }

            float4 basisValue = LPV_SH_Basis(propagateDir);
            float intensityR = max(0.0, dot(neighbourR, basisValue));
            float intensityG = max(0.0, dot(neighbourG, basisValue));
            float intensityB = max(0.0, dot(neighbourB, basisValue));

            float4 lobe = LPV_SH_Lobe(-faceNormal);
            float solidAngle = (offset / 2 == face / 2) ? LPV_DIRECT_FACE_SOLID_ANGLE : LPV_SIDE_FACE_SOLID_ANGLE;
            float scale = solidAngle * LPV_INV_PI * attenuation;
            result.r += (scale * intensityR) * lobe;
            result.g += (scale * intensityG) * lobe;
            result.b += (scale * intensityB) * lobe;
        }
    }

    LPV_StoreVolumeSH(writeVolume, cascade, cell, result);
    LPV_AddAccumSH(cascade, cell, result);
}