int getVoxelIndex(float3 worldPos, float3 voxelGroupStartPos, float3 voxelSize, int voxelCount) {
    /*
    We determine whether a pos is located in a node by checking whether it is contained within the aabb of the node in the order of 0-7.
    Therefore, we should assign the pos of the boundary to the smaller node to avoid issues caused by floating-point number comparisons
    */
    int3 voxelIndex = int3((worldPos - voxelGroupStartPos) / voxelSize);
    int voxelTotalCount = voxelCount * voxelCount * voxelCount;
    int voxelIndexU = 0;
    while (voxelTotalCount > 1) {
        voxelCount = voxelCount / 2;
        voxelTotalCount = voxelTotalCount / 8;
        //if (voxelIndex.z / voxelCount == 1) {
        //    voxelIndexU += 4 * voxelTotalCount;
        //    voxelIndex.z -= voxelCount;
        //}
        //if (voxelIndex.y / voxelCount == 1) {
        //    voxelIndexU += 2 * voxelTotalCount;
        //    voxelIndex.y -= voxelCount;
        //}
        //if (voxelIndex.x / voxelCount == 1) {
        //    voxelIndexU += voxelTotalCount;
        //    voxelIndex.x -= voxelCount;
        //}

        uint3 offset = voxelIndex / voxelCount;

        voxelIndexU += offset.z * 4 * voxelTotalCount;
        voxelIndex.z -= offset.z * voxelCount;

        voxelIndexU += offset.y * 2 * voxelTotalCount;
        voxelIndex.y -= offset.y * voxelCount;

        voxelIndexU += offset.x * voxelTotalCount;
        voxelIndex.x -= offset.x * voxelCount;
    }
    return voxelIndexU;
}

inline int getVGBIndex(float3 normal) {
    float3 absNormal = abs(normal);
    //float maxAxis = max(max(absNormal.x, absNormal.y), absNormal.z);
    //int3 mask = int3(absNormal == maxAxis);
    //int axis = mask.x * 0 + mask.y * 1 + mask.z * 2;
    //return axis * 2 + (sign(normal[axis]) + 1) / 2;

    int maxAxis = absNormal.x > absNormal.y ? (absNormal.x > absNormal.z ? 0 : 2) : (absNormal.y > absNormal.z ? 1 : 2);
    return maxAxis * 2 + (sign(normal[maxAxis]) + 1) / 2;
}