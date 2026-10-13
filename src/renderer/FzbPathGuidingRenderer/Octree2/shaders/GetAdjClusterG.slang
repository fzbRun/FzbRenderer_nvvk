#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/FzbPathGuidingRenderer/FzbPathGuidingShaderio.h"
#include "renderer/FzbPathGuidingRenderer/Octree2/Octree2Shaderio_FzbPG.h"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"

/*
    computeMain_getAdjClusterG
    ---------------------------------------------------------------------------
    对每个 G 不可分节点(几何簇) label，在其它几何簇中找"最接近"的
    ADJ_CLUSTER_COUNT - 1 个；并把自己的 label 固定写在槽 0，按距离升序写入
    AdjClusterInfoBuffer。

    输出布局（扁平 uint 数组）：
        AdjClusterInfoBuffer[label * ADJ_CLUSTER_COUNT + 0] = label（自身，恒在第一位）
        AdjClusterInfoBuffer[label * ADJ_CLUSTER_COUNT + k] = 第 k 近的"其它簇"的 indivisibleNodeIndex
                                                             (k >= 1，按距离升序)
        不足时用 ADJ_CLUSTER_INVALID 填充。
    消费端判有效： value < GlobalInfoBuffer[0].indivisibleNodeCount_G
                  （哨兵 0xFFFFFFFF 恒 >= 节点总数，天然落在无效区）
    消费端可依赖： k == 0 恒为自身
                  => 空间复用里"某一条的着色侧是当前簇"就等价于 (i == 0)

    线程映射： 1 个线程 = 1 个几何簇(label = threadIndex)，线程内串行扫描所有候选。
              dispatch 组数 = ceil(IndivisibleNodeCount_G_FZBPG / 1024) = 2，即可覆盖
              N_G <= 2000 的全部情形（无间接 dispatch、无组内 barrier，早点退安全）。

    依赖顺序： 必须在 getOctreeLabel 之后（IndivisibleNodeInfoBuffer_G 已写好），
              且要有一条 compute -> compute 的 barrier。

    性能说明： 内层循环的 j 对整组线程一致 => IndivisibleNodeInfoBuffer_G[j] /
              OctreeNodeDataBuffer_G[layer][nodeIndex] 是"组内同地址"的 broadcast load，
              实际访存约 N_G * (N_G / 1024) * 48B（N_G=2000 时约 190KB），可忽略；
              top-K 用 [unroll] 静态索引 => 数组常驻寄存器，不会落 local memory。
*/

//---------------------------------------- 可调参数 ----------------------------------------
#define ADJ_CLUSTER_COUNT           Candidate_Samples_Count     // 每个几何簇保存的条目数(32)：1 个自身 + 31 个邻居
#define ADJ_CLUSTER_NORMAL_COS_MIN  0.7f                        // 平均法线相似度门限(1=完全同向)
#define ADJ_CLUSTER_INVALID         0xFFFFFFFFu                 // 空槽哨兵

// 距离度量：默认用"盒子间隙"(AABBDistance, 0 表示相接/嵌套, 无 NaN 风险)。
// 想改成"朝向面之间的距离"，把它换成 AABBDistance2(a, b) 即可。
#define ADJ_CLUSTER_DISTANCE(a, b)  AABBDistance(a, b)

[[vk::push_constant]] ConstantBuffer<OctreePushConstant_FzbPG> pushConst;

[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeNodeData_G)]] RWStructuredBuffer<OctreeNodeData_G_FzbPG, ScalarDataLayout> OctreeNodeDataBuffer_G[];

[[vk::binding(BindingPoints_Octree_FzbPG::eGlobalInfo)]] RWStructuredBuffer<OctreeGlobalInfo_FzbPG, ScalarDataLayout> GlobalInfoBuffer;

[[vk::binding(BindingPoints_Octree_FzbPG::eIndivisibleNodeInfos_G)]] RWStructuredBuffer<uint2> IndivisibleNodeInfoBuffer_G;

[[vk::binding(BindingPoints_Octree_FzbPG::eAdjClusterInfo_G)]] RWStructuredBuffer<uint> AdjClusterInfoBuffer;
//------------------------------------------------------------------------------------------
[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_getAdjClusterG(uint threadIndex: SV_DispatchThreadID) {
    uint indivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;

    uint nodeLabel = threadIndex;
    if (nodeLabel >= indivisibleNodeCount_G) return;    // N_G 是运行时的，超出即退出

    //-------------------------------------------------- 自身
    uint2 myInfo = IndivisibleNodeInfoBuffer_G[nodeLabel];              // x = layer, y = nodeIndex
    OctreeNodeData_G_FzbPG me = OctreeNodeDataBuffer_G[myInfo.x][myInfo.y];

    float  myNormalLength = length(me.meanNormal.xyz);
    bool   myNormalValid = myNormalLength > 1e-8f;
    float3 myNormal = myNormalValid ? me.meanNormal.xyz / myNormalLength : float3(0.0f, 0.0f, 1.0f);

    //-------------------------------------------------- top-K 初始化(距离升序)
    // 槽 0 固定留给自身（恒在第一位，不参与下面的竞争，也不受法线门控/距离门限影响）；
    // 槽 1 .. ADJ_CLUSTER_COUNT-1 才是"其它簇"的竞争位。
    float bestDistance[ADJ_CLUSTER_COUNT];
    uint  bestLabel[ADJ_CLUSTER_COUNT];
    bestDistance[0] = 0.0f;                             // 自身距离按 0 处理
    bestLabel[0] = nodeLabel;                           // ← 保证自身出现在第一个位置
    [unroll]
    for (int k = 1; k < ADJ_CLUSTER_COUNT; ++k) {
        bestDistance[k] = 3.402823466e+38F;             // FLT_MAX
        bestLabel[k] = ADJ_CLUSTER_INVALID;
    }
    float worstDistance = 3.402823466e+38F;

    //-------------------------------------------------- 扫描所有其它几何簇
    for (uint j = 0; j < indivisibleNodeCount_G; ++j) {
        // 这两个 load 对组内所有线程是同一地址(j 一致) => 硬件广播，开销可忽略
        uint2 otherInfo = IndivisibleNodeInfoBuffer_G[j];
        OctreeNodeData_G_FzbPG other = OctreeNodeDataBuffer_G[otherInfo.x][otherInfo.y];

        if (j == nodeLabel) continue;                                   // 自身已固定占用槽 0，扫描时排除，避免重复
        if (other.meanNormal.w <= 0.0f) continue;                       // 无数据(hasData)兜底

        if (myNormalValid) {                                            // 法线兼容性门控
            float  otherNormalLength = length(other.meanNormal.xyz);
            float3 otherNormal = otherNormalLength > 1e-8f ? other.meanNormal.xyz / otherNormalLength
                                                          : float3(0.0f, 0.0f, 1.0f);
            if (dot(myNormal, otherNormal) < ADJ_CLUSTER_NORMAL_COS_MIN) continue;

            // 可选：改成"同轴向法线"这种更严格的门控（与非 RIS 的 NEARBYNODE 一致）
            // uint myAxis    = myInfo.y    / (1u << (3u * myInfo.x));
            // uint otherAxis = otherInfo.y / (1u << (3u * otherInfo.x));
            // if (myAxis != otherAxis) continue;
        }

        float distance = ADJ_CLUSTER_DISTANCE(me.aabb, other.aabb);
        if (!(distance < worstDistance)) continue;                      // 比当前第 K 名还远(同时挡 NaN/相等)

        // 冒泡插入：只在槽 1..K-1 内下沉，第 K-1 名被挤出（槽 0 是自身，绝不能被挤掉）
        float carryDistance = distance;
        uint  carryLabel = j;
        [unroll]
        for (int k = 1; k < ADJ_CLUSTER_COUNT; ++k) {
            if (carryDistance < bestDistance[k]) {
                float swapDistance = bestDistance[k];
                bestDistance[k] = carryDistance;
                carryDistance = swapDistance;

                uint swapLabel = bestLabel[k];
                bestLabel[k] = carryLabel;
                carryLabel = swapLabel;
            }
        }
        worstDistance = bestDistance[ADJ_CLUSTER_COUNT - 1];
    }

    //-------------------------------------------------- 写回(槽 0 = 自身，其余含无效槽填充)
    uint outBase = nodeLabel * (uint)ADJ_CLUSTER_COUNT;
    [unroll]
    for (int k = 0; k < ADJ_CLUSTER_COUNT; ++k) {
        AdjClusterInfoBuffer[outBase + k] = bestLabel[k];
    }
    // 可选：如果空间复用想按"远近"加权，可以再开一个 float 缓冲把 bestDistance[k] 一并写出
    //       （注意 bestDistance[0] 恒为 0，即自身的距离）
}
