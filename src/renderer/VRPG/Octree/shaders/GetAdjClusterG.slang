#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/VRPG/VRPGShaderio.h"
#include "renderer/VRPG/Octree/OctreeShaderio_VRPG.h"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"

[[vk::push_constant]] ConstantBuffer<OctreePushConstant_VRPG> pushConst;

[[vk::binding(BindingPoints_Octree_VRPG::eOctreeNodeData_G)]] RWStructuredBuffer<OctreeNodeData_G_VRPG, ScalarDataLayout> OctreeNodeDataBuffer_G[];

[[vk::binding(BindingPoints_Octree_VRPG::eGlobalInfo)]] RWStructuredBuffer<OctreeGlobalInfo_VRPG, ScalarDataLayout> GlobalInfoBuffer;

[[vk::binding(BindingPoints_Octree_VRPG::eIndivisibleNodeInfos_G)]] RWStructuredBuffer<uint2> IndivisibleNodeInfoBuffer_G;
[[vk::binding(BindingPoints_Octree_VRPG::eAdjClusterInfo_G)]] RWStructuredBuffer<uint> AdjClusterInfoBuffer;

[[vk::binding(BindingPoints_Octree_VRPG::eNearbyNodeTempInfos)]] RWStructuredBuffer<IndivisibleNodeNearbyNodeTempInfo_VRPG, ScalarDataLayout> NearbyNodeTempInfoBuffer;
[[vk::binding(BindingPoints_Octree_VRPG::eNearbyNodeInfos)]] RWStructuredBuffer<OctreeNearbyNodeInfo_VRPG, ScalarDataLayout> NearbyNodeInfoBuffer;
//-------------------------------------------------------------------------------------------------------------------------------
[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain_dispatchGetNearbyClusterG() {
    GlobalInfoBuffer[0].cmd.x = GlobalInfoBuffer[0].indivisibleNodeCount_G * (GlobalInfoBuffer[0].indivisibleNodeCount_G + GETNEARBYNODES_CS_THREADGROUP_SIZE - 1) / GETNEARBYNODES_CS_THREADGROUP_SIZE;
}

struct IndivisibleNodeData_G {
    AABB aabb;
    uint normalIndex;
    uint layer;
    float extent;
    float3 normal;
};

groupshared uint groupIndivisibleNodeCount_G;
groupshared IndivisibleNodeData_G groupIndivisibleNodeData;

groupshared float groupDirWarpMinDistance[4][GETNEARBYNODES_CS_THREADGROUP_SIZE / 32];
groupshared float groupDirMinDistances[4];
groupshared int2 groupDirMinDistanceNodeInfos[4];

[numthreads(GETNEARBYNODES_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getNearbyNodes1(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) groupIndivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
    if (groupThreadIndex < 4) {
        groupDirMinDistances[groupThreadIndex] = 1e30f;
        groupDirMinDistanceNodeInfos[groupThreadIndex] = int2(-1);
    }
    GroupMemoryBarrierWithGroupSync();
    int indivisibleNodeCount = groupIndivisibleNodeCount_G;
    uint threadGroupCount_perIndivisibleNode = (indivisibleNodeCount + GETNEARBYNODES_CS_THREADGROUP_SIZE - 1) / GETNEARBYNODES_CS_THREADGROUP_SIZE;
    uint indivisibleNodeLabel = threadGroupIndex / threadGroupCount_perIndivisibleNode;
    if (groupThreadIndex == 0) {
        uint2 nodeInfos = IndivisibleNodeInfoBuffer_G[indivisibleNodeLabel];
        OctreeNodeData_G_VRPG nodeData = OctreeNodeDataBuffer_G[nodeInfos.x][nodeInfos.y];
        IndivisibleNodeData_G nodeInfo;

        // nodeInfo.normal = normalize(nodeData.meanNormal.xyz);
        uint octreeLayerNode = 1 << (3 * nodeInfos.x);
        nodeInfo.normalIndex = nodeInfos.y / octreeLayerNode;
        nodeInfo.layer = nodeInfos.x;
        float3 selectedExtent = nodeData.aabb.maximum - nodeData.aabb.minimum;
        nodeInfo.extent = max(selectedExtent.x, max(selectedExtent.y, selectedExtent.z));
        nodeInfo.normal = normalize(nodeData.meanNormal.xyz + float3(1e-8f));

        nodeInfo.aabb = nodeData.aabb;
        groupIndivisibleNodeData = nodeInfo;
    }
    GroupMemoryBarrierWithGroupSync();
    uint threadGroupOffset = threadGroupIndex % threadGroupCount_perIndivisibleNode;
    uint nearbyNodeLabel = threadGroupOffset * GETNEARBYNODES_CS_THREADGROUP_SIZE + groupThreadIndex;
    if (nearbyNodeLabel >= indivisibleNodeCount) return;

    uint remainingThreadCount = indivisibleNodeCount - (threadGroupCount_perIndivisibleNode - 1) * GETNEARBYNODES_CS_THREADGROUP_SIZE;
    uint warpCount = (threadGroupIndex % threadGroupCount_perIndivisibleNode) == threadGroupCount_perIndivisibleNode - 1 ? (remainingThreadCount + 31) / 32 : GETNEARBYNODES_CS_THREADGROUP_SIZE / 32;
    uint warpLane = groupThreadIndex & 31;
    uint warpIndex = groupThreadIndex / 32;

    IndivisibleNodeData_G indivisibleNodeData = groupIndivisibleNodeData;
    uint2 nearbyNodeInfo = IndivisibleNodeInfoBuffer_G[nearbyNodeLabel];
    OctreeNodeData_G_VRPG nearbyNodeData = OctreeNodeDataBuffer_G[nearbyNodeInfo.x][nearbyNodeInfo.y];
    uint octreeLayerNode = 1 << (3 * nearbyNodeInfo.x);
    uint nearbyNormalIndex = nearbyNodeInfo.y / octreeLayerNode;

    float3 indivisibleNodeCenter = (indivisibleNodeData.aabb.minimum + indivisibleNodeData.aabb.maximum) * 0.5f;
    float3 nearbyNodeCenter = (nearbyNodeData.aabb.minimum + nearbyNodeData.aabb.maximum) * 0.5f;

    // Use the tangent plane of the selected G normal. The two axes are the
    // world axes other than the normal axis, and each gets +/- slots.
    int normalAxis = int(indivisibleNodeData.normalIndex) / 2;
    int axisA = (normalAxis + 1) % 3;
    int axisB = (normalAxis + 2) % 3;

    float3 delta = nearbyNodeCenter - indivisibleNodeCenter;
    float projA = delta[axisA];
    float projB = delta[axisB];

    int dirIndex = 0;
    int extentAxis = axisA;
    if (abs(projA) >= abs(projB)) {
        dirIndex = projA >= 0.0f ? 0 : 1;
        extentAxis = axisA;
    } else {
        dirIndex = projB >= 0.0f ? 2 : 3;
        extentAxis = axisB;
    }

    // Requirement 2: only clusters with exactly the same normal partition.
    // Requirement 3: boundary AABB distance is the primary key.
    float boundaryDistance2 = AABBDistance2(indivisibleNodeData.aabb, nearbyNodeData.aabb);
    float3 nearbyExtent3 = nearbyNodeData.aabb.maximum - nearbyNodeData.aabb.minimum;
    float nearbyExtent = max(nearbyExtent3.x, max(nearbyExtent3.y, nearbyExtent3.z));
    float minExtent = max(min(indivisibleNodeData.extent, nearbyExtent), 1e-6f);
    float distance = boundaryDistance2 / (minExtent * minExtent);

    // A side neighbor must overlap the selected cluster on the other tangent
    // axis. Otherwise a cluster that is entirely below/above can be chosen as
    // the "left/right" neighbor just because its boundary is close.
    int otherAxis = extentAxis == axisA ? axisB : axisA;
    float overlap = min(indivisibleNodeData.aabb.maximum[otherAxis],
                        nearbyNodeData.aabb.maximum[otherAxis]) -
                    max(indivisibleNodeData.aabb.minimum[otherAxis],
                        nearbyNodeData.aabb.minimum[otherAxis]);
    if (overlap < 0.01f * minExtent)
        distance = 1e30f;

    // Direction alignment from the center delta is more stable than the
    // boundary vector when the AABBs overlap/touch: the boundary vector can be
    // zero there and would make every candidate look equally aligned.
    float3 directionVector = nearbyNodeCenter - indivisibleNodeCenter;
    float directionLength = length(directionVector);
    float alignment = 0.0f;
    if (directionLength > 1e-6f)
        alignment = abs(directionVector[extentAxis]) / directionLength;
    distance += (1.0f - clamp(alignment, 0.0f, 1.0f)) * 1e-6f;

    // Requirement 4: when both distance and direction are equal, prefer the
    // cluster that is longer along the selected cardinal direction.
    float extentAlong = max(nearbyNodeData.aabb.maximum[extentAxis] -
                            nearbyNodeData.aabb.minimum[extentAxis], 1e-6f);
    float tieBreak = (1.0f - clamp(extentAlong / max(indivisibleNodeData.extent, 1e-6f), 0.0f, 1.0f)) * 1e-9f;
    distance += tieBreak;

    if (nearbyNormalIndex != indivisibleNodeData.normalIndex)
        distance = 1e30f;
    if (indivisibleNodeLabel == nearbyNodeLabel) distance = 1e30f;

    for (int i = 0; i < 4; i++) {
        float dirDistance = i == dirIndex ? distance : 1e30f;
        float warpMinDistance = WaveActiveMin(dirDistance);
        if (warpLane == 0) groupDirWarpMinDistance[i][warpIndex] = warpMinDistance;
        uint warpMinDistanceLabel = WavePrefixCountBits(dirDistance == warpMinDistance);
        GroupMemoryBarrierWithGroupSync();

        warpMinDistance = 1e30f;
        if (warpLane < warpCount) warpMinDistance = groupDirWarpMinDistance[i][warpLane];
        float groupDirMinDistance = WaveActiveMin(warpMinDistance);
        uint groupMinDistanceLabel = WavePrefixCountBits(warpMinDistance == groupDirMinDistance);
        groupMinDistanceLabel = WaveReadLaneAt(groupMinDistanceLabel, warpIndex);

        if (dirDistance == groupDirMinDistance && warpMinDistanceLabel == 0 && groupMinDistanceLabel == 0)
            groupDirMinDistanceNodeInfos[i] = nearbyNodeInfo;

        if (groupThreadIndex == 0 && threadGroupCount_perIndivisibleNode > 1) groupDirMinDistances[i] = groupDirMinDistance;
    }

    if (groupThreadIndex == 0) {
        if (threadGroupCount_perIndivisibleNode == 1) {
            NearbyNodeInfoBuffer[indivisibleNodeLabel].nearbyNodeInfos = groupDirMinDistanceNodeInfos;
        } else {
            uint infoIndex = indivisibleNodeLabel * threadGroupCount_perIndivisibleNode + threadGroupOffset;
            NearbyNodeTempInfoBuffer[infoIndex].nodeLabel = indivisibleNodeLabel;
            NearbyNodeTempInfoBuffer[infoIndex].nearbyNodeInfos = groupDirMinDistanceNodeInfos;
            NearbyNodeTempInfoBuffer[infoIndex].nearbyNodeDistances = groupDirMinDistances;
        }
        if (threadIndex == 0) {
            DispatchIndirectCommand cmd;
            if (threadGroupCount_perIndivisibleNode == 1) cmd.x = 0;
            else cmd.x = (indivisibleNodeCount * 32 + GETNEARBYNODES2_CS_THREADGROUP_SIZE - 1) / GETNEARBYNODES2_CS_THREADGROUP_SIZE;
            cmd.y = 1; cmd.z = 1;

            if (threadGroupCount_perIndivisibleNode * NEARBY_NODE_COUNT_VRPG > 32) {
                printf("Error: Too many threads per indivisible node for getNearbyNodes. currentThreadCount: %u, maxThreadCount: %u\n",
                       threadGroupCount_perIndivisibleNode * NEARBY_NODE_COUNT_VRPG, 32);
                cmd.x = 0; cmd.y = 0; cmd.z = 0;
            }

            GlobalInfoBuffer[0].cmd = cmd;
        }
    }
}

[numthreads(GETNEARBYNODES2_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getNearbyNodes2(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) groupIndivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
    GroupMemoryBarrierWithGroupSync();
    int indivisibleNodeCount = groupIndivisibleNodeCount_G;
    uint indivisibleNodeLabel = threadIndex / 32;
    if (indivisibleNodeLabel >= indivisibleNodeCount) return;

    uint threadGroupCount_perIndivisibleNode = (indivisibleNodeCount + GETNEARBYNODES_CS_THREADGROUP_SIZE - 1) / GETNEARBYNODES_CS_THREADGROUP_SIZE;

    uint warpLane = groupThreadIndex & 31;
    uint warpIndex = groupThreadIndex / 32;

    float distance = 1e30f; int2 nearbyNodeIndex = -1; uint nodeLabel = 0;
    if (warpLane < threadGroupCount_perIndivisibleNode * 4) {
        uint infoIndex = indivisibleNodeLabel * threadGroupCount_perIndivisibleNode + warpLane / 4;
        distance = NearbyNodeTempInfoBuffer[infoIndex].nearbyNodeDistances[warpLane % 4];
        nearbyNodeIndex = NearbyNodeTempInfoBuffer[infoIndex].nearbyNodeInfos[warpLane % 4];

        nodeLabel = NearbyNodeTempInfoBuffer[indivisibleNodeLabel * threadGroupCount_perIndivisibleNode].nodeLabel;
    }

    int dirIndex = warpLane & 3;
    for (int i = 0; i < 4; ++i) {
        float dirDistance = dirIndex == i ? distance : 1e30f;
        float minDistance = WaveActiveMin(dirDistance);

        uint minDistanceLabel = WavePrefixCountBits(dirDistance == minDistance);
        if (dirDistance == minDistance && minDistanceLabel == 0)
            NearbyNodeInfoBuffer[nodeLabel].nearbyNodeInfos[i] = nearbyNodeIndex;
    }
}

#ifdef SpatialReuse_VRPG
/*
    computeMain_getAdjClusterG
    ---------------------------------------------------------------------------
    闂佽绨肩徊濠氾綖婢舵劖鍎婇柍鈺佸暟閳?G 濠电偞鍨堕幐鍝ョ矓绾惧浜归柡灞诲劚缁€鍡涙煕閳╁啳澹橀摶锟犳⒑?闂備礁鎲￠崹瑙勬叏瀹曞洨绀婄€广儱娲╂慨? label闂備焦瀵х粙鎴︽嚐椤栨熬鑰挎い蹇撶墕缁€鍌炴偣妤︽寧顏犻柡鍡楋躬閺屾盯骞嬮悩娈嬶紕绱撳鍛ら柤娲憾瀹曞爼鏁冩担绋跨畱闂?闂備礁鎼悧鍐磻閹剧粯鐓欓柟顓熷笒閸斻儳绱?闂?    ADJ_CLUSTER_COUNT - 1 濠电偞鍨堕幖鈺傜妞嬪孩顫曢梺顒€绉跺畵渚€鏌ら幖浣规锭閹喖姊洪悡搴疇濞存粍绮庨幑銏＄瑹閳ь剟骞?label 闂備焦鎮堕崕閬嶅箹椤愶附鍋╁Δ锝呭暙缁€鍐╃箾閸℃绠查棅顒夊墯娣?0闂備焦瀵х粙鎴﹀嫉椤掑倵鍋撻崹顐ょ煂闁圭懓瀚埞鎴﹀醇椤愩倐鍋撻弻銉︾厱婵°倐鍋撴繛灞傚€楅懞閬嶎敇閵忕姷顦ㄥ┑鐐叉缁绘劕顕?    AdjClusterInfoBuffer闂?
    闂佸搫顦悧濠囧箰閹间礁鍚规い鎾跺У閺嗘粓鏌涢幇鈺佸婵犫偓娴煎瓨鐓ユ繛鎴烆焽婢ф稑顭胯闁帮綁鎮?uint 闂備浇妗ㄩ悞锔界珶閸℃瑥鍨濈€广儱顦伴弲顒€顭块懜鐢点€掔紒?        AdjClusterInfoBuffer[label * ADJ_CLUSTER_COUNT + 0] = label闂備焦瀵х粙鎴︽偋閹版澘鍚规繝濠傜墢閸欐洘銇勯幇闈涘缂佲偓婢舵劖鐓欐い鎾跺枎缁楁帗绻濋埀顒勵敂閸℃瑧鐣跺銈嗗灱濞夋洟顢旈崡鐏诲綊鏁愰崨顖楁瀸缂?        AdjClusterInfoBuffer[label * ADJ_CLUSTER_COUNT + k] = 缂?k 闂佸搫顦弲婊堝垂閻熸壋鏋?闂備胶顭堢换妤€鈻斿☉銏″仼闁绘ü绲绘慨?闂?indivisibleNodeIndex
                                                             (k >= 1闂備焦瀵х粙鎴﹀嫉椤掑倵鍋撻崹顐ょ煂闁圭懓瀚埞鎴﹀醇椤愩倐鍋撻弻銉︾厱婵°倐鍋撴繛灞傚€楅懞?
        濠电偞鍨堕幐鍝ョ矓閻戣棄绀冨┑鐘宠壘缁秹鏌涢幘鑼妽闁?ADJ_CLUSTER_INVALID 濠电娀娼чˇ浠嬪磻閸曨垰鍌ㄩ柡宥庡幖杩?    婵犵數鍋為崹鐢告偋閹版澘鍨傞柛灞剧矌閸楁碍銇勯弽銊ㄥ闁诲繑鐓￠弻锟犲醇濠垫劖效闂佸搫鎳岄崕鐢稿极?value < GlobalInfoBuffer[0].indivisibleNodeCount_G
                  闂備焦瀵х粙鎴︽偋閸℃稑绠洪柣妯款嚙缁€?0xFFFFFFFF 闂?>= 闂備胶鍘ч幖顐﹀磹婵犳艾纾婚柨婵嗩槸缁犳垹绱撴担鑲℃垵鈻撻崼鏇熺叆婵炴垶顭堢€氫即鏌曢崶褜妯€闁诡喚鍋ら獮鍡涙偋閸繂濮嶉梻渚€娼荤拹鐔煎礉瀹ュ憘娑㈠Χ婢跺﹥鐎梺鍓插亝缁诲倿姊惧鈧弻?    婵犵數鍋為崹鐢告偋閹版澘鍨傞柛灞剧矌閸楁碍銇勯弽銊ㄥ閻犱焦褰冮妴鎺戭潩椤撶姷鐣剧紓浣稿€稿Λ婵嬪极?k == 0 闂備浇顕栭崢鐣屾暜婵犲嫭瀚婚柣鏂垮悑閸ゅ﹥銇勮箛鎾村枠闁?                  => 缂傚倷绀侀張顒€顪冮挊澹╂盯骞愭惔锛勭獮閻庡箍鍎卞ú锕傚汲韫囨稒鈷?闂備礁鎼張顒勫礉閹烘梻鍗氶柟缁㈠枛缁狙囨煥濞戞ê顏柣锝変憾閺岋綁顢曢悢铚傚闂備浇鍋愰、濠傤嚕閹捐鐤鹃柛顐ｆ礀閸欏﹥銇勯弽銊︾殤缂佸銈搁弻娑㈠箻瀹曞泦銈夋煟?闂佽绻愮换妤佺椤掑嫭鍋ら柟瀵稿仧椤╃兘鏌у顒€鈧鎳?(i == 0)

    缂傚倷鐒﹀畷妯衡枖閺囥垹鐓濆┑鍌滎焾閸欏﹪鏌ｅΟ鍨毢婵炲牊鎮傞弻?1 濠电偞鍨堕幖鈺傜閻愬搫鍚归幖杈剧悼閻?= 1 濠电偞鍨堕幖鈺傜濠婂牆鍚归柟瀵稿仦婵挳鏌￠崶鈺佇ユ繛?label = threadIndex)闂備焦瀵х粙鎴︽儔婵傜鍚归幖杈剧悼閻瑩鎮楅敐搴濈盎闁轰浇娉涢埥澶愬箻閻熸媽鏁垮銈冨灪閸ㄥ潡鐛澶樻晝闁挎繂鎳嶇槐鐔兼⒑閸︻収鏆柛瀣崌閺岋繝宕煎┑鎰﹂梺閫炲苯澧柣鐔叉櫊閸┾偓妞ゆ巻鍋撴い锔诲弮閸┾偓?              dispatch 缂傚倸鍊风粈浣衡偓姘煎弮瀵?= ceil(IndivisibleNodeCount_G_VRPG / 1024) = 2闂備焦瀵х粙鎴︽嚐椤栨氨鏆ら柛鈩冪☉閻銇勯弽銊х婵炲樊鍓熼弻?              N_G <= 2000 闂備焦鐪归崝宀€鈧凹鍓熷畷锝堢疀濞戞瑧鍔搁梺闈涱煭缁犳垿宕滈幍顔藉枑闂侇叏闄勭欢娑氱磼鏉堛劎鎳囩€殿噮鍣ｉ幃鈺冪磼濮楀牏鍚归梻?dispatch闂備線娼уΛ鏃堟倿閿曗偓铻為柕鍫濇川绾惧ジ鏌涢弴銊ヤ簻闁?barrier闂備焦瀵х粙鎴﹀嫉椤掆偓铻炴い鎾卞灪閸婇鎲搁悧鍫濆潑闁稿鎹囬崺鈧い鎺戝€甸崑鎾绘偡闁附笑闂佸憡鐟ョ换姗€寮鍡楃窞闁割偅鑹炬禍?
    濠电偞鎸荤喊宥囨崲閸℃瑧鐭夐柛鈩兦滄禍婊堟煕濠靛棗顏痪鏉跨Ч閺?闂傚鍋勫ú銈夊箠濮椻偓婵＄绠涘☉妯诲祶?getOctreeLabel 濠电偞鍨堕弻銊╊敄閸涙潙绠栨俊銈呮噺閺咁剟鏌涢敂鑺ュ礋divisibleNodeInfoBuffer_G 闁诲海鎳撻幉陇銇愰崘顔肩柈闁规儳顕梽鍕煛閸屾ê鈧绮堥崱娑欑叆?              濠电偞鍨堕幐璇参涙笟鈧妴鍛存晝閸屾艾鐝樻繝銏ｆ硾椤﹂亶顢旈柆宥嗙厸?compute -> compute 闂?barrier闂?
    闂備浇顕栭崕鎼佸幢濡も偓楠炲鏌ｉ悩鍙夋悙闁宦板妼鍗遍柟杈鹃檮閺?闂備礁鎲￠崝鏇㈠箠鎼淬劍鍎戦柡澶嬵儥閸ゆ銇勮箛鎾搭棞缂佺姵宀搁弻?j 闂佽绨肩徊濠氾綖婢舵劕姹查柨婵嗘川绾惧ジ鏌涢弴銊ュ箹婵炲牏濮风槐鎺斺偓锝庝憾閸庡海绱掗弮鈧幐鎶藉箠?=> IndivisibleNodeInfoBuffer_G[j] /
              OctreeNodeDataBuffer_G[layer][nodeIndex] 闂?缂傚倸鍊风粈浣衡偓姘煎墴瀹曨剟骞橀鑹靶曢柣蹇曞仩婵倝宕熸ィ鍐╃厱鐟滃宕?闂?broadcast load闂?              闂佽楠稿﹢閬嶅磻閵堝拋鐎舵い鏍ㄧ矋婵ジ鏌涢…鎴濇灓闁圭晫濮风槐?N_G * (N_G / 1024) * 48B闂備焦瀵х粙鎴濓耿椤﹀尀=2000 闂備礁鎼崯顐ｆ櫠鎼达絺鍋?190KB闂備焦瀵х粙鎴βㄩ埀顒傜磼鏉堛劎绠炵€规洩缍侀、娑橆潩椤撶儐浼撻梻浣风串缁插宕曢懡銈嗩潟?              top-K 闂?[unroll] 闂傚倸鍊搁悧濠囨偡閿曞倸鐒垫い鎴ｅ劵閸忓矂鏌涚仦璇插妞?=> 闂備浇妗ㄩ悞锔界珶閸℃瑥鍨濈€广儱娲﹂弳婊堟煙闂傜顔夐柡浣告閹妫冨☉姗嗕患闂佸鏉垮鐎规洘濞婇垾锕傚箳閺冨偆妲峰┑鐐村灦閹稿摜绮斿畷鍥潟濞寸厧鐡ㄦ刊?local memory闂?*/
//---------------------------------------- 闂備礁鎲￠悷顖炲垂閻㈠憡鍎嶉柣鏃傚帶閻鏌涚仦鍓р姇婵?----------------------------------------
#define ADJ_CLUSTER_COUNT           Candidate_Samples_Count     // 婵犳鍣徊鐣屾崲鐎ｎ喗鐓傛繝濠傜墕缁€鍕煟濡じ鍚紒澶庢硶缁辨帡濡搁妷锔剧厬缂備胶绮銊╁焵椤掍胶鈯曟い顓炴搐閳绘捇骞嬮敃鈧痪褔鏌ㄥ☉妯侯仾缂佹劖顨婇弻?32)闂? 濠电偞鍨堕幖鈺傜閿濆鍚规繝濠傜墢閸?+ 31 濠电偞鍨堕幖鈺傜椤忓牆纾婚柨鏂垮綖濞?#define ADJ_CLUSTER_NORMAL_COS_MIN  0.7f                        // 婵°倗濮烽崑娑㈠疮閸噮鐒介幖绮规閺嬫牠鏌￠崶鈺佇㈡繛鍫㈠█閺岋綁鍩￠崨顔界彸闂佺懓鐏堥崑鎾绘煙閼圭増褰х紒鐘虫尭铻炴い鎺戝閳?1=闂佽娴烽幊鎾绘嚐椤栫偛鐭楅煫鍥ㄧ☉鐟欙箓鎮橀悙璺盒撻柛?
#define ADJ_CLUSTER_INVALID         0xFFFFFFFFu                 // 缂傚倷绀侀惉濂稿磿閵堝牊缍嶉梻浣告啞缁哄潡宕濆畝鍕煑?
// 闂佽崵濮烽崕銈囨崲閸曨垪鈧牠宕堕妸锕€鍔呴梺鎸庢礀閸婄鈻嶉妶澶嬬叆婵炴垶鐟ラ崫铏圭磼椤曞懎鐏ｉ柟宄邦儔婵″爼宕堕妸锔筋吂"闂備胶鍎甸弲婊呮暜閹烘鍋ゆ繛鍡樻尰閳锋帡鏌熺紒銏犳灍妞?(AABBDistance, 0 闂佽崵鍋炵粙蹇涘礉鎼淬劌桅婵鍩栭崕搴ㄦ煕濞戞﹫宸ュ?闁诲氦顫夐幐鎼佹嚐椤栫偑鈧? 闂?NaN 濠碉紕鍋涢鍛存煀閿濆應鏋?闂?// 闂備浇顕栭崹闈涒枍閺囥垹鏋佸┑鍌滎焾缁?闂備礁鎼悧濠勬崲閸岀偛绠栨繝濠傜墛椤ュ牊銇勯顐㈠箻妞ゆ挻妞藉濠氬磼濮樼厧娈跺┑鐐茬墛閸ㄥ綊骞戦崟顐熸斀闁割偒鍋嗛埀?闂備焦瀵х粙鎴﹀嫉椤掍椒绻嗛柛婵勫劗閸嬫捇鎮烽弶鎸庢閻庤娲橀悡锟犵嵁?AABBDistance2(a, b) 闂備礁鎲￠〃鍛村疮閼愁垬浜归柡灞诲劚杩?#define ADJ_CLUSTER_DISTANCE(a, b)  AABBDistance(a, b)
//------------------------------------------------------------------------------------------
[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_getAdjClusterG(uint threadIndex: SV_DispatchThreadID) {
    uint indivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;

    uint nodeLabel = threadIndex;
    if (nodeLabel >= indivisibleNodeCount_G) return;    // N_G 闂備礁鎼€氱兘宕归悽鍨床鐎广儱娲﹂崑姗€鎮橀悙闈涗壕妞は佸洦鐓熼柕濞垮劚椤忕數绱掓潏銊х畼闁瑰ジ绠栧畷銊╊敇閻愬妲橀梻浣告啞椤ㄥ棙绻涙繝鍥х劦妞ゆ垼妫勬禍楣冩⒑?
    //-------------------------------------------------- 闂備胶鍘ч〃搴㈢閿濆顥?    uint2 myInfo = IndivisibleNodeInfoBuffer_G[nodeLabel];              // x = layer, y = nodeIndex
    OctreeNodeData_G_VRPG me = OctreeNodeDataBuffer_G[myInfo.x][myInfo.y];

    float  myNormalLength = length(me.meanNormal.xyz);
    bool   myNormalValid = myNormalLength > 1e-8f;
    float3 myNormal = myNormalValid ? me.meanNormal.xyz / myNormalLength : float3(0.0f, 0.0f, 1.0f);

    //-------------------------------------------------- top-K 闂備礁鎲＄敮妤冩崲閸岀儑缍栭柟鐗堟緲缁€?闂佽崵濮烽崕銈囨崲閸曨垪鈧牠宕堕鈧涵鈧梺鎼炲劀閸涱垼鏆?
    // 婵?0 闂備焦鎮堕崕閬嶅箹椤愶附鍋╁Δ锝呭暞閸嬧晜绻涢崱妯虹仼闁肩鐖奸弻銈嗙附婢跺鎹ｉ梻鍌毰堥弲鐘诲极瀹ュ閱囬柣鏃€鍎奸崺鍛存⒑闂堚晞绀嬮柛鏂块叄椤㈡牠宕堕妸褉鏋栭梺閫炲苯澧柟宄版噹椤撹壈绠涢弴鐙€妲峰┑鐐村灦閹稿摜绮旈弶鎴殨妞ゆ帊鑳堕埢鏃堝箹鐎涙绠橀柣鐔告崌濮婂宕煎☉妯间紕濠电偛鐗婇崹鍦垝閳哄懏鍊烽柟娈垮枟鐏忔劙姊洪幐搴ｂ槈濠靛倹姊诲褔宕橀埡鍐ｆ灃閻庡箍鍎遍幊搴ｇ礊閳ь剙鈹戦悙宸Ч婵炲弶绮撳畷鍨償閵婏腹鎷婚梺闈涱煭缁犳垶绂?闂佽崵濮烽崕銈囨崲閸曨垪鈧牠宕堕浣叉嫽闂侀潧艌閺呮粓顢撳▎鎴炲枑闁哄鐏濋懜褰掓煙閼测晛鍘撮柡浣哥У瀵板嫬鈽夊顒冣偓?    // 婵?1 .. ADJ_CLUSTER_COUNT-1 闂備礁缍婂褏绮旈棃娴?闂備胶顭堢换妤€鈻斿☉銏″仼闁绘ü绲绘慨?闂備焦鐪归崝宀€鈧凹鍣ｉ幃鐢稿垂椤愩値娲告繝銏ｆ硾椤﹁京绮旈幘顔界厪?    float bestDistance[ADJ_CLUSTER_COUNT];
    uint  bestLabel[ADJ_CLUSTER_COUNT];
    bestDistance[0] = 0.0f;                             // 闂備胶鍘ч〃搴㈢閿濆顥婇柍鍝勫暞瀹曞绻濋棃娑氬⒈闁诲孩鐓￠弻?0 濠电姰鍨煎▔娑氣偓姘煎櫍楠?    bestLabel[0] = nodeLabel;                           // 闂?濠电儑绲藉ú锔炬崲閸℃稒鍎婂鑸靛姈閸ゅ﹥銇勮箛鎾村枠闁绘挴鏅犻弻娑㈠箣閿濆棭浠╁Δ鐘靛仜濞差厼鐣烽悜钘壩╃憸婊堝磿閹存績妲堥柟缁㈠灠娴滅偓绻涢幋鐐存儎濞存粍鐟х槐鐐哄川椤栨粎锛?    [unroll]
    for (int k = 1; k < ADJ_CLUSTER_COUNT; ++k) {
        bestDistance[k] = 3.402823466e+38F;             // FLT_MAX
        bestLabel[k] = ADJ_CLUSTER_INVALID;
    }
    float worstDistance = 3.402823466e+38F;

    //-------------------------------------------------- 闂備浇顫夐鏍磻閸涱収鍤曢柛褎顨呯粻銉╂煃瑜滈崜鐔奉嚕閸偄绶炲┑鐘插€哥欢顓㈡煟鎼达絾鍤€闁哥噥鍋婂畷鐢稿箣閻愭潙顎撻梺鍝勬川婵嘲鈻?    for (uint j = 0; j < indivisibleNodeCount_G; ++j) {
        // 闂佸搫顦弲婊堟偡閵堝洦鍋栭弶鍫涘妿閳?load 闂佽閰ｅ褍锕㈡潏鈺佸灊鐎广儱顦粈鍐煕濞戝崬鏋涙い锝嗙叀閺岋繝宕煎┑鎰ㄩ梺鎼炲€曞ù椋庡垝濞嗘垟鍋撻敐搴′簼缁楁垿姊洪崨濠呭濠靛倹姊荤划鈺呭箻椤斿吋宓嶉梺鍝勵槹鐎笛呭緤?j 濠电偞鍨堕幐鎾磻閹剧粯鐓? => 缂備胶铏庨崣搴ㄥ礂濞戞ǚ鏋庨柕蹇嬪€楀畵渚€鏌熺粙鍨劉閻忓骏绻濋弻銊モ槈濡灝顏紓浣介哺閸ㄥ潡寮婚妸鈺佺劦妞ゆ帒瀚惌妤併亜閺嶃劎顣查悗姘秺閺?        uint2 otherInfo = IndivisibleNodeInfoBuffer_G[j];
        OctreeNodeData_G_VRPG other = OctreeNodeDataBuffer_G[otherInfo.x][otherInfo.y];

        if (j == nodeLabel) continue;                                   // 闂備胶鍘ч〃搴㈢閿濆顥婇柍杞扮贰閸熷懘鏌熺紒妯虹瑲缂佷礁銈搁幃妤冩喆閸曨喖顥濋悗瑙勬磻閸楁娊骞冩禒瀣╅敎?0闂備焦瀵х粙鎴﹀嫉椤掍胶鐝堕柛宀€鍋涚粻鐢哥叓閸ャ劍鈷愭い蟻鍥ㄧ厵闁绘垶锚濞搭喗绻涢崨顓㈠弰闁轰礁绉撮悾婵嬪礋椤忓棛绀嗛梻浣侯焾缁绘劗绮旇ぐ鎺戦棷闁告稑锕ょ欢?        if (other.meanNormal.w <= 0.0f) continue;                       // 闂備礁鎼崯鐗堟叏閻㈢姹查柣鏃傚帶缁?hasData)闂備胶顭堢换鎴犲垝瀹€鈧懞?
        if (myNormalValid) {                                            // 婵犵數鍋涢ˇ鏉棵哄鍫濆惞閹兼番鍔岀粈鍌炴煟閹惧啿鐦ㄩ柛鐔锋喘閺岀喖顢栭鈧ú锕傛晬瀹€鍕厵?            float  otherNormalLength = length(other.meanNormal.xyz);
            float3 otherNormal = otherNormalLength > 1e-8f ? other.meanNormal.xyz / otherNormalLength
                                                          : float3(0.0f, 0.0f, 1.0f);
            if (dot(myNormal, otherNormal) < ADJ_CLUSTER_NORMAL_COS_MIN) continue;

            // 闂備礁鎲￠悷顖炲垂閸洖鐒垫い鎺嗗亾妞ぱ€鍋撶紓浣稿綁閸楀啿顕ｉ妸褎鍠嗛柛鏇ㄥ亞娴?闂備礁鎲￠懝楣冩儗娓氣偓閺屽﹪鏌嗗鍛靶曢梺鐟邦嚟閸嬬喖濡存繝鍕＜?闂佸搫顦弲婊堟偡瑜旈敐鐐哄箳濡も偓閸楀磭绱撻崼銏犘㈢紒槌栧墴閺屸€愁吋閸繀姹楀┑鐐茬墛閸ㄥ灝顫忔繝姘╅柨鏇楀亾濞寸姵锕㈤弻銊モ槈濡厧顣圭紓浣诡殔閻倸螞?RIS 闂?NEARBYNODE 濠电偞鍨堕幐鎾磻閹剧粯鐓犻柣銈庡灡瑜把呯磼?            // uint myAxis    = myInfo.y    / (1u << (3u * myInfo.x));
            // uint otherAxis = otherInfo.y / (1u << (3u * otherInfo.x));
            // if (myAxis != otherAxis) continue;
        }

        float distance = ADJ_CLUSTER_DISTANCE(me.aabb, other.aabb);
        if (!(distance < worstDistance)) continue;                      // 婵犳鍣徊楣冨蓟閵娧呯閻庯綆鍠栫粈鍫⑩偓骞垮劚濞村倿宕?K 闂備礁鎲￠懝鍓х矓妞嬪孩宕查柡鍥╁枑娴?闂備礁鎲￠懝楣冨嫉椤掆偓椤啴宕掗悙鑼唶?NaN/闂備胶鍎甸弲娑㈠箺濠婂牊鍋?

        // 闂備礁鎲￠崝鏍暜濡ゅ懎鎹舵繛宸簻缁犵敻鏌熼悜妯虹仴鐎殿喗濞婇弻銊モ槈濞嗘帒顥濋悷婊勫Ω閸パ勫祶闂侀潧脿閳?1..K-1 闂備礁鎲￠崝鏇㈠箠瀹ュ洨绠旈柛娑卞幐閺岋箑顭块懜鐢点€掔紒鈧径宀€纾?K-1 闂備礁鎲￠懝鍓х矓閻戞枻缍栭柨鏇炲€哥粻浼存煏閸繃鍣规慨锝咁樀閺屻劌鈽夊Ο鐓庮杸闁?0 闂備礁鎼€氱兘宕归悽绋垮惞婵犲﹤鐗忛崣鏇熴亜閹伴潧澧扮紒鈧径宀€纾藉ù锝呮贡缁犳垹绱掑Δ鈧ˇ鐢稿箠濡ゅ懎纾归柣鏇氱劍閽戝姊虹憴鍕仩闁稿﹥鐗滈弫顕€鍩€椤掑嫭鐓?        float carryDistance = distance;
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

    //-------------------------------------------------- 闂備礁鎲￠崝鏍偡閵夈儍?婵?0 = 闂備胶鍘ч〃搴㈢閿濆顥婇柍鍝勬噺閺咁剟鎮橀悙鑸殿棄鐎电増鎸搁湁闁挎繂妫涢幊鍐煙椤栨氨澧︾€殿噮鍣ｉ幃鈺呮惞椤愶絿褰欐繝寰枫倕钄兼俊顐㈠閺佹劙鎮欓崫鍕槯?
    uint outBase = nodeLabel * (uint)ADJ_CLUSTER_COUNT;
    [unroll]
    for (int k = 0; k < ADJ_CLUSTER_COUNT; ++k) {
        AdjClusterInfoBuffer[outBase + k] = bestLabel[k];
    }
    // 闂備礁鎲￠悷顖炲垂閸洖鐒垫い鎺嗗亾妞ぱ€鍋撶紓浣戒含婢ф绌辨繝鍥х＜婵炴垶鐭崙鐣岀磽娴ｅ憡婀版俊鐐舵铻為柟绋块缁剁偟鈧箍鍎卞ú锕傚汲韫囨稒鐓欐い鏍ㄧ箖閻濐亪鎮?闂佸搫顦弲婊呭垝韫囨洘宕?闂備礁鎲″缁樻叏闂堟侗鐒介柛顐犲劜閺咁剟鎮橀悙鑸殿棄閻犱焦褰冮湁闁绘﹩鍠栭崝姘舵煕閻旀彃浜鹃柣搴㈩問閸犳帡宕戦幘鏂ユ闁圭虎鍨版禍鐐箾?float 缂傚倸鍊搁崐褰掑箰閹间礁鐤柛褎顨呯粻?bestDistance[k] 濠电偞鍨堕幐鎾磻閹炬惌娈介柣鎰煐绾箖鏌涢悢鍛婂磳鐎?    //       闂備焦瀵х粙鎴︽偋婵犲洤鏋侀柣鎰惈缁?bestDistance[0] 闂備浇顕栭崢鐣屾暜婵犲嫭瀚?0闂備焦瀵х粙鎴︽嚐椤栨氨鏆ら柛鈩冪⊕閸ゅ﹥銇勮箛鎾村枠闁绘挴鏅犻弻锝夊Ω閵夈儺浼傜紓浣瑰姈缁嬫挾鍒掔€ｎ偄顕辨繛鍡樺灩椤?}
#endif