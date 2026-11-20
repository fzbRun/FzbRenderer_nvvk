#ifndef FZBRENDERER_VRPG_2_SHADER_IO_H
#define FZBRENDERER_VRPG_2_SHADER_IO_H

#include <common/Shader/shaderStructType.h>
NAMESPACE_SHADERIO_BEGIN()

//------------------------------------------------------------
#define MAX_OCTREE_LAYER_VRPG2 8
#define ClusterMaxCount_VRPG 4000

#define OCTREE_CLUSTER_TOP_LAYER_VRPG 3
static const uint OctreeLayerNodeCount_VRPG2[MAX_OCTREE_LAYER_VRPG2] = { 6, 48, 384, 3072, 24576, 196608, 1572864 };
static const uint OctreeLayerStartIndex_VRPG2[4] = { 0, 8, 56, 384 };

#define VRPG_NEARBY_CLUSTER_COUNT 4
#ifndef NEARBY_NODE_COUNT_VRPG
#define NEARBY_NODE_COUNT_VRPG 4
#endif		//don't change
// Neighbour jitter across G clusters (see GetAdjClusterG.slang / VRPGGuide.slang):
//   0 (default) : the original hard requirement that a neighbour lives in the same normal partition
//   1           : cosine + coherence weighting, so the jitter can cross normal partition boundaries
//                 on curved (circle/sphere) and flat surfaces while creases are still refused
//                 (the length of meanNormal is the consistency of the averaged normal)
#ifndef VRPG_COSINE_NEIGHBOR
#define VRPG_COSINE_NEIGHBOR 1
#endif
#define VRPG_COSINE_MIN_DOT 0.70f    // cos(45 deg): reject candidates facing away more than that
#define VRPG_COSINE_POWER   2.0f     // sharpness of the cosine weight
//------------------------------------------------------------
#define Candidate_Samples_Count 64
#ifndef VRPG_Candidate_Samples_SampleMode
#define VRPG_Candidate_Samples_SampleMode 1

#if VRPG_Candidate_Samples_SampleMode == 1
#undef Candidate_Samples_Count
//don't change
#define Candidate_Samples_Count 32
#endif

#endif

#define HITTEST_COUNT_PER_CHILDNODE_VRPG 4		//must <= 4

#define VRPG_SPATIAL_REUSE 0
#define VRPG_TEMPORAL_REUSE 0

//#define SpatialReuse_VRPG
#define Pairwise_MIS
//------------------------------------------------------------
// Compile-time VRPG switches (change requires rebuilding the app).
#define VRPG_RANDOM_FRAME_SOURCE 1
#ifndef VRPG_PATH_SAMPLE_MODE
// How the BSDF and the guide techniques are combined (see computeMain_VRPG):
	//   0 (default) two sample balance MIS: both rays traced, one sample chosen for the
	//     continuation, every weighted term divided by the selection probability
	//   1 one sample 50/50: only the chosen technique traced, its terms divided by q
#define VRPG_PATH_SAMPLE_MODE 1
#endif
//------------------------------------------------------------------------------------------
struct OctreeNodeInfo_VRPG2 {
	int layerIndex;
	int nodeIndex;
};

struct OctreeLayerInfo_VRPG2 {
	uint32_t divisibleNodeCount;
	uint32_t indivisibleNodeCount;
};

struct GlobalInfo_VRPG2 {
	DispatchIndirectCommand cmd;
	uint ClusterCount;
	OctreeLayerInfo_VRPG2 layerInfos_G[MAX_OCTREE_LAYER_VRPG2];
};

NAMESPACE_SHADERIO_END()
#endif
