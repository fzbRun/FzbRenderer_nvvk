#pragma once

#include "feature/Feature.h"
#include "./OctreeShaderio_VRPG.h"
#include <renderer/PathTracingRenderer/hard/AccelerationStructure.h>

#ifndef FZBRENDERER_OCTREE_2_VRPG_H
#define FZBRENDERER_OCTREE_2_VRPG_H
namespace FzbRenderer {
struct OctreeCreateInfo_VRPG {
	std::vector<nvvk::Buffer> VGBs;       // geometry grid
	shaderio::float3 VGBStartPos;
	shaderio::float3 VGBVoxelSize;
	float VGBSize;

	std::vector<nvvk::Buffer> lightVGBs;  // light grid, independent resolution
	shaderio::float3 lightVGBStartPos;
	shaderio::float3 lightVGBVoxelSize;
	float lightVGBSize;

	AccelerationStructureManager* asManager;
};

class Octree_VRPG : public Feature {
public:
	Octree_VRPG() = default;
	virtual ~Octree_VRPG() = default;

	Octree_VRPG(pugi::xml_node& featureNode);

	void init(OctreeCreateInfo_VRPG createInfo);
	void clean();
	void uiRender();
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender();
	void render(VkCommandBuffer cmd);
	void postProcess(VkCommandBuffer cmd);

	void createOctreeArray();
	void createDescriptorSetLayout() override;
	void createDescriptorSet();
	void createPipeline();
	void compileAndCreateShaders();
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	void initOctreeArray(VkCommandBuffer cmd);
	void createOctreeArray(VkCommandBuffer cmd);
	void createLightTree(VkCommandBuffer cmd);
	void getOctreeLabel(VkCommandBuffer cmd);
	void getOctreeNodePairData(VkCommandBuffer cmd);
	void getNearbyClusterInfo(VkCommandBuffer cmd);

	shaderio::OctreePushConstant_VRPG pushConstant{};

	uint32_t octreeMaxLayer = 6;
	uint32_t lightOctreeMaxLayer = 4;

	float clusterNormalThreshold = 0.707f;
	float clusterFillRateThreshold = 0.75f;
	float clusterVolumeTolerance = 0.3f;
	bool lightBackfaceCull = true;
	bool stableCandidateCut = false;

	std::vector<nvvk::Buffer> octreeNodeDataBuffer_G;
	std::vector<nvvk::Buffer> octreeNodeInfoBuffer_G;			//label_indivisible,  label: indivisible node闂傚倸鍊搁崐鎼佸磹妞嬪孩顐芥慨姗嗗墻閻掍粙鏌ゆ慨鎰偓鏍偓姘煼閺岋綁寮崒姘粯缂備讲鍋撳璺哄閸嬫捇宕楁径濠佸闂備線鈧偛鑻晶鐗堜繆閸欏濮嶉柟顔界懅閸犲﹤螣鏉炴澘顥氶梻浣瑰缁诲倹顨ラ幖浣稿嚑婵炴垶姘ㄧ壕濂告煃瑜滈崜鐔风暦婵傜鐓戦柍杞版婢规洖螖閻橀潧浠滈柣蹇旂箞瀹曟繂顫濋懜鐢靛幍闂佹儳娴氶崑鍕叏閸ヮ剚鐓ラ柡鍥悘鍙夘殽閻愯揪鑰挎い銏＄懇閹墽浠﹂挊澶岊唶闂傚倸鍊风粈渚€骞夐垾瓒佹椽鎮㈡總澶屽姺閻熸粍妫冮獮鍐潨閳ь剟寮幇鏉垮耿闁归偊浜為惄搴ㄦ⒒娴ｇ儤鍤€闁宦板妿閹广垽宕橀鑲╃崶濠碘槅鍨伴惃鐑藉磻閹捐埖鍠嗛柛鏇ㄥ墰椤︺劑姊洪懡銈呮毐闁哄懐濞€閻涱噣宕橀鑲╁幐闂佸憡渚楅崰姘跺礉閸洘鍊垫鐐茬仢閸旀岸鏌熼崘鑼闁诡喗瀵х粚閬嶆儍閳х惄ivisibleNode闂傚倸鍊搁崐鎼佸磹閻戣姤鍊块柨鏃堟暜閸嬫挾绮☉妯诲櫧闁活厽鐟╅弻鐔衡偓鐢殿焾娴犙囨⒒閸曨偄顏柡宀嬬節瀹曟﹢濡搁妷銏犱壕闁归棿绀侀崹?闂傚倸鍊峰ù鍥敋瑜忛埀顒佺▓閺呯娀銆佸▎鎾冲唨妞ゆ挾鍋熼悰銉╂⒑閸︻厼鍔嬫い銊ユ噽婢规洘绻濆顓犲幍闂佺粯鍔﹂崜姘舵倶闁秵鐓涢柍褜鍓熼幊鐐哄Ψ閿濆嫮鐩庨梻浣告惈閸燁偊宕愰悽绋跨闁跨喓濮甸悡鏇㈠箹濞ｎ剙鐏柣顓炵灱缁辨帗娼忛妸銉х懖濠电偟鍘х换姗€宕洪敓鐘茬＜婵犲﹤鎳忓鏉梔ivisible闂傚倸鍊搁崐鎼佸磹閻戣姤鍊块柨鏃堟暜閸嬫挾绮☉妯诲櫧闁活厽鐟╅弻鐔兼倻濮楀棙鐣烽梺鎼炲€曢懟顖濈亙闂佹寧绻傞幊搴ㄥ汲濞嗘挻鐓曢柕濞垮劚閳ь剚鎮傚﹢渚€姊虹紒妯忣亜螣婵犲洤纾块煫鍥ㄦ煣缁诲棙銇勯幇鍓佺Ф闁兼澘娼￠弻鈥崇暆鐎ｎ剛袦闂佽鍠掗弲娑㈠煝閹捐绠伴幖绮光偓鏂ュ彠de闂傚倸鍊搁崐鎼佸磹妞嬪海鐭嗗〒姘ｅ亾妤犵偞鐗犻、鏇㈠Χ閸屾矮澹曞┑顔矫畷顒勫储鐎电硶鍋撶憴鍕缂傚秴锕妴浣糕槈濮楀棛鍙嗛梺鍛婁緱閸ㄦ娊寮抽妷銉㈡斀闁绘﹩鍠栭悘閬嶆煕閳哄倻澧电€规洘绻堝鎾閻樼數鍘┑鐘垫暩婵潙煤閿曞倹鍋傞柣鏂垮悑閻撳繐顭跨拠鈥崇仩闁哥喓澧楃粋宥夊级娑旂晬isible闂?
	std::vector<nvvk::Buffer> octreeNodeDataBuffer_E;

	nvvk::Buffer globalInfoBuffer;
	nvvk::Buffer indivisibleNodeInfosBuffer_G;					//闂傚倸鍊搁崐鎼佸磹妞嬪海鐭嗗ù锝夋交閼板潡姊洪鈧粔鏌ュ焵椤掆偓閸婂湱绮嬮幒鏂哄亾閿濆簼绨介柨娑欑洴濮婃椽鎮烽弶搴撴寖缂備緡鍣崹鍫曞箖閳ユ枼妲堥柕蹇ョ磿閸樺崬顪冮妶鍡楀Ё缂佸弶瀵ф穱濠囧炊椤掍胶鍘撻悷婊勭矒瀹曟粌顫濋鈺傜☉椤顬婄€典巩sibleNode闂傚倸鍊搁崐鎼佸磹妞嬪海鐭嗗ù锝夋交閼板潡姊洪鈧粔顕€鍩€椤掑﹦鐣电€规洖銈告俊椋庘偓锝庝簼閸犳﹢鏌＄仦鑺ヮ棞妞ゆ挸銈稿畷銊╊敍濠婂啫绠戦梻浣筋嚙濞寸兘骞婇幘璇茬疅闁斥晛鍟伴々鏌ユ煟鎼绰板仱l闂傚倸鍊搁崐鎼佸磹妞嬪海鐭嗗ù锝夋交閼板潡姊洪鈧粔瀵稿閸ф鐓忛柛顐ｇ箖婢跺嫰鏌￠崱妯肩煉闁哄瞼鍠栭幃婊兾熷ú缁樷挅濠电姵顔栭崰娑樷枍閺囥垹桅闁告洦鍨扮粻濠氭煕濡ゅ啫浠уù鐘荤畺濮婃椽宕崟顒€顎涢梺鍛婃尵閸犳牠鐛崘顭戞建闁逞屽墴楠炲啫鈻庨幘鍐茬€銈嗗姧缁茶儻銇愯濮婄粯鎷呴崨濠呯闂佸搫鎷嬮崑鍛崲濞戙垹鐒垫い鎺戝閻撴稓鈧厜鍋撻悗锝庡墰閻﹀牓鎮楃憴鍕闁告鍥х厴闁硅揪绠戠粈瀣煏婵炲灝鍔欏瑙勬礀閳规垿顢欑涵閿嬫暰濡炪倖宸婚弳?闂傚倸鍊搁崐鎼佸磹閻戣姤鍊块柨鏃堟暜閸嬫挾绮☉妯诲櫧闁活厽鐟╅弻鐔兼倻濮楀棙顓归梺杞扮濡瑩濡甸崟顖氱厸濞达絽婀辫ぐ鍗恟Index + nodeIndex

	nvvk::Buffer adjClusterInfoBuffer;

	nvvk::Buffer clusterPairInfoBuffer;
	nvvk::Buffer clusterPairGlobalInfoBuffer;
	nvvk::Buffer clusterPairHitInfoBuffer;

	nvvk::Buffer candidateSubTreeInfoBuffer;
	nvvk::Buffer candidateLeafNodeDataBuffer;

	nvvk::Buffer candidateSampleDataBuffer;
	nvvk::Buffer candidateSamplePrefixWSumBuffer;

	nvvk::Buffer candidateSampleIndexBuffer;
	nvvk::Buffer candidateSampleAABBBuffer;
	nvvk::Buffer candidateSampleData2Buffer;
	nvvk::Buffer candidateSampleDataHistoryBuffer;
	nvvk::Buffer spatialReservoirBuffer;
	nvvk::Buffer reservoirCurrentBuffer;
	nvvk::Buffer reservoirHistoryBuffer;

	nvvk::Buffer nearbyNodeTempInfoBuffer;
	nvvk::Buffer nearbyNodeInfoBuffer;

	bool spatialReuse = false;
	bool temporalReuse = false;
	bool temporalReset = false;
	uint32_t temporalFrameIndex = 0;
	float temporalBlend = 0.1f;
	bool ditherSample = true;
	bool rotateOutgoing = true;
private:
	OctreeCreateInfo_VRPG setting;

	nvvk::Buffer blockInfoBuffer_G;
	nvvk::Buffer blockInfoBuffer_E;
	nvvk::Buffer hasDataBlockIndexBuffer_G;
	nvvk::Buffer hasDataBlockIndexBuffer_E;
	nvvk::Buffer hasDataBlockCountBuffer;

	nvvk::Buffer divisibleNodeInfoBuffer_G;
	nvvk::Buffer threadGroupInfoBuffer;

	VkShaderEXT computeShader_initOctreeArray{};
	VkShaderEXT computeShader_initHasDataBlockInfo{};
	VkShaderEXT computeShader_getGlobalInfo{};
	VkShaderEXT computeShader_createOctreeArray{};
	VkShaderEXT computeShader_createOctreeArray2{};
	VkShaderEXT computeShader_initLightTreeLeaves{};
	VkShaderEXT computeShader_aggregateLightTree{};

	VkShaderEXT computeShader_getOctreeLabel1{};
	VkShaderEXT computeShader_getOctreeLabel2{};
	VkShaderEXT computeShader_getOctreeLabel3{};
	VkShaderEXT computeShader_getOctreeLabel4{};

	VkShaderEXT computeShader_initWeights{};
	VkShaderEXT computeShader_getCandidateNodes{};
	VkShaderEXT computeShader_dispatchGetCandidateLeafNodes{};
	VkShaderEXT computeShader_getCandidateLeafNodes{};
	VkShaderEXT computeShader_clusterPairHitTest{};
	VkShaderEXT computeShader_getCandidateNodeWeights{};;
	VkShaderEXT computeShader_buildSpatialReservoir{};
	VkShaderEXT computeShader_reuseReservoir{};
	VkShaderEXT computeShader_temporalReuse{};
	VkShaderEXT computeShader_createCandidatePrefixWSum{};;


	VkShaderEXT computeShader_getAdjClusterG{};

	VkShaderEXT computeShader_reservoirSample{};;
	VkShaderEXT computeShader_SpatialReuse{};;

	VkShaderEXT computeShader_dispatchGetNearbyClusterG{};
	VkShaderEXT computeShader_getNearbyClusterG1{};
	VkShaderEXT computeShader_getNearbyClusterG2{};

	VkBindDescriptorSetsInfo bindDescriptorSetsInfo;
	VkPushConstantsInfo pushInfo;

	enum class GBuffers_Octree_VRPG {
		eGeometryClusterResult,
		eLightClusterResult,
		eGeometrySVO,
		eClusterPairResult,
		eBufferCount,
	};

	shaderio::OctreeDebugPushConstant_VRPG debugPushConstant;
	void geometryTreeDebug(VkCommandBuffer cmd);
	void lightTreeDebug(VkCommandBuffer cmd);
	void geometrySVODebug(VkCommandBuffer cmd);
	VkShaderEXT vertexShader_TreeDebug{};
	VkShaderEXT fragmentShader_TreeDebug{};

	bool changeClusterG = true;
	nvvk::Buffer clusterPairDebugGlobalInfoBuffer;
	shaderio::ClusterPairDebugPushConstant_VRPG debugPushConstant2;
	void clusterPairDebug(VkCommandBuffer cmd);
	VkShaderEXT computeShader_GetDebugClusterG{};
	VkShaderEXT vertexShader_ClusterPairDebug{};
	VkShaderEXT fragmentShader_ClusterPairDebug{};
	bool showNearbyClusters = false;
	int debugClusterIndex = -1;
};
}
#endif
