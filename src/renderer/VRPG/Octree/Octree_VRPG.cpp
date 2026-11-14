#include "./Octree_VRPG.h"
#include <nvutils/timers.hpp>
#include <common/Application/Application.h>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <bit>
#include <nvvk/default_structs.hpp>
#include <nvgui/property_editor.hpp>
#include "../RasterVoxelization/RasterVoxelization_VRPG.h"
#include "feature/PathTracing/shaderio.h"
#include <cstdio>
#include <cstdlib>


using namespace FzbRenderer;

static_assert(sizeof(shaderio::OctreePushConstant_VRPG) <= 256, "Octree push constant exceeds 256 bytes");

Octree_VRPG::Octree_VRPG(pugi::xml_node& featureNode) {}

void Octree_VRPG::init(OctreeCreateInfo_VRPG createInfo) {
	this->setting = createInfo;

	createOctreeArray();
	createDescriptorSetLayout();
	createDescriptorSet();
	createPipeline();
	compileAndCreateShaders();

#ifndef NDEBUG
	Feature::createGBuffer(true, true, (uint32_t)GBuffers_Octree_VRPG::eBufferCount);
	{
		nvutils::PrimitiveMesh primitive = FzbRenderer::MeshSet::createWireframe();
		FzbRenderer::MeshSet mesh = FzbRenderer::MeshSet("Wireframe", primitive);
		scene.addMeshSet(mesh);

		primitive = FzbRenderer::MeshSet::createCube(false, false);
		mesh = FzbRenderer::MeshSet("Cube", primitive);
		scene.addMeshSet(mesh);

		scene.createSceneInfoBuffer();
	}
#endif
}
void Octree_VRPG::clean() {
	Feature::clean();
	for (int i = 0; i < octreeNodeDataBuffer_G.size(); ++i) Application::allocator.destroyBuffer(octreeNodeDataBuffer_G[i]);
	for (int i = 0; i < octreeNodeInfoBuffer_G.size(); ++i) Application::allocator.destroyBuffer(octreeNodeInfoBuffer_G[i]);
	for (int i = 0; i < octreeNodeDataBuffer_E.size(); ++i) Application::allocator.destroyBuffer(octreeNodeDataBuffer_E[i]);

	Application::allocator.destroyBuffer(globalInfoBuffer);
	Application::allocator.destroyBuffer(indivisibleNodeInfosBuffer_G);

	Application::allocator.destroyBuffer(divisibleNodeInfoBuffer_G);
	Application::allocator.destroyBuffer(threadGroupInfoBuffer);

	Application::allocator.destroyBuffer(adjClusterInfoBuffer);

	Application::allocator.destroyBuffer(blockInfoBuffer_G);
	Application::allocator.destroyBuffer(blockInfoBuffer_E);
	Application::allocator.destroyBuffer(hasDataBlockIndexBuffer_G);
	Application::allocator.destroyBuffer(hasDataBlockIndexBuffer_E);
	Application::allocator.destroyBuffer(hasDataBlockCountBuffer);

	Application::allocator.destroyBuffer(clusterPairInfoBuffer);
	Application::allocator.destroyBuffer(clusterPairGlobalInfoBuffer);
	Application::allocator.destroyBuffer(clusterPairHitInfoBuffer);

#if VRPG_Candidate_Samples_SampleMode == 1
	Application::allocator.destroyBuffer(candidatePoolInfoBuffer);
	Application::allocator.destroyBuffer(candidatePoolNodeInfoBuffer);
#endif

	Application::allocator.destroyBuffer(candidateSubTreeInfoBuffer);
	Application::allocator.destroyBuffer(candidateLeafNodeDataBuffer);

	Application::allocator.destroyBuffer(candidateSampleDataBuffer);
	Application::allocator.destroyBuffer(candidateSamplePrefixWSumBuffer);

	Application::allocator.destroyBuffer(candidateSampleIndexBuffer);
	Application::allocator.destroyBuffer(candidateSampleAABBBuffer);
	Application::allocator.destroyBuffer(candidateSampleData2Buffer);
	Application::allocator.destroyBuffer(candidateSampleDataHistoryBuffer);
	Application::allocator.destroyBuffer(spatialReservoirBuffer);
	Application::allocator.destroyBuffer(reservoirCurrentBuffer);
	Application::allocator.destroyBuffer(reservoirHistoryBuffer);

	Application::allocator.destroyBuffer(nearbyNodeTempInfoBuffer);
	Application::allocator.destroyBuffer(nearbyNodeInfoBuffer);

	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_initOctreeArray, nullptr);
	vkDestroyShaderEXT(device, computeShader_initHasDataBlockInfo, nullptr);
	vkDestroyShaderEXT(device, computeShader_getGlobalInfo, nullptr);
	vkDestroyShaderEXT(device, computeShader_createOctreeArray, nullptr);
	vkDestroyShaderEXT(device, computeShader_createOctreeArray2, nullptr);
	vkDestroyShaderEXT(device, computeShader_initLightTreeLeaves, nullptr);
	vkDestroyShaderEXT(device, computeShader_aggregateLightTree, nullptr);

	vkDestroyShaderEXT(device, computeShader_getOctreeLabel1, nullptr);
	vkDestroyShaderEXT(device, computeShader_getOctreeLabel2, nullptr);
	vkDestroyShaderEXT(device, computeShader_getOctreeLabel3, nullptr);
	vkDestroyShaderEXT(device, computeShader_getOctreeLabel4, nullptr);

	vkDestroyShaderEXT(device, computeShader_initWeights, nullptr);
#if VRPG_Candidate_Samples_SampleMode == 1
	vkDestroyShaderEXT(device, computeShader_getCandidateNodes_init, nullptr);
	vkDestroyShaderEXT(device, computeShader_getCandidateNodes_1, nullptr);
	vkDestroyShaderEXT(device, computeShader_getCandidateNodes_2, nullptr);
	vkDestroyShaderEXT(device, computeShader_getCandidateNodes_3, nullptr);
	vkDestroyShaderEXT(device, computeShader_getCandidateNodes_4, nullptr);
	vkDestroyShaderEXT(device, computeShader_getCandidateNodes_5, nullptr);
#endif
	vkDestroyShaderEXT(device, computeShader_getCandidateNodes, nullptr);
	vkDestroyShaderEXT(device, computeShader_dispatchGetCandidateLeafNodes, nullptr);
	vkDestroyShaderEXT(device, computeShader_getCandidateLeafNodes, nullptr);
	vkDestroyShaderEXT(device, computeShader_clusterPairHitTest, nullptr);
	vkDestroyShaderEXT(device, computeShader_getCandidateNodeWeights, nullptr);
	vkDestroyShaderEXT(device, computeShader_buildSpatialReservoir, nullptr);
	vkDestroyShaderEXT(device, computeShader_reuseReservoir, nullptr);
	vkDestroyShaderEXT(device, computeShader_temporalReuse, nullptr);
	vkDestroyShaderEXT(device, computeShader_createCandidatePrefixWSum, nullptr);

	vkDestroyShaderEXT(device, computeShader_getAdjClusterG, nullptr);
	
	vkDestroyShaderEXT(device, computeShader_reservoirSample, nullptr);
	vkDestroyShaderEXT(device, computeShader_SpatialReuse, nullptr);

	vkDestroyShaderEXT(device, computeShader_dispatchGetNearbyClusterG, nullptr);
	vkDestroyShaderEXT(device, computeShader_getNearbyClusterG1, nullptr);
	vkDestroyShaderEXT(device, computeShader_getNearbyClusterG2, nullptr);

	vkDestroyShaderEXT(device, vertexShader_TreeDebug, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_TreeDebug, nullptr);

	Application::allocator.destroyBuffer(clusterPairDebugGlobalInfoBuffer);
	vkDestroyShaderEXT(device, computeShader_GetDebugClusterG, nullptr);
	vkDestroyShaderEXT(device, vertexShader_ClusterPairDebug, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_ClusterPairDebug, nullptr);
}
void Octree_VRPG::uiRender() {
#ifndef NDEBUG
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;

	std::vector<std::string> normalIndexNames = { "left", "right", "bottom", "up", "back", "forward" };
	std::vector<const char*> normalIndexNames_pointers;
	for (const auto& normalIndexName : normalIndexNames)
		normalIndexNames_pointers.push_back(normalIndexName.c_str());

	if (ImGui::Begin("Debug")) {
		ImGui::SeparatorText("Geometry Clustering");
		UIModified |= ImGui::SliderFloat("Fill Rate", &clusterFillRateThreshold, 0.0f, 1.0f, "%.3f");


		uint32_t geometryShowLayerCount = pushConstant.octreeMaxLayer - OCTREE_CLUSTER_LAYER_VRPG + 1;
		std::vector<std::string> layerMapNames(geometryShowLayerCount);
		for (int i = 0; i < geometryShowLayerCount; ++i) layerMapNames[i] = "Layer" + std::to_string(i + OCTREE_CLUSTER_LAYER_VRPG);

		std::vector<const char*> layerMapNames_pointers;
		for (const auto& layerMapName : layerMapNames)
			layerMapNames_pointers.push_back(layerMapName.c_str());

		if (ImGui::CollapsingHeader("Geometry Tree Debug", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Indent(20.0f);

			static bool showGeometryTreeDebugMap = false;
			static int debugLayer = 0;
			ImGui::Combo("normal_G", &debugPushConstant.normalIndex_G, normalIndexNames_pointers.data(), static_cast<int>(normalIndexNames_pointers.size()));

			ImGui::Combo("Geometry Tree Layer", &debugLayer, layerMapNames_pointers.data(), static_cast<int>(layerMapNames_pointers.size()));
			if (PE::begin()) {
				if (PE::entry("GeometryTreeLayerClusterResult", [&] {
					static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
					ImVec4 selectedColor = showGeometryTreeDebugMap ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
					ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
					ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
					ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
					ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

					bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet((uint32_t)GBuffers_Octree_VRPG::eGeometryClusterResult),
						ImVec2(100 * gBuffers.getAspectRatio(), 100));

					ImGui::PopStyleColor(2);
					ImGui::PopStyleVar();
					return result;
					}))
				{
					showGeometryTreeDebugMap = !showGeometryTreeDebugMap;
				}
			}
			PE::end();

			debugPushConstant.curLayer_G = debugLayer + OCTREE_CLUSTER_LAYER_VRPG;
			if (showGeometryTreeDebugMap) Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)GBuffers_Octree_VRPG::eGeometryClusterResult);

			ImGui::Unindent(20.0f);
		}

		if (ImGui::CollapsingHeader("Geometry SVO Debug", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Indent(20.0f);

			std::vector<const char*> geometrySVONormalNames_pointers = normalIndexNames_pointers;
			geometrySVONormalNames_pointers.push_back("All");
			ImGui::Combo("normal-G2", &debugPushConstant.normalIndex_G_2,
				geometrySVONormalNames_pointers.data(), static_cast<int>(geometrySVONormalNames_pointers.size()));

			static bool showGeometrySVODebugMap = false;
			static int debugLayer2 = 0;
			std::vector<const char*> geometrySVOLayerNames_pointers = layerMapNames_pointers;
			geometrySVOLayerNames_pointers.push_back("All Layers");
			ImGui::Combo("Geometry SVO Layer", &debugLayer2,
				geometrySVOLayerNames_pointers.data(), static_cast<int>(geometrySVOLayerNames_pointers.size()));
			if (PE::begin()) {
				if (PE::entry("GeometrySVOLayerClusterResult", [&] {
					static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
					ImVec4 selectedColor = showGeometrySVODebugMap ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
					ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
					ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
					ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
					ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

					bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet((uint32_t)GBuffers_Octree_VRPG::eGeometrySVO),
						ImVec2(100 * gBuffers.getAspectRatio(), 100));

					ImGui::PopStyleColor(2);
					ImGui::PopStyleVar();
					return result;
					}))
				{
					showGeometrySVODebugMap = !showGeometrySVODebugMap;
				}
			}
			PE::end();

			debugPushConstant.curLayer_G_2 = debugLayer2 >= int(geometryShowLayerCount)
				? pushConstant.octreeMaxLayer + 1
				: debugLayer2 + OCTREE_CLUSTER_LAYER_VRPG;
			if (showGeometrySVODebugMap) Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)GBuffers_Octree_VRPG::eGeometrySVO);

			ImGui::Unindent(20.0f);
		}

		if (ImGui::CollapsingHeader("Light Tree Debug", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Indent(20.0f);

			uint32_t lightShowLayerCount = pushConstant.lightOctreeMaxLayer + 1;
			std::vector<std::string> layerMapNames(lightShowLayerCount);
			for (int i = 0; i < lightShowLayerCount; ++i) layerMapNames[i] = "Layer" + std::to_string(i);

			std::vector<const char*> layerMapNames_pointers;
			for (const auto& layerMapName : layerMapNames)
				layerMapNames_pointers.push_back(layerMapName.c_str());

			static bool showLightTreeDebugMap = false;
			static int debugLayer = 0;

			ImGui::Combo("normal_E", &debugPushConstant.normalIndex_E, normalIndexNames_pointers.data(), static_cast<int>(normalIndexNames_pointers.size()));

			ImGui::Combo("Light Tree Layer", &debugLayer, layerMapNames_pointers.data(), static_cast<int>(layerMapNames_pointers.size()));
			if (PE::begin()) {
				if (PE::entry("LightTreeLayerClusterResult", [&] {
					static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
					ImVec4 selectedColor = showLightTreeDebugMap ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
					ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
					ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
					ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
					ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

					bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet((uint32_t)GBuffers_Octree_VRPG::eLightClusterResult),
						ImVec2(100 * gBuffers.getAspectRatio(), 100));

					ImGui::PopStyleColor(2);
					ImGui::PopStyleVar();
					return result;
					}))
				{
					showLightTreeDebugMap = !showLightTreeDebugMap;
				}
			}
			PE::end();

			debugPushConstant.curLayer_E = debugLayer;
			if (showLightTreeDebugMap) Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)GBuffers_Octree_VRPG::eLightClusterResult);

			ImGui::Unindent(20.0f);
		}

		if (ImGui::CollapsingHeader("Cluster Pair Debug", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Indent(20.0f);

			std::vector<const char*> showModeNames = { "StochasticLightCut LeafNode", "Reservoir Node" };
			ImGui::Combo("Cluster Struct", &debugPushConstant2.mode2, showModeNames.data(), static_cast<int>(showModeNames.size()));

			ImGui::DragInt("OutogingIndex", (int*)&debugPushConstant2.outgoingIndex, 0, OUTGOING_COUNT_VRPG, 1);

			static bool showClusterPairDebugMap = false;
			ImGui::Checkbox("Show Nearby Clusters", &showNearbyClusters);
			if (PE::begin()) {
				if (PE::entry("ClusterPairResult", [&] {
					static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
					ImVec4 selectedColor = showClusterPairDebugMap ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
					ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
					ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
					ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
					ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

					bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet((uint32_t)GBuffers_Octree_VRPG::eLightClusterResult),
						ImVec2(100 * gBuffers.getAspectRatio(), 100));

					ImGui::PopStyleColor(2);
					ImGui::PopStyleVar();
					return result;
					}))
				{
					showClusterPairDebugMap = !showClusterPairDebugMap;
				}
			}
			PE::end();

			if (showClusterPairDebugMap) Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)GBuffers_Octree_VRPG::eClusterPairResult);

			ImGui::Unindent(20.0f);
		}
	}
	ImGui::End();

	{
		const ImVec2 mousePos = ImGui::GetMousePos();
		const ImVec2 vpOrigin = Application::viewportScreenPos;
		const ImVec2 vpSize = Application::viewportContentSize;

		if (changeClusterG = ImGui::IsMouseClicked(ImGuiMouseButton_Right)
			&& vpSize.x > 0.0f && vpSize.y > 0.0f
			&& mousePos.x >= vpOrigin.x && mousePos.x < vpOrigin.x + vpSize.x
			&& mousePos.y >= vpOrigin.y && mousePos.y < vpOrigin.y + vpSize.y)
		{
			// 1) 闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ょ紓宥咃躬瀵鎮㈤崗灏栨嫽闁诲酣娼ф竟濠偽ｉ鍓х＜闁绘劦鍓欓崝銈囩磽瀹ュ拑韬€殿喖顭烽弫鎰緞婵犲嫷鍚呴梻浣瑰缁诲倿骞夊☉銏犵缂備焦顭囬崢閬嶆⒑闂堟稓澧曢柟鍐查叄椤㈡棃顢橀姀锛勫幐闁诲繒鍋涙晶钘壝虹€涙ǜ浜滈柕蹇婂墲缁€瀣煛娴ｇ懓濮嶇€规洖宕埢搴∥熼幁宥嗘皑缁辨捇宕掑▎鎺戝帯闂佺顑嗛幑鍥х暦閺囥垹绀冪€瑰壊鍠氶崝鐑芥⒑鐠恒劌鏋斿┑顔芥尦閹锋垿鎮㈤崗鑲╁弳濠电娀娼уΛ顓炍ｉ崨濞夸簻妞ゆ劑鍨洪崵鍥煛鐏炵偓绀夌紒鐘崇⊕缁绘繈宕橀埡鍐ㄧ悼闂傚倷绀侀幉鈥愁潖瑜版帒绠栭柛灞惧嚬濞兼牠鏌涘┑鍡楃彅鐟滄柨鐣烽幆閭︽Х闂佺顑嗛幐姝岀亙闂佸憡渚楅崢楣冩晬濞戙垺鈷戦悷娆忓缁€鍐煕閳哄倻澧柣锝囧厴瀹曪絾寰勫畝鈧惁鍫ユ⒒閸屾氨澧涘〒姘殜瀹曟洟骞囬婊€绨婚梺闈涱槶閸庤櫕鏅跺☉姘辩＜缂備焦顭囧ú瀛橆殽閻愬樊鍎忛柍瑙勫灴瀹曞崬顫濋鍌氬箣闂傚倸鍊风粈渚€骞夐敓鐘茬闁冲搫鎳庣粈鍫ユ煟閺傛娈犳繛鍏肩墬缁绘稑顔忛鑽ゅ嚬闂佺粯甯掗…鐑藉蓟閿濆绠涙い鏍ㄤ亢閺佸潡鏌ｆ惔銏㈩暡婵犮垺锕㈤垾锕傛嚄椤栵絾些濠电姭鎷冮崨顔界彧濠碘€冲级閸旀瑩鐛幒鎳虫梹鎷呴梹鎰潖闂傚倷鐒﹂崕鍐裁瑰璺虹；闁告洦鍘奸弸鍫⑩偓骞垮劚椤︿即鍩涢幋婢濆綊宕楅懖鈺傚櫘缂備礁顦介崳锝夊蓟閿熺姴鐒垫い鎺嶇劍缂嶅洭鏌嶉崫鍕殶濠㈣泛瀚板鐑樺濞嗘垵鍩岄梺娲诲墮閵堟悂骞嗙仦瑙ｆ瀻闁圭偓娼欐禒顖炴⒑閹肩偛鍔电紒鐘冲灩閳ь剚鑹鹃柊锝夊蓟濞戙垹妫橀悹鎭掑壉瑜嶉埞鎴︽晬閸曨剚姣堥悗瑙勬礈閸犳牠銆佸☉姗嗘僵閺夊牃鏅滅紞渚€姊婚崒娆戝妽闁诡喖鐖煎畷鏇灻洪鍛偓鑸电節闂堟侗鍎忕紒鐘差煼閹妫冨☉娆愬枑濡炪倖姊瑰ú鐔奉潖濞差亜宸濆┑鐘辫兌缁讳線姊洪崫銉バｆい銊ワ躬瀹曞搫鈽夐姀鐘电杸濡炪倖鍨崹褰掓煀閿濆懐鏆﹂柟顖炲亰濡绢亪姊绘繝搴″⒒闁告挾鍠栧濠氭偄绾拌鲸鏅╅梺浼欑到閺堫剟宕抽弶娆炬富闁靛牆鍟悘顏呬繆椤愩垹鏆欓柣锝囧厴閹垽鎮￠獮顑芥櫊閺屾洘寰勯崼婵冨亾閺団懞澶婎潩閼哥鎷绘繛杈剧到閹诧繝骞嗛崼銉﹀仯闁搞儮鏅涙禒褍霉閻欏懐鐣电€规洘绮忛ˇ鎾煃鐠囧樊妲圭紒缁樼箞閹粙妫冨ù韬插劦閺屸剝鎷呴崨濠傛殘闂侀€涚┒閸旀垿骞冩禒瀣窛濠电姴鍊归悿鍕⒒娴ｅ憡鍟為柟绋挎瀹曟﹢鎮欓澶嬶紖濠电姷鏁告慨鐑藉极閸涘﹥鍙忛柣銏ゆ涧閺嬪牏鈧箍鍎遍ˇ浼村磻閸屾稓绠鹃柛鈩兩戠亸浼存煟閹惧鎳囬柟顔煎槻閳诲氦绠涢幙鍐ф偅濠电姵顔栭崳顖滅不閹捐钃熸繛鎴欏灩鍥撮梺绯曞墲閻燂箓宕抽鍓х＝濞达絼绮欓崫娲偨椤栥倗绡€闁绘侗鍣ｅ畷鍫曨敆婢跺娅嶉柣鐔哥矊闁帮綁骞冮幐搴ｇ懝闁逞屽墮椤繐煤椤忓嫪绱堕梺鍛婃处閸撴岸骞忛柆宥嗏拺闁绘垟鏅滃▍鎾绘煕閵娿劍纭鹃柣锝呭槻鐓ゆい蹇撳椤︽澘顪冮妶鍡欏婵ǜ鍔戣棟闁靛鍎哄〒濠氭煏閸繃鍣界紒鐙呯悼缁辨帡鍩€椤掍胶鐟归柍褜鍓熼弫鎰版倷閼碱剚娈曢梺鍛婃处閸忔﹢骞忕紒妯肩閺夊牆澧界粔顒佺箾閸滃啰鎮奸柡渚囧枛閳藉濮€閿涘嫬骞嶆繝娈垮枟缁诲倿鎯岄崼銉ュ惞婵炲棙鎸婚崑鍌炴煏閸繍妲归柣鎾跺枛閺岋繝宕橀妸銉㈠亾閸濄儲鏆滈柛顐ｆ礃閻撱垽鏌涢幇鍏哥盎闁哄妫冮弻宥堫檨闁告挻鐟х划璇差吋婢跺﹦锛熼梻渚囧墮閸楁洟宕堕澶嬫櫖闂佺粯鍔栧娆愮濡も偓閳规垿鎮欓弶鎴犱桓闂佸磭顑曢崐娑㈠Φ閹版澘绠抽柟鎯х摠閻濇娊姊绘担鍛婂暈濞撴碍顨婂畷浼村冀椤撶偞杈堥梺鍓插亖閸庢煡鍩涢幋锔藉仯闁搞儺浜滈惃铏圭磼閻樺磭鍙€闁哄瞼鍠栧畷銊︾節閸愩劉鍋撻幇鐗堢厱闁冲搫鍟禒杈殽閻愬樊鍎旈柡浣稿€块幐濠冨緞閸℃ぞ澹曢梺缁樺灱婵倝鎮￠悩鐢垫／闁哄鐏濋懜鐟邦熆瑜滈崰妤呭Φ閸曨垰围闁稿本鐭竟鏇炩攽閻樺灚鏆╁┑顔炬暬椤㈡瑩寮介鐐电崶闂佸搫绋侀崢浠嬪疾濠靛鐓忓┑鐐茬仢閸斻倝鏌￠崱顓犵暤婵﹤顭峰畷濂告偄瀹勬澘绠梻浣瑰缁诲倿藝椤栨粎涓嶆繛鎴欏灪閻撴稑顭跨捄鍝勵劉缁绢叀鍩栭妵鍕籍閳ь剛鏁垾鎰佹綎濡わ箒锟ユ禍褰掓煙閻戞绠栨繛鍫涘劤缁辨挻鎷呴幓鎺嶅濠电偞鎸婚崺鍐磻閹炬惌娈介柣鎰絻閺嗭絾顨ラ悙宸剶闁轰礁鍊婚幉鎾晲閸℃浼栫紓鍌氬€搁崐椋庢閿熺姴鍨傞柛妤冧紳濞差亜鍐€闁靛闄勯悵鐑芥⒑閸濆嫯顫﹂柛搴㈢叀閹繝濡烽敂钘夊伎濠碘槅鍨伴悘婵嬫偂閹扮増鐓熼柟鎯ь嚟濞叉挳鏌＄仦鍓ф创闁糕晛瀚板畷妯诲緞閸濄儴绻戝┑鐘垫暩閸嬫盯鎮ф繝鍕灊鐎广儱顦粻鐐烘煏婵犲繐顩紒鈾€鍋撻梻浣告啞閸斿繘寮崒娑氼浄闁靛繈鍊栭埛鎴︽煕濠靛棗顏悗姘嵆閺岀喖鎼归顒冣偓鎸庮殽閻愭彃鏆欓柍璇查叄楠炴﹢寮堕幋鐑嗗悋闂備浇宕垫慨鎯ь浖閵娧勫闁挎棁妫勯閬嶆煕閹扳晛濡烽柡鈧禒瀣厽婵妫楁禍婵嗏攽椤栨瑥宓嗛柟顔斤耿閹瑧鎹勯惄鎺撴崌閺岋綀绠涢弮鈧亸鐢电磼鏉堛劍灏伴柟宄版噺缁傛帞鈧綆浜濋幊娆撴⒒娴ｇ儤鍤€闁硅绻濋獮鍐磼閻愵亖鍋撴担鑲濈喖鎳￠妶澶嬵€嶇紓鍌欑椤戝棛鈧瑳鍥︽繝闈涱儐閳锋垿鏌涘☉姗堝伐濠殿喖鍊块弻娑㈠棘濞嗘儳鍓冲┑鈥冲级閸旀瑩寮幇鏉垮耿闁哄洨濮烽悾楣冩⒒娴ｈ櫣甯涢柛鏃撻檮缁傚秴顭ㄩ崼婵堝姦濡炪倖鍨堕崕鎶藉焵椤掍胶绠撻柣锝囧厴婵偓闁靛牆妫楁禍妤€螖閻橀潧浠ч柛鐘冲姇椤啯绂掔€ｃ劉鍋撴担鍓叉建闁逞屽墯閹便劑鍩€椤掑嫭鐓熸俊顖氬悑閺嗏晠鏌＄€ｎ亜鏆ｆ慨濠勭帛閹峰懘鎳為妷褋鈧﹪姊洪崫銉バｉ柟绋款煼楠炲牓濡搁埡浣洪獓闂佺懓顕崑鐔煎礄娴犲鈷掗柛灞捐壘閳ь剛鍏橀幃鐐烘晝閸屾氨鐣洪梺鍓插亝濞叉牜澹曢幐搴涗簻闁哄洦顨呮禍楣冩倵濞堝灝鏋熼柣鎿勭節閵嗕線寮崼婵嗚€垮┑顔筋殔濡瑧绱炲Δ鍛拻濞达絿鎳撻崝銈嗙箾鐎涙ê鍝虹€规洘鍨垮畷鍗炩枎韫囨挃姘舵⒑闁偛鑻晶顕€鏌嶇憴鍕伌闁诡喗鐟╅幊鐘活敆閳ь剟銆傚ú顏呪拺閻犲洩灏欑粻鑼磼鐠囪尙澧曟い鏇秮楠炴牗鎷呴悷棰佺盎闂備胶顢婇崑鎰板磻濡壈濮抽悹鍥ф▕濞撳鏌曢崼婵囶棡閻忓繒鏁婚弻娑氣偓锝庡墮閺嬫垿鏌￠崨鐗堢【闁宠棄顦～婵囨綇閳哄搴婇梻鍌欒兌椤牓寮甸鈧～婵嬪Ω閿旇姤鐝℃繝纰夌磿閸嬫垿宕愰幋锕€鍨傛繛宸簵閳ь剙鍟换婵嬪磼濠婂嫭顔曢梻浣告贡閸庛倝宕洪崼銉﹀亗闁靛鏅滈悡鍐喐濠婂牆绀堟繛鎴炶壘閸ㄦ繃銇勯幘璺烘瀻缂佸墎鍋ら弻鐔兼焽閿曗偓楠炴劙鏌熼柨瀣仢婵﹥妞藉畷銊︾節閸曨厾鏆ら梺璇插閸戝綊宕滈悢濂夊殨闁告劖绁撮弸搴ㄦ煙閹屽殶闁告妫勯埞鎴炲箠闁稿﹥鍔欏畷鎴﹀箻缂佹鍙嗛梺鍝勬储閸斿鏌囬娑辨闁绘劖褰冮弳锝夋煛娴ｈ宕岄柟顔规櫊楠炴捇骞掗幋鐘仏闂傚倸鍊风欢姘焽瑜忛幑銏ゅ幢濞戞鍔﹀銈嗗笂閻掞箓鎮橀柆宥嗙厱閻庯綆鍋呭畷宀€鈧娲栭妶绋款嚕閹绢喗鍋勯柛婵嗗缁犳椽姊婚崒娆戭槮缂傚秴锕畷鎴炵節閸パ冨亶闂佸吋绁撮弲娑氭閵堝應鏀介柣妯虹－椤ｆ煡鏌ｉ幘鍐叉殻婵﹥妞藉畷褰掝敋閸涱厼澹夋俊鐐€戦崕閬嶆偋閹捐钃熼柍銉ョ－閺嗗棝鎮楅敐搴″鐞氾附淇婇悙顏勨偓鎴﹀垂濞差亝鏅濋柕蹇嬪€曠粻鏍ㄧ節婵犲倻澧曟鐐灪娣囧﹪濡堕崒姘濠电姰鍨婚幊鎾绘煀閿濆拋娼栫紓浣股戞刊鎾煕濞戞﹫鏀婚柛搴㈡崌濮婃椽宕崟顓炩拡闂佸憡鎸鹃崰鏍嵁閸℃稑绀嬫い鏍电稻閺咁剟姊虹紒妯哄閻忓繑鐟╅幃鎯洪鍛嫼闂佸憡鍔曞鍫曞箚閸儲鐓欓柛鎰叀閸欏嫰鏌熼姘冲閾绘牠鏌嶈閸撶喖鍨鹃敃鍌氶敜婵°倓绀佸▓婵嬫⒑濞茶澧诲ù婊勭箞瀹曞綊宕稿Δ鈧弸浣衡偓骞垮劚椤︻垶鐛姀锛勭闁瑰鍋熼幉鍧楁煙妞嬪海甯涚紒缁樼⊕濞煎繘宕滆閸╁矂姊虹涵鍜佸殝缂佺粯绻堥悰顕€宕卞☉姗€鍞堕梺闈涱檧婵″洭宕㈤挊澶樻富闁靛牆妫楅崸濠囨煕鐎ｎ偅宕岄柡灞炬礋瀹曞ジ濮€閳哄啫濮奸梻浣风串缂嶄線寮ㄦ潏鈹惧亾闂堟稏鍋㈢€殿喖鐖奸獮瀣攽閸℃ɑ娈㈡繝纰夌磿閸嬫垿宕愰弽顓炵濡わ絽鍟壕濠氭煟閺冨倵鎷￠柡浣革躬閺岀喖骞嶉纰辨毉闂佺顑傞崜婵堟崲濠靛洨绡€闁稿本绋戝▍锝咁渻閵堝繒鍒伴柕鍫熸倐楠炲啯绂掔€ｅ灚鏅┑鐐村灱妞存悂鎮挎笟鈧幃妤冩喆閸曨剛顦ㄩ梺鎼炲妼濞尖€崇暦瑜版帒绠氱憸婊堟偄閸℃稒鐓曢柡鍥ュ妼婢т即鏌嶉娑欑闁哄被鍔岄埥澶娾枎韫囨搩娼旈梻浣烘嚀閸㈡煡骞婂Ο铏规殾妞ゆ劧绠戠粈瀣亜閹伴潧浜為柍褜鍓氶幐鍐差潖婵犳艾纾兼繛鍡樺笒閸橈紕绱掔紒銏犲箹闁绘牜鍘ч锝夊箵閹烘繄鍙嗛梺鍛婁緱閸ㄦ娊鎮￠幘缁樷拺閻熸瑥瀚粈鍐煕閳哄倻澧甸柨婵堝仱瀵挳濮€閿涘嫬骞嶉梻浣侯攰椤宕濋敃鍌氬惞婵炲棙鍨瑰Λ顖炴煙椤栧棗鍟崐顖滅磽娴ｈ櫣甯涢柣鈺婂灠閻ｇ兘宕￠悙宥囧枑濞碱亪寮堕崹顔句簴闂傚倸鍊风粈渚€骞楀鍫濈；婵炴垶姘ㄧ粈濠傗攽閻樻彃浜炴繛鍏肩墵閺屟嗙疀閹剧纭€闂佹椿鍘介悷锔炬崲濞戙垹骞㈡俊顖氭惈婵垽姊洪崨濠冨磳濞存粏娉涢～蹇撁洪鍕祶濡炪倖鎸鹃崰搴♀枔鐏炶В鏀介柣鎰级閳绘洟鏌涘Ο鎭掑仮妞ゃ垺宀搁弫鎰板炊閵娿儳褰存繝鐢靛仦濞兼瑩顢栭崱娑崇稏鐎广儱顦伴埛鎴︽⒒閸喓銆掔紒鐘插暱閳规垿顢欓懞銉х▏闂佷紮绲块弫璇茬暦閸楃偐妲堟繛鍡楁湰鐎氬ジ姊婚崒姘偓鎼佹偋婵犲啰鐟规俊銈呮噺閸婂灝螖閿濆懎鏆為柣鎾存礋楠炴牕菐椤掆偓閳ь剚鐗犻幃鐢割敂閸℃瑧锛滈梺绋挎湰缁嬫垵霉椤斿浜滈柕蹇婃閼板灝鈹戦鐟颁壕闂備焦濞婇弫顕€宕戦幘缁樼厵闁汇値鍨遍鐘电磼鏉堛劍宕岀€规洘甯掗埢搴ㄥ箣濠靛棭鐎撮梻浣芥閸熷瓨绂嶉崼鏇炵畺婵☆垵娉曢悿鈧梺鍝勬川閸庢劙寮抽鐘电＝濞达綀娅ｇ敮娑㈡煟閳哄﹤鐏犻柣锝囧厴閺佹劙宕卞▎鎴濋獎闂備焦瀵х换鍌毼涘Δ鍛€堕柍鍝勫暟绾惧吋銇勯弮鍌涙珪妞わ絽寮剁换娑㈡偂鎼达絿顔掓繝纰樺墲閻℃洟藝閹绢喗鐓熼柨鏇楀亾闁绘鎸搁～蹇撁洪鍕啇闂佺粯鍔栬ぐ鍐╂叏閸洘鈷戦悹鍥皺缁犺尙绱掔€ｎ偆澧甸挊婵堟喐閻楀牆绗氶柣鎾寸洴閹﹢鎮欓幓鎺嗘寖濠电偞褰冮顓㈠焵椤掍緡鍟忛柛鐘愁殜楠炴劙鎼归锛勭畾闂佸綊妫跨粈浣告暜闂備線娼ч敍蹇曚沪閼恒儲鐝濆┑鐘垫暩婵兘寮崨濠冨弿闁绘垵顫曢埀顒€鍊块、娆撴倷椤掑缍楅梻浣告惈濞层垽宕硅ぐ鎺撳仾闁绘劦鍏欐禍婊堟煙閹佃櫕娅呴柍褜鍓氶悧鐘绘晲閻愬搫鍗抽柕蹇ョ磿閸樼敻姊虹紒妯虹仸闁挎洍鏅涢埢鎾诲即閵忥紕鍘辨繝鐢靛Т濞诧箓寮柆宥嗙厪闁搞儜鍐句純濡ょ姷鍋炵敮鈥崇暦閸楃儐娓婚柟顖嗗本顥＄紓鍌氬€搁崐鎼佸磹妞嬪孩濯奸柡灞诲劚绾惧鏌熼崜褏甯涢柣鎾存礋閺岀喖寮堕崹顔藉€庣紓浣靛妽閹告娊寮诲☉姘ｅ亾閿濆骸浜濋悘蹇曟暬閺岀喖宕ｆ径瀣攭閻庤娲滈崰鏍€佸Δ鍛＜闁靛牆鏌婇妶澶嬧拻闁稿本鐟︾粊鐗堜繆濡炵厧濮傞柟铏殜瀹曞ジ寮村璇蹭壕闁挎洖鍊搁柋鍥ㄧ節闂堟稒顥犻柕?= 闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ょ紓宥咃躬瀵鎮㈤崗灏栨嫽闁诲酣娼ф竟濠偽ｉ鍓х＜闁诡垎鍐ｆ寖闂佺娅曢幑鍥灳閺冨牆绀冩い蹇庣娴滈箖鏌ㄥ┑鍡欏嚬缂併劌銈搁弻鐔兼儌閸濄儳袦闂佸搫鐭夌紞渚€銆佸鈧幃娆撳箹椤撶噥妫ч梻鍌欑窔濞佳兾涘▎鎴炴殰闁圭儤顨愮紞鏍ㄧ節闂堟侗鍎愰柡鍛叀閺屾稑鈽夐崡鐐差潻濡炪們鍎查懝楣冨煘閹寸偛绠犻梺绋匡攻椤ㄥ棝骞堥妸鈺傚€婚柦妯侯槺閿涙盯姊虹紒妯哄闁稿簺鍊濆畷鎴犫偓锝庡枟閻撶喐淇婇婵嗗惞婵犫偓娴犲鐓冪憸婊堝礂濞戞碍顐芥慨姗嗗墻閸ゆ洟鏌熺紒銏犳灈妞ゎ偄鎳橀弻锝呂熼懡銈呯仼闂佹悶鍎崝搴ㄥ储闁秵鐓熼煫鍥ㄦ礀娴犳粌顭胯缁瑩骞冮敓鐙€鏁嶆慨妯垮亹閸炵敻鏌ｉ悢鍝ユ噧閻庢凹鍘剧划鍫ュ焵椤掑嫭鈷戦悗鍦濞兼劙鏌涢妸銉﹀仴闁靛棔绀侀埢搴ㄥ箣閻樼儵鏀洪梺璇插嚱缂嶅棝宕滃▎鎰弿婵炲樊浜濋埛鎺懨归敐鍛暈闁哥喓鍋為妵鍕敇閻愭惌妫﹂悗瑙勬礈閸犳牠銆佸Δ浣瑰闁告繂瀚埢蹇旂節绾板纾块柛瀣灴瀹曟劙寮介鐔蜂罕濠电姴锕ら悧鍡欏婵犳碍鐓熼柟閭﹀枛閸斿鏌涚€ｎ亜鈧湱鎹㈠☉銏犵婵炲棗绻掓禒楣冩⒑缂佹ɑ灏版繛鍙夘焽閹广垹鈽夐姀鐘殿吅闂佺粯鍔曢悘姘跺闯椤撶姷纾藉ù锝堟閻撴劙鏌ｉ幙鍕瘈鐎殿喖顭烽弫鎰緞婵犲嫷鍟嬮梻浣告惈椤︿即宕归悽闈╃稏閹艰揪绲跨壕钘壝归敐鍕煓闁告繃妞介幃浠嬵敍濡炶浜剧€规洖娲﹀▓楣冩⒑閸︻厼鍔嬮柛銊у枎鍗遍柛顐ゅ枍缁诲棙銇勯弽銊ヮ棜闁瑰嚖绻濋弻娑欐償閿涘嫮鍔┑顔硷功缁垳绮悢鐓庣劦妞ゆ帒瀚悞鍨亜閹烘垵鈧憡绂掑鍫熺厾婵炶尪顕ч悘锟犳煛閸涱厾鍩ｇ€规洘绮忛ˇ瀵哥棯閸撗勬珪闁逞屽墮缁犲秹宕曢柆宥呯疇闊洦铔嬫径鎰潊闁靛牆妫涢崢閬嶆⒑鐎圭媭娼愰柛搴ゆ珪缁傚秹鎮欓璺ㄧ畾闂佺粯鍔︽禍婊堝焵椤掍胶澧遍柡渚囧櫍楠炴帒顪冨┑鍡樺枠妞ゃ垺绋戦～婵嬫偂鎼淬埄鍚欓梻鍌欒兌椤宕熼崹顐ゆ殾闂備焦濞婇弨杈╂暜閿熺姴钃熸繛鎴欏灩閸楁娊鏌曟繛鍨姍缂併劏顕ч—鍐Χ閸涱喚顩伴梺鍛娒肩划娆撳Υ閸愵喖唯闁冲搫鍊搁埀顒傚厴閺屻倗鍠婇崡鐐差潻闂佹剚浜滈悘姘辨崲濠靛牆鏋堟俊顖氭惈閳峰姊虹拠鈥虫灈闁稿﹨顫夌粚杈ㄧ節閸ャ劎鍙嗛柣搴秵閸嬪嫰宕濋崨瀛樷拺闁圭娴风粻鎾绘煙閸愯尙绠婚柟顔斤耿瀹曠厧顭块鍛棃闁诡喒鏅犲Λ鍐ㄢ槈濡や礁濯伴梻鍌欑閹测€愁潖瑜版帗鍋嬮柣妯垮吹瀹撲線鏌涢埄鍐槈缂侇偄绉归弻娑㈩敃閵堝懏鐎惧┑鐐叉噺缁挸顫忓ú顏勫窛濠电姴鍊歌〖缂傚倷娴囩亸顏呮叏閵堝洨鐭夌€广儱妫庨崑鍛存煕閹般劍娅囬柛妯兼暬濮婄粯绗熼崶褍浼庣紓浣哄У閸ㄥ灝顕ｉ幖浣肝ч柛銉㈡櫆閿涘繘姊虹拠鈥崇€诲ù锝夋櫜閸掓帡姊绘担鍛婃儓闁活剙銈稿畷浼村冀椤撴壕鍋撴担绯曟闁靛繒濮撮懓鍨攽鎺抽崐鎰板磻閹剧粯鐓涢柛鈩冪懃閺嬫梻绱掓潏銊ョ闁逞屽墾缂嶅棙绂嶉悙鐢典笉鐎规洖娲ㄧ壕濂稿级閸稑濡介悘蹇斿缁辨帒螖閸愩劍鐏堥悗瑙勬礃閸庡ジ藝閹惰姤鐓熼煫鍥ㄦ尵鏁堝┑顔硷攻濡炰粙骞婇敓鐘参ч柛娑卞枟閻︼綁姊绘担鍛婃儓闁硅櫕鍔栫换娑㈠焵椤掑嫭鐓熸繛鎴濆船濞呭秶鈧娲橀敃銏ゃ€佸▎鎾村殟闁靛瀵屾禒褔姊婚崒娆掑厡缂侇噮鍨堕幃鐑藉閵堝懐锛涢梺鐟板⒔缁垶鎮″☉妯忓綊鏁愰崶鍓佸姼闂佸搫妫寸紞渚€寮婚敍鍕勃閻犲洦褰冮～鍥倵濞堝灝鏋熸繛鍏肩懆閻忓啴姊洪柅鐐茶嫰婢ь垳绱掗崒姘毙ｉ柕鍫秮瀹曟﹢鍩為悙顒€顏圭紓鍌氬€搁崐鐑芥倿閿曞倹鏅┑鐘愁問閸犳牕煤椤撱垹钃熼柕濞炬櫅閸楁娊鏌曟繛鍨偓婵嬪磻濡ゅ懏鍊垫繛鍫濈仢閺嬫稒銇勯鐐叉Щ妞ゆ洩缍侀獮姗€顢欓挊澶夌盎闂備焦鍎崇换鎰耿閸楃倣锝夊炊椤掍讲鎷绘繝鐢靛Т妤犲憡绂嶅鍕閻犲泧鍛殼濡ょ姷鍋為悧鐘诲春閳╁啫顕遍弶鍫涘妽閵囨繃銇勯姀鈽呰€块柟顔界矒瀹曞崬螣閾忚袙闂傚倸鍊风欢姘焽瑜旈幃褔宕卞▎鎰簥闂佽澹嗘晶妤呭磿婵犲啰绡€濠电姴鍊绘晶娑㈡煟閹惧瓨绀冨ǎ鍥э躬椤㈡稑鈽夊Δ鍕劵闂備胶顭堥鍐垂娴犲钃熺€广儱顦敮闂佹寧姊婚悺鏂课涢崘顔藉€甸悷娆忓缁€鍐煥閺囨ê鐏查柕鍡曠閳诲酣骞橀崘鑼搸濠电姰鍨奸鏍垂閺夋５娲Ω閳轰胶鐓戦梺鐟板⒔缁垶鎮￠弴銏＄厪濠㈣埖绋撻崚鏉库攽閿涘嫭娅曠紒杈ㄥ浮閸┾偓妞ゆ帒瀚柋鍥煏婢舵稑顩柛妯绘倐濮婅櫣鍖栭弴鐐测拤闂佸憡姊瑰ú鐔煎春閳ь剚銇勯幒宥堝厡濠⒀囦憾閺岀喖鐛崹顔句患闂佸疇妫勯ˇ顖烇綖濠靛鏅查柛娑卞墮椤ユ碍绻濋悽闈涗沪闁搞劌鐖奸幃鐑筋敍閻愭潙鈧灝霉閻樺樊鍎愰柣鎾跺枑閵囧嫰骞樼捄鍝勫闂傚鍓﹂崜鐔煎蓟閳ュ磭鏆嗛悗锝庡墰琚﹂柣搴㈩問閸犳洜鍒掑▎鎾扁偓渚€寮撮姀鐙€娼婇梺缁橆焾娴滎剛鏁☉妯锋斀?
			const float localX = mousePos.x - vpOrigin.x;
			const float localY = mousePos.y - vpOrigin.y;

			// 2) 闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ょ紓宥咃躬瀵鎮㈤崗灏栨嫽闁诲酣娼ф竟濠偽ｉ鍓х＜闁绘劦鍓欓崝銈囩磽瀹ュ拑韬€殿喖顭烽弫鎰緞婵犲嫷鍚呴梻浣瑰缁诲倿骞夊☉銏犵缂備焦顭囬崢閬嶆⒑闂堟稓澧曢柟鍐查叄椤㈡棃顢橀姀锛勫幐闁诲繒鍋涙晶钘壝虹€涙ǜ浜滈柕蹇婂墲缁€瀣煛娴ｇ懓濮嶇€规洖宕埢搴∥熼幁宥嗘皑缁辨捇宕掑顑藉亾閻戣姤鍤勯柤绋跨仛閸欏繘姊洪崹顕呭剳缂佺娀绠栭弻宥堫檨闁告挾鍠栧濠氬灳瀹曞洦娈曢梺閫炲苯澧寸€规洑鍗冲浠嬵敇閻樿尙銈﹂梻浣虹《閸撴繈宕欓悷鎷旓絿鎷犲顔藉瘜闂侀潧鐗嗙换鎺楀礆娴煎瓨鐓曢柣鏃堟敱閸嬨儵鏌涢埞鍨姕鐎垫澘瀚伴獮鍥敆閸屻倕鏅梺璇叉唉椤煤閺嵮€鏋栭柨鏇炲€稿洿婵犮垼娉涢鍥储閹剧粯鍋℃繝濠傚閻帞鈧娲樼划宀勫煝鎼淬劌绠ｉ柣妯夸含閻熸繈鏌ｆ惔锛勭暛闁稿酣浜惰棟妞ゆ牗绮岄ˉ姘舵煕瑜庨〃鍡涘煕閹寸偑浜滈柟鍝勬娴滈箖姊洪崨濠庢畷婵炶尙鍠庨悾鐑藉醇閺囥劍鏅㈡繝銏ｆ硾椤戝棗鈻嶉弽顓熲拻濞达綀娅ｇ敮娑㈡煕閺冣偓閻楃偤濡甸幇鏉跨妞ゆ棁妫勯埀顒€鐏氶妵鍕箻閸楃偟浠鹃梺鎶芥敱閸ㄥ湱妲愰幒鏂哄亾閿濆骸浜滄い鏇熺矋缁绘繈鍩€椤掍礁顕遍悗娑欘焽閸橀亶姊虹€圭姵銆冪紒鈧笟鈧鎶芥晲閸ワ絽浜炬繛鍫濈仢閺嬫棃鏌涢弬璺ㄐょ紒顔碱儔楠炴帡寮崫鍕闂佹寧绻傛鍛婄閻愮儤鐓曟俊顖涘椤ュ鏌嶇憴鍕伌闁糕斂鍎靛畷鍗炍旈崘褎妯婂┑鐘垫暩閸嬫盯骞忛幋鐘电濞撴埃鍋撶€殿噮鍋勯濂稿椽娴ｅ憡鍊梻浣虹《閸撴繈宕濋幒鏃囧С濠电姵纰嶉埛鎴︽煙缁嬫寧鎹ｇ紒鐘虫尰娣囧﹪顢曢敐鍥ㄥ垱閻庤娲╃紞渚€宕洪埀顒併亜閹烘垵顏柍閿嬪灴閺屾稑鈹戦崱妤婁痪濠电姭鍋撻柟娈垮枤绾惧ジ鎮楅敐搴′簼鐎规洖鐭傞弻娑㈠煘閹傚濠碉紕鍋戦崐鏍暜閹烘鏅濋柕澶嗘櫅閻掑灚銇勯幋鐐插姢闁告棑濡囬埀顒冾潐濞叉粓宕伴弽顓溾偓浣肝旈崨顓狅紲濠殿喗锚瀹曨剟宕哄畝鍕拻闁稿本鑹鹃埀顒佹倐瀹曟劖顦版惔锝囩劶婵炴挻鍩冮崑鎾绘煙椤斿吋鍋ョ€规洖鐖奸、妤佺節閸曨厸妫ㄦ繝鐢靛О閸ㄥジ宕洪弽顓炵闁哄洢鍨归崹鍌炴煕瀹€鈧崑鐐烘偂閺囩姵鍠愰幖娣妸閳ь剙鍟村畷銊╊敍濠娾偓缁楀姊洪崫鍕枆闁告ü绮欓幃锟犲礃椤旂晫鍘卞┑鐘绘涧濡顢旈埡鍛厸闁糕剝顨愰煬顒勬煛鐏炲墽銆掗柍褜鍓ㄧ紞鍡涘磻閸涱厾鏆︾€光偓閳ь剟鍩€椤掑喚娼愭繛鍙夌矒瀹曟顫滈埀顒勫Υ娴ｅ壊娼ㄩ柍褜鍓熼獮鍐ㄢ枎閹惧磭鐤€闂佸疇妗ㄩ懗鑸电妤ｅ啯鈷戦梻鍫熺⊕閹兼劙鎮楀鍗烆暭濞ｅ洤锕幃婊堟嚍閵夛附鐝曢梻浣告啞閹稿棝宕ㄩ鐣屾缂傚倸鍊搁崐鎼佸磹瀹勬噴褰掑炊閵婏絼绮撻梺褰掓？缁€浣虹不閺夊簱鏀介柣妯虹枃婢规绱掗悪鈧崹鍫曞蓟濞戞ǚ妲堟繛鍡樺姉缁嬪洨绱撴担鍝勑ラ柡浣割煼瀵鎮㈤悡搴濈炊闂佸憡娲﹂崢婊堝鎺虫禍婊堟煏婢舵稓鐣遍柛婵堝劋閹便劍绻濋崒銈囧悑閻庤娲樼敮鎺楀煝鎼淬劌绠抽柟瀛樼箓閼垫劗绱撻崒娆掑厡闁惧繐閰ｅ畷銏＄附閸涘﹤浜梺缁樻尭濞寸兘寮抽敂鐣岀瘈濠电姴鍊搁弳濠囨煛閳ь剚绂掔€ｎ偆鍘卞銈嗗姉婵挳鎮橀鈧弻锝夊箻鐎涙ê纾冲┑顔硷攻濡炶棄鐣烽悜绛嬫晣闁绘劖褰冮‖鍡樹繆閻愵亜鈧垿宕濈仦杞挎稑鈹戦崱娆愭闂佺鎻粻鎴犲婵傚憡鐓熼柟閭﹀墻閸ょ喖鏌涘Ο鍦煓婵﹥妞藉畷顐﹀礋椤愶絾顔勯梻浣虹帛椤ㄥ懘鏁冮妶澶婄厺鐎广儱鎷嬮悡銉╂煕濞戝崬鐏ｉ柨娑欑矒閺岋綁鎮╅崣澶嬫倷閻庢鍠栭悥鑹版＂闂侀潧绻堥崐鏍偂濞戙垺鍊堕柣鎰版涧娴滃墽绱掗埀顒傗偓锝庡厴閸嬫挾鎲撮崟顒傤槶闂佸憡顭嗛崶褏鍘撮梺纭呮彧缁犳垿鐛姀銈囧彄闁搞儯鍔嶇亸顐㈩熆鐟欏嫭绀嬫慨濠冩そ瀹曘劍绻濋崟銊︻潔闂備焦瀵уú鈺冪不閹捐违闁告稒娼欑粻鐢告煙楠炲灝濡芥俊顐㈠暣瀵偊骞囬弶鍨獩濡炪倖鎸鹃崰鎰邦敊婢舵劖鈷掑ù锝囨嚀閳绘洟鏌￠埀顒勬焼瀹ュ懎鐎梺闈╁瘜閸樹粙锝為弴銏＄厵闁诡垎鍛喖婵犳鍨遍幐鎶藉蓟閿濆牜妯勯梺鍝勬嚀鐏忔瑥鈻庨姀銈嗩棃婵炴垶甯楅弬鈧梻浣虹帛閸旀浜稿▎鎰珷婵炴垶鐟х粻楣冩煠绾板崬澧柡瀣〒閳ь剚顔栭崰娑㈩敋瑜旈崺銉﹀緞閹邦剦娼婇梺缁橆焽閺佺顭囨径鎰拻濞撴埃鍋撻柍褜鍓涢崑娑㈡嚐椤栨稒娅犻柟缁㈠枟閻撴瑦銇勯弮鈧崕鎶藉储閹绢喗鐓曢柍鍝勵儑缁♀偓閻庤娲樼敮鎺楀煝鎼淬劌绠ｉ柣妯虹仛濮ｅ酣姊婚崒姘偓椋庣矆娴ｉ潻鑰块梺顒€绉甸幆鐐哄箹濞ｎ剙濡肩紒鎰殜閺屸€崇暤椤旇壈瀚伴柡鍛Т閳规垿鎮╃紒妯婚敪濡炪倖鍨靛Λ婵嬪箖閳ユ枼鏋庨柟瀵稿У閿涘繘姊洪崫鍕垫Ц闁诲繑绻堥幊婊呮喆閸曞灚顔旈梺缁樺姈閹苯鈻撳鍫熺厸鐎光偓鐎ｎ剛鐦堥悗瑙勬磸閸旀垿銆佸▎鎾崇鐟滃繘寮堕懞銉х瘈婵炲牆鐏濋弸娑㈡煥閺囨ê鍔氭い顏勫暞缁傛帞鈧綆鍓欑粊锕傛⒑瑜版帒浜伴柛妯圭矙瀹曟洟鎮㈤崗鑲╁帗闂佹悶鍔嶇粙鎺戔枍閸℃稒鍊垫慨姗嗗亞閻瑦鎱ㄦ繝鍕笡闁瑰嘲鎳樺畷銊︾節閸愩劌澹嶉梻鍌欒兌閹虫捇宕查弻銉晞濠㈣泛锕﹂弳锕傛煙鐎电浠х紒鈾€鍋撻梻浣侯焾閺堫剛鍒掑鍛灁濡わ絽鍟埛鎴︽⒒閸喓銆掔紒鐘靛仱閺屾稒绻濋崘鈺冾槶缂備緡鍠栭…鐑藉箖閻戣棄绠犲┑鐘插€荤粔娲煛娴ｇ懓濮嶇€规洏鍔戦、娆戞崉椤垶鏁ら梻鍌氬€风粈渚€骞夐敍鍕畳缂傚倷绶￠崰妤呮偡閵夈劎浜遍梻浣告啞缁嬫帡骞嗗畝鍕瀬闁告劦鍠楅悡蹇涚叓閸ヮ灒鍫ュ磻閹捐绠涘ù锝呭ⅲ閻斿皝鏀介柣妯虹仛閺嗏晠鏌涚€ｎ偆鈽夋い鏇秮瀹曞ジ寮撮悢铚傜綍婵犲痉鏉库偓鏇㈠疮娴煎瓨鍎楀鑸靛姈閻撴瑩鏌ｉ幋鐐嗘垶寰勯崟顑跨箚妞ゆ牗鑹鹃顓熸叏婵犲啯銇濈€规洦鍋婂畷鐔碱敆娴ｇ懓顏扮紓鍌氬€风粈渚€顢栭崱娑樼闁搞儺鍓欓弰銉╂煃瑜滈崜姘跺Φ閸曨垰绠抽柟鎼灠婵稓绱撴担鍓叉Ш闁轰礁顭峰璇测槈濮橈絽浜鹃柨婵嗙凹缁ㄥジ鏌涢敐鍫燁棄闂囧绻濇繝鍌氼仾闁稿﹥鍔栨穱濠囶敃閵忕姵娈梻鍥ь樀閺屾洝绠涚€ｎ亞浠鹃梺鎼炲€愰崑鎾翠繆閻愵亜鈧倝宕㈣ぐ鎺戠闁瑰瓨绻勯弳锕傛煟閺冨倵鎷￠柡浣告閺屽秷顧侀柛鎾跺枛瀹曟椽鍩€椤掍降浜滈柟鍝勭Ф椤︼箑鈹戦鑲╁ⅵ闁哄本娲熷畷鎯邦槻妞ゅ浚鍘奸埞鎴︻敊閻熼澹曢梻鍌欑閻ゅ洤螞閸曨垰绠犻柟鎹愵嚙閻掑灚銇勯幒鍡椾壕闂佽绻戠换鍫ュ春閻愬搫绠ｉ柨鏃傜帛閺呪晠姊洪懡銈呮灈妞わ綇闄勭粩鐔煎即閵忊檧鎷婚梺绋挎湰閻熝囧礉瀹ュ鍊电紒妤佺☉閹虫劙鎯屽▎鎾寸厵閻庣數顭堣ⅴ闂佺顑嗛幐鎼侊綖濠靛鍋傞幖绮规閹奉偊姊绘担铏瑰笡濞撴碍顨婂畷鎶芥晲閸涱垱娈炬繝闈涘€婚…鍫濇暜闂備焦瀵уΛ渚€锝炴径灞稿亾濮樿櫕顥夐柍瑙勫灴閹瑧鈧稒锚闂夊秹姊虹化鏇熸珔闁哥喐娼欓悾鐑藉箣閿曗偓缁犲鎮楅棃娑欐喐妞ゆ梹娲樼换娑欐綇閸撗冨煂闂佸摜鍠庨悘婵堝弲闂佺粯顭堥褔鎮為崹顐犱簻闁圭儤鍩婇崝鐔兼煟閿濆洦鏆╅柍褜鍓濋～澶娒洪弽顓炍ч柛娑橈梗缁诲棝鏌ｉ幋锝嗩棄閸烆垶鎮峰鍕畼婵″弶鍔欓幃浠嬪川婵炵偓瀚介梻浣侯焾閺堫剟鎮疯閹疯瀵肩€涙鍘遍梺瑙勫劤椤曨參骞婇崨顖滅＜妞ゆ梻鏅幊鍕磼濡ゅ啫鏋涢柛鈹惧亾濡炪倖宸婚崑鎾淬亜閺囶亞绉い銏℃礋閺佸啴鍩€椤掑嫮宓侀柕蹇ョ磿缁犻箖鏌熺€电浠ч柟鍐叉嚇閺屾稑顫滈崱鏇㈠仐濠碘槅鍋勯幊姗€銆侀弴銏狀潊闁炽儱鍟挎禍鍫曟⒒娴ｄ警鐒惧Δ鐘叉憸婢规洟顢橀悢铏瑰骄闂佸搫娲ㄦ慨闈涚暦婢舵劖鐓忓┑鐐戝啯瀚呯紒杈ㄧ叀濮婄粯鎷呴搹鐟扮濡炪値鍘奸悧鍡涙偩閻戣姤鍋ㄩ柛娑橈工娴犵厧鈹戦悙鍙夆枙濞存粍绻堥幃鈥斥槈濮橈絽浜鹃柛蹇擃槸娴滈箖鏌ｆ惔顖滅У闁告挻绋栭埅闈涒攽鎺抽崐妤佹叏閻戣棄纾婚柣鎰劋閺呮繈鏌曡箛瀣偓鏇犵不閸︻厽鍠愰柣妤€鐗嗘穱顖炴煛娴ｇ鏆ｉ柡灞诲妼閳规垿宕卞Ο鐑樼暬婵犵妲呴崑濠囁夐幘鑸殿潟闁规崘顕х壕鍏兼叏濮楀棗鍘撮柛瀣崌閹粓鎳為妷銉︾暠闂備浇鍋愰…鍫ユ倶濮樿泛鐭楅煫鍥ㄦ嫻閺冨牊鏅查柛娑卞幗閻忔捇姊烘潪鎵槮妞ゆ垵娲獮澶愬箹娴ｅ摜鐫勯梺鍓插亝缁诲啫危椤掑嫭鈷戠紓浣姑慨鍥煙绾板崬浜炵紒顔碱煼閺佹劖寰勭€ｎ剙寮抽梻浣告惈濞诧箓銆冮崨顔绢洸濡わ絽鍟埛鎴犵磽娴ｅ顏嗙箔濮橆厺绻嗛柣鎰閻瑧鈧鍠涢褔鍩ユ径鎰潊闁绘鏁搁弶鎼佹⒒娴ｄ警鐒剧紒缁樺姍楠炴顭ㄩ崼婵堫槱閻熸粎澧楃敮妤呭煕閹达附鐓熼柣鏂挎啞缁惰尙绱掗悪娆忔处閻撶喖鏌ｉ弮鈧娆撳礉閿曞倹鐓冮柕澶涢檮椤ュ牏鈧娲橀敃銏ゃ€佸▎鎾冲簥濠㈣鍨板ú銊╁矗閹剧粯鐓曢柕澶涚到婵″潡鏌嶉娑欑闁靛洤瀚幆鏃堟晬閸曟嚪鍕弿濠电姴鎳忛鐘电磼鏉堛劌绗掗摶锝囩磽娴ｅ顏堝礈閻㈢數纾介柛灞剧懆閸忓瞼绱掗鍛仯闁瑰箍鍨藉畷鎺楁倻閸℃ɑ娅旈梻浣瑰缁诲倸螞濞戞艾濮柍褜鍓欓埞鎴︻敊閺傘倓绶甸梺绋跨凹缁插藝閸欏浜滄い鎾墲绾爼鏌熼悷鏉款仾缂佹鍠栭、娑橆潩椤掆偓椤ュ秹姊婚崒娆戠獢闁逞屽墰閸嬫盯鎳熼娑欐珷闁规鍠氱壕鍏笺亜閺囩偞鍣归柣蹇ョ磿缁辨帒鐣濋崘顔兼懙缂備胶绮换鍫濈暦閸洘鍤嬮柛顭戝亝閻濐偄鈹戦悩鍨毄濠电偐鍋撳┑鐐板尃閸涱亜浜炬慨妯峰亾闁搞儜鍜冪幢闂備浇顫夊畷姗€顢氳閹偤宕归鐘辩盎闂佸湱鍎ら崹鐢割敂椤忓牊鐓熼柟鎯ь嚟濞叉挳鏌熼绛嬫疁闁轰焦鍔栭幆鏃堝灳閼碱剚鏅奸梻鍌欒兌閹虫捇宕崹顔氭椽鏁冮崒姘辩杽闂侀潧顭俊鍥╁姬閳ь剟姊洪崨濠傚闁哄倸鍊圭粋鎺楀閵堝棗鈧敻鎮峰▎蹇擃仾濠㈣泛瀚伴弻娑氣偓锝庝悍瀹搞儵鏌ｉ敐鍥у幋闁诡喓鍨婚崰濠冩綇閵婏附鐝滄繝鐢靛仩閹活亞寰婇崸妤€绠犻柟鐐た閺佸倿鏌熷▓鍨灓缁惧彞绮欓弻娑氫沪閹规劕顥濋梺閫炲苯澧存い銉︽尵閸掓帡宕奸悢铏规嚌闂侀€炲苯澧撮柕鍡曠窔瀵粙顢橀悙鑼垛偓鍨攽閻愭潙鐏﹂柨姘舵煟鎼搭喖寮慨濠呮缁棃宕卞Δ鈧銉ッ归悩闈涒枅闁哄苯绉烽¨渚€鏌涜箛鏃傘€掔紒顔肩墛閹峰懘鎮剧仦鑹板焻闂傚倸鍊烽懗鍫曞储瑜旈敐鐐哄即閵忕姷锛欏┑鐘绘涧濡參鎮楅懜鐐逛簻闁哄倸鐏濋埛鏃傜棯閹规劖顥夐棁澶愭煥濠靛棙鍣规い銉ョ箻閺屾稒鎯旈敍鍕啋闂佸搫鐭夌紞鈧紒鐘崇洴瀵噣宕掑鍛瘔缂傚倸鍊风欢锟犲闯椤栨粌绶ゅù鐘差儏閻撯€愁熆鐠轰警鐓繛绗哄姂閺屾盯鍩勯崗鐙€浜幆浣割煥閸啿鎷绘繛鎾村焹閸嬫挻绻涙担鍐叉处閸嬪鏌涢埄鍐︿簵婵炴垶菤濡插牊绻涢崱妯哄闁绘繂鍢查—鍐Χ閸℃鐟ㄩ梺绋匡龚瀹曞灚绔熼弴銏犵缂佹妗ㄧ花濠氭⒑閸濆嫬鏆婇柛瀣崌閺屾稒鎯旈姀鐘灆闂佺硶鏂傞崹钘夌暦婵傜鐓戦柍杞版婢规洟鏌ｉ悢鍝ユ噧閻庢凹鍘藉鍕礋椤栨稓鍘遍梺瑙勫礃濞夋盯寮稿☉娆樻闁绘劕顕晶鎶芥煏閸ャ劌濮嶆鐐村浮楠炲鎮欓懠顑惧仩婵犵绱曢崑鎴﹀磹閺嶎厼绠伴柣鎰靛墮閸ㄦ棃鎮楅棃娑欏暈妞ゎ偅娲熼弻鐔煎箚閺夊晝鎾绘煃闁垮绗掗棁澶愭煥濠靛棗顏い鎺斿枛閺岋綁骞掑Δ鍐毇闂佸搫鐭夌换婵嗙暦婵傚壊鏁冮柕蹇曞Л閹奉偊姊绘担绛嬪殐闁哥姵宀稿畷妤€螣鐠佸磭绠氶梺褰掓？缁€渚€鎮″☉妯锋斀闁绘ɑ褰冮弳鐐寸箾閸涱喗灏柍瑙勫灴閹晠顢欓懖鈺€绱橀梻浣虹《閺呮粓鎮ч悩鑼殾婵犻潧顑呴崘鈧銈嗘尵閸犲孩绂?
			const VkExtent2D renderSize = gBuffers.getSize();
			const float imageX = localX / vpSize.x * float(renderSize.width);
			const float imageY = localY / vpSize.y * float(renderSize.height);

			LOGI("[RightClick] screen = (%.1f, %.1f)  viewportLocal = (%.1f, %.1f)  imagePixel = (%.1f, %.1f)  size = %ux%u\n",
				mousePos.x, mousePos.y, localX, localY, imageX, imageY, renderSize.width, renderSize.height);

			debugPushConstant2.pixelIndex = shaderio::uint2(imageX, imageY);
			//Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)GBuffers_Octree_VRPG::eClusterPairResult);
		}
	}

	// Headless/autonomous debug: VRPG_AUTO_PICK="x,y" picks one pixel
	// without any mouse input.
	if (const char* autoPick = std::getenv("VRPG_AUTO_PICK")) {
		int x = 0, y = 0;
		if (std::sscanf(autoPick, "%d,%d", &x, &y) == 2) {
			changeClusterG = true;
			debugPushConstant2.pixelIndex = shaderio::uint2(x, y);
		}
	}
	if (const char* debugCluster = std::getenv("VRPG_DEBUG_CLUSTER")) {
		int cluster = -1;
		if (std::sscanf(debugCluster, "%d", &cluster) == 1 && cluster >= 0) {
			debugClusterIndex = cluster;
			changeClusterG = false;
		}
	}
#endif
}
void Octree_VRPG::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
#ifndef NDEBUG
	NVVK_CHECK(gBuffers.update(cmd, size));
	//婵犵數濮烽弫鍛婃叏閻戣棄鏋侀柛娑橈攻閸欏繘鏌ｉ幋锝嗩棄闁哄绶氶弻娑樷槈濮楀牊鏁鹃梺鍛婄懃缁绘﹢寮婚敐澶婄闁挎繂妫Λ鍕⒑閸濆嫷鍎庣紒鑸靛哺瀵鈽夊Ο閿嬵潔濠殿喗顨呴悧濠囧极妤ｅ啯鈷戦柛娑橈功閹冲啰绱掔紒姗堣€跨€殿喖顭烽弫鎰緞婵犲嫷鍚呴梻浣瑰缁诲倸螞椤撶倣娑㈠礋椤栨稈鎷洪梺鍛婄箓鐎氱兘宕曟惔锝囩＜闁兼悂娼ч崫铏光偓娈垮枛椤兘骞冮姀銈呯閻忓繑鐗楃€氫粙姊虹拠鏌ュ弰婵炰匠鍕彾濠电姴浼ｉ敐澶樻晩闁告挆鍜冪床闂備胶绮崝锕傚礈濞嗘挸绀夐柕鍫濇娴滄粍銇勯幘璺盒㈤柛妯侯嚟閳ь剚顔栭崰鏍€﹂悜钘夋瀬闁归偊鍘肩欢鐐测攽閻樻彃顏柡澶婃啞娣囧﹪鎮欓鍕ㄥ亾閺嶎偅鏆滃┑鐘叉处閸婂潡鏌ㄩ弴鐐测偓鍝ュ閸ф鐓欓柟顖嗗喚鏆㈢紒鐐礃濡嫰婀侀梺鎸庣箓閹冲繘骞嗛崼顫剨闊洦绋掗埛鎴︽煕濞戞﹩鐓繛鍫熸礀閳规垿鎮欑拠褍浼愬銈嗘穿缁插墽鎹㈠┑鍡╂僵妞ゆ挾鍋樼花濠氭⒒娴ｄ警鐒剧紒缁樺姍閹啴鎮滈挊澶岊唵婵犵數濮电喊宥夋偂閻旂厧绠规繛锝庡墮婵¤姤绻涢幊宄板枤閻斿棝鏌ｉ悢宄扮盎鐎规悶鍎靛畷鈩冩綇閵婏絼绨婚梺鍝勫€搁悘婵嬪煕閺冨倻妫柟瑙勫姈椤ュ妫佹径鎰叆婵犻潧妫欓崳娲煙椤栨氨澧︾€规洦鍨跺畷銊︾節閸曨厾妲囬梻浣圭湽閸ㄨ棄顭囪缁傚秷銇愰幒鎾跺幈闁诲函缍嗛崑鍛暦鐏炵偓鍙忓┑鐘插暞閵囨繄鈧娲﹂崑濠傜暦閻旂⒈鏁囬柣妯夸含缁€鍐⒒閸屾瑧顦﹂柟鑺ョ矋閹便劑鎮介崨濠備罕闂佺粯顭堝畷闈涚暤娓氣偓閺屾盯骞囬棃娑樺濠碘槅鍨扮€氫即骞冨Δ鍐╁枂闁告洦鍓涢ˇ銉╂⒑闂堟稓澧涢柟顔煎€块悰顕€宕橀纰辨綂闂侀潧鐗嗛幊宥囨閸洘鈷戦柛娑橈攻婢跺嫰鏌涚€ｎ亝鍤囨い銏★耿閹虫牠鍩￠崘顏庣闯闂備胶顭堥張顒勬嚌妤ｅ啫鐒垫い鎺嶇劍閸婃劗鈧娲橀崝娆撳箖濠婂牊鍤嶉柕澶堝劜閻ｉ亶姊绘担钘変汗妞ゎ厼鐗撻幃妯侯潩鐠鸿櫣鐓戦棅顐㈡处缁嬫帡鎮￠悢鍏肩厪闊洦娲栨牎闂佸摜鍠撻崑銈夊蓟濞戞埃鍋撻敐搴濇喚闁稿孩妫冮弻鈥崇暆鐎ｎ剛鐦堥悗瑙勬礀閻栧吋淇婇悜钘壩ㄧ憸宀勫箖閿濆鈷掑ù锝堟鐢盯鏌熺粙娆剧吋闁轰礁鍟存慨鈧柕鍫濇嚀閹芥洟姊虹紒妯荤叆闁硅绱曠槐鐐哄炊椤剚妫冮弫鎰板川椤撶喐顔夋俊鐐€曟鍝ョ矓閻熼偊娼栭柧蹇撴贡閻瑦绻涢崱妯哄姢闁告捇浜跺娲川婵犲啠鎷归梺鑽ゅ枂閸庢挳鎮樼€ｎ喗鈷戦柛娑橈功閳洟鏌熼姘辩劯鐎殿噮鍋婂畷姗€鍩炴径鍝ョ泿闂備浇顫夋竟瀣疾濞戙垺鍊堕柍鍝勫暟绾惧ジ鏌ｅΟ鐑樻儓闁告艾婀辩槐鎺撴綇閵婏箑闉嶉梺鐟板槻閹虫﹢鐛幘璇茬鐎广儱鎷嬪Λ婊堟⒒閸屾艾鈧绮堟担鍦彾濠电姴娲ょ壕璇层€掑锝呬壕閻庤娲樺ú鐔奉嚕婵犳艾唯闁挎洍鍋撳ù婊勵殔閳规垿鎮欓崣澶樻！闂佸湱顭堥…鐑藉箖閿熺姴鍗抽柕蹇嬪灩瑜板嫰姊洪幖鐐插姌闁告柨绻戠粩鐔肺熸總鐣屾嚀椤劑宕橀鍠般劑姊洪崫鍕効缂佺粯绻傞悾鐑藉础閻戝棙寤洪梺鍓插亾缁叉椽宕戦幘璇查敜婵°倓鑳堕崢鎼佹煟韫囨洖浠滃褑妫勭叅閻庣數纭堕崑鎾斥枔閸喗鐝梺鍛婃尰閻╊垶鐛幋锕€绀嬫い鏍嚤閳哄啯鍠愮€光偓閳ь剙危閹邦儵鐔兼嚒閵堝洨妲囬梻浣圭湽閸ㄨ棄顭囪缁傛帡鏁冮崒娑氬幈闂侀潧顭堥崕宕囩矓閾忓厜鍋撳▓鍨珮闁稿鎳愰幑銏犫攽閸♀晜鍍靛銈嗗笂閻掞箒鈪查梻鍌氬€风欢姘跺焵椤掑倸浠滈柤娲诲灡閺呭爼顢涢悙瀵稿帾闂佹悶鍎滈崘鍙ョ磾婵°倗濮烽崑鐐垫暜閿熺姷宓侀悗锝庝簴閺€浠嬫煙闁缚绨芥俊顐灦濮婂宕掑▎鎴濆婵犵數鍋涢敃顏勵嚕閵娾晜鐒肩€广儱鎳庨悗顓㈡⒑闂堚晛鐦滈柛妯圭矙閿濈偤寮撮姀锛勫幍闂佽鍨庨崨顔藉枛濠电姵顔栭崹鍗烆焽濞嗘挸鐓橀柟杈惧瘜閺佸﹪鏌﹀Ο渚Ш闁告柨鎲＄换娑氣偓娑欘焽閻绱掔拠鑼ⅵ闁绘侗鍣ｅ畷濂稿Ψ閵夊簶鏅犻弻宥嗘姜閹峰苯鍘￠梺绋匡功閸嬨倕顫忓ú顏咁棃婵炴垼浜崝鎼佹⒑缁嬪尅鏀绘い銊ョ墢閸掓帞鎷犲顔藉兊濡炪倖甯掗ˇ鎵偓闈涚焸濮婃椽妫冨☉姘暫濠碘槅鍋呯粙鎺楀疾閸洦鏁嗛柛鏇ㄥ墰閸樹粙姊洪崫鍕殲闁搞劌鐏氶弲鍫曞箻椤旂晫鍘介梺鐟扮仢閸燁偅鏅剁€涙ǜ浜滈柕蹇婃濞堟粎鈧娲樼敮鎺曠亙闂侀€炲苯澧紒鍌氱У閵堬綁宕橀埡浣风敾婵犵數鍋涘Λ妤冩崲閹伴偊鏁傛い蹇撳閻斿棝鎮归崫鍕儓妞ゅ浚鍙冮弻锛勪沪閸撗勫垱濡ょ姷鍋為…鍥箲閸曨垱鍊绘俊顖炴？闁垰鈹戞幊閸婃鎱ㄩ悜钘夌；闁绘劗鍎ら崑瀣煟閹邦剚鎯堝鍛存⒑閸︻厼顣兼繝銏☆焽缁骞庨懞銉у幍闁诲海鏁搁…鍫熺濠婂懐纾煎ù锝堟閻撴劕菐閸パ嶈含闁诡喗鐟╅、鏃堝礋閵娿儰澹曢梺鍝勭▉閸樿偐绮绘导瀛樼厽闁逛即娼ф晶顕€鏌涢妸锔筋棃闁诡喗顨婂Λ鍐ㄢ槈濞嗗繑娈樻俊鐐€ら崑鍛洪悢鐓庤摕婵炴垯鍨圭粻娑欍亜閹捐泛袨闁稿鍋ゅ铏规偘閳ュ厖澹曞┑鐐差嚟婵挳顢栭崱娑欏亗闁绘柨鍚嬮悡娆撴⒑椤撱劎鐣遍柣蹇曞█閺岋綁骞橀弶鎴濐潕闂佺懓寮堕幃鍌炵嵁鎼淬垻绡€闁稿本绋戝▍褏绱掗悙顒€鍔ら柕鍫熸倐瀵鎮㈤搹鍦紲濠碘槅鍨靛▍锝夋偡閺屻儲鈷戠紒瀣儥閸庡秹鏌涙惔銏㈠弨闁靛棗鍟存俊鐑藉煛閸屾埃鍋撻悜鑺ュ仯闁搞儱娲ゅΛ妤呭箚閸喍绻嗛柕鍫濇搐鍟搁梺绋款儑閸嬨倝鍨鹃敃鍌氱倞妞ゆ巻鍋撶紒鐙€鍨堕弻锝呂熼懖鈺佺闂佹悶鍊栭崹鍫曞Φ閸曨垰绠抽柟鎼幗鏁堥梺鍝ュ枑閹倸顫忔繝姘＜婵﹩鍏橀崑鎾搭槹鎼淬埄鍋ㄥ┑顔斤供閸擄箓鎮為悾灞惧枑闁哄啫鐗嗙粻鏍喐閺傝法鏆﹂柛妤冨剱濞撳鏌曡箛銉х？闁宠棄顦甸弻宥夋寠婢舵ɑ笑濡炪値鍋呯换鍫ャ€佸Δ鍛劦妞ゆ帒鍟犻崣銏ゆ⒒閸屾艾鈧娆㈠璺虹劦妞ゆ帒鍊告禒婊堟煠濞茶鐏￠柡鍛閳ь剚绋掕摫闁告瑥绻愰埞鎴︽偐閹绘帗娈查梺闈涙处缁诲牓寮婚敍鍕勃闁兼亽鍎卞▓妤呮⒑鐠団€虫灍闁荤啿鏅犻獮濠囨偐閸偄纾繛鎾村嚬閸垳妲愰弮鈧换婵嬫偨闂堟刀娑㈡煙瀹勬澘鏆ｇ€规洜鍠栭、妤呭焵椤掑嫬鑸瑰鑸靛姈閳锋垿鏌涘┑鍡楊伀闁绘帟娉曠槐鎺撴綇閵婏妇顦ㄧ紓浣稿€圭敮锟犲极閸愵喖纾兼繛鎴炆戦悿鍛存⒒娓氣偓濞佳囁囨禒瀣獥婵°倕鎯ゆ径鎰櫜闁糕剝娲滅粻姘舵⒑缂佹ê濮﹀ù婊勭矒閸┾偓妞ゆ帊鑳舵晶鍨殽閻愭潙濮嶆鐐寸墬閹峰懘鎳栧顓炴噽绾惧ジ鎮楅敐搴″箺缁绢厼鐖奸弻锟犲幢椤撶姷鏆ら梺鍝勬湰濞叉鎹㈠☉銏″€锋い鎺嶈兌瑜扮喖姊绘担鑺ャ€冪紒鈧担铏圭煋闁汇垹鐏氬畷鍙夌節闂堟稒宸濈紒鈾€鍋撻梻浣侯焾閺堫剟宕欑憴鍕浄濡わ絽鍟埛?
	{
		VkSamplerCreateInfo samplerInfo = DEFAULT_VkSamplerCreateInfo;
		samplerInfo.magFilter = VK_FILTER_NEAREST;
		samplerInfo.minFilter = VK_FILTER_NEAREST;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
		Application::samplerPool.acquireSampler(gBuffers.m_res.gBufferDepth.descriptor.sampler, samplerInfo);

		const VkImageLayout layout{ VK_IMAGE_LAYOUT_GENERAL };
		VkImageMemoryBarrier2 barrier = nvvk::makeImageMemoryBarrier({ .image = gBuffers.m_res.gBufferDepth.image,
														.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
														.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
														.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
		const VkDependencyInfo depInfo{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
									   .imageMemoryBarrierCount = 1,
									   .pImageMemoryBarriers = &barrier };

		vkCmdPipelineBarrier2(cmd, &depInfo);

		VkClearDepthStencilValue clearDepth = { 1.0f, 0 };
		VkImageSubresourceRange range = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
		vkCmdClearDepthStencilImage(cmd, gBuffers.m_res.gBufferDepth.image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearDepth, 1, &range);

		// Setting the layout to the final one
		barrier = nvvk::makeImageMemoryBarrier(
			{ .image = gBuffers.m_res.gBufferDepth.image, .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.newLayout = layout, .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
		gBuffers.m_res.gBufferDepth.descriptor.imageLayout = layout;
		vkCmdPipelineBarrier2(cmd, &depInfo);
	}

	debugPushConstant2.pixelIndex = { size.width / 2, size.height / 2 };
#endif
};
void Octree_VRPG::preRender() {
	if (temporalReset) temporalFrameIndex = 0;
	pushConstant.frameIndex = temporalReuse ? int(temporalFrameIndex++) : Application::frameIndex;
	pushConstant.randomFrameIndex = (VRPG_RANDOM_FRAME_SOURCE != 0)
		? int(Application::randomFrameIndex)
		: Application::frameIndex;

	pushConstant.VGBVoxelTotalCount = setting.VGBSize * setting.VGBSize * setting.VGBSize;
	pushConstant.voxelVolume = setting.VGBVoxelSize.x * setting.VGBVoxelSize.y * setting.VGBVoxelSize.z;
	pushConstant.VGBStartPos_Size = glm::vec4(setting.VGBStartPos, setting.VGBSize);
	pushConstant.VGBVoxelSize = glm::vec4(setting.VGBVoxelSize, 1.0f);

	pushConstant.lightVGBStartPos_Size = glm::vec4(setting.lightVGBStartPos, setting.lightVGBSize);
	pushConstant.lightVGBVoxelSize = glm::vec4(setting.lightVGBVoxelSize, 1.0f);
	pushConstant.temporalReuse = temporalReuse ? 1 : 0;
	pushConstant.temporalBlend = temporalBlend;
	pushConstant.clusterFillRateThreshold = clusterFillRateThreshold;


	{
		static constexpr float GOLDEN = 0.6180339887498949f; // 1/phi
		if (rotateOutgoing) {
			float t = static_cast<float>(pushConstant.randomFrameIndex);
			float angle = glm::two_pi<float>() * glm::fract(t * GOLDEN + 0.5f);
			pushConstant.randomRotateMatrix =
				glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0, 0, 1));
		}
		else {
			pushConstant.randomRotateMatrix = glm::mat4(1.0f);
		}
	}

	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
}
void Octree_VRPG::renderStatic(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd, "Octree_render");

	updateDataPerFrame(cmd);

	pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = 256,
		.pValues = &pushConstant,
	};

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	nvvk::WriteSetContainer write{};
	write.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), setting.asManager->asBuilder.tlas);
	vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 1, write.size(), write.data());

	vkCmdPushConstants2(cmd, &pushInfo);

	initOctreeArray(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
	createOctreeArray(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	getOctreeLabel(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	createLightTree(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	// Nearby clusters are needed by the reservoir spatial merge.
	getNearbyClusterInfo(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
}

// Per-frame randomized candidate work: leaf sampling, hit tests and weights.
// The structural G/E trees, labels, light tree, nearby clusters and cut nodes
// are retained from renderStatic().
void Octree_VRPG::renderCandidates(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd, "Octree_candidates");

	updateDataPerFrame(cmd);

	pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = 256,
		.pValues = &pushConstant,
	};

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	nvvk::WriteSetContainer write{};
	write.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), setting.asManager->asBuilder.tlas);
	vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 1, write.size(), write.data());

	vkCmdPushConstants2(cmd, &pushInfo);
	getOctreeNodePairData(cmd);
}
void Octree_VRPG::postProcess(VkCommandBuffer cmd) {
#ifndef NDEBUG
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);
	pushInfo.pValues = &debugPushConstant;

	debugPushConstant.frameIndex = pushConstant.frameIndex;
	debugPushConstant.maxLayer = pushConstant.octreeMaxLayer;
	debugPushConstant.sceneInfoAddress = pushConstant.sceneInfoAddress;
	debugPushConstant2.spatialReuse = spatialReuse;

	debugPushConstant.curLayerNodeCount_G = (1 << (3 * debugPushConstant.curLayer_G));
	debugPushConstant.curLayerNodeCount_G_2 =
		debugPushConstant.curLayer_G_2 > int(pushConstant.octreeMaxLayer)
		? 0 : (1 << (3 * debugPushConstant.curLayer_G_2));
	debugPushConstant.curLayerNodeCount_E = (1 << (3 * debugPushConstant.curLayer_E));

	geometryTreeDebug(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);
	lightTreeDebug(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);
	geometrySVODebug(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);
	clusterPairDebug(cmd);
#endif
};

void Octree_VRPG::createOctreeArray() {
	uint32_t VGBSize = uint32_t(setting.VGBSize);
	octreeMaxLayer = std::countr_zero(VGBSize);	//start from 0
	lightOctreeMaxLayer = std::countr_zero(uint32_t(setting.lightVGBSize));
	if (octreeMaxLayer >= MAX_OCTREE_LAYER_VRPG || lightOctreeMaxLayer >= MAX_OCTREE_LAYER_VRPG)
		throw std::runtime_error("闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ょ紓宥咃躬瀵鎮㈤崗灏栨嫽闁诲酣娼ф竟濠偽ｉ鍓х＜闁绘劦鍓欓崝銈囩磽瀹ュ拑韬€殿喖顭烽弫鎰緞婵犲嫷鍚呴梻浣瑰缁诲倿骞夊☉銏犵缂備焦顭囬崢閬嶆⒑闂堟稓澧曢柟鍐查叄椤㈡棃顢橀姀锛勫幐闁诲繒鍋涙晶钘壝虹€涙ǜ浜滈柕蹇婂墲缁€瀣煛娴ｇ懓濮嶇€规洖宕埢搴♀枎閹寸姭鏁嶉梻鍌氬€搁崐椋庢濮樿泛鐒垫い鎺戝€告禒婊堟煠濞茶鐏￠柡鍛埣瀹曟粏顦寸痪鎯с偢閺岋絽螣閹稿海褰ч柣蹇撴禋閸欏啴寮婚敐澶嬫櫜闁搞儜鍐ㄧ闂備浇顕栭崰鎺楀礈濠靛绠氶柡鍐ㄧ墱閺佸﹪鏌熼鍡楀濮ｅ嫭绻濋悽闈浶ラ柡浣告啞缁绘盯鍩€椤掍胶绠惧ù锝呭暱濞诧箓宕戠€ｎ亖鏀介柣妯哄级閹兼劙鏌涘顒傜Ш闁哄苯绉烽¨渚€鏌涢幘瀵告噰闁炽儲妫冨畷姗€顢欓崲澹洤绠圭紒顔煎帨閸嬫捇鎳犻鈧崵顒傜磽閸屾艾鈧嘲霉閸ヮ剦鏁嬬憸蹇涘焵椤掍胶鐓柛妤佸▕閹即顢氶埀顒勭嵁鐎ｎ喗鏅濋柍褜鍓涚划鍫⑩偓锝庡亖娴滄粓鏌″鍐ㄥ闁愁垱娲熼弻娑欐償椤斿鍋楀┑顔硷龚濞咃綁骞忛锝嗗劅闁炽儴娅曞▓鏌ユ⒑鐟欏嫬绲婚柟璇х節閵嗗啴濡烽埡鍌氣偓閿嬨亜韫囨挸顏ù鐓庣墕閳规垿顢欑涵宄颁紣濡炪値鍘奸崲鏌ユ偩閻戣棄绠抽柟瀛樻⒐閻庡姊虹憴鍕剹闁告ê鈧啿鍙洪梻鍌氬€搁崐宄懊归崶顒婄稏濠㈣泛鐬奸惌鍫ユ煙缂併垹鏋涢柣鎺戠仛閵囧嫰骞掗幋婵囨閻炴熬绠戦埞鎴︽倷閸欏鏋欐繛瀛樼矋缁诲牆顕ｉ幓鎺嗘斀閻庯綆鍋嗛崢閬嶆煟韫囨洖浠ч柛瀣崌閹啴骞嬮敂鐣屽弳濠电偞鍨堕悷褍煤鐎涙ɑ鍙忓┑鐘插暞閵囨繄鈧娲滈崗姗€銆佸鈧幃娆撳箵閹烘繄鈧崬鈹戦敍鍕杭闁稿﹥鍨垮畷婵嗙暆閸曨偆鐤囧┑鈽嗗灥濞咃綁寮抽敃鍌涚厪濠㈣泛鐗嗘俊绋棵瑰鍐Ш闁哄瞼鍠栭幃婊兾熺拠鍙夋闂備礁鎲¤摫闁诡喖鍊搁～蹇撁洪鍕啇闂佺粯鍔栬ぐ鍐╂叏瀹€鍕拺缂佸娼￠崣鍕瑰鍐煟妤犵偛鍟～婊堝焵椤掑嫨鈧線寮崼婵嬪敹闂佺粯鏌ㄩ幖顐︾嵁濡ゅ懏鈷掑ù锝囩摂濞兼劙鏌涙惔銏犫枙闁诡喗妞芥俊姝岊槾闁活厼妫濋弻娑㈠箛闂堟稒鐏嶉梺缁樻尭缁绘﹢寮诲☉銏犖ㄩ柟瀛樼箓閺嬨倕霉閻撳骸鏆欓摶鏍煟濮椻偓濞佳勭濠婂嫮绠剧€瑰壊鍠栭獮鏍煟閿濆懎妲绘い顓滃姂瀹曠喖顢楅埀顒勵敁閹剧粯鈷戦梻鍌氼嚟閻瑩鏌￠埀顒勬焼瀹ュ懎鐎梺闈╁瘜閸樹粙锝為弴銏＄厵闁诡垎鍛喖婵犳鍨遍幐鎶藉蓟閻旇櫣纾兼俊顖濇閻熸劖绻濋姀锝庢綈婵炶尙鍠庨～蹇曠磼濡偐鎳濋梺閫炲苯澧撮柛鈹惧亾濡炪倖甯掗崐鍛婄濠婂牊鐓犳繛鑼额嚙閻忥繝鏌￠崨顓烆暢闁逞屽墯缁嬫帟鎽梺鍛婄懃缁绘劙鍩為幋锔藉亹闁圭粯甯楀▓鏌ユ⒑闂堟稒澶勭紒璇插閸┾偓妞ゆ帒鍠氬鎰箾閹绘帞绠荤€规洘绻傞濂稿炊閵娿儱绨ユ繝鐢靛█濞佳囶敄閸涱垰顥氶柛蹇撳悑閸欏繑淇婇姘儓闁诡喗鍨圭槐鎾愁吋閸涱厸鍋撻崹顕呮綎濠电姵鑹剧壕鍏肩箾閸℃ê绗掗柛妯封偓鎰佹富闁靛牆楠搁獮鏍煟濡や焦宕岄柕鍡曠窔瀵粙顢橀悙鑼垛偓鍨攽閻愭潙鐏︽い顓炴喘椤㈡瑦寰勯幇顓涙嫼闂佺鍋愰崑娑㈠礉閳ь剟鏌ｈ箛鏇炰粶濠⒀呮櫕缁瑦鎷呴搹鍦紳闂佺鏈懝楣冨焵椤掑倸鍘撮柟顔惧仱閺佸啯鎷呯憴鍕濠电姷鏁告慨鐢割敊閺嶎厼鍨傞梺顒€绉寸粻浼存煙閹増顥夐柦鍐枛閺屻劌鈹戦崱鈺傂︾紓浣哄缂嶄線寮婚敐鍫㈢杸闁哄啠鍋撻柕鍥╁枎闇夋繝濠傚閻帡鏌涢埞鍨伈妤犵偞锕㈠鐢稿垂椤愩垹顎忕紓鍌氬€烽悞锔剧矙閹达富鏁勯柛鈩冪憿閸嬫捇妫冨☉姘卞姱濠殿喖锕ュ钘壩涢崘銊㈡婵﹩鍓﹂弳顐ｇ節绾版ɑ顫婇柛瀣嚇閹兘鏁冮埀顒勵敋閵夆晛绀嬫い鏍ㄦ皑椤︽澘顪冮妶鍡欏缂佸鎹囧畷銏ゅ础閻愨晜鏂€闂佺粯鍔曞鍫曀夊鍛＜缂備焦锚婵秶鈧鍠栭崯鍧楀煝鎼淬劌绠婚悗鐢殿焾楠炴姊绘担鐟邦嚋缂佽鍊哥叅闁靛牆顦崒銊╂倵濞戞瑱渚涙繛鍫滅矙閺岋綁骞囬浣叉灆闂佺瀛╁Λ鍐蓟閳╁啯濯寸紒瀣氨閸嬫捇宕稿Δ鈧粻鏉库攽閻樻彃鏆欑紒鍓佸仱閺屾盯鍩勯崘锔挎勃婵犮垼顫夊ú鐔奉潖閾忓湱纾兼俊顖濇娴煎洦绻濆▓鍨灁闁稿﹥绻傞悾鐑藉箛閻楀牆浠洪梺鍛婄☉鑹岄柟宄邦煼濮婃椽宕滈幓鎺嶇敖闂佸摜濮甸幑鍥春閵夛箑绶為悗锝冨妺缁ㄥ姊洪棃娑辨Ф闁稿骸纾划濠氭嚑椤掑倻锛滅紓鍌欑劍椤洤煤鐎电硶鍋撳▓鍨灈闁绘牕銈搁悰顔锯偓锝庝簴閺€浠嬫煙闁缚绨奸柛瀣嚇濮婂宕掑▎鎴М闂佹眹鍊曞ú顓㈡晲閻愭潙绶為柟閭﹀墮閻庮參姊虹粔鍡楀濞堟棃鏌￠崟鈺佸姦闁哄本娲樺鍕幢濡崵褰呮俊銈囧Х閸嬫垿宕归悜妯尖攳濠电姴娲ゅ洿闂佸憡渚楅崢钘夆枔閵娾晜鈷戦柣鐔煎亰閸ょ喖鏌涜箛鏂嗩亪锝炶箛鎾佹椽顢旈崨顓燁吋闂備線娼ч悧鍡涘箟濞戞ǚ妲堥柕蹇ョ磿閸橀亶鏌ｆ惔顖滅У闁稿鎳庨悾宄扮暆閸曨剛鍘搁柣蹇曞仩椤曆勬叏閸屾壕鍋撳▓鍨灍闁瑰憡濞婇獮鍐ㄢ枎瀵版繂婀遍埀顒婄秵娴滄瑦绔熼弴銏″仭婵犲﹤鎳庨。濂告偨椤栨稑绗у瑙勬礃缁绘繂顫濋鐘插箞闂備礁鎼ú銏ゅ礉瀹€鍕€堕柨鐔哄У閻撶喖鏌熼幆褏鎽犵紒鈧€ｎ喗鐓涢悘鐐垫櫕鏁堥梺绯曟杹閸撴繈骞忛崨鏉戝窛濠电姴鍠氶崯鈧梻鍌氬€烽悞锕傚礈濮樿泛鍨傞柛妤冧紳濞差亝鍋勯柣鎾虫唉閹芥洟姊洪幐搴ｇ畵妞わ富鍨崇划缁樼節濮橆厼鈧灚绻涢幋鐐垫噽闁绘帟濮ら妵鍕晜閸濆嫬顫囧┑顔硷攻濡炶棄螞閸愩劉妲堟繛鍡樺姈閸婄兘姊虹拠鎻掝劉闁活収鍣ｅ畷锟犲箮閼恒儱浠掑銈嗘磵閸嬫挾鈧娲栭悥濂搞€佸Δ鍛劦妞ゆ帒瀚崑鍌涚箾閹寸偐妫ㄧ憸鐗堝笚閺呮煡鏌涢銈呮瀭闁逞屽墮閸熸挳寮婚敐澶娢ㄦい鏃囧吹閸戣绻涢敐鍛悙闁挎洦浜獮鍐ㄢ枎閹垮啯鏅滈梺鍛婃磸閸斿本绂嶆ィ鍐╃厵閺夊牆澧介悾杈╃棯閹勫仴闁哄被鍊楅崰濠囧础閻愬樊娼婚梻浣规偠閸斿秶鎹㈤崼銉ヨ摕闁绘梻鍘х粻鏌ユ煙娴煎瓨娑ф繛鍫幘缁辨挻鎷呴崫鍕戯綁鏌ｅΔ渚囨畷缂佸矁椴哥换婵嬪炊椤儸鍥ㄧ厱閻忕偛澧介悡顖氼熆鐟欏嫭绀冨ǎ鍥э躬閹瑩顢旈崟銊ヤ壕闁靛牆顦壕濠氭煕濞戝崬鐏＄€规洘鐓￠弻娑㈩敃閻樻彃濮庣紓浣哄У閻楃娀寮诲澶婄厸濞达絽鎲″▓鏌ユ⒑缁嬫寧鎹ｉ柛鐘崇墵瀵寮撮敍鍕澑闂佹寧绻傚Λ娑氱玻閺冣偓缁绘繈濮€閿濆孩缍堝┑鐐插级缁骸危閹版澘绠婚悹鍥皺閿涙粌鈹戦鏂や緵闁稿繑锕㈠畷鎴﹀箻閹碱厽效闁瑰吋鐣崺鍕磻閵娾晜鈷戦悹鎭掑妼閺嬫柨鈹戦纰卞殶缂侇噯绲借灃闁告侗鍠氶崢鎼佹倵楠炲灝鍔氬Δ鐘虫倐閻涱噣骞囬悧鍫㈠帗閻熸粍绮撳畷婊堝Ω瑜忔稉宥夋煙閹规劦鍤欑紒鐘崇墬缁绘繃绻濋崒婊冾杸闂佺顑傞弲鐘诲箖濮椻偓閹瑩宕归锝囧涧闂備線娼ч悧鐐电不閹炬剚娼栫紓浣股戞刊鎾煕濞戞﹫榫氱痪顓涘亾闂傚倷鑳堕…鍫㈣姳濞差亜纾规繝闈涙閺嗭箓鏌ｉ姀鐘冲暈闁哄懏绻堥弻娑氫沪閻愵剛娈ら梺娲诲亜鐎氭澘顫忓ú顏勭闁圭粯甯婄花鑲╃磽娴ｇ瓔鍤欓悗姘緲閻ｅ嘲鈹戠€ｎ偅娅囬梺閫炲苯澧撮柍銉︽瀹曟﹢顢欓崲澹洦鐓曢柍鈺佸枤濞堟﹢鏌ｉ悢鏉戝婵﹨娅ｇ划娆撳礌閳ュ啿顫犵紓鍌欑贰閸ｎ噣宕瑰畷鍥у灊婵炲棙鍔曠欢鐐测攽閻樻煡顎楅柟顔藉灴濮婃椽骞栭悙鎻掑闂佸憡鏌ㄩ惌鍌氼嚕閹间礁绠ｉ柨鏃囆掗幏铏圭磽閸屾瑧鍔嶉柨鏇楁櫅閳绘捇寮崼鐔哄幗濠德板€愰崑鎾翠繆椤愶絿銆掗柛鎺撳浮椤㈡﹢濮€閻樻鍞洪梻浣告惈閸燁偊宕愭繝姘櫖婵犲﹤鐗婇埛鎴犵磽娴ｈ鐒介柟鍐插閺岋綁鎮㈤弶鎴濆闁绘挶鍊濋弻銊╁即閻愭祴鍋撹ぐ鎺戠；闁稿瞼鍋為悡娑㈡煕閵夈垺娅呴柡瀣€搁湁婵犲﹤瀚惌鎺楁煛瀹€鈧崰鏍ь嚕閸洖鍨傛い鏇炴噺濞堟椽姊绘担绛嬪殐闁哥姵鐩畷鎴﹀箛椤旂瓔娼熼梺鍦劋椤ㄥ棝宕戦幇鐗堢叄闊浄绲芥禍婊呯棯閻愵剙鈻曟慨濠冩そ瀹曨偊宕熼纰变純缂傚倷绀侀ˇ顖滅礊婵犲偆鍤曞┑鐘宠壘鎯熼悷婊冾樀瀹曟垿骞樼紒妯绘珳闁硅偐琛ラ埀顒冨皺閻栭亶姊绘担鍛婅础闁哄苯顦悾鐑芥偨閸偒娼熼梺瑙勫劤閻°劍鍒婇弶鎳ㄥ綊鎮╁顔煎壈闂佸湱鏅弫濠氬蓟閿濆顫呴柍杞拌兌娴犳岸姊洪柅鐐茶嫰婢ь喚鐥紒銏犲籍鐎殿喗濞婇弫鍐磼濞戞艾寮虫繝鐢靛█濞佳兾涘▎鎾崇柈闁圭儤顨嗛悡鐔兼煛閸愩劌鈧摜鏁捄濂藉綊鎳栭埡浣叉瀰闂佸搫鏈惄顖氼嚕閹绢喖惟闁靛鍎抽澶愭⒒娴ｄ警鐒炬い鎴濇閹囨偐瀹割喗缍?-" + std::to_string(MAX_OCTREE_LAYER_VRPG - 1));

	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();

	octreeNodeDataBuffer_G.resize(octreeMaxLayer + 1);
	octreeNodeInfoBuffer_G.resize(octreeMaxLayer + 1);
	octreeNodeDataBuffer_E.resize(octreeMaxLayer + 1);
	uint32_t layerNodeCount = 6;
	for (int layerIndex = 0; layerIndex <= octreeMaxLayer; ++layerIndex) {
		uint32_t bufferSize = layerNodeCount * sizeof(shaderio::OctreeNodeData_G_VRPG);
		allocator->createBuffer(octreeNodeDataBuffer_G[layerIndex], bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(octreeNodeDataBuffer_G[layerIndex].buffer);

		bufferSize = layerNodeCount * sizeof(shaderio::OctreeNodeInfo_G_VRPG);
		allocator->createBuffer(octreeNodeInfoBuffer_G[layerIndex], bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(octreeNodeInfoBuffer_G[layerIndex].buffer);

		bufferSize = layerNodeCount * sizeof(shaderio::OctreeNodeData_E_VRPG);
		allocator->createBuffer(octreeNodeDataBuffer_E[layerIndex], bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(octreeNodeDataBuffer_E[layerIndex].buffer);

		layerNodeCount *= 8;
	}
	//-----------------------------------------------------------------------------------------------------------------------------------------
	uint32_t bufferSize = 0;
	{
		bufferSize = uint32_t(pow(8, octreeMaxLayer)) * 6 / 8 * sizeof(uint32_t);
		allocator->createBuffer(blockInfoBuffer_G, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(blockInfoBuffer_G.buffer);

		allocator->createBuffer(blockInfoBuffer_E, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(blockInfoBuffer_E.buffer);

		bufferSize = int(pow(8, octreeMaxLayer)) * 6 / 8 * sizeof(uint32_t);
		allocator->createBuffer(hasDataBlockIndexBuffer_G, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(hasDataBlockIndexBuffer_G.buffer);

		allocator->createBuffer(hasDataBlockIndexBuffer_E, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(hasDataBlockIndexBuffer_E.buffer);

		bufferSize = sizeof(shaderio::HasDataOctreeBlockCount_VRPG);
		allocator->createBuffer(hasDataBlockCountBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(hasDataBlockCountBuffer.buffer);

		bufferSize = sizeof(shaderio::OctreeGlobalInfo_VRPG);
		allocator->createBuffer(globalInfoBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT);
		NVVK_DBG_NAME(globalInfoBuffer.buffer);
	}
	//-----------------------------------------------------------------------------------------------------------------------------------------
	{
		bufferSize = sizeof(shaderio::uint2) * ((IndivisibleNodeCount_G_VRPG + 7) / 8);
		allocator->createBuffer(divisibleNodeInfoBuffer_G, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(divisibleNodeInfoBuffer_G.buffer);

		uint32_t maxLayerNodeCount = (1 << (3 * octreeMaxLayer)) * 6;
		bufferSize = sizeof(shaderio::OctreeThreadGroupInfo_VRPG) * ((maxLayerNodeCount + GETOCTREELABEL_CS_THREADGROUP_SIZE - 1) / GETOCTREELABEL_CS_THREADGROUP_SIZE);
		allocator->createBuffer(threadGroupInfoBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(threadGroupInfoBuffer.buffer);

		bufferSize = IndivisibleNodeCount_G_VRPG * sizeof(shaderio::uint2);
		allocator->createBuffer(indivisibleNodeInfosBuffer_G, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(indivisibleNodeInfosBuffer_G.buffer);
	}
	//-----------------------------------------------------------------------------------------------------------------------------------------
	{
		bufferSize = IndivisibleNodeCount_G_VRPG * Candidate_Samples_Count * sizeof(shaderio::uint);
		allocator->createBuffer(adjClusterInfoBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(adjClusterInfoBuffer.buffer);
	}
	//-----------------------------------------------------------------------------------------------------------------------------------------
	{
		bufferSize = IndivisibleNodeCount_G_VRPG * Candidate_Samples_Count * sizeof(shaderio::ClusterPairInfo_VRPG);
		allocator->createBuffer(clusterPairInfoBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(clusterPairInfoBuffer.buffer);

		bufferSize = sizeof(shaderio::ClusterPairGlobalInfo_VRPG);
		allocator->createBuffer(clusterPairGlobalInfoBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(clusterPairGlobalInfoBuffer.buffer);

		bufferSize = IndivisibleNodeCount_G_VRPG * Candidate_Samples_Count * HITTEST_COUNT_VRPG * sizeof(shaderio::ClusterPairHitInfo_VRPG);
		allocator->createBuffer(clusterPairHitInfoBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(clusterPairHitInfoBuffer.buffer);
	}
	//-----------------------------------------------------------------------------------------------------------------------------------------
	{
		bufferSize = IndivisibleNodeCount_G_VRPG * Candidate_Samples_Count * sizeof(shaderio::CandidateSubTreeInfo_VRPG);
		allocator->createBuffer(candidateSubTreeInfoBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(candidateSubTreeInfoBuffer.buffer);

		bufferSize = IndivisibleNodeCount_G_VRPG * Candidate_Samples_Count * sizeof(shaderio::CandidateLeafNodeData_VRPG);
		allocator->createBuffer(candidateLeafNodeDataBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(candidateLeafNodeDataBuffer.buffer);
	}
	//-----------------------------------------------------------------------------------------------------------------------------------------
	{
		bufferSize = OUTGOING_COUNT_VRPG * IndivisibleNodeCount_G_VRPG * Candidate_Samples_Count * sizeof(shaderio::CandidateSampleData_VRPG);
		allocator->createBuffer(candidateSampleDataBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(candidateSampleDataBuffer.buffer);

		bufferSize = OUTGOING_COUNT_VRPG * IndivisibleNodeCount_G_VRPG * Candidate_Samples_Count * sizeof(shaderio::CandidateSampleData_VRPG);
		allocator->createBuffer(candidateSampleDataHistoryBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(candidateSampleDataHistoryBuffer.buffer);

		bufferSize = OUTGOING_COUNT_VRPG * IndivisibleNodeCount_G_VRPG * sizeof(shaderio::ReservoirSample_VRPG);
		allocator->createBuffer(spatialReservoirBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(spatialReservoirBuffer.buffer);
		allocator->createBuffer(reservoirCurrentBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(reservoirCurrentBuffer.buffer);
		allocator->createBuffer(reservoirHistoryBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(reservoirHistoryBuffer.buffer);

		bufferSize = OUTGOING_COUNT_VRPG * IndivisibleNodeCount_G_VRPG * Candidate_Samples_Count * sizeof(float);
		allocator->createBuffer(candidateSamplePrefixWSumBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(candidateSamplePrefixWSumBuffer.buffer);

		bufferSize = OUTGOING_COUNT_VRPG * IndivisibleNodeCount_G_VRPG * sizeof(int);
		allocator->createBuffer(candidateSampleIndexBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(candidateSampleIndexBuffer.buffer);

		bufferSize = OUTGOING_COUNT_VRPG * IndivisibleNodeCount_G_VRPG * Candidate_Samples_Count * sizeof(shaderio::AABB);
		allocator->createBuffer(candidateSampleAABBBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(candidateSampleAABBBuffer.buffer);

		bufferSize = OUTGOING_COUNT_VRPG * IndivisibleNodeCount_G_VRPG * Candidate_Samples_Count * sizeof(shaderio::CandidateSampleData_VRPG);
		allocator->createBuffer(candidateSampleData2Buffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(candidateSampleData2Buffer.buffer);
	}
	//-----------------------------------------------------------------------------------------------------------------------------------------
	{
		bufferSize = IndivisibleNodeCount_G_VRPG * (IndivisibleNodeCount_G_VRPG + GETNEARBYNODES_CS_THREADGROUP_SIZE - 1) / GETNEARBYNODES_CS_THREADGROUP_SIZE * sizeof(shaderio::IndivisibleNodeNearbyNodeTempInfo_VRPG);
		allocator->createBuffer(nearbyNodeTempInfoBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(nearbyNodeTempInfoBuffer.buffer);

		bufferSize = IndivisibleNodeCount_G_VRPG * sizeof(shaderio::OctreeNearbyNodeInfo_VRPG);
		allocator->createBuffer(nearbyNodeInfoBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(nearbyNodeInfoBuffer.buffer);
	}
	//-----------------------------------------------------------------------------------------------------------------------------------------
	{
#if VRPG_Candidate_Samples_SampleMode == 1
		bufferSize = IndivisibleNodeCount_G_VRPG * sizeof(shaderio::CandidatePoolInfo_VRPG);
		allocator->createBuffer(candidatePoolInfoBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(candidatePoolInfoBuffer.buffer);

		bufferSize = IndivisibleNodeCount_G_VRPG * Candidate_Samples_Count * sizeof(shaderio::CandidatePoolNodeInfo_VRPG);
		allocator->createBuffer(candidatePoolNodeInfoBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(candidatePoolNodeInfoBuffer.buffer);
#endif
	}

	pushConstant.octreeMaxLayer = octreeMaxLayer;
	pushConstant.lightOctreeMaxLayer = lightOctreeMaxLayer;
	pushConstant.octreeNodeTotalCount = int(pow(8, octreeMaxLayer + 1) - 1) / 7 * 6;

#ifndef NDEBUG
	allocator->createBuffer(clusterPairDebugGlobalInfoBuffer, sizeof(shaderio::ClusterPairDebugGlobalInfo_VRPG),
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(clusterPairDebugGlobalInfoBuffer.buffer);

	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();
	vkCmdFillBuffer(cmd, clusterPairDebugGlobalInfoBuffer.buffer, 0, sizeof(shaderio::ClusterPairDebugGlobalInfo_VRPG), -1);
	Application::app->submitAndWaitTempCmdBuffer(cmd);
#endif
}
void Octree_VRPG::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;

	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eVGB,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)setting.VGBs.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eLightVGB,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)setting.lightVGBs.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	{
		bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eOctreeNodeData_G,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)octreeNodeDataBuffer_G.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eOctreeNodeInfo_G,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = (uint32_t)octreeNodeInfoBuffer_G.size(),
			.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eOctreeNodeData_E,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = (uint32_t)octreeNodeDataBuffer_E.size(),
			.stageFlags = VK_SHADER_STAGE_ALL });
	}
	{
		bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eBlockInfos_G,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eBlockInfos_E,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eHasDataBlockIndices_G,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eHasDataBlockIndices_E,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eHasDataBlockCount,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eGlobalInfo,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
	}
	{
		bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eDivisibleNodeInfos_G,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eThreadGroupInfos,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eIndivisibleNodeInfos_G,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
	}
	{
		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eAdjClusterInfo_G,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
	}
	{
		bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eClusterPairInfo_VRPG,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eClusterPairGlobalInfo_VRPG,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eClusetPairHitTestInfo,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
	}
	{
		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eCandidateSubTreeInfo_VRPG,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eCandidateLeafNodeData_VRPG,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
	}
	{
		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eCandidateSampleData,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eCandidateSamplePrefixWSum,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eCandidateSampleIndex,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eCandidateSampleAABB,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eCandidateSampleData2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
	}
	{
		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eNearbyNodeTempInfos,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eNearbyNodeInfos,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eCandidateSampleDataHistory,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		for (auto binding : { shaderio::BindingPoints_Octree_VRPG::eSpatialReservoir,
							 shaderio::BindingPoints_Octree_VRPG::eReservoirCurrent,
							 shaderio::BindingPoints_Octree_VRPG::eReservoirHistory }) {
			bindings.addBinding({
				.binding = (uint32_t)binding,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_ALL });
		}
	}
	{
#if VRPG_Candidate_Samples_SampleMode == 1
		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eCandidatePoolInfo_VRPG,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::BindingPoints_Octree_VRPG::eCandidatePoolNodeInfo_VRPG,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
#endif
	}

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	NVVK_DBG_NAME(staticDescPack.getLayout());
	NVVK_DBG_NAME(staticDescPack.getPool());
	NVVK_DBG_NAME(staticDescPack.getSet(0));

	bindings.clear();
	bindings.addBinding({
			.binding = shaderio::DynamicSetBindingPoints_PT::eTlas_PT,
			.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL
		});
	dynamicDescPack.init(bindings, Application::app->getDevice(), 0, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT);
}
void Octree_VRPG::createDescriptorSet() {
	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    VGBWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eVGB, 0, 0, setting.VGBs.size());
	nvvk::Buffer* VGBsPtr = setting.VGBs.data();
	write.append(VGBWrite, VGBsPtr);

	VkWriteDescriptorSet lightVGBWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eLightVGB, 0, 0, setting.lightVGBs.size());
	nvvk::Buffer* lightVGBsPtr = setting.lightVGBs.data();
	write.append(lightVGBWrite, lightVGBsPtr);

	{
		VkWriteDescriptorSet    OctreeArrayWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eOctreeNodeData_G, 0, 0, octreeNodeDataBuffer_G.size());
		nvvk::Buffer* octreeArraysPtr = octreeNodeDataBuffer_G.data();
		write.append(OctreeArrayWrite, octreeArraysPtr);

		OctreeArrayWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eOctreeNodeInfo_G, 0, 0, octreeNodeInfoBuffer_G.size());
		octreeArraysPtr = octreeNodeInfoBuffer_G.data();
		write.append(OctreeArrayWrite, octreeArraysPtr);

		OctreeArrayWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eOctreeNodeData_E, 0, 0, octreeNodeDataBuffer_E.size());
		octreeArraysPtr = octreeNodeDataBuffer_E.data();
		write.append(OctreeArrayWrite, octreeArraysPtr);
	}
	{
		VkWriteDescriptorSet    HasDataInfoWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eBlockInfos_G, 0, 0, 1);
		write.append(HasDataInfoWrite, blockInfoBuffer_G, 0, blockInfoBuffer_G.bufferSize);

		HasDataInfoWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eBlockInfos_E, 0, 0, 1);
		write.append(HasDataInfoWrite, blockInfoBuffer_E, 0, blockInfoBuffer_E.bufferSize);

		HasDataInfoWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eHasDataBlockIndices_G, 0, 0, 1);
		write.append(HasDataInfoWrite, hasDataBlockIndexBuffer_G, 0, hasDataBlockIndexBuffer_G.bufferSize);

		HasDataInfoWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eHasDataBlockIndices_E, 0, 0, 1);
		write.append(HasDataInfoWrite, hasDataBlockIndexBuffer_E, 0, hasDataBlockIndexBuffer_E.bufferSize);

		HasDataInfoWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eHasDataBlockCount, 0, 0, 1);
		write.append(HasDataInfoWrite, hasDataBlockCountBuffer, 0, hasDataBlockCountBuffer.bufferSize);

		VkWriteDescriptorSet    GlobalInfoWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eGlobalInfo, 0, 0, 1);
		write.append(GlobalInfoWrite, globalInfoBuffer, 0, globalInfoBuffer.bufferSize);
	}
	{
		VkWriteDescriptorSet    LabelInfoWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eDivisibleNodeInfos_G, 0, 0, 1);
		write.append(LabelInfoWrite, divisibleNodeInfoBuffer_G, 0, divisibleNodeInfoBuffer_G.bufferSize);

		LabelInfoWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eThreadGroupInfos, 0, 0, 1);
		write.append(LabelInfoWrite, threadGroupInfoBuffer, 0, threadGroupInfoBuffer.bufferSize);

		VkWriteDescriptorSet    IndivisibleInfoWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eIndivisibleNodeInfos_G, 0, 0, 1);
		write.append(IndivisibleInfoWrite, indivisibleNodeInfosBuffer_G, 0, indivisibleNodeInfosBuffer_G.bufferSize);
	}
	{
		VkWriteDescriptorSet    adjClusterInfoWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eAdjClusterInfo_G, 0, 0, 1);
		write.append(adjClusterInfoWrite, adjClusterInfoBuffer, 0, adjClusterInfoBuffer.bufferSize);
	}
	{
		VkWriteDescriptorSet 	candidatePairsWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eClusterPairInfo_VRPG, 0, 0, 1);
		write.append(candidatePairsWrite, clusterPairInfoBuffer, 0, clusterPairInfoBuffer.bufferSize);

		candidatePairsWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eClusterPairGlobalInfo_VRPG, 0, 0, 1);
		write.append(candidatePairsWrite, clusterPairGlobalInfoBuffer, 0, clusterPairGlobalInfoBuffer.bufferSize);

		candidatePairsWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eClusetPairHitTestInfo, 0, 0, 1);
		write.append(candidatePairsWrite, clusterPairHitInfoBuffer, 0, clusterPairHitInfoBuffer.bufferSize);
	}
	{
		VkWriteDescriptorSet candidateNodesWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eCandidateSubTreeInfo_VRPG, 0, 0, 1);
		write.append(candidateNodesWrite, candidateSubTreeInfoBuffer, 0, candidateSubTreeInfoBuffer.bufferSize);

		candidateNodesWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eCandidateLeafNodeData_VRPG, 0, 0, 1);
		write.append(candidateNodesWrite, candidateLeafNodeDataBuffer, 0, candidateLeafNodeDataBuffer.bufferSize);
	}
	{
		VkWriteDescriptorSet candidateNodesWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eCandidateSampleData, 0, 0, 1);
		write.append(candidateNodesWrite, candidateSampleDataBuffer, 0, candidateSampleDataBuffer.bufferSize);

		candidateNodesWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eCandidateSamplePrefixWSum, 0, 0, 1);
		write.append(candidateNodesWrite, candidateSamplePrefixWSumBuffer, 0, candidateSamplePrefixWSumBuffer.bufferSize);

		candidateNodesWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eCandidateSampleIndex, 0, 0, 1);
		write.append(candidateNodesWrite, candidateSampleIndexBuffer, 0, candidateSampleIndexBuffer.bufferSize);

		candidateNodesWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eCandidateSampleAABB, 0, 0, 1);
		write.append(candidateNodesWrite, candidateSampleAABBBuffer, 0, candidateSampleAABBBuffer.bufferSize);

		candidateNodesWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eCandidateSampleData2, 0, 0, 1);
		write.append(candidateNodesWrite, candidateSampleData2Buffer, 0, candidateSampleData2Buffer.bufferSize);
	}
	{
		VkWriteDescriptorSet nearbyNodesWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eNearbyNodeTempInfos, 0, 0, 1);
		write.append(nearbyNodesWrite, nearbyNodeTempInfoBuffer, 0, nearbyNodeTempInfoBuffer.bufferSize);

		nearbyNodesWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eNearbyNodeInfos, 0, 0, 1);
		write.append(nearbyNodesWrite, nearbyNodeInfoBuffer, 0, nearbyNodeInfoBuffer.bufferSize);

		nearbyNodesWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eCandidateSampleDataHistory, 0, 0, 1);
		write.append(nearbyNodesWrite, candidateSampleDataHistoryBuffer, 0, candidateSampleDataHistoryBuffer.bufferSize);

		nearbyNodesWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eSpatialReservoir, 0, 0, 1);
		write.append(nearbyNodesWrite, spatialReservoirBuffer, 0, spatialReservoirBuffer.bufferSize);
		nearbyNodesWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eReservoirCurrent, 0, 0, 1);
		write.append(nearbyNodesWrite, reservoirCurrentBuffer, 0, reservoirCurrentBuffer.bufferSize);
		nearbyNodesWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eReservoirHistory, 0, 0, 1);
		write.append(nearbyNodesWrite, reservoirHistoryBuffer, 0, reservoirHistoryBuffer.bufferSize);
	}
	{
#if VRPG_Candidate_Samples_SampleMode == 1
		VkWriteDescriptorSet getCandidateNodesWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eCandidatePoolInfo_VRPG, 0, 0, 1);
		write.append(getCandidateNodesWrite, candidatePoolInfoBuffer, 0, candidatePoolInfoBuffer.bufferSize);

		getCandidateNodesWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_VRPG::eCandidatePoolNodeInfo_VRPG, 0, 0, 1);
		write.append(getCandidateNodesWrite, candidatePoolNodeInfoBuffer, 0, candidatePoolNodeInfoBuffer.bufferSize);
#endif
	}

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void Octree_VRPG::createPipeline() {
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = 256
	};

	std::array<VkDescriptorSetLayout, 2> layouts = { {staticDescPack.getLayout(), dynamicDescPack.getLayout()} };
	const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = layouts.size(),
		.pSetLayouts = layouts.data(),
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};
	NVVK_CHECK(vkCreatePipelineLayout(Application::app->getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout));
	NVVK_DBG_NAME(pipelineLayout);
}
void Octree_VRPG::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);


	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource;
	VkShaderModuleCreateInfo shaderCode;

	// Slang rounds the constant-block size up to 256 bytes for this layout.
	// The declared range must cover the whole shader block, while the C++
	// pushConstant payload (sizeof) can stay at its natural size.
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = 256,
	};

	std::array<VkDescriptorSetLayout, 2> layouts = { {staticDescPack.getLayout(), dynamicDescPack.getLayout()} };
	VkShaderCreateInfoEXT shaderInfo{
		.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
		.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
		.pName = "main",
		.setLayoutCount = layouts.size(),
		.pSetLayouts = layouts.data(),
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};
	VkDevice device = Application::app->getDevice();
	//--------------------------------------------------------------------------------------
	{
		shaderSource = shaderPath / "Clustering.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_initOctreeArray, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_initOctreeArray";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_initOctreeArray);
		NVVK_DBG_NAME(computeShader_initOctreeArray);

		vkDestroyShaderEXT(device, computeShader_initHasDataBlockInfo, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_initHasDataBlockInfo";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_initHasDataBlockInfo);
		NVVK_DBG_NAME(computeShader_initHasDataBlockInfo);

		vkDestroyShaderEXT(device, computeShader_getGlobalInfo, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getGlobalInfo";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getGlobalInfo);
		NVVK_DBG_NAME(computeShader_getGlobalInfo);

		vkDestroyShaderEXT(device, computeShader_createOctreeArray, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createOctreeArray";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createOctreeArray);
		NVVK_DBG_NAME(computeShader_createOctreeArray);

		vkDestroyShaderEXT(device, computeShader_createOctreeArray2, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createOctreeArray2";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createOctreeArray2);
		NVVK_DBG_NAME(computeShader_createOctreeArray2);
	}
	//---------------------------------getOctreeIndivisibleNodeLabel-------------------------------------
	{
		shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
		shaderSource = shaderPath / "CreateSVO.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_getOctreeLabel1, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getOctreeLabel1";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getOctreeLabel1);
		NVVK_DBG_NAME(computeShader_getOctreeLabel1);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_getOctreeLabel2, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getOctreeLabel2";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getOctreeLabel2);
		NVVK_DBG_NAME(computeShader_getOctreeLabel2);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_getOctreeLabel3, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getOctreeLabel3";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getOctreeLabel3);
		NVVK_DBG_NAME(computeShader_getOctreeLabel3);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_getOctreeLabel4, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getOctreeLabel4";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getOctreeLabel4);
		NVVK_DBG_NAME(computeShader_getOctreeLabel4);
	}
	//---------------------------------light E-tree-------------------------------------
	{
		shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
		shaderSource = shaderPath / "LightTree.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_initLightTreeLeaves, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_initLightTreeLeaves";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_initLightTreeLeaves);
		NVVK_DBG_NAME(computeShader_initLightTreeLeaves);

		vkDestroyShaderEXT(device, computeShader_aggregateLightTree, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_aggregateLightTree";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_aggregateLightTree);
		NVVK_DBG_NAME(computeShader_aggregateLightTree);
	}
	//---------------------------------getOctreeNodePairData-------------------------------------
	{
		shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
		shaderSource = shaderPath / "GetClusterWeight.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_initWeights, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_initCandidateNodes";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_initWeights);
		NVVK_DBG_NAME(computeShader_initWeights);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_getCandidateNodes, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getCandidateNodes";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getCandidateNodes);
		NVVK_DBG_NAME(computeShader_getCandidateNodes);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_dispatchGetCandidateLeafNodes, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_dispatchGetCandidateLeafNodes";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_dispatchGetCandidateLeafNodes);
		NVVK_DBG_NAME(computeShader_dispatchGetCandidateLeafNodes);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_getCandidateLeafNodes, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getCandidateLeafNodes";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getCandidateLeafNodes);
		NVVK_DBG_NAME(computeShader_getCandidateLeafNodes);
		//--------------------------------------------------------------------------------------
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_clusterPairHitTest, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_clusterPairHitTest";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_clusterPairHitTest);
		NVVK_DBG_NAME(computeShader_clusterPairHitTest);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_getCandidateNodeWeights, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getCandidateNodeWeight";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getCandidateNodeWeights);
		NVVK_DBG_NAME(computeShader_getCandidateNodeWeights);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_buildSpatialReservoir, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_buildSpatialReservoir";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_buildSpatialReservoir);
		NVVK_DBG_NAME(computeShader_buildSpatialReservoir);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_reuseReservoir, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_reuseReservoir";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_reuseReservoir);
		NVVK_DBG_NAME(computeShader_reuseReservoir);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_temporalReuse, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_temporalReuse";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_temporalReuse);
		NVVK_DBG_NAME(computeShader_temporalReuse);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_createCandidatePrefixWSum, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createCandidatePrefixWSum";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createCandidatePrefixWSum);
		NVVK_DBG_NAME(computeShader_createCandidatePrefixWSum);
	}
	{
#ifdef SpatialReuse_VRPG
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_reservoirSample, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_reservoirSample";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_reservoirSample);
		NVVK_DBG_NAME(computeShader_reservoirSample);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_SpatialReuse, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_SpatialReuse";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_SpatialReuse);
		NVVK_DBG_NAME(computeShader_SpatialReuse);
		//--------------------------------------------------------------------------------------
		shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
		shaderSource = shaderPath / "GetAdjClusterG.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_getAdjClusterG, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getAdjClusterG";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getAdjClusterG);
		NVVK_DBG_NAME(computeShader_getAdjClusterG);
#endif
	}
	{
#if VRPG_Candidate_Samples_SampleMode == 1
		shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
		shaderSource = shaderPath / "GetCandidateNodes.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_getCandidateNodes_init, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getCandidateNodes_init";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getCandidateNodes_init);
		NVVK_DBG_NAME(computeShader_getCandidateNodes_init);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_getCandidateNodes_1, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getCandidateNodes_1";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getCandidateNodes_1);
		NVVK_DBG_NAME(computeShader_getCandidateNodes_1);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_getCandidateNodes_2, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getCandidateNodes_2";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getCandidateNodes_2);
		NVVK_DBG_NAME(computeShader_getCandidateNodes_2);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_getCandidateNodes_3, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getCandidateNodes_3";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getCandidateNodes_3);
		NVVK_DBG_NAME(computeShader_getCandidateNodes_3);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_getCandidateNodes_4, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getCandidateNodes_4";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getCandidateNodes_4);
		NVVK_DBG_NAME(computeShader_getCandidateNodes_4);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_getCandidateNodes_5, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getCandidateNodes_5";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getCandidateNodes_5);
		NVVK_DBG_NAME(computeShader_getCandidateNodes_5);
#endif
	}
	//---------------------------------getNearbyNode-------------------------------------
	{
		shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
		shaderSource = shaderPath / "GetAdjClusterG.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_dispatchGetNearbyClusterG, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_dispatchGetNearbyClusterG";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_dispatchGetNearbyClusterG);
		NVVK_DBG_NAME(computeShader_dispatchGetNearbyClusterG);

		vkDestroyShaderEXT(device, computeShader_getNearbyClusterG1, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getNearbyNodes1";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getNearbyClusterG1);
		NVVK_DBG_NAME(computeShader_getNearbyClusterG1);

		vkDestroyShaderEXT(device, computeShader_getNearbyClusterG2, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getNearbyNodes2";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getNearbyClusterG2);
		NVVK_DBG_NAME(computeShader_getNearbyClusterG2);
	}
	//---------------------------------Debug-------------------------------------
	{
		shaderSource = shaderPath / "Debug.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, vertexShader_TreeDebug, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.pName = "vertexMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &vertexShader_TreeDebug);
		NVVK_DBG_NAME(vertexShader_TreeDebug);

		vkDestroyShaderEXT(device, fragmentShader_TreeDebug, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "fragmentMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &fragmentShader_TreeDebug);
		NVVK_DBG_NAME(fragmentShader_TreeDebug);
	}
	{
		shaderSource = shaderPath / "Debug_ClusterPair.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_GetDebugClusterG, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_GetDebugClusterG";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_GetDebugClusterG);
		NVVK_DBG_NAME(computeShader_GetDebugClusterG);

		vkDestroyShaderEXT(device, vertexShader_ClusterPairDebug, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.pName = "vertexMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &vertexShader_ClusterPairDebug);
		NVVK_DBG_NAME(vertexShader_ClusterPairDebug);

		vkDestroyShaderEXT(device, fragmentShader_ClusterPairDebug, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "fragmentMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &fragmentShader_ClusterPairDebug);
		NVVK_DBG_NAME(fragmentShader_ClusterPairDebug);
	}
}
void Octree_VRPG::updateDataPerFrame(VkCommandBuffer cmd) {}

void Octree_VRPG::initOctreeArray(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initOctreeArray);

	VkExtent2D groupSize = nvvk::getGroupCounts({ pushConstant.octreeNodeTotalCount, 1 }, VkExtent2D{ 1024, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}
void Octree_VRPG::createOctreeArray(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;

	uint32_t layerBlockCount = uint32_t(pow(8, pushConstant.octreeMaxLayer)) * 6 / 8;
	for (int layerIndex = pushConstant.octreeMaxLayer; layerIndex > OCTREE_CLUSTER_LAYER_VRPG; --layerIndex) {
		pushConstant.currentLayer = layerIndex;
		pushConstant.currentLayerBlockCount = layerBlockCount;
		vkCmdPushConstants2(cmd, &pushInfo);

		vkCmdFillBuffer(cmd, hasDataBlockCountBuffer.buffer, 0, sizeof(shaderio::HasDataOctreeBlockCount_VRPG), 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initHasDataBlockInfo);
		VkExtent2D groupSize = nvvk::getGroupCounts({ layerBlockCount, 1 }, VkExtent2D{ 1024, 1 });
		vkCmdDispatch(cmd, groupSize.width, 1, 1);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getGlobalInfo);
		vkCmdDispatch(cmd, 1, 1, 1);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createOctreeArray);
		vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);

		layerBlockCount /= 8;
	}

	for (int layerIndex = OCTREE_CLUSTER_LAYER_VRPG; layerIndex > 0; --layerIndex) {
		uint32_t layerNodeCount = shaderio::OctreeLayerNodeCount_VRPG[layerIndex];
		pushConstant.currentLayer = layerIndex;
		pushConstant.currentLayerNodeCount = layerNodeCount;
		vkCmdPushConstants2(cmd, &pushInfo);

		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createOctreeArray2);
		VkExtent2D groupSize = nvvk::getGroupCounts({ layerNodeCount, 1 }, VkExtent2D{ CREATEOCTREE_CS_THREADGROUP_SIZE, 1 });
		vkCmdDispatch(cmd, groupSize.width, 1, 1);
	}
}
void Octree_VRPG::getOctreeLabel(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdPushConstants2(cmd, &pushInfo);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getOctreeLabel1);
	vkCmdDispatch(cmd, 1, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	int startLayer = 2;
	for (int layerIndex = startLayer; layerIndex <= pushConstant.octreeMaxLayer; ++layerIndex) {
		pushConstant.currentLayer = layerIndex;
		vkCmdPushConstants2(cmd, &pushInfo);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getOctreeLabel2);
		vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getOctreeLabel3);
		vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
		if (layerIndex < pushConstant.octreeMaxLayer)
			nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
		else nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getOctreeLabel4);
	vkCmdDispatch(cmd, 1, 1, 1);
}
void Octree_VRPG::createLightTree(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);
	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;

	// Leaves: directly from the light VGB at lightOctreeMaxLayer.
	pushConstant.lightOctreeMaxLayer = lightOctreeMaxLayer;
	pushConstant.currentLayer = lightOctreeMaxLayer;
	vkCmdPushConstants2(cmd, &pushInfo);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initLightTreeLeaves);
	uint32_t leafCount = 6u << (3 * lightOctreeMaxLayer);
	VkExtent2D groupSize = nvvk::getGroupCounts({ leafCount, 1 }, VkExtent2D{ 1024, 1 });
	vkCmdDispatch(cmd, groupSize.width, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	// Parents: bottom-up aggregation in the independent light tree.
	for (int layer = int(lightOctreeMaxLayer) - 1; layer >= 0; --layer) {
		pushConstant.currentLayer = uint32_t(layer);
		vkCmdPushConstants2(cmd, &pushInfo);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_aggregateLightTree);
		uint32_t layerNodeCount = 6u << (3 * layer);
		groupSize = nvvk::getGroupCounts({ layerNodeCount, 1 }, VkExtent2D{ CREATEOCTREE_CS_THREADGROUP_SIZE, 1 });
		vkCmdDispatch(cmd, groupSize.width, 1, 1);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}
}

void Octree_VRPG::getOctreeNodePairData(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initWeights);
	uint32_t threadTotalCount = OUTGOING_COUNT_VRPG * IndivisibleNodeCount_G_VRPG * Candidate_Samples_Count;
	VkExtent2D groupSize = nvvk::getGroupCounts({ threadTotalCount, 1 }, VkExtent2D{ INIT_CANDIDATE_NODES_CS_THREADGROUP_SIZE, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

#if VRPG_Candidate_Samples_SampleMode == 0
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getCandidateNodes);
	vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
#else
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getCandidateNodes_init);
	vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	for (int i = 0; i < 4; ++i) {
		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getCandidateNodes_1);
		vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
	}

	for (int i = 0; i < 2; ++i) {
		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getCandidateNodes_2);
		vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getCandidateNodes_3);
		vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getCandidateNodes_4);
		vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
		
		vkCmdFillBuffer(cmd, clusterPairGlobalInfoBuffer.buffer, offsetof(shaderio::ClusterPairGlobalInfo_VRPG, clusterPairCount), sizeof(uint32_t), 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}
	
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getCandidateNodes_5);
	vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
#endif

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_dispatchGetCandidateLeafNodes);
	vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getCandidateLeafNodes);
	vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_clusterPairHitTest);
	vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getCandidateNodeWeights);
	vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	if (spatialReuse || temporalReuse) {
		if (temporalReset) {
			vkCmdFillBuffer(cmd, reservoirHistoryBuffer.buffer, 0, reservoirHistoryBuffer.bufferSize, 0);
			nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
			temporalReset = false;
		}
		uint32_t reservoirCount = OUTGOING_COUNT_VRPG * IndivisibleNodeCount_G_VRPG;
		VkExtent2D reservoirGroups = nvvk::getGroupCounts({ reservoirCount, 1 }, VkExtent2D{ 256, 1 });

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_buildSpatialReservoir);
		vkCmdDispatch(cmd, reservoirGroups.width, 1, 1);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_reuseReservoir);
		vkCmdDispatch(cmd, reservoirGroups.width, 1, 1);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}

	pushConstant.candidateSampleBufferIndex = 0;
	vkCmdPushConstants2(cmd, &pushInfo);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createCandidatePrefixWSum);
	vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

#ifdef SpatialReuse_VRPG
	if (!spatialReuse) return;
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getAdjClusterG);
	vkCmdDispatch(cmd, (IndivisibleNodeCount_G_VRPG + 1023) / 1024, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_reservoirSample);
	vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_SpatialReuse);
	vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	pushConstant.candidateSampleBufferIndex = 1;
	vkCmdPushConstants2(cmd, &pushInfo);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createCandidatePrefixWSum);
	vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
#endif
}
void Octree_VRPG::getNearbyClusterInfo(VkCommandBuffer cmd) {
	if (ditherSample == false) return;
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_dispatchGetNearbyClusterG);
	vkCmdDispatch(cmd, 1, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getNearbyClusterG1);
	vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getNearbyClusterG2);
	vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
}

void Octree_VRPG::geometryTreeDebug(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(GBuffers_Octree_VRPG::eGeometryClusterResult)), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView(uint32_t(GBuffers_Octree_VRPG::eGeometryClusterResult));
	colorAttachment.clearValue = { .color = {Application::sceneResource.sceneInfo.backgroundColor.x,
											Application::sceneResource.sceneInfo.backgroundColor.y,
											Application::sceneResource.sceneInfo.backgroundColor.z, 1.0f} };

	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };
	depthAttachment.imageView = gBuffers.getDepthImageView();

	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = { {0, 0}, gBuffers.getSize() };
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	renderingInfo.pDepthAttachment = &depthAttachment;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdBeginRendering(cmd, &renderingInfo);

	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	graphicsDynamicPipeline.rasterizationState.lineWidth = 2.0f;
	graphicsDynamicPipeline.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_FALSE;
	graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_FALSE;

	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, gBuffers.getSize());
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_TreeDebug, .fragment = fragmentShader_TreeDebug });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	bool useWireframe = true;
	uint32_t meshIndex = useWireframe ? 0 : 1;
	const shaderio::Mesh& mesh = scene.meshes[meshIndex];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	debugPushConstant.mode = 0;
	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t bufferIndex = scene.getMeshBufferIndex(meshIndex);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];

	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

	uint32_t instanceCount = debugPushConstant.curLayerNodeCount_G;
	vkCmdDrawIndexed(cmd, triMesh.indices.count, instanceCount, 0, 0, 0);

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(GBuffers_Octree_VRPG::eGeometryClusterResult)), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}
void Octree_VRPG::lightTreeDebug(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(GBuffers_Octree_VRPG::eLightClusterResult)), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView(uint32_t(GBuffers_Octree_VRPG::eLightClusterResult));
	colorAttachment.clearValue = { .color = {Application::sceneResource.sceneInfo.backgroundColor.x,
											Application::sceneResource.sceneInfo.backgroundColor.y,
											Application::sceneResource.sceneInfo.backgroundColor.z, 1.0f} };

	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };
	depthAttachment.imageView = gBuffers.getDepthImageView();

	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = { {0, 0}, gBuffers.getSize() };
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	renderingInfo.pDepthAttachment = &depthAttachment;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdBeginRendering(cmd, &renderingInfo);

	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	graphicsDynamicPipeline.rasterizationState.lineWidth = 2.0f;
	graphicsDynamicPipeline.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_FALSE;
	graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_FALSE;

	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, gBuffers.getSize());
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_TreeDebug, .fragment = fragmentShader_TreeDebug });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	bool useWireframe = true;
	uint32_t meshIndex = useWireframe ? 0 : 1;
	const shaderio::Mesh& mesh = scene.meshes[meshIndex];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	debugPushConstant.mode = 1;
	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t bufferIndex = scene.getMeshBufferIndex(meshIndex);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];

	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

	uint32_t instanceCount = debugPushConstant.curLayerNodeCount_E; //pushConstant.octreeNodeTotalCount * 6;
	vkCmdDrawIndexed(cmd, triMesh.indices.count, instanceCount, 0, 0, 0);

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(GBuffers_Octree_VRPG::eLightClusterResult)), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}
void Octree_VRPG::geometrySVODebug(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(GBuffers_Octree_VRPG::eGeometrySVO)), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView(uint32_t(GBuffers_Octree_VRPG::eGeometrySVO));
	colorAttachment.clearValue = { .color = {Application::sceneResource.sceneInfo.backgroundColor.x,
											Application::sceneResource.sceneInfo.backgroundColor.y,
											Application::sceneResource.sceneInfo.backgroundColor.z, 1.0f} };

	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };
	depthAttachment.imageView = gBuffers.getDepthImageView();

	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = { {0, 0}, gBuffers.getSize() };
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	renderingInfo.pDepthAttachment = &depthAttachment;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdBeginRendering(cmd, &renderingInfo);

	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	graphicsDynamicPipeline.rasterizationState.lineWidth = 2.0f;
	graphicsDynamicPipeline.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_FALSE;
	graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_FALSE;

	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, gBuffers.getSize());
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_TreeDebug, .fragment = fragmentShader_TreeDebug });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	bool useWireframe = true;
	uint32_t meshIndex = useWireframe ? 0 : 1;
	const shaderio::Mesh& mesh = scene.meshes[meshIndex];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	uint32_t bufferIndex = scene.getMeshBufferIndex(meshIndex);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];

	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

	const int selectedNormal = debugPushConstant.normalIndex_G_2;
	const int normalPartitionCount = 6;
	const int firstNormal = selectedNormal == normalPartitionCount ? 0 : selectedNormal;
	const int lastNormal = selectedNormal == normalPartitionCount
		? normalPartitionCount : selectedNormal + 1;

	const int selectedLayer = debugPushConstant.curLayer_G_2;
	const int selectedLayerNodeCount = debugPushConstant.curLayerNodeCount_G_2;
	const int maxLayer = int(pushConstant.octreeMaxLayer);
	const int firstLayer = selectedLayer > maxLayer
		? OCTREE_CLUSTER_LAYER_VRPG : selectedLayer;
	const int lastLayer = selectedLayer > maxLayer
		? maxLayer + 1 : selectedLayer + 1;

	for (int layer = firstLayer; layer < lastLayer; ++layer) {
		debugPushConstant.curLayer_G_2 = layer;
		debugPushConstant.curLayerNodeCount_G_2 = (1 << (3 * layer));
		const uint32_t instanceCount = debugPushConstant.curLayerNodeCount_G_2;
		for (int normalIndex = firstNormal; normalIndex < lastNormal; ++normalIndex) {
			debugPushConstant.normalIndex_G_2 = normalIndex;
			debugPushConstant.mode = 2;
			vkCmdPushConstants2(cmd, &pushInfo);
			vkCmdDrawIndexed(cmd, triMesh.indices.count, instanceCount, 0, 0, 0);
		}
	}
	debugPushConstant.normalIndex_G_2 = selectedNormal;
	debugPushConstant.curLayer_G_2 = selectedLayer;
	debugPushConstant.curLayerNodeCount_G_2 = selectedLayerNodeCount;

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(GBuffers_Octree_VRPG::eGeometrySVO)), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}

void Octree_VRPG::clusterPairDebug(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	{
		debugPushConstant2.frameIndex = Application::frameIndex;
		debugPushConstant2.VGBVoxelTotalCount = pushConstant.VGBVoxelTotalCount;
		debugPushConstant2.VGBStartPos_Size = pushConstant.VGBStartPos_Size;
		debugPushConstant2.VGBVoxelSize = pushConstant.VGBVoxelSize;
		debugPushConstant2.octreeMaxLayer = pushConstant.octreeMaxLayer;
		debugPushConstant2.screenSize = { gBuffers.getSize().width, gBuffers.getSize().height };
		debugPushConstant2.sceneInfoAddress = pushConstant.sceneInfoAddress;
		debugPushConstant2.globalInfoAddress = (shaderio::ClusterPairDebugGlobalInfo_VRPG*)clusterPairDebugGlobalInfoBuffer.address;
		pushInfo.pValues = &debugPushConstant2;
	}

	if (debugClusterIndex >= 0) {
		uint32_t clusterIndex = uint32_t(debugClusterIndex);
		vkCmdUpdateBuffer(cmd, clusterPairDebugGlobalInfoBuffer.buffer, 0, sizeof(clusterIndex), &clusterIndex);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
			VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);
	}

	//get clusterIndex_G by mouse
	if(changeClusterG){
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

		nvvk::WriteSetContainer write{};
		write.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), setting.asManager->asBuilder.tlas);
		vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 1, write.size(), write.data());

		vkCmdPushConstants2(cmd, &pushInfo);
		VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_GetDebugClusterG);
		vkCmdDispatch(cmd,1, 1, 1);
	}
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);

	shaderio::TriangleMesh triMesh;
	{
		nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(GBuffers_Octree_VRPG::eClusterPairResult)), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
		nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });

		VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.imageView = gBuffers.getColorImageView(uint32_t(GBuffers_Octree_VRPG::eClusterPairResult));
		colorAttachment.clearValue = { .color = {Application::sceneResource.sceneInfo.backgroundColor.x,
												Application::sceneResource.sceneInfo.backgroundColor.y,
												Application::sceneResource.sceneInfo.backgroundColor.z, 1.0f} };

		VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };
		depthAttachment.imageView = gBuffers.getDepthImageView();

		VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
		renderingInfo.renderArea = { {0, 0}, gBuffers.getSize() };
		renderingInfo.colorAttachmentCount = 1;
		renderingInfo.pColorAttachments = &colorAttachment;
		renderingInfo.pDepthAttachment = &depthAttachment;

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

		vkCmdBeginRendering(cmd, &renderingInfo);

		graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
		graphicsDynamicPipeline.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
		graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
		graphicsDynamicPipeline.rasterizationState.lineWidth = 2.0f;
		graphicsDynamicPipeline.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
		graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_FALSE;
		graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_FALSE;

		graphicsDynamicPipeline.cmdApplyAllStates(cmd);
		graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, gBuffers.getSize());
		graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_ClusterPairDebug, .fragment = fragmentShader_ClusterPairDebug });

		VkVertexInputBindingDescription2EXT bindingDescription{};
		VkVertexInputAttributeDescription2EXT attributeDescription = {};
		vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

		bool useWireframe = true;
		uint32_t meshIndex = useWireframe ? 0 : 1;
		const shaderio::Mesh& mesh = scene.meshes[meshIndex];
		triMesh = mesh.triMesh;

		uint32_t bufferIndex = scene.getMeshBufferIndex(meshIndex);
		const nvvk::Buffer& v = scene.bDatas[bufferIndex];

		vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));
	}

	{
		debugPushConstant2.mode = 0;
		vkCmdPushConstants2(cmd, &pushInfo);
		vkCmdDrawIndexed(cmd, triMesh.indices.count, 1, 0, 0, 0);
	}

	if (showNearbyClusters) {
		debugPushConstant2.mode = 3;
		vkCmdPushConstants2(cmd, &pushInfo);
		vkCmdDrawIndexed(cmd, triMesh.indices.count, NEARBY_NODE_COUNT_VRPG + 1, 0, 0, 0);
	}

	{
		debugPushConstant2.mode = 1;
		vkCmdPushConstants2(cmd, &pushInfo);
		vkCmdDraw(cmd, 2, 1, 0, 0);
	}

	{
		debugPushConstant2.mode = 2;
		vkCmdPushConstants2(cmd, &pushInfo);

		vkCmdDrawIndexed(cmd, triMesh.indices.count, Candidate_Samples_Count, 0, 0, 0);
	}

	{
		vkCmdEndRendering(cmd);

		nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(GBuffers_Octree_VRPG::eClusterPairResult)), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
		nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
	}
}
