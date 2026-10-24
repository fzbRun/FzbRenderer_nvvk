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
		UIModified |= ImGui::SliderFloat("Normal Threshold", &clusterNormalThreshold, 0.0f, 1.0f, "%.3f");
		UIModified |= ImGui::SliderFloat("Fill Rate", &clusterFillRateThreshold, 0.0f, 1.0f, "%.3f");
		UIModified |= ImGui::SliderFloat("Volume Tolerance", &clusterVolumeTolerance, 0.0f, 5.0f, "%.3f");
		UIModified |= ImGui::Checkbox("Light Backface Cull", &lightBackfaceCull);
		UIModified |= ImGui::Checkbox("Stable Candidate Cut", &stableCandidateCut);

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

			ImGui::Combo("normal-G2", &debugPushConstant.normalIndex_G_2, normalIndexNames_pointers.data(), static_cast<int>(normalIndexNames_pointers.size()));

			static bool showGeometrySVODebugMap = false;
			static int debugLayer2 = 0;
			ImGui::Combo("Geometry SVO Layer", &debugLayer2, layerMapNames_pointers.data(), static_cast<int>(layerMapNames_pointers.size()));
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

			debugPushConstant.curLayer_G_2 = debugLayer2 + OCTREE_CLUSTER_LAYER_VRPG;
			if (showGeometrySVODebugMap) Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)GBuffers_Octree_VRPG::eGeometrySVO);

			ImGui::Unindent(20.0f);
		}

		if (ImGui::CollapsingHeader("Light Tree Debug", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Indent(20.0f);

			uint32_t lightShowLayerCount = pushConstant.octreeMaxLayer + 1;
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
			// 1) 闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚敐澶婄闁挎繂鎲涢幘缁樼厱闁靛牆鎳庨顓㈡煛鐏炲墽娲存い銏℃礋閺佹劙宕卞▎妯恍氱紓鍌欒兌閸嬫捇宕曢崘宸劷闁跨喓濮撮拑鐔兼煥濠靛棭妲告い顐㈡嚇閺屾洝绠涙繝鍐╃彇闂佸憡姊归幃鍌氼潖閸濆嫅褔宕惰閸嬫挸螖閸涱厾锛涢梺瑙勫礃閸╂牕鐣烽崣澶岀瘈闂傚牊渚楅崕鎰亜閵夈儳澧涚紒缁樼洴楠炲鈹戦崱姘厴闂備礁鎲￠幖鈺呭礈閻斿娼栨繛宸簼閻掑鏌ｉ幇顖氳敿閻庢碍婢橀…鑳槻濠⒀呮櫕濡叉劙骞掑Δ浣镐汗闂佹儳娴氶崑鍕枈瀹ュ鈷戝ù鍏肩懅缁夘噣鏌￠埀顒佹綇閳哄啰澶勯梺浼欑到閻ジ鎯屽▎鎾村仯闁搞儻绠戠€氬酣鏌涢弬璺ㄐу┑锛勬暬瀹曠喖顢涘杈╂綁闂備焦鎮堕崕娲倶濠靛绠婚悗闈涙憸椤旀洟姊洪崨濠佺繁闁哥姵顨婇崺娑㈠箛椤戙垹閰ｅ畷鎯邦檪闂婎剦鍓熼弻鐔碱敊閻ｅ本鍣板銈冨灪濡啫鐣烽悢鐓幬╅柕澶堝€曢ˉ姘舵⒒娴ｅ憡鎯堥悹鍥╁仧娴狅箓宕滆閳诲绻濋悽闈浶ラ柡浣告啞閹便劑鎮介崨濠傛疄闂佸憡鎸嗛崘鐐棥濠电姷鏁告慨鐢靛枈瀹ュ鍋傛繛鍡樻尰閻撴瑩鎮峰▎蹇擃仼濠殿噯绠撻弻娑㈠Χ閸℃瑦鍣紓浣介哺鐢繝鐛崶顒夋晣鐟滃酣鎮挎笟鈧娲传閸曨剚鎷遍梺鐑╂櫓閸ㄨ泛鐣峰ú顏勭妞ゆ牗姘ㄩˇ銊ヮ渻閵堝懐绠ｇ紒鈧笟鈧敐鐐茬暆閸曨兘鎷绘繛杈剧秬濞咃絿鏁☉銏＄厵婵繂鐭堥崵娆撴偂閵堝鐓熼柡鍐ㄥ€哥敮鍫曟煢閸愵亜鏋旈柍褜鍓欑粻宥夊磿闁单鍥敍閻愭潙浜楀┑鐐村灦閳笺倛銇愰幒鎾存珳闂佸憡渚楅崰鏍汲閸儲鈷戦悹鍥皺缁犳煡鏌熼崨濠冨€愰柛鈺冨仱楠炲鏁傞挊澶嗗亾閻戣姤鐓犵痪鏉垮船婢ь喗銇勯妶鍌氫壕婵犵數濮甸鏍窗閺嶎厽鏅濋柕濞垮劤閺嗭箓姊婚崼鐔峰幏闁挎繂顦粻娑㈡煛婢跺孩纭舵い鏃€甯″楦裤亹閹烘繃娈ョ紓浣插亾濞撴埃鍋撴鐐插暣楠炲鏁傜憴锝嗗缂傚倷绶￠崹鍗灻洪弽銊︽珷闁哄被鍎查悡銉︽叏濡灝鐓愰柣鎾寸洴閺屾盯鍩勯崘澶哥返濠电偛鐗婄划宥夊箞閵婏妇绡€闁告劏鏂傛禒銏ゆ⒑閸濆嫯瀚伴柟铏耿楠炲啫鈻庨幘鏉戞闁荤姾妗ㄧ拃锔藉閸愩劉鏀介柣姗嗗亝婵即鎮橀悙璺盒撻柣蹇旀尵缁辨挻鎷呮禒瀣懙闂佺懓鎲￠幃鍌炪€佸Ο鑽ら檮缂佸瀵ч妵婵嬫⒑閸涘﹤濮傞柛鏂款儑濡叉劙鏁愰崱鏇犵畾闂佺粯鍔︽禍婊堝焵椤戞儳鈧繂鐣烽姀銈嗗亗妤犵偟鍠愬浠嬨€佸鈧慨鈧柣妯兼暩閺嬪啴姊绘担鑺ョ《闁哥姵鎸婚幈銊╁级閹寸偟鍑介梻鍌氬€烽悞锕傚箖閸洖纾挎い鏇楀亾鐎殿喗鐓￠、鏃堝幢濮楀棙缍楅梻浣瑰劤濞存岸宕戦崱娆忓К闁逞屽墮閳规垿鎮欓懠顒€顣洪梺璇茬箲缁诲牆顕ｉ幖浣瑰亜闁硅偐鍋樼花璇差渻閵堝棙灏甸柛瀣枑閺呰泛鈽夊杈╋紲闂佺鏈銊︽櫏婵＄偑鍊戦崹娲€冩繝鍌滄殾闁圭儤鍩堥悡銉╂煃瑜滈崜娑氬垝椤撶喎绶為柟閭﹀幐閹锋椽姊洪崨濠勭畵閻庢凹鍓熷鍐测枎閹惧鍘藉┑掳鍊愰崑鎾绘煥閺囥劋閭鐐插暣閺佹捇鎮╅幓鎺戠ギ闂備線娼х换鍡涘焵椤掍焦鐏遍柛瀣崌瀹曞ジ寮撮悢鍝勫箺闂備礁缍婇崑濠囧储閼测晛绶ゅ┑鐘崇閻撳啴姊洪崹顕呭剱妞ゃ儱鐗撻弻鈥崇暆鐎ｎ剛袦濡炪們鍨洪敃銏ゅ箖濠婂牆骞㈡俊銈傚亾濠㈢懓閰ｅ缁樻媴閸涘﹥鍎撻梺鍝勭墱閸撶喖骞嗛崟顖ｆ晬闁绘劘灏欓悾娲⒑閸愬弶鎯堥柟鍐查叄閹苯螖閳ь剟鈥﹂崸妤佸殝闁活剦浜濋崹鍧楀极瀹ュ骞㈡繛鎴炵憿閹风粯绻涙潏鍓у埌闁硅绻濋獮鍡涘醇閵夛妇鍘藉銈嗗坊閸嬫捇鏌曢崼銏╃劸闁伙綁鏀辩€靛ジ寮堕幋婵嗘暏婵＄偑鍊栭幐楣冨磻閻旂⒈鏁婇柟鎵閸嬧剝绻濇繝鍌涘櫣妞わ絽銈搁幃浠嬵敍濞戣鲸鐤侀悗瑙勬礃閸ㄥ潡宕归幆褜鐔嗛柛鏇ㄥ墰閳洟鏌ㄥ☉娆忓闁瑰嘲鎳樺畷顐﹀礋椤愵偆鈧挳姊婚崒娆掑厡缁绢厼鐖煎鎻掝煥閸涱垳鐓撻梺瑙勫礃閸╂牠锝為弴銏＄厵闁硅鍔﹂崵娆戠棯閹冩倯缂佺粯鐩畷鍗炍熸潪鏉垮灁闂備礁鎽滈崑娑氱礊婵犲偆娼栭柧蹇撴贡閻瑦绻涢崱妯哄姢闁告捇浜跺娲捶椤撶喎娈屽┑鐐叉▕閸樺ジ顢氶敐鍥ㄥ珰婵炴潙顑嗛～宥夋⒑闂堟稓绠冲┑顔芥綑閻ｇ敻宕卞☉娆屾嫼闂傚倸鐗婃笟妤呮倿妤ｅ啯鐓曢幖娣€栭悵顏嗙磼閸屾稑娴い銏★耿婵偓闁抽敮鍋撻柟鐤缁辨捇宕掑▎鎴濆闁藉啴浜堕幃妯跨疀閺囩儐鏆紓浣虹帛閻╊垰鐣烽敐鍡欑彾闁挎繂妫楅悘鈺冪磼椤旂晫顣叉繛鐓庣箻婵℃瓕顦撮柣锕€鐗撳娲箹閻愭彃濮堕梺璇茬箲缁嬫牠藝瑜版帗鐓涢悘鐐插⒔濞插鈧鍠楅幐铏繆閹间礁唯闁靛／灞芥櫍濠电姷鏁告慨鐑姐€傞鐐潟闁哄洢鍨归悿鐐亜閹扳晛鈧牠锝為弴銏＄厵闁绘垶锕╁▓鏇㈡煕婵犲洦鏁遍柕鍥у楠炴帡宕卞鎯ь棜缂傚倸鍊风拋鏌ュ磻閹剧粯鐓曢柟鏉垮悁缁ㄦ挳鏌涚€ｎ偅灏甸柟鍙夋尦瀹曠喖顢楅崒銈喰ら梻鍌欐祰濡椼劑鎳楅崼鏇椻偓锕傚醇濠靛棛銈?= 闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鎯у⒔閹虫捇鈥旈崘顏佸亾閿濆簼绨奸柟鐧哥秮閺岋綁顢橀悙鎼闂侀潧妫欑敮鎺楋綖濠靛鏅查柛娑卞墮椤ユ艾鈹戞幊閸婃鎱ㄩ悜钘夌；闁绘劗鍎ら崑瀣煟濡崵婀介柍褜鍏涚欢姘嚕閹绢喖顫呴柣妯荤垹閸ャ劎鍘遍柣蹇曞仜婢т粙鎮￠姘肩唵閻熸瑥瀚粈鈧梺瀹狀潐閸ㄥ潡銆佸▎鎴犻┏閻庯綆鍓欐慨娲⒒娴ｅ憡鍟炴い顓炴瀹曠喖顢楁担鍝勫▏濠碉紕鍋戦崐鏍偋濠婂牆纾婚柣鎰劋閸嬪倹绻涢幋娆忕仾闁绘挻娲樼换娑㈠箣閻愬灚鍣紓浣藉煐閻擄繝寮婚敐澶婄闁哄鍨甸～搴ｇ磽娴ｄ粙鍝洪悽顖椻偓宕囨殾闁圭儤鍨熷Σ鍫熶繆椤栨壕鎷￠柛搴＄焸濮婄粯绗熼埀顒勫焵椤掑倸浠滈柤娲诲灡閺呭爼宕滆绾剧晫鈧箍鍎遍幊蹇浰夐姀銈嗙厱闁宠鍎虫禍鐐繆閻愵亜鈧牜鏁幒妞濆洭顢涘鐓庢闂佺鍕垫畷闁抽攱甯￠弻娑㈠即閵娿儰绨诲銈呮禋閸樹粙銆冮妷鈺傚€烽柤纰卞墮閳灚绻濈喊妯哄⒉闁诡喖鍊规穱濠囨煥鐎ｎ偄鍔呴梺鎸庣箓閹冲繘鎮橀崼婢棃鎮╅棃娑楁勃闂佸憡姊归悧鐘荤嵁閸℃稑绫嶉柛顐ゅ枑濞呮粓姊洪崨濠傚Ё缂佽尪濮ょ粋宥嗐偅閸愨晝鍘甸梻渚囧弿缁犳垿寮搁妶鍥╂／闁诡垎浣镐划闂佸搫鐬奸崰鏍€佸☉銏犲耿婵°倐鍋撻柡鍡欏枔缁辨帞鈧綆浜炵粻宕囩磼鏉堛劍灏扮紒妤冨枛瀹曟儼顦抽柣蹇撶Ч濮婃椽鎳￠妶鍛瘣闂佸搫鎳愭繛鈧柣娑卞櫍瀹曟﹢顢欓懞銉︻仧闂備胶绮悷銏ゅ磻閹剧粯鐓涘ù锝囶焾閺嗭綁鏌＄仦璇插鐎殿喗娼欒灃闁逞屽墯缁傚秵銈ｉ崘鈹炬嫼缂傚倷鐒﹂敋濠殿喖娲弻銊╁即閵娿倝鍋楅悗娈垮枛椤嘲顕ｉ幘顔藉亜闁惧繐婀卞Σ鍥⒒婵犲骸浜滄繛璇х畵楠炴牠鍩℃导杈ㄦそ椤㈡﹢鎮滈崱妯虹槣闂備線娼ч悧鍡欐崲閹扮増鍎婃繝濠傜墛閻撴洘淇婇娑楄埅闁稿鍨介弻宥夊Ψ閿旂瓔妫冮悗瑙勬礃閿曘垽銆佸▎鎾冲簥濠㈣鍨板ú銊╁矗閹剧粯鐓曢柕澶涚到婵＄晫绱掗埀顒勫醇閵夛妇鍘搁梺绯曞墲閸旀洟鍩€椤掍胶澧遍柟骞垮灩閳藉顫濋敐鍛濠电偞鍨堕悷顖炴倿娴犲鐓熸い鎾跺剳闊剟鏌″畝瀣М鐎殿喕绮欓、鏇㈠閻樿京甯涘┑?
			const float localX = mousePos.x - vpOrigin.x;
			const float localY = mousePos.y - vpOrigin.y;

			// 2) 闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚敐澶婄闁挎繂鎲涢幘缁樼厱闁靛牆鎳庨顓㈡煛鐏炲墽娲存い銏℃礋閺佹劙宕卞▎妯恍氱紓鍌氬€烽懗鑸垫叏闁垮绠鹃柍褜鍓熼弻鈥崇暆閳ь剟宕伴弽顓犲祦闁糕剝鍑瑰Σ璇差渻閵堝繒鍒伴柛鐔锋健閸┿儲寰勯幇顒傤啋閻庤娲栧▔锕傚Χ婢跺鍘撻悷婊勭矒瀹曟粌鈻庨幋鐘辩瑝閻庡箍鍎遍ˇ顖滃閸ф鈷戞い鎺嗗亾闁告娲熷畷鍫曨敆婢跺娅栭梻浣虹帛閸旀牞銇愰崘顔藉€垫い鎾卞灪閳锋垹绱掔€ｎ偄顕滄繝鈧导瀛樼厱闁宠桨绀侀弳锝団偓娈垮枔閸斿秶绮嬮幒鏂哄亾閿濆骸浜為柛妯挎閳规垿鍩ラ崱妤冧淮濠电偛鎷戠紞渚€寮鍜佸悑闁糕剝鍔掔花濠氭⒑閹稿海绠撴俊顐ｇ洴瀹曪綁鍩€椤掑嫭鈷戦柛婵嗗濠€鎵磼鐎ｎ偅宕岄柛鈹惧亾濡炪倖甯掗敃銉╁焵椤戞儳鍔︾€规洜鍠栭、妤呭磼濮樺崬鍩岄梻鍌氬€搁崐椋庣矆娓氣偓閹潡宕堕濠勭◤婵犮垼鍩栭崝鏇㈠垂閸岀偞鐓曠憸搴ㄣ€冮崨顖滀笉闁哄稁鍘介悡鍐煕濠靛棗顏╅柡鍡楋躬閺屾稓鈧綆鍋呭畷宀€鈧娲滈崗姗€銆佸鈧幃娆撳箛閳轰胶浠鹃梺闈涙搐鐎氱増淇婇悜鑺ユ櫆闁告挆鍛瘔缂傚倸鍊峰ù鍥ㄣ仈閹间礁绠板┑鐘宠壘缁狀垶鏌涘☉娆愮稇缂佺姷鏁婚弻鐔煎传閸曨厜銏°亜閵夛箑鍝烘慨濠傤煼瀹曟帒鈻庨幒鎴濆腐缂備胶铏庨崢濂告偉閻撳海鏆︽繝濠傚暊閺€浠嬫煕椤愮姴鐏柣鎾存崌濮婃椽宕烽鐐插濡炪們鍔屽Λ婵嗙暦閹达箑纾奸柣鎰嚟閸樹粙姊洪崫鍕殭闁绘锕ら埢宥咁煥閸涱垳锛滈柣鐘叉搐瀵爼藟閵忋倖鐓涢悘鐐靛亾缁€瀣偓瑙勬礈閸樠囧煘閹达箑骞㈤煫鍥ㄦ尫婢规洟姊洪崨濠勨槈闁挎洩绠撻妴鍛村箵閹广劍妫冮弫鎰板川椤撶喐顔夐梻浣瑰▕閺€閬嶅垂閸︻厽顫曢柟鎯х摠婵挳鏌ｈ閺呰尙娆㈤棃娑掓斀闁绘劘灏欐晶娑欑箾閼碱剙鏋涚€殿喛顕ч埥澶愬閻樼數娼夐梻渚€鈧偛鑻晶鎾煙椤旀儳鍘撮柛鈺嬬節瀹曟帒鈻庨幋鐘垫毎闂傚倷绀佸﹢閬嶆惞鎼淬劌绐楅柡宓苯鏅犲┑鐘绘涧椤戝棝鎮″▎鎴犳／闁哄鐏濋懜瑙勵殽閻愭惌娈滈柡宀嬬節瀹曘劑顢欓崜褏鏉芥繝娈垮枛閿曘儱顪冩禒瀣祦闁归偊鍘介崕鐔兼煃閸ㄦ稒娅嗛悗姘矙濮婄粯鎷呴崨濠冨創闂佺懓鍢查澶婄暦閹寸偟绡€闁稿本绮嶅▓楣冩⒑闂堟稓绠為柛濠冩礈缁鎮烽幊濠傜秺閺佹劙宕ㄩ璺攨闂備礁鎲＄湁缂侇喗鐟ヨ灋闁告稒鎯岄弫鍐煏韫囧ň鍋撻幇浣剐熷┑鐘垫暩閸嬫稑顕ｉ崼鏇熷亱婵°倕鍟伴惌澶愭煙閻戞ɑ澶勬俊顖氬濮婃椽宕崟顒佹嫳缂備礁顑嗛幑鍥春閳ь剚銇勯幒鎴姛缂佸鏁婚弻娑氣偓锝庝簼閸ｈ棄霉濠婂嫭鍊愭俊顐㈠暙闇夐柕蹇嬪灪閸も偓濡炪値鍘归崝鎴濈暦閻旂⒈鏁嗛柍褜鍓熼崺鈧い鎺嶇婵牓鏌曢崶褍顏┑顔瑰亾闂佸疇妫勯幊鎾诲焵椤掆偓閻忔繈鍩為幋锔绘晩闁荤喖顣︽竟鏇㈡⒒閸屾瑧鍔嶉悗绗涘懐鐭欓柟瀵稿Л閸嬫挸顫濋悡搴♀拫闂佺粯渚楅崳锝呯暦婵傜唯闁挎棁顫夌€氳棄鈹戦悙瀛樺鞍闁糕晛鍟村畷鎴﹀箻鐎靛摜顔曟繛杈剧到閸熷灝煤閻樿鐒垫い鎺戯功閻ｇ數鈧娲栭妶鍛婁繆閻戣棄唯鐟滄繄妲愰悽鍛娾拻闁稿本鐟х拹浼存煕閹惧鎳呯紒顔界懄缁楃喖鍩€椤掆偓椤曪綁顢曢敃鈧獮銏＄箾閹寸偟鎳呴柛姗嗕邯濡懘顢曢姀鈥冲亶闂佺楠哥壕顓熺珶閺囩姵宕夐柕濞у拑绱查梻浣虹帛椤牆鐣烽悽鍛婂仼婵炲樊浜濋悡娑氣偓鍏夊亾閻庯綆鍓涜ⅵ婵°倗濮烽崑鐐烘晝閵忋倕绠圭憸鐗堝俯閺佸啴鏌ㄥ┑鍡樺殗婵☆偊顥撶槐鎾诲磼濮橆兘鍋撻悜鑺ュ殑闁肩鐏氶崣蹇撐旈敐鍛灓闁轰礁顑囬幉鎼佸籍閸繃妲梺缁樺姇閹碱偆绮婚敐澶嬬叆闁哄洦顨呮禍楣冩⒑缂佹ê绗氭俊鐐扮矙瀵鈽夐姀鐘电杸闂佺绻愰幗婊堝礄瑜版帗鈷戦柣鐔告緲缁狙囨煟閻旀潙鍔﹂柍銉︽瀹曟﹢顢欓崲澹洨鍙撻柛銉ｅ妿閳洘銇勬惔锛勑ф慨濠呮缁辨帒螣缂佹ê鍓电紓鍌欒兌缁垳鎹㈤崒鐑嗘晪闁挎繂妫涚弧鈧┑顔斤供閸庝絻顦叉い顓℃硶閹瑰嫭绗熼姘闂備線鈧偛鑻晶鎵磼椤曞懎鐏＄紒宀冮哺缁绘繈宕堕懜鍨珫婵犵數濞€濞佳呪偓姘€鍥х；闁规崘顕ч悞鍨亜閹烘垵顏柣鎾寸洴閹鏁愭惔鈥茬敖闂佺懓鍟垮ú锕傚箞閵婏妇绡€闁告劏鏂傛禒銏ゆ倵鐟欏嫭澶勯柛瀣工閻ｇ兘鎮㈢喊杈ㄦ櫌婵炶揪绲块幊鎾活敂閹殿喚纾介柛灞捐壘閳ь剟顥撶划鍫熺瑹閳ь剟銆侀弽顓炲耿婵炴垶锚閻庮參姊虹粔鍡楀娴犵娀鏌嶈閸忔稓绮堟担鐣屼航闂備胶鍘ч～鏇㈠磹濠靛棭鐎舵い鏂垮⒔绾捐棄霉閿濆懎顥忛柛搴＄箻閺岋綀绠涢弮鍌滅暭缂備緡鍣徊浠嬪煡婢跺娼ㄩ柛鈩兠惁婊堟⒒娓氣偓濞佳囨偋閸℃あ娑樷槈濞嗘劖鐝峰銈嗗笒閸婅崵澹曢崗绗轰簻闁哄啠鍋撻柛搴㈠▕閸┿垽宕奸埗鈺佷壕閻熸瑥瀚径鍕煕閹捐泛鏋涙鐐寸墳閵囨劙骞掗幋鐐茬ザ婵＄偑鍊栭幐鐐垔鐎靛憡顫曢柟鎯板Г閳锋帒霉閿濆嫯顒熼柣鎺楃畺閺岋繝宕奸銏犫拫闂佽鍠氶崗妯讳繆閹间礁鐓涘┑鐘插暞濞呮挸鈹戦悩顔肩伇闁糕晜鐗犲畷婵嬪冀椤撶喎浜?
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
	if (const char* stable = std::getenv("VRPG_STABLE_CUT")) stableCandidateCut = std::atoi(stable) != 0;
	if (const char* cull = std::getenv("VRPG_BACKFACE_CULL")) lightBackfaceCull = std::atoi(cull) != 0;
#endif
}
void Octree_VRPG::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
#ifndef NDEBUG
	NVVK_CHECK(gBuffers.update(cmd, size));
	//婵犵數濮烽弫鍛婃叏閻戣棄鏋侀柛娑橈攻閸欏繘鏌ｉ幋锝嗩棄闁哄绶氶弻娑樷槈濮楀牊鏁鹃梺鍛婄懃缁绘﹢寮婚敐澶婄闁挎繂妫Λ鍕⒑閸濆嫷鍎庣紒鑸靛哺瀵鎮㈤崗灏栨嫽闁诲酣娼ф竟濠偽ｉ鍓х＜闁绘劦鍓欓崝銈嗐亜椤撶姴鍘寸€殿喖顭烽弫鎰緞婵犲嫮鏉告俊鐐€栫敮濠囨倿閿曞倸纾块柟鎯у绾捐棄霉閿濆懏鎯堢亸蹇涙⒑閸涘娈曞┑鐐诧工椤曪絾绻濆顓炰簻闂佺绻愰惃鐑藉箯婵犳碍鐓熼幖娣妽濞懷冾熆閻熷府宸ラ崡杈ㄣ亜閺傚灝鈷旂痪鎹愭闇夐柨婵嗘噺閹牓宕崨濠勭瘈闁汇垽娼ф禍褰掓煕鐎ｎ偅宕屾慨濠冩そ瀹曨偊宕熼鐘辩礃闂備礁鎽滄慨鐢告偋閻樿崵宓侀柛鎰靛弾濡嫰鎮楃憴鍕闁靛牊鎮傞獮鍐閵堝懍绱堕梺鍛婃处閸嬪懘顢橀懖鈺冪＝闁稿本鑹鹃埀顒佹倐瀹曟劙鎮滈懞銉ユ畱闂佽偐顭堥悘姘跺矗韫囨稒鐓熼柕蹇曞У閸熺偤鏌涚€ｎ亶鍎旈柡宀嬬節瀹曞爼濡烽妷褌鎮ｉ梻浣虹帛閹稿鏁冮姀銈呰摕闁绘柨鎲＄紞鍥旈敐鍛殲妞ゅ骸绉瑰铏圭矙濞嗘儳鍓遍梺鍛婃⒐閻熴儴鐏嬮梺鍛婄⊕閹矂寮崘鈹夸簻闁规壋鏅涢悘鈺冪磼閻樺搫鍚圭紒杈ㄦ尰閹峰懘骞撻幒宥咁棜闂傚倷绀佹竟濠囧磻娓氣偓瀹曟洟寮婚妷锕€浜楀┑鐐叉閹稿鎮￠弴銏㈠彄闁搞儯鍔忔竟妯好瑰鍐Ш闁哄矉绻濆畷鍗烆潨閸絺鍋撻崸妤佺厸閻忕偛澧藉ú瀵糕偓娈垮櫘閸撴盯骞戦崟顖毼╃憸宀€妲愬Ο鑽ょ瘈闁汇垽娼ф禒锕傛煕閵娿儳绉虹€殿噮鍋呯换婵嗩潩椤掍焦袣闂備線鈧偛鑻晶顖炴煏閸パ冾伃妤犵偞甯￠獮瀣攽閹邦亞妫梻鍌欑劍婵炲﹪寮ㄩ柆宥呭瀭闁革富鍘介～鏇㈡煙閹呮憼濠殿垱娼欓埞鎴︻敊閽樺鍔告繛瀛樼矋缁诲牓鐛崼銉ノ╅柍杞拌兌閸婄偤姊洪棃娑辩劸闁稿﹥顨堢划璇差潩椤掑瀵岄梺闈涚墕濡稒鏅堕鍕厽闁哄啯鍨垫晶鎾煟閹垮啫澧存い銏☆殜瀹曟帒螖閳ь剚绂嶆ィ鍐╁仭婵炲棗绻愰顏勵熆鐠哄搫顏柡灞剧洴楠炴鎹勯悜妯间邯婵°倗濮烽崑鐐烘偋閻愬搫桅闁圭増婢樼粻鎶芥煙鐎电浠滅紒浣藉煐娣囧﹪鎮欓鍕ㄥ亾閺嶎厼绠伴柣鎰靛墮閸ㄦ棃鎮楅棃娑欏暈妞ゎ偅娲熼弻娑㈠箛椤撶姰鍋為梺琛″亾濞寸姴顑嗛悡鐔兼煙闁箑鐏犻柣鎾村姉閹叉悂骞庢繝鍌涘櫧缁炬儳銈搁弻鐔虹磼濡櫣鐟查梺绋款儍閸庢煡銆冮妷鈺傚€烽悹鍥囧棗鎯堟俊銈囧Х閸嬬偤鈥﹂崶顒€绠柣妯肩帛閸ゆ垶銇勯幒鎴Ч閸熸悂姊婚崒姘偓椋庢濮橆剦鐒界憸鏃堝箖瑜斿畷鍗烆渻閵忥紕鈽夐柍钘夘槸椤繈顢楅埀顒冣叢闂傚倷绶氶埀顒傚仜閼活垱鏅剁€涙ɑ鍙忓┑鐘插暞閵囨繈鏌＄仦鑺ュ殗闁诡喗鐟╅幊鐘垫崉娓氼垯绱旀繝鐢靛Л閹峰啴宕熼鈧崬澶愭⒑閸濆嫮鐒跨紒杈ㄦ礃缁傛帡鏁冮崒娑樻疅闂侀潧顦介崰妤吽夐敐鍡曠箚闁绘劦浜滈埀顒佺墵楠炴劙骞栨担鑲澭呯磼鐎ｎ厽纭堕柡鍡畵閺屾洘绻涢悙顒佺彟闂佽桨绀佺粔鐢垫崲濠靛洨绡€闁稿本鍑规禒楣冩⒑?
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

	pushConstant.VGBVoxelTotalCount = setting.VGBSize * setting.VGBSize * setting.VGBSize;
	pushConstant.voxelVolume = setting.VGBVoxelSize.x * setting.VGBVoxelSize.y * setting.VGBVoxelSize.z;
	pushConstant.VGBStartPos_Size = glm::vec4(setting.VGBStartPos, setting.VGBSize);
	pushConstant.VGBVoxelSize = glm::vec4(setting.VGBVoxelSize, 1.0f);

	pushConstant.lightVGBStartPos_Size = glm::vec4(setting.lightVGBStartPos, setting.lightVGBSize);
	pushConstant.lightVGBVoxelSize = glm::vec4(setting.lightVGBVoxelSize, 1.0f);
	pushConstant.temporalReuse = temporalReuse ? 1 : 0;
	pushConstant.temporalBlend = temporalBlend;
	pushConstant.clusterNormalThreshold = clusterNormalThreshold;
	pushConstant.clusterFillRateThreshold = clusterFillRateThreshold;
	pushConstant.clusterVolumeTolerance = clusterVolumeTolerance;
	pushConstant.lightBackfaceCull = lightBackfaceCull ? 1 : 0;
	pushConstant.stableCandidateCut = stableCandidateCut ? 1 : 0;

	{
		static constexpr float GOLDEN = 0.6180339887498949f; // 1/phi
		if (rotateOutgoing) {
			float t = static_cast<float>(Application::frameIndex);
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
void Octree_VRPG::render(VkCommandBuffer cmd) {
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
	debugPushConstant.curLayerNodeCount_G_2 = (1 << (3 * debugPushConstant.curLayer_G_2));
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
		throw std::runtime_error("闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚敐澶婄闁挎繂鎲涢幘缁樼厱闁靛牆鎳庨顓㈡煛鐏炲墽娲存い銏℃礋閺佹劙宕卞▎娆戠┛闂傚倷绶氶埀顒傚仜閼活垱鏅堕崜褏纾奸柣妯挎珪鐏忣參鏌ｉ敐鍥у幋闁诡喒鍓濋幆鏃堫敊閹勬毄濠电姷鏁告繛鈧繛浣冲洦鍋嬪┑鐘叉搐閸氬綊鏌嶈閸撴瑩鈥旈崘顔嘉ч幖绮光偓鑼嚬缂傚倷娴囬褏鈧矮鍗抽悰顕€骞嬮敃鈧粈瀣亜閺嶃劎鈯曢柛搴邯濮婅櫣鎷惔鈥虫殲闁规彃鎲￠〃銉╂倷椤忓嫮浼堝┑顔硷工椤嘲鐣烽幒鎴旀瀻闁规惌鍘借ⅵ闂傚倷娴囬～澶岀矈閹绢喖鐒垫い鎺戝暟琚﹀┑鐐叉▕娴滄繈寮插┑瀣厱閻忕偟鍋撻惃鎴︽煥濞戞瑧娲存慨濠冩そ瀹曠兘顢橀悙鎻掝瀱婵＄偑鍊戦崝宀勫箠濡櫣鏆﹂柕澶堝妸娴滃綊鏌熼悜妯诲暗闁告ɑ鎮傚娲箹閻愭彃濮岄梺绋块叄娴滃爼骞冨鈧、鏃堝醇閻斿搫骞楅梻浣筋潐閸庢娊鎮洪妸褏鐭嗛柛鎰靛枟閻撳繘鏌涢妷鎴濆枤娴煎啫螖閻橀潧浠滄繛宸幖閻ｅ嘲顭ㄩ崟顒€顎撻梺闂寸矙閺€閬嶅垂閸︻厽顫曢柟鎯х摠婵挳鏌熺紒妯虹瑐濠㈣娲熷缁樼瑹閳ь剟鍩€椤掑倸浠滈柤娲诲灡閺呭墎鈧稒蓱閸欏繐鈹戦悩鎻掓殲闁靛洦绻勯埀顒冾潐濞插繘宕濆鍥ㄥ床婵犻潧顑呯壕鍏兼叏濡搫鎮戠紓宥呭€垮濠氬磼濞嗘帒鍘″銈庡幖閻楁捇銆侀弽顓炲耿婵炴垶顭囬澶愭⒑閹肩偛鍔€閻忕偛澧界粙浣虹磽閸屾艾鈧兘鎮為敃浣规噷濠电姵顔栭崹閬嶅箰閹惰棄钃熼柨婵嗩槸缁狅綁鏌ｈ箛鏃€銇熷ù婊勭矒閸┿垽骞橀張鍨媰缂備焦绋戦鍡欌偓闈涚焸濮婃椽妫冨☉姘暫濠碘槅鍋呴悷锕€顕ラ崟顖氱妞ゆ牗绋撻崢鍗炩攽閻愬弶顥滅紒缁樺姍瀵啿鈻庨幘瀵稿幗闂佹寧绻傚ú銈夊储鐎涙﹩娈介柣鎰▕閸庢棃鏌℃担绋库偓鍧楀箰婵犲啫绶為柛鈩冦仦婢规洟姊虹紒妯虹伇濠殿喓鍊濆畷鎰版偨閸涘ň鎷婚梺鍓插亞閸犳捇鍩ユ径瀣ㄤ簻闁靛鍎崇粻鑼磼缂佹娲寸€殿喖鐖奸獮瀣攽閹邦亝鍋呴梻鍌欑閻ゅ洭锝炴径鎰瀭闁秆勵殔閺勩儵鏌曟径鍡樻珔妤犵偑鍨烘穱濠囧Χ閸涱厽娈ㄩ梺鐟邦嚟閸忔ê顫忓ú顏呭殥闁靛牆鎲涘☉銏＄厱閻庯綆鍋呭畷宀勬煛鐏炶濮傜€殿喗鎸抽幃娆徝圭€ｎ亙澹曢悷婊呭鐢帞澹曟繝姘厱闁哄洢鍔岄悘锟犳煟閹惧瓨绀嬮柡灞界Ч閸┾剝鎷呴崨濠冾啀闂備焦鍓氶崹鍗灻洪悢鐓庤摕闁挎繂顦粻濠氭倵濞戞瑯鐒芥い锔哄姂濮婃椽妫冨☉娆愭倷闁诲孩鐭崡鎶芥偘椤曗偓瀹曞爼顢楅埀顒勬偂濞戞◤褰掓晲閸喆鈧啴鏌ｉ妷顔荤凹濞ｅ洤锕幃娆擃敂閸曘劌浜鹃柟杈剧畱绾惧潡鏌ょ喊鍗炲闁搞劍绻堥弻鐔煎箲閹伴潧娈紓浣哄У閻楁绌辨繝鍥ч柛灞剧煯婢规洘淇婇悙顏勨偓銈夊磻閸涱垱宕查柛顐犲劚缁犳牠鏌嶉埡浣告殲闁稿海鍠栭弻鏇＄疀閿濆棛绐旀繝銏ｎ潐濞叉粎妲愰幘璇茬＜婵﹩鍏橀崑鎾搭槹鎼淬埄鍋ㄩ梺璺ㄥ枔婵绮ｅΔ鍛厸鐎广儱楠搁獮鎰版煃瑜滈崜銊х不閹捐绠栨繝濠傜墛閸嬧晠鎮橀悙鍨珪闁靛牞绠撳缁樻媴閸涘﹨纭€闂佺绨洪崐婵嗙暦閻㈠憡鏅濋柛灞炬皑閻嫰姊洪崜鎻掍簼缂佽瀚板畷婵嬫晝閳ь剟鈥旈崘顔嘉ч柛鈩冾殘閻熴劑姊虹粙鍖″姛缂侇噮鍨崇划娆愬緞婵犲骸鎮戦梺鎼炲劘閸斿矂寮搁幋锔解拺缂備焦锕╁▓鏃堟煟濡も偓濡稓鍒掗銏犵闁哄啫鍊婚敍婊堟⒑缂佹鎳勯柣鐔村劚鐓ら柨鏇炲€归崑鍌炴煛閸モ晛鏋傚ù婊勭矒閺岀喖寮堕崹顕呮殺闂佽鍠涢崑鎰閹烘鍋愰柧蹇ｅ亜绾炬娊姊洪崫鍕缂佸缍婂濠氬Ω瑜夐崑鎾绘晲鎼粹€茬爱闂佸湱鏌夊畷鐢垫閹惧瓨濯村ù鐘差儏閹界敻鏌ｉ姀鈺佺仚闁逞屽墮绾绢參寮抽敃鍌涚叆婵犻潧妫欓崯鎺楁煟閺冨倸甯跺ù鑲╁█閺屾盯寮撮妸銉ョ闂佺顑嗛悧鐘差潖?-" + std::to_string(MAX_OCTREE_LAYER_VRPG - 1));

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

	for (int layerIndex = OCTREE_CLUSTER_LAYER_VRPG; layerIndex <= pushConstant.octreeMaxLayer; ++layerIndex) {
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

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getCandidateNodes);
	vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

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

	debugPushConstant.mode = 2;
	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t bufferIndex = scene.getMeshBufferIndex(meshIndex);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];

	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

	uint32_t instanceCount = debugPushConstant.curLayerNodeCount_G_2;
	vkCmdDrawIndexed(cmd, triMesh.indices.count, instanceCount, 0, 0, 0);

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