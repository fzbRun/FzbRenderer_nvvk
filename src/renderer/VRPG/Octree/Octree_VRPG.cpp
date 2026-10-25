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
			// 1) 闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ょ紓宥咃躬瀵鏁愭径濠勵吅闂佹寧绻傞幉娑㈠箻缂佹鍘遍梺闈涚墕閹冲酣顢旈銏＄厸閻忕偛澧藉ú瀛樸亜閵忊剝绀嬮柡浣瑰姍瀹曞崬鈻庡Ο鎭嶆氨绱撻崒娆掑厡闁稿鎹囧畷鏇㈠礃瀹割喚鍔烽梺璺ㄥ枔婵挳鎷戦悢鍏肩叆婵犻潧妫Σ鍛娿亜椤愩垺鍤囬柡灞炬礉缁犳稒绻濋崘鈺冨絿闂備礁鎲″褰掑箖閸屾凹娼栭柛婵嗗珔瑜斿畷鎯邦槷闁稿鎸歌灃闁告侗鍘鹃敍娑㈡⒑鐟欏嫬绀冮柛鈺傜墪閻ｇ兘宕ｆ径宀€鐦堥梻鍌氱墛娓氭宕曢幇顓滀簻闁靛鍎虫晶娑氱磼缂佹娲存鐐差儏閳规垿宕卞顒傚幋闂傚倷绀侀幉锟犲箹閳哄懎绀堥柣鏂款殠濞兼牗绻涘顔荤凹闁绘帒顭烽弻锝夊箛椤栨俺鏁块柣搴㈢濠㈡﹢鈥﹂懗顖ｆЩ婵犫拃鍛珪婵″弶鍔欓獮鎺懳旀担闀愭睏闂備焦鍎冲ù姘跺磻閸曨厼鏋堢€广儱顦伴埛鎴澝归崗鑲╂噮缂佸鍣ｉ弻锟犲焵椤掍焦缍囬柍鍝勫暟婢跺嫰姊烘导娆戝埌闁活剙銈搁幆灞解枎閹炬潙浠梺鎼炲劵缁犳垹鈧艾閰ｉ弻娑㈠棘鐠恒剱褍鈹戦敍鍕毈鐎规洜鍠栭、娑橆潩鏉堚晜缍侀梻鍌欑劍閹爼宕曞ú顏呭€舵繝闈涱儏缁犲鎮楅棃娑欐喐妞ゆ梹娲熷娲川婵犱胶绻侀梺鍝ュУ椤ㄥ﹪宕哄☉銏犵疀妞ゆ垯鍨归柊锝呯暦閹偊妾梻濠庡墻閸撶喖寮婚悢纰辨晩闁伙絽鏈崳鏉款渻閵堝啫鐏俊顐㈠暙閻ｇ兘鎮㈤悡骞晠鏌曟径鍫濃偓鏇⑺夊鑸碘拻濞达絽鎲￠幆鍫ユ偣閸モ晛浠уù鐙呯畵瀹曟粏顦撮柍璇差槹缁绘繈鎮介棃娴躲儵鏌℃担鍛婂暈闁逛究鍔戦幃浠嬪川婵犲倹鐤勯梻浣告啞閹稿棝宕橀悙顒夋％婵犵數濮烽弫鍛婃叏閻㈤潧鏋堢€广儱顦伴崑鍌涚箾閸℃ɑ灏伴柣鎾寸懇閹嘲鈻庤箛鎿冧患婵犳鍣粻鎾诲蓟濞戙垹围闁糕剝鐟﹂崳顔剧磽娴ｄ粙鍝洪悽顖ょ節閻涱噣宕堕澶嬫櫍閻熸粌閰ｉ幃鎸庣瑹閳ь剙顫忓ú顏勪紶闁告洦鍓氶幏閬嶆⒑閻戔晜娅撻柛銊ㄦ硾閻ｅ嘲煤椤忓嫮顦板銈嗙墬濮樸劑藝閵娿儺娓婚柕鍫濇噽缁狅絿绱掗埀顒佺瑹閳ь剟鏁愰悙鑼殕闁告洦鍏橀幏缁樼箾鏉堝墽绉繛鍜冪悼閺侇喖鈽夐姀锛勫幍濠殿喗绻傞惌鍫ュ吹濞嗘挻鍋傞柕鍫濐槹閻撶喖鏌￠崘銊モ偓鍝ユ暜閸洘鐓㈤柛鎰典簻閺嬫棃鏌嶈閸撴瑧绮诲澶婄？闂侇剙鍗曢崶顒夋晬闁绘劖娼欐禍妤€鈹戦悙鏉戠仸闁崇鍊涢妵鎰板箳閹惧瓨鐝抽梻浣告啞娓氭宕伴弽顓炴辈闁割偁鍎查埛鎴︽偣閸ヮ亜鐨虹紒鐘崇叀閺岀喖宕ㄦ繝鍐ㄢ偓鎰版煕閳哄啫浠辨鐐差儔閺佸倿鎸婃径鍡椾壕闁绘垼濮ら悡鐘电棯閺夊灝鑸瑰褜鍠楅妵鍕Χ閸屾矮澹曞┑鐘垫暩婵敻顢欓弽顓炵獥闁哄稁鍘介弲婵嬫煏婵炲灝鍔ら柡鍡畵濮婂宕奸悢宄板箯闂佹寧绻傞ˇ顖滅不濞戙垺鐓涘璺哄绾埖銇勯弮鈧敮鈥愁潖妤﹁￥浜归柟鐑樼箖濞堛儳绱撴担鎻掍壕婵炴挻鍩冮崑鎾搭殽閻愭彃鏆ｆ鐐差儔閺佸倻鎲撮敐鍡楊伖缂傚倸鍊风欢锟犲垂閸楃伝娲冀閵婏附鐝烽梺鍝勮閸庢煡鎮￠妷锔藉弿婵☆垰鐏濋悡鎰版煟閹惧娲撮柡灞剧洴閸╁嫰宕樻径鍝ヨ繑婵犵數鍋涢悧濠勫垝瀹ュ绠為柕濠忓缁♀偓闂佸憡鍔忛弬鍌涚閵忋倖鈷戦柛婵嗗鐎氫即鏌熼搹顐€挎鐐插暙閳诲酣骞橀弶鎴烆吇闂佽崵濮惧銊ф媰閿旇棄顕遍柛鎰╁妷閺€浠嬫煟濮楀棗浜濆┑顖欏嵆閹﹢鎮欑捄鐩掓捇鏌ｈ箛鏃€灏电紒杈ㄦ尰閹峰懏绂掔€ｎ亝鎳欓梻浣烘嚀閹诧繝骞冮崒鐐偓浣肝熼懡銈夋缂備礁顑嗙€笛囧Φ濠靛鈷戦柛娑橈工婵倿鏌涢弬娆惧剳婵″弶鍔欓弫鎰板幢閺囩姷鐣鹃梻浣虹帛閸旓附绂嶅鍫濈劦妞ゆ垶鍎抽埀顒佺箓閻ｇ兘濮€閵堝棗浜楀Δ鐘靛仧閸犳劕顪冩禒瀣ㄢ偓浣割潨閳ь剚鎱ㄩ埀顒勬煟濡吋鏆╅柡瀣暣濮婄粯鎷呴懞銉с€婇梺鍝ュУ閹稿骞堥妸鈺佺骇闁瑰鍋熼崙浠嬫⒒閸屾艾鈧兘鎮為敃鍌氱畺闁割偅娲栫壕鎸庛亜閺囨浜鹃悗娈垮枟閻擄繝銆侀弮鍫濆耿婵妫欑紞妤呮⒒娴ｇ懓鍔ゆ繛瀛樺哺瀹曟垿宕卞▎蹇撔氶梺閫炲苯澧柍瑙勫灴閹瑩鎳犻鈧。娲⒑鐠囪尙绠茬紒璇茬墕椤曪綁骞栨担鐟颁簻闂佺鍋愰崑妯艰姳鐠囧樊娓婚柕鍫濇鐏忕敻鏌涚€ｎ剙鏋戦柡鍛版硾閳藉顫濇潏鈺嬬床闂備胶顭堥張顒€顫濋妸锔芥珡濠碉紕鍋戦崐鎴﹀垂濞差亗鈧啯绻濋崒婊勬闂佸湱鍎ら崺鍫ユ偂閵夆晜鐓冪憸婊堝礈濞戞艾鍨濇い鎾跺枎缁剁偤鏌熼柇锕€骞愰柟閿嬫そ濮婃椽宕ㄦ繝鍕暤闁诲孩鍑归崜鐔奉嚕閸愭祴鏋庨柟鎯ь嚟閸樿棄鈹戞幊閸婃劙宕戦幘缁樼叆闁哄洢鍔嬮柇顖涱殽閻愭彃鏆ｉ柡浣规崌閹晠骞撻幒鎴犮偖闂傚倷绶氬褏鎹㈤崱娑樼劦妞ゆ帊鐒﹂悘閬嶆煕鐎ｎ偅宕岀€规洖銈稿鎾偄閸濆嫬绠洪梻鍌欑缂嶅﹪宕戞繝鍥у偍闁兼祴鏅涚欢銈呪攽閻樺磭顣查柣鎾冲暣濮婃椽宕归鍛壉濡炪們鍎遍悧鎾诲蓟閳ュ磭鏆嗛悗锝庡墰琚︽俊鐐€戦崹娲晝閵忋倕绠栨繝濠傜墕楠炪垺淇婇妶鍌氫壕婵犮垻鎳撻柊锝咁潖缂佹ɑ濯撮柛娑橈攻閸庢捇姊洪崫鍕⒈闁告挾鍠栭獮鍡涘礋椤栵絾鏅梺缁樺姌鐏忔瑩鎮惧ú顏呪拺闁告劕寮堕幆鍫ユ煙閸愭煡鍙勯柟顖氳嫰铻栭柍褜鍓熼垾锕傚锤濡や礁娈濋梺娲诲墻娴滄繈宕归崸妤€鏋佺€广儱顦獮銏＄箾閹寸偟鎲块柟椋庣帛缁绘稒娼忛崜褍鍩岄梺纭咁嚋缁绘繈鐛崱娑橀唶闁靛濡囬崢钘夘渻閵堝棗鍧婇柛瀣崌閺屾洟宕奸姀鈺冨姼闂佷紮缍侀弨杈┾偓闈涖偢瀵爼骞嬪┑鍡樻殢濠碉紕鍋戦崐鏍箰妤ｅ啫纾婚柣鏃傗拡閺佸﹪鏌熼幍顔碱暭闁稿鍓濈换婵囩節閸屾稑娅ｅ銈忕到閵堟悂骞冩禒瀣垫晬婵炴垼椴搁悿渚€鎮楃憴鍕闁搞劌娼″畷褰掑箚瑜滈悢鍡涙煕閺囥劌澧伴柍顖涙礋閺屻劌鈽夊▎蹇擃潔闂佺懓鍢查幊妯虹暦椤愶箑绀嬫い鎰靛亞閳ь剚鎸冲濠氬磼濞嗘帒鍘＄紒缁㈠幖閻栫厧顕ｉ幓鎺濈叆闁告侗鍨抽悡鎾绘⒑鐟欏嫬绀冮柛鈺傜墵閿濈偤寮撮姀锛勫幍闂佺顫夐崝锕傚吹濞嗘垹妫柟顖嗗啯鍊紓浣虹帛閻╊垰鐣烽崡鐐嶇喐娼弶鍨亖闂傚倷绀侀幗婊堝磻濞戞氨绀婂┑鐘插亞濞兼牠鏌ц箛鎾磋础闁活厽鐟︾换娑㈠幢濡搫濮㈤梺鍛婃崌娴滆泛顫忓ú顏勬嵍妞ゆ挾鍠庡▓灞解攽閻愬弶鈻曢柛妯恒偢椤㈡岸鏁愰崶銊ョ彴濠电偞娼欓鍡涳綖瀹ュ鈷戦梻鍫熺〒缁犲啿鈹戦鑺ョ稇闁伙絿鏁诲畷鍗炩槈濞嗗本瀚奸梻鍌氬€搁悧濠冪瑹濡ゅ懏鍊垮Δ锝呭暞閻撴洟骞栧ǎ顒€鈧牠鎮甸鍡欑＜闁稿本绋戝ù顔姐亜閵忊槄鑰垮┑顔瑰亾闂佹娊鏁崑鎾绘煙閻ゎ垱顏犵紒杈ㄦ崌瀹曟帒鈻庨幋婵嗩瀴闂佽棄鍟存禍鍫曞箖濡法鐤€闁哄洨鍎愰弳顓犵磽娴ｈ櫣甯涢柣鈺婂灠閻ｇ兘鏁愰崱娆戝骄闂佹寧绻傚Λ妤呮倶閳哄啰纾兼い鏃傛櫕椤ｅ弶绻涢悡搴ｇ濠碘剝鐡曢ˇ鎾煟閿曗偓閻楁挸顫忓ú顏勭闁绘劖褰冩慨鍫曟⒑鐠囪尙绠茬紒瀣墵钘濈憸鐗堝笚閻撴盯鎮橀悙鎻掆挃婵炴彃顕埀顒侇問閸犳骞愰搹顐＄箚闁归棿绀佸敮闂侀潧锛忕仦鑺ユ珝婵犵數濮烽弫鍛婃叏閻戝鈧倿顢欓悙顒夋綗闂佸搫娲㈤崹褰掓偪閻愵兙浜滈柟鎵虫櫅閳ь剚鐗犻敐鐐哄即閵忥紕鍘甸梺缁樺灦閿曗晛鈻撻弴銏＄厱濠电姴娲﹂弫閬嶆煏閸パ冾伃妤犵偞甯″畷鍗烆渻閹屾缂傚倸鍊搁崐椋庢媼閺屻儱纾婚柟鍓х帛閻撴洟鏌熼弶鍨倎缂併劍鎸抽弻娑氣偓锝庡亝鐏忕敻鏌熼崣澶嬪唉鐎规洜鍠栭、妤呭磼閵堝柊銈夋⒒閸屾瑦绁版俊妞煎姂閹虫宕奸弴妞诲亾閿曞倸閱囨繝闈涙閵?= 闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閹冣挃闁硅櫕鎹囬垾鏃堝礃椤忎礁浜鹃柨婵嗙凹缁ㄥジ鏌熼惂鍝ョМ闁哄矉缍侀、姗€鎮欓幖顓燁棧闂備線娼уΛ娆戞暜閹烘缍栨繝闈涱儐閺呮煡鏌涘☉鍗炲妞ゃ儲鑹鹃埞鎴炲箠闁稿﹥顨嗛幈銊╂倻閽樺锛涢梺缁樺姉閸庛倝宕戠€ｎ喗鐓熸俊顖濆吹濠€浠嬫煃瑜滈崗娑氭濮橆剦鍤曢柟缁㈠枛椤懘鏌ｅΟ鑽ゅ灩闁搞儯鍔庨崢閬嶆煟韫囨洖浠滃褌绮欓幃锟狀敍濮樿偐鍞甸柣鐔哥懃鐎氼厾绮堥埀顒勬⒑鐎圭媭娼愰柛銊ユ健閵嗕礁鈻庨幋鐘烩攺闁诲函缍嗛崜娆愭叏濞差亝鈷掑ù锝呮啞閸熺偞銇勯鐐搭棦鐎规洜鍠栭、妤佹媴閸濆嫬鈻忔繝纰夌磿閸嬫垿宕愰弽顓熷亱婵犲﹤鐗嗙壕濠氭煟閹邦喖鍔嬮柛瀣€圭换娑㈠箣濞嗗繒浠鹃梺缁樻尰濞叉鎹㈠☉銏犵闁绘劕鐏氶崳顓犵磽娴ｈ棄鐓愰柣鎿勭節瀵鏁愭径濠勵啋闂佸搫顦伴崹鐢革綖鎼达絿纾藉ù锝勭矙閸濇椽鎮介妞诲亾瀹曞洦娈鹃梺鍦劋閸ㄧ喎危閸喍绻嗘い鏍ㄥ閹凤繝鏌涙惔锛勭劯婵﹦绮粭鐔煎焵椤掑嫬鐒垫い鎺戝€告禒婊堟煠濞茶鐏￠柡鍛埣瀹曟粏顦寸痪鍓ф櫕閳ь剙绠嶉崕閬嶅箠韫囨蛋澶愬閵堝棛鍘遍梺瀹狀潐閸庤櫕绂嶉悙顑跨箚闁绘劦浜滈埀顒佺墱閺侇噣骞掑婵嗘喘椤㈡稑顫濋悡搴㈩吙闂備胶顫嬮崟鍨暦闂佹娊鏀辩敮锟犲蓟濞戙垹鍗抽柕濞垮劙缁ㄨ顪冮妶鍛闁告ü绮欓妴鍐Ψ閳哄倸鈧兘鏌ょ喊鍗炲闁愁亜鐏氱换婵堝枈濡搫鈷夐梺璇″枛閸婅绌辨繝鍥ㄧ叆閻庯綆鍋勯崝鍛存⒑閹稿海绠撻柟鍐茬箻閹﹢宕煎顏呮閹晠妫冨☉妤佸媰闂備礁鎲″褰掓偋閻樿崵宓侀柛鈩冪☉缁秹鏌涢銈呮瀾婵炲懏绮撳娲川婵犲倸衼缂備浇灏慨銈囩矉瀹ュ棎鍋呴柛鎰ㄦ櫇閸樼敻姊绘笟鍥у伎缂佺姵鍨垮鎼佸Χ閸モ晜锛忛梺璇″瀻娴ｉ晲鍒掗梻浣告惈閻ジ宕伴弽顓溾偓浣糕槈閵忕姴鑰垮┑掳鍊愰崑鎾绘煛閸℃瑥鏋旂紒杈ㄥ笧閳ь剨缍嗘禍鐐电不瀹曞洨纾奸弶鍫涘妽鐏忔壆绱掑Δ鍐ㄦ灈鐎规洘鍎奸ˇ鎶芥煟韫囨挾效婵﹥妞介幊锟犲Χ閸涱喗鐦ｉ梻浣告惈閹虫劖绻涢埀顒勬煟濞戝崬娅嶇€规洘锕㈤、娆撴嚍閵夛富浠ч梻鍌欒兌缁垶鎮烽姀銈呯；闁瑰墽绮悡娑樏归敐鍥剁劸闁哄棴缍侀弻锛勪沪鐠囨彃顫囬悗娈垮枟濞兼瑨鐏冮梺閫炲苯澧紒鍌氱У閵堬綁宕橀埞鐐缂傚倸鍊烽悞锕傛晪婵犳鍠栧ú顓㈠蓟閵娾晛鍗抽柕濞垮€濋崑妤呮倵濞堝灝鏋涙い顓炲槻椤曪綁骞橀钘変簻闂佹儳绻愬﹢鍗炍ｉ崶顒佲拻濠电姴楠告禍婊勭箾鐠囇呯暤妤犵偞鐗犻崺鈩冨鏉堛劍銇濇い銏★耿閹粓宕卞Ο铏规В闂傚倷绶氬褔鎮ч崱娆愬床闁规壆澧楅崕濠冪節婵犲倻澧涢柣鎾存礃娣囧﹪顢涘☉妤勫焻闂佺顑冮崹浠嬪蓟瀹ュ唯闁挎梻鐡斿Λ鍐倵鐟欏嫭绀冮柨鏇樺灲閵嗕礁鈻庨幘鍐茬哎婵犮垼顕栭崹鏉棵洪妸鈺佺煑闁瑰墽绮悡鏇㈡煏婢舵稓鍒板┑锛勬櫕缁辨帡鍩€椤掑嫬閱囬柕澶涘閸樻悂姊虹化鏇炲⒉闁告梹娲熼崺鈧い鎺嶈兌婢ч亶鏌熼獮鍨仼闁宠棄顦～婵嬫晲閸涱剙顥氭繝鐢靛仦閸ㄥ爼鎮烽鐐村€垮ù鐘差儐閻撶喐銇勯幘璺哄壋闂婎剦鍓熼弻鈥崇暆鐎ｎ剛袦閻庢鍠曠划娆撱€侀弴銏狀潊闁绘浜敮娑樷攽?
			const float localX = mousePos.x - vpOrigin.x;
			const float localY = mousePos.y - vpOrigin.y;

			// 2) 闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ょ紓宥咃躬瀵鏁愭径濠勵吅闂佹寧绻傞幉娑㈠箻缂佹鍘遍梺闈涚墕閹冲酣顢旈銏＄厸閻忕偛澧藉ú瀛樸亜閵忊剝绀嬮柡浣瑰姍瀹曞崬鈻庡Ο鎭嶆氨绱撻崒姘偓鐑芥嚄閼稿灚鍙忛梺鍨儑缁犻箖鏌嶈閸撶喖寮婚垾宕囨殕闁逞屽墴瀹曚即寮介鐘茬ウ闂佺硶鍓濋崙鐟拔ｇ拠宸富闁靛牆绻掗崚浼存煕閻旈攱鍋ラ柛鈹垮劜瀵板嫰骞囬鍌ゅ晪闁诲氦顫夊ú鏍р枖閿曞倸围濠㈣泛顑囬崢鎾绘偡濠婂嫮鐭掔€规洘绮岄埢搴ㄥ箣閻樿京鐟濋柣搴＄畭閸庨亶藝椤栨粌顥氶柛褎顨嗛埛鎴炪亜閹哄棗浜鹃梺鍛婎殕濞茬喎鐣烽崼鏇ㄦ晢濠㈣泛顑嗗▍鏍⒒娴ｈ櫣甯涢柛鏃€鐗為妵鎰板礃椤旇棄鈧灚銇勯幘鍗炵仾闁抽攱鍨圭槐鎺斺偓锝庡亜椤曟粍绻濋埀顒佸鐎涙鍘遍梺瀹犳〃缁€渚€寮抽敐鍥ｅ亾濞堝灝鏋旈柛鏂跨Ф缁骞掗弬鍝勪壕闁挎繂楠告禍鐐烘煕濡寧顥夐柍瑙勫灴閸┿儵宕卞Δ鍐ф樊婵犵數鍋涢幏鎴犵礊娓氣偓瀵偄顓奸崪浣告倯闂佺硶鍓濋崝鎺旇姳婵犳碍鈷戦柟绋挎捣缁犳挻淇婇锝囨创鐎规洩缍侀崺鈧い鎺戝閳锋垿鏌涘┑鍡楊仾婵犫偓閹殿喚纾奸悗锝庡亝瀹曞矂鏌涢埞鎯т壕婵＄偑鍊栫敮鎺楁晝閵夆晛鐒垫い鎴炲劤閸旓妇鈧娲滈崰鏍€佸Δ鍛＜婵ê宕崺宀勬⒒閸屾艾鈧悂宕愭搴ｇ焼濞撴埃鍋撻柟顔兼健瀹曞爼顢楁繝鍕棨濠电姰鍨奸崺鏍礉閺囥垹鍨傞柛宀€鍋為悡鏇犳喐鎼淬劊鈧啴宕ㄩ婊€绗夐梺鍝勭▉閸樹粙鎮￠崘顔界厱婵犻潧妫楅鈺呮煛閸℃韬柡灞剧〒閳ь剨缍嗛崑鍛暦瀹€鈧埀顒冾潐濞叉粓宕楀鈧妴浣割潨閳ь剟骞冨▎鎾崇疀闁宠桨鑳舵禒楣冩⒑闂堟稒鎼愰悗姘卞娣囧﹪鎮滈懞銉︽珕闂佸憡鎸嗛崨顔界様缂傚倸鍊搁崐宄懊归崶銊ｄ粓闁归棿绀佺粻鏉库攽閻樺疇澹樼紒鐙€鍨堕弻娑樷槈濞嗘劗绋囩紓浣哄Х閺佸寮婚悢鐓庝紶闁告洦鍘滈姀掳浜滈柕澶涚畱閸濈儤鎱ㄦ繝鍌ょ吋鐎规洘甯掗埢搴ㄥ箳閹存繂鑵愮紓鍌欒兌閾忓酣宕㈡總鍛婂亯闁绘挸娴烽弳锔界節婵犲倸鏆婇柡鈧禒瀣厱妞ゆ劗濮撮悘顕€鏌ｉ幘瀛樺磳婵﹥妞藉畷鐑筋敇閻愭彃顬嗘俊鐐€戦崝灞轿涘┑鍡欐殾闁硅揪绠戠壕濂告煟閹邦剦鍤熼柛妯圭矙濮婃椽宕崟顐ｆ闂佺粯顨呴敃銈夊煝瀹ュ拋鐓ラ柛娑卞灣閿涙粓鏌ｉ悩鍙夋悙鐎殿喖鐖艰棢闁靛繈鍊栭悡娑㈡倶閻愰潧浜剧紒鈧€ｎ兘鍋撶憴鍕闁告鍥х厴闁硅揪绠戦獮銏ょ叓閸ャ劍灏瑙勬礋濮婃椽宕ㄦ繝鍕ㄦ闂佹寧娲╃粻鎾诲Υ閸涙潙绠甸柟骞垮妽濡啴寮幇鏉垮窛妞ゆ挾鍠愰澶愭⒒娴ｇ懓鈻曢柡鈧柆宥呭瀭闁革富鍘介～鏇㈡煙閹呮憼濠殿垱鎸抽弻锝堫樄闁哄懓灏欏▎銏ゆ濞戞帗鏂€闂佺粯鍔樼亸娆愭櫠濞戞瑧绠鹃柤纰卞墮閺嬫稓鈧鍠涢褔鍩ユ径鎰潊闁绘鏁稿澶愭⒒娓氣偓閳ь剛鍋涢懟顖涙櫠閹绢喗鐓欐い鏃€鍎抽崢鎾煕閳哄绡€鐎规洘甯掗埢搴ㄥ箣閻樺灚姣庨梻鍌氬€风粈浣革耿闁秵鎯為幖娣妼缁愭鏌″畵顔艰嫰閺呯姴鈹戦悩缁樻锭妞ゆ垵妫濋幃鈥斥枎閹寸姵锛忛梺鍝勵槸閻忔繈鎳滅憴鍕垫闁绘劖鎯屽▓婊堟煛瀹€瀣瘈鐎规洏鍔戦、娆撳礈瑜忛弶鑺ョ節濞堝灝鏋涢柨鏇樺劚椤啯绂掔€ｎ剙绁﹂梺褰掑亰閸樹粙宕曢悢鍏肩厓闁搞劍绋掑▍鍡涙倵濮橆偆鐭欐慨濠勭帛閹峰懘宕ㄦ繝鍐ㄥ壍闂備胶鎳撻崲鏌ヮ敄婢跺﹦鏆﹂柟瀵稿仧缁♀偓闂佺鏈划宥呪枔妤ｅ啯鈷戦梻鍫熺〒缁犵偤鏌涙繝鍐╃缂侇喖顭烽幃鐑藉箠婵犲倻绉洪柡浣瑰姍瀹曘劑顢欑捄顭戞敤闂傚倷绀侀幉锛勬箒缂備緡鍠楅悷銉ㄧ亱闂佸憡绋掗幆宀勫极閸愵喗鐓忛煫鍥堥崑鎾诲箛娴ｅ墣鐔封攽閻樺灚鏆╅柛瀣☉椤曪綁宕奸弴鐔蜂罕濠德板€曢崯浼存儗婢舵劖鐓欓柣鎴炆戞径鍕繆椤栨艾顏慨濠冩そ瀹曨偊宕熼浣瑰缂傚倷绀侀鍡涘箲閸ヮ剙鏄ラ柍褜鍓氶妵鍕箳閹搭垰濮涚紓浣割槺閺佸寮诲☉姘ｅ亾閿濆簼绨奸柛锝堟闇夋繝濠傚閸婃劖淇婇銏犳殭闂囧鏌曡箛瀣仾闁搞倐鍋撴俊鐐€ら崢褰掑礉閹存繄鏆﹂柣鏃傗拡閺佸棝鏌嶈閸撶喖宕洪埀顒併亜閹哄秶顦﹀┑顔肩墦閺屾洟宕惰椤忣厼鈹戦鐟颁壕闂備礁鐤囧Λ鍕箠閹捐鐒垫い鎺嗗亾闁诲繑绻堥崺鐐哄箣閿旂粯鏅╅梺鑽ゅ枛椤ｏ附绔熼弴銏♀拻闁稿本鐟ч崝宥夋倵缁楁稑鎳愰惌娆撴煙鐎电袥闁稿鎸搁～婵嬫偂鎼粹檧鎷梻浣虹帛娓氭宕抽敐鍛殾濠靛倻顭堝敮闂佹寧妫侀～澶屸偓姘虫閳规垿鎮欑€涙ê闉嶉梺绯曟櫅閸熸潙鐣烽幋锕€绠婚悗闈涙憸椤旀洘绻涙潏鍓у埌闁哥喎鐏濈叅闁绘顕ч悞鍨亜閹烘埊鍔熼柣锝囨暩閳ь剝顫夊ú鏍Χ閸涘﹣绻嗛柣鎴ｆ鍞悷婊勭箘濡叉劙鎮介崨濞炬嫽闂佺鏈悷褏鎷规导瀛樼厱闁规儳顕幊鍛磼椤旂晫鎳勭紒妤冨枛閸┾偓妞ゆ巻鍋撴い鏇秮椤㈡洟鏁冮埀顒勭嵁閵忥紕绠鹃柟瀵稿仧閹冲懘鏌涘鍡曢偗婵☆偄鎳橀、鏇㈠閳ュ啿浜堕梻浣侯焾妤犲摜澹曢鐔虹彾闁哄洨濮靛畷澶愭煏婵炑冩嫅缁辨煡姊绘担铏瑰笡妞ゎ厼鐗嗛悾鐑芥偨閸涘﹤浠煎┑鐐叉▕娴滄繈鎮″☉姘ｅ亾閸忓浜鹃柣搴秵閸撴稖鈪靛┑掳鍊楁慨鐑藉磻閻愮儤鏅濋柕蹇嬪€曠粻鍦喐閻楀牆淇柡浣稿暣閺屻劌鈹戦崱妯烘畻濠碘槅鍋婇ˉ鎾舵閹捐纾兼慨姗嗗厴閸嬫捇鎮滈懞銉ユ畱闂佽偐顭堥悘姘跺矗韫囨拹鏃堟晲閸涱厽鐏撻梺杞扮椤戝洭骞夐幖浣哥睄闁割偅绻冨Σ顒勬⒑缂佹ê濮囬柟纰卞亞缁鏁愭径瀣弳闂佸搫娲﹂〃鍛妤ｅ啯鈷戠紓浣姑粭姘繆閻愭壆鐭欑€殿喛顕ч埥澶愬閻樼數鏉搁梻浣侯焾缁绘劙骞楀鍫濈鐟滅増甯楅埛鎴︽煟閻斿憡绶茬紒鐙欏洦鐓熼柣鏃€娼欓崝锕傛煃閵夛附顥堢€规洘锕㈤、娆撳床婢诡垰娲ㄩ崣鎾绘煕閵夛絽濡块柍顖涙礃閵囧嫭鎯旈敍鍕懷勬叏婵犲懏顏犵紒杈ㄥ笒铻ｇ紓浣姑崜鐢电磽閸屾瑨鍏岀紒顕呭灣閹广垽宕掗悜鍡樻櫔闂佹寧绻傚Λ娑氬姬閳ь剙鈹戦鏂や緵闁稿簼绲婚ˇ鍙夈亜椤撯剝纭堕柟鐟板缁楃喖顢涘顒€顥嶉梻鍌欑窔閳ь剛鍋涢懟顖涙櫠閹殿喚纾兼い鏇炴噹閻忥紕绱掑畝鍐摵缂佺粯绻堝畷鍫曟嚋閸偅鐝┑鐘垫暩婵炩偓婵炰匠鍛亾濮樷偓閸パ咃紱闂佽宕橀褔鎮為崹顐犱簻闁圭儤鍨甸顏堟煟閹惧娲撮柟顔筋殜閺佹劖鎯旈垾鑼晼闂備胶鎳撻崯鍨洪敃鍌氱疄闁靛濡囩弧鈧梺鍛婂姀閺傚倹绂掗姀銈嗗€甸悷娆忓婢跺嫰鏌涚€ｎ亷宸ラ柣锝囧厴閹垻鍠婃潏銊︽珜濠电偠鎻徊鍧楀箠閹炬椿鏁傞柟娈垮枤绾句粙鏌涚仦鎹愬闁逞屽墴椤ユ挾鍒掗崼鐔虹懝闁逞屽墴閵嗕線寮介鐐茶€垮┑鐐村灦閿氶柣搴弮濮婅櫣绮旈崱妤€顏ュù鐘靛█閺屽秷顧侀柛蹇旂〒缁牊鎷呴悾灞艰埅闂傚倷鑳堕崢褔锝為弴銏犵９婵犻潧妫悗鑸点亜閺傚灝鈷旂痪鎹愭闇夐柨婵嗘噹椤ュ繘鏌涙惔锛勭闁哄矉缍€缁犳盯寮崒婊呮毉缂傚倷绶￠崳顖滃緤娴犲鐓″璺侯煬濞笺劑鏌涢埄鍏狀亪鎯佸鍫熲拻濞撴埃鍋撴繛浣冲洦鍋嬮柛鈩冦亗濞戞ǚ妲堟繛鍡樺姈閻濆嘲顪冮妶鍡楃瑨闁稿﹨宕垫竟鏇㈠礂缁楄桨绨婚梺鍝勫暊閸嬫捇鏌涙惔銏犫枙闁糕斂鍨藉畷濂稿煑閳轰椒澹曢柣鐔哥懃鐎氼厽寰勯崟顖涚厱闁规崘娉涢弸娑欘殽閻愬澧抽柕鍥ㄥ姍楠炴帡骞嬮悙鑼偠濠碉紕鍋戦崐鏍箰閻愵剙鍨旈悗闈涙啞椤洟鏌熼幆鏉啃撻柍閿嬪笒闇夐柨婵嗗椤掔喖鏌ｉ幒妤冪暫闁哄矉绻濆畷濂割敃閵忕姭鎷梻浣筋嚃閸犳岸宕楀Ο璁崇箚闁归棿绀侀悡娑樷攽閻樻彃鏆炴繛鍛尭閳规垿鎮╅鑲╀紘闂佺硶鏅滈悧鐘茬暦濠靛鍐€妞ゆ挾鍠庢禍?
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
	//濠电姷鏁告慨鐑藉极閸涘﹥鍙忛柣鎴ｆ閺嬩線鏌涘☉姗堟敾闁告瑥绻橀弻锝夊箣閿濆棭妫勯梺鍝勵儎缁舵岸寮诲☉妯锋婵鐗婇弫楣冩⒑閸涘﹦鎳冪紒缁橈耿瀵鏁愭径濠勵吅闂佹寧绻傚Λ顓炍涢崟顖涒拺闁告繂瀚烽崕搴ｇ磼閼搁潧鍝虹€殿喖顭烽幃銏ゅ礂鐏忔牗瀚介梺璇查叄濞佳勭珶婵犲伣锝夘敊閸撗咃紲闂佺粯鍔﹂崜娆撳礉閵堝棎浜滄い鎾跺Т閸樺鈧鍠栭…鐑藉极閹邦厼绶炲┑鐘插閺夊憡淇婇悙顏勨偓鏍暜婵犲洦鍊块柨鏇炲€哥壕鍧楁煙閹冾暢缁炬崘妫勯湁闁挎繂鎳忛幆鍫焊韫囨稒鈷戦柛娑橆煬濞堟洖鈹戦悙璇у伐妞ゆ洩绲剧换婵嗩潩椤撶偘绨婚梻浣侯焾缁绘劙鎯冮悜钘夌濠电姵纰嶉悡鐔煎箹濞ｎ剙濡芥繛鎳峰喚鐔嗛柣鐔峰簻瀹搞儵宕℃潏銊ｄ簻闁哄倸鐏濋埛鏃傜棯閹规劖顥夐棁澶愭煥濠靛棙鍣洪柟顖氱墦瀹曨垶宕ㄦ繝鍕槇闂佹眹鍨藉褎绂嶈ぐ鎺撶厱閻庯綆鍋呭畷灞炬叏婵犲啯銇濈€规洦鍋婂畷鐔碱敇閻樿京绀冮梻鍌欑閹芥粍鎱ㄩ悽鍛婂亱闁绘宕靛畵渚€鏌涢幇闈涘季婵☆垰瀚伴幃妤冩喆閸曨剛顦ラ梺闈涚墛閹倿鐛崘顔碱潊闁靛牆鎳嶇槐鍫曟⒑閸涘﹥澶勯柛瀣嚇椤㈡﹢鎳栭埡鍐紳闂佺鏈懝楣冨焵椤掍焦鍊愮€规洘鍔欓幃婊堟嚍閵夈儲鐣遍梻浣藉亹椤牓鎮樺璺虹煑闊洦绋掗悡鐔兼煏韫囨洖校闁哥喓鍋ら弻娑氣偓锝庝憾閸庢棃鏌″畝瀣瘈鐎规洖鐖兼俊鐑藉Ψ瑜岄幃锝夋⒒娴ｈ櫣甯涢柟绋款煼閺佸啴濮€閵堝懓鎽曢梺缁樻煥閹诧紕绱為崶顬棃鏁愰崨顓熸濡炪倕楠哥粔鐟邦潖閾忓湱鐭欐繛鍡樺劤閸撻亶姊洪崨濠冣拹闁荤喆鍎撮悘瀣⒑閸涘﹦鈯曢柟顖氱焸瀵偊宕橀埞澶哥盎闂佽澹嬮弲娑㈡倶閳哄啰纾奸柣妯烘惈閸氬湱绱掓潏銊﹀鞍闁瑰嘲鎳橀獮鎾诲箳瀹ュ拋妫滈梻鍌氬€风粈浣圭珶婵犲洤纾诲〒姘ｅ亾鐎规洘娲熷濠氬Ψ閿曗偓娴滄鈹戦悙鍙夘棡闁圭顭烽幃锟犲即閵忋垹褰勯梺鎼炲劘閸斿繑绔熷Ο濂界懓顭ㄩ崘顏喰ㄩ梺鍝勭焿缁绘繂鐣烽崡鐑嗘建闁割偓绲洪崑鎾诲锤濡や胶鍘搁柣蹇曞仜婢ц棄煤鐎电硶鍋撳▓鍨珮闁告挻鐩獮鎴﹀礋椤栨鈺冩喐瀹€鈧Σ鎰熼懡銈囩槇闂佹眹鍨藉褎绂掗敃鍌涚厱闁靛鍎崇粔铏光偓娈垮櫘閸嬪懐鎹㈠┑鍡╂僵妞ゆ帊鐒﹁ⅲ闂傚倷绶氶埀顒傚仜閼活垱鏅堕鐐寸厪闁搞儜鍐句純濡ょ姷鍋炵敮锟犵嵁鐎ｎ亖鏀介柟閭︿簽濡垶姊婚崒娆戝妽濠电偛锕銊╂焼瀹ュ懎鐎梺闈╁瘜閸樹粙锝為弴銏＄厵闁诡垎鍛喖婵犳鍨卞娆撳煘閹达富鏁婇柦妯侯槺閸斿憡绻涚€涙鐭嬬紒璇茬墦閻涱噣宕奸妷銉庘晠鏌嶆潪鎷屽厡闁稿﹦鍋ゅ娲濞戣京鍔搁梺绋匡攻椤ㄥ牏鍒掔拠宸僵妞ゆ帒顦扮€靛矂姊洪棃娑氬婵☆偅绋掗弲鍫曨敆閸曨剛鍘介梺鍝勫暞閸ㄥ灚鏅堕幘顔界厽闁瑰灝鍟晶瀛樸亜閵忊槅娈滅€规洘甯掕灃闁逞屽墯缁傚秵銈ｉ崘鈺佷画濠电偛妫楃换鎰邦敂椤忓嫷鐔嗛悹鍝勬惈椤忣參鏌＄仦鍓ф创妤犵偞顭囬幑鍕倻濡棿閭┑掳鍊楁慨鐑藉磻閻愮儤鍋嬮柣鎰惈妗呴梺鍦濠㈡绮婚幎鑺ョ厵閻庣數顭堟禒婊呯磼娴ｈ棄鐓愬ǎ鍥э躬閹瑩顢旈崟銊ヤ壕闁哄稁鍘肩粻浼存煟閹伴潧澧柛銊︽閹妫冨☉娆忔殘濡炪値鍋呭ú鐔煎蓟濞戙垹绠涙い鎾跺О閸嬬偤姊虹悰鈥充壕婵炲濮撮鍡涙偂閻斿吋鐓欓梺顓ㄧ畱閻忕娀鏌ｉ幘鏉戝闁瑰弶鎮傞獮搴㈢節閸屾稑娅х紒鐐劤閵堟悂寮婚悢铏圭＜婵☆垵娅ｉ悷鏌ユ⒑缁嬫鍎嶉柛搴㈢叀閵嗗啴濡烽埡鍌氣偓鐑芥偣閸ュ洤妫楅幆鍫熶繆閵堝洤啸闁稿鍋ら垾锕傚炊椤掆偓缁狀垶鏌ｅΟ鑲╁笡闁搞倖鍨堕妵鍕箳閹搭厽效闁哥喐鎮傚濠氬磼濮橆兘鍋撴搴㈩偨婵﹩鍓﹂悞鐣屾喐閺冨牆绠栫憸鏂跨暦閸楃儐娓婚柕蹇ョ磿閳藉鏌嶉挊澶樻Ц妞ゎ偅绻堥、妤呭焵椤掑啠鍙㈤梻鍌氬€风欢姘跺焵椤掑倸浠滈柤娲诲灡閺呭墎鈧稒蓱閸欏繐鈹戦悩鎻掓殲闁靛洦绻堥弻锛勪沪閼恒儱娈楅梺璇″枟閻熲晠骞婇悩鍨磯濞撴凹鍨槐鏃€绻濋悽闈浶涢柟宄板暣瀹曠喖顢楅埀顒勫船婢舵劖鈷戦柛婵嗗閻掕法绱掓潏銊︾缂佸倹甯￠弫鍐磼濞戞ɑ鐤呴梻渚€娼чˇ浠嬪窗濡ゅ惤澶愭晲閸℃洜绠氶梺缁樺姦娴滄粓鍩€椤掍胶澧垫鐐村姍楠炴牗鎷呴懖婢懐纾奸悗锝庡幗绾爼鏌￠崱顓犵暤闁哄本娲樼换娑㈡倷椤掍胶褰熼梻浣芥〃缁€浣虹矓閻㈠灚宕叉繝闈涙川缁♀偓闂佺鏈崙瑙勭妤ｅ啯鈷?
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
		throw std::runtime_error("闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ょ紓宥咃躬瀵鏁愭径濠勵吅闂佹寧绻傞幉娑㈠箻缂佹鍘遍梺闈涚墕閹冲酣顢旈銏＄厸閻忕偛澧藉ú瀛樸亜閵忊剝绀嬮柡浣瑰姍瀹曞崬鈻庡▎鎴犫敍闂傚倸鍊风欢姘跺焵椤掑倸浠滈柤娲诲灡閺呭爼宕滆绾惧ジ鏌ｅΟ鎸庣彧閻忓浚鍙冮弻锝夋晲閸パ冨箣闂佽鍠掗崜婵嬪箚閺冨牜鏁婇柟顖嗗嫭姣勬繝鐢靛Х閺佸憡绻涢埀顒佺箾娴ｅ啿娲﹂崑瀣攽閻樺弶鎼愰柛姘秺閺屽秷顧侀柛鎾寸懇閳ユ棃宕橀鍢壯囧箹缁厜鍋撻懠顒€鍤紓鍌氬€峰ù鍥敋瑜忛埀顒佺煯閸楁娊鎮伴鈧獮瀣晝閳ь剛绮堢€ｎ偁浜滈柡宥冨妿閳洟鏌涙惔顔婚偗婵﹨娅ｉ幏顐ｆ償閳ヨ櫕娈查梺瑙勫絻閹诧繝銆冮妷鈺傚€锋い蹇撳娴煎牆鈹戦纭峰伐妞ゎ厼鍢查悾鐑藉箳閹存梹鐎婚梺瑙勬儗閸樺€熲叺闂傚倸鍊峰ù鍥綖婢跺瞼鐭堥柟缁㈠枛閻掑灚銇勯幒鎴濇殶鐞氾箑鈹戦悙鍙夆枙濞存粍绻堝鎻掆攽鐎ｎ偆鍘遍柣蹇曞仧閸嬫捇鎯冮幋锔界叆婵炴垶鐟уú瀛樻叏婵犲啯銇濈€规洜鍏橀、姗€鎮欓幓鎺濈€卞┑锛勫亼閸婃垿宕濆畝鍕疇婵☆垵娅ｉ弳锕傛煏婢跺牆濡稿ù婊冪秺閺岀喖鎮滃Ο璇叉殫闂佸憡蓱閹倸顫忓ú顏勭闁绘劖褰冩慨宀勬⒑缁嬪潡鍙勫ù婊冪埣楠炲啫顫滈埀顒勩€侀弮鍫濋唶闁绘柨鎼獮妤呮⒒娴ｇ瓔娼愰柛搴㈠▕閹椽濡歌閻棝鏌涢幇闈涙灍闁绘挸绻橀弻娑㈠Ψ閹存繂鏋ゅù鐓庡暙铻栭柣姗€娼ф禒婊勭箾瀹割喖骞栭柣锝呭槻椤劑宕熼鈧鎾绘⒑闂傚鐭欓柡鈧柆宥呭瀭闁革富鍘介～鏇㈡煙閹呮憼濠殿垱鎸抽弻鐔虹磼濡櫣鐟愭繝銏ｎ潐濞茬喎顫忕紒妯肩懝闁逞屽墴閸┾偓妞ゆ帒鍊告禒婊堟煠濞茶鐏￠柡鍛閳ь剚绋掕摫闁告瑥绻愰埞鎴︽偐閹绘帗娈查梺闈涙处缁诲嫰鍩€椤掑喚娼愭繛鎻掔箻瀹曟繂顓奸崶銊ュ簥濠电娀娼ч鍛閸忓吋鍙忔俊顖氭惈閹垹绱撳鍛偓鍨潖婵犳艾纾兼繛鍡樺笒閸樷€愁渻閵堝骸骞栭柣妤佹崌閵嗕線寮介鐐茶€垮┑鐐村灦椤洭顢欐径鎰拺闁硅偐鍋涢崝鈧柣蹇曞仜婢х晫绮欐担铏圭＝闁稿本鑹鹃埀顒傚厴閹偤鏁冩担瑙勫櫡婵犵數濮甸鏍垂闁秴绠伴柟鎯版閽冪喖鏌ㄥ┑鍡╂Ц缂佺媴缍侀弻锝堢疀閺冣偓閵囩喎霉濠婂嫮鐭掗柛鈹垮灲楠炴﹢寮甸崹顐㈠缂傚倷鐒︾粙鎴︻敄閸℃瑢鍋撻棃娑氱劯婵﹥妞藉Λ鍐ㄢ槈濮橆剦鏆繝纰樻閸嬪懘鎮烽敃鈧銉╁礋椤栨氨顦板銈嗙墬缁嬫捇宕㈤崡鐐╂斀闁绘劕寮堕ˉ婊呯磼缂佹ê濮嶇€殿喖鍟块埢搴ㄥ箻鐎电骞楅梻浣瑰缁诲倸煤閵堝鍌ㄩ悗娑欙供濞堜粙鏌ｉ幇顒佲枙闁稿孩妫冮弻鈩冩媴缁嬪簱鍋撻崸妤€绠板┑鐘插暙缁剁偤鏌涢埄鍐︿沪濠㈣娲熷铏圭磼濡櫣浼囨繝娈垮枔閸婃繂鐣烽幇鐗堝仺闁告稑艌閹峰姊洪崜鎻掍簽闁哥姵鎹囬崺銉﹀緞鐎ｃ劋绨婚梺闈涱槶閸庡磭绮婚懠顒傜＜缂備焦顭囧ú瀵糕偓娈垮枛閻栧ジ鐛€ｎ亖鏀介柟閭︿簼閸嬪懘姊婚崒娆戭槮闁汇倕娲敐鐐村緞閹邦剙鐎梺绉嗗嫷娈旈柡鍕╁劦閺屾洘寰勯崱妯荤彅濡ょ姷鍋戦崹鐑樼┍婵犲洤围闁告侗鍘藉▓銊╂⒑閻熼偊鍤熼柛蹇斆～蹇撁洪鍛闂侀潧鐗嗛幉娑樷槈閵忥紕鍘遍柣搴秵閸嬪懎鐣峰畝鍕厸閻忕偠顕ф慨鍌溾偓娈垮枟閹告娊骞冨▎寰濆湱鈧綆浜欐竟鏇㈡偡濠婂懎顣奸悽顖涘笧婢规洘绻濆顓犲幈闂佸搫娲㈤崝宀勬倶閿熺姵鐓熼柟鎯х摠缁€瀣煛鐏炵晫效闁糕斁鍓濋幏鍛村川婵犲喚鍟€闂傚倷鐒﹂崜姘跺垂閸楃伝娲偄閻撳氦鎽曢梺鎸庣箓椤︻垳绮绘繝姘€垫繛鎴炵懐閻掕姤銇勯敂鍝勫婵﹥妞藉Λ鍐ㄢ槈濞嗘劖鍊烽梺璇插閻噣宕￠幎鑺ュ仒妞ゆ洍鍋撶€规洖鐖奸、妤呭焵椤掑嫭鍋傛繛鎴炩棨瑜版帗鏅查柛顐ゅ枂閳ь剙鍟撮弻锝夊Ψ椤旇崵鍑规繛锝呮搐閿曨亪骞冨▎鎿冩晜闁告洏鍔屾禍楣冩煙鏉堝墽鐣辩痪鎯ф健閺屻倗鍠婇崡鐐差潻闂佹悶鍔嶇换鍫ュ蓟閻旂厧绠查柟浼存涧濞堫厾绱撴担鍝勑ｉ柣妤侇殕缁岃鲸绻濋崶顬囨煕鐏炲墽鐓瑙勬礃娣囧﹪鎮欓鍕ㄥ亾閵堝纾婚柛娑卞灡瀹曟煡鏌涢鐘插姎缂佺姵鐗犻弻宥夊煛娴ｅ憡娈查梺绋挎捣閸犳牠寮婚弴锛勭杸闁挎繂妫涚粣鏃€绻濋姀锝庢綈婵炲弶绮庡Σ鎰板箻鐠囪尙锛滃┑顔斤供閸忔﹢宕戦幘鎼Ч閹兼番鍩勯崑銊╂⒑鐠恒劌鏋斿┑顔碱嚟缁絽螖閸涱喚鍘搁悗骞垮劚妤犳悂鐛幇鐗堢厓鐟滄粓宕滈妸褏涓嶉柟鎹愵嚙缁犳牗绻濇繝鍌滃闁稿鏅犻幃姗€鎮欓崹顐ｇ彧闂侀潧鐗炵粻鎾愁潖缂佹ɑ濯撮柛娑橈龚绾偓闂備胶顭堢花娲磹濠靛棛鏆﹂柣銏犳啞閺呮繈鏌涚仦鐐殤闁活偄瀚板娲礈閹绘帊绨肩紓浣筋嚙鐎氭澘鐣峰┑瀣櫇闁逞屽墴閳ユ棃宕橀鍢壯囨煕閳╁喚娈橀柣鐔村姂濮婅櫣绮欓崠鈥冲缂備緡鍣崹宕囧垝濞嗘劕绶炲┑鐘查閹垿姊洪幖鐐插姌闁告柨鐭傚鎼佸箣閿旇В鎷虹紓鍌欑劍閿曗晛鈻撻弮鍫熺厽婵°倐鍋撴俊顐ｇ〒閸掓帡顢橀姀鐘殿唺闂佸搫鍟崐濠氭晬濠婂牊鈷戠紓浣诡焽閹冲嫰鏌ｉ悢鏉戝姎閻撱倝鏌ㄩ弴鐐测偓褰掑磻閸岀偞鐓涢柛銉㈡櫅閺嬪倸霉濠婂嫮鐭掗柡宀€鍠栧鍫曞垂椤曞懏娈洪梻浣筋嚃閸犳盯宕戦幇顔筋潟闁圭儤顨嗛崑鎰版煣韫囷絽浜滅痪鐐▕濮婃椽宕崟顔碱伃缂備礁顦紞濠傤嚕婵犳艾惟鐟滃宕戦幘缁樻櫜閹肩补鈧尙鐖遍梻浣告贡閺屽鐣烽悽鍨潟闁规儳鐡ㄦ刊鏉懨归悩宸剰闁圭晫鏁婚弻锝夊閳轰胶浠氶梺閫炲苯澧痪缁㈠弮瀵娊鏁冮崒娑氬弳濠电娀娼уΛ娆撳疮閹烘鐓熼柡鍐ㄥ€哥敮璺好归懖鈺佲枅闁哄本鐩鎾Ω閵夈儳顔夐梻浣侯焾椤戝棝鎮ч悩宸綎?-" + std::to_string(MAX_OCTREE_LAYER_VRPG - 1));

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