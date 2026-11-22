#include "./LightClustering.h"
#include <nvutils/timers.hpp>
#include <common/Application/Application.h>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <bit>
#include <nvvk/default_structs.hpp>
#include <nvgui/property_editor.hpp>
#include "../VRPG2.h"
#include "feature/PathTracing/shaderio.h"
#include <cstdio>
#include <cstdlib>

using namespace FzbRenderer;

LightClustering_VRPG::LightClustering_VRPG(pugi::xml_node& featureNode) {
	uint32_t lightVoxelResolution = 16;
	if (pugi::xml_node resolutionNode = featureNode.child("resolution"))
		lightVoxelResolution = std::stof(resolutionNode.attribute("value").value());
	VRPG2Renderer::setting.lightOctreeDepth = std::countr_zero(lightVoxelResolution);
}

void LightClustering_VRPG::init() {
	createSource();
	createDescriptorSetLayout();
	createDescriptorSet();
	createPipeline();
	compileAndCreateShaders();

#ifndef NDEBUG
	Feature::createGBuffer(true, false, 1);
	debugPushConstant.curLayer_E = int(VRPG2Renderer::setting.lightOctreeDepth);	// default to the leaf layer
	{
		nvutils::PrimitiveMesh primitive = FzbRenderer::MeshSet::createWireframe();
		FzbRenderer::MeshSet mesh = FzbRenderer::MeshSet("Wireframe", primitive);
		scene.addMeshSet(mesh);
		scene.createSceneInfoBuffer();
	}
#endif
}
void LightClustering_VRPG::clean() {
	Feature::clean();

	Application::allocator.destroyBuffer(clusteringGlobalInfoBuffer);

	Application::allocator.destroyBuffer(lightNodeBlockInfoBuffer);
	Application::allocator.destroyBuffer(hasDataLightNodeBlockIndexBuffer);


	VkDevice device = Application::app->getDevice();

	vkDestroyShaderEXT(device, computeShader_init, nullptr);
	vkDestroyShaderEXT(device, computeShader_GetHasDataLightNodeBlockInfo, nullptr);
	vkDestroyShaderEXT(device, computeShader_DispatchClustering, nullptr);
	vkDestroyShaderEXT(device, computeShader_Clustering1, nullptr);
	vkDestroyShaderEXT(device, computeShader_Clustering2, nullptr);
#ifndef NDEBUG
	vkDestroyShaderEXT(device, vertexShader_TreeDebug, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_TreeDebug, nullptr);
#endif
}
void LightClustering_VRPG::uiRender() {
#ifndef NDEBUG
	namespace PE = nvgui::PropertyEditor;
	if (ImGui::CollapsingHeader(std::string("Light Clustering ").c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Indent(20.0f);

		std::vector<std::string> normalIndexNames = { "left", "right", "bottom", "up", "back", "forward", "All" };
		std::vector<const char*> normalIndexNames_pointers;
		for (const auto& normalIndexName : normalIndexNames)
			normalIndexNames_pointers.push_back(normalIndexName.c_str());
		ImGui::Combo("normal_E", &debugPushConstant.normalIndex_E, normalIndexNames_pointers.data(), static_cast<int>(normalIndexNames_pointers.size()));

		int maxLayer = int(VRPG2Renderer::setting.lightOctreeDepth);
		std::vector<std::string> layerMapNames(maxLayer + 1);
		for (int i = 0; i <= maxLayer; ++i) layerMapNames[i] = "Layer" + std::to_string(i);
		std::vector<const char*> layerMapNames_pointers;
		for (const auto& layerMapName : layerMapNames)
			layerMapNames_pointers.push_back(layerMapName.c_str());
		ImGui::Combo("Light Tree Layer", &debugPushConstant.curLayer_E, layerMapNames_pointers.data(), static_cast<int>(layerMapNames_pointers.size()));

		ImGui::Checkbox("Show Light Tree", &showLightTree);

		ImGui::Unindent(20.0f);
	}

	if (showLightTree) Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)shaderio::GBuffers_LightClustering_VRPG::eLightTree);
#endif
}
void LightClustering_VRPG::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
#ifndef NDEBUG
	NVVK_CHECK(gBuffers.update(cmd, size));
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
#endif
};
void LightClustering_VRPG::preRender() {}
void LightClustering_VRPG::updateDataPerFrame(VkCommandBuffer cmd) {
	pushConstant.frameIndex = VRPG2Renderer::setting.frameIndex;
	pushConstant.randomFrameIndex = (VRPG_RANDOM_FRAME_SOURCE != 0) ? int(VRPG2Renderer::setting.randomFrameIndex) : VRPG2Renderer::setting.frameIndex;

	pushConstant.lightOctreeMaxLayer = VRPG2Renderer::setting.lightOctreeDepth;
	pushConstant.lightOctreeNodeTotalCount = ((1u << (3 * (VRPG2Renderer::setting.lightOctreeDepth + 1))) - 1) / 7 * 6;
	pushConstant.lightLeafNodeTotalCount = (1 << (3 * VRPG2Renderer::setting.lightOctreeDepth)) * 6;

	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
}
void LightClustering_VRPG::render(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd, "GeometryClustering_render");

	updateDataPerFrame(cmd);

	pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::LightClusteringPushConstant_VRPG),
		.pValues = &pushConstant,
	};

	// the light clustering shaders never trace rays, so only set 0 is needed
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdPushConstants2(cmd, &pushInfo);

	createLightNodeOctree(cmd);
}
void LightClustering_VRPG::postProcess(VkCommandBuffer cmd) {
#ifndef NDEBUG
	if (!showLightTree) return;

	debugPushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::LightClusteringDebugPushConstant_VRPG),
		.pValues = &debugPushConstant,
	};

	debugPushConstant.frameIndex = pushConstant.frameIndex;
	debugPushConstant.sceneInfoAddress = pushConstant.sceneInfoAddress;
	debugPushConstant.maxLayer = int(VRPG2Renderer::setting.lightOctreeDepth);
	debugPushConstant.curLayerNodeCount_E = 1 << (3 * debugPushConstant.curLayer_E);

	lightTreeDebug(cmd);
#endif
};

void LightClustering_VRPG::createSource() {
	int octreeMaxLayer = VRPG2Renderer::setting.lightOctreeDepth;
	if (octreeMaxLayer >= MAX_OCTREE_LAYER_VRPG2)
		throw std::runtime_error("闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ょ紓宥咃躬瀵鏁愭径濠勵吅闂佹寧绻傞幉娑㈠箻缂佹鍘遍梺闈涚墕閹冲酣顢旈銏＄厸閻忕偛澧藉ú瀛樸亜閵忊剝绀嬮柡浣瑰姍瀹曞崬鈻庡▎鎴犫敍闂傚倸鍊风欢姘跺焵椤掑倸浠滈柤娲诲灡閺呭爼宕滆绾惧ジ鏌ｅΟ鎸庣彧閻忓浚鍙冮弻锝夋晲閸パ冨箣闂佽鍠掗崜婵嬪箚閺冨牜鏁婇柟顖嗗嫭姣勬繝鐢靛Х閺佸憡绻涢埀顒佺箾娴ｅ啿娲﹂崑瀣攽閻樺弶鎼愰柛姘秺閺屽秷顧侀柛鎾寸懇閳ユ棃宕橀鍢壯囧箹缁厜鍋撻懠顒€鍤紓鍌氬€峰ù鍥敋瑜忛埀顒佺煯閸楁娊鎮伴鈧獮瀣晝閳ь剛绮堢€ｎ偁浜滈柡宥冨妿閳洟鏌涙惔顔婚偗婵﹨娅ｉ幏顐ｆ償閳ヨ櫕娈查梺瑙勫絻閹诧繝銆冮妷鈺傚€锋い蹇撳娴煎牆鈹戦纭峰伐妞ゎ厼鍢查悾鐑藉箳閹存梹鐎婚梺瑙勬儗閸樺€熲叺闂傚倸鍊峰ù鍥綖婢跺瞼鐭堥柟缁㈠枛閻掑灚銇勯幒鎴濇殶鐞氾箑鈹戦悙鍙夆枙濞存粍绻堝鎻掆攽鐎ｎ偆鍘遍柣蹇曞仧閸嬫捇鎯冮幋锔界叆婵炴垶鐟уú瀛樻叏婵犲啯銇濈€规洜鍏橀、姗€鎮欓幓鎺濈€卞┑锛勫亼閸婃垿宕濆畝鍕疇婵☆垵娅ｉ弳锕傛煏婢跺牆濡稿ù婊冪秺閺岀喖鎮滃Ο璇叉殫闂佸憡蓱閹倸顫忓ú顏勭闁绘劖褰冩慨宀勬⒑缁嬪潡鍙勫ù婊冪埣楠炲啫顫滈埀顒勩€侀弮鍫濋唶闁绘柨鎼獮妤呮⒒娴ｇ瓔娼愰柛搴㈠▕閹椽濡歌閻棝鏌涢幇闈涙灍闁绘挸绻橀弻娑㈠Ψ閹存繂鏋ゅù鐓庡暙铻栭柣姗€娼ф禒婊勭箾瀹割喖骞栭柣锝呭槻椤劑宕熼鈧鎾绘⒑闂傚鐭欓柡鈧柆宥呭瀭闁革富鍘介～鏇㈡煙閹呮憼濠殿垱鎸抽弻鐔虹磼濡櫣鐟愭繝銏ｎ潐濞茬喎顫忕紒妯肩懝闁逞屽墴閸┾偓妞ゆ帒鍊告禒婊堟煠濞茶鐏￠柡鍛閳ь剚绋掕摫闁告瑥绻愰埞鎴︽偐閹绘帗娈查梺闈涙处缁诲嫰鍩€椤掑喚娼愭繛鎻掔箻瀹曟繂顓奸崶銊ュ簥濠电娀娼ч鍛閸忓吋鍙忔俊顖氭惈閹垹绱撳鍛偓鍨潖婵犳艾纾兼繛鍡樺笒閸樷€愁渻閵堝骸骞栭柣妤佹崌閵嗕線寮介鐐茶€垮┑鐐村灦椤洭顢欐径鎰拺闁硅偐鍋涢崝鈧柣蹇曞仜婢х晫绮欐担铏圭＝闁稿本鑹鹃埀顒傚厴閹偤鏁冩担瑙勫櫡婵犵數濮甸鏍垂闁秴绠伴柟鎯版閽冪喖鏌ㄥ┑鍡╂Ц缂佺媴缍侀弻锝堢疀閺冣偓閵囩喎霉濠婂嫮鐭掗柛鈹垮灲楠炴﹢寮甸崹顐㈠缂傚倷鐒︾粙鎴︻敄閸℃瑢鍋撻棃娑氱劯婵﹥妞藉Λ鍐ㄢ槈濮橆剦鏆繝纰樻閸嬪懘鎮烽敃鈧銉╁礋椤栨氨顦板銈嗙墬缁嬫捇宕㈤崡鐐╂斀闁绘劕寮堕ˉ婊呯磼缂佹ê濮嶇€殿喖鍟块埢搴ㄥ箻鐎电骞楅梻浣瑰缁诲倸煤閵堝鍌ㄩ悗娑欙供濞堜粙鏌ｉ幇顒佲枙闁稿孩妫冮弻鈩冩媴缁嬪簱鍋撻崸妤€绠板┑鐘插暙缁剁偤鏌涢埄鍐︿沪濠㈣娲熷铏圭磼濡櫣浼囨繝娈垮枔閸婃繂鐣烽幇鐗堝仺闁告稑艌閹峰姊洪崜鎻掍簽闁哥姵鎹囬崺銉﹀緞鐎ｃ劋绨婚梺闈涱槶閸庡磭绮婚懠顒傜＜缂備焦顭囧ú瀵糕偓娈垮枛閻栧ジ鐛€ｎ亖鏀介柟閭︿簼閸嬪懘姊婚崒娆戭槮闁汇倕娲敐鐐村緞閹邦剙鐎梺绉嗗嫷娈旈柡鍕╁劦閺屾洘寰勯崱妯荤彅濡ょ姷鍋戦崹鐑樼┍婵犲洤围闁告侗鍘藉▓銊╂⒑閻熼偊鍤熼柛蹇斆～蹇撁洪鍛闂侀潧鐗嗛幉娑樷槈閵忥紕鍘遍柣搴秵閸嬪懎鐣峰畝鍕厸閻忕偠顕ф慨鍌溾偓娈垮枟閹告娊骞冨▎寰濆湱鈧綆浜欐竟鏇㈡偡濠婂懎顣奸悽顖涘笧婢规洘绻濆顓犲幈闂佸搫娲㈤崝宀勬倶閿熺姵鐓熼柟鎯х摠缁€瀣煛鐏炵晫效闁糕斁鍓濋幏鍛村川婵犲喚鍟€闂傚倷鐒﹂崜姘跺垂閸楃伝娲偄閻撳氦鎽曢梺鎸庣箓椤︻垳绮绘繝姘€垫繛鎴炵懐閻掕姤銇勯敂鍝勫婵﹥妞藉Λ鍐ㄢ槈濞嗘劖鍊烽梺璇插閻噣宕￠幎鑺ュ仒妞ゆ洍鍋撶€规洖鐖奸、妤呭焵椤掑嫭鍋傛繛鎴炩棨瑜版帗鏅查柛顐ゅ枂閳ь剙鍟撮弻锝夊Ψ椤旇崵鍑规繛锝呮搐閿曨亪骞冨▎鎿冩晜闁告洏鍔屾禍楣冩煙鏉堝墽鐣辩痪鎯ф健閺屻倗鍠婇崡鐐差潻闂佹悶鍔嶇换鍫ュ蓟閻旂厧绠查柟浼存涧濞堫厾绱撴担鍝勑ｉ柣妤侇殕缁岃鲸绻濋崶顬囨煕鐏炲墽鐓瑙勬礃娣囧﹪鎮欓鍕ㄥ亾閵堝纾婚柛娑卞灡瀹曟煡鏌涢鐘插姎缂佺姵鐗犻弻宥夊煛娴ｅ憡娈查梺绋挎捣閸犳牠寮婚弴锛勭杸闁挎繂妫涚粣鏃€绻濋姀锝庢綈婵炲弶绮庡Σ鎰板箻鐠囪尙锛滃┑顔斤供閸忔﹢宕戦幘鎼Ч閹兼番鍩勯崑銊╂⒑鐠恒劌鏋斿┑顔碱嚟缁絽螖閸涱喚鍘搁悗骞垮劚妤犳悂鐛幇鐗堢厓鐟滄粓宕滈妸褏涓嶉柟鎹愵嚙缁犳牗绻濇繝鍌滃闁稿鏅犻幃姗€鎮欓崹顐ｇ彧闂侀潧鐗炵粻鎾愁潖缂佹ɑ濯撮柛娑橈龚绾偓闂備胶顭堢花娲磹濠靛棛鏆﹂柣銏犳啞閺呮繈鏌涚仦鐐殤闁活偄瀚板娲礈閹绘帊绨肩紓浣筋嚙鐎氭澘鐣峰┑瀣櫇闁逞屽墴閳ユ棃宕橀鍢壯囨煕閳╁喚娈橀柣鐔村姂濮婅櫣绮欓崠鈥冲缂備緡鍣崹宕囧垝濞嗘劕绶炲┑鐘查閹垿姊洪幖鐐插姌闁告柨鐭傚鎼佸箣閿旇В鎷虹紓鍌欑劍閿曗晛鈻撻弮鍫熺厽婵°倐鍋撴俊顐ｇ〒閸掓帡顢橀姀鐘殿唺闂佸搫鍟崐濠氭晬濠婂牊鈷戠紓浣诡焽閹冲嫰鏌ｉ悢鏉戝姎閻撱倝鏌ㄩ弴鐐测偓褰掑磻閸岀偞鐓涢柛銉㈡櫅閺嬪倸霉濠婂嫮鐭掗柡宀€鍠栧鍫曞垂椤曞懏娈洪梻浣筋嚃閸犳盯宕戦幇顔筋潟闁圭儤顨嗛崑鎰版煣韫囷絽浜滅痪鐐▕濮婃椽宕崟顔碱伃缂備礁顦紞濠傤嚕婵犳艾惟鐟滃宕戦幘缁樻櫜閹肩补鈧尙鐖遍梻浣告贡閺屽鐣烽悽鍨潟闁规儳鐡ㄦ刊鏉懨归悩宸剰闁圭晫鏁婚弻锝夊閳轰胶浠氶梺閫炲苯澧痪缁㈠弮瀵娊鏁冮崒娑氬弳濠电娀娼уΛ娆撳疮閹烘鐓熼柡鍐ㄥ€哥敮璺好归懖鈺佲枅闁哄本鐩鎾Ω閵夈儳顔夐梻浣侯焾椤戝棝鎮ч悩宸�?-" + std::to_string(MAX_OCTREE_LAYER_VRPG2 - 1));

	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();
	uint32_t bufferSize = 0;
	//-----------------------------------------------------------Octree------------------------------------------------------------------------------
	VRPG2Renderer::source.lightNodeDataBuffer_Octree.resize(octreeMaxLayer + 1);
	uint32_t layerNodeCount = 6;
	for (int layerIndex = 0; layerIndex <= octreeMaxLayer; ++layerIndex) {
		bufferSize = layerNodeCount * sizeof(shaderio::LightNodeData_Octree_VRPG);
		allocator->createBuffer(VRPG2Renderer::source.lightNodeDataBuffer_Octree[layerIndex], bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(VRPG2Renderer::source.lightNodeDataBuffer_Octree[layerIndex].buffer);

		layerNodeCount *= 8;
	}
	//-----------------------------------------------------------Global------------------------------------------------------------------------------
	{
		bufferSize = sizeof(shaderio::LightClusteringGlobalInfo_VRPG);
		allocator->createBuffer(clusteringGlobalInfoBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT);
		NVVK_DBG_NAME(clusteringGlobalInfoBuffer.buffer);
	}
	//------------------------------------------------------------CreateOctree-----------------------------------------------------------------------------
	{
		bufferSize = uint32_t(pow(8, octreeMaxLayer)) * 6 / 8 * sizeof(shaderio::LightNodeBlockInfo);
		allocator->createBuffer(lightNodeBlockInfoBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(lightNodeBlockInfoBuffer.buffer);

		bufferSize = int(pow(8, octreeMaxLayer)) * 6 / 8 * sizeof(shaderio::LightNodeBlockIndex);
		allocator->createBuffer(hasDataLightNodeBlockIndexBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(hasDataLightNodeBlockIndexBuffer.buffer);
	}
}
void LightClustering_VRPG::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;

	{
		bindings.addBinding({
			.binding = (uint32_t)shaderio::LightClusteringBindingPoints_VRPG::eGlobalInfo,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::LightClusteringBindingPoints_VRPG::eClusteringGlobalInfo,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
	}
	{
		bindings.addBinding({
			.binding = (uint32_t)shaderio::LightClusteringBindingPoints_VRPG::eLightNodeData_Octree,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = (uint32_t)VRPG2Renderer::source.lightNodeDataBuffer_Octree.size(),
			.stageFlags = VK_SHADER_STAGE_ALL });
	}
	{
		bindings.addBinding({
			.binding = (uint32_t)shaderio::LightClusteringBindingPoints_VRPG::eLightNodeBlockInfos,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::LightClusteringBindingPoints_VRPG::eHasDataLightNodeBlockIndices,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
	}

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	NVVK_DBG_NAME(staticDescPack.getLayout());
	NVVK_DBG_NAME(staticDescPack.getPool());
	NVVK_DBG_NAME(staticDescPack.getSet(0));
}
void LightClustering_VRPG::createDescriptorSet() {
	nvvk::WriteSetContainer write{};
	{
		VkWriteDescriptorSet    GlobalInfoWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::LightClusteringBindingPoints_VRPG::eGlobalInfo, 0, 0, 1);
		write.append(GlobalInfoWrite, VRPG2Renderer::source.globalInfoBuffer, 0, VRPG2Renderer::source.globalInfoBuffer.bufferSize);

		GlobalInfoWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::LightClusteringBindingPoints_VRPG::eClusteringGlobalInfo, 0, 0, 1);
		write.append(GlobalInfoWrite, clusteringGlobalInfoBuffer, 0, clusteringGlobalInfoBuffer.bufferSize);
	}
	{
		VkWriteDescriptorSet    OctreeWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::LightClusteringBindingPoints_VRPG::eLightNodeData_Octree, 0, 0, VRPG2Renderer::source.lightNodeDataBuffer_Octree.size());
		nvvk::Buffer* octreeArraysPtr = VRPG2Renderer::source.lightNodeDataBuffer_Octree.data();
		write.append(OctreeWrite, octreeArraysPtr);
	}
	{
		VkWriteDescriptorSet    createOctreeWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::LightClusteringBindingPoints_VRPG::eLightNodeBlockInfos, 0, 0, 1);
		write.append(createOctreeWrite, lightNodeBlockInfoBuffer, 0, lightNodeBlockInfoBuffer.bufferSize);

		createOctreeWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::LightClusteringBindingPoints_VRPG::eHasDataLightNodeBlockIndices, 0, 0, 1);
		write.append(createOctreeWrite, hasDataLightNodeBlockIndexBuffer, 0, hasDataLightNodeBlockIndexBuffer.bufferSize);
	}

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void LightClustering_VRPG::createPipeline() {
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = 256
	};

	std::array<VkDescriptorSetLayout, 1> layouts = { {staticDescPack.getLayout()} };
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
void LightClustering_VRPG::compileAndCreateShaders() {
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

	std::array<VkDescriptorSetLayout, 1> layouts = { {staticDescPack.getLayout()} };
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
		shaderSource = shaderPath / "init.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_init, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_init";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_init);
		NVVK_DBG_NAME(computeShader_init);
	}
	//--------------------------------------------------------------------------------------
	{
		shaderSource = shaderPath / "clustering.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_GetHasDataLightNodeBlockInfo, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_GetHasDataLightNodeBlockInfo";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_GetHasDataLightNodeBlockInfo);
		NVVK_DBG_NAME(computeShader_GetHasDataLightNodeBlockInfo);

		vkDestroyShaderEXT(device, computeShader_DispatchClustering, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_DispatchClustering";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_DispatchClustering);
		NVVK_DBG_NAME(computeShader_DispatchClustering);

		vkDestroyShaderEXT(device, computeShader_Clustering1, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_Clustering1";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_Clustering1);
		NVVK_DBG_NAME(computeShader_Clustering1);

		vkDestroyShaderEXT(device, computeShader_Clustering2, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_Clustering2";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_Clustering2);
		NVVK_DBG_NAME(computeShader_Clustering2);
	}
	//----------------------------------------debug----------------------------------------
#ifndef NDEBUG
	{
		shaderSource = shaderPath / "debug.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, vertexShader_TreeDebug, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.pName = "vertexMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &vertexShader_TreeDebug);
		NVVK_DBG_NAME(vertexShader_TreeDebug);

		vkDestroyShaderEXT(device, fragmentShader_TreeDebug, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "fragmentMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &fragmentShader_TreeDebug);
		NVVK_DBG_NAME(fragmentShader_TreeDebug);
	}
#endif
}

void LightClustering_VRPG::createLightNodeOctree(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;

	int iterationCount = std::min(VRPG2Renderer::setting.lightOctreeDepth, 3);
	uint32_t layerBlockCount = uint32_t(pow(8, VRPG2Renderer::setting.lightOctreeDepth)) * 6 / 8;

	{
		vkCmdPushConstants2(cmd, &pushInfo);

		vkCmdFillBuffer(cmd, lightNodeBlockInfoBuffer.buffer, 0, lightNodeBlockInfoBuffer.bufferSize, 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_init);
		VkExtent2D leafGroupSize = nvvk::getGroupCounts({ (uint32_t)pushConstant.lightOctreeNodeTotalCount, 1 }, VkExtent2D{ 1024, 1 });
		vkCmdDispatch(cmd, leafGroupSize.width, 1, 1);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
	}

	for (int layerIndex = VRPG2Renderer::setting.lightOctreeDepth; layerIndex > iterationCount; --layerIndex) {
		pushConstant.currentLayer = layerIndex;
		pushConstant.currentLayerBlockCount = layerBlockCount;
		vkCmdPushConstants2(cmd, &pushInfo);

		vkCmdFillBuffer(cmd, clusteringGlobalInfoBuffer.buffer, 0, sizeof(shaderio::LightClusteringGlobalInfo_VRPG), 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_GetHasDataLightNodeBlockInfo);
		VkExtent2D groupSize = nvvk::getGroupCounts({ layerBlockCount, 1 }, VkExtent2D{ 1024, 1 });
		vkCmdDispatch(cmd, groupSize.width, 1, 1);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_DispatchClustering);
		vkCmdDispatch(cmd, 1, 1, 1);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_Clustering1);
		vkCmdDispatchIndirect(cmd, VRPG2Renderer::source.globalInfoBuffer.buffer, 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);

		layerBlockCount /= 8;
	}

	// aggregate the remaining layers one by one (layer 3 -> 0)
	for (int layer = iterationCount; layer > 0; --layer) {
		uint32_t layerNodeCount = shaderio::OctreeLayerNodeCount_VRPG2[layer];
		pushConstant.currentLayer = layer;
		pushConstant.currentLayerNodeCount = layerNodeCount;
		vkCmdPushConstants2(cmd, &pushInfo);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_Clustering2);
		VkExtent2D groupSize = nvvk::getGroupCounts({ layerNodeCount, 1 }, VkExtent2D{ 384, 1 });
		vkCmdDispatch(cmd, groupSize.width, 1, 1);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}
}

void LightClustering_VRPG::lightTreeDebug(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage((uint32_t)shaderio::GBuffers_LightClustering_VRPG::eLightTree), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView((uint32_t)shaderio::GBuffers_LightClustering_VRPG::eLightTree);
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
	graphicsDynamicPipeline.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_FALSE;
	graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_FALSE;
	graphicsDynamicPipeline.depthStencilState.depthBoundsTestEnable = VK_FALSE;
	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, gBuffers.getSize());
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_TreeDebug, .fragment = fragmentShader_TreeDebug });

	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	const shaderio::Mesh& mesh = scene.meshes[0];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	uint32_t bufferIndex = scene.getMeshBufferIndex(0);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];
	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

	// normalIndex == 6 means "all six normal partitions"
	const int selectedNormal = debugPushConstant.normalIndex_E;
	const int firstNormal = selectedNormal == 6 ? 0 : selectedNormal;
	const int lastNormal = selectedNormal == 6 ? 6 : selectedNormal + 1;

	for (int normalIndex = firstNormal; normalIndex < lastNormal; ++normalIndex) {
		debugPushConstant.normalIndex_E = normalIndex;
		debugPushConstant.mode = 1;
		vkCmdPushConstants2(cmd, &debugPushInfo);
		vkCmdDrawIndexed(cmd, triMesh.indices.count, (uint32_t)debugPushConstant.curLayerNodeCount_E, 0, 0, 0);
	}
	debugPushConstant.normalIndex_E = selectedNormal;

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage((uint32_t)shaderio::GBuffers_LightClustering_VRPG::eLightTree), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}

