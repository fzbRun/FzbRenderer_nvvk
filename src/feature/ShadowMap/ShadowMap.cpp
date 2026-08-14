#include "./ShadowMap.h"
#include <common/Application/Application.h>
#include <nvvk/formats.hpp>
#include <common/Image/Image.h>
#include <common/Shader/Shader.h>
#include <nvvk/default_structs.hpp>
#include <nvgui/property_editor.hpp>
#include <nvvk/compute_pipeline.hpp>

using namespace FzbRenderer;

ShadowMap::ShadowMap() {
	//Application::vkContextInitInfo.deviceExtensions.push_back({ VK_NV_VIEWPORT_ARRAY2_EXTENSION_NAME });
	//Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_MULTIVIEW_EXTENSION_NAME });
	//Application::vkContextInitInfo.deviceExtensions.push_back({ VK_NV_VIEWPORT_SWIZZLE_EXTENSION_NAME });
}

void ShadowMap::init(ShadowMapCreateInfo createInfo) {
	this->setting = createInfo;

	createShadowMap();
	if (shadowMaps.size() == 0) return;
#ifndef NDEBUG
	Feature::createGBuffer(false, false, shadowMaps.size(), setting.resolution);
#endif
	createPipeline();
	compileAndCreateShaders();

	{
		sceneAABB.minimum = { FLT_MAX, FLT_MAX, FLT_MAX };
		sceneAABB.maximum = -sceneAABB.minimum;
		FzbRenderer::Scene& sceneResource = Application::sceneResource;
		for (int i = 0; i < sceneResource.instances.size(); ++i) {
			uint32_t meshIndex = sceneResource.instances[i].meshIndex;
			MeshInfo meshInfo = sceneResource.getMeshInfo(meshIndex);
			shaderio::AABB meshAABB = meshInfo.getAABB(sceneResource.instances[i].transform);	//对于动态物体，这里需要修改

			sceneAABB.minimum.x = std::min(meshAABB.minimum.x, sceneAABB.minimum.x);
			sceneAABB.minimum.y = std::min(meshAABB.minimum.y, sceneAABB.minimum.y);
			sceneAABB.minimum.z = std::min(meshAABB.minimum.z, sceneAABB.minimum.z);
			sceneAABB.maximum.x = std::max(meshAABB.maximum.x, sceneAABB.maximum.x);
			sceneAABB.maximum.y = std::max(meshAABB.maximum.y, sceneAABB.maximum.y);
			sceneAABB.maximum.z = std::max(meshAABB.maximum.z, sceneAABB.maximum.z);
		}
	}
}
void ShadowMap::clean() {
	Feature::clean();
	for (int i = 0; i < shadowMaps.size(); ++i) shadowMaps[i].clean();

	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, vertexShader_directionLight, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_directionLight, nullptr);

	vkDestroyShaderEXT(device, vertexShader_pointLight, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_pointLight, nullptr);

#ifndef NDEBUG
	vkDestroyShaderEXT(device, computeShader_debug, nullptr);
#endif
}
void ShadowMap::uiRender() {
	return;
#ifndef NDEBUG
	uint32_t mapCount = shadowMaps.size();
	if (mapCount == 0) return;
	bool& UIModified = Application::UIModified;

	std::vector<std::string> shadowMapNames(mapCount);
	for (int i = 0; i < mapCount; ++i) shadowMapNames[i] = "shadowMap " + std::to_string(i);

	std::vector<const char*> shadowMapNames_pointers;
	for (const auto& shadowMapName : shadowMapNames)
		shadowMapNames_pointers.push_back(shadowMapName.c_str());

	int selectedShadowMapIndex = 0;

	namespace PE = nvgui::PropertyEditor;
	if (ImGui::Begin("ShadowMap")) {
		ImGui::Combo("ShadowMap Index", &selectedShadowMapIndex, shadowMapNames_pointers.data(), static_cast<int>(shadowMapNames_pointers.size()));
		if (PE::begin()) {
			if (PE::entry("ShadowMap Result", [&] {
				static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
				ImVec4 selectedColor = showShadowMap ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
				ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
				ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));
	
				bool result = ImGui::ImageButton("##but", (ImTextureID)shadowMaps[selectedShadowMapIndex].uiDescriptorSet,
					ImVec2(100, 100));		//����Ӧ����һ��������
	
				ImGui::PopStyleColor(2);
				ImGui::PopStyleVar();
				return result;
				}))
			{
				showShadowMap = !showShadowMap;
				showRestructResultMap = false;
			}
		}
		PE::end();

		//if (PE::begin()) {
		//	if (PE::entry("Depth Restruct result", [&] {
		//		static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
		//		ImVec4 selectedColor = showRestructResultMap ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
		//		ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
		//		ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
		//		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
		//		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));
		//
		//		bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet(selectedShadowMapIndex),
		//			ImVec2(100 * gBuffers.getAspectRatio(), 100));
		//
		//		ImGui::PopStyleColor(2);
		//		ImGui::PopStyleVar();
		//		return result;
		//		}))
		//	{
		//		showRestructResultMap = !showRestructResultMap;
		//		showShadowMap = false;
		//	}
		//}
		//PE::end();
	}
	ImGui::End();

	if (showShadowMap) Application::viewportImage = shadowMaps[selectedShadowMapIndex].uiDescriptorSet;
	if (showRestructResultMap) Application::viewportImage = gBuffers.getDescriptorSet(selectedShadowMapIndex);
#endif
};
void ShadowMap::resize(VkCommandBuffer cmd, const VkExtent2D& size) {}
void ShadowMap::preRender(){}
void ShadowMap::render(VkCommandBuffer cmd) {
	if (shadowMaps.size() == 0) return;

	NVVK_DBG_SCOPE(cmd, "ShadowMap_render");

	pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::ShadowMapPushConstant),
		.pValues = &pushConstant,
	};
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;

	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	//graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_FRONT_BIT;
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
	graphicsDynamicPipeline.rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_TRUE;
	graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_TRUE;
	graphicsDynamicPipeline.depthStencilState.stencilTestEnable = VK_FALSE;

	for (int i = 0; i < shadowMaps.size(); ++i) {
		VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };
		depthAttachment.imageView = shadowMaps[i].image.descriptor.imageView;

		VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
		renderingInfo.renderArea = { {0, 0}, setting.resolution };
		renderingInfo.colorAttachmentCount = 0;
		renderingInfo.pColorAttachments = nullptr;
		renderingInfo.pDepthAttachment = &depthAttachment;

		shaderio::Light& light = Application::sceneResource.sceneInfo.lights[lightIndices[i]];
		VkShaderEXT vertexShader{}, fragmentShader{};
		if (light.type == shaderio::LightType::Direction) {
			//shaderio::float4x4 viewMatrix = glm::lookAt(light.pos, light.pos + light.direction, glm::vec3(0.0f, 1.0f, 0.0f));
			//float near_plane = 0.1f, far_plane = 20.0f;
			//glm::mat4 orthoMatrix = glm::orthoRH_ZO(-10.0f, 10.0f, -10.0f, 10.0f, near_plane, far_plane);
			//orthoMatrix[1][1] *= -1;
			//pushConstant.lightVP = orthoMatrix * viewMatrix;

			shaderio::float3 sceneSize = sceneAABB.maximum - sceneAABB.minimum;
			shaderio::float3 sceneStartPos = sceneAABB.minimum;

			glm::vec3 distance = sceneSize * 1.1f;
			glm::vec3 center = (sceneAABB.maximum + sceneAABB.minimum) * 0.5f;
			glm::vec3 minimum = center - distance * 0.5f;
			glm::vec3 maximum = center + distance * 0.5f;
			float sceneLength = glm::length(distance);
			shaderio::float3 lightPos = center - light.direction * sceneLength;

			glm::vec3 up = fabs(light.direction.y) > 0.99f ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);
			glm::mat4 viewMatrix = glm::lookAt(lightPos, lightPos + light.direction, up);

			std::vector<glm::vec3> corners = {
				glm::vec3(minimum.x, minimum.y, minimum.z),
				glm::vec3(maximum.x, minimum.y, minimum.z),
				glm::vec3(minimum.x, maximum.y, minimum.z),
				glm::vec3(minimum.x, minimum.y, maximum.z),
				glm::vec3(maximum.x, maximum.y, minimum.z),
				glm::vec3(maximum.x, minimum.y, maximum.z),
				glm::vec3(minimum.x, maximum.y, maximum.z),
				glm::vec3(maximum.x, maximum.y, maximum.z)
			};

			glm::vec3 minView(FLT_MAX);
			glm::vec3 maxView(-FLT_MAX);
			for (auto& v : corners) {
				glm::vec4 p = viewMatrix * glm::vec4(v, 1.0f);
				glm::vec3 p3 = glm::vec3(p) / p.w;
				minView = glm::min(minView, p3);
				maxView = glm::max(maxView, p3);
			}
			{
				float texelSizeX = (maxView.x - minView.x) / setting.resolution.width;
				float texelSizeY = (maxView.y - minView.y) / setting.resolution.height;

				minView.x = floor(minView.x / texelSizeX) * texelSizeX;
				maxView.x = floor(maxView.x / texelSizeX) * texelSizeX;

				minView.y = floor(minView.y / texelSizeY) * texelSizeY;
				maxView.y = floor(maxView.y / texelSizeY) * texelSizeY;
			}

			glm::mat4 orthoMatrix = glm::orthoRH_ZO(minView.x, maxView.x, minView.y, maxView.y, -maxView.z, -minView.z);
			orthoMatrix[1][1] *= -1;
			pushConstant.lightVP = orthoMatrix * viewMatrix;

			graphicsDynamicPipeline.rasterizationState.depthBiasEnable = VK_TRUE;
			graphicsDynamicPipeline.rasterizationState.depthBiasConstantFactor = 1.5f;
			graphicsDynamicPipeline.rasterizationState.depthBiasSlopeFactor = 1.0f;
			graphicsDynamicPipeline.rasterizationState.depthBiasClamp = 0.0f;
			graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, setting.resolution);

			//VkViewportSwizzleNV swizzle = {
			//	VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_X_NV,  // r
			//	VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Y_NV,  // g
			//	VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Z_NV,  // b
			//	VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV   // a
			//};
			//vkCmdSetViewportSwizzleNV(cmd, 0, 1, &swizzle);

			vertexShader = vertexShader_directionLight;
			fragmentShader = fragmentShader_directionLight;

			nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(i), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL });
			nvvk::cmdImageMemoryBarrier(cmd, { shadowMaps[i].image.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
				{ VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
		}
		else if (light.type == shaderio::LightType::Point) {
			shaderio::float4x4 viewMatrix = glm::lookAt(light.pos, light.pos + shaderio::float3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
			float near_plane = 0.1f, far_plane = 20.0f;
			glm::mat4 projMatrix = glm::perspectiveRH_ZO(glm::radians(90.0f), 1.0f, far_plane, near_plane);
			//projMatrix[1][1] *= -1;
			pushConstant.lightVP = projMatrix * viewMatrix;

			VkViewport viewports[6];
			VkRect2D scissors[6];
			for (int j = 0; j < 6; j++){
				viewports[j].x = 0;
				viewports[j].y = 0;
				viewports[j].width = setting.resolution.width;
				viewports[j].height = setting.resolution.height;
				viewports[j].minDepth = 0.0f;
				viewports[j].maxDepth = 1.0f;

				scissors[j].offset = { 0, 0 };
				scissors[j].extent = setting.resolution;
			}
			vkCmdSetViewport(cmd, 0, 6, viewports);
			vkCmdSetScissor(cmd, 0, 6, scissors);

			VkViewportSwizzleNV swizzles[6] = {
				{ VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Z_NV, VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Y_NV,
				  VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV, VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_X_NV }, // +X
				{ VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_Z_NV, VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Y_NV,
				  VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV, VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_X_NV }, // -X
				{ VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_X_NV, VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Z_NV,
				  VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV, VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Y_NV }, // +Y
				{ VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_X_NV, VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_Z_NV,
				  VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV, VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_Y_NV }, // -Y
				{ VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_X_NV, VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_Y_NV,
				  VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV, VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Z_NV }, // +Z
				{ VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_X_NV, VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_Y_NV,
				  VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV, VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_Z_NV }  // -Z
			};
			vkCmdSetViewportSwizzleNV(cmd, 0, 6, swizzles);

			graphicsDynamicPipeline.rasterizationState.depthClampEnable = VK_TRUE;
			graphicsDynamicPipeline.depthStencilState.depthCompareOp = VK_COMPARE_OP_GREATER;

			depthAttachment.imageView = shadowMaps[i].imageViews[0];	//VK_IMAGE_VIEW_TYPE_2D_ARRAY
			depthAttachment.clearValue.depthStencil = { 0.0f, 0 };
			renderingInfo.layerCount = 6;
			renderingInfo.viewMask = 0x3F;

			nvvk::cmdImageMemoryBarrier(cmd, { shadowMaps[i].image.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
				{ VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, 6} });

			vertexShader = vertexShader_pointLight;
			fragmentShader = fragmentShader_pointLight;
		}

		vkCmdBeginRendering(cmd, &renderingInfo);

		vkCmdSetDepthBias(cmd, 1.5f, 0.0f, 1.0f);
		vkCmdSetDepthBiasEnable(cmd, VK_TRUE);
		graphicsDynamicPipeline.cmdApplyAllStates(cmd);
		graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader, .fragment = fragmentShader });

		VkVertexInputBindingDescription2EXT bindingDescription{};
		VkVertexInputAttributeDescription2EXT attributeDescription = {};
		vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

		pushConstant.lightIndex = lightIndices[i];
		for (size_t j = 0; j < Application::sceneResource.instances.size(); ++j)
		{
			uint32_t meshIndex = Application::sceneResource.instances[j].meshIndex;
			const shaderio::Mesh& mesh = Application::sceneResource.meshes[meshIndex];
			const shaderio::TriangleMesh& triMesh = mesh.triMesh;

			if (Application::sceneResource.materials[Application::sceneResource.instances[j].materialIndex].type == shaderio::MaterialType::RoughDielectric) continue;

			pushConstant.instanceIndex = int(j);
			vkCmdPushConstants2(cmd, &pushInfo);

			uint32_t bufferIndex = Application::sceneResource.getMeshBufferIndex(meshIndex);
			const nvvk::Buffer& v = Application::sceneResource.bDatas[bufferIndex];

			vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

			vkCmdDrawIndexed(cmd, triMesh.indices.count, 1, 0, 0, 0);
		}

		vkCmdEndRendering(cmd);

		if (light.type == shaderio::LightType::Point) {
			VkViewportSwizzleNV identitySwizzle = {
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_X_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Y_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Z_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV
			};
			vkCmdSetViewportSwizzleNV(cmd, 0, 1, &identitySwizzle);
		}

		nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(i), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
		nvvk::cmdImageMemoryBarrier(cmd, { shadowMaps[i].image.image, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
			{ VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
	}
}
void ShadowMap::postProcess(VkCommandBuffer cmd) {
	//debug_Visualization(cmd);
}

VkResult ShadowMap::createShadowMap() {
	int lightCount = 0; lightIndices.resize(0);
	for (int i = 0; i < Application::sceneResource.sceneInfo.numLights; ++i) {
		const shaderio::Light& light = Application::sceneResource.sceneInfo.lights[i];
		if (light.type == shaderio::LightType::Non) break;
		if (light.type == shaderio::LightType::Direction) {	// || light.type == shaderio::LightType::Point
			++lightCount;
			lightIndices.push_back(i);
		}
	}
	shadowMaps.resize(lightCount);
	if (lightCount == 0) return VK_SUCCESS;

	VkSamplerCreateInfo sampleCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_NEAREST,
		.minFilter = VK_FILTER_NEAREST,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.compareEnable = VK_TRUE,
		.compareOp = VK_COMPARE_OP_LESS,
		.maxLod = 0.0f,
		.unnormalizedCoordinates = VK_FALSE

	};

	for (int i = 0; i < lightCount; ++i) {
		sampleCreateInfo.compareOp = VK_COMPARE_OP_LESS;

		std::string shadowMapName = "";
		FzbRenderer::ImageCreateInfo colorImageCreateInfo = FzbRenderer::createDefaultImageCreateInfo();
		shaderio::LightType lightType = (shaderio::LightType)Application::sceneResource.sceneInfo.lights[lightIndices[i]].type;
		if (lightType == shaderio::LightType::Direction) {
			shadowMapName = "shadowMap_direction_" + std::to_string(i);
			
			colorImageCreateInfo.info.extent = { setting.resolution.width, setting.resolution.height, 1 };
			colorImageCreateInfo.info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT
				| VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

			colorImageCreateInfo.viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;

			colorImageCreateInfo.viewInfo.subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT, .levelCount = 1, .layerCount = 1 };
		}
		else if (lightType == shaderio::LightType::Point) {
			sampleCreateInfo.compareOp = VK_COMPARE_OP_GREATER;

			shadowMapName = "shadowMap_Point_" + std::to_string(i);

			colorImageCreateInfo = FzbRenderer::createDefaultImageCreateInfo(2);

			colorImageCreateInfo.info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
			colorImageCreateInfo.info.arrayLayers = 6;

			colorImageCreateInfo.viewInfos[0].format = colorImageCreateInfo.info.format;
			colorImageCreateInfo.viewInfos[0].viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;	//用于渲染
			colorImageCreateInfo.viewInfos[0].subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 6 };
			colorImageCreateInfo.viewInfos[1].format = colorImageCreateInfo.info.format;
			colorImageCreateInfo.viewInfos[1].viewType = VK_IMAGE_VIEW_TYPE_CUBE;	//用于采样
			colorImageCreateInfo.viewInfos[1].subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 6 };

			colorImageCreateInfo.viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;		//用于采样
			colorImageCreateInfo.viewInfo.subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 6 };
		}
		shadowMaps[i] = FzbRenderer::Image(shadowMapName);
		
		colorImageCreateInfo.info.imageType = VK_IMAGE_TYPE_2D;
		colorImageCreateInfo.info.extent = { setting.resolution.width, setting.resolution.height, 1 };
		colorImageCreateInfo.info.format = VK_FORMAT_D32_SFLOAT;
		colorImageCreateInfo.info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT
			| VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

		colorImageCreateInfo.viewInfo.format = colorImageCreateInfo.info.format;

		colorImageCreateInfo.samplerInfo = sampleCreateInfo;

		shadowMaps[i].init(colorImageCreateInfo);
	}


	/*
	#ifndef NDEBUG
	uiImageViews.resize(lightCount);
#endif

		FzbRenderer::ImageCreateInfo createInfo = createDefaultImageCreateInfo();
	createInfo.info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
		| VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
	createInfo.info.format = nvvk::findDepthFormat(Application::app->getPhysicalDevice());
	createInfo.viewInfo.format = nvvk::findDepthFormat(Application::app->getPhysicalDevice());
	createInfo.viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	createInfo.samplerInfo = sampleCreateInfo;
	createInfo.info.extent = { setting.resolution.width, setting.resolution.height, 1 };
	for (int i = 0; i < lightCount; ++i) {
		const shaderio::Light& light = Application::sceneResource.sceneInfo.lights[lightIndices[i]];
		if (light.type == shaderio::LightType::Direction) {}
		else if (light.type == shaderio::LightType::Point) {
			throw std::runtime_error("��û��ʵ�ֵ��Դ��Ӱ");
		}
		NVVK_FAIL_RETURN(createImage(shadowMaps[i], createInfo));

		#ifndef NDEBUG
		{
			nvvk::DebugUtil& dutil = nvvk::DebugUtil::getInstance();
			dutil.setObjectName(shadowMaps[i].image, "ShadowMap" + std::to_string(i));
			dutil.setObjectName(shadowMaps[i].descriptor.imageView, "ShadowMap" + std::to_string(i));

			// UI Image color view
			VkImageViewCreateInfo uiViewInfo = createInfo.viewInfo;
			uiViewInfo.image = shadowMaps[i].image;
			uiViewInfo.components.a = VK_COMPONENT_SWIZZLE_ONE;  // Forcing the VIEW to have a 1 in the alpha channel
			NVVK_FAIL_RETURN(vkCreateImageView(Application::app->getDevice(), &uiViewInfo, nullptr, &uiImageViews[i]));
			dutil.setObjectName(uiImageViews[i], "UI ShadowMap" + std::to_string(i));
		}
		#endif
	}

	const VkImageLayout layout{ VK_IMAGE_LAYOUT_GENERAL };
	{  // Clear all images and change layout
		std::vector<VkImageMemoryBarrier2> barriers(lightCount);
		for (uint32_t i = 0; i < lightCount; i++){
			// Best layout for clearing color
			barriers[i] = nvvk::makeImageMemoryBarrier({ .image = shadowMaps[i].image,
														.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
														.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
														.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
		}
		const VkDependencyInfo depInfo{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
									   .imageMemoryBarrierCount = (uint32_t)lightCount,
									   .pImageMemoryBarriers = barriers.data() };

		VkCommandBuffer cmd = Application::app->createTempCmdBuffer();

		vkCmdPipelineBarrier2(cmd, &depInfo);

		for (uint32_t i = 0; i < lightCount; i++){
			VkClearDepthStencilValue clearDepth = { 1.0f, 0 };
			VkImageSubresourceRange range = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
			vkCmdClearDepthStencilImage(cmd, shadowMaps[i].image,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearDepth, 1, &range);

			// Setting the layout to the final one
			barriers[i] = nvvk::makeImageMemoryBarrier(
				{ .image = shadowMaps[i].image, .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				.newLayout = layout, .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
			shadowMaps[i].descriptor.imageLayout = layout;
		}
		vkCmdPipelineBarrier2(cmd, &depInfo);

		Application::app->submitAndWaitTempCmdBuffer(cmd);
	}

#ifndef NDEBUG
	VkDevice device = Application::app->getDevice();
	descriptorPool = Application::app->getTextureDescriptorPool();
	uiDescriptorSets.resize(lightCount);

	const VkDescriptorSetLayoutBinding binding = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT };
	const VkDescriptorSetLayoutCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &binding };
	NVVK_FAIL_RETURN(vkCreateDescriptorSetLayout(device, &info, nullptr, &descLayout));

	std::vector<VkDescriptorSetLayout> layouts(lightCount, descLayout);

	std::vector<VkDescriptorImageInfo> descImages(lightCount);
	std::vector<VkWriteDescriptorSet>  writeDesc(lightCount);
	const VkDescriptorSetAllocateInfo  allocInfos = {
		 .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		 .descriptorPool = descriptorPool,
		 .descriptorSetCount = (uint32_t)lightCount,
		 .pSetLayouts = layouts.data(),
	};
	NVVK_FAIL_RETURN(vkAllocateDescriptorSets(device, &allocInfos, uiDescriptorSets.data()));

	for (uint32_t i = 0; i < lightCount; ++i){
		descImages[i] = { shadowMaps[i].descriptor.sampler, uiImageViews[i], layout };
		writeDesc[i] = {
			 .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			 .dstSet = uiDescriptorSets[i],
			 .descriptorCount = 1,
			 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			 .pImageInfo = &descImages[i],
		};
	}
	vkUpdateDescriptorSets(device, uint32_t(uiDescriptorSets.size()), writeDesc.data(), 0, nullptr);
#endif
	*/


	return VK_SUCCESS;
}
void ShadowMap::createPipeline() {
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::ShadowMapPushConstant)
	};

	const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 0,
		.pSetLayouts = nullptr,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};
	NVVK_CHECK(vkCreatePipelineLayout(Application::app->getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout));
	NVVK_DBG_NAME(pipelineLayout);
}
void ShadowMap::compileAndCreateShaders(){
	if (shadowMaps.size() == 0) return;

	SCOPED_TIMER(__FUNCTION__);

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "ShadowMap.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = sizeof(shaderio::ShadowMapPushConstant),
	};

	VkShaderCreateInfoEXT shaderInfo{
		.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
		.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
		.pName = "main",
		.setLayoutCount = 0,
		.pSetLayouts = nullptr,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};
	VkDevice device = Application::app->getDevice();
	//--------------------------------------------------------------------------------------
	vkDestroyShaderEXT(device, vertexShader_directionLight, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_directionLight, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.pName = "vertexMain_directionLight";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &vertexShader_directionLight);
	NVVK_DBG_NAME(vertexShader_directionLight);

	shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "fragmentMain_directionLight";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &fragmentShader_directionLight);
	NVVK_DBG_NAME(fragmentShader_directionLight);
	//--------------------------------------------------------------------------------------
	//vkDestroyShaderEXT(device, vertexShader_pointLight, nullptr);
	//vkDestroyShaderEXT(device, fragmentShader_pointLight, nullptr);
	//
	//shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	//shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
	//shaderInfo.pName = "vertexMain_pointLight";
	//shaderInfo.codeSize = shaderCode.codeSize;
	//shaderInfo.pCode = shaderCode.pCode;
	//vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &vertexShader_pointLight);
	//NVVK_DBG_NAME(vertexShader_pointLight);
	//
	//shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	//shaderInfo.nextStage = 0;
	//shaderInfo.pName = "fragmentMain_pointLight";
	//shaderInfo.codeSize = shaderCode.codeSize;
	//shaderInfo.pCode = shaderCode.pCode;
	//vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &fragmentShader_pointLight);
	//NVVK_DBG_NAME(fragmentShader_pointLight);
//#ifndef NDEBUG
//	//--------------------------------------------------------------------------------------
//	vkDestroyShaderEXT(device, computeShader_debug, nullptr);
//
//	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
//	shaderInfo.nextStage = 0;
//	shaderInfo.pName = "computeMain_debug";
//	shaderInfo.codeSize = shaderCode.codeSize;
//	shaderInfo.pCode = shaderCode.pCode;
//	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_debug);
//	NVVK_DBG_NAME(computeShader_debug);
//#endif
}

#ifndef NDEBUG
void ShadowMap::debug_prepare() {
	Feature::createGBuffer(false, false, shadowMaps.size(), setting.resolution);

	nvvk::DescriptorBindings bindings;
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_ShadowMap::eShadowMaps,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = (uint32_t)shadowMaps.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_ShadowMap::eDepthRestructResultMaps,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = (uint32_t)shadowMaps.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    shadowMapsWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_ShadowMap::eShadowMaps, 0, 0, shadowMaps.size());
	std::vector<nvvk::Image> shadowMaps_nvvk(shadowMaps.size());
	for (int i = 0; i < shadowMaps.size(); ++i) shadowMaps_nvvk[i] = shadowMaps[i].image;
	nvvk::Image* shadowMapsPtr = shadowMaps_nvvk.data();
	write.append(shadowMapsWrite, shadowMapsPtr);

	VkWriteDescriptorSet    depthRestructResultMapsWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_ShadowMap::eDepthRestructResultMaps, 0, 0, shadowMaps.size());
	nvvk::Image* depthRestructResultMapsPtr = gBuffers.m_res.gBufferColor.data();
	for (int i = 0; i < shadowMaps.size(); ++i) depthRestructResultMapsPtr[i].descriptor.sampler = VK_NULL_HANDLE;
	write.append(depthRestructResultMapsWrite, depthRestructResultMapsPtr);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::ShadowMapPushConstant)
	};

	const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = staticDescPack.getLayoutPtr(),
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};
	NVVK_CHECK(vkCreatePipelineLayout(Application::app->getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout));
	NVVK_DBG_NAME(pipelineLayout);
}
void ShadowMap::debug_Visualization(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_debug);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	pushConstant.inverse_lightVP = glm::inverse(pushConstant.lightVP);
	pushConstant.frameIndex = Application::frameIndex;
	for (int i = 0; i < shadowMaps.size(); ++i) {
		pushConstant.lightIndex = i;
		vkCmdPushConstants2(cmd, &pushInfo);

		VkExtent2D groupSize = nvvk::getGroupCounts(gBuffers.getSize(), VkExtent2D{ 32, 32 });
		vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
	}
}
#endif