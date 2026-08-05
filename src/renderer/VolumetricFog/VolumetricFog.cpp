#include "./VolumetricFog.h"

#include <common/Application/Application.h>
#include <nvgui/sky.hpp>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/default_structs.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

using namespace FzbRenderer;

void testScene_onePlane() {

}

VolumetricFog::VolumetricFog(pugi::xml_node& rendererNode) {
	derivFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR;
	derivFeatures.pNext = nullptr;
	derivFeatures.computeDerivativeGroupQuads = VK_TRUE;
	derivFeatures.computeDerivativeGroupLinear = VK_FALSE;
	//Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME, &derivFeatures });

	Application::vkContext->getPhysicalDeviceFeatures_notConst().fragmentStoresAndAtomics = VK_TRUE;

	atomicFloatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
	atomicFloatFeatures.shaderBufferFloat32AtomicAdd = VK_TRUE;
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, &atomicFloatFeatures });

	{
		HeightFogSetCreateInfo heightFogSetCreateInfo = { &this->volumetricFogInfos };
		heightFogSet = std::make_unique<HeightFogSet>(heightFogSetCreateInfo);

		shaderio::HeightFogInfo heightFogInfo = {
			.color = {1.0f, 1.0f, 1.0f},
			.ambientIntensity = 0.0f,
			.absorption = 0.01f,
			.scattering = 0.03f,
			.phase = -0.7f,
			.heightScale = 1.0f,
		};

		shaderio::AABB heightFogAABB;
		heightFogAABB.minimum = { 5084.0f, -69.5f, -4486.0f };
		heightFogAABB.maximum = heightFogAABB.minimum + 16.0f * shaderio::float3(7.0f, 2.0f, 5.0f);

		heightFogSet->addFog(heightFogInfo, heightFogAABB);
		//--------------------------------------------------------
		heightFogInfo = {
			.color = {1.0f, 1.0f, 1.0f},
			.ambientIntensity = 0.001f,
			.absorption = 0.01f,
			.scattering = 0.7f,
			.phase = 0.5,
			.heightScale = 1.0f,
		};

		heightFogAABB.minimum = { 5098.0f, -69.5f, -4470.0f };
		heightFogAABB.maximum = heightFogAABB.minimum + 16.0f * shaderio::float3(2.0f, 0.1f, 2.0f);

		heightFogSet->addFog(heightFogInfo, heightFogAABB);
	}
	{
		FluidFogSetCreateInfo fluidFogSetCreateInfo = { &this->volumetricFogInfos };
		fluidFogSet = std::make_unique<FluidFogSet>(fluidFogSetCreateInfo);

		shaderio::FluidFogInfo fluidFogInfo = {
			.startUp = 0,
			.gridSize = {32, 32, 32},
			.voxelSize = {1.0, 0.2, 1.0},
			.viscosity = 0.001f,
			.FIntensity = 10,
			.restoreSpeed = 10,
		};
		FluidFogCreateInfo fluidFogCreateInfo = {
			.fogInfo = fluidFogInfo,
			.fogRange = {},
			.follow = true,
			.followInstanceID = "mainCharacter"
		};
		fluidFogSet->addFog(fluidFogCreateInfo);
	}
	{
		GridFogSetCreateInfo gridFogSetCreateInfo = { &this->volumetricFogInfos };
		gridFogSet = std::make_unique<GridFogSet>(gridFogSetCreateInfo);

		shaderio::GridFogInfo gridFogInfo = {
			.gridSize = {32, 32, 32},
			.voxelSize = {1, 0.1, 1},
			.absorption = 0.1f,
			.scattering = 0.5f,
			.phase = -0.7f,
			.ambientIntensity = 0.2f,
			.color = {1.0f, 1.0f, 1.0f},
		};

		shaderio::AABB gridFogAABB;
		gridFogAABB.minimum = { 5094.0f, -69.5f, -4467.0f };
		gridFogAABB.maximum = gridFogAABB.minimum + gridFogInfo.voxelSize * (shaderio::float3)gridFogInfo.gridSize;

		shaderio::GridFogGenerationInfo generationInfo = {
			.cloudInfo = {
				.cloudScale = {0.5f, 0.2f, 0.5f},
				.cloudFlowSpeed = 0.05f,
				.cloudCoverage = {0.3f, 0.3f},
				.cloudTypePreference = {1.0f, 0.0f},
				.weatherScale = 0.01f,
			},
		};

		GridFogCreateInfo gridFogCreateInfo = {
			.fogInfo = gridFogInfo,
			.fogRange = gridFogAABB,
			.generationInfo = generationInfo
		};
		//gridFogSet->addFog(gridFogCreateInfo);
	}

	volumetricFogCount = volumetricFogInfos.size();
	if (volumetricFogCount > MAX_VOLUMETRIC_FOG_COUNT) throw std::runtime_error("体积雾数量超出上限，请扩大上限");
	pushConstant.volumetricFogCount = volumetricFogCount;
	volumetricFogInfoModified.resize(volumetricFogCount);

	pushConstant.heightFogCount = heightFogSet->fogCount;
	pushConstant.fluidFogCount = fluidFogSet->fogCount;
	pushConstant.gridFogCount = gridFogSet->fogCount;

	pushConstant.randomStepping = 1;
	pushConstant.useAccFog = 1;
	pushConstant.compressionParams = 3.0f;
	frustumGridSize = { 160, 160, 80 };
	pushConstant.frustumGridSize = { frustumGridSize.width, frustumGridSize.height, frustumGridSize.depth };
	pushConstant.time = 0.0f;


#ifndef NDEBUG
	showVolumetricFogVoxelGrids.resize(volumetricFogCount);
#endif
}

void VolumetricFog::init() {
	VkSamplerCreateInfo samplerInfo = DEFAULT_VkSamplerCreateInfo;
	samplerInfo.magFilter = VK_FILTER_NEAREST;
	samplerInfo.minFilter = VK_FILTER_NEAREST;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	Renderer::createGBuffer(true, true, (uint32_t)GBuffers_VolumetricFog::eTonemapping, { 1, 1 }, samplerInfo);
	Application::samplerPool.acquireSampler(gBuffers.m_res.gBufferDepth.descriptor.sampler, samplerInfo);

#ifndef NDEBUG
	//---------------------------------------------wireframe-----------------------------------
	nvutils::PrimitiveMesh primitive = FzbRenderer::MeshSet::createWireframe();
	FzbRenderer::MeshSet mesh = FzbRenderer::MeshSet("Wireframe", primitive);
	scene.addMeshSet(mesh);

	scene.createSceneInfoBuffer();
#endif

#ifdef USE_SVGF
	SVGFCreateInfo svgfCreateInfo = {
		.albedoImage = gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eAlbedo],
		.depthImage = gBuffers.m_res.gBufferDepth,
		.normalImage = gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eNormal],
		.velocityImage = gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eVelocity],
		.renderTarget = gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eRendered],
	};
	svgf.init(svgfCreateInfo);
#endif

#ifdef USE_TAA
	TAACreateInfo taaCreateInfo = {
		.depthImage = gBuffers.m_res.gBufferDepth,
		.velocityImage = gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eVelocity],
		.renderTarget = gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eRendered]
	};
	taa.init(taaCreateInfo);
#endif
	shadowMap.init({ 2048, 2048 });

	heightFogSet->init();
	fluidFogSet->init();
	gridFogSet->init();

	createVolumetricFogData();
	createDescriptorSetLayout();
	createDescriptorSet();
	createPipelineLayout();
	compileAndCreateShaders();

	Renderer::init();

	pushConstant.cameraNearPlane = Application::sceneResource.cameraManip->getClipPlanes().x;
	pushConstant.cameraFarPlane = 2000.0f;	// Application::sceneResource.cameraManip->getClipPlanes().y * 0.02f;
	pushConstant.tanCameraFov_2 = glm::tan(glm::radians(Application::sceneResource.cameraManip->getFov() * 0.5f));
	pushConstant.aspectRatio = Application::sceneResource.cameraManip->getAspectRatio();

	if (fluidFogSet->fogCount > 0) {
		FzbRenderer::Scene& scene = Application::sceneResource;
		for (int i = 0; i < scene.meshes.size(); ++i) {
			int meshSetIndex = scene.meshIndexToMeshSetIndex[i];
			scene.meshSets[meshSetIndex].getAABB();
		}
	}
}
void VolumetricFog::clean() {
	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, vertexShader_createGBuffer, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_createGBuffer, nullptr);
	vkDestroyShaderEXT(device, computeShader_getVisibleVolumetricFog, nullptr);

	vkDestroyShaderEXT(device, computeShader_initFluidFog, nullptr);
	vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_A, nullptr);
	vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_D, nullptr);
	vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_F, nullptr);
	vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_P, nullptr);
	vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_S, nullptr);

	vkDestroyShaderEXT(device, computeShader_createEnvionmentFog, nullptr);

	vkDestroyShaderEXT(device, computeShader_createLightAttenuationEstimator, nullptr);

	vkDestroyShaderEXT(device, computeShader_createFrustumAccFog, nullptr);
#ifdef FOG_ACC_DELETE_NOFOGVOXEL
	vkDestroyShaderEXT(device, computeShader_getHasFogVoxelInfo, nullptr);
#endif
#ifdef FOG_ACC_TWO_PASS
	vkDestroyShaderEXT(device, computeShader_createFrustumAccFog_Pass1, nullptr);
	vkDestroyShaderEXT(device, computeShader_createFrustumAccFog_Pass2, nullptr);
#endif

	vkDestroyShaderEXT(device, computeShader_deferredRenderring, nullptr);

	vkDestroyShaderEXT(device, computeShader_getDepthGradient, nullptr);
	vkDestroyShaderEXT(device, computeShader_varianceConvolution, nullptr);
	vkDestroyShaderEXT(device, computeShader_blurFog_X, nullptr);
	vkDestroyShaderEXT(device, computeShader_blurFog_Y, nullptr);
	vkDestroyShaderEXT(device, computeShader_addFog, nullptr);

	vkDestroyShaderEXT(device, vertexShader_renderTransparentMaterial, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_renderTransparentMaterial, nullptr);

	GlobalInfoBuffer.clean();
	volumetricFogInfosBuffer.clean();

	heightFogSet->clean();
	fluidFogSet->clean();
	gridFogSet->clean();

#ifdef FOG_ACC_DIVIDE_PART
	fogAccResultBuffer.clean();
#elif defined(FOG_ACC_ONE_DISPATCH)
	fogAccSyncBuffer.clean();
#endif
#ifdef FOG_ACC_DELETE_NOFOGVOXEL
	fogAccHasFogVoxelInfoBuffer.clean();
#endif
	volumetricFogAccResultImage.clean();
	envVolumetricFogInfoImage.clean();

#ifdef USE_SVGF
	svgf.clean();
#endif
#ifdef USE_TAA
	taa.clean();
#endif
	shadowMap.clean();

#ifndef NDEBUG
	vkDestroyShaderEXT(device, vertexShader_renderVoxelGrid, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_renderVoxelGrid, nullptr);

	vkDestroyShaderEXT(device, vertexShader_renderCameraFrustum, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_renderCameraFrustum, nullptr);

	vkDestroyShaderEXT(device, computeShader_test, nullptr);

	Application::allocator.destroyBuffer(bShowCameraInfo);
#endif

	Renderer::clean();
};
void VolumetricFog::uiRender() {
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;
	Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)GBuffers_VolumetricFog::eTonemapping);

	const char* fogTypeItems[] = { "Height", "Fluid", "Noise"};

	uint32_t heightFogIndex = 0, fluidFogIndex = 0, noiseFogIndex = 0;
	if (ImGui::Begin("Volumetric Fog Setting")) {
		if (ImGui::CollapsingHeader("Global Setting", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::DragFloat("Light Attenuation Strength", (float*)&pushConstant.lightAttenuationStrength, 0.1, 0, 2);
			ImGui::Checkbox("Use SVGF", (bool*)&useSVGF);
			ImGui::Checkbox("Use Fog Blur", (bool*)&pushConstant.useFogBlur);
			ImGui::DragInt("Fog Filter Count", (int*)&FogFilterCount, 1, 1, 10);
		}

#ifndef NDEBUG
		if (ImGui::CollapsingHeader("Frustum Setting", ImGuiTreeNodeFlags_DefaultOpen)) {
			UIModified |= ImGui::DragFloat("Compression Params ", (float*)&pushConstant.compressionParams, 0.1f, 0.0f, 10);
			if (ImGui::Checkbox("Show Camera Frustum ", (bool*)&showCameraFrustum)) {
				UIModified = true;
				showCameraInfo = Application::sceneResource.sceneInfo;
			}
		}
#endif
		
		#ifdef USE_ENVFOG
		if (ImGui::CollapsingHeader("Environment Fog Setting", ImGuiTreeNodeFlags_DefaultOpen)) {
			#ifdef Uniform_EnvFog_Grid
			envChange = false;
			envChange |= ImGui::DragFloat3("Environment Fog Start Pos ", (float*)&envStartPos);
			ImGui::BeginDisabled(true);
			bool change = ImGui::DragInt3("Environment Fog Voxel Grid Size ", (int*)&envGridSize);
			ImGui::EndDisabled();
			envChange |= ImGui::DragFloat3("Environment Fog Voxel Size ", (float*)&envVoxelSize);

			ImGui::Checkbox("show Environment Fog grid ", (bool*)&showEnvGrid);
			#else
			
			#endif
			ImGui::DragInt("Environment Sample Count ", (int*)&sampleCount_env, 1, 1, 50);
			ImGui::DragFloat("Environment Jitter Strength", (float*)&jitterStrength_env, 1, 0, 10);
		}
		#endif
		if (ImGui::CollapsingHeader("Fog Acc Setting", ImGuiTreeNodeFlags_DefaultOpen)) {
			UIModified |= ImGui::Checkbox("Start Up", (bool*)&pushConstant.useAccFog);
			UIModified |= ImGui::Checkbox("Random Stepping Fog Acc", (bool*)&randomStepping_fogAcc);
			UIModified |= ImGui::DragInt("Fog ACC Sample Count ", (int*)&rmSampleCountSampleCount_fogAcc, 1, 1, 50);
			UIModified |= ImGui::DragFloat("Acc Jitter Strength", (float*)&accJitterStrength, 1, 0, 100);
			UIModified |= ImGui::DragFloat("Acc Interpolation Jitter Strength", (float*)&interpolationJitterStrength_fogAcc, 1, 0, 100);
		}
		if (ImGui::CollapsingHeader("opaque render Setting", ImGuiTreeNodeFlags_DefaultOpen)) {
			UIModified |= ImGui::Checkbox("Random Stepping Opaque", (bool*)&randomStepping_opaque);
			UIModified |= ImGui::DragInt("No Fog Acc RayMarching Sample Count ", (int*)&rmSampleCount_opaque_noFogAcc, 1, 1, 200);
			//UIModified |= ImGui::DragInt("Fog Acc RayMarching Sample Count ", (int*)&rmSampleCount_opaque_FogAcc, 1, 1, 50);
			//UIModified |= ImGui::DragInt("Forward Sample Number", (int*)&forwardSampleCount, 1, 0, 100);
			UIModified |= ImGui::DragFloat("Attenuation Interpolation Jitter Strength", (float*)&interpolationJitterStrength_attenuation, 1, 0, 100);
			UIModified |= ImGui::DragFloat("L Interpolation Jitter Strength", (float*)&interpolationJitterStrength_L, 1, 0, 100);
		}
		if (ImGui::CollapsingHeader("transparent render Setting", ImGuiTreeNodeFlags_DefaultOpen)) {
			UIModified |= ImGui::DragFloat("Attenuation Interpolation Jitter Strength transparent", (float*)&interpolationJitterStrength_attenuation_transparent, 1, 0, 100);
			UIModified |= ImGui::DragFloat("L Interpolation Jitter Strength transparent", (float*)&interpolationJitterStrength_L_transparent, 1, 0, 100);
		}

		heightFogSet->uiRender();
		fluidFogSet->uiRender();
		gridFogSet->uiRender();
	}
	ImGui::End();

	shadowMap.uiRender();
}
void VolumetricFog::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
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

	nvvk::WriteSetContainer write{};

	VkWriteDescriptorSet    imageWrite = staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eAlbedoImage, 0, 0, 1);
	write.append(imageWrite, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eAlbedo]);

	imageWrite = staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eNormalImage, 0, 0, 1);
	write.append(imageWrite, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eNormal]);

	imageWrite = staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eDepthImage, 0, 0, 1);
	write.append(imageWrite, gBuffers.m_res.gBufferDepth);

	imageWrite = staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eEmissiveImage, 0, 0, 1);
	write.append(imageWrite, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eEmissive]);

	imageWrite = staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eRenderedImage, 0, 0, 1);
	write.append(imageWrite, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eRendered]);

#ifdef BLUR_FOG
	imageWrite = staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eRenderedFogResultImage, 0, 0, 1);
	write.append(imageWrite, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eRenderedResultFog]);

	imageWrite = staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eDepthGradientImage, 0, 0, 1);
	write.append(imageWrite, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eDepthGradient]);

	imageWrite = staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFilterImages, 0, 0, 2);
	std::vector<nvvk::Image> vairanceImages = { 
		gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eFilter0],  
		gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eFilter1],
	};
	write.append(imageWrite, vairanceImages.data());
#endif

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);

#ifdef USE_SVGF
	nvvk::Image images_svgf[6] = {
		gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eAlbedo],
		gBuffers.m_res.gBufferDepth,
		gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eNormal],
		gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eVelocity],
		gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eVertexInfo],
		gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eRendered],
	};
	svgf.resize(cmd, size, images_svgf);
#endif

#ifdef USE_TAA
	nvvk::Image images_taa[3] = { gBuffers.m_res.gBufferDepth, 
		gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eVelocity], 
		gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eRendered] };
	taa.resize(cmd, size, images_taa);
#endif
	shadowMap.resize(cmd, size);

	pushConstant.screenSize = { size.width, size.height };
}
void VolumetricFog::preRender() {
	Scene& scene = Application::sceneResource;
	if (scene.cameraChange) Application::frameIndex = 0;
	pushConstant.frameIndex = Application::frameIndex;
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
#ifndef NDEBUG
	pushConstant.showCameraInfoAddress = (shaderio::SceneInfo*)bShowCameraInfo.address;
#endif

	#ifdef USE_SVGF
	svgf.preRender();
	#endif

	#ifdef USE_TAA
	taa.preRender();
	#endif

	shadowMap.preRender();

	heightFogSet->preRender();
	fluidFogSet->preRender();
	gridFogSet->preRender();

	// 计算相机移动方向
	{
		const glm::mat4& viewMatrix = Application::sceneResource.sceneInfo.viewMatrix;
		glm::vec3 currentCameraPos = glm::vec3(glm::inverse(viewMatrix)[3]);
		glm::vec3 moveDir = currentCameraPos - m_lastCameraPos;
		//pushConstant.cameraMoveDir = shaderio::float3(moveDir.x, moveDir.y, moveDir.z);
		m_lastCameraPos = currentCameraPos;

		if(pushConstant.time == 0.0f)
			VPMatrix_lastFrame = Application::sceneResource.cameraManip->getPerspectiveMatrix() * Application::sceneResource.sceneInfo.viewMatrix;
	}
}
void VolumetricFog::render(VkCommandBuffer* cmdPtr) {
	VkCommandBuffer cmd = cmdPtr[0];
	NVVK_DBG_SCOPE(cmd);

	static float time = 0.0f;
	pushConstant.time = time;
	pushConstant.dt = ImGui::GetIO().DeltaTime;
	time += pushConstant.dt;

	updateDataPerFrame(cmd);

	pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::VolumetricFogPushConstant),
		.pValues = &pushConstant,
	};

	getVisibleFog(cmd);
	initFluidFog(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);

	createGBuffers(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	shadowMap.render(cmd);
	fluidSimulation(cmd);
	if (Application::sceneResource.sceneInfo.useSky){
		const glm::mat4& viewMatrix = Application::sceneResource.sceneInfo.viewMatrix;
		const glm::mat4& projMatrix = Application::sceneResource.sceneInfo.projMatrix;
		Application::skySimple.runCompute(cmd, Application::app->getViewportSize(), viewMatrix, projMatrix,
			Application::sceneResource.sceneInfo.skySimpleParam, gBuffers.getDescriptorImageInfo((uint32_t)GBuffers_VolumetricFog::eRendered));
	}
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

#ifdef USE_ENVFOG
	createEnvFog(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	//envFogLightAttenuationEstimate(cmd);
	//nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
#endif

	createFrustumAccFog(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	deferredRenderring(cmd);
#ifdef BLUR_FOG
	if (pushConstant.useFogBlur) {
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
		fogBlur(cmd);
	}
#endif
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);

	renderTransparentMaterial(cmd);

#ifndef NDEBUG
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
	renderVolumetricFogVoxelGrid(cmd);
	renderCameraFrustum(cmd);
#endif
	nvvk::cmdMemoryBarrier(cmd,
		VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

#ifdef USE_SVGF
	if (useSVGF) {
		svgf.render(cmd);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}
#endif

#ifdef USE_TAA
	taa.mergeResult(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
#endif

	//Renderer::postProcess(cmd, &gBuffers.m_res.gBufferColor[((uint32_t)GBuffers_VolumetricFog::eTonemapping)].descriptor);
	Application::tonemapper.runCompute(cmd, gBuffers.getSize(), Application::tonemapperData,
		gBuffers.getDescriptorImageInfo(((uint32_t)GBuffers_VolumetricFog::eRendered)),
		gBuffers.getDescriptorImageInfo(((uint32_t)GBuffers_VolumetricFog::eTonemapping)));
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
}

void VolumetricFog::createVolumetricFogImage(FzbRenderer::Image& image, shaderio::uint3 size, bool linear){
	image.clean();

	static int imageCount = 0;
	image = FzbRenderer::Image("volumetricFog3DTexture" + std::to_string(imageCount));
	++imageCount;

	FzbRenderer::ImageCreateInfo colorImageCreateInfo = FzbRenderer::createDefaultImageCreateInfo();
	colorImageCreateInfo.info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	colorImageCreateInfo.info.imageType = VK_IMAGE_TYPE_3D;
	colorImageCreateInfo.info.extent = { size.x, size.y, size.z };

	colorImageCreateInfo.viewInfo.format = colorImageCreateInfo.info.format;
	colorImageCreateInfo.viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;

	if (linear) {
		colorImageCreateInfo.samplerInfo.magFilter = VK_FILTER_LINEAR;
		colorImageCreateInfo.samplerInfo.minFilter = VK_FILTER_LINEAR;
		colorImageCreateInfo.samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	}
	else {
		colorImageCreateInfo.samplerInfo.magFilter = VK_FILTER_NEAREST;
		colorImageCreateInfo.samplerInfo.minFilter = VK_FILTER_NEAREST;
		colorImageCreateInfo.samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	}
	colorImageCreateInfo.samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	colorImageCreateInfo.samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	colorImageCreateInfo.samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	//colorImageCreateInfo.samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
	//colorImageCreateInfo.samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	//colorImageCreateInfo.samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	//colorImageCreateInfo.samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;

	image.init(colorImageCreateInfo);
}
void VolumetricFog::createVolumetricFogData() {
	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();

	GlobalInfoBuffer = FzbRenderer::Buffer("GlobalInfoBuffer", false);
	GlobalInfoBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(shaderio::GlobalInfo_VolumetricFog),
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});

	volumetricFogInfosBuffer = FzbRenderer::Buffer("volumetricFogInfosBuffer", false);
	volumetricFogInfosBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(shaderio::VolumetricFogInfo) * volumetricFogCount,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});
	vkCmdUpdateBuffer(cmd, volumetricFogInfosBuffer.buffer.buffer, 0, sizeof(shaderio::VolumetricFogInfo) * volumetricFogCount, volumetricFogInfos.data());
	//-------------------------------------------------------------------------------------------------------------------
	{
#ifdef FOG_ACC_DELETE_NOFOGVOXEL
		fogAccHasFogVoxelInfoBuffer = FzbRenderer::Buffer("fogAccHasFogVoxelInfoBuffer", false);
		fogAccHasFogVoxelInfoBuffer.init({
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = sizeof(shaderio::FogAccHasFogVoxelInfo) * frustumGridSize.width * frustumGridSize.height * frustumGridSize.depth,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});
#endif

		uint32_t threadGroupCount_oneDepth_X = (frustumGridSize.width + FOG_ACC_THREADGROUP_SIZE - 1) / FOG_ACC_THREADGROUP_SIZE;
		uint32_t threadGroupCount_oneDepth_Y = (frustumGridSize.height + FOG_ACC_THREADGROUP_SIZE - 1) / FOG_ACC_THREADGROUP_SIZE;
#if defined(FOG_ACC_DIVIDE_PART)
		fogAccResultBuffer = FzbRenderer::Buffer("fogAccResultBuffer", false);
		fogAccResultBuffer.init({
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = sizeof(shaderio::float2) * threadGroupCount_oneDepth_X * FOG_ACC_THREADGROUP_SIZE * threadGroupCount_oneDepth_Y * FOG_ACC_THREADGROUP_SIZE,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});
#elif defined(FOG_ACC_ONE_DISPATCH)
		fogAccSyncBuffer = FzbRenderer::Buffer("fogAccSyncBuffer", false);
		fogAccSyncBuffer.init({
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = sizeof(int) * threadGroupCount_oneDepth_X * threadGroupCount_oneDepth_Y,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});
#endif

		createVolumetricFogImage(volumetricFogAccResultImage, { frustumGridSize.width, frustumGridSize.height, frustumGridSize.depth }, true);

#ifdef Uniform_EnvFog_Grid
		createVolumetricFogImage(envVolumetricFogInfoImage, { envGridSize.x, envGridSize.y, envGridSize.z }, true);
#else
		createVolumetricFogImage(envVolumetricFogInfoImage, { frustumGridSize.width, frustumGridSize.height, frustumGridSize.depth }, true);
#endif
	}
#ifndef NDEBUG
	NVVK_CHECK(Application::allocator.createBuffer(bShowCameraInfo,
		std::span<const shaderio::SceneInfo>(&showCameraInfo, 1).size_bytes(),
		VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT));
#endif
	
	Application::app->submitAndWaitTempCmdBuffer(cmd);
}
void VolumetricFog::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;
	{
		bindings.addBinding({ .binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eTextures,
					 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					 .descriptorCount = std::max(uint32_t(Application::sceneResource.textures.size()), 1u),
					 .stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
				.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eAlbedoImage,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eNormalImage,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eDepthImage,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eEmissiveImage,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
	}
	//---------------------------------------------------------------------------------------------------------------------------------------------
	{
		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eGlobalInfoBuffer,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFogInfosBuffer,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	}
	//----------------------------------------------------------------高度-----------------------------------------------------------------------------
	{
		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eHeightFogInfoBuffer,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	}
	//----------------------------------------------------------------流体-----------------------------------------------------------------------------
	{
		uint32_t FluidFogCount = fluidFogSet->fogCount;

		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFluidFogInfoBuffer,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = FluidFogCount,
			.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFluidFogVoxelInfoBuffer,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = FluidFogCount,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFluidFogVoxelVelocityImage,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = FluidFogCount,
			.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFluidFogVoxelVelocityImage_sample,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = FluidFogCount,
			.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFluidFogVoxelInfoImages,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = FluidFogCount,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFluidFogVoxelInfoImages_sampler,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = FluidFogCount,
			.stageFlags = VK_SHADER_STAGE_ALL });
	}
	//----------------------------------------------------------------网格-----------------------------------------------------------------------------
	{
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eGridFogInfoBuffer,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = (uint32_t)gridFogSet->fogCount,
			.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eGridFogImages,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = (uint32_t)gridFogSet->fogCount,
			.stageFlags = VK_SHADER_STAGE_ALL });
	}
	//---------------------------------------------------------------------------------------------------------------------------------------------
	{
#ifdef FOG_ACC_DIVIDE_PART
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFogAccResultBuffer,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
#elif defined(FOG_ACC_ONE_DISPATCH)
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFogAccSyncBuffer,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		.stageFlags = VK_SHADER_STAGE_ALL
	});
#endif
#ifdef FOG_ACC_DELETE_NOFOGVOXEL
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFogAccHasFogVoxelInfoBuffer,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL
		});
#endif

		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFogAccResultImage,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFogAccResultImage_sample,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eEnvFogInfoImage,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eEnvFogInfoImage_sample,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
	}
	//-----------------------------------------------------------------Blur----------------------------------------------------------------------------
#ifdef BLUR_FOG
	{
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eRenderedFogResultImage,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eDepthGradientImage,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFilterImages,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 2,
			.stageFlags = VK_SHADER_STAGE_ALL });
	}
#endif
	//---------------------------------------------------------------------------------------------------------------------------------------------
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eShadowMap,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eRenderedImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	LOGI("Volumetric Fog static descriptor layout created\n");
	NVVK_DBG_NAME(staticDescPack.getLayout());
	NVVK_DBG_NAME(staticDescPack.getPool());
	NVVK_DBG_NAME(staticDescPack.getSet(0));
}
void VolumetricFog::createDescriptorSet() {
	nvvk::WriteSetContainer write{};

	{
		if (!Application::sceneResource.textures.empty()) {
			VkWriteDescriptorSet    allTextures =
				staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eTextures, 0, 0, uint32_t(Application::sceneResource.textures.size()));
			nvvk::Image* allImages = Application::sceneResource.textures.data();
			write.append(allTextures, allImages);
		}

		VkWriteDescriptorSet	gBuffersWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eAlbedoImage, 0, 0, 1);
		write.append(gBuffersWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBuffers_VolumetricFog::eAlbedo));

		gBuffersWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eNormalImage, 0, 0, 1);
		write.append(gBuffersWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBuffers_VolumetricFog::eNormal));

		gBuffersWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eDepthImage, 0, 0, 1);
		write.append(gBuffersWrite, gBuffers.getDepthImageView(), VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL, gBuffers.m_res.gBufferDepth.descriptor.sampler);

		gBuffersWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eEmissiveImage, 0, 0, 1);
		write.append(gBuffersWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBuffers_VolumetricFog::eEmissive));
	}
	//---------------------------------------------------------------------------------------------------------------------------------------------
	{
		VkWriteDescriptorSet globalInfoWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eGlobalInfoBuffer, 0, 0, 1);
		write.append(globalInfoWrite, GlobalInfoBuffer.buffer);

		VkWriteDescriptorSet	volumetricFogInfoWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFogInfosBuffer, 0, 0, 1);
		write.append(volumetricFogInfoWrite, volumetricFogInfosBuffer.buffer);
	}
	//------------------------------------------------------------------高度---------------------------------------------------------------------------
	if (heightFogSet->fogCount > 0) {
		VkWriteDescriptorSet	volumetricFogHeightWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eHeightFogInfoBuffer, 0, 0, 1);
		write.append(volumetricFogHeightWrite, heightFogSet->heightFogInfoBuffer.buffer);
	}
	//------------------------------------------------------------------流体---------------------------------------------------------------------------
	int FluidFogCount = fluidFogSet->fogCount;
	if (FluidFogCount > 0) {
		VkWriteDescriptorSet	FluidFogWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFluidFogInfoBuffer, 0, 0, 1);
		write.append(FluidFogWrite, fluidFogSet->fluidFogInfoBuffer.buffer);

		FluidFogWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFluidFogVoxelInfoBuffer, 0, 0, FluidFogCount);
		write.append(FluidFogWrite, fluidFogSet->getfluidFogVoxelInfoBuffersPtr());

		FluidFogWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFluidFogVoxelVelocityImage, 0, 0, FluidFogCount);
		write.append(FluidFogWrite, fluidFogSet->getfluidFogVoxelVelocityImagesPtr());

		FluidFogWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFluidFogVoxelVelocityImage_sample, 0, 0, FluidFogCount);
		write.append(FluidFogWrite, fluidFogSet->getfluidFogVoxelVelocityImagesPtr());

		FluidFogWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFluidFogVoxelInfoImages, 0, 0, FluidFogCount);
		write.append(FluidFogWrite, fluidFogSet->getfluidFogVoxelInfoImagesPtr());

		FluidFogWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFluidFogVoxelInfoImages_sampler, 0, 0, FluidFogCount);
		write.append(FluidFogWrite, fluidFogSet->getfluidFogVoxelInfoImagesPtr());
	}
	//------------------------------------------------------------------网格---------------------------------------------------------------------------
	if (gridFogSet->fogCount > 0) {
		VkWriteDescriptorSet	volumetricFogGridWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eGridFogInfoBuffer, 0, 0, 1);
		write.append(volumetricFogGridWrite, gridFogSet->gridFogInfoBuffer.buffer);

		volumetricFogGridWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eGridFogImages, 0, 0, gridFogSet->fogCount);
		write.append(volumetricFogGridWrite, gridFogSet->getFogGridImagesPtr());
	}
	//------------------------------------------------------------------------------------------------------------------------------------------------
	{
		VkWriteDescriptorSet	fogAccImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFogAccResultImage, 0, 0, 1);
		write.append(fogAccImageWrite, volumetricFogAccResultImage.image);

#ifdef FOG_ACC_DIVIDE_PART
		fogAccImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFogAccResultBuffer, 0, 0, 1);
		write.append(fogAccImageWrite, fogAccResultBuffer.buffer);
#elif defined(FOG_ACC_ONE_DISPATCH)
		fogAccImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFogAccSyncBuffer, 0, 0, 1);
		write.append(fogAccImageWrite, fogAccSyncBuffer.buffer);
#endif
#ifdef FOG_ACC_DELETE_NOFOGVOXEL
		fogAccImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFogAccHasFogVoxelInfoBuffer, 0, 0, 1);
		write.append(fogAccImageWrite, fogAccHasFogVoxelInfoBuffer.buffer);
#endif

		fogAccImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFogAccResultImage_sample, 0, 0, 1);
		write.append(fogAccImageWrite, volumetricFogAccResultImage.image);

		VkWriteDescriptorSet	envCalInfoImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eEnvFogInfoImage, 0, 0, 1);
		write.append(envCalInfoImageWrite, envVolumetricFogInfoImage.image);

		envCalInfoImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eEnvFogInfoImage_sample, 0, 0, 1);
		write.append(envCalInfoImageWrite, envVolumetricFogInfoImage.image);
	}
	//---------------------------------------------------------------------------------------------------------------------------------------------
	VkWriteDescriptorSet	shadowMapWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eShadowMap, 0, 0, 1);
	write.append(shadowMapWrite, shadowMap.shadowMaps[0].image);

	VkWriteDescriptorSet	renderedImageWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eRenderedImage, 0, 0, 1);
	write.append(renderedImageWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBuffers_VolumetricFog::eRendered));

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void VolumetricFog::createPipelineLayout() {
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::VolumetricFogPushConstant)
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
void VolumetricFog::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource;
	VkShaderModuleCreateInfo shaderCode;

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = sizeof(shaderio::VolumetricFogPushConstant),
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
	//-------------------------------------------Step1-------------------------------------------
	{
		shaderSource = shaderPath / "step1_getVisibleFog.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_getVisibleVolumetricFog, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getVisibleVolumetricFog";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getVisibleVolumetricFog);
		NVVK_DBG_NAME(computeShader_getVisibleVolumetricFog);
	}
	//-------------------------------------------Step2-------------------------------------------
	{
		shaderSource = shaderPath / "step2_initFluid.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_initFluidFog, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_initFluidFog";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_initFluidFog);
		NVVK_DBG_NAME(computeShader_initFluidFog);
	}
	//-------------------------------------------Step3-------------------------------------------
	{
		shaderSource = shaderPath / "step3_createGBuffers.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, vertexShader_createGBuffer, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.pName = "vertexMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &vertexShader_createGBuffer);
		NVVK_DBG_NAME(vertexShader_createGBuffer);

		vkDestroyShaderEXT(device, fragmentShader_createGBuffer, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "fragmentMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &fragmentShader_createGBuffer);
		NVVK_DBG_NAME(fragmentShader_createGBuffer);
	}
	//-------------------------------------------Step4-------------------------------------------
	{
		shaderSource = shaderPath / "step4_fluidSimulation.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_A, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createVolumetricFog_Fluid_A";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createVolumetricFog_Fluid_A);
		NVVK_DBG_NAME(computeShader_createVolumetricFog_Fluid_A);

		vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_D, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createVolumetricFog_Fluid_D";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createVolumetricFog_Fluid_D);
		NVVK_DBG_NAME(computeShader_createVolumetricFog_Fluid_D);

		vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_F, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createVolumetricFog_Fluid_F";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createVolumetricFog_Fluid_F);
		NVVK_DBG_NAME(computeShader_createVolumetricFog_Fluid_F);

		vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_P, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createVolumetricFog_Fluid_P";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createVolumetricFog_Fluid_P);
		NVVK_DBG_NAME(computeShader_createVolumetricFog_Fluid_P);

		vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_S, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createVolumetricFog_Fluid_S";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createVolumetricFog_Fluid_S);
		NVVK_DBG_NAME(computeShader_createVolumetricFog_Fluid_S);
	}
	//-------------------------------------------Step5-------------------------------------------
	#ifdef USE_ENVFOG
	{
		shaderSource = shaderPath / "step5_createEnvironmentFog.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_createEnvionmentFog, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createEnvionmentFog";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createEnvionmentFog);
		NVVK_DBG_NAME(computeShader_createEnvionmentFog);
	}
	#endif
	//-------------------------------------------Step6-------------------------------------------
	#ifdef USE_ENVFOG
	{
		shaderSource = shaderPath / "step6_createLightAttenuatonEstimator.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_createLightAttenuationEstimator, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createLightAttenuationEstimator";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createLightAttenuationEstimator);
		NVVK_DBG_NAME(computeShader_createLightAttenuationEstimator);
	}
	#endif
	//-------------------------------------------Step7-------------------------------------------
	{
		shaderSource = shaderPath / "step7_createFrustumAccFog.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_createFrustumAccFog, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		#ifdef Fog_Acc_Stepping
			#if defined(FOG_ACC_DIVIDE_PART)
					shaderInfo.pName = "computeMain_createFrustumAccFog_Stepping_GPUGriven";
			#elif defined(FOG_ACC_ONE_DISPATCH)
					shaderInfo.pName = "computeMain_createFrustumAccFog_Stepping_oneDispatch";
			#elif defined(FOG_ACC_SERIAL)
					shaderInfo.pName = "computeMain_createFrustumAccFog_Stepping_Serial";
			#else
					shaderInfo.pName = "computeMain_createFrustumAccFog_Stepping";
			#endif
		#else
		shaderInfo.pName = "computeMain_createFrustumAccFog";
		#endif
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createFrustumAccFog);
		NVVK_DBG_NAME(computeShader_createFrustumAccFog);

#ifdef FOG_ACC_DELETE_NOFOGVOXEL
		vkDestroyShaderEXT(device, computeShader_getHasFogVoxelInfo, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getHasFogVoxelInfo";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getHasFogVoxelInfo);
		NVVK_DBG_NAME(computeShader_getHasFogVoxelInfo);
#endif

#ifdef FOG_ACC_TWO_PASS
		vkDestroyShaderEXT(device, computeShader_createFrustumAccFog_Pass1, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createFrustumAccFog_Stepping_Pass1";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createFrustumAccFog_Pass1);
		NVVK_DBG_NAME(computeShader_createFrustumAccFog_Pass1);

		vkDestroyShaderEXT(device, computeShader_createFrustumAccFog_Pass2, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createFrustumAccFog_Stepping_Pass2";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createFrustumAccFog_Pass2);
		NVVK_DBG_NAME(computeShader_createFrustumAccFog_Pass2);
#endif
	}
	//-------------------------------------------Step8-------------------------------------------
	{
		shaderSource = shaderPath / "step8_deferredRendering.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_deferredRenderring, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_deferredRendering";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_deferredRenderring);
		NVVK_DBG_NAME(computeShader_deferredRenderring);
		//-------------------------------------Step8.5-------------------------------------------
		{
#ifdef BLUR_FOG
			shaderSource = shaderPath / "step8_5_fogBlur.slang";
			shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

			vkDestroyShaderEXT(device, computeShader_getDepthGradient, nullptr);
			shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			shaderInfo.nextStage = 0;
			shaderInfo.pName = "computeMain_getDepthGradient";
			shaderInfo.codeSize = shaderCode.codeSize;
			shaderInfo.pCode = shaderCode.pCode;
			vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getDepthGradient);
			NVVK_DBG_NAME(computeShader_getDepthGradient);

			/*
			vkDestroyShaderEXT(device, computeShader_varianceConvolution, nullptr);
			shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			shaderInfo.nextStage = 0;
			shaderInfo.pName = "computeMain_varianceConvolution";
			shaderInfo.codeSize = shaderCode.codeSize;
			shaderInfo.pCode = shaderCode.pCode;
			vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_varianceConvolution);
			NVVK_DBG_NAME(computeShader_varianceConvolution);
			*/

			vkDestroyShaderEXT(device, computeShader_blurFog_X, nullptr);
			shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			shaderInfo.nextStage = 0;
			shaderInfo.pName = "computeMain_blurFog_X";
			shaderInfo.codeSize = shaderCode.codeSize;
			shaderInfo.pCode = shaderCode.pCode;
			vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_blurFog_X);
			NVVK_DBG_NAME(computeShader_blurFog_X);

			vkDestroyShaderEXT(device, computeShader_blurFog_Y, nullptr);
			shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			shaderInfo.nextStage = 0;
			shaderInfo.pName = "computeMain_blurFog_Y";
			shaderInfo.codeSize = shaderCode.codeSize;
			shaderInfo.pCode = shaderCode.pCode;
			vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_blurFog_Y);
			NVVK_DBG_NAME(computeShader_blurFog_Y);

			vkDestroyShaderEXT(device, computeShader_addFog, nullptr);
			shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			shaderInfo.nextStage = 0;
			shaderInfo.pName = "computeMain_addFog";
			shaderInfo.codeSize = shaderCode.codeSize;
			shaderInfo.pCode = shaderCode.pCode;
			vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_addFog);
			NVVK_DBG_NAME(computeShader_addFog);
#endif
		}
	}
	//-------------------------------------------Step9-------------------------------------------
	{
		shaderSource = shaderPath / "step9_renderTransparentMaterial.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, vertexShader_renderTransparentMaterial, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.pName = "vertexMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &vertexShader_renderTransparentMaterial);
		NVVK_DBG_NAME(vertexShader_renderTransparentMaterial);

		vkDestroyShaderEXT(device, fragmentShader_renderTransparentMaterial, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "fragmentMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &fragmentShader_renderTransparentMaterial);
		NVVK_DBG_NAME(fragmentShader_renderTransparentMaterial);
	}
	//-------------------------------------------Debug-------------------------------------------
#ifndef NDEBUG
	shaderSource = shaderPath / "renderVolumetricFogVoxelGrid.slang";
	shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	vkDestroyShaderEXT(device, vertexShader_renderVoxelGrid, nullptr);
	shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.pName = "vertexMain";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &vertexShader_renderVoxelGrid);
	NVVK_DBG_NAME(vertexShader_renderVoxelGrid);

	vkDestroyShaderEXT(device, fragmentShader_renderVoxelGrid, nullptr);
	shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "fragmentMain";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &fragmentShader_renderVoxelGrid);
	NVVK_DBG_NAME(fragmentShader_renderVoxelGrid);
	//--------------------------------------------------------------------------------------
	shaderSource = shaderPath / "renderCameraFrustum.slang";
	shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	vkDestroyShaderEXT(device, vertexShader_renderCameraFrustum, nullptr);
	shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.pName = "vertexMain";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &vertexShader_renderCameraFrustum);
	NVVK_DBG_NAME(vertexShader_renderCameraFrustum);

	vkDestroyShaderEXT(device, fragmentShader_renderCameraFrustum, nullptr);
	shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "fragmentMain";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &fragmentShader_renderCameraFrustum);
	NVVK_DBG_NAME(fragmentShader_renderCameraFrustum);

	{
		shaderSource = shaderPath / "test.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_test, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_test";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_test);
		NVVK_DBG_NAME(computeShader_test);
	}
#endif
}
void VolumetricFog::updateDataPerFrame(VkCommandBuffer cmd) {
	#ifdef Uniform_EnvFog_Grid
	if (envChange || pushConstant.time == 0.0f) {
		nvvk::cmdBufferMemoryBarrier(cmd, { GlobalInfoBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT });
		vkCmdUpdateBuffer(cmd, GlobalInfoBuffer.buffer.buffer, offsetof(shaderio::GlobalInfo_VolumetricFog, envStartPos), sizeof(shaderio::float3), &envStartPos);
		vkCmdUpdateBuffer(cmd, GlobalInfoBuffer.buffer.buffer, offsetof(shaderio::GlobalInfo_VolumetricFog, envGridSize), sizeof(shaderio::uint3), &envGridSize);
		vkCmdUpdateBuffer(cmd, GlobalInfoBuffer.buffer.buffer, offsetof(shaderio::GlobalInfo_VolumetricFog, envVoxelSize), sizeof(shaderio::float3), &envVoxelSize);

		nvvk::cmdBufferMemoryBarrier(cmd, { GlobalInfoBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT });
	}
	#endif

	#ifdef USE_TAA
	nvvk::cmdBufferMemoryBarrier(cmd, { GlobalInfoBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT });
	vkCmdUpdateBuffer(cmd, GlobalInfoBuffer.buffer.buffer, offsetof(shaderio::GlobalInfo_VolumetricFog, VPMatrix_lastFrame), sizeof(shaderio::float4x4), &VPMatrix_lastFrame);
	nvvk::cmdBufferMemoryBarrier(cmd, { GlobalInfoBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT });

	VPMatrix_lastFrame = Application::sceneResource.cameraManip->getPerspectiveMatrix() * Application::sceneResource.cameraManip->getViewMatrix();
	#endif

	if (volumetricFogCount > 0) nvvk::cmdBufferMemoryBarrier(cmd, { volumetricFogInfosBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT });
	heightFogSet->updateDataPerFrame(cmd, pushConstant.time == 0.0f, volumetricFogInfosBuffer);
	fluidFogSet->updateDataPerFrame(cmd, pushConstant.time == 0.0f, volumetricFogInfosBuffer);
	gridFogSet->updateDataPerFrame(cmd, pushConstant.time == 0.0f, volumetricFogInfosBuffer);
	if (volumetricFogCount > 0) nvvk::cmdBufferMemoryBarrier(cmd, { volumetricFogInfosBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT });

#ifndef NDEBUG
	static bool showCameraGetTrue = false;
	if (showCameraGetTrue == false && showCameraFrustum) {
		nvvk::cmdBufferMemoryBarrier(cmd, { bShowCameraInfo.buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT });
		vkCmdUpdateBuffer(cmd, bShowCameraInfo.buffer, 0, sizeof(shaderio::SceneInfo), &showCameraInfo);
		nvvk::cmdBufferMemoryBarrier(cmd, { bShowCameraInfo.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT });
	}
	showCameraGetTrue = showCameraFrustum;
#endif
}

void VolumetricFog::getVisibleFog(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getVisibleVolumetricFog);
	vkCmdPushConstants2(cmd, &pushInfo);
	VkExtent3D groupSize = nvvk::getGroupCounts(VkExtent3D{ volumetricFogCount, 1, 1 }, VkExtent3D{ 1024, 1, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
}
void VolumetricFog::initFluidFog(VkCommandBuffer cmd) {
	int FluidFogCount = fluidFogSet->fogCount;
	if (FluidFogCount == 0) return;
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);
	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	for (int i = 0; i < FluidFogCount; ++i) {
		if (!fluidFogSet->getFogStartUp(i)) continue;
		shaderio::uint3 gridSize = fluidFogSet->getFogGridSize(i);
		VkExtent3D groupSize = nvvk::getGroupCounts(VkExtent3D{ gridSize.x, gridSize.y, gridSize.z }, VkExtent3D{ 4, 4, 4 });

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initFluidFog);
		float time = pushConstant.time;

		if (fluidFogSet->getFogModified(i)) pushConstant.time = 0.0f;
		pushConstant.instanceIndex = fluidFogSet->getFogIndexMap(i);
		vkCmdPushConstants2(cmd, &pushInfo);

		pushConstant.time = time;
		vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
	}
}
void VolumetricFog::createGBuffers(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	uint32_t numColorAttachments = (uint32_t)GBuffers_VolumetricFog::eRendered;
	std::vector<VkRenderingAttachmentInfo> colorAttachments(numColorAttachments);
	for (int i = 0; i < (uint32_t)GBuffers_VolumetricFog::eRendered; ++i) {
		nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(i), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

		colorAttachments[i] = DEFAULT_VkRenderingAttachmentInfo;
		colorAttachments[i].clearValue = { .color = {0, 0, 0, 1.0f} };
		colorAttachments[i].imageView = gBuffers.getColorImageView(i);
	}

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.imageView = gBuffers.getDepthImageView();
	depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };

	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = DEFAULT_VkRect2D(gBuffers.getSize());
	renderingInfo.colorAttachmentCount = colorAttachments.size();
	renderingInfo.pColorAttachments = colorAttachments.data();
	renderingInfo.pDepthAttachment = &depthAttachment;

	vkCmdBeginRendering(cmd, &renderingInfo);

	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;	//背面物体可能影响流体
	graphicsDynamicPipeline.depthStencilState.stencilTestEnable = VK_FALSE;
	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, Application::app->getViewportSize());

	VkColorComponentFlags writeMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	VkBool32 blendEnable = VK_FALSE;
	for (uint32_t i = 0; i < numColorAttachments; ++i) {
		vkCmdSetColorWriteMaskEXT(cmd, i, 1, &writeMask);
		vkCmdSetColorBlendEnableEXT(cmd, i, 1, &blendEnable);
	}

	vkCmdSetDepthTestEnable(cmd, VK_TRUE);
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_createGBuffer, .fragment = fragmentShader_createGBuffer });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

#ifdef USE_TAA
	static int iteration = 0;
	pushConstant.iteration = iteration;
	++iteration;
#endif

	for (size_t i = 0; i < Application::sceneResource.instances.size(); ++i)
	{
		uint32_t meshIndex = Application::sceneResource.instances[i].meshIndex;
		const shaderio::Mesh& mesh = Application::sceneResource.meshes[meshIndex];
		const shaderio::TriangleMesh& triMesh = mesh.triMesh;

		FzbRenderer::InstanceSet* instanceSet;
		pushConstant.instanceVelocity = shaderio::float3(0.0f);
		if (Application::sceneResource.periodInstanceIndexToInstanceSetIndex.count(i)) {
			uint32_t instanceSetIndex = Application::sceneResource.periodInstanceIndexToInstanceSetIndex[i];
			instanceSet = &Application::sceneResource.periodInstanceSets[instanceSetIndex];

			int meshSetIndex = Application::sceneResource.meshIndexToMeshSetIndex[i];
			shaderio::AABB meshAABB = Application::sceneResource.meshSets[meshSetIndex].getAABB();
			shaderio::float3 localCenter = (meshAABB.minimum + meshAABB.maximum) * 0.5f;

			shaderio::float3 pos0 = shaderio::float3(instanceSet->transform * shaderio::float4(localCenter, 1.0f));
			shaderio::float3 pos1 = shaderio::float3(instanceSet->transform_lastTime * shaderio::float4(localCenter, 1.0f));
			pushConstant.instanceVelocity = (pos0 - pos1) / pushConstant.dt;
			//pushConstant.transofrm_lastTime = instanceSet->transform_lastTime;
		}
		else {		
			uint32_t instanceSetIndex = Application::sceneResource.staticInstanceIndexToInstanceSetIndex[i];
			instanceSet = &Application::sceneResource.staticInstanceSets[instanceSetIndex];
			//静态的先用AABB与流体AABB进行判断，如果不相交，则几何无需与流体进行判断; 动态的每帧需要重新计算AABB，还不如直接FS中进行判断呢
			//shaderio::AABB instanceAABB = instanceSet->aabb;
			//
			//for (int j = 0; j < FluidFogCount; ++j) {
			//	uint32_t fluidFogIndex = FluidFogIndexMap[j];
			//	shaderio::VolumetricFogInfo fogInfo = volumetricFogInfos[fluidFogIndex];
			//	shaderio::AABB fogAABB = { .minimum = fogInfo.fogStartPos, .maximum = fogInfo.fogStartPos + (shaderio::float3)fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize };
			//
			//	if (instanceAABB.maximum.x >= fogAABB.minimum.x && instanceAABB.minimum.x <= fogAABB.maximum.x &&
			//		instanceAABB.maximum.y >= fogAABB.minimum.y && instanceAABB.minimum.y <= fogAABB.maximum.y &&
			//		instanceAABB.maximum.z >= fogAABB.minimum.z && instanceAABB.minimum.z <= fogAABB.maximum.z
			//		) {
			//		pushConstant.FluidFogIndex = 1;
			//		break;
			//	}
			//}
		}

		pushConstant.normalMatrix = glm::transpose(glm::inverse(glm::mat3(Application::sceneResource.instances[i].transform)));
		pushConstant.instanceIndex = int(i);
		pushConstant.lightVP = instanceSet->transform_lastTime;
		vkCmdPushConstants2(cmd, &pushInfo);

		uint32_t bufferIndex = Application::sceneResource.getMeshBufferIndex(meshIndex);
		const nvvk::Buffer& v = Application::sceneResource.bDatas[bufferIndex];

		vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

		vkCmdDrawIndexed(cmd, triMesh.indices.count, 1, 0, 0, 0);
	}
	pushConstant.lightVP = shadowMap.pushConstant.lightVP;

	vkCmdEndRendering(cmd);

	for (int i = 0; i < (uint32_t)GBuffers_VolumetricFog::eRendered; ++i)
		nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(i), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}
void VolumetricFog::fluidSimulation(VkCommandBuffer cmd) {
	int FluidFogCount = fluidFogSet->fogCount;
	if (FluidFogCount == 0) return;
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	VkExtent3D groupSize = nvvk::getGroupCounts(VkExtent3D{ volumetricFogCount, 1, 1 }, VkExtent3D{ 1024, 1, 1 });
	{
		auto barrierImage = [&](VkImage image) {
			VkImageMemoryBarrier2 b = nvvk::makeImageMemoryBarrier({
				.image = image,
				.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
				.newLayout = VK_IMAGE_LAYOUT_GENERAL,
				.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
				.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
				.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
				.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
				.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
			});
			VkDependencyInfo depInfo{ 
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.imageMemoryBarrierCount = 1, 
				.pImageMemoryBarriers = &b };
			vkCmdPipelineBarrier2(cmd, &depInfo);
		};
		auto barrierBuffer = [&](VkBuffer buffer) {
			VkBufferMemoryBarrier2 b = nvvk::makeBufferMemoryBarrier({
				.buffer = buffer,
				.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
				.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
				.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
				.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
				});
			VkDependencyInfo depInfo{
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.bufferMemoryBarrierCount = 1,
				.pBufferMemoryBarriers = &b };
			vkCmdPipelineBarrier2(cmd, &depInfo);
			};

		auto barrierVolumeAll = [&](int fluidIndex) {
			barrierImage(fluidFogSet->getfluidFogVoxelVelocityImagesPtr()[fluidIndex].image);
			barrierBuffer(fluidFogSet->getfluidFogVoxelInfoBuffersPtr()[fluidIndex].buffer);
			barrierImage(fluidFogSet->getfluidFogVoxelInfoImagesPtr()[fluidIndex].image);
		};

		auto barrierVolumePingPong = [&](int idx) {
			barrierBuffer(fluidFogSet->getfluidFogVoxelInfoBuffersPtr()[idx].buffer);
		};

		int FluidFogCount = fluidFogSet->fogCount;
		for (int i = 0; i < FluidFogCount; ++i) {
			if (!fluidFogSet->getFogStartUp(i)) continue;
			shaderio::uint3 gridSize = fluidFogSet->getFogGridSize(i);
			groupSize = nvvk::getGroupCounts(VkExtent3D{ gridSize.x, gridSize.y, gridSize.z }, VkExtent3D{ 4, 4, 4 });
			pushConstant.instanceIndex = fluidFogSet->getFogIndexMap(i);

			// --- Stage A: Advection ---
			vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createVolumetricFog_Fluid_A);
			vkCmdPushConstants2(cmd, &pushInfo);
			vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
			barrierVolumeAll(i);

			uint32_t iterationStart = 0;

			// --- Stage D: Diffusion (Jacobi) ---
			vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createVolumetricFog_Fluid_D);
#ifdef FLUID_SIMPLIFY_VISCOSITY
			pushConstant.iteration = 0;
			vkCmdPushConstants2(cmd, &pushInfo);
			vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
			barrierVolumePingPong(i);
#else
			for (uint32_t iter = 0; iter < Jacobi_Iteration_Count; ++iter) {
				pushConstant.iteration = iter;
				vkCmdPushConstants2(cmd, &pushInfo);
				vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
				barrierVolumePingPong(i);
			}
			iterationStart = Jacobi_Iteration_Count & 1;
#endif
			

			// --- Stage F: External forces ---
			vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createVolumetricFog_Fluid_F);
			pushConstant.iteration = iterationStart;
			vkCmdPushConstants2(cmd, &pushInfo);
			vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
			barrierVolumeAll(i);

#ifndef FLUID_SIMPLIFY_PRESSURE
			// --- Stage P: Pressure solve (Jacobi) ---
			vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createVolumetricFog_Fluid_P);
			for (uint32_t iter = 0; iter < Jacobi_Iteration_Count; ++iter) {
				pushConstant.iteration = iter + iterationStart;
				vkCmdPushConstants2(cmd, &pushInfo);
				vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
				barrierVolumePingPong(i);
			}

			// --- Stage S: Subtract pressure gradient ---
			vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createVolumetricFog_Fluid_S);
			pushConstant.iteration = Jacobi_Iteration_Count;
			vkCmdPushConstants2(cmd, &pushInfo);
			vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
			barrierVolumeAll(i);
#endif
		}
	}
}
void VolumetricFog::createEnvFog(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createEnvionmentFog);
	pushConstant.sampleCount = sampleCount_env;
	pushConstant.jitterStrength0 = jitterStrength_env;
	vkCmdPushConstants2(cmd, &pushInfo);
#ifdef Uniform_EnvFog_Grid
	VkExtent3D groupSize = nvvk::getGroupCounts(VkExtent3D{ envGridSize.x, envGridSize.y, envGridSize.z }, VkExtent3D{ 4, 4, 4 });
#else
	VkExtent3D groupSize = nvvk::getGroupCounts(VkExtent3D{ frustumGridSize.width, frustumGridSize.height, frustumGridSize.depth }, VkExtent3D{ 4, 4, 4 });
#endif
	
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
}
void VolumetricFog::envFogLightAttenuationEstimate(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createLightAttenuationEstimator);
	pushConstant.sampleCount = 10;
	//pushConstant.forwardSampleCount = 0;
	vkCmdPushConstants2(cmd, &pushInfo);
	vkCmdDispatch(cmd, 1, 1, 1);
}
void VolumetricFog::createFrustumAccFog(VkCommandBuffer cmd) {
	if (!pushConstant.useAccFog) return;
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createFrustumAccFog);

	pushConstant.useAccFog = 0;
	pushConstant.sampleCount = rmSampleCountSampleCount_fogAcc;
	pushConstant.randomStepping = randomStepping_fogAcc;
	pushConstant.jitterStrength0 = accJitterStrength;
	pushConstant.jitterStrength1 = interpolationJitterStrength_fogAcc;

#ifdef FOG_ACC_DELETE_NOFOGVOXEL
	vkCmdFillBuffer(cmd, GlobalInfoBuffer.buffer.buffer, offsetof(shaderio::GlobalInfo_VolumetricFog, hasFogCount), sizeof(uint32_t), 0);
	{
		VkImageSubresourceRange range = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = VK_REMAINING_MIP_LEVELS,
			.baseArrayLayer = 0,
			.layerCount = VK_REMAINING_ARRAY_LAYERS
		};

		VkClearColorValue clearColor = { .float32 = {1.0f, 0.0f, 0.0f, 0.0f} };
		vkCmdClearColorImage(cmd, volumetricFogAccResultImage.image.image, VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &range);
	}
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getHasFogVoxelInfo);
	vkCmdPushConstants2(cmd, &pushInfo);
	VkExtent3D groupSize = nvvk::getGroupCounts(VkExtent3D{ frustumGridSize.width, frustumGridSize.height, frustumGridSize.depth }, VkExtent3D{ 8, 8, 8 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
#endif

#ifdef FOG_ACC_ONE_DISPATCH
	uint32_t threadGroupCount_oneDepth_X = (frustumGridSize.width + FOG_ACC_THREADGROUP_SIZE - 1) / FOG_ACC_THREADGROUP_SIZE;
	uint32_t threadGroupCount_oneDepth_Y = (frustumGridSize.height + FOG_ACC_THREADGROUP_SIZE - 1) / FOG_ACC_THREADGROUP_SIZE;
	vkCmdFillBuffer(cmd, fogAccSyncBuffer.buffer.buffer, 0, sizeof(int) * threadGroupCount_oneDepth_X * threadGroupCount_oneDepth_Y, frustumGridSize.depth + 1);
	nvvk::cmdBufferMemoryBarrier(cmd, { fogAccSyncBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT });
#endif

#ifdef Fog_Acc_Stepping
#if defined(FOG_ACC_DIVIDE_PART)
	{
		uint32_t groupThreadCount = FOG_ACC_THREADGROUP_SIZE * FOG_ACC_THREADGROUP_SIZE;
		for (int depthIndex = 0; depthIndex < frustumGridSize.depth; ++depthIndex) {
			pushConstant.instanceIndex = depthIndex;
			vkCmdPushConstants2(cmd, &pushInfo);
			VkExtent3D groupSize = nvvk::getGroupCounts(VkExtent3D{ frustumGridSize.width, frustumGridSize.height, 1 }, VkExtent3D{ FOG_ACC_THREADGROUP_SIZE, FOG_ACC_THREADGROUP_SIZE, 1 });
			vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);

			uint32_t threadGroupCount_oneDepth_X = (frustumGridSize.width + FOG_ACC_THREADGROUP_SIZE - 1) / FOG_ACC_THREADGROUP_SIZE;
			uint32_t threadGroupCount_oneDepth_Y = (frustumGridSize.height + FOG_ACC_THREADGROUP_SIZE - 1) / FOG_ACC_THREADGROUP_SIZE;
			for (int offsetX = 0; offsetX < threadGroupCount_oneDepth_X; ++offsetX) {
				for (int offsetY = 0; offsetY < threadGroupCount_oneDepth_Y; ++offsetY) {
					uint32_t offset = (offsetY * threadGroupCount_oneDepth_X * groupThreadCount + offsetX * groupThreadCount) * sizeof(shaderio::float2);
					nvvk::BufferMemoryBarrierParams barrierParams = {
						.buffer = fogAccResultBuffer.buffer.buffer,
						.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
						.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
						.offset = offset,
						.size = groupThreadCount * sizeof(shaderio::float2)
					};
					nvvk::cmdBufferMemoryBarrier(cmd, barrierParams);
				}
			}
		}
	}
#elif defined(FOG_ACC_ONE_DISPATCH)
	vkCmdPushConstants2(cmd, &pushInfo);
	VkExtent3D groupSize = nvvk::getGroupCounts(VkExtent3D{ frustumGridSize.width, frustumGridSize.height, frustumGridSize.depth }, VkExtent3D{ FOG_ACC_THREADGROUP_SIZE, FOG_ACC_THREADGROUP_SIZE, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
#elif defined(FOG_ACC_SERIAL)
	{
		vkCmdPushConstants2(cmd, &pushInfo);
		VkExtent3D groupSize = nvvk::getGroupCounts(VkExtent3D{ frustumGridSize.width, frustumGridSize.height, 1 }, VkExtent3D{ 32, 32, 1 });
		vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
	}
#elif defined(FOG_ACC_TWO_PASS)
	{
		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createFrustumAccFog_Pass1);
		vkCmdPushConstants2(cmd, &pushInfo);
#ifdef FOG_ACC_DELETE_NOFOGVOXEL
		groupSize = nvvk::getGroupCounts(VkExtent3D{ frustumGridSize.width * frustumGridSize.height * frustumGridSize.depth, 1, 1 }, VkExtent3D{ 1024, 1, 1 });
#else
		groupSize = nvvk::getGroupCounts(VkExtent3D{ frustumGridSize.width, frustumGridSize.height, frustumGridSize.depth }, VkExtent3D{ FOG_ACC_THREADGROUP_SIZE, FOG_ACC_THREADGROUP_SIZE, 1 });
#endif
		vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createFrustumAccFog_Pass2);
		vkCmdPushConstants2(cmd, &pushInfo);
		groupSize = nvvk::getGroupCounts(VkExtent3D{ frustumGridSize.width, frustumGridSize.height, 1 }, VkExtent3D{ FOG_ACC_THREADGROUP_SIZE, FOG_ACC_THREADGROUP_SIZE, 1 });
		vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
	}
#else
	{
		for (int i = 0; i < frustumGridSize.depth; ++i) {
			pushConstant.instanceIndex = i;
			vkCmdPushConstants2(cmd, &pushInfo);

			VkExtent3D groupSize = nvvk::getGroupCounts(VkExtent3D{ frustumGridSize.width, frustumGridSize.height, 1 }, VkExtent3D{ 32, 32, 1 });
			vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);

			nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
		}
	}
#endif
#else
	vkCmdPushConstants2(cmd, &pushInfo);
	VkExtent3D groupSize = nvvk::getGroupCounts(VkExtent3D{ frustumGridSize.width, frustumGridSize.height, frustumGridSize.depth }, VkExtent3D{ 4, 4, 4 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
#endif
	pushConstant.useAccFog = 1;
}
void VolumetricFog::deferredRenderring(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	//nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_deferredRenderring);

	pushConstant.lightVP = shadowMap.pushConstant.lightVP;
	pushConstant.sampleCount = pushConstant.useAccFog == 1 ? rmSampleCount_opaque_FogAcc : rmSampleCount_opaque_noFogAcc;
	pushConstant.randomStepping = randomStepping_opaque;
	pushConstant.jitterStrength0 = interpolationJitterStrength_attenuation;
	pushConstant.jitterStrength1 = interpolationJitterStrength_L;
	//if (pushConstant.useAccFog) pushConstant.forwardSampleCount = forwardSampleCount;
	//else pushConstant.forwardSampleCount = 0;
	//pushConstant.forwardSampleCount = forwardSampleCount;
	vkCmdPushConstants2(cmd, &pushInfo);

	VkExtent2D groupSize = nvvk::getGroupCounts(gBuffers.getSize(), VkExtent2D{16, 16});
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);

	//nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}
void VolumetricFog::fogBlur(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	VkExtent2D groupSize = nvvk::getGroupCounts(gBuffers.getSize(), VkExtent2D{ 32, 32 });

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getDepthGradient);
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	pushConstant.iteration = FogFilterCount;
	for (int i = 0; i < FogFilterCount; ++i) {
		pushConstant.instanceIndex = i;
		vkCmdPushConstants2(cmd, &pushInfo);

		//vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_varianceConvolution);
		//vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
		//nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_blurFog_X);
		vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		//vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_varianceConvolution);
		//vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
		//nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_blurFog_Y);
		vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_addFog);
	vkCmdPushConstants2(cmd, &pushInfo);
	groupSize = nvvk::getGroupCounts(gBuffers.getSize(), VkExtent2D{ 32, 32 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}
void VolumetricFog::renderTransparentMaterial(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	uint32_t renderImageIndex = (uint32_t)GBuffers_VolumetricFog::eRendered;
	std::vector<VkRenderingAttachmentInfo> colorAttachments(1);
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(renderImageIndex), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	colorAttachments[0] = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	colorAttachments[0].imageView = gBuffers.getColorImageView(renderImageIndex);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.imageView = gBuffers.getDepthImageView();
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;

	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = DEFAULT_VkRect2D(gBuffers.getSize());
	renderingInfo.colorAttachmentCount = colorAttachments.size();
	renderingInfo.pColorAttachments = colorAttachments.data();
	renderingInfo.pDepthAttachment = &depthAttachment;

	vkCmdBeginRendering(cmd, &renderingInfo);

	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_TRUE;
	graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_FALSE;
	graphicsDynamicPipeline.depthStencilState.stencilTestEnable = VK_FALSE;

	graphicsDynamicPipeline.colorBlendEnables = { VK_TRUE };
	graphicsDynamicPipeline.colorWriteMasks = { VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT };
	graphicsDynamicPipeline.colorBlendEquations = { {
		.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
		.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
		.colorBlendOp = VK_BLEND_OP_ADD,
		.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
		.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
		.alphaBlendOp = VK_BLEND_OP_ADD,
	} };

	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, Application::app->getViewportSize());
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_renderTransparentMaterial, .fragment = fragmentShader_renderTransparentMaterial });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	if (pushConstant.useAccFog) pushConstant.sampleCount = 0;	//no rayMarching
	else pushConstant.sampleCount = 10;
	pushConstant.jitterStrength0 = interpolationJitterStrength_attenuation_transparent;
	pushConstant.jitterStrength1 = interpolationJitterStrength_L_transparent;
	//pushConstant.forwardSampleCount = 0;
	for (size_t i = 0; i < Application::sceneResource.instances.size(); ++i){
		shaderio::Instance instance = Application::sceneResource.instances[i];
		const shaderio::Mesh& mesh = Application::sceneResource.meshes[instance.meshIndex];
		const shaderio::TriangleMesh& triMesh = mesh.triMesh;

		shaderio::BSDFMaterial material = Application::sceneResource.materials[instance.materialIndex];
		if (material.type != shaderio::MaterialType::RoughDielectric) continue;

		pushConstant.normalMatrix = glm::transpose(glm::inverse(glm::mat3(Application::sceneResource.instances[i].transform)));
		pushConstant.instanceIndex = int(i);
		vkCmdPushConstants2(cmd, &pushInfo);

		uint32_t bufferIndex = Application::sceneResource.getMeshBufferIndex(instance.meshIndex);
		const nvvk::Buffer& v = Application::sceneResource.bDatas[bufferIndex];

		vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

		vkCmdDrawIndexed(cmd, triMesh.indices.count, 1, 0, 0, 0);
	}

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(renderImageIndex), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}

#ifndef NDEBUG
void VolumetricFog::renderVolumetricFogVoxelGrid(VkCommandBuffer cmd) {
	bool show = false;
	for (int i = 0; i < heightFogSet->fogCount; ++i) show |= heightFogSet->showFogGrid[i] == 1;
	for (int i = 0; i < fluidFogSet->fogCount; ++i) show |= fluidFogSet->getFogGridShow(i);
	for (int i = 0; i < gridFogSet->fogCount; ++i) show |= gridFogSet->getFogGridShow(i);
	show |= showEnvGrid;
	if (!show) return;

	NVVK_DBG_SCOPE(cmd);

	uint32_t renderedImage = uint32_t(GBuffers_VolumetricFog::eRendered);
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(renderedImage), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	colorAttachment.imageView = gBuffers.getColorImageView(renderedImage);

	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.imageView = gBuffers.getDepthImageView();
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;

	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = { {0, 0}, gBuffers.getSize() };
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	renderingInfo.pDepthAttachment = &depthAttachment;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdBeginRendering(cmd, &renderingInfo);

	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	graphicsDynamicPipeline.rasterizationState.lineWidth = 2.0f;
	graphicsDynamicPipeline.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_TRUE;
	graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_FALSE;
	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, gBuffers.getSize());
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_renderVoxelGrid, .fragment = fragmentShader_renderVoxelGrid });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	uint32_t wireframeMeshIndex = 0;
	const shaderio::Mesh& mesh = scene.meshes[wireframeMeshIndex];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t bufferIndex = scene.getMeshBufferIndex(wireframeMeshIndex);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];

	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));
	for (int i = 0; i < volumetricFogCount; ++i) {
		pushConstant.instanceIndex = i;

		shaderio::VolumetricFogInfo fogInfo = volumetricFogInfos[i];
		shaderio::uint3 gridSize;
		if (fogInfo.type == shaderio::VolumetricFogType::Height) {
			if (heightFogSet->showFogGrid[fogInfo.typeFogIndex] == 0) continue;
			gridSize = { 1, 1, 1 };
		}
		else if (fogInfo.type == shaderio::VolumetricFogType::Fluid) {
			if (!fluidFogSet->getFogGridShow(fogInfo.typeFogIndex)) continue;
			gridSize = fluidFogSet->getFogGridSize(fogInfo.typeFogIndex);
		}
		else if (fogInfo.type == shaderio::VolumetricFogType::Grid) {
			if (!gridFogSet->getFogGridShow(fogInfo.typeFogIndex)) continue;
			gridSize = gridFogSet->getFogGridSize(fogInfo.typeFogIndex);
		}
		else continue;

		vkCmdPushConstants2(cmd, &pushInfo);
		vkCmdDrawIndexed(cmd, triMesh.indices.count, gridSize.x * gridSize.y * gridSize.z, 0, 0, 0);
	}
#ifdef USE_ENVFOG
	if (showEnvGrid) {
		pushConstant.instanceIndex = -100;
		vkCmdPushConstants2(cmd, &pushInfo);
		vkCmdDrawIndexed(cmd, triMesh.indices.count, envGridSize.x * envGridSize.y * envGridSize.z, 0, 0, 0);
	}
#endif

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(renderedImage), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}
void VolumetricFog::renderCameraFrustum(VkCommandBuffer cmd) {
	if (!showCameraFrustum) return;
	NVVK_DBG_SCOPE(cmd);

	uint32_t renderedImage = uint32_t(GBuffers_VolumetricFog::eRendered);
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(renderedImage), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	colorAttachment.imageView = gBuffers.getColorImageView(renderedImage);

	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.imageView = gBuffers.getDepthImageView();
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;

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
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_TRUE;
	graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_FALSE;
	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, gBuffers.getSize());
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_renderCameraFrustum, .fragment = fragmentShader_renderCameraFrustum });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	uint32_t wireframeMeshIndex = 0;
	const shaderio::Mesh& mesh = scene.meshes[wireframeMeshIndex];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t bufferIndex = scene.getMeshBufferIndex(wireframeMeshIndex);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];

	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));
	//vkCmdDrawIndexed(cmd, triMesh.indices.count, pushConstant.frustumGridSize.x * pushConstant.frustumGridSize.y * pushConstant.frustumGridSize.z, 0, 0, 0);
	vkCmdDrawIndexed(cmd, triMesh.indices.count, pushConstant.frustumGridSize.z, 0, 0, 0);

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(renderedImage), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}

void VolumetricFog::test(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_test);
	vkCmdPushConstants2(cmd, &pushInfo);
	vkCmdDispatch(cmd, 1, 1, 1);
}
#endif
