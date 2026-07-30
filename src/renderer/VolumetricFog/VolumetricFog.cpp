#include "./VolumetricFog.h"

#include <common/Application/Application.h>
#include <nvgui/sky.hpp>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/default_structs.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

using namespace FzbRenderer;

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
	//if (pugi::xml_node volumetricFogCountNode = rendererNode.child("volumetricFogCount_local"))
	//	volumetricFogCount = std::stoi(volumetricFogCountNode.attribute("value").value());
	volumetricFogCount = 2;
	if (volumetricFogCount > MAX_VOLUMETRIC_FOG_COUNT) throw std::runtime_error("体积雾数量超出上限，请扩大上限");
	pushConstant.volumetricFogCount = volumetricFogCount;
	volumetricFogInfos.resize(volumetricFogCount);
	volumetricFogInfoModified.resize(volumetricFogCount);
#ifndef NDEBUG
	showVolumetricFogVoxelGrids.resize(volumetricFogCount);
#endif

	volumetricFogHeightCount = 1;
	if (volumetricFogHeightCount > MAX_HEIGHT_FOG_COUNT) throw std::runtime_error("高度雾数量超出上限，请扩大上限");
	pushConstant.heightFogCount = volumetricFogHeightCount;
	volumetricFogHeightInfos.resize(volumetricFogHeightCount);

	volumetricFogFluidCount = 1;
	if (volumetricFogFluidCount > MAX_FLUID_FOG_COUNT) throw std::runtime_error("流体数量超出上限，请扩大上限");
	pushConstant.fluidFogCount = volumetricFogFluidCount;
	volumetricFogFluidInfos.resize(volumetricFogFluidCount);
	volumetricFogFluidVoxelInfoBuffers.resize(volumetricFogFluidCount);
	volumetricFogFluidVoxelVelocityImages.resize(volumetricFogFluidCount);
	volumetricFogFluidVoxelInfoImages.resize(volumetricFogFluidCount);

	volumetricFogNoiseCount = 0;
	if (volumetricFogNoiseCount > MAX_NOISE_FOG_COUNT) throw std::runtime_error("噪声雾数量超出上限，请扩大上限");
	pushConstant.noiseFogCount = volumetricFogNoiseCount;
	volumetricFogNoiseInfos.resize(volumetricFogNoiseCount);

	/*
	volumetricFogInfos[0] = {
		.fogStartPos = {5098.0f, -69.5f, -4470.0f},
		.fogVoxelGridSize = {16, 16, 16},
		.fogVoxelSize = {2.0f, 0.05f, 2.0f},
		.color = {1.0f, 1.0f, 1.0f},
		.ambientIntensity = 0.1f,
		.absorption = { 3.0, 3.0 },
		.scattering = 0.7f,
		.phase = 0.5,
		.type = shaderio::VolumetricFogType::Noise,
		.volumetricFogTypeIndex = 0,
	};
	volumetricFogNoiseInfos[0] = {
		.cloudScale = {0.5f, 0.5f, 0.5f},
		.cloudFlowSpeed = 0.05f,
		.cloudCoverage = {0.3f, 0.2f},
		.cloudTypePreference = {1.0f, 0.0f},
		.weatherScale = 0.01f,
	};
	volumetricFogNoiseIndexMap.insert({ 0, 0 });
	*/
	volumetricFogInfos[0] = {
		.fogStartPos = {5098.0f, -69.5f, -4470.0f},
		.fogVoxelGridSize = {16, 16, 16},
		.fogVoxelSize = {2.0f, 0.1f, 2.0f},
		.color = {1.0f, 1.0f, 1.0f},
		.ambientIntensity = 0.001f,
		.absorption = { 0.01, 100.0 },
		.scattering = 0.7f,
		.phase = 0.5,
		.type = shaderio::VolumetricFogType::Height,
		.volumetricFogTypeIndex = 0,
	};
	volumetricFogHeightInfos[0] = {
		.heightScale = 1.0f,
	};
	volumetricFogNoiseIndexMap.insert({ 0, 0 });

	volumetricFogInfos[1] = {
		//.fogStartPos = {5106.0f, -69.0f, -4459.0f},
		.fogVoxelGridSize = {32, 32, 32},
		.fogVoxelSize = { 0.4, 0.2, 0.4 },
		.color = {1.0f, 1.0f, 1.0f},
		.ambientIntensity = 0.0f,
		.absorption = { 0.1, 0.3 },
		.scattering = 0.7f,
		.phase = 0.5f,
		.type = shaderio::VolumetricFogType::Fluid,
		.volumetricFogTypeIndex = 0
	};
	volumetricFogFluidInfos[0] = {
		.startUp = 0,
		.viscosity = 0.01f,
		.FIntensity = 10.0f,
		.lightAttenuationEstimator = 1.0f,
		.restoreSpeed = 10.0f,
	};
	volumetricFogFluidIndexMap.insert({ 0, 1 });
	pushConstant.fluidFogIndex = 1;

	pushConstant.randomStepping = 1;

	pushConstant.useAccFog = 0;
	pushConstant.compressionParams = 3.0f;
	pushConstant.forwardSampleCount = 0;
	frustumGridSize = { 160, 160, 80 };
	pushConstant.frustumGridSize = { frustumGridSize.width, frustumGridSize.height, frustumGridSize.depth };

	pushConstant.time = 0.0f;
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

	createVolumetricFogData();
	createDescriptorSetLayout();
	createDescriptorSet();
	createPipelineLayout();
	compileAndCreateShaders();

	Renderer::init();

	//for (size_t i = 0; i < Application::sceneResource.instances.size(); ++i) {
	//	if (!Application::sceneResource.staticInstanceIndexToInstanceSetIndex.count(i)) continue;
	//
	//	uint32_t instanceSetIndex = Application::sceneResource.staticInstanceIndexToInstanceSetIndex[i];
	//	FzbRenderer::InstanceSet* instanceSet = &Application::sceneResource.staticInstanceSets[instanceSetIndex];
	//
	//	uint32_t meshIndex = Application::sceneResource.instances[i].meshIndex;
	//	MeshInfo& meshInfo = Application::sceneResource.getMeshInfo(meshIndex);
	//	instanceSet->aabb = meshInfo.getAABB(Application::sceneResource.instances[i].transform, false);
	//}

	if (volumetricFogFluidCount > 0) {
		std::pair<uint32_t, uint32_t> instanceSetPair = Application::sceneResource.instanceIDToInstanceSet["mainCharacter"];
		InstanceSet mainCharacter = Application::sceneResource.getInstanceSet((InstanceType)instanceSetPair.first, instanceSetPair.second);
		shaderio::AABB mainCharacterAABB;
		mainCharacterAABB.minimum = { FLT_MAX, FLT_MAX, FLT_MAX };
		mainCharacterAABB.maximum = -mainCharacterAABB.minimum;
		for (int i = 0; i < mainCharacter.childInstances.size(); ++i) {
			shaderio::Instance childInstance = mainCharacter.childInstances[i];
			uint32_t meshIndex = childInstance.meshIndex;
			MeshInfo& meshInfo = Application::sceneResource.getMeshInfo(meshIndex);
			shaderio::AABB childAABB = meshInfo.getAABB();
			{
				mainCharacterAABB.minimum.x = std::min(mainCharacterAABB.minimum.x, childAABB.minimum.x);
				mainCharacterAABB.minimum.y = std::min(mainCharacterAABB.minimum.y, childAABB.minimum.y);
				mainCharacterAABB.minimum.z = std::min(mainCharacterAABB.minimum.z, childAABB.minimum.z);
				mainCharacterAABB.maximum.x = std::max(mainCharacterAABB.maximum.x, childAABB.maximum.x);
				mainCharacterAABB.maximum.y = std::max(mainCharacterAABB.maximum.y, childAABB.maximum.y);
				mainCharacterAABB.maximum.z = std::max(mainCharacterAABB.maximum.z, childAABB.maximum.z);
			}
		}
		fluidLocalStartPos = (mainCharacterAABB.minimum + mainCharacterAABB.maximum) * 0.5f;
		fluidLocalStartPos.y = 0.0f;
	}

	pushConstant.cameraNearPlane = Application::sceneResource.cameraManip->getClipPlanes().x;
	pushConstant.cameraFarPlane = 2000.0f;	// Application::sceneResource.cameraManip->getClipPlanes().y * 0.02f;
	pushConstant.tanCameraFov_2 = glm::tan(glm::radians(Application::sceneResource.cameraManip->getFov() * 0.5f));
	pushConstant.aspectRatio = Application::sceneResource.cameraManip->getAspectRatio();
}
void VolumetricFog::clean() {
	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, vertexShader_createGBuffer, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_createGBuffer, nullptr);
	vkDestroyShaderEXT(device, computeShader_getVisibleVolumetricFog, nullptr);

	vkDestroyShaderEXT(device, computeShader_initVolumetricFogFluid, nullptr);
	vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_A, nullptr);
	vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_D, nullptr);
	vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_F, nullptr);
	vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_P, nullptr);
	vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_S, nullptr);

	vkDestroyShaderEXT(device, computeShader_createEnvionmentFog, nullptr);

	vkDestroyShaderEXT(device, computeShader_createLightAttenuationEstimator, nullptr);
	vkDestroyShaderEXT(device, computeShader_createFrustumAccFog, nullptr);
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

	volumetricFogHeightInfoBuffer.clean();

	volumetricFogFluidInfoBuffer.clean();
	for (int i = 0; i < volumetricFogFluidCount; ++i) {
		volumetricFogFluidVoxelVelocityImages[i].clean();
		volumetricFogFluidVoxelInfoBuffers[i].clean();
		volumetricFogFluidVoxelInfoImages[i].clean();
	}

	volumetricFogNoiseInfoBuffer.clean();

	volumetricFogAttenuationImage.clean();
	volumetricFogLImage.clean();
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

		if (ImGui::CollapsingHeader("Frustum Setting", ImGuiTreeNodeFlags_DefaultOpen)) {
			UIModified |= ImGui::DragFloat("Compression Params ", (float*)&pushConstant.compressionParams, 0.1f, 0.0f, 10);
			if (ImGui::Checkbox("Show Camera Frustum ", (bool*)&showCameraFrustum)) {
				UIModified = true;
				showCameraInfo = Application::sceneResource.sceneInfo;
			}
		}
		
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
			if (ImGui::Checkbox("Start Up", (bool*)&pushConstant.useAccFog)) {
				if (pushConstant.useAccFog) interpolationJitterStrength = 30.0f;
				else interpolationJitterStrength = 10.0f;
				UIModified = true;
			}
			UIModified |= ImGui::Checkbox("Random Stepping Fog Acc", (bool*)&randomStepping_fogAcc);
			UIModified |= ImGui::DragInt("Fog ACC Sample Count ", (int*)&rmSampleCountSampleCount_fogAcc, 1, 1, 50);
			UIModified |= ImGui::DragFloat("Acc Jitter Strength", (float*)&accJitterStrength, 1, 0, 100);
			UIModified |= ImGui::DragFloat("Acc Interpolation Jitter Strength", (float*)&interpolationJitterStrength_fogAcc, 1, 0, 100);
		}
		if (ImGui::CollapsingHeader("opaque render Setting", ImGuiTreeNodeFlags_DefaultOpen)) {
			UIModified |= ImGui::Checkbox("Random Stepping Opaque", (bool*)&randomStepping_opaque);
			UIModified |= ImGui::DragInt("No Fog Acc RayMarching Sample Count ", (int*)&rmSampleCount_opaque_noFogAcc, 1, 1, 200);
			UIModified |= ImGui::DragInt("Fog Acc RayMarching Sample Count ", (int*)&rmSampleCount_opaque_FogAcc, 1, 1, 50);
			UIModified |= ImGui::DragInt("Forward Sample Number", (int*)&forwardSampleCount, 1, 0, 100);
			UIModified |= ImGui::DragFloat("Interpolation Jitter Strength", (float*)&interpolationJitterStrength, 1, 0, 100);
		}

		for (int i = 0; i < volumetricFogCount; ++i) {
			volumetricFogInfoModified[i] = false;
			if (ImGui::CollapsingHeader(std::string("Volumetric Fog Properties " + std::to_string(i)).c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
				ImGui::BeginDisabled(true);
				volumetricFogInfoModified[i] |= ImGui::Combo(std::string("Type " + std::to_string(i)).c_str(), (int*)&volumetricFogInfos[i].type, fogTypeItems, IM_ARRAYSIZE(fogTypeItems));
				ImGui::EndDisabled();

				if (volumetricFogInfos[i].type != shaderio::VolumetricFogType::Fluid)
					volumetricFogInfoModified[i] |= ImGui::DragFloat3(std::string("Volumetric Fog Start Pos " + std::to_string(i)).c_str(), (float*)&volumetricFogInfos[i].fogStartPos);

				ImGui::BeginDisabled(true);
				bool change = ImGui::DragInt3(std::string("Volumetric Fog Voxel Grid Size " + std::to_string(i)).c_str(), (int*)&volumetricFogInfos[i].fogVoxelGridSize);
				ImGui::EndDisabled();
				if (change) {
					//createVolumetricFogData();
					volumetricFogInfoModified[i] = true;
				}

				volumetricFogInfoModified[i] |= ImGui::DragFloat3(std::string("Volumetric Fog Voxel Size " + std::to_string(i)).c_str(), (float*)&volumetricFogInfos[i].fogVoxelSize);

				volumetricFogInfoModified[i] |= ImGui::DragFloat3(std::string("Volumetric Fog Color " + std::to_string(i)).c_str(), (float*)&volumetricFogInfos[i].color);
				volumetricFogInfoModified[i] |= ImGui::DragFloat(std::string("Ambient Intensity " + std::to_string(i)).c_str(), (float*)&volumetricFogInfos[i].ambientIntensity, 0.1f, 0.0f, 10.0f);

				volumetricFogInfoModified[i] |= ImGui::DragFloat2(std::string("Extinction Coefficient " + std::to_string(i)).c_str(), (float*)&volumetricFogInfos[i].absorption, 0.1f, 0.0f);
				volumetricFogInfoModified[i] |= ImGui::DragFloat(std::string("Scatter Coefficient " + std::to_string(i)).c_str(), (float*)&volumetricFogInfos[i].scattering, 0.1f, 0.0f, 1.0f);
				volumetricFogInfoModified[i] |= ImGui::DragFloat(std::string("Asymmetric Parameters " + std::to_string(i)).c_str(), (float*)&volumetricFogInfos[i].phase, 0.1f, -1.0f, 1.0f);

				if (volumetricFogInfos[i].type == shaderio::VolumetricFogType::Height) {
					if (ImGui::CollapsingHeader(std::string("Height Properties " + std::to_string(i)).c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
						volumetricFogInfoModified[i] |= 
							ImGui::DragFloat(std::string("Height Attenuation " + std::to_string(i)).c_str(), (float*)&volumetricFogHeightInfos[heightFogIndex].heightScale, 1.0f, 0.0f, 1000.0f);
					}
					++heightFogIndex;
				}
				else if(volumetricFogInfos[i].type == shaderio::VolumetricFogType::Fluid) {
					if (ImGui::CollapsingHeader(std::string("Fluid Properties " + std::to_string(i)).c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
						volumetricFogInfoModified[i] |= ImGui::Checkbox(std::string("Start Up " + std::to_string(i)).c_str(), (bool*)&volumetricFogFluidInfos[fluidFogIndex].startUp);
						volumetricFogInfoModified[i] |= ImGui::DragFloat(std::string("Viscosity " + std::to_string(i)).c_str(), (float*)&volumetricFogFluidInfos[fluidFogIndex].viscosity, 0.1f, 0.0f, 1.0f);
						volumetricFogInfoModified[i] |= ImGui::DragFloat(std::string("F Intensity " + std::to_string(i)).c_str(), (float*)&volumetricFogFluidInfos[fluidFogIndex].FIntensity, 1.0f, 0.0f, 100.0f);
						volumetricFogInfoModified[i] |= ImGui::DragFloat(std::string("Restore Speed " + std::to_string(i)).c_str(), (float*)&volumetricFogFluidInfos[fluidFogIndex].restoreSpeed, 1.0f, 0.0f, 100.0f);
					}
					++fluidFogIndex;
				}
				else if (volumetricFogInfos[i].type == shaderio::VolumetricFogType::Noise) {
					if (ImGui::CollapsingHeader(std::string("Noise Properties " + std::to_string(i)).c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
						volumetricFogInfoModified[i] |= ImGui::DragFloat3(std::string("Cloud Scale " + std::to_string(i)).c_str(), (float*)&volumetricFogNoiseInfos[noiseFogIndex].cloudScale, 0.01f, 0.0f, 2.0f);
						volumetricFogInfoModified[i] |= ImGui::DragFloat(std::string("Cloud Flow Speed " + std::to_string(i)).c_str(), (float*)&volumetricFogNoiseInfos[noiseFogIndex].cloudFlowSpeed, 0.1f, 0.0f, 10.0f);
						volumetricFogInfoModified[i] |= ImGui::DragFloat2(std::string("Cloud Coverage " + std::to_string(i)).c_str(), (float*)&volumetricFogNoiseInfos[noiseFogIndex].cloudCoverage, 0.1f, 0.0f, 2.0f);
						volumetricFogInfoModified[i] |= ImGui::DragFloat(std::string("Cloud Type Preference " + std::to_string(i)).c_str(), (float*)&volumetricFogNoiseInfos[noiseFogIndex].cloudTypePreference, 0.1f, 0.0f, 1.0f);
						volumetricFogInfoModified[i] |= ImGui::DragFloat(std::string("Weather Scale " + std::to_string(i)).c_str(), (float*)&volumetricFogNoiseInfos[noiseFogIndex].weatherScale, 0.01f, 0.0f, 2.0f);
					}
					++noiseFogIndex;
				}

				volumetricFogInfoModified[i] |= ImGui::Checkbox(std::string("show voxel grid " + std::to_string(i)).c_str(), (bool*)&showVolumetricFogVoxelGrids[i]);
			}
			UIModified |= volumetricFogInfoModified[i];
		}
	}
	ImGui::End();

	shadowMap.uiRender();

	/*
	// 鼠标点击施加力
	if ((ImGui::IsMouseDown(ImGuiMouseButton_Left) || ImGui::IsMouseClicked(ImGuiMouseButton_Left)) && !ImGui::IsAnyItemHovered()) {
		ImVec2 mousePos = ImGui::GetMousePos();
		ImVec2 vpPos = Application::viewportScreenPos;
		ImVec2 vpSize = Application::viewportContentSize;

		// 使用 viewport 局部坐标
		float mouseRelX = mousePos.x - vpPos.x;
		float mouseRelY = mousePos.y - vpPos.y;
		if (vpSize.x <= 0 || vpSize.y <= 0 ||
			mouseRelX < 0 || mouseRelX > vpSize.x ||
			mouseRelY < 0 || mouseRelY > vpSize.y) return; // 鼠标不在 viewport 内

		float ndcX = (2.0f * mouseRelX / vpSize.x) - 1.0f;
		float ndcY = (2.0f * mouseRelY / vpSize.y) - 1.0f;  // OpenGL NDC：投影矩阵已含Vulkan Y翻转，这里不要再翻

		const glm::mat4& viewMat = Application::sceneResource.sceneInfo.viewMatrix;
		const glm::mat4& projMat = Application::sceneResource.s;
		glm::mat4 invProj = glm::inverse(projMat);
		glm::mat4 invView = glm::inverse(viewMat);

		glm::vec4 clipNear(ndcX, ndcY, 0.0f, 1.0f);
		glm::vec4 clipFar(ndcX, ndcY, 1.0f, 1.0f);
		glm::vec4 viewNear = invProj * clipNear; viewNear /= viewNear.w;
		glm::vec4 viewFar = invProj * clipFar; viewFar /= viewFar.w;
		glm::vec3 worldNear = glm::vec3(invView * viewNear);
		glm::vec3 worldFar = glm::vec3(invView * viewFar);
		glm::vec3 rayDir = glm::normalize(worldFar - worldNear);

		// 射线与流体 AABB 求交
		bool hit = false;
		for (int i = 0; i < volumetricFogCount; ++i) {
			if (volumetricFogInfos[i].type != shaderio::VolumetricFogType::Fluid) continue;
			if (volumetricFogFluidInfos[volumetricFogInfos[i].volumetricFogTypeIndex].startUp == 0) continue;

			glm::vec3 fogMin = glm::vec3(volumetricFogInfos[i].fogStartPos.x, volumetricFogInfos[i].fogStartPos.y, volumetricFogInfos[i].fogStartPos.z);
			glm::vec3 fogMax = fogMin + glm::vec3(volumetricFogInfos[i].fogVoxelGridSize.x * volumetricFogInfos[i].fogVoxelSize.x,
												   volumetricFogInfos[i].fogVoxelGridSize.y * volumetricFogInfos[i].fogVoxelSize.y,
												   volumetricFogInfos[i].fogVoxelGridSize.z * volumetricFogInfos[i].fogVoxelSize.z);

			float tMin = 0.0f, tMax = std::numeric_limits<float>::max();
			for (int axis = 0; axis < 3; ++axis) {
				float invD = 1.0f / (std::abs(rayDir[axis]) > 1e-8f ? rayDir[axis] : (rayDir[axis] >= 0 ? 1e-8f : -1e-8f));
				float t0 = (fogMin[axis] - worldNear[axis]) * invD;
				float t1 = (fogMax[axis] - worldNear[axis]) * invD;
				if (invD < 0.0f) std::swap(t0, t1);
				tMin = std::max(tMin, t0);
				tMax = std::min(tMax, t1);
				if (tMin > tMax) break;
			}

			if (tMin <= tMax && tMax >= 0.0f) {
				float tHit = std::max(tMin, 0.0f);
				mouseForcePosition = worldNear + rayDir * tHit;
				mouseForceStrength = 10.0f;
				hit = true;
				break;
			}
		}
	}
	*/
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
	imageWrite = staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eRenderedFogImage, 0, 0, 1);
	write.append(imageWrite, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eRenderedFog]);

	imageWrite = staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eDepthGradientImage, 0, 0, 1);
	write.append(imageWrite, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eDepthGradient]);

	imageWrite = staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFogVarianceImages, 0, 0, 2);
	std::vector<nvvk::Image> vairanceImages = { gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eFogVariance0],  gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eFogVariance1] };
	write.append(imageWrite, vairanceImages.data());

	imageWrite = staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFilterImages, 0, 0, 2);
	std::vector<nvvk::Image> filterImages = { gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eFilter0],  gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eFilter1] };
	write.append(imageWrite, filterImages.data());
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

	//pushConstant.mouseForcePosition = { mouseForcePosition.x, mouseForcePosition.y, mouseForcePosition.z };
	//pushConstant.mouseForceStrength = mouseForceStrength;
	//pushConstant.mouseForceRadius = mouseForceRadius;
	//
	//// 力衰减
	//if (mouseForceStrength > 0.0f) {
	//	mouseForceStrength *= 0.8f;
	//	if (mouseForceStrength < 0.01f) mouseForceStrength = 0.0f;
	//}
	if (volumetricFogFluidCount > 0) {
		std::pair<uint32_t, uint32_t> instanceSetPair = Application::sceneResource.instanceIDToInstanceSet["mainCharacter"];
		InstanceSet mainCharacter = Application::sceneResource.getInstanceSet((InstanceType)instanceSetPair.first, instanceSetPair.second);
		shaderio::float3 fogStartPos = mainCharacter.transform * shaderio::float4(fluidLocalStartPos, 1.0f);
		shaderio::float3 fogStartPos_lastTime = mainCharacter.transform_lastTime * shaderio::float4(fluidLocalStartPos, 1.0f);

		uint32_t fogIndex = volumetricFogFluidIndexMap[0];
		shaderio::VolumetricFogInfo fogInfo = volumetricFogInfos[fogIndex];
		fogStartPos -= shaderio::float3(0.5f, 0.0f, 0.5f) * (shaderio::float3)fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize;
		fogStartPos -= 3.0f * fogInfo.fogVoxelSize.y;
		volumetricFogInfos[fogIndex].fogStartPos = fogStartPos;

		fogStartPos_lastTime -= shaderio::float3(0.5f, 0.0f, 0.5f) * (shaderio::float3)fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize;
		fogStartPos_lastTime -= 3.0f * fogInfo.fogVoxelSize.y;
		volumetricFogFluidInfos[0].fogStartPos_lastTime = fogStartPos_lastTime;
	}

	#ifdef USE_SVGF
	svgf.preRender();
	#endif

	#ifdef USE_TAA
	taa.preRender();
	#endif

	shadowMap.preRender();

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
	initVolumetricFogFluid(cmd);
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
	//-----------------------------------------------------------高度----------------------------------------------------
	if (volumetricFogHeightCount > 0) {
		volumetricFogHeightInfoBuffer = FzbRenderer::Buffer("volumetricFogHeightInfoBuffer", false);
		volumetricFogHeightInfoBuffer.init({
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = sizeof(shaderio::HeightFogInfo) * volumetricFogHeightCount,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
			});
	}
	//-----------------------------------------------------------流体----------------------------------------------------
	if (volumetricFogFluidCount > 0) {
		volumetricFogFluidInfoBuffer = FzbRenderer::Buffer("volumetricFogFluidInfoBuffer", false);
		volumetricFogFluidInfoBuffer.init({
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = sizeof(shaderio::FluidFogInfo) * volumetricFogFluidCount,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
			});
	}
	for (int i = 0; i < volumetricFogFluidCount; ++i) {
		uint32_t fogIndex = volumetricFogFluidIndexMap[i];

		createVolumetricFogImage(volumetricFogFluidVoxelVelocityImages[i], volumetricFogInfos[fogIndex].fogVoxelGridSize);

		volumetricFogFluidVoxelInfoBuffers[i] = FzbRenderer::Buffer("volumetricFogFluidVoxelInfoBuffer" + std::to_string(i), false);
		volumetricFogFluidVoxelInfoBuffers[i].init({
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = sizeof(shaderio::VolumetricFogFluidVoxelInfo) * volumetricFogInfos[fogIndex].fogVoxelGridSize.x * volumetricFogInfos[fogIndex].fogVoxelGridSize.y * volumetricFogInfos[fogIndex].fogVoxelGridSize.z,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
			});

		createVolumetricFogImage(volumetricFogFluidVoxelInfoImages[i], volumetricFogInfos[fogIndex].fogVoxelGridSize);

		//image init的时候会全部置0
		//vkCmdFillBuffer(cmd, volumetricFogVoxelInfoBuffers[i].buffer.buffer, 0, volumetricFogVoxelInfoBuffers[i].allocMemSize, 0);
	}
	//-----------------------------------------------------------噪声----------------------------------------------------
	if (volumetricFogNoiseCount > 0) {
		volumetricFogNoiseInfoBuffer = FzbRenderer::Buffer("volumetricFogNoiseInfoBuffer", false);
		volumetricFogNoiseInfoBuffer.init({
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = sizeof(shaderio::NoiseFogInfo) * volumetricFogNoiseCount,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
			});
	}
	//-------------------------------------------------------------------------------------------------------------------
	{
		createVolumetricFogImage(volumetricFogAttenuationImage, { frustumGridSize.width, frustumGridSize.height, frustumGridSize.depth }, true);
		createVolumetricFogImage(volumetricFogLImage, { frustumGridSize.width, frustumGridSize.height, frustumGridSize.depth }, true);

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
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogInfosBuffer,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	}
	//----------------------------------------------------------------高度-----------------------------------------------------------------------------
	{
		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogHeightInfoBuffer,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = volumetricFogHeightCount,
		.stageFlags = VK_SHADER_STAGE_ALL });
	}
	//----------------------------------------------------------------流体-----------------------------------------------------------------------------
	{
		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidInfoBuffer,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = volumetricFogFluidCount,
		.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelInfoBuffer,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = volumetricFogFluidCount,
		.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelVelocityImage,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = volumetricFogFluidCount,
			.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelVelocityImage_sample,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = volumetricFogFluidCount,
			.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelInfoImages,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = volumetricFogFluidCount,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelInfoImages_sampler,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = volumetricFogFluidCount,
			.stageFlags = VK_SHADER_STAGE_ALL });
	}
	//----------------------------------------------------------------噪声-----------------------------------------------------------------------------
	{
		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogNoiseInfoBuffer,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = volumetricFogNoiseCount,
		.stageFlags = VK_SHADER_STAGE_ALL });
	}
	//---------------------------------------------------------------------------------------------------------------------------------------------
	{
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogAttenuationImage,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogAttenuationImage_sample,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogLImage,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogLImage_sample,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eEnvVolumetricFogInfoImage,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eEnvVolumetricFogInfoImage_sample,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
	}
	//-----------------------------------------------------------------Blur----------------------------------------------------------------------------
#ifdef BLUR_FOG
	{
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eRenderedFogImage,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eDepthGradientImage,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eFogVarianceImages,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 2,
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

	LOGI("Fzb PathGuiding static descriptor layout created\n");
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
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogInfosBuffer, 0, 0, 1);
		write.append(volumetricFogInfoWrite, volumetricFogInfosBuffer.buffer);
	}
	//------------------------------------------------------------------高度---------------------------------------------------------------------------
	if (volumetricFogHeightCount > 0) {
		VkWriteDescriptorSet	volumetricFogHeightWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogHeightInfoBuffer, 0, 0, 1);
		write.append(volumetricFogHeightWrite, volumetricFogHeightInfoBuffer.buffer);
	}
	//------------------------------------------------------------------流体---------------------------------------------------------------------------
	if (volumetricFogFluidCount > 0) {
		std::vector<nvvk::Buffer> volumetricFogBuffer_nvvk(volumetricFogFluidCount);
		nvvk::Buffer* volumetricFogBufferPtr = nullptr;
		auto getNvvkBufferPtr = [&](std::vector<FzbRenderer::Buffer> buffers) {
			for (int i = 0; i < volumetricFogFluidCount; ++i) volumetricFogBuffer_nvvk[i] = buffers[i].buffer;
			volumetricFogBufferPtr = volumetricFogBuffer_nvvk.data();
		};
		std::vector<nvvk::Image> volumetricFogImage_nvvk(volumetricFogFluidCount);
		nvvk::Image* volumetricFogImagePtr;
		auto getNvvkImagePtr = [&](std::vector<FzbRenderer::Image> images) {
			for (int i = 0; i < volumetricFogFluidCount; ++i) volumetricFogImage_nvvk[i] = images[i].image;
			volumetricFogImagePtr = volumetricFogImage_nvvk.data();
		};

		VkWriteDescriptorSet	volumetricFogFluidWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidInfoBuffer, 0, 0, 1);
		write.append(volumetricFogFluidWrite, volumetricFogFluidInfoBuffer.buffer);

		volumetricFogFluidWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelInfoBuffer, 0, 0, volumetricFogFluidCount);
		getNvvkBufferPtr(volumetricFogFluidVoxelInfoBuffers);
		write.append(volumetricFogFluidWrite, volumetricFogBufferPtr);

		volumetricFogFluidWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelVelocityImage, 0, 0, volumetricFogFluidCount);
		getNvvkImagePtr(volumetricFogFluidVoxelVelocityImages);
		write.append(volumetricFogFluidWrite, volumetricFogImagePtr);

		volumetricFogFluidWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelVelocityImage_sample, 0, 0, volumetricFogFluidCount);
		write.append(volumetricFogFluidWrite, volumetricFogImagePtr);

		volumetricFogFluidWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelInfoImages, 0, 0, volumetricFogFluidCount);
		getNvvkImagePtr(volumetricFogFluidVoxelInfoImages);
		write.append(volumetricFogFluidWrite, volumetricFogImagePtr);

		volumetricFogFluidWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelInfoImages_sampler, 0, 0, volumetricFogFluidCount);
		write.append(volumetricFogFluidWrite, volumetricFogImagePtr);
	}
	//------------------------------------------------------------------噪声---------------------------------------------------------------------------
	if (volumetricFogNoiseCount > 0) {
		VkWriteDescriptorSet	volumetricFogNoiseWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogNoiseInfoBuffer, 0, 0, 1);
		write.append(volumetricFogNoiseWrite, volumetricFogNoiseInfoBuffer.buffer);
	}
	//------------------------------------------------------------------------------------------------------------------------------------------------
	{
		VkWriteDescriptorSet	attenuationImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogAttenuationImage, 0, 0, 1);
		write.append(attenuationImageWrite, volumetricFogAttenuationImage.image);

		attenuationImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogAttenuationImage_sample, 0, 0, 1);
		write.append(attenuationImageWrite, volumetricFogAttenuationImage.image);

		attenuationImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogLImage, 0, 0, 1);
		write.append(attenuationImageWrite, volumetricFogLImage.image);

		attenuationImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogLImage_sample, 0, 0, 1);
		write.append(attenuationImageWrite, volumetricFogLImage.image);

		VkWriteDescriptorSet	envCalInfoImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eEnvVolumetricFogInfoImage, 0, 0, 1);
		write.append(envCalInfoImageWrite, envVolumetricFogInfoImage.image);

		envCalInfoImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eEnvVolumetricFogInfoImage_sample, 0, 0, 1);
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

		vkDestroyShaderEXT(device, computeShader_initVolumetricFogFluid, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_initVolumetricFogFluid";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_initVolumetricFogFluid);
		NVVK_DBG_NAME(computeShader_initVolumetricFogFluid);
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
	//-------------------------------------------Step6-------------------------------------------
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
	//-------------------------------------------Step7-------------------------------------------
	{
		shaderSource = shaderPath / "step7_createFrustumAccFog.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_createFrustumAccFog, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		#ifdef Fog_Acc_Stepping
		shaderInfo.pName = "computeMain_createFrustumAccFog_Stepping";
		#else
		shaderInfo.pName = "computeMain_createFrustumAccFog";
		#endif
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createFrustumAccFog);
		NVVK_DBG_NAME(computeShader_createFrustumAccFog);
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

			vkDestroyShaderEXT(device, computeShader_varianceConvolution, nullptr);
			shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			shaderInfo.nextStage = 0;
			shaderInfo.pName = "computeMain_varianceConvolution";
			shaderInfo.codeSize = shaderCode.codeSize;
			shaderInfo.pCode = shaderCode.pCode;
			vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_varianceConvolution);
			NVVK_DBG_NAME(computeShader_varianceConvolution);

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
	if(volumetricFogHeightCount > 0) nvvk::cmdBufferMemoryBarrier(cmd, { volumetricFogHeightInfoBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT });
	if(volumetricFogFluidCount > 0) nvvk::cmdBufferMemoryBarrier(cmd, { volumetricFogFluidInfoBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT });
	if(volumetricFogNoiseCount > 0) nvvk::cmdBufferMemoryBarrier(cmd, { volumetricFogNoiseInfoBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT });


	bool heightModified = false, fluidModified = false, noiseModified = false;
	if (volumetricFogFluidCount > 0) {
		uint32_t fogIndex = volumetricFogFluidIndexMap[0];
		shaderio::VolumetricFogInfo fogInfo = volumetricFogInfos[fogIndex];
		uint32_t typeFogIndex = fogInfo.volumetricFogTypeIndex;

		fluidModified = true;
		vkCmdUpdateBuffer(cmd, volumetricFogInfosBuffer.buffer.buffer, sizeof(shaderio::VolumetricFogInfo) * fogIndex, sizeof(shaderio::VolumetricFogInfo), &volumetricFogInfos[fogIndex]);
		vkCmdUpdateBuffer(cmd, volumetricFogFluidInfoBuffer.buffer.buffer, sizeof(shaderio::FluidFogInfo) * typeFogIndex, sizeof(shaderio::FluidFogInfo), &volumetricFogFluidInfos[typeFogIndex]);
	}

	for (int i = 0; i < volumetricFogCount; ++i) {
		if (!volumetricFogInfoModified[i] && pushConstant.time > 0.0f) continue;
		vkCmdUpdateBuffer(cmd, volumetricFogInfosBuffer.buffer.buffer, sizeof(shaderio::VolumetricFogInfo) * i, sizeof(shaderio::VolumetricFogInfo), &volumetricFogInfos[i]);

		shaderio::VolumetricFogInfo fogInfo = volumetricFogInfos[i];
		uint32_t typeFogIndex = fogInfo.volumetricFogTypeIndex;
		if (fogInfo.type == shaderio::VolumetricFogType::Height) {
			heightModified = true;
			vkCmdUpdateBuffer(cmd, volumetricFogHeightInfoBuffer.buffer.buffer, sizeof(shaderio::HeightFogInfo) * typeFogIndex, sizeof(shaderio::HeightFogInfo), &volumetricFogHeightInfos[typeFogIndex]);
		}
		else if (fogInfo.type == shaderio::VolumetricFogType::Fluid) {
			//fluidModified = true;
			//vkCmdUpdateBuffer(cmd, volumetricFogFluidInfoBuffer.buffer.buffer, sizeof(shaderio::FluidFogInfo) * typeFogIndex, sizeof(shaderio::FluidFogInfo), &volumetricFogFluidInfos[typeFogIndex]);
		}
		else if (fogInfo.type == shaderio::VolumetricFogType::Noise) {
			noiseModified = true;
			vkCmdUpdateBuffer(cmd, volumetricFogNoiseInfoBuffer.buffer.buffer, sizeof(shaderio::NoiseFogInfo) * typeFogIndex, sizeof(shaderio::NoiseFogInfo), &volumetricFogNoiseInfos[typeFogIndex]);
		}
	}
	if (volumetricFogCount > 0) nvvk::cmdBufferMemoryBarrier(cmd, { volumetricFogInfosBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT });
	if (heightModified) nvvk::cmdBufferMemoryBarrier(cmd, { volumetricFogHeightInfoBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT });
	if (fluidModified) nvvk::cmdBufferMemoryBarrier(cmd, { volumetricFogFluidInfoBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT });
	if (noiseModified) nvvk::cmdBufferMemoryBarrier(cmd, { volumetricFogNoiseInfoBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT });

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
void VolumetricFog::initVolumetricFogFluid(VkCommandBuffer cmd) {
	if (volumetricFogFluidCount == 0) return;
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);
	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	for (int i = 0; i < volumetricFogFluidCount; ++i) {
		uint32_t fogIndex = volumetricFogFluidIndexMap[i];
		pushConstant.instanceIndex = fogIndex;
		shaderio::VolumetricFogInfo fogInfo = volumetricFogInfos[fogIndex];
		shaderio::FluidFogInfo fluidFogInfo = volumetricFogFluidInfos[i];
		if (fluidFogInfo.startUp == 0) continue;
		VkExtent3D groupSize = nvvk::getGroupCounts(VkExtent3D{ fogInfo.fogVoxelGridSize.x, fogInfo.fogVoxelGridSize.y, fogInfo.fogVoxelGridSize.z }, VkExtent3D{ 4, 4, 4 });

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initVolumetricFogFluid);
		float time = pushConstant.time;
		if (volumetricFogInfoModified[fogIndex]) pushConstant.time = 0.0f;
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
		//pushConstant.volumetricFogFluidIndex = -1;
		pushConstant.instanceVelocity = shaderio::float3(0.0f);
		if (Application::sceneResource.periodInstanceIndexToInstanceSetIndex.count(i)) {
			uint32_t instanceSetIndex = Application::sceneResource.periodInstanceIndexToInstanceSetIndex[i];
			instanceSet = &Application::sceneResource.periodInstanceSets[instanceSetIndex];

			//先不考虑旋转带来的力
			shaderio::float3 pos0 = shaderio::float3(instanceSet->transform * shaderio::float4(1.0f, 1.0f, 1.0f, 1.0f));
			shaderio::float3 pos1 = shaderio::float3(instanceSet->transform_lastTime * shaderio::float4(1.0f, 1.0f, 1.0f, 1.0f));
			pushConstant.instanceVelocity = (pos0 - pos1) / pushConstant.dt;
			//pushConstant.transofrm_lastTime = instanceSet->transform_lastTime;

			//pushConstant.volumetricFogFluidIndex = 1;	//表示会与流体进行交互，因此几何需要与流体进行判断
		}
		else {		
			uint32_t instanceSetIndex = Application::sceneResource.staticInstanceIndexToInstanceSetIndex[i];
			instanceSet = &Application::sceneResource.staticInstanceSets[instanceSetIndex];
			//静态的先用AABB与流体AABB进行判断，如果不相交，则几何无需与流体进行判断; 动态的每帧需要重新计算AABB，还不如直接FS中进行判断呢
			//shaderio::AABB instanceAABB = instanceSet->aabb;
			//
			//for (int j = 0; j < volumetricFogFluidCount; ++j) {
			//	uint32_t fluidFogIndex = volumetricFogFluidIndexMap[j];
			//	shaderio::VolumetricFogInfo fogInfo = volumetricFogInfos[fluidFogIndex];
			//	shaderio::AABB fogAABB = { .minimum = fogInfo.fogStartPos, .maximum = fogInfo.fogStartPos + (shaderio::float3)fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize };
			//
			//	if (instanceAABB.maximum.x >= fogAABB.minimum.x && instanceAABB.minimum.x <= fogAABB.maximum.x &&
			//		instanceAABB.maximum.y >= fogAABB.minimum.y && instanceAABB.minimum.y <= fogAABB.maximum.y &&
			//		instanceAABB.maximum.z >= fogAABB.minimum.z && instanceAABB.minimum.z <= fogAABB.maximum.z
			//		) {
			//		pushConstant.volumetricFogFluidIndex = 1;
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
		auto barrierBuffer = [&](FzbRenderer::Buffer buffer) {
			VkBufferMemoryBarrier2 b = nvvk::makeBufferMemoryBarrier({
				.buffer = buffer.buffer.buffer,
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
			barrierImage(volumetricFogFluidVoxelVelocityImages[fluidIndex].image.image);
			barrierBuffer(volumetricFogFluidVoxelInfoBuffers[fluidIndex]);
			barrierImage(volumetricFogFluidVoxelInfoImages[fluidIndex].image.image);
		};

		auto barrierVolumePingPong = [&](int idx) {
			barrierBuffer(volumetricFogFluidVoxelInfoBuffers[idx]);
		};

		for (int i = 0; i < volumetricFogFluidCount; ++i) {
			uint32_t fogIndex = volumetricFogFluidIndexMap[i];
			pushConstant.instanceIndex = fogIndex;
			shaderio::VolumetricFogInfo fogInfo = volumetricFogInfos[fogIndex];
			shaderio::FluidFogInfo fluidFogInfo = volumetricFogFluidInfos[i];
			if (fluidFogInfo.startUp == 0) continue;
			groupSize = nvvk::getGroupCounts(VkExtent3D{ fogInfo.fogVoxelGridSize.x, fogInfo.fogVoxelGridSize.y, fogInfo.fogVoxelGridSize.z }, VkExtent3D{ 4, 4, 4 });

			// --- Stage A: Advection ---
			vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createVolumetricFog_Fluid_A);
			vkCmdPushConstants2(cmd, &pushInfo);
			vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
			barrierVolumeAll(i);

			// --- Stage D: Diffusion (Jacobi) ---
			vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createVolumetricFog_Fluid_D);
			for (uint32_t iter = 0; iter < Jacobi_Iteration_Count; ++iter) {
				pushConstant.iteration = iter;
				vkCmdPushConstants2(cmd, &pushInfo);
				vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
				barrierVolumePingPong(i);
			}

			// --- Stage F: External forces ---
			vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createVolumetricFog_Fluid_F);
			pushConstant.iteration = Jacobi_Iteration_Count & 1;
			vkCmdPushConstants2(cmd, &pushInfo);
			vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
			barrierVolumeAll(i);

			// --- Stage P: Pressure solve (Jacobi) ---
			vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createVolumetricFog_Fluid_P);
			for (uint32_t iter = 0; iter < Jacobi_Iteration_Count; ++iter) {
				pushConstant.iteration = iter + (Jacobi_Iteration_Count & 1);
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
		}
	}
}
void VolumetricFog::createEnvFog(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createEnvionmentFog);
	pushConstant.sampleCount = sampleCount_env;
	pushConstant.jitterStrength = jitterStrength_env;
	vkCmdPushConstants2(cmd, &pushInfo);
#ifdef Uniform_EnvFog_Grid
	VkExtent3D groupSize = nvvk::getGroupCounts(VkExtent3D{ envGridSize.x, envGridSize.y, envGridSize.z }, VkExtent3D{ 4, 4, 4 });
#else
	VkExtent3D groupSize = nvvk::getGroupCounts(VkExtent3D{ frustumGridSize.width, frustumGridSize.height, frustumGridSize.depth }, VkExtent3D{ 4, 4, 4 });
#endif
	
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
}
void VolumetricFog::envFogLightAttenuationEstimate(VkCommandBuffer cmd) {
#ifndef Uniform_EnvFog_Grid
	return;
#endif
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createLightAttenuationEstimator);
	pushConstant.sampleCount = 10;
	pushConstant.forwardSampleCount = 0;
	vkCmdPushConstants2(cmd, &pushInfo);
	vkCmdDispatch(cmd, 1, 1, 1);
}
void VolumetricFog::createFrustumAccFog(VkCommandBuffer cmd) {
	if (!pushConstant.useAccFog) return;
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
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createFrustumAccFog);

	pushConstant.useAccFog = 0;
	pushConstant.sampleCount = rmSampleCountSampleCount_fogAcc;
	pushConstant.randomStepping = randomStepping_fogAcc;
	pushConstant.forwardSampleCount = 0;
	pushConstant.jitterStrength = interpolationJitterStrength_fogAcc;
	pushConstant.accJitterStrength = accJitterStrength;
#ifdef Fog_Acc_Stepping
	for (int i = 0; i < frustumGridSize.depth; ++i) {
		pushConstant.instanceIndex = i;
		vkCmdPushConstants2(cmd, &pushInfo);

		VkExtent3D groupSize = nvvk::getGroupCounts(VkExtent3D{ frustumGridSize.width, frustumGridSize.height, 1 }, VkExtent3D{ 32, 32, 1 });
		vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);

		barrierImage(volumetricFogAttenuationImage.image.image);
		barrierImage(volumetricFogLImage.image.image);
	}
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
	pushConstant.jitterStrength = interpolationJitterStrength;
	//if (pushConstant.useAccFog) pushConstant.forwardSampleCount = forwardSampleCount;
	//else pushConstant.forwardSampleCount = 0;
	pushConstant.forwardSampleCount = forwardSampleCount;
	vkCmdPushConstants2(cmd, &pushInfo);

	VkExtent2D groupSize = nvvk::getGroupCounts(gBuffers.getSize(), VkExtent2D{16, 16});
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);

	//nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}
void VolumetricFog::fogBlur(VkCommandBuffer cmd) {
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

	//vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_addFog);
	//vkCmdPushConstants2(cmd, &pushInfo);
	//VkExtent2D groupSize = nvvk::getGroupCounts(gBuffers.getSize(), VkExtent2D{ 32, 32 });
	//vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
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
	pushConstant.forwardSampleCount = 0;
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
	bool show = showVolumetricFogVoxelGrid;
	for (int i = 0; i < volumetricFogCount; ++i) show |= showVolumetricFogVoxelGrids[i] == 1;
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
		if (showVolumetricFogVoxelGrids[i] == 1) {
			pushConstant.instanceIndex = i;
			vkCmdPushConstants2(cmd, &pushInfo);
			vkCmdDrawIndexed(cmd, triMesh.indices.count, volumetricFogInfos[i].fogVoxelGridSize.x * volumetricFogInfos[i].fogVoxelGridSize.y * volumetricFogInfos[i].fogVoxelGridSize.z, 0, 0, 0);
		}
	}
	if (showEnvGrid) {
		pushConstant.instanceIndex = -100;
		vkCmdPushConstants2(cmd, &pushInfo);
		vkCmdDrawIndexed(cmd, triMesh.indices.count, envGridSize.x * envGridSize.y * envGridSize.z, 0, 0, 0);
	}

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
	vkCmdDrawIndexed(cmd, triMesh.indices.count, pushConstant.frustumGridSize.x * pushConstant.frustumGridSize.y * pushConstant.frustumGridSize.z, 0, 0, 0);

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
