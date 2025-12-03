#define TINYGLTF_IMPLEMENTATION         // Implementation of the GLTF loader library
#define STB_IMAGE_IMPLEMENTATION        // Implementation of the image loading library
#define STB_IMAGE_WRITE_IMPLEMENTATION  // Implementation of the image writing library
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1  // Use dynamic Vulkan functions for VMA (Vulkan Memory Allocator)
#define VMA_IMPLEMENTATION              // Implementation of the Vulkan Memory Allocator
#define VMA_LEAK_LOG_FORMAT(format, ...)                                                                               \
  {                                                                                                                    \
    printf((format), __VA_ARGS__);                                                                                     \
    printf("\n");                                                                                                      \
  }
// #define USE_NSIGHT_AFTERMATH  (not always on, as it slows down the application)

#include "./Application.h"
#include "pugixml.hpp"
#include <nvvk/validation_settings.hpp>
#include <common/utils.hpp>
#include <nvgui/camera.hpp>
#include <nvgui/sky.hpp>
#include <nvgui/camera.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <nvgui/tonemapper.hpp>
#include "common/Shader/spv/sky_simple.slang.h"
#include "common/Shader/spv/tonemapper.slang.h"

void FzbRenderer::Application::getAppInfoFromXML(nvapp::ApplicationCreateInfo& appInfo, nvvk::ContextInitInfo& vkContextInitInfo) {
	std::filesystem::path exePath = nvutils::getExecutablePath().parent_path();
	std::filesystem::path rendererInfoXMLPath = std::filesystem::absolute(exePath / TARGET_EXE_TO_SOURCE_DIRECTORY / "rendererInfo") / "rendererInfo.xml";
	pugi::xml_document doc;
	auto result = doc.load_file(rendererInfoXMLPath.c_str());
	if (!result) {
		throw std::runtime_error("rendererInfoXML打开失败");
	}
	pugi::xml_node rendererInfo = doc.document_element();

	appInfo.name = rendererInfo.child("name").attribute("value").value();
	appInfo.windowSize.x = std::stoi(rendererInfo.child("resolution").attribute("width").value());
	appInfo.windowSize.y = std::stoi(rendererInfo.child("resolution").attribute("height").value());

	sceneResource.scenePath = rendererInfo.child("sceneXML").attribute("path").value();
	sceneResource.scenePath = std::filesystem::absolute(exePath / TARGET_EXE_TO_SOURCE_DIRECTORY / "resources") / sceneResource.scenePath;

	if (pugi::xml_node rendererNode = rendererInfo.child("renderer")) {
		std::string rendererType = rendererNode.attribute("type").value();
		RendererCreateInfo rendererCreateInfo{
			.rendererTypeStr = rendererType,
			.rendererNode = rendererNode,
			.vkContextInfo = vkContextInitInfo,
		};
		renderer = FzbRenderer::createRenderer(rendererCreateInfo);
	}
	else throw std::runtime_error("sceneInfoXML必须指定一个renderer！");

	doc.reset();
}
FzbRenderer::Application::Application(nvapp::ApplicationCreateInfo& appInfo, nvvk::Context& vkContext) {
	VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT };
	nvvk::ContextInitInfo vkSetup{
		.instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
		.deviceExtensions =
			{
				{VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME},
				{VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjectFeatures},
			},
	};

	getAppInfoFromXML(appInfo, vkSetup);

	if (!appInfo.headless)
		nvvk::addSurfaceExtensions(vkSetup.instanceExtensions, &vkSetup.deviceExtensions);

#ifdef NDEBUG
#else
	nvvk::ValidationSettings validationSettings;
	validationSettings.setPreset(nvvk::ValidationSettings::LayerPresets::eStandard);
	vkSetup.instanceCreateInfoExt = validationSettings.buildPNextChain();
#endif

#if defined(USE_NSIGHT_AFTERMATH)   //GPU崩溃之后的调试工具
	auto& aftermath = AftermathCrashTracker::getInstance();
	aftermath.initialize();
	aftermath.addExtensions(vkSetup.deviceExtensions);
	nvvk::CheckError::getInstance().setCallbackFunction([&](VkResult result) { aftermath.errorCallback(result); });
#endif

	if (vkContext.init(vkSetup) != VK_SUCCESS)
	{
		LOGE("Error in Vulkan context creation\n");
		throw std::runtime_error("VulkanContext 初始化失败");
	}

	appInfo.instance = vkContext.getInstance();
	appInfo.device = vkContext.getDevice();
	appInfo.physicalDevice = vkContext.getPhysicalDevice();
	appInfo.queues = vkContext.getQueueInfos();
}

void FzbRenderer::Application::onAttach(nvapp::Application* app) {
	this->app = app;
	VmaAllocatorCreateInfo allocatorInfo = {
		.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
		.physicalDevice = app->getPhysicalDevice(),
		.device = app->getDevice(),
		.instance = app->getInstance(),
		.vulkanApiVersion = VK_API_VERSION_1_4,
	};
	allocator.init(allocatorInfo);
	stagingUploader.init(&allocator, true);   //所有的CPU、GPU只一方可见的缓冲的交互都要经过暂存缓冲区
	initSlangCompiler();
	samplerPool.init(app->getDevice());
	
	sceneResource.createSceneFromXML();
	renderer->init();

	skySimple.init(&allocator, std::span(sky_simple_slang));
	tonemapper.init(&allocator, std::span(tonemapper_slang));
}
void FzbRenderer::Application::initSlangCompiler() {
	//必须要有一个，否则在查询shader的for循环不会进入（数量为0），那么直接找不到;
//后面给绝对地址也没关系 commonShaderPath/xxx。会会删去commonShaderPath的，直接得到xxx
	std::filesystem::path commonShaderPath = FzbRenderer::getCommonDir() / "Shader";
	slangCompiler.addSearchPaths({ commonShaderPath });

	slangCompiler.defaultTarget();
	slangCompiler.defaultOptions();

	slangIncludes.resize(0);
	slangIncludes.push_back(commonShaderPath.string());
	slangCompiler.addOption({ .name = slang::CompilerOptionName::Include,
		.value = {.kind = slang::CompilerOptionValueKind::String, .stringValue0 = slangIncludes[0].c_str()}
		});
	std::filesystem::path nvpro_core2Path = FzbRenderer::getProjectRootDir() / "third_party/nvpro_core2";
	slangIncludes.push_back(nvpro_core2Path.string());
	slangCompiler.addOption({ .name = slang::CompilerOptionName::Include,
		.value = {.kind = slang::CompilerOptionValueKind::String, .stringValue0 = slangIncludes[1].c_str()}
		});

	slangCompiler.addOption({ slang::CompilerOptionName::DebugInformation,
		{slang::CompilerOptionValueKind::Int, SLANG_DEBUG_INFO_LEVEL_MAXIMAL} });

#if defined(AFTERMATH_AVAILABLE)
	slangCompiler.setCompileCallback([&](const std::filesystem::path& sourceFile, const uint32_t* spirvCode, size_t spirvSize) {
		std::span<const uint32_t> data(spirvCode, spirvSize / sizeof(uint32_t));
		AftermathCrashTracker::getInstance().addShaderBinary(data);
		});
#endif
}

void FzbRenderer::Application::onDetach() {
	NVVK_CHECK(vkQueueWaitIdle(app->getQueue(0).queue));

	VkDevice device = app->getDevice();
	
	renderer->clean();
	sceneResource.clean();

	stagingUploader.deinit();
	skySimple.deinit();
	tonemapper.deinit();
	samplerPool.deinit();
	allocator.deinit();
}
void FzbRenderer::Application::onUIRender() {
	namespace PE = nvgui::PropertyEditor;
	UIModified = false;
	if (ImGui::Begin("Settings"))
	{
		//ImGui::Checkbox("Use Ray Tracing", &m_useRayTracing);
		if (ImGui::CollapsingHeader("Camera"))
			nvgui::CameraWidget(sceneResource.cameraManip);
		if (ImGui::CollapsingHeader("Environment"))
		{
			UIModified |= ImGui::Checkbox("USE Sky", (bool*)&sceneResource.sceneInfo.useSky);
			if (sceneResource.sceneInfo.useSky)
				UIModified |= nvgui::skySimpleParametersUI(sceneResource.sceneInfo.skySimpleParam);
			else
			{
				PE::begin();
				UIModified |= PE::ColorEdit3("Background", (float*)&sceneResource.sceneInfo.backgroundColor);
				PE::end();
				// Light
				PE::begin();
				if (sceneResource.sceneInfo.punctualLights[0].type == shaderio::GltfLightType::ePoint
					|| sceneResource.sceneInfo.punctualLights[0].type == shaderio::GltfLightType::eSpot)
					UIModified |= PE::DragFloat3("Light Position", glm::value_ptr(sceneResource.sceneInfo.punctualLights[0].position),
						1.0f, -20.0f, 20.0f, "%.2f", ImGuiSliderFlags_None, "Position of the light");
				if (sceneResource.sceneInfo.punctualLights[0].type == shaderio::GltfLightType::eDirectional
					|| sceneResource.sceneInfo.punctualLights[0].type == shaderio::GltfLightType::eSpot)
					UIModified |= PE::SliderFloat3("Light Direction", glm::value_ptr(sceneResource.sceneInfo.punctualLights[0].direction),
						-1.0f, 1.0f, "%.2f", ImGuiSliderFlags_None, "Direction of the light");

				UIModified |= PE::SliderFloat("Light Intensity", &sceneResource.sceneInfo.punctualLights[0].intensity, 0.0f, 1000.0f,
					"%.2f", ImGuiSliderFlags_Logarithmic, "Intensity of the light");
				UIModified |= PE::ColorEdit3("Light Color", glm::value_ptr(sceneResource.sceneInfo.punctualLights[0].color),
					ImGuiColorEditFlags_NoInputs, "Color of the light");
				UIModified |= PE::Combo("Light Type", (int*)&sceneResource.sceneInfo.punctualLights[0].type,
					"Point\0Spot\0Directional\0", 3, "Type of the light (Point, Spot, Directional)");
				if (sceneResource.sceneInfo.punctualLights[0].type == shaderio::GltfLightType::eSpot)
					UIModified |= PE::SliderAngle("Cone Angle", &sceneResource.sceneInfo.punctualLights[0].coneAngle, 0.f, 90.f, "%.2f",
						ImGuiSliderFlags_AlwaysClamp, "Cone angle of the spot light");
				PE::end();
			}
		}
		if (ImGui::CollapsingHeader("Tonemapper"))
			nvgui::tonemapperWidget(tonemapperData);

		//ImGui::Separator();
		//PE::begin();
		//PE::SliderFloat2("Metallic/Roughness Override", glm::value_ptr(metallicRoughnessOverride), -0.01f, 1.0f,
		//	"%.2f", ImGuiSliderFlags_AlwaysClamp, "Override all material metallic and roughness");
		//PE::end();
	}
	ImGui::End();
	renderer->uiRender();
}
void FzbRenderer::Application::onResize(VkCommandBuffer cmd, const VkExtent2D& size) {
	renderer->resize(cmd, size);
}
void FzbRenderer::Application::onRender(VkCommandBuffer cmd) {
	updateDataPerFrame(cmd);
	renderer->render(cmd);
}
void FzbRenderer::Application::updateDataPerFrame(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	++frameIndex;
	renderer->updateDataPerFrame(cmd);

	const glm::mat4& viewMatrix = sceneResource.cameraManip->getViewMatrix();
	const glm::mat4& projMatrix = sceneResource.cameraManip->getPerspectiveMatrix();

	sceneResource.sceneInfo.viewProjMatrix = projMatrix * viewMatrix;
	sceneResource.sceneInfo.projInvMatrix = glm::inverse(projMatrix);
	sceneResource.sceneInfo.viewInvMatrix = glm::inverse(viewMatrix);
	sceneResource.sceneInfo.cameraPosition = sceneResource.cameraManip->getEye();
	sceneResource.sceneInfo.instances = (shaderio::GltfInstance*)sceneResource.bInstances.address;
	sceneResource.sceneInfo.meshes = (shaderio::GltfMesh*)sceneResource.bMeshes.address;
	sceneResource.sceneInfo.materials = (shaderio::GltfMetallicRoughness*)sceneResource.bMaterials.address;

	nvvk::cmdBufferMemoryBarrier(cmd, { sceneResource.bSceneInfo.buffer, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
									   VK_PIPELINE_STAGE_2_TRANSFER_BIT });
	vkCmdUpdateBuffer(cmd, sceneResource.bSceneInfo.buffer, 0, sizeof(shaderio::GltfSceneInfo), &sceneResource.sceneInfo);
	nvvk::cmdBufferMemoryBarrier(cmd, { sceneResource.bSceneInfo.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
									   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT });
}

void FzbRenderer::Application::onUIMenu() {
	bool reload = false;
	if (ImGui::BeginMenu("Tools"))
	{
		reload |= ImGui::MenuItem("Reload shaders", "F5");
		ImGui::EndMenu();
	}
	reload |= ImGui::IsKeyPressed(ImGuiKey_F5);
	if (reload)
	{
		vkQueueWaitIdle(app->getQueue(0).queue);
		renderer->compileAndCreateShaders();
	}
}
void FzbRenderer::Application::onLastHeadlessFrame() {
	renderer->onLastHeadlessFrame();
}