#pragma once

#include <nvapp/application.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvslang/slang.hpp>
#include <nvshaders_host/sky.hpp>
#include <nvshaders_host/tonemapper.hpp>
#include <common/path_utils.hpp>
#include <nvaftermath/aftermath.hpp>
#include <renderer/Renderer.h>
#include <common/Scene/Scene.h>
#include <nvvk/context.hpp>

#include <nvutils/camera_manipulator.hpp>
#include <common/Mesh/nvvk/gltf_utils.hpp>
#include <nvvk/acceleration_structures.hpp>
#include <random>

#ifndef FZB_APPLICATION_H
#define FZB_APPLICATION_H

#define MAX_FRAME 1 << 30
//#define PathTracingMotionBlur

//#ifdef NDEBUG
//#define IF_DEBUG(debug, release) do { release; } while (0)
//#else
//#define IF_DEBUG(debug, release) do { debug; } while (0)
//#endif

#ifdef NDEBUG
#define IF_DEBUG(debug, release) (release)
#else
#define IF_DEBUG(debug, release) (debug)
#endif

namespace FzbRenderer {

class Application : public nvapp::IAppElement {
public:
	Application() = default;
	~Application() override = default;

	Application(nvapp::ApplicationCreateInfo& appInfo, nvvk::Context& vkContext);

	void onAttach(nvapp::Application* app) override;
	void onDetach() override;
	void onUIRender() override;
	void onResize(VkCommandBuffer cmd, const VkExtent2D& size);

	void onPreRender() override;
	void onRender(VkCommandBuffer* cmd) override;

	void onUIMenu() override;
	void onLastHeadlessFrame() override;
	std::shared_ptr<nvutils::CameraManipulator> getCameraManipulator() const { return sceneResource.cameraManip; };

	inline static nvvk::ContextInitInfo vkContextInitInfo{};
	inline static nvvk::Context* vkContext = nullptr;

	inline static nvapp::Application* app{};
	inline static nvvk::ResourceAllocator allocator{};
	inline static nvvk::StagingUploader   stagingUploader{};
	inline static nvvk::ResourceAllocatorExport allocatorExport{};
	inline static nvvk::StagingUploader   stagingUploaderExport{};
	inline static nvvk::SamplerPool       samplerPool{};
	inline static nvslang::SlangCompiler     slangCompiler{};

	inline static FzbRenderer::Scene sceneResource;

	inline static nvshaders::SkySimple skySimple{};
	inline static nvshaders::Tonemapper tonemapper{};
	inline static shaderio::TonemapperData tonemapperData{};

	inline static uint32_t cmdCount = 1;
	inline static int frameIndex = -1;
	inline static bool UIModified = false;
	inline static VkDescriptorSet viewportImage = nullptr;
	inline static ImVec2 viewportScreenPos = ImVec2(0, 0);
	inline static ImVec2 viewportContentSize = ImVec2(0, 0);

	static void uploadResource();
private:
	void getAppInfoFromXML(nvapp::ApplicationCreateInfo& appInfo);
	void initSlangCompiler();

	std::vector<std::string> slangIncludes;	//slang��include��ַ

	std::shared_ptr<FzbRenderer::Renderer> renderer;
};
}

#endif