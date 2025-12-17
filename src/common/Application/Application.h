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

#include <nvutils/camera_manipulator.hpp>
#include <common/Mesh/nvvk/gltf_utils.hpp>
#include <nvvk/acceleration_structures.hpp>

#ifndef FZB_APPLICATION_H
#define FZB_APPLICATION_H

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
	void onRender(VkCommandBuffer cmd);

	void onUIMenu() override;
	void onLastHeadlessFrame() override;
	std::shared_ptr<nvutils::CameraManipulator> getCameraManipulator() const { return sceneResource.cameraManip; };

	//所有的全局共用资源
	inline static nvapp::Application* app{};
	inline static nvvk::ResourceAllocator allocator{};
	inline static nvvk::StagingUploader   stagingUploader{};
	inline static nvvk::SamplerPool       samplerPool{};
	inline static nvslang::SlangCompiler     slangCompiler{};

	inline static FzbRenderer::Scene sceneResource;

	inline static nvshaders::SkySimple skySimple{};
	inline static nvshaders::Tonemapper tonemapper{};
	inline static shaderio::TonemapperData tonemapperData{};

	inline static int frameIndex = 0;
	inline static bool UIModified = false;
private:
	/*
		这个函数会从项目根目录/rendererInfo/rendererInfo.xml中读取信息，包括
		1. 渲染器名称、分辨率
		2. 渲染sceneInfo.xml的地址
		3. 渲染器的类型，如前向渲染、路径追踪，并初始化相应的渲染器
	*/
	void getAppInfoFromXML(nvapp::ApplicationCreateInfo& appInfo, nvvk::ContextInitInfo& vkContextInitInfo);
	void initSlangCompiler();

	void updateDataPerFrame(VkCommandBuffer cmd);

	std::vector<std::string> slangIncludes;	//slang的include地址

	std::shared_ptr<FzbRenderer::Renderer> renderer;
};
}

#endif