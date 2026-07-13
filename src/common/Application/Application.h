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
	/*
	nvapp���߼���
	1. m_maxFramesInFlight��¼����������󻺳壬�ȷ�˵FIFO�ǵ����壻V-Sync��˫���壻һ����fast-Sync��������
	2. ��ôCPU���ͬʱ����m_maxFramesInFlight��ָ���GPU����һ֡һִ֡�е�
	3. m_maxFramesInFlight=3����ô��һ����СΪ3�������ʾ�ȴ��ź���������һ��ʼԪ����0��1��2����ʱ�����ź���Ϊ2
	4. ��ô0��1��2֡CPU��������ȴ�GPUִ����ֱ�Ӵ���ָ�������Ӧ������Ԫ��+3��������֡3 < 2����ִ��
	5. ����һ֡GPUִ�����ʱ�����ź���+1��Ϊ3����ô����֡����ִ�У���������Ԫ������Ϊ6
	6. ����

	��������Ⱦ˳��ΪUIRender->PreRender->Render
	*/
	void onPreRender() override;
	void onRender(VkCommandBuffer* cmd) override;

	void onUIMenu() override;
	void onLastHeadlessFrame() override;
	std::shared_ptr<nvutils::CameraManipulator> getCameraManipulator() const { return sceneResource.cameraManip; };

	inline static nvvk::ContextInitInfo vkContextInitInfo{};
	inline static nvvk::Context* vkContext = nullptr;

	//���е�ȫ�ֹ�����Դ
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
	/*
		������������Ŀ��Ŀ¼/rendererInfo/rendererInfo.xml�ж�ȡ��Ϣ������
		1. ��Ⱦ�����ơ��ֱ���
		2. ��ȾsceneInfo.xml�ĵ�ַ
		3. ��Ⱦ�������ͣ���ǰ����Ⱦ��·��׷�٣�����ʼ����Ӧ����Ⱦ��
	*/
	void getAppInfoFromXML(nvapp::ApplicationCreateInfo& appInfo);
	void initSlangCompiler();

	std::vector<std::string> slangIncludes;	//slang��include��ַ

	std::shared_ptr<FzbRenderer::Renderer> renderer;
};
}

#endif