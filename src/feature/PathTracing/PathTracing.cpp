#include "./PathTracing.h"
#include <common/Application/Application.h>

using namespace FzbRenderer;

void PathTracingContext::setContextInfo() {
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature });
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature });
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME });
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_NV_RAY_TRACING_MOTION_BLUR_EXTENSION_NAME, &rtMotionBlurFeatures });

	rtPosFetchFeature.rayTracingPositionFetch = VK_TRUE;
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME, &rtPosFetchFeature });
}
void PathTracingContext::getRayTracingPropertiesAndFeature() {
	//查询是否支持rtPosFetchFeature
	VkPhysicalDeviceFeatures2 deviceFeatures2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
	deviceFeatures2.pNext = &rtPosFetchFeature;
	rtPosFetchFeature.pNext = nullptr;
	rtPosFetchFeature.rayTracingPositionFetch = VK_FALSE;
	vkGetPhysicalDeviceFeatures2(Application::app->getPhysicalDevice(), &deviceFeatures2);
	//查询设备参数
	VkPhysicalDeviceProperties2 prop2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
	prop2.pNext = &rtProperties;
	vkGetPhysicalDeviceProperties2(Application::app->getPhysicalDevice(), &prop2);
}

void PathTracing::init(PathTracingSetting setting) {
	this->setting = setting;
	sbtGenerator.init(Application::app->getDevice(), setting.ptContext->rtProperties);
}
void PathTracing::clean() {
	Feature::clean();
	sbtGenerator.deinit();
	Application::allocator.destroyBuffer(sbtBuffer);
}

void FzbRenderer::createShaderBindingTable(
	const VkRayTracingPipelineCreateInfoKHR& rtPipelineInfo, VkPipeline& rtPipeline,
	nvvk::SBTGenerator& sbtGenerator, nvvk::Buffer& sbtBuffer) {
	/*
	STB，顾名思义，就是shader大的绑定表；
	其将pipeline各个阶段的shader组进行绑定
	每个实例只能用每个阶段的shader组中的一个条目
	每个组和条目都需要对齐
	*/
	SCOPED_TIMER(__FUNCTION__);
	Application::allocator.destroyBuffer(sbtBuffer);

	size_t bufferSize = sbtGenerator.calculateSBTBufferSize(rtPipeline, rtPipelineInfo);
	NVVK_CHECK(Application::allocator.createBuffer(sbtBuffer, bufferSize, VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR,
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
		sbtGenerator.getBufferAlignment()));
	NVVK_DBG_NAME(sbtBuffer.buffer);
	NVVK_CHECK(sbtGenerator.populateSBTBuffer(sbtBuffer.address, bufferSize, sbtBuffer.mapping));

	LOGI(" Shader binding table created and populated\n");
}