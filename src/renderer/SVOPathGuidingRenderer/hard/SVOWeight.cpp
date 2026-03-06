#include "SVOWeight.h"
#include <common/Shader/Shader.h>
#include <nvutils/timers.hpp>
#include <common/Application/Application.h>
#include <nvvk/default_structs.hpp>
#include <nvgui/property_editor.hpp>
#include <nvvk/compute_pipeline.hpp>

using namespace FzbRenderer;

SVOWeight::SVOWeight(pugi::xml_node& featureNode) {}
void SVOWeight::init(SVOWeightSetting setting) {
	this->setting = setting;
	sbtGenerator.init(Application::app->getDevice(), setting.ptContext->rtProperties);

	createDescriptorSetLayout();
	createDescriptorSet();
	Feature::createPipelineLayout(sizeof(shaderio::SVOWeightPushConstant));
	compileAndCreateShaders();
}
void SVOWeight::clean() {
	PathTracing::clean();
	asManager.clean();

	VkDevice device = Application::app->getDevice();
	vkDestroyPipeline(device, rtPipeline, nullptr);
	vkDestroyPipelineLayout(device, rtPipelineLayout, nullptr);
}
void SVOWeight::uiRender() {};
void SVOWeight::resize(VkCommandBuffer cmd, const VkExtent2D& size) {};
void SVOWeight::preRender() {
	if (Application::sceneResource.cameraChange) Application::frameIndex = 0;
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;

#ifndef NDEBUG
	pushConstant.frameIndex = Application::frameIndex;
#endif
}
void SVOWeight::render(VkCommandBuffer cmd) {

}
void SVOWeight::postProcess(VkCommandBuffer cmd) {};

void SVOWeight::createDescriptorSetLayout() {

}
void SVOWeight::createDescriptorSet() {

}
void SVOWeight::createPipeline() {

}
void SVOWeight::compileAndCreateShaders() {

}
void SVOWeight::updateDataPerFrame(VkCommandBuffer cmd) {

}