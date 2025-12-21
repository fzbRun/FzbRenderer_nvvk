#include <common/Application/Application.h>
#include "./RasterVoxelization.h"
#include "common/utils.hpp"

FzbRenderer::RasterVoxelization::RasterVoxelization(pugi::xml_node& featureNode) {
	if (pugi::xml_node voxelCountNode = featureNode.child("voxelCount"))
		setting.pushConstant.voxelCount = FzbRenderer::getRGBFromString(voxelCountNode.attribute("value").value());
	else setting.pushConstant.voxelSize = glm::uvec3(64);

	if (pugi::xml_node debugModeNode = featureNode.child("debugMode")) {
		std::string debugModeStr = debugModeNode.attribute("type").value();
		if (debugModeStr == "Cube") setting.debugMode = DebugMode::Cube;
		else setting.debugMode = DebugMode::Wireframe;
	}
	else setting.debugMode = DebugMode::None;

	if (setting.debugMode == DebugMode::Wireframe) {
		Application::vkContext->getPhysicalDeviceFeatures_notConst().fillModeNonSolid = VK_TRUE;
		Application::vkContext->getPhysicalDeviceFeatures_notConst().wideLines = VK_TRUE;
		setting.lineWidth = std::stof(featureNode.child("debugMode").attribute("value").value());
	}
}

void FzbRenderer::RasterVoxelization::init() {

}
void FzbRenderer::RasterVoxelization::clean() {

}
void FzbRenderer::RasterVoxelization::uiRender() {

}
void FzbRenderer::RasterVoxelization::render(VkCommandBuffer cmd) {

}
void FzbRenderer::RasterVoxelization::createGBuffer(bool useDepth) {

}
void FzbRenderer::RasterVoxelization::createGraphicsDescriptorSetLayout() {

}
void FzbRenderer::RasterVoxelization::createGraphicsPipelineLayout(uint32_t pushConstantSize) {

}
void FzbRenderer::RasterVoxelization::compileAndCreateShaders() {

}
void FzbRenderer::RasterVoxelization::updateDataPerFrame(VkCommandBuffer cmd) {

}