#include "./SparseVoxelOctree.h"
#include "common/Application/Application.h"

using namespace FzbRenderer;

SparseVoxelOctree::SparseVoxelOctree(pugi::xml_node& featureNode) {

}
void SparseVoxelOctree::init(SVOSetting setting) {
	this->setting = setting;

	
}

void SparseVoxelOctree::clean() {
	Feature::clean();
	for(int i = 0; i < SVOArray_G.size(); ++i) Application::allocator.destroyBuffer(SVOArray_G[i]);
	for(int i = 0; i < SVOArray_E.size(); ++i) Application::allocator.destroyBuffer(SVOArray_E[i]);
	Application::allocator.destroyBuffer(SVOIndivisibleNodes_G);
}

void SparseVoxelOctree::resize(VkCommandBuffer cmd, const VkExtent2D& size) {

}
void SparseVoxelOctree::uiRender() {

}
void SparseVoxelOctree::preRender() {

}
void SparseVoxelOctree::render(VkCommandBuffer cmd) {

}
void SparseVoxelOctree::postProcess(VkCommandBuffer cmd) {

}

void SparseVoxelOctree::createSVOArray() {

}