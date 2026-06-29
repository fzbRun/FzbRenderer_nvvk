#pragma once
#include <vulkan/vulkan_core.h>
#include <nvvk/resources.hpp>
#include <string>
#include <filesystem>
#include <volk.h>

#ifndef FZBRENDERER_IMAGE_FZBPATHGUIDING_H
#define FZBRENDERER_IMAGE_FZBPATHGUIDING_H
namespace FzbRenderer {
struct ImageCreateInfo {
	VkImageCreateInfo info;
	VkImageViewCreateInfo viewInfo;
	std::vector<VkImageViewCreateInfo> viewInfos;
	VkSamplerCreateInfo samplerInfo;
};

ImageCreateInfo createDefaultImageCreateInfo(uint32_t imageViewCount = 0);
VkResult createImage(nvvk::Image& image, ImageCreateInfo createInfo);
void destroyImage(nvvk::Image& image);

class Image {
public:
	Image(std::string name, bool external = false);
	Image() = default;
	~Image() = default;

	VkResult init(ImageCreateInfo createInfo);
	VkResult init(std::filesystem::path& imagePath);
	void clean();

	std::string name = "image";

	ImageCreateInfo setting;
	nvvk::Image image;
	std::vector<VkImageView> imageViews;

	uint32_t allocMemSize;
	uint32_t allocMemTotalSize;
	uint32_t allocMemOffset;

	VkImageView     uiImageView{};
	VkDescriptorPool descriptorPool = nullptr;
	VkDescriptorSetLayout descLayout{};
	VkDescriptorSet uiDescriptorSet{};

	bool external = false;
	HANDLE handle = nullptr;
};
}

#endif