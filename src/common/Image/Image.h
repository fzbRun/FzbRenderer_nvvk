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
	VkSampler sampler;
};

ImageCreateInfo createDefaultImageCreateInfo();
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

	uint32_t imageSize;

	VkImageView     uiImageView{};
	VkDescriptorPool descriptorPool = nullptr;
	VkDescriptorSetLayout descLayout{};
	VkDescriptorSet uiDescriptorSet{};

	bool external = false;
	HANDLE handle = nullptr;
};
}

#endif