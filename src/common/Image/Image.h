#pragma once
#include <vulkan/vulkan_core.h>
#include <nvvk/resources.hpp>

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
}

#endif