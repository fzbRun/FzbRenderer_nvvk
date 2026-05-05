#include "./Image.h"
#include <common/Application/Application.h>

FzbRenderer::ImageCreateInfo FzbRenderer::createDefaultImageCreateInfo() {
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT
        | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    const VkImageCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
        .extent = {1024, 1024, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .usage = usage,
    };
    VkImageViewCreateInfo viewInfo = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
        .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1},
    };

    ImageCreateInfo createInfo = {
        .info = info,
        .viewInfo = viewInfo,
        .sampler = nullptr
    };
    return createInfo;
}
VkResult FzbRenderer::createImage(nvvk::Image& image, ImageCreateInfo createInfo) {
    NVVK_FAIL_RETURN(Application::allocator.createImage(image, createInfo.info, createInfo.viewInfo));  //iamge.descriptor.imageView whill be writed
    image.descriptor.sampler = createInfo.sampler;
    return VK_SUCCESS;
}
void FzbRenderer::destroyImage(nvvk::Image& image) {
    Application::allocator.destroyImage(image);
}