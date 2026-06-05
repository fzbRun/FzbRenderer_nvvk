#include "./Image.h"
#include <common/Application/Application.h>
#include <nvvk/formats.hpp>
#include <nvvk/default_structs.hpp>

#include <stb/stb_image.h>
#include <common/utils.hpp>

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

    VkSamplerCreateInfo sampleCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR
    };

    ImageCreateInfo createInfo = {
        .info = info,
        .viewInfo = viewInfo,
        .samplerInfo = sampleCreateInfo
    };
    return createInfo;
}
VkResult FzbRenderer::createImage(nvvk::Image& image, ImageCreateInfo createInfo) {
    NVVK_FAIL_RETURN(Application::allocator.createImage(image, createInfo.info, createInfo.viewInfo));  //iamge.descriptor.imageView whill be writed
    Application::samplerPool.acquireSampler(image.descriptor.sampler, createInfo.samplerInfo);
    return VK_SUCCESS;
}
void FzbRenderer::destroyImage(nvvk::Image& image) {
    Application::allocator.destroyImage(image);
}

FzbRenderer::Image::Image(std::string name, bool external) {
	this->name = name;
	this->external = external;
}
VkResult FzbRenderer::Image::init(ImageCreateInfo createInfo) {
    clean();
	setting = createInfo;

    if (external) {
        nvvk::StagingUploader& staging = Application::stagingUploaderExport;
        nvvk::ResourceAllocatorExport* allocator = dynamic_cast<nvvk::ResourceAllocatorExport*>(staging.getResourceAllocator());

        NVVK_CHECK(allocator->createImageExport(image, createInfo.info, createInfo.viewInfo));
        
        VmaAllocationInfo allocInfo;
        vmaGetAllocationInfo(Application::allocator, image.allocation, &allocInfo);
        imageSize = allocInfo.size;

        VkMemoryGetWin32HandleInfoKHR handleInfo = {};
        handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        handleInfo.memory = allocInfo.deviceMemory;
        handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        FzbRenderer::GetMemoryWin32HandleKHR(&handleInfo, &this->handle);
    }
    else NVVK_FAIL_RETURN(Application::allocator.createImage(image, createInfo.info, createInfo.viewInfo));  //iamge.descriptor.imageView whill be writed
    Application::samplerPool.acquireSampler(image.descriptor.sampler, createInfo.samplerInfo);

    VkDevice device = Application::app->getDevice();

    nvvk::DebugUtil& dutil = nvvk::DebugUtil::getInstance();
    dutil.setObjectName(image.image, name);
    dutil.setObjectName(image.descriptor.imageView, name + "View");

    VkImageViewCreateInfo uiViewInfo = createInfo.viewInfo;
    uiViewInfo.image = image.image;
    uiViewInfo.components.a = VK_COMPONENT_SWIZZLE_ONE;  // Forcing the VIEW to have a 1 in the alpha channel
    NVVK_FAIL_RETURN(vkCreateImageView(device, &uiViewInfo, nullptr, &uiImageView));
    dutil.setObjectName(uiImageView, name + "UI View");

    const VkImageLayout imageLayout{ VK_IMAGE_LAYOUT_GENERAL };
    {
		bool isDepth = createInfo.info.format == nvvk::findDepthFormat(Application::app->getPhysicalDevice());

        VkImageMemoryBarrier2 barrier = nvvk::makeImageMemoryBarrier({ .image = image.image,
                                                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                                    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL });
        if (isDepth) barrier.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS };

        const VkDependencyInfo depInfo{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                       .imageMemoryBarrierCount = 1,
                                       .pImageMemoryBarriers = &barrier };

        VkCommandBuffer cmd = Application::app->createTempCmdBuffer();

        vkCmdPipelineBarrier2(cmd, &depInfo);

        uint32_t mipLevels = createInfo.info.mipLevels;
        uint32_t arrayLayers = createInfo.info.arrayLayers;
        if (isDepth) {
            VkClearDepthStencilValue clearDepth = { 1.0f, 0 };
            std::vector<VkImageSubresourceRange> ranges;
            for (uint32_t m = 0; m < mipLevels; ++m) {
                for (uint32_t a = 0; a < arrayLayers; ++a)
                    ranges.push_back({ VK_IMAGE_ASPECT_DEPTH_BIT, m, 1, a, 1 });
            }
            vkCmdClearDepthStencilImage(cmd, image.image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearDepth, ranges.size(), ranges.data());
        }
        else {
            const VkClearColorValue                      clear_value = { {0.F, 0.F, 0.F, 0.F} };
            std::vector<VkImageSubresourceRange> ranges;
            for (uint32_t m = 0; m < mipLevels; ++m) {
                for (uint32_t a = 0; a < arrayLayers; ++a)
                    ranges.push_back({ VK_IMAGE_ASPECT_COLOR_BIT, m, 1, a, 1 });
            }
            vkCmdClearColorImage(cmd, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                &clear_value, ranges.size(), ranges.data());
        }

        // Setting the layout to the final one
        barrier = nvvk::makeImageMemoryBarrier(
            { .image = image.image, .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,.newLayout = imageLayout });
        if (isDepth) barrier.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS };

        image.descriptor.imageLayout = imageLayout;
        vkCmdPipelineBarrier2(cmd, &depInfo);

        Application::app->submitAndWaitTempCmdBuffer(cmd);
    }

    descriptorPool = Application::app->getTextureDescriptorPool();

    const VkDescriptorSetLayoutBinding binding = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT };
    const VkDescriptorSetLayoutCreateInfo descriptorLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &binding };
    NVVK_FAIL_RETURN(vkCreateDescriptorSetLayout(device, &descriptorLayoutInfo, nullptr, &descLayout));

    VkDescriptorImageInfo descImage;
    VkWriteDescriptorSet  writeDesc;
    const VkDescriptorSetAllocateInfo  allocInfos = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
         .descriptorPool = descriptorPool,
         .descriptorSetCount = 1,
         .pSetLayouts = &descLayout,
    };
    NVVK_FAIL_RETURN(vkAllocateDescriptorSets(device, &allocInfos, &uiDescriptorSet));

    descImage = { image.descriptor.sampler, uiImageView, imageLayout };
    writeDesc = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = uiDescriptorSet,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &descImage,
    };
    vkUpdateDescriptorSets(device, 1, &writeDesc, 0, nullptr);

    return VK_SUCCESS;
}
VkResult FzbRenderer::Image::init(std::filesystem::path& imagePath) {
    clean();

    int            w, h, comp, req_comp{ 4 };
    std::string    filenameUtf8 = nvutils::utf8FromPath(imagePath);
    const stbi_uc* data = stbi_load(filenameUtf8.c_str(), &w, &h, &comp, req_comp);
    assert((data != nullptr) && "Could not load texture image!");

    // Define how to create the image
    setting.info = DEFAULT_VkImageCreateInfo;
    setting.info.format = VK_FORMAT_R8G8B8A8_UNORM;
    setting.info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT
        | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    setting.info.extent = { uint32_t(w), uint32_t(h), 1 };

    setting.viewInfo = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = setting.info.format,
        .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1},
    };
   
    //----------------------------------------------------------------------------------------------------------------
    const std::span dataSpan(data, w * h * req_comp);
    if (external) {
        nvvk::StagingUploader& staging = Application::stagingUploaderExport;
        nvvk::ResourceAllocatorExport* allocator = dynamic_cast<nvvk::ResourceAllocatorExport*>(staging.getResourceAllocator());

        NVVK_CHECK(allocator->createImageExport(image, setting.info, setting.viewInfo));
        //Image will not have data immediately. After the staginguploader submits, the data will be copied in
        NVVK_CHECK(staging.appendImage(image, dataSpan, VK_IMAGE_LAYOUT_GENERAL));
        Application::samplerPool.acquireSampler(image.descriptor.sampler);

        VmaAllocationInfo allocInfo;
        vmaGetAllocationInfo(Application::allocator, image.allocation, &allocInfo);
        imageSize = allocInfo.size;

        VkMemoryGetWin32HandleInfoKHR handleInfo = {};
        handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        handleInfo.memory = allocInfo.deviceMemory;
        handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        GetMemoryWin32HandleKHR(&handleInfo, &this->handle);
    }
    else {
        nvvk::StagingUploader& staging = Application::stagingUploader;;
        nvvk::ResourceAllocator* allocator = staging.getResourceAllocator();

        // Use the VMA allocator to create the image
        NVVK_CHECK(allocator->createImage(image, setting.info, setting.viewInfo));
        //Image will not have data immediately. After the staginguploader submits, the data will be copied in
        NVVK_CHECK(staging.appendImage(image, dataSpan, VK_IMAGE_LAYOUT_GENERAL));
        Application::samplerPool.acquireSampler(image.descriptor.sampler);
    }

    //----------------------------------------------------------------------------------------------------------------
    VkDevice device = Application::app->getDevice();

    nvvk::DebugUtil& dutil = nvvk::DebugUtil::getInstance();
    dutil.setObjectName(image.image, name);
    dutil.setObjectName(image.descriptor.imageView, name + "View");

    VkImageViewCreateInfo uiViewInfo = setting.viewInfo;
    uiViewInfo.image = image.image;
    uiViewInfo.components.a = VK_COMPONENT_SWIZZLE_ONE;  // Forcing the VIEW to have a 1 in the alpha channel
    NVVK_FAIL_RETURN(vkCreateImageView(device, &uiViewInfo, nullptr, &uiImageView));
    dutil.setObjectName(uiImageView, name + "UI View");

    const VkImageLayout imageLayout{ VK_IMAGE_LAYOUT_GENERAL };
    descriptorPool = Application::app->getTextureDescriptorPool();

    const VkDescriptorSetLayoutBinding binding = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT };
    const VkDescriptorSetLayoutCreateInfo descriptorLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &binding };
    NVVK_FAIL_RETURN(vkCreateDescriptorSetLayout(device, &descriptorLayoutInfo, nullptr, &descLayout));

    VkDescriptorImageInfo descImage;
    VkWriteDescriptorSet  writeDesc;
    const VkDescriptorSetAllocateInfo  allocInfos = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
         .descriptorPool = descriptorPool,
         .descriptorSetCount = 1,
         .pSetLayouts = &descLayout,
    };
    NVVK_FAIL_RETURN(vkAllocateDescriptorSets(device, &allocInfos, &uiDescriptorSet));

    descImage = { image.descriptor.sampler, uiImageView, imageLayout };
    writeDesc = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = uiDescriptorSet,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &descImage,
    };
    vkUpdateDescriptorSets(device, 1, &writeDesc, 0, nullptr);

    return VK_SUCCESS;
}
void FzbRenderer::Image::clean() {
    if(external) Application::allocatorExport.destroyImage(image);
    else Application::allocator.destroyImage(image);

    VkDevice device = Application::app->getDevice();
    vkDestroyImageView(device, uiImageView, nullptr);
    if (descriptorPool) vkFreeDescriptorSets(device, descriptorPool, 1, &uiDescriptorSet);
    vkDestroyDescriptorSetLayout(device, descLayout, nullptr);
    descLayout = VK_NULL_HANDLE;

    if (handle != nullptr) { // HANDLE 在 Win32 下通常是 void* / HANDLE
        CloseHandle(handle);
        handle = nullptr;
    }
}