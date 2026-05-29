#include "./ImageRecognition.cuh"


//------------------------------------------------------------------------------------------------------------------------
ImageRecognition::ImageRecognition(ImageRecognition_CreateInfo createInfo) {
	if (getCudaDeviceForVulkanPhysicalDevice(createInfo.physicalDevice) == cudaInvalidDeviceId)
		throw std::runtime_error("CUDA与Vulkan用的不是同一个GPU！！！");

	FzbRenderer::Image& image = createInfo.image;
	VkImageCreateInfo imagInfo = image.setting.info;

	imageWidth = imagInfo.extent.width;
	imageHeight = imagInfo.extent.height;

	uint32_t imageSize = image.imageSize;
	fromVulkanImageToCudaSurface(
		createInfo.physicalDevice,
		image, image.handle, imageSize,
		false, imageExtMem, imageMipmap, imageObject
	);

	startSemaphore = importVulkanSemaphoreObjectFromNTHandle(createInfo.startSemaphoreHandle);
	endSemaphore = importVulkanSemaphoreObjectFromNTHandle(createInfo.endSemaphoreHandle);

	CHECK(cudaStreamCreate(&stream));
}