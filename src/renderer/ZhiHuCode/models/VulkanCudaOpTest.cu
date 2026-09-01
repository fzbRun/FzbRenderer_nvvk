#include "./VulkanCudaOpTest.cuh"

Image_yReversal::Image_yReversal(Image_yReversal_CreateInfo createInfo) {
	if (getCudaDeviceForVulkanPhysicalDevice(createInfo.physicalDevice) == cudaInvalidDeviceId)
		throw std::runtime_error("CUDA与Vulkan用的不是同一个GPU！！！");

	FzbRenderer::Image& image = createInfo.image;
	VkImageCreateInfo imagInfo = image.setting.info;

	imageWidth = imagInfo.extent.width;
	imageHeight = imagInfo.extent.height;

	fromVulkanImageToCudaSurface(
		createInfo.physicalDevice,
		image, false, imageExtMem, imageMipmap, imageObject
	);

	startSemaphore = importVulkanSemaphoreObjectFromNTHandle(createInfo.startSemaphoreHandle);
	endSemaphore = importVulkanSemaphoreObjectFromNTHandle(createInfo.endSemaphoreHandle);

	CHECK(cudaStreamCreate(&stream));
}

__global__ void yReversalKernel(cudaSurfaceObject_t imageObj, uint32_t width, uint32_t height, uint32_t frameIndex) {
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height / 2) return;
	//if (x == 0 && y == 0) printf("yReversalKernel start\n");

	uint32_t y_up = y;
	uchar4 pixel_up = surf2Dread<uchar4>(imageObj, x * sizeof(uchar4), y_up);

	//if(x == 0 && y == 0) printf("cuda frameIndex: %d\n", frameIndex);
	//return;

	uint32_t y_bottom = height - y - 1;
	uchar4 pixel_bottom = surf2Dread<uchar4>(imageObj, x * sizeof(uchar4), y_bottom);

	surf2Dwrite(pixel_up, imageObj, x * sizeof(uchar4), y_bottom);
	surf2Dwrite(pixel_bottom, imageObj, x * sizeof(uchar4), y_up);
}
void Image_yReversal::infer(uint32_t frameIndex, uint64_t waitTimeline) {
	CHECK(waitExternalSemaphore(startSemaphore, stream, waitTimeline));

	uint32_t halfHeight = imageHeight / 2;

	dim3 blockSize(16, 16, 1);
	dim3 gridSize = dim3((imageWidth + blockSize.x - 1) / blockSize.x, (halfHeight + blockSize.y - 1) / blockSize.y, 1);

	yReversalKernel << < gridSize, blockSize, 0, stream >> > (imageObject, imageWidth, imageHeight, frameIndex);

	CHECK(signalExternalSemaphore(endSemaphore, stream, waitTimeline));
}
void Image_yReversal::clean() {
	CHECK(cudaDestroyTextureObject(imageObject));
	CHECK(cudaFreeMipmappedArray(imageMipmap));
	CHECK(cudaDestroyExternalMemory(imageExtMem));

	cudaDestroyExternalSemaphore(startSemaphore);
	cudaDestroyExternalSemaphore(endSemaphore);

	CHECK(cudaStreamDestroy(stream));
}