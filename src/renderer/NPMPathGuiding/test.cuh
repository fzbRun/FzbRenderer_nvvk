#pragma once

#include "common/CUDA/vulkanCudaInterop.cuh"

struct Image_yReversal_CreateInfo {
	VkPhysicalDevice physicalDevice;
	FzbRenderer::Image image;

	HANDLE startSemaphoreHandle;
	HANDLE endSemaphoreHandle;
};
class Image_yReversal {
public:
	Image_yReversal() = default;

	Image_yReversal(Image_yReversal_CreateInfo createInfo);
	void reversal(uint32_t frameIndex, uint64_t waitTimeline = 1);

	void clean();

private:
	uint32_t imageWidth;
	uint32_t imageHeight;

	cudaExternalMemory_t imageExtMem;
	cudaMipmappedArray_t imageMipmap;
	cudaSurfaceObject_t imageObject;

	cudaExternalSemaphore_t startSemaphore;
	cudaExternalSemaphore_t endSemaphore;

	cudaStream_t stream = nullptr;
};
