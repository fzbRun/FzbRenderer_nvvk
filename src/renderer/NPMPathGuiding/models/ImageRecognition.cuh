#pragma once

#include "common/CUDA/vulkanCudaInterop.cuh"

struct ImageRecognition_CreateInfo {
	VkPhysicalDevice physicalDevice;
	FzbRenderer::Image image;

	HANDLE startSemaphoreHandle;
	HANDLE endSemaphoreHandle;
};
class ImageRecognition {
public:
	ImageRecognition() = default;

	ImageRecognition(ImageRecognition_CreateInfo createInfo);
	void recognition(uint32_t frameIndex, uint64_t waitTimeline = 1);

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
