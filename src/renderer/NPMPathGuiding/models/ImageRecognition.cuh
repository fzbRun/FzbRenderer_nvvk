#pragma once

#include "common/CUDA/vulkanCudaInterop.cuh"
#include <common/CUDA/TensorRT/TensorRT.cuh>

struct ImageRecognition_CreateInfo {
	VkPhysicalDevice physicalDevice;
	FzbRenderer::Buffer buffer;

	HANDLE startSemaphoreHandle;
	HANDLE endSemaphoreHandle;
};
class ImageRecognition {
public:
	ImageRecognition() = default;
	virtual ~ImageRecognition() = default;

	ImageRecognition(ImageRecognition_CreateInfo createInfo);
	ImageRecognition& operator=(ImageRecognition&&) noexcept;
	void recognition(uint64_t waitTimeline = 1);

	void clean();

private:
	cudaExternalMemory_t inputTensorExtMem;
	float* inputTensor;

	cudaExternalSemaphore_t startSemaphore;
	cudaExternalSemaphore_t endSemaphore;

	cudaStream_t stream = nullptr;

	FzbRenderer::Model resNet34;
};
