#pragma once

#include "common/CUDA/vulkanCudaInterop.cuh"
#include <common/CUDA/TensorRT/TensorRT.cuh>

#ifndef FZBRENDERER_KPCNN_CUH
#define FZBRENDERER_KPCNN_CUH

struct KPCNNDenoiser_CreateInfo {
	VkPhysicalDevice physicalDevice;
	FzbRenderer::Buffer inputBuffer_diff;
	FzbRenderer::Buffer inputBuffer_spec;
	VkExtent2D imageSize;

	HANDLE startSemaphoreHandle;
	HANDLE endSemaphoreHandle;
};

class KPCNNDenoiser {
public:
	KPCNNDenoiser() = default;
	virtual ~KPCNNDenoiser() = default;

	KPCNNDenoiser(KPCNNDenoiser_CreateInfo createInfo);
	KPCNNDenoiser& operator=(KPCNNDenoiser&&) noexcept;
	void denoising(uint64_t waitTimeline = 1);

	void clean();

private:
	KPCNNDenoiser_CreateInfo setting;

	cudaExternalMemory_t inputBufferExtMem_diff;
	float* inputBuffer_diff;

	cudaExternalMemory_t inputBufferExtMem_spec;
	float* inputBuffer_spec;

	cudaExternalSemaphore_t startSemaphore;
	cudaExternalSemaphore_t endSemaphore;

	cudaStream_t stream = nullptr;

	FzbRenderer::Model KPCNN_diff;
	FzbRenderer::Model KPCNN_spec;
};

#endif