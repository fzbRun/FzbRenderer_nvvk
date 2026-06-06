#include "./KPCNN.cuh"
#include <common/path_utils.hpp>
#include "common/CUDA/commonCudaFunction.cuh"
#include "../PathTracing_KPCNNDenosingShaderio.h"

//------------------------------------------------------------------------------------------------------------------------
KPCNNDenoiser::KPCNNDenoiser(KPCNNDenoiser_CreateInfo createInfo) {
	if (getCudaDeviceForVulkanPhysicalDevice(createInfo.physicalDevice) == cudaInvalidDeviceId)
		throw std::runtime_error("CUDA与Vulkan用的不是同一个GPU！！！");

	setting = createInfo;

	FzbRenderer::Buffer& buffer_diff = createInfo.inputBuffer_diff;
	inputBufferExtMem_diff = importVulkanMemoryObjectFromNTHandle(buffer_diff.handle, buffer_diff.buffer.bufferSize, false);
	inputBuffer_diff = (float*)mapBufferOntoExternalMemory(inputBufferExtMem_diff, 0, buffer_diff.buffer.bufferSize);

	FzbRenderer::Buffer& buffer_spec = createInfo.inputBuffer_spec;
	inputBufferExtMem_spec = importVulkanMemoryObjectFromNTHandle(buffer_spec.handle, buffer_spec.buffer.bufferSize, false);
	inputBuffer_spec = (float*)mapBufferOntoExternalMemory(inputBufferExtMem_spec, 0, buffer_spec.buffer.bufferSize);

	FzbRenderer::Buffer& buffer_albedo = createInfo.albedoBuffer;
	albedoBufferExtMem_diff = importVulkanMemoryObjectFromNTHandle(buffer_albedo.handle, buffer_albedo.buffer.bufferSize, false);
	albedoBuffer_diff = (float*)mapBufferOntoExternalMemory(albedoBufferExtMem_diff, 0, buffer_albedo.buffer.bufferSize);

	uint32_t imageSize = createInfo.outputImage.imageSize;
	fromVulkanImageToCudaSurface(
		createInfo.physicalDevice,
		createInfo.outputImage, createInfo.outputImage.handle, imageSize,
		false, imageExtMem, imageMipmap, imageObject
	);

	startSemaphore = importVulkanSemaphoreObjectFromNTHandle(createInfo.startSemaphoreHandle);
	endSemaphore = importVulkanSemaphoreObjectFromNTHandle(createInfo.endSemaphoreHandle);

	CHECK(cudaStreamCreate(&stream));

	std::string enginePath = FzbRenderer::getProjectRootDir().string() + "src/renderer/PathTracing_KPCNNDenoising/models/kpcnn_diffuse.engine";
	KPCNN_diff = std::move(FzbRenderer::Model({ enginePath }));

	enginePath = FzbRenderer::getProjectRootDir().string() + "src/renderer/PathTracing_KPCNNDenoising/models/kpcnn_specular.engine";
	KPCNN_spec = std::move(FzbRenderer::Model({ enginePath }));
}

__global__ void denoisingCuda(
	uint imageWidth, uint imageHeight, uint kernelSize,
	float* inputBuffer_diff, float* inputBuffer_spec,
	float* kernel_diff, float* kernel_spec,
	float3* albedoBuffer,
	cudaSurfaceObject_t outputImage
) {
	extern __shared__ float3 groupImageColors[];	//16KB
	float3* groupImageColors_diff = (float3*)groupImageColors;
	float3* groupImageColors_spec = (float3*)(groupImageColors + (blockDim.x + kernelSize / 2) * (blockDim.y + kernelSize / 2));

	uint threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
	uint threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
	uint2 threadIndex = uint2(threadIndexX, threadIndexY);
	uint2 groupThreadIndex = uint2(blockDim.x * blockIdx.x, blockDim.y * blockIdx.y);
	uint pixelIndex = threadIndexY * imageWidth + threadIndexX;

	uint halfKernelSize = kernelSize / 2;
	uint2 groupNeighborPixelStartIndex = groupThreadIndex - uint2(halfKernelSize, halfKernelSize);
	for (int y = threadIdx.y; y < blockDim.y + 2 * halfKernelSize; y += blockDim.y) {
		uint neighborPixelIndexY = groupNeighborPixelStartIndex.y + y;
		for (int x = threadIdx.x; x < blockDim.x + 2 * halfKernelSize; x += blockDim.x) {
			uint neighborPixelIndexX = groupNeighborPixelStartIndex.x + x;

			float3 neighborPixelColor_diff = float3(0.0f), neighborPixelColor_spec = float3(0.0f);
			if (neighborPixelIndexX >= 0 && neighborPixelIndexX < imageWidth && neighborPixelIndexY >= 0 && neighborPixelIndexY < imageHeight) {
				neighborPixelColor_diff = reinterpret_cast<float3*>(inputBuffer_diff)[neighborPixelIndexY * imageWidth + neighborPixelIndexX];
				neighborPixelColor_spec = reinterpret_cast<float3*>(inputBuffer_spec)[neighborPixelIndexY * imageWidth + neighborPixelIndexX];
			}

			uint groupImageColorIndex = y * (blockDim.x + 2 * halfKernelSize) + x;
			groupImageColors_diff[groupImageColorIndex] = neighborPixelColor_diff;
			groupImageColors_spec[groupImageColorIndex] = neighborPixelColor_spec;
		}
	}
	__syncthreads();

	float3 pixelColor_diff = float3(0.0f), pixelColor_spec = float3(0.0f);
	for (int y = 0; y < kernelSize; ++y) {
		uint neighborPixelIndexY = threadIndexY - halfKernelSize + y;
		for (int x = 0; x < kernelSize; ++x) {
			uint neighborPixelIndexX = threadIndexX - halfKernelSize + x;
			uint groupNeighborPixelColorIndex = (neighborPixelIndexY - groupNeighborPixelStartIndex.y) * (blockDim.x + 2 * halfKernelSize) +
				(neighborPixelIndexX - groupNeighborPixelStartIndex.x);
			float3 neighborPixelColor_diff = groupImageColors_diff[groupNeighborPixelColorIndex];
			float3 neighborPixelColor_spec = groupImageColors_spec[groupNeighborPixelColorIndex];

			//kernel数组是[1, C, H, W]形式的，所有像素的第i个核元素连续存储
			uint kernelIndex = (y * kernelSize + x) * imageWidth * imageHeight + pixelIndex;
			float kernelValue_diff = kernel_diff[kernelIndex];
			float kernelValue_spec = kernel_spec[kernelIndex];

			pixelColor_diff += neighborPixelColor_diff * kernelValue_diff;
			pixelColor_spec += neighborPixelColor_spec * kernelValue_spec;
		}
	}

	float3 pixelAlbedo = albedoBuffer[pixelIndex];
	float3 filteredPixelColor = (pixelColor_diff + eps_kpcnn) * pixelAlbedo + float3(exp(pixelColor_spec.x), exp(pixelColor_spec.y), exp(pixelColor_spec.z)) - 1.0;
	surf2Dwrite({ filteredPixelColor , 1.0f}, outputImage, threadIndexX * sizeof(float4), threadIndexY);
}
void KPCNNDenoiser::denoising(uint64_t waitTimeline) {
	CHECK(waitExternalSemaphore(startSemaphore, stream, waitTimeline));

	FzbRenderer::InputTensorInfo inputTensorInfo = {
		"input",
		(void*)inputBuffer_diff,
		{ 1, 27, (int)setting.imageSize.height, (int)setting.imageSize.width }
	};
	KPCNN_diff.infer({ inputTensorInfo }, stream);
	float* kernel_diff = (float*)KPCNN_diff.outputTensors["kernel_weights"];

	inputTensorInfo = {
		"input",
		(void*)inputBuffer_spec,
		{ 1, 27, (int)setting.imageSize.height, (int)setting.imageSize.width }
	};
	KPCNN_spec.infer({ inputTensorInfo }, stream);
	float* kernel_spec = (float*)KPCNN_spec.outputTensors["kernel_weights"];

	uint outputSize = KPCNN_diff.outputTensorSizes["kernel_weights"];
	uint imageSize = setting.imageSize.width * setting.imageSize.height;
	uint kernelSize = sqrt(outputSize / imageSize);

	dim3 blockSize = dim3(16, 16, 1);
	dim3 gridSize = dim3((setting.imageSize.width + blockSize.x - 1) / blockSize.x, (setting.imageSize.height + blockSize.y - 1) / blockSize.y, 1);

	uint groupSharedMemorySize = (blockSize.x + kernelSize / 2) * (blockSize.y + kernelSize / 2) * sizeof(float3) * 2;

	denoisingCuda << <gridSize, blockSize, groupSharedMemorySize, stream >> > (
		setting.imageSize.width, setting.imageSize.height,
		inputBuffer_diff, inputBuffer_spec,
		kernel_diff, kernel_spec,
		albedoBuffer_diff,
		imageObject
		);

	CHECK(signalExternalSemaphore(endSemaphore, stream, waitTimeline));
}

void KPCNNDenoiser::clean() {
	CHECK(cudaFree(inputBuffer_diff));
	CHECK(cudaDestroyExternalMemory(inputBufferExtMem_diff));
	CHECK(cudaFree(inputBuffer_spec));
	CHECK(cudaDestroyExternalMemory(inputBufferExtMem_spec));

	CHECK(cudaFree(albedoBuffer_diff));
	CHECK(cudaDestroyExternalMemory(albedoBufferExtMem_diff));

	CHECK(cudaDestroyTextureObject(imageObject));
	CHECK(cudaFreeMipmappedArray(imageMipmap));
	CHECK(cudaDestroyExternalMemory(imageExtMem));

	cudaDestroyExternalSemaphore(startSemaphore);
	cudaDestroyExternalSemaphore(endSemaphore);

	CHECK(cudaStreamDestroy(stream));

	KPCNN_diff.clean();
	KPCNN_spec.clean();
}
KPCNNDenoiser& KPCNNDenoiser::operator=(KPCNNDenoiser&& other) noexcept {
	if (this != &other) {
		KPCNN_diff = std::move(other.KPCNN_diff);
		KPCNN_spec = std::move(other.KPCNN_spec);

		inputBuffer_diff = other.inputBuffer_diff;
		inputBufferExtMem_diff = other.inputBufferExtMem_diff;
		inputBuffer_spec = other.inputBuffer_spec;
		inputBufferExtMem_spec = other.inputBufferExtMem_spec;

		albedoBuffer_diff = other.albedoBuffer_diff;
		albedoBufferExtMem_diff = other.albedoBufferExtMem_diff;

		imageExtMem = other.imageExtMem;
		imageMipmap = other.imageMipmap;
		imageObject = other.imageObject;

		startSemaphore = other.startSemaphore;
		endSemaphore = other.endSemaphore;
		stream = other.stream;
	}
	return *this;
}