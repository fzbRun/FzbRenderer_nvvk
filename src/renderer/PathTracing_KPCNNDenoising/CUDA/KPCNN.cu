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
	inputBufferExtMem_diff = importVulkanMemoryObjectFromNTHandle(buffer_diff.handle, buffer_diff.allocMemTotalSize, false);
	inputBuffer_diff = (float*)mapBufferOntoExternalMemory(inputBufferExtMem_diff, buffer_diff.allocMemOffset, buffer_diff.allocMemSize);

	FzbRenderer::Buffer& buffer_spec = createInfo.inputBuffer_spec;
	inputBufferExtMem_spec = importVulkanMemoryObjectFromNTHandle(buffer_spec.handle, buffer_spec.allocMemTotalSize, false);
	inputBuffer_spec = (float*)mapBufferOntoExternalMemory(inputBufferExtMem_spec, buffer_spec.allocMemOffset, buffer_spec.allocMemSize);

	FzbRenderer::Buffer& buffer_albedo = createInfo.albedoBuffer;
	albedoBufferExtMem_diff = importVulkanMemoryObjectFromNTHandle(buffer_albedo.handle, buffer_albedo.allocMemTotalSize, false);
	albedoBuffer = (float3*)mapBufferOntoExternalMemory(albedoBufferExtMem_diff, buffer_albedo.allocMemOffset, buffer_albedo.allocMemSize);

	fromVulkanImageToCudaSurface(
		createInfo.physicalDevice,
		createInfo.outputImage,
		false, imageExtMem, imageMipmap, imageObject
	);

	startSemaphore = importVulkanSemaphoreObjectFromNTHandle(createInfo.startSemaphoreHandle);
	endSemaphore = importVulkanSemaphoreObjectFromNTHandle(createInfo.endSemaphoreHandle);

	CHECK(cudaStreamCreate(&stream));

	std::string enginePath = FzbRenderer::getProjectRootDir().string() + "src/renderer/PathTracing_KPCNNDenoising/models/kpcnn_diffuse.engine";
	FzbRenderer::ModelCreateInfo modelCreateInfo = {
		enginePath,                                           // enginePath
		nvinfer1::BuilderFlag::kFP16,                         // precision
		{1, 27, setting.imageSize.height, setting.imageSize.width}, // inputShape_min
		{1, 27, setting.imageSize.height, setting.imageSize.width}, // inputShape_opt
		{1, 27, setting.imageSize.height, setting.imageSize.width}, // inputShape_max
		{1, 21 * 21, setting.imageSize.height, setting.imageSize.width} // outputShape
	};
	KPCNN_diff = std::move(FzbRenderer::Model(modelCreateInfo));

	enginePath = FzbRenderer::getProjectRootDir().string() + "src/renderer/PathTracing_KPCNNDenoising/models/kpcnn_specular.engine";
	modelCreateInfo.enginePath = enginePath;
	KPCNN_spec = std::move(FzbRenderer::Model(modelCreateInfo));
}

__global__ void denoisingCuda(
	uint imageWidth, uint imageHeight, uint kernelSize,
	float* inputBuffer_diff, float* inputBuffer_spec,
	float* kernel_diff, float* kernel_spec,
	float3* albedoBuffer,
	cudaSurfaceObject_t outputImage
) {
	extern __shared__ float3 groupImageDatas[];	//16KB

	int halo = kernelSize / 2;
	int smemWidth = blockDim.x + 2 * halo;
	int smemHeight = blockDim.y + 2 * halo;

	float3* groupImageIrradiance = groupImageDatas;
	float3* groupImageSpecular = groupImageDatas + smemWidth * smemHeight;

	uint threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
	uint threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
	uint2 threadIndex = make_uint2(threadIndexX, threadIndexY);
	uint2 groupThreadIndex = make_uint2(blockDim.x * blockIdx.x, blockDim.y * blockIdx.y);
	uint pixelIndex = threadIndexY * imageWidth + threadIndexX;

	uint2 groupNeighborPixelStartIndex = groupThreadIndex - make_uint2(halo, halo);
	for (int y = threadIdx.y; y < smemHeight; y += blockDim.y) {
		uint neighborPixelIndexY = groupNeighborPixelStartIndex.y + y;
		for (int x = threadIdx.x; x < smemWidth; x += blockDim.x) {
			uint neighborPixelIndexX = groupNeighborPixelStartIndex.x + x;

			float3 neighborPixelIrradiance = make_float3(0.0f), neighborPixelSpecular = make_float3(0.0f);
			if (neighborPixelIndexX >= 0 && neighborPixelIndexX < imageWidth && neighborPixelIndexY >= 0 && neighborPixelIndexY < imageHeight) {
				neighborPixelIrradiance = reinterpret_cast<float3*>(inputBuffer_diff)[neighborPixelIndexY * imageWidth + neighborPixelIndexX];
				neighborPixelSpecular = reinterpret_cast<float3*>(inputBuffer_spec)[neighborPixelIndexY * imageWidth + neighborPixelIndexX];
			}

			uint groupImageColorIndex = y * smemWidth + x;
			groupImageIrradiance[groupImageColorIndex] = neighborPixelIrradiance;
			groupImageSpecular[groupImageColorIndex] = neighborPixelSpecular;
		}
	}
	__syncthreads();
	if (threadIndexX >= imageWidth || threadIndexY >= imageHeight) return;

	float3 pixelIrradiance = make_float3(0.0f), pixelSpecular = make_float3(0.0f);
	for (int y = 0; y < kernelSize; ++y) {
		uint neighborPixelIndexY = threadIndexY - halo + y;
		for (int x = 0; x < kernelSize; ++x) {
			uint neighborPixelIndexX = threadIndexX - halo + x;
			uint groupNeighborPixelDataIndex = (neighborPixelIndexY - groupNeighborPixelStartIndex.y) * smemWidth +
				(neighborPixelIndexX - groupNeighborPixelStartIndex.x);
			float3 neighborPixelIrradiance = groupImageIrradiance[groupNeighborPixelDataIndex];
			float3 neighborPixelSpecular = groupImageSpecular[groupNeighborPixelDataIndex];

			//kernel数组是[1, C, H, W]形式的，所有像素的第i个核元素连续存储
			uint kernelIndex = (y * kernelSize + x) * imageWidth * imageHeight + pixelIndex;
			float kernelValue_diff = kernel_diff[kernelIndex];
			float kernelValue_spec = kernel_spec[kernelIndex];

			pixelIrradiance += neighborPixelIrradiance * kernelValue_diff;
			pixelSpecular += neighborPixelSpecular * kernelValue_spec;
		}
	}

	float3 pixelAlbedo = albedoBuffer[pixelIndex];
	float3 filteredPixelColor = pixelIrradiance * (pixelAlbedo + eps_kpcnn) + make_float3(exp(pixelSpecular.x), exp(pixelSpecular.y), exp(pixelSpecular.z)) - 1.0f;
	surf2Dwrite(make_float4(filteredPixelColor, 1.0f), outputImage, threadIndexX * sizeof(float4), threadIndexY);
	//surf2Dwrite(make_float4(pixelAlbedo, 1.0f), outputImage, threadIndexX * sizeof(float4), threadIndexY);
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

	uint groupSharedMemorySize = (blockSize.x + kernelSize / 2 * 2) * (blockSize.y + kernelSize / 2 * 2) * sizeof(float3) * 2;

	denoisingCuda << <gridSize, blockSize, groupSharedMemorySize, stream >> > (
		setting.imageSize.width, setting.imageSize.height, 21,
		inputBuffer_diff, inputBuffer_spec,
		kernel_diff, kernel_spec,
		albedoBuffer,
		imageObject
		);

	CHECK(signalExternalSemaphore(endSemaphore, stream, waitTimeline));
}

void KPCNNDenoiser::clean() {
	CHECK(cudaFree(inputBuffer_diff));
	CHECK(cudaDestroyExternalMemory(inputBufferExtMem_diff));
	CHECK(cudaFree(inputBuffer_spec));
	CHECK(cudaDestroyExternalMemory(inputBufferExtMem_spec));

	CHECK(cudaFree(albedoBuffer));
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
		setting = other.setting;

		KPCNN_diff = std::move(other.KPCNN_diff);
		KPCNN_spec = std::move(other.KPCNN_spec);

		inputBuffer_diff = other.inputBuffer_diff;
		inputBufferExtMem_diff = other.inputBufferExtMem_diff;
		inputBuffer_spec = other.inputBuffer_spec;
		inputBufferExtMem_spec = other.inputBufferExtMem_spec;

		albedoBuffer = other.albedoBuffer;
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