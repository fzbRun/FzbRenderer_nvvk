#include "./KPCNN.cuh"
#include <common/path_utils.hpp>
#include "common/CUDA/commonCudaFunction.cuh"

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

	startSemaphore = importVulkanSemaphoreObjectFromNTHandle(createInfo.startSemaphoreHandle);
	endSemaphore = importVulkanSemaphoreObjectFromNTHandle(createInfo.endSemaphoreHandle);

	CHECK(cudaStreamCreate(&stream));

	std::string enginePath = FzbRenderer::getProjectRootDir().string() + "src/renderer/PathTracing_KPCNNDenoising/models/kpcnn_diffuse.engine";
	KPCNN_diff = std::move(FzbRenderer::Model({ enginePath }));

	enginePath = FzbRenderer::getProjectRootDir().string() + "src/renderer/PathTracing_KPCNNDenoising/models/kpcnn_specular.engine";
	KPCNN_spec = std::move(FzbRenderer::Model({ enginePath }));
}

__global__ void denoisingCuda(float* inputBuffer_diff, float* inputBuffer_spec, float* kernel_diff, float* kernel_spec, uint imageWidth, uint imageHeight, uint kernelSize) {
	extern __shared__ float3 groupImageColor[];

	uint threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
	uint threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;
	if()


	if (x >= imageWidth || y >= imageHeight)
		return;
	uint pixelIndex = y * imageWidth + x;
	uint kernelIndex = pixelIndex * kernelSize * kernelSize;
	// 这里可以根据需要进行去噪处理，以下是一个简单的示例
	float diffValue = inputBuffer_diff[pixelIndex];
	float specValue = inputBuffer_spec[pixelIndex];
	// 简单的加权平均作为示例
	float denoisedValue = (diffValue + specValue) / 2.0f;
	// 将结果写回到输出缓冲区
	inputBuffer_diff[pixelIndex] = denoisedValue; // 或者写入另一个输出缓冲区
}
void KPCNNDenoiser::denoising(uint64_t waitTimeline) {
	CHECK(waitExternalSemaphore(startSemaphore, stream, waitTimeline));

	FzbRenderer::InputTensorInfo inputTensorInfo = {
		"input",
		(void*)inputBuffer_diff,
		{ 1, 27, (int)setting.imageSize.height, (int)setting.imageSize.width }
	};
	KPCNN_diff.infer({ inputTensorInfo }, stream);

	inputTensorInfo = {
		"input",
		(void*)inputBuffer_spec,
		{ 1, 27, (int)setting.imageSize.height, (int)setting.imageSize.width }
	};
	KPCNN_spec.infer({ inputTensorInfo }, stream);

	uint outputSize = KPCNN_diff.outputTensorSizes["kernel_weights"];
	uint imageSize = setting.imageSize.width * setting.imageSize.height;
	uint kernelSize = sqrt(outputSize / imageSize);

	dim3 blockSize = dim3(16, 16, 1);
	dim3 gridSize = dim3((setting.imageSize.width + blockSize.x - 1) / blockSize.x, (setting.imageSize.height + blockSize.y - 1) / blockSize.y, 1);

	uint groupSharedMemorySize = (blockSize.x + kernelSize / 2) * (blockSize.y + kernelSize / 2) * sizeof(float3);

	denoisingCuda << <gridSize, blockSize, groupSharedMemorySize, stream >> > ((float*)resNet34.outputTensors["output"], outputCount);

	CHECK(signalExternalSemaphore(endSemaphore, stream, waitTimeline));
}

void KPCNNDenoiser::clean() {
	CHECK(cudaFree(inputBuffer_diff));
	CHECK(cudaDestroyExternalMemory(inputBufferExtMem_diff));
	CHECK(cudaFree(inputBuffer_spec));
	CHECK(cudaDestroyExternalMemory(inputBufferExtMem_spec));

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

		startSemaphore = other.startSemaphore;
		endSemaphore = other.endSemaphore;
		stream = other.stream;
	}
	return *this;
}