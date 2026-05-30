#include "./ImageRecognition.cuh"
#include <common/path_utils.hpp>
#include "common/CUDA/commonCudaFunction.cuh"

//------------------------------------------------------------------------------------------------------------------------
ImageRecognition::ImageRecognition(ImageRecognition_CreateInfo createInfo) {
	if (getCudaDeviceForVulkanPhysicalDevice(createInfo.physicalDevice) == cudaInvalidDeviceId)
		throw std::runtime_error("CUDA与Vulkan用的不是同一个GPU！！！");

	FzbRenderer::Buffer& buffer = createInfo.buffer;
	inputTensorExtMem = importVulkanMemoryObjectFromNTHandle(buffer.handle, buffer.buffer.bufferSize, false);
	inputTensor = (float*)mapBufferOntoExternalMemory(inputTensorExtMem, 0, buffer.buffer.bufferSize);

	startSemaphore = importVulkanSemaphoreObjectFromNTHandle(createInfo.startSemaphoreHandle);
	endSemaphore = importVulkanSemaphoreObjectFromNTHandle(createInfo.endSemaphoreHandle);

	CHECK(cudaStreamCreate(&stream));

	std::string enginePath = FzbRenderer::getProjectRootDir().string() + "src/renderer/NPMPathGuiding/models/resnet34.engine";
	resNet34 = std::move(FzbRenderer::Model({ enginePath }));
}

//resnet34 识别1000个类型，所以blockSize = 1024可以在一个线程组中处理
__global__ void printfOutput(float* output, int outputCount) {
	__shared__ int groupWarpMaxProbabilitiesU[32];	//No initialization is required. Each warp will put information
	__shared__ uint groupWarpMaxProbabilityTypes[32];

	uint threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	//uint activeMask = __ballot_sync(0xffffffff, threadIndex >= outputCount);
	//if (threadIndex >= outputCount) return;

	uint warpIndex = threadIndex / 32;
	uint warpLane = threadIndex & 31;

	uint type = threadIndex;
	float typeProbability = threadIndex >= outputCount ? -INFINITY : output[threadIndex];
	int typeProbabilityU = FloatToOrderedInt(typeProbability);
	int warpMaxProbabilityU = __reduce_max_sync(0xffffffff, typeProbabilityU);
	if (warpLane == 0) groupWarpMaxProbabilitiesU[warpIndex] = warpMaxProbabilityU;

	unsigned warpSameMaxProbabilityMask = __ballot_sync(0xffffffff, typeProbabilityU == warpMaxProbabilityU);
	int warpSameMaxProbabilityIndex = __popc(warpSameMaxProbabilityMask & ((1u << warpLane) - 1));
	if (typeProbabilityU == warpMaxProbabilityU && warpSameMaxProbabilityIndex == 0)
		groupWarpMaxProbabilityTypes[warpIndex] = type;
	__syncthreads();

	warpMaxProbabilityU = groupWarpMaxProbabilitiesU[warpLane];
	int groupWarpMaxProbabilityU = __reduce_max_sync(0xffffffff, warpMaxProbabilityU);

	if (warpIndex == 0) {
		unsigned groupSameMaxProbabilityMask = __ballot_sync(0xffffffff, warpMaxProbabilityU == groupWarpMaxProbabilityU);
		int groupSameMaxProbabilityIndex = __popc(groupSameMaxProbabilityMask & ((1u << warpLane) - 1));
		if (warpMaxProbabilityU == groupWarpMaxProbabilityU && groupSameMaxProbabilityIndex == 0)
			printf("type = %d;  probability = %f\n", groupWarpMaxProbabilityTypes[warpLane], OrderedIntToFloat(groupWarpMaxProbabilityU));
	}
}
void ImageRecognition::recognition(uint64_t waitTimeline) {
	CHECK(waitExternalSemaphore(startSemaphore, stream, waitTimeline));

	FzbRenderer::InputTensorInfo inputTensorInfo = {
		"input",
		(void*)inputTensor,
		{ 1, 3, 224, 224 }
	};
	resNet34.infer({ inputTensorInfo }, stream);

	uint outputCount = resNet34.outputTensorSizes["output"];
	uint blockSize = 1024;
	uint gridSize = (outputCount + blockSize - 1) / blockSize;
	printfOutput<<<gridSize, blockSize, 0, stream >>>((float*)resNet34.outputTensors["output"], outputCount);

	CHECK(signalExternalSemaphore(endSemaphore, stream, waitTimeline));
}

void ImageRecognition::clean() {
	CHECK(cudaFree(inputTensor));
	CHECK(cudaDestroyExternalMemory(inputTensorExtMem));

	cudaDestroyExternalSemaphore(startSemaphore);
	cudaDestroyExternalSemaphore(endSemaphore);

	CHECK(cudaStreamDestroy(stream));

	resNet34.clean();
}

ImageRecognition& ImageRecognition::operator=(ImageRecognition&& other) noexcept {
	if (this != &other) {
		resNet34 = std::move(other.resNet34);
		inputTensor = other.inputTensor;
		inputTensorExtMem = other.inputTensorExtMem;
		startSemaphore = other.startSemaphore;
		endSemaphore = other.endSemaphore;
		stream = other.stream;
	}
	return *this;
}