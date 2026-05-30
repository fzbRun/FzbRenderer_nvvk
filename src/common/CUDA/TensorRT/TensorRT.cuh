#pragma once

#include "common/CUDA/vulkanCudaInterop.cuh"
#include <NvInfer.h>
#include <unordered_map>

#ifndef FZBRENDERER_TENSORRT_CUH
#define FZBRENDERER_TENSORRT_CUH

namespace FzbRenderer {
template<typename T>
struct TrtDeleter {
	void operator()(T* obj) const {
		if (obj) delete obj;
	}
};
template<typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

struct TrtModel {
	TrtUniquePtr<nvinfer1::IRuntime> runtime;
	TrtUniquePtr<nvinfer1::ICudaEngine> engine;
	TrtUniquePtr<nvinfer1::IExecutionContext> context;
};
struct ModelCreateInfo {
	std::string enginePath;
	nvinfer1::BuilderFlag precision = nvinfer1::BuilderFlag::kFP16;
};
struct InputTensorInfo {
	std::string name;
	void* inputTensor;			//device point
	std::vector<int> shape;		//batch channels height width;
};

class Model {
public:
	Model() = default;
	virtual ~Model() = default;

	Model(ModelCreateInfo createInfo);
	Model& operator=(Model&&) noexcept;
	void clean();

	void infer(std::vector<InputTensorInfo> inputInfos, cudaStream_t stream);

	TrtModel model;
	std::unordered_map<std::string, void*> outputTensors;
	std::unordered_map<std::string, int> outputTensorSizes;
private:
	std::vector<char> loadModelData(const std::string& onnxPath);
	ModelCreateInfo setting;
};
}


#endif
