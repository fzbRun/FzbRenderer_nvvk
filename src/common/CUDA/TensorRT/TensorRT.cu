#include "./TensorRT.cuh"

#include <NvOnnxParser.h>
#include <fstream>
#include <sstream>

class Logger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) noexcept override {
		if (severity <= Severity::kWARNING)
			std::cout << msg << std::endl;
	}
} gLogger;

std::vector<char> FzbRenderer::Model::loadModelData(const std::string& onnxPath){
    TrtUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(gLogger));
    if (!builder) throw std::runtime_error("Failed to create builder");

    const uint32_t explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    TrtUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicitBatch));
    if (!network) throw std::runtime_error("Failed to create network");

    TrtUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, gLogger));
    if (!parser) throw std::runtime_error("Failed to create parser");

    std::cout << "Parsing ONNX: " << onnxPath << std::endl;

    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))){
        std::stringstream ss;

        ss << "Failed to parse ONNX:\n";

        for (int i = 0; i < parser->getNbErrors(); ++i) {
            ss << parser->getError(i)->desc() << "\n";
        }

        throw std::runtime_error(ss.str());
    }

    TrtUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) throw std::runtime_error("Failed to create config");

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
    //if (builder->platformHasFastFp16()) config->setFlag(nvinfer1::BuilderFlag::kFP16);
    config->setFlag(setting.precision);

    bool hasDynamicInput = false;
    auto* profile = builder->createOptimizationProfile();
    if (!profile) throw std::runtime_error("Failed to create optimization profile");
    for (int inputIndex = 0; inputIndex < network->getNbInputs(); ++inputIndex){
        auto* inputTensor = network->getInput(inputIndex);
        if (!inputTensor) continue;

        auto dims = inputTensor->getDimensions();
        bool isDynamic = false;
        for (int d = 0; d < dims.nbDims; ++d) {
            if (dims.d[d] == -1) {
                isDynamic = true;
                break;
            }
        }
        if (!isDynamic) continue;

        hasDynamicInput = true;
        nvinfer1::Dims minDims = dims;
        nvinfer1::Dims optDims = dims;
        nvinfer1::Dims maxDims = dims;

        for (int d = 0; d < dims.nbDims; ++d){
            if (dims.d[d] == -1){
                // batch dimension
                if (d == 0){
                    minDims.d[d] = 1;
                    optDims.d[d] = 1;
                    maxDims.d[d] = 8;
                } else{
                    // spatial / sequence dimension
                    minDims.d[d] = 1;
                    optDims.d[d] = 224;
                    maxDims.d[d] = 1024;
                }
            }
        }
        std::cout << "Dynamic input detected: " << inputTensor->getName() << std::endl;

        profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMIN, minDims);
        profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kOPT, optDims);
        profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMAX, maxDims);

        auto printDims = [](const char* name, const nvinfer1::Dims& d){
                std::cout << name << ": [";
                for (int i = 0; i < d.nbDims; ++i){
                    std::cout << d.d[i];
                    if (i + 1 < d.nbDims) std::cout << ", ";
                }
                std::cout << "]\n";
            };
        printDims("MIN", minDims);
        printDims("OPT", optDims);
        printDims("MAX", maxDims);
    }
    if (hasDynamicInput) config->addOptimizationProfile(profile);

    std::cout << "Building TensorRT engine..." << std::endl;

    TrtUniquePtr<nvinfer1::IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
    if (!serialized) throw std::runtime_error("Failed to build engine");

    std::vector<char> engineData(serialized->size());
    memcpy(engineData.data(), serialized->data(), serialized->size());

    return engineData;
}
FzbRenderer::Model::Model(ModelCreateInfo createInfo) {
	this->setting = createInfo;

    std::vector<char> engineData;
    std::ifstream engineFile(createInfo.enginePath, std::ios::binary | std::ios::ate);
    if (engineFile.good()){
        size_t size = static_cast<size_t>(engineFile.tellg());
        engineData.resize(size);
        engineFile.seekg(0, std::ios::beg);
        engineFile.read(engineData.data(), size);
        engineFile.close();
        std::cout << "Loaded engine: " << createInfo.enginePath << std::endl;
    }else{
        std::cout << "Engine not found, building from ONNX..." << std::endl;
        std::string onnxPath = createInfo.enginePath.substr(0, createInfo.enginePath.find_last_of('.')) + ".onnx";
        engineData = loadModelData(onnxPath);

        std::ofstream outFile(createInfo.enginePath, std::ios::binary);
        outFile.write(engineData.data(), engineData.size());
        outFile.close();

        std::cout << "Saved engine: " << createInfo.enginePath << std::endl;
    }

    model.runtime.reset(nvinfer1::createInferRuntime(gLogger));
    if (!model.runtime) throw std::runtime_error("Failed to create runtime");

    model.engine.reset(model.runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!model.engine) throw std::runtime_error("Failed to deserialize engine");

    model.context.reset(model.engine->createExecutionContext());
    if (!model.context) throw std::runtime_error("Failed to create execution context");

    //暂时不考虑动态output shape
    int nbTensors = model.engine->getNbIOTensors();
    for (int i = 0; i < nbTensors; ++i) {
        const char* tensorName = model.engine->getIOTensorName(i);
        if (model.engine->getTensorIOMode(tensorName) != nvinfer1::TensorIOMode::kOUTPUT) continue;

        nvinfer1::Dims dims = model.context->getTensorShape(tensorName);
        size_t count = 1;
        for (int d = 0; d < dims.nbDims; ++d) {
            if (dims.d[d] < 0) {
                if (d == 0) continue;   //batch temporarily set to 1
                throw std::runtime_error(std::string("Output shape still dynamic: ") + tensorName);
            }
            count *= static_cast<size_t>(dims.d[d]);
        }

        void* dPtr = nullptr;
        CHECK(cudaMalloc(&dPtr, count * sizeof(float)));

        if (!model.context->setTensorAddress(tensorName, dPtr))
            throw std::runtime_error(std::string("Failed to bind output: ") + tensorName);

		outputTensors.insert({ tensorName, dPtr });
        outputTensorSizes.insert({ tensorName, count });
    }
}
void FzbRenderer::Model::infer(std::vector<InputTensorInfo> inputInfos, cudaStream_t stream) {
    for (const auto& inputInfo : inputInfos){
        nvinfer1::Dims dims{};
        dims.nbDims = static_cast<int>(inputInfo.shape.size());
        for (int j = 0; j < dims.nbDims; ++j) dims.d[j] = inputInfo.shape[j];

        if (!model.context->setInputShape(inputInfo.name.c_str(), dims))
            throw std::runtime_error("Failed to set input shape: " + inputInfo.name);

        if (!model.context->setTensorAddress(inputInfo.name.c_str(), inputInfo.inputTensor))
            throw std::runtime_error("Failed to set tensor address: " + inputInfo.name);
    }

    //暂时不考虑动态output shape
    int nbTensors = model.engine->getNbIOTensors();
    for (int i = 0; i < nbTensors; ++i){
        const char* tensorName = model.engine->getIOTensorName(i);
        if (model.engine->getTensorIOMode(tensorName) != nvinfer1::TensorIOMode::kOUTPUT) continue;

        if (!model.context->setTensorAddress(tensorName, outputTensors[tensorName]))
            throw std::runtime_error(std::string("Failed to bind output: ") + tensorName);
    }

    if (!model.context->enqueueV3(stream)) throw std::runtime_error( "TensorRT enqueueV3 failed");
}

FzbRenderer::Model& FzbRenderer::Model::operator=(FzbRenderer::Model&& other) noexcept {
    if (this != &other) {
        model = std::move(other.model);
        outputTensors = std::move(other.outputTensors);
		outputTensorSizes = std::move(other.outputTensorSizes);
        setting = std::move(other.setting);
        other.outputTensors.clear();
        other.outputTensorSizes.clear();
    }
    return *this;
}

void FzbRenderer::Model::clean() {
    for (auto& [name, dPtr] : outputTensors) {
        if (dPtr) cudaFree(dPtr);
    }
	outputTensors.clear();
}

