#pragma once

#include "common/CUDA/vulkanCudaInterop.cuh"
#include <NvInfer.h>
#include <unordered_map>

#include <torch/torch.h>
#include <torch/script.h>

#ifndef FZBRENDERER_PATH_TRACING_KPCNN_DENOISING_DATASET_CUH
#define FZBRENDERER_PATH_TRACING_KPCNN_DENOISING_DATASET_CUH

struct KPCNNSample {
    torch::Tensor input_diff;
    torch::Tensor input_spec;
    torch::Tensor gtDiff;
    torch::Tensor gtSpec;
    torch::Tensor gtColor;
    torch::Tensor sampleAlbedo;
    torch::Tensor sampleColor;
    torch::Tensor sampleNormal;
};

struct KPCNNDataSetSampleBuffers {
    FzbRenderer::Buffer inputBuffer_diff;
    FzbRenderer::Buffer inputBuffer_spec;
    FzbRenderer::Buffer gtDiff;
    FzbRenderer::Buffer gtSpec;
    FzbRenderer::Buffer gtColor;
    FzbRenderer::Buffer sampleAlbedo;
    FzbRenderer::Buffer sampleColor;
    FzbRenderer::Buffer sampleNormal;
};
class KPCNNDataSet {
public:
    KPCNNDataSet() = default;
    ~KPCNNDataSet() = default;

    KPCNNDataSet(std::string& folder);

private:
    std::vector<KPCNNSample> samples;
};

#endif