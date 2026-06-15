/*
#include "./DataSet.cuh"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

KPCNNDataSetSampleBuffers createBuffers(const std::string& sampleFile, const std::string& gtFile) {

}

std::vector<KPCNNSample> get_cropped_patches(const std::string& sampleFile, const std::string& gtFile) {

}

KPCNNDataSet::KPCNNDataSet(std::string& folder) {
    for (const auto& entry : std::filesystem::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;

        const auto path = entry.path();
        const auto filename = path.filename().string();

        if (filename.rfind("sample", 0) != 0 || path.extension() != ".exr") continue;

        const auto num_start = std::string("sample").size();
        const auto num_end = filename.find('.');
        const std::string num = filename.substr(num_start, num_end - num_start);

        const std::string sample_name = (std::filesystem::path(folder) / ("sample" + num + ".exr")).string();
        const std::string gt_name = (std::filesystem::path(folder) / ("gt" + num + ".exr")).string();

        std::cout << sample_name << "  " << gt_name << "\n";
        std::vector<KPCNNSample> patches = get_cropped_patches(sample_name, gt_name);
        samples.insert(samples.end(), patches.begin(), patches.end());
    }
}
*/
