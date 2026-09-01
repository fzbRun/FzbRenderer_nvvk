#include "./GPUInfo.cuh"

void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << "Error: " << cudaGetErrorString(err) << " (" << func << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void getGPUInfo() {
    int deviceCount = 0;
    // 获取可用的 CUDA 设备数量
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cout << "没有找到支持 CUDA 的设备。" << std::endl;
        return;
    }

    std::cout << "找到 " << deviceCount << " 个支持 CUDA 的设备。" << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, dev));

        std::cout << "\n================== 设备 " << dev << ": " << prop.name << " ==================" << std::endl;
        std::cout << "  计算能力:                " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  全局内存总量:             " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  多处理器数量 (SM):        " << prop.multiProcessorCount << std::endl;
        std::cout << "  CUDA 核心/SM (近似):     根据计算能力估算，实际核心数需查表" << std::endl; // 不直接给出数值，因为 prop 中没有直接的核心数
        std::cout << "  每个 SM 的最大线程数:      " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  每个块的最大线程数:        " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  每个 SM 的最大线程束数量:  " << prop.maxThreadsPerMultiProcessor / prop.warpSize << std::endl;
        std::cout << "  线程束大小:               " << prop.warpSize << std::endl;
        std::cout << "  每个块的最大共享内存:      " << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "  每个 SM 的最大共享内存:    " << prop.sharedMemPerMultiprocessor / 1024.0 << " KB" << std::endl;
        std::cout << "  每个块的最大寄存器数量:    " << prop.regsPerBlock << std::endl;
        std::cout << "  每个 SM 的最大寄存器数量:  " << prop.regsPerMultiprocessor << std::endl;
        std::cout << "  最大网格维度:             [" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;
        std::cout << "  最大块维度:               [" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]" << std::endl;
        std::cout << "  纹理对齐要求:             " << prop.textureAlignment << " bytes" << std::endl;
        //std::cout << "  设备是否支持统一内存:      " << (prop.unifiedAddressing ? "是" : "否") << std::endl;
        std::cout << "  内存总线位宽:              " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "  峰值内存时钟频率:          " << prop.memoryClockRate << " kHz" << std::endl;
        std::cout << "  L2 缓存大小:               " << prop.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << "  最大线程块中的线程数 (X):  " << prop.maxThreadsDim[0] << std::endl;
        // 更多属性可以参考 cudaDeviceProp 结构体定义
    }

    return;
}