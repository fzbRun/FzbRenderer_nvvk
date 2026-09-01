#pragma once

#include "commonCudaFunction.cuh"

#ifndef COMMON_CUDA_FUNCTION_CU
#define COMMON_CUDA_FUNCTION_CU

double cpuSecond() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

void checkKernelFunction() {
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        printf("核函数启动失败: %s\n", cudaGetErrorString(launchErr));
        return; // 或适当的错误处理
    }
    else {
        printf("核函数已成功启动\n");
    }

    // 然后等待核函数完成并检查执行错误
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        printf("核函数执行错误: %s\n", cudaGetErrorString(syncErr));
    }
    else {
        printf("核函数已成功执行完成\n");
    }
}
void checkLaunchConfiguration(uint32_t gridSize, uint32_t blockSize, uint32_t sharedMemSize) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("启动配置检查:\n");
    printf("  网格大小: %u\n", gridSize);
    printf("  块大小: %u\n", blockSize);
    printf("  共享内存: %u bytes\n", sharedMemSize);
    printf("  设备限制:\n");
    printf("    - 每块最大线程数: %d\n", prop.maxThreadsPerBlock);
    printf("    - 每块最大共享内存: %zu bytes\n", prop.sharedMemPerBlock);
    printf("    - 最大网格维度: (%d, %d, %d)\n",
        prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    if (blockSize > prop.maxThreadsPerBlock) {
        printf("错误: 块大小超出设备限制!\n");
    }
    if (sharedMemSize > prop.sharedMemPerBlock) {
        printf("错误: 共享内存超出设备限制!\n");
    }
}

//------------------------------------------warp内部操作----------------------------------------------

// 假设 value 是每个线程的输入，返回该 warp 内的最大值
__device__ int warpMax(int value, uint32_t mask) {
    //// 全活跃线程掩码（32 线程 warp 中通常都是活跃的）
    //unsigned mask = __activemask();
    //// 每次向下偏移 offset 并与自己的值做比较
    //for (int offset = 16; offset > 0; offset >>= 1) {
    //    int other = __shfl_down_sync(mask, value, offset);
    //    value = (value > other ? value : other);
    //}
    //// 到这里，lane 0 上的 value 就是整个 warp 的最大值
    //return value;
    return __reduce_max_sync(mask, value);
}
__device__ int warpMin(int value, uint32_t mask) {
    //unsigned mask = __activemask();
    //for (int offset = 16; offset > 0; offset >>= 1) {
    //    int other = __shfl_down_sync(mask, value, offset);
    //    value = (value < other ? value : other);
    //}
    //return value;
    return __reduce_min_sync(mask, value);
}

__device__ float warpMax(float value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, value, offset);
        value = fmaxf(value, other);
    }
    return value;
}
__device__ float warpMin(float value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, value, offset);
        value = fminf(value, other);
    }
    return value;
}
__device__ int warpReduce(int localSum, uint32_t mask)
{
    //localSum += __shfl_xor_sync(0xFFFFFFFFu, localSum, 16);
    //localSum += __shfl_xor_sync(0xFFFFFFFFu, localSum, 8);
    //localSum += __shfl_xor_sync(0xFFFFFFFFu, localSum, 4);
    //localSum += __shfl_xor_sync(0xFFFFFFFFu, localSum, 2);
    //localSum += __shfl_xor_sync(0xFFFFFFFFu, localSum, 1);
    //return localSum;
    return __reduce_add_sync(mask, localSum);
}
__device__ float warpReduce(float localSum, uint32_t mask)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        localSum += __shfl_down_sync(mask, localSum, offset);
    }
    return localSum;
}

__device__ bool valueEqual(int val, uint32_t mask) {
    int ref = __shfl_sync(mask, val, __ffs(mask) - 1);
    unsigned vote = __ballot_sync(mask, val == ref); // 哪些 lane 等于 ref
    return vote == mask; // 如果所有参与 lane 都等于 ref
}
//------------------------------------------------求和--------------------------------------------
__global__ void sumFloatKernel(float* input, float* output, uint32_t dataNum) {
    extern __shared__ float sharedData[];
    int threadIndex = threadIdx.x;

    sharedData[threadIndex] = 0;
    int dataIndex = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    if (dataIndex >= dataNum) {
        return;
    }
    float inputData = input[dataIndex];
    if (dataIndex + blockDim.x < dataNum) {
        inputData += input[dataIndex + blockDim.x];
    }
    sharedData[threadIndex] = inputData;
    __syncthreads();

    if (dataNum > 1024) {
        if (threadIndex >= blockDim.x / 2) {
            return;
        }
    }
    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (threadIndex < s) {
            sharedData[threadIndex] += sharedData[threadIndex + s];
        }
        __syncthreads();
    }
    if (threadIndex < 32) { //这里只是针对4MB，最后一个块有1024个线程，如果是其他数量，需要额外判断
        float sum = warpReduce(sharedData[threadIndex]);
        if (threadIndex == 0) {
            output[blockIdx.x] = sum;
        }
    }

}
float* sumFloat(float* data, uint32_t dataNum, cudaStream_t stream) {
    uint32_t gridSize = ceil((float)dataNum / 2048);
    //gridSize = ceil((float)gridSize / 2);
    uint32_t blockSize = dataNum > 1024 ? 1024 : ceil(float(dataNum) / 2);

    float* input;
    CHECK(cudaMalloc((void**)&input, sizeof(float) * dataNum));
    CHECK(cudaMemcpy(input, data, sizeof(float) * dataNum, cudaMemcpyDeviceToDevice));
    float* output;
    CHECK(cudaMalloc((void**)&output, sizeof(float) * gridSize));
    CHECK(cudaDeviceSynchronize());

    while (dataNum > 1) {
        sumFloatKernel << <gridSize, blockSize, blockSize * sizeof(float), stream >> > (input, output, dataNum);
        CHECK(cudaStreamSynchronize(stream));

        //float* testData_host = (float*)malloc(sizeof(float) * gridSize);
        //CHECK(cudaMemcpy(testData_host, output, sizeof(float) * gridSize, cudaMemcpyDeviceToHost));
        //for (int i = 0; i < gridSize; i++) {
        //    std::cout << testData_host[i] << std::endl;
        //}
        //std::cout << std::endl;
        //free(testData_host);

        dataNum = gridSize;
        blockSize = dataNum > 1024 ? 1024 : ceil(float(dataNum) / 2);
        gridSize = ceil((float)gridSize / 2048);
        //gridSize = ceil((float)gridSize / 2);
        float* temp = input;
        input = output;
        output = temp;
    }

    float* sum;
    CHECK(cudaMalloc((void**)&sum, sizeof(float)));
    CHECK(cudaMemcpy(sum, input, sizeof(float), cudaMemcpyDeviceToDevice));

    CHECK(cudaFree(input));
    CHECK(cudaFree(output));

    return sum;
}

//------------------------------------------打包操作-----------------------------------------------
__device__ uint32_t packUint3(uint3 valueU3) {
    return ((valueU3.x & 0x3FF) << 20) | ((valueU3.y & 0x3FF) << 10) | (valueU3.z & 0x3FF);
}

__device__ uint3 unpackUint(uint32_t value) {
    return make_uint3((value >> 20) & 0x3FF, (value >> 10) & 0x3FF, value & 0x3FF);
}

__device__ __forceinline__ uint32_t packUnorm4x8(const float4 v) {
    // clamp to [0,1], then map to 0..255 and round
    float4 c;
    c.x = fminf(fmaxf(v.x, 0.0f), 1.0f);
    c.y = fminf(fmaxf(v.y, 0.0f), 1.0f);
    c.z = fminf(fmaxf(v.z, 0.0f), 1.0f);
    c.w = fminf(fmaxf(v.w, 0.0f), 1.0f);

    uint32_t r = (uint32_t)(c.x * 255.0f + 0.5f) & 0xFFu;
    uint32_t g = (uint32_t)(c.y * 255.0f + 0.5f) & 0xFFu;
    uint32_t b = (uint32_t)(c.z * 255.0f + 0.5f) & 0xFFu;
    uint32_t a = (uint32_t)(c.w * 255.0f + 0.5f) & 0xFFu;

    // pack: lowest byte = R, then G, B, highest = A
    return (a << 24) | (b << 16) | (g << 8) | r;
}
__device__ __forceinline__ float4 unpackUnorm4x8(const uint32_t v) {
    uint32_t r = v & 0xFFu;
    uint32_t g = (v >> 8) & 0xFFu;
    uint32_t b = (v >> 16) & 0xFFu;
    uint32_t a = (v >> 24) & 0xFFu;
    float4 out;
    out.x = (float)r / 255.0f;
    out.y = (float)g / 255.0f;
    out.z = (float)b / 255.0f;
    out.w = (float)a / 255.0f;
    return out;
}

__device__ int FloatToOrderedInt(float value) {
    unsigned int bits = __float_as_uint(value);
    unsigned int ordered = (bits & 0x80000000u) != 0
        ? ~bits
        : (bits ^ 0x80000000u);
    return (int)(ordered ^ 0x80000000u);
}
__device__ float OrderedIntToFloat(int value) {
    unsigned int ordered = (unsigned int)value ^ 0x80000000u;
    unsigned int bits = (ordered & 0x80000000u) != 0
        ? (ordered ^ 0x80000000u)
        : ~ordered;
    return __uint_as_float(bits);
}
//------------------------------------------原子操作-----------------------------------------------
__device__ float atomicAddFloat(float* addr, float val) {
    int* iaddr = reinterpret_cast<int*>(addr);
    int old_i = atomicCAS(iaddr, 0, 0); // __float_as_int(*addr);
    float old_f;
    int new_i;
    do {
        old_f = __int_as_float(old_i);
        float tmp = old_f + val;
        new_i = __float_as_int(tmp);
        // atomicCAS 返回的是 *addr 执行之前的旧位模式
        // 把它赋给 old_i，失败就用它做下一次的比较基准
        old_i = atomicCAS(iaddr, old_i, new_i);
    } while (old_f != __int_as_float(old_i));
    // 或者： while (old_i != new_i)；两种条件都行

    return old_f;
}

__device__ float atomicMinFloat(float* addr, float val) {
    int* iaddr = reinterpret_cast<int*>(addr);
    int old_i = atomicCAS(iaddr, 0, 0); //__float_as_int(*addr);
    float old_f;
    int new_i;
    do {
        old_f = __int_as_float(old_i);
        // 选择老值和 val 里的最小者
        float tmp = (old_f < val ? old_f : val);
        new_i = __float_as_int(tmp);
        // atomicCAS 返回的是 *addr 执行之前的旧位模式
        // 把它赋给 old_i，失败就用它做下一次的比较基准
        old_i = atomicCAS(iaddr, old_i, new_i);
    } while (old_f != __int_as_float(old_i));
    // 或者： while (old_i != new_i)；两种条件都行

    return old_f;
}

__device__ float atomicMaxFloat(float* addr, float val) {
    int* iaddr = reinterpret_cast<int*>(addr);
    int old_i = atomicCAS(iaddr, 0, 0); //__float_as_int(*addr);
    float old_f;
    int new_i;
    do {
        old_f = __int_as_float(old_i);
        // 选择老值和 val 里的最大者
        float tmp = (old_f > val ? old_f : val);
        new_i = __float_as_int(tmp);
        // atomicCAS 返回的是 *addr 执行之前的旧位模式
        // 把它赋给 old_i，失败就用它做下一次的比较基准
        old_i = atomicCAS(iaddr, old_i, new_i);
    } while (old_f != __int_as_float(old_i));
    // 或者： while (old_i != new_i)；两种条件都行

    return old_f;
}

__device__ void atomicMeanFloat4(uint32_t* addr, float4 val) {
    uint32_t newVal = packUnorm4x8(val);
    uint32_t curStoredVal = *addr;
    uint32_t prevStoredVal = curStoredVal;
    while (true) {
        uint32_t old = atomicCAS(addr, prevStoredVal, newVal);
        if (old == prevStoredVal) break;
        prevStoredVal = old;
        float4 rval = unpackUnorm4x8(prevStoredVal);

        rval.x = rval.x * rval.w;
        rval.y = rval.y * rval.w;
        rval.z = rval.z * rval.w;

        float4 curValF;
        curValF.x = rval.x + val.x;
        curValF.y = rval.y + val.y;
        curValF.z = rval.z + val.z;
        curValF.w = rval.w + val.w;

        if (curValF.w > 0.0f) {
            curValF.x /= curValF.w;
            curValF.y /= curValF.w;
            curValF.z /= curValF.w;
        }
        else {
            curValF.x = 0.0f;
            curValF.y = 0.0f;
            curValF.z = 0.0f;
        }
        newVal = packUnorm4x8(curValF);
    }
}


//----------------------------------------随机数------------------------------------------------
__device__ uint32_t pcg(uint32_t& state)
{
    //uint32_t prev = state * 747796405u + 2891336453u;
    //uint32_t word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
    //state = prev;
    state = state * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ uint2 pcg2d(uint2 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    //v = v ^ (v >> 16u);
    uint2 t;
    t.x = v.x >> 16u;
    t.y = v.y >> 16u;
    v.x = v.x ^ t.x;
    v.y = v.y ^ t.y;

    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    //v = v ^ (v >> 16u);
    t.x = v.x >> 16u;
    t.y = v.y >> 16u;
    v.x = v.x ^ t.x;
    v.y = v.y ^ t.y;

    return v;
}

__device__ float rand(uint32_t& seed)
{
    uint32_t val = pcg(seed);
    return (float(val) * (1.0 / float(0xffffffffu)));
}


//低差异序列
__device__ float RadicalInverse_VdC(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}
__device__ glm::vec2 Hammersley(uint i, uint N)
{
    return glm::vec2(float(i) / float(N), RadicalInverse_VdC(i));
}

__global__ void init_curand_states(curandState* states, unsigned long seed, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}
//__device__ float getCudaRandomNumber() {
//    uint32_t threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
//    return curand_uniform(&systemRandomNumberStates[threadIndex]);    //0-1
//}

//---------------------------------------------------------------------------------------

__global__ void addDate_float_device(float* data, float date, uint32_t dataNum) {
    uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadIndex >= dataNum) return;
    data[threadIndex] += date;
}
void addDateCUDA_float(float* data, float date, uint32_t dataNum) {
    float* data_device;
    CHECK(cudaMalloc((void**)&data_device, sizeof(float) * dataNum));
    CHECK(cudaMemcpy(data_device, data, sizeof(float) * dataNum, cudaMemcpyHostToDevice));
    uint32_t gridSize = (dataNum + 511) / 512;
    uint32_t blockSize = dataNum > 512 ? 512 : dataNum;
    addDate_float_device << < gridSize, blockSize >> > (data_device, date, dataNum);
    CHECK(cudaMemcpy(data, data_device, sizeof(float) * dataNum, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(data_device));
}

__global__ void addDate_uint_device(uint32_t* data, uint32_t date, uint32_t dataNum) {
    uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadIndex >= dataNum) return;
    data[threadIndex] += date;
}
void addDateCUDA_uint(uint32_t* data, uint32_t date, uint32_t dataNum) {
    uint32_t* data_device;
    CHECK(cudaMalloc((void**)&data_device, sizeof(uint32_t) * dataNum));
    CHECK(cudaMemcpy(data_device, data, sizeof(uint32_t) * dataNum, cudaMemcpyHostToDevice));
    uint32_t gridSize = (dataNum + 511) / 512;
    uint32_t blockSize = dataNum > 512 ? 512 : dataNum;
    addDate_uint_device << < gridSize, blockSize >> > (data_device, date, dataNum);
    CHECK(cudaMemcpy(data, data_device, sizeof(uint32_t) * dataNum, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(data_device));
}

#endif