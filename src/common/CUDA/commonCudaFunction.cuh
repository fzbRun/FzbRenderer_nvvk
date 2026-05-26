#pragma once

#include <chrono>

#include <texture_indirect_functions.h>

#include <cuda_runtime.h> 
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <helper_math.h>
#include <curand_kernel.h>
#include "../Shader/shaderStructType.h"

#ifndef COMMON_CUDA_FUNCTION_CUH
#define COMMON_CUDA_FUNCTION_CUH

//__constant__ bool useCudaRandom;
//__constant__ curandState* systemRandomNumberStates;
//__constant__ uint32_t systemRandomNumberSeed;
//----------------------------------------------------------------------------------------------

#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

/*
clock()会计算CPU时间，而非挂钟时间。即若有3个线程并行运行，执行3秒，那么clock会返回3x3=9秒，并且挂起时间不算在内，因此无法得到核函数运行时间。
*/
double cpuSecond();

void checkKernelFunction();
void checkLaunchConfiguration(uint32_t gridSize, uint32_t blockSize, uint32_t sharedMemSize);

/*
这里是关于warp内部操作（原语）的学习
首先，明确几点重要的点
1. __activemask()是指执行到此刻的线程的掩码。但是如果在__activemask()前有warp级别的分支，则可能不是所有线程同时到达__activemask()，
    导致__activemask()给出的结果不一定代表所有激活线程，这成为偶然分支，所以我们可以在__activemask()前先同步warp再使用。
    一般来说需要在分支前使用__ballot_sync()来得到掩码。
2. 大部分原语需要掩码显示执行线程，但是volta架构后可以不需要管，直接用0xffffffff即可，原语内部会处理，对于不激活的线程，原语结果会是未定义（但资料说就是0）
    这样，在分支中，如果想对激活线程进行累加，可以直接用0xffffffff。但是对于大小值判断则不行，因为不激活线程会返回0
3. 同样在volta架构之后，__syncwarp()可以处理条件分支，只不过在原语之后线程会再次发散。
4. 由于编译环境和硬件的不同，隐式同步是不安全的，比如在if-else后分支同步，但可能由于环境不同，同步没有发生，即使在前后使用__syncwarp()，如
     __syncwarp(); v = __shfl(0); __syncwarp() != __shfl(0)
    因此我们需要使用显式同步，即 使用带sync的原语，如__shfl_sync。
5. warp内部操作可以指定mask，但调用warp操作函数的线程必须在mask中，且warp操作的对象也必须在mask中，如线程0去拿线程16的数据，必须保证0和16都在mask中
*/
__device__ int warpReduce(int localSum, uint32_t mask = 0xffffffff);
__device__ float warpReduce(float localSum, uint32_t mask = 0xffffffff);
__device__ int warpMax(int value, uint32_t mask = 0xffffffff);
__device__ int warpMin(int value, uint32_t mask = 0xffffffff);

__device__ float warpMax(float value);
__device__ float warpMin(float value);

__device__ bool valueEqual(int val, uint32_t mask = 0xffffffff);

float* sumFloat(float* input, float* output);

__device__ uint32_t packUint3(uint3 valueU3);
__device__ uint3 unpackUint(uint32_t value);
__device__ uint32_t packUnorm4x8(const float4 v);
__device__ float4 unpackUnorm4x8(const uint32_t v);

__device__ float atomicAddFloat(float* addr, float val);
__device__ float atomicMinFloat(float* addr, float val);
__device__ float atomicMaxFloat(float* addr, float val);
__device__ void atomicMeanFloat4(uint32_t* addr, float4 val);

__device__ uint32_t pcg(uint32_t& state);
__device__ uint2 pcg2d(uint2 v);
__device__ float rand(uint32_t& seed);
__device__ float RadicalInverse_VdC(uint bits);
__device__ glm::vec2 Hammersley(uint i, uint N);
__global__ void init_curand_states(curandState* states, unsigned long seed, int n);
__device__ float getCudaRandomNumber();
//__device__ float getRandomNumberFromSeed(uint32_t& randomNumberSeed);

__global__ void addDate_float_device(float* data, float date, uint32_t dataNum);
void addDateCUDA_float(float* data, float date, uint32_t dataNum);

__global__ void addDate_uint_device(uint32_t* data, uint32_t date, uint32_t dataNum);
void addDateCUDA_uint(uint32_t* data, uint32_t date, uint32_t dataNum);


#endif