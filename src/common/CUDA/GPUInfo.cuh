#include <iostream>
#include <cuda_runtime.h>

// 辅助函数：将 cudaError_t 转换为可读字符串并检查错误
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line);
void getGPUInfo();