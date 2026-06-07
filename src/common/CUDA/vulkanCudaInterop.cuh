#pragma once

#include<iostream>
#include <chrono>
#include "./commonCudaFunction.cuh"
#include "../Image/Image.h"
#include "../Buffer/Buffer.h"

//#ifndef __CUDACC__
//#define __CUDACC__
//#endif
//#ifndef __cplusplus
//#define __cplusplus
//#endif

#include <texture_indirect_functions.h>

#include <cuda_runtime.h> 
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <volk.h>

#ifndef VULKAN_CUDA_INTEROP_CUH
#define VULKAN_CUDA_INTEROP_CUH

int getCudaDeviceForVulkanPhysicalDevice(VkPhysicalDevice vkPhysicalDevice);
cudaExternalMemory_t importVulkanMemoryObjectFromFileDescriptor(int fd, unsigned long long size, bool isDedicated);
cudaExternalMemory_t importVulkanMemoryObjectFromNTHandle(HANDLE handle, unsigned long long size, bool isDedicated);
cudaExternalMemory_t importVulkanMemoryObjectFromNamedNTHandle(LPCWSTR name, unsigned long long size, bool isDedicated);
cudaExternalMemory_t importVulkanMemoryObjectFromKMTHandle(HANDLE handle, unsigned long long size, bool isDedicated);
void* mapBufferOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, unsigned long long size);
cudaMipmappedArray_t mapMipmappedArrayOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, cudaChannelFormatDesc* formatDesc, cudaExtent* extent, unsigned int flags, unsigned int numLevels);
cudaChannelFormatDesc getCudaChannelFormatDescForVulkanFormat(VkFormat format);
cudaExtent getCudaExtentForVulkanExtent(VkExtent3D vkExt, uint32_t arrayLayers, VkImageViewType vkImageViewType);
unsigned int getCudaMipmappedArrayFlagsForVulkanImage(VkImageViewType vkImageViewType, VkImageUsageFlags vkImageUsageFlags, bool allowSurfaceLoadStore);
cudaExternalSemaphore_t importVulkanSemaphoreObjectFromFileDescriptor(int fd);
cudaExternalSemaphore_t importVulkanSemaphoreObjectFromNTHandle(HANDLE handle, bool timeline = true);
cudaExternalSemaphore_t importVulkanSemaphoreObjectFromNamedNTHandle(LPCWSTR name);
cudaExternalSemaphore_t importVulkanSemaphoreObjectFromKMTHandle(HANDLE handle);
cudaError_t signalExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream, uint64_t signalValue = 1);
cudaError_t waitExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream, uint64_t waitValue = 1);
void fromVulkanImageToCudaTexture(VkPhysicalDevice vkPhysicalDevice, FzbRenderer::Image& vkImage,
    bool isDedicated, cudaExternalMemory_t& extMem, cudaMipmappedArray_t& mipmap, cudaTextureObject_t& texObj, bool sampleNormal = false);
void fromVulkanImageToCudaSurface(VkPhysicalDevice vkPhysicalDevice, FzbRenderer::Image& vkImage,
    bool isDedicated, cudaExternalMemory_t& extMem, cudaMipmappedArray_t& mipmap, cudaSurfaceObject_t& surfObj);
#endif