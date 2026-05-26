#pragma once

#include "vulkanCudaInterop.cuh"

#ifndef VULKAN_CUDA_INTEROP_CU
#define VULKAN_CUDA_INTEROP_CU

//------------------------------------------------------------Vulkan交互基础函数-----------------------------------------------------------------

/*
When importing memory and synchronization objects exported by Vulkan, they must be imported and mapped on the same device as they were created on. 
The CUDA device that corresponds to the Vulkan physical device on which the objects were created can be determined by comparing the UUID of a CUDA device with that of the Vulkan physical device, 
as shown in the following code sample. Note that the Vulkan physical device should not be part of a device group that contains more than one Vulkan physical device. 
The device group as returned by vkEnumeratePhysicalDeviceGroups that contains the given Vulkan physical device must have a physical device count of 1.
*/
int getCudaDeviceForVulkanPhysicalDevice(VkPhysicalDevice vkPhysicalDevice) {

    VkPhysicalDeviceIDProperties vkPhysicalDeviceIDProperties = {};
    vkPhysicalDeviceIDProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    vkPhysicalDeviceIDProperties.pNext = NULL;

    VkPhysicalDeviceProperties2 vkPhysicalDeviceProperties2 = {};
    vkPhysicalDeviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    vkPhysicalDeviceProperties2.pNext = &vkPhysicalDeviceIDProperties;

    vkGetPhysicalDeviceProperties2(vkPhysicalDevice, &vkPhysicalDeviceProperties2);

    int cudaDeviceCount;
    cudaGetDeviceCount(&cudaDeviceCount);

    for (int cudaDevice = 0; cudaDevice < cudaDeviceCount; cudaDevice++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, cudaDevice);
        if (!memcmp(&deviceProp.uuid, vkPhysicalDeviceIDProperties.deviceUUID, VK_UUID_SIZE)) {
            return cudaDevice;
        }
    }

    return cudaInvalidDeviceId;

}

/*
On Linux and Windows 10, both dedicated and non-dedicated memory objects exported by Vulkan can be imported into CUDA. 
On Windows 7, only dedicated memory objects can be imported. 
When importing a Vulkan dedicated memory object, the flag cudaExternalMemoryDedicated must be set.
*/

/*
A Vulkan memory object exported using VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT can be imported into CUDA using the file descriptor associated with that object as shown below.
Note that CUDA assumes ownership of the file descriptor once it is imported. 
Using the file descriptor after a successful import results in undefined behavior.
*/
cudaExternalMemory_t importVulkanMemoryObjectFromFileDescriptor(int fd, unsigned long long size, bool isDedicated) {

    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = fd;
    desc.size = size;
    if (isDedicated) {
        desc.flags |= cudaExternalMemoryDedicated;
    }

    // Input parameter 'fd' should not be used beyond this point as CUDA has assumed ownership of it
    cudaImportExternalMemory(&extMem, &desc);

    return extMem;

}

/*
A Vulkan memory object exported using VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT can be imported into CUDA using the NT handle associated with that object as shown below.
Note that CUDA does not assume ownership of the NT handle and it is the application’s responsibility to close the handle when it is not required anymore. 
The NT handle holds a reference to the resource, so it must be explicitly freed before the underlying memory can be freed.
*/
cudaExternalMemory_t importVulkanMemoryObjectFromNTHandle(HANDLE handle, unsigned long long size, bool isDedicated) {

    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    desc.handle.win32.handle = handle;
    desc.size = size;
    if (isDedicated) {
        desc.flags |= cudaExternalMemoryDedicated;
    }

    CHECK(cudaImportExternalMemory(&extMem, &desc));

    // Input parameter 'handle' should be closed if it's not needed anymore
    //CloseHandle(handle);

    return extMem;

}

/*
A Vulkan memory object exported using VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT can also be imported using a named handle if one exists as shown below.
*/
cudaExternalMemory_t importVulkanMemoryObjectFromNamedNTHandle(LPCWSTR name, unsigned long long size, bool isDedicated) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    desc.handle.win32.name = (void*)name;
    desc.size = size;
    if (isDedicated) {
        desc.flags |= cudaExternalMemoryDedicated;
    }

    cudaImportExternalMemory(&extMem, &desc);

    return extMem;
}

/*
A Vulkan memory object exported using VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT can be imported into CUDA using the globally shared D3DKMT handle 
associated with that object as shown below. 
Since a globally shared D3DKMT handle does not hold a reference to the underlying memory it is automatically destroyed when all other references to the resource are destroyed.
*/
cudaExternalMemory_t importVulkanMemoryObjectFromKMTHandle(HANDLE handle, unsigned long long size, bool isDedicated) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
    desc.handle.win32.handle = (void*)handle;
    desc.size = size;
    if (isDedicated) {
        desc.flags |= cudaExternalMemoryDedicated;
    }

    cudaImportExternalMemory(&extMem, &desc);

    return extMem;
}

//----------------------------------------------------------------缓冲区——-----------------------------------------------------
/*
A device pointer can be mapped onto an imported memory object as shown below. 
The offset and size of the mapping must match that specified when creating the mapping using the corresponding Vulkan API.
All mapped device pointers must be freed using cudaFree().
*/
/*
貌似只能从vulkan中传递数据到cuda，而不能从cuda中传递数据到vulkan
我们只能在vulkan中创建一个buffer，然后在cuda中将输出copy进去
*/
void* mapBufferOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, unsigned long long size) {

    void* ptr = NULL;

    cudaExternalMemoryBufferDesc desc = {};
    memset(&desc, 0, sizeof(desc));
    desc.offset = offset;
    desc.size = size;

    cudaExternalMemoryGetMappedBuffer(&ptr, extMem, &desc);

    // Note: ‘ptr’ must eventually be freed using cudaFree()
    return ptr;

}

//----------------------------------------------------------------纹理------------------------------------------------------------
/*
A CUDA mipmapped array can be mapped onto an imported memory object as shown below. 
The offset, dimensions, format and number of mip levels must match that specified when creating the mapping using the corresponding Vulkan API. 
Additionally, if the mipmapped array is bound as a color target in Vulkan, the flagcudaArrayColorAttachment must be set. 
All mapped mipmapped arrays must be freed using cudaFreeMipmappedArray(). 
The following code sample shows how to convert Vulkan parameters into the corresponding CUDA parameters when mapping mipmapped arrays onto imported memory objects.
*/
//貌似没有不是mipmap的，我看nvida的runtime API文档中
cudaMipmappedArray_t mapMipmappedArrayOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, cudaChannelFormatDesc* formatDesc, cudaExtent* extent, unsigned int flags, unsigned int numLevels) {

    cudaMipmappedArray_t mipmap = NULL;
    cudaExternalMemoryMipmappedArrayDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.offset = offset;
    desc.formatDesc = *formatDesc;
    desc.extent = *extent;
    desc.flags = flags;
    desc.numLevels = numLevels;

    // 检查外部内存是否有效
    if (extMem == NULL) {
        printf("ERROR: extMem is NULL\n");
        return NULL;
    }

    // Note: 'mipmap' must eventually be freed using cudaFreeMipmappedArray()
    CHECK(cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &desc));

    return mipmap;

}

cudaChannelFormatDesc getCudaChannelFormatDescForVulkanFormat(VkFormat format)
{
    cudaChannelFormatDesc d;

    memset(&d, 0, sizeof(d));

    switch (format) {
        case VK_FORMAT_R8_UINT:             d.x = 8;  d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
        case VK_FORMAT_R8_SINT:             d.x = 8;  d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
        case VK_FORMAT_R8G8_UINT:           d.x = 8;  d.y = 8;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
        case VK_FORMAT_R8G8_SINT:           d.x = 8;  d.y = 8;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
        case VK_FORMAT_R8G8B8A8_UINT:       d.x = 8;  d.y = 8;  d.z = 8;  d.w = 8;  d.f = cudaChannelFormatKindUnsigned; break;
        case VK_FORMAT_R8G8B8A8_SINT:       d.x = 8;  d.y = 8;  d.z = 8;  d.w = 8;  d.f = cudaChannelFormatKindSigned;   break;
        case VK_FORMAT_R16_UINT:            d.x = 16; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
        case VK_FORMAT_R16_SINT:            d.x = 16; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
        case VK_FORMAT_R16G16_UINT:         d.x = 16; d.y = 16; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
        case VK_FORMAT_R16G16_SINT:         d.x = 16; d.y = 16; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
        case VK_FORMAT_R16G16B16A16_UINT:   d.x = 16; d.y = 16; d.z = 16; d.w = 16; d.f = cudaChannelFormatKindUnsigned; break;
        case VK_FORMAT_R16G16B16A16_SINT:   d.x = 16; d.y = 16; d.z = 16; d.w = 16; d.f = cudaChannelFormatKindSigned;   break;
        case VK_FORMAT_R32_UINT:            d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
        case VK_FORMAT_R32_SINT:            d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
        case VK_FORMAT_R32_SFLOAT:          d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindFloat;    break;
        case VK_FORMAT_R32G32_UINT:         d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
        case VK_FORMAT_R32G32_SINT:         d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
        case VK_FORMAT_R32G32_SFLOAT:       d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindFloat;    break;
        case VK_FORMAT_R32G32B32A32_UINT:   d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindUnsigned; break;
        case VK_FORMAT_R32G32B32A32_SINT:   d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindSigned;   break;
        case VK_FORMAT_R32G32B32A32_SFLOAT: d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindFloat;    break;
        case VK_FORMAT_R8G8B8A8_SRGB:       d.x = 8;  d.y = 8;  d.z = 8;  d.w = 8;  d.f = cudaChannelFormatKindUnsigned; break;
        case VK_FORMAT_R8G8B8A8_UNORM:      d.x = 8;  d.y = 8;  d.z = 8;  d.w = 8;  d.f = cudaChannelFormatKindUnsigned; break;
        default: assert(0);
    }

    return d;
}

cudaExtent getCudaExtentForVulkanExtent(VkExtent3D vkExt, uint32_t arrayLayers, VkImageViewType vkImageViewType) {

    cudaExtent e = { 0, 0, 0 };
    //srgb貌似必须是256的倍数
    // vkExt.width -= vkExt.width & 255;
    // vkExt.height -= vkExt.height & 255;

    switch (vkImageViewType) {
    case VK_IMAGE_VIEW_TYPE_1D:         e.width = vkExt.width; e.height = 0;            e.depth = 0;           break;
    case VK_IMAGE_VIEW_TYPE_2D:         e.width = vkExt.width; e.height = vkExt.height; e.depth = 0;           break;
    case VK_IMAGE_VIEW_TYPE_3D:         e.width = vkExt.width; e.height = vkExt.height; e.depth = vkExt.depth; break;
    case VK_IMAGE_VIEW_TYPE_CUBE:       e.width = vkExt.width; e.height = vkExt.height; e.depth = arrayLayers; break;
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY:   e.width = vkExt.width; e.height = 0;            e.depth = arrayLayers; break;
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY:   e.width = vkExt.width; e.height = vkExt.height; e.depth = arrayLayers; break;
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: e.width = vkExt.width; e.height = vkExt.height; e.depth = arrayLayers; break;
    default: assert(0);
    }

    return e;
}

unsigned int getCudaMipmappedArrayFlagsForVulkanImage(VkImageViewType vkImageViewType, VkImageUsageFlags vkImageUsageFlags, bool allowSurfaceLoadStore) {

    unsigned int flags = 0;

    switch (vkImageViewType) {
        case VK_IMAGE_VIEW_TYPE_CUBE:       flags |= cudaArrayCubemap;                    break;
        case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: flags |= cudaArrayCubemap | cudaArrayLayered; break;
        case VK_IMAGE_VIEW_TYPE_1D_ARRAY:   flags |= cudaArrayLayered;                    break;
        case VK_IMAGE_VIEW_TYPE_2D_ARRAY:   flags |= cudaArrayLayered;                    break;
        default: break;
    }

    if (vkImageUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
        flags |= cudaArrayColorAttachment;
    }

    if (allowSurfaceLoadStore) {
        flags |= cudaArraySurfaceLoadStore;
    }

    return flags;

}

void fromVulkanImageToCudaTexture(VkPhysicalDevice vkPhysicalDevice, FzbRenderer::Image& vkImage, HANDLE handle, unsigned long long size,
    bool isDedicated, cudaExternalMemory_t& extMem, cudaMipmappedArray_t& mipmap, cudaTextureObject_t& texObj, bool sampleNormal) {

    //先判断是否是同一个物理设备
    if (getCudaDeviceForVulkanPhysicalDevice(vkPhysicalDevice) == cudaInvalidDeviceId) {
        throw std::runtime_error("CUDA与Vulkan用的不是同一个GPU！！！");
    }

    //获得Vulkan导出的内存对象
    extMem = importVulkanMemoryObjectFromNTHandle(handle, size, isDedicated);

    //将纹理映射到外部内存对象
	VkImageCreateInfo imageInfo = vkImage.setting.info;
	VkImageViewCreateInfo imageViewInfo = vkImage.setting.viewInfo;
    cudaChannelFormatDesc format = getCudaChannelFormatDescForVulkanFormat(imageInfo.format);
    cudaExtent extent = getCudaExtentForVulkanExtent({ imageInfo.extent.width, imageInfo.extent.height, imageInfo.extent.depth }, imageInfo.arrayLayers, imageViewInfo.viewType);
    unsigned int flags = getCudaMipmappedArrayFlagsForVulkanImage(imageViewInfo.viewType, imageInfo.usage, false);   //cudaArraySurfaceLoadStore表示是否可写
    mipmap = mapMipmappedArrayOntoExternalMemory(extMem, 0, &format, &extent, flags, imageInfo.mipLevels);    //cudaMipmappedArray_t是只读的

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeMipmappedArray;
    resDesc.res.mipmap.mipmap = mipmap;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = sampleNormal;
    //texDesc.sRGB = true;    //转回线性空间

    CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
}

void fromVulkanImageToCudaSurface(VkPhysicalDevice vkPhysicalDevice, FzbRenderer::Image& vkImage, HANDLE handle, unsigned long long size,
    bool isDedicated, cudaExternalMemory_t& extMem, cudaMipmappedArray_t& mipmap, cudaSurfaceObject_t& surfObj) {

    //先判断是否是同一个物理设备
    if (getCudaDeviceForVulkanPhysicalDevice(vkPhysicalDevice) == cudaInvalidDeviceId) {
        throw std::runtime_error("CUDA与Vulkan用的不是同一个GPU！！！");
    }

    //获得Vulkan导出的内存对象
    extMem = importVulkanMemoryObjectFromNTHandle(handle, size, isDedicated);

    //将纹理映射到外部内存对象
    VkImageCreateInfo imageInfo = vkImage.setting.info;
    VkImageViewCreateInfo imageViewInfo = vkImage.setting.viewInfo;
    cudaChannelFormatDesc format = getCudaChannelFormatDescForVulkanFormat(imageInfo.format);
    cudaExtent extent = getCudaExtentForVulkanExtent({ imageInfo.extent.width, imageInfo.extent.height, imageInfo.extent.depth }, imageInfo.arrayLayers, imageViewInfo.viewType);
    unsigned int flags = getCudaMipmappedArrayFlagsForVulkanImage(imageViewInfo.viewType, imageInfo.usage, true);   //cudaArraySurfaceLoadStore表示是否可写
    mipmap = mapMipmappedArrayOntoExternalMemory(extMem, 0, &format, &extent, flags, imageInfo.mipLevels);

    cudaArray_t cuArray;    //cudaArray_t不能直接在核函数中读写，包括原子运算
    CHECK(cudaGetMipmappedArrayLevel(&cuArray, mipmap, 0)); // 选择 Mipmap 层级

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    CHECK(cudaCreateSurfaceObject(&surfObj, &resDesc));

}

//-------------------------------------------------------------信号量---------------------------------------------------------
/*
A Vulkan semaphore object exported using VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BITcan be imported into CUDA using the file descriptor associated with that object as shown below.
Note that CUDA assumes ownership of the file descriptor once it is imported.
Using the file descriptor after a successful import results in undefined behavior.
*/
cudaExternalSemaphore_t importVulkanSemaphoreObjectFromFileDescriptor(int fd) {

    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
    desc.handle.fd = fd;

    cudaImportExternalSemaphore(&extSem, &desc);

    // Input parameter 'fd' should not be used beyond this point as CUDA has assumed ownership of it
    return extSem;

}

/*
A Vulkan semaphore object exported using VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT can be imported into CUDA using the NT handle associated with that object as shown below.
Note that CUDA does not assume ownership of the NT handle and it is the application’s responsibility to close the handle when it is not required anymore.
The NT handle holds a reference to the resource, so it must be explicitly freed before the underlying semaphore can be freed.
*/
cudaExternalSemaphore_t importVulkanSemaphoreObjectFromNTHandle(HANDLE handle, bool timeline) {

    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = timeline ? cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32: cudaExternalSemaphoreHandleTypeOpaqueWin32;
    desc.handle.win32.handle = handle;

    CHECK(cudaImportExternalSemaphore(&extSem, &desc));

    // Input parameter 'handle' should be closed if it's not needed anymore
    //CloseHandle(handle);

    return extSem;
}

/*
A Vulkan semaphore object exported using VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT can also be imported using a named handle if one exists as shown below.
*/
cudaExternalSemaphore_t importVulkanSemaphoreObjectFromNamedNTHandle(LPCWSTR name) {

    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
    desc.handle.win32.name = (void*)name;

    cudaImportExternalSemaphore(&extSem, &desc);

    return extSem;
}

/*
A Vulkan semaphore object exported using VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT can be imported into CUDA using the globally shared D3DKMT handle
associated with that object as shown below.
Since a globally shared D3DKMT handle does not hold a reference to the underlying semaphore it is automatically destroyed when all other references to the resource are destroyed.
*/
cudaExternalSemaphore_t importVulkanSemaphoreObjectFromKMTHandle(HANDLE handle) {

    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
    desc.handle.win32.handle = (void*)handle;

    cudaImportExternalSemaphore(&extSem, &desc);

    return extSem;

}

/*
An imported Vulkan semaphore object can be signaled as shown below.
Signaling such a semaphore object sets it to the signaled state.
The corresponding wait that waits on this signal must be issued in Vulkan.
Additionally, the wait that waits on this signal must be issued after this signal has been issued.
*/
cudaError_t signalExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream, uint64_t signalValue) {

    cudaExternalSemaphoreSignalParams params = {};

    memset(&params, 0, sizeof(params));
    params.params.fence.value = signalValue;

    return cudaSignalExternalSemaphoresAsync(&extSem, &params, 1, stream);

}

/*
An imported Vulkan semaphore object can be waited on as shown below.
Waiting on such a semaphore object waits until it reaches the signaled state and then resets it back to the unsignaled state.
The corresponding signal that this wait is waiting on must be issued in Vulkan.
Additionally, the signal must be issued before this wait can be issued.
*/
cudaError_t waitExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream, uint64_t waitValue) {

    cudaExternalSemaphoreWaitParams params = {};

    memset(&params, 0, sizeof(params));
    params.params.fence.value = waitValue;

    return cudaWaitExternalSemaphoresAsync(&extSem, &params, 1, stream);

}

#endif
