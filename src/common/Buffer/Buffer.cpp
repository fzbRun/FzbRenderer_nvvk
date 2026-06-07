#include "./Buffer.h"
#include <common/Application/Application.h>
#include <nvvk/formats.hpp>
#include <nvvk/default_structs.hpp>
#include <common/utils.hpp>

using namespace FzbRenderer;

FzbRenderer::Buffer::Buffer(std::string name, bool external) {
	this->name = name;
	this->external = external;
}

VkResult FzbRenderer::Buffer::init(VkBufferCreateInfo createInfo) {
	if (external) {
		Application::allocatorExport.createBufferExport(buffer, createInfo.size, createInfo.usage);

		VmaAllocationInfo2 allocInfo;
		vmaGetAllocationInfo2(Application::allocator, buffer.allocation, &allocInfo);
		allocMemTotalSize = allocInfo.blockSize;
		allocMemSize = allocInfo.allocationInfo.size;
		allocMemOffset = allocInfo.allocationInfo.offset;

		VkMemoryGetWin32HandleInfoKHR handleInfo{};
		handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
		handleInfo.memory = allocInfo.allocationInfo.deviceMemory;
		handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
		GetMemoryWin32HandleKHR(&handleInfo, &handle);
	}
	else {
		Application::allocator.createBuffer(buffer, createInfo.size, createInfo.usage);

		VmaAllocationInfo2 allocInfo;
		vmaGetAllocationInfo2(Application::allocator, buffer.allocation, &allocInfo);
		allocMemTotalSize = allocInfo.blockSize;
		allocMemSize = allocInfo.allocationInfo.size;
		allocMemOffset = allocInfo.allocationInfo.offset;
	}

	return VK_SUCCESS;
}
void FzbRenderer::Buffer::clean() {
	if (buffer.buffer != VK_NULL_HANDLE) {
		Application::allocator.destroyBuffer(buffer);
		buffer = {};
	}

	if (handle != nullptr && allocMemOffset == 0) {		//sometime many buffer use same handle(distinguish by offet)
		CloseHandle(handle);
		handle = nullptr;
	}
}
