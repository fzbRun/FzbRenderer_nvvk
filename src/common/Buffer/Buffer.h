#pragma once
#include <vulkan/vulkan_core.h>
#include <nvvk/resources.hpp>
#include <string>
#include <filesystem>
#include <volk.h>

#ifndef FZBRENDERER_BUFFER_FZBPATHGUIDING_H
#define FZBRENDERER_BUFFER_FZBPATHGUIDING_H
namespace FzbRenderer {
class Buffer {
public:
	Buffer(std::string name, bool external = false);
	Buffer() = default;
	~Buffer() = default;

	VkResult init(VkBufferCreateInfo createInfo);
	//template<typename T>
	//VkResult fill(std::vector<T>& data, uint32_t offset = 0) {
	//	uint32_t dataSize = data.size() * sizeof(T);
	//	if (dataSize + offset >= buffer.bufferSize) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
	//
	//	if (external) NVVK_CHECK(Application::stagingUploaderExport.appendBuffer(buffer, offset, dataSize, data.data(), {}));
	//	else NVVK_CHECK(Application::stagingUploader.appendBuffer(buffer, offset, dataSize, data.data(), {}));
	//	Application::uploadResource();
	//
	//	return VK_SUCCESS;
	//}
	void clean();

	void save(std::string path);
	void load(std::string path);

	std::string name = "buffer";
	nvvk::Buffer buffer;
	
	uint32_t allocMemSize;
	uint32_t allocMemTotalSize;
	uint32_t allocMemOffset;

	bool external = false;
	HANDLE handle = nullptr;
};
}

#endif