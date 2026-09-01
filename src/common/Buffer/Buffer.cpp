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
		if(external) Application::allocatorExport.destroyBuffer(buffer);
		else Application::allocator.destroyBuffer(buffer);
		buffer = {};
	}

	/*
	这个handle的释放其实挺麻烦的，因为当clean然后重新init后，可能还是一块，但是offset变了，导致可能没有任何buffer的offset是0，导致无法closeHandle
	不过这也不是什么大事情，漏就漏吧，屁点大的东西
	*/
	if (handle != nullptr && allocMemOffset == 0) {		//sometime many buffer use same handle(distinguish by offet)
		CloseHandle(handle);
		handle = nullptr;
	}
}

void FzbRenderer::Buffer::save(std::string path) {
	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();

	nvvk::BufferRange stagingSpace;
	stagingUploader.acquireStagingSpace(stagingSpace, allocMemSize, nullptr);

	VkBufferCopy2 copyRegionInfo{
		.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
		.srcOffset = 0,
		.dstOffset = stagingSpace.offset,
		.size = allocMemSize,
	};

	VkCopyBufferInfo2 copyBufferInfo{
		.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
		.srcBuffer = buffer.buffer,
		.dstBuffer = stagingSpace.buffer,
		.regionCount = 1,
		.pRegions = &copyRegionInfo,
	};

	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();
	vkCmdCopyBuffer2(cmd, &copyBufferInfo);
	Application::app->submitAndWaitTempCmdBuffer(cmd);

	std::error_code ec;
	std::filesystem::path filePath(path);
	std::filesystem::create_directories(filePath.parent_path(), ec);
	if (ec) throw std::system_error(ec, "创建目录失败: " + filePath.parent_path().string());

	FILE* fp = fopen(path.c_str(), "wb");
	if (!fp) {
		int err = errno;
		std::string msg = "打开文件失败: " + path + " - " + std::strerror(err);
		throw std::runtime_error(msg);
	}

	uint64_t size = static_cast<uint64_t>(allocMemSize);
	fwrite(&size, sizeof(size), 1, fp);
	fwrite(stagingSpace.mapping, 1, allocMemSize, fp);
	fclose(fp);
}
void FzbRenderer::Buffer::load(std::string path) {
	FILE* fp = fopen(path.c_str(), "rb");
	if (!fp) {
		int err = errno;
		std::string msg = "打开文件失败: " + path + " - " + std::strerror(err);
		throw std::runtime_error(msg);
	}

	uint64_t size;
	fread(&size, sizeof(size), 1, fp);
	std::vector<uint8_t> fileData(size);
	fread(fileData.data(), 1, size, fp);
	fclose(fp);

	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();

	nvvk::BufferRange stagingSpace;
	stagingUploader.acquireStagingSpace(stagingSpace, size, nullptr);

	memcpy(stagingSpace.mapping, fileData.data(), size);

	
	if (buffer.buffer != VK_NULL_HANDLE) {
		Application::allocator.destroyBuffer(buffer);
		buffer = {};
	}

	init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});

	VkBufferCopy2 copyRegionInfo{
		.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
		.srcOffset = stagingSpace.offset,
		.dstOffset = 0,
		.size = size,
	};

	VkCopyBufferInfo2 copyBufferInfo{
		.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
		.srcBuffer = stagingSpace.buffer,
		.dstBuffer = buffer.buffer,
		.regionCount = 1,
		.pRegions = &copyRegionInfo,
	};

	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();
	vkCmdCopyBuffer2(cmd, &copyBufferInfo);
	Application::app->submitAndWaitTempCmdBuffer(cmd);
}
