#include "./Semaphore.h"
#include <common/Application/Application.h>

void GetSemaphoreWin32HandleKHR(VkDevice device, VkSemaphoreGetWin32HandleInfoKHR* handleInfo, HANDLE* handle)
{
	auto func = (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR");
	if (func != nullptr)
	{
		func(device, handleInfo, handle);
	}
}
VkResult FzbRenderer::Semaphore::init(bool timeline, uint64_t initialValue, bool external) {
	VkDevice device = Application::allocator.getDevice();

	void* pNext = nullptr;
	VkSemaphoreTypeCreateInfo timelineSemaphoreCreateInfo{};
	if (timeline) {
		timelineSemaphoreCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
			.pNext = pNext,
			.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
			.initialValue = 0	//时间线信号量的初始值必须为0，后续通过信号量状态来设置实际的时间线值
		};
		pNext = &timelineSemaphoreCreateInfo;
	}

	VkExportSemaphoreCreateInfoKHR exportInfo{};
	if (external) {
		exportInfo = {
			.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
			.pNext = pNext,
			.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
		};
		pNext = &exportInfo;
	}

	VkSemaphoreCreateInfo semaphoreInfo{};
	semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	semaphoreInfo.pNext = pNext;
	semaphoreInfo.flags = 0;

	VkSemaphore semaphore;
	if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS) {
		throw std::runtime_error("failed to create semaphores!");
	}
	
	if (timeline) semaphoreState = nvvk::SemaphoreState::makeDynamic(semaphore);
	else semaphoreState = nvvk::SemaphoreState::makeFixed(semaphore, initialValue);

	if (external) {
		VkSemaphoreGetWin32HandleInfoKHR handleInfo = {};
		handleInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
		handleInfo.semaphore = semaphore;
		handleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
		GetSemaphoreWin32HandleKHR(device, &handleInfo, &handle);
	}

	return VK_SUCCESS;
}
void FzbRenderer::Semaphore::clean() {
	VkDevice device = Application::allocator.getDevice();
	vkDestroySemaphore(device, semaphoreState.getSemaphore(), nullptr);

	if (handle != nullptr) { // HANDLE 在 Win32 下通常是 void* / HANDLE
		CloseHandle(handle);
		handle = nullptr;
	}
}