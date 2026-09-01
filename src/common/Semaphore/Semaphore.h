#pragma once
#include <vulkan/vulkan_core.h>
#include <nvvk/resources.hpp>
#include <nvvk/semaphore.hpp>
#include <volk.h>

#ifndef FZBRENDERER_SEMAPHORE_FZBPATHGUIDING_H
#define FZBRENDERER_SEMAPHORE_FZBPATHGUIDING_H
namespace FzbRenderer {
class Semaphore {
public:
	Semaphore() = default;
	~Semaphore() = default;

	VkResult init(bool timeline = true, uint64_t initialValue = 0, bool external = false);
	void clean();

	nvvk::SemaphoreState semaphoreState;
	HANDLE handle = nullptr;
};
}

#endif