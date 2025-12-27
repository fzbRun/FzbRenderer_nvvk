#pragma once
#include <vulkan/vulkan_core.h>
#include <filesystem>
#include <span>
#include <common/utils.hpp>

#ifndef FZB_SHADER_H
#define FZB_SHADER_H

namespace FzbRenderer{
	VkShaderModuleCreateInfo compileSlangShader(const std::filesystem::path& shaderSource, const std::span<const uint32_t>& spirv);
	void saveShaderToFile(const VkShaderModuleCreateInfo& shaderInfo, std::filesystem::path& shaderSource);
	bool validateSPIRVFile(std::filesystem::path& shaderSource);
}

#endif