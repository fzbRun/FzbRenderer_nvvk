#include "common/Application/Application.h"
#include <nvutils/timers.hpp>
#include "common/Shader/Shader.h"
#include <iostream>


VkShaderModuleCreateInfo FzbRenderer::compileSlangShader(const std::filesystem::path& shaderSource, const std::span<const uint32_t>& spirv) {
	SCOPED_TIMER(__FUNCTION__);

	VkShaderModuleCreateInfo shaderCode = nvsamples::getShaderModuleCreateInfo(spirv);
	if (Application::slangCompiler.compileFile(shaderSource))
	{
		shaderCode.codeSize = Application::slangCompiler.getSpirvSize();
		shaderCode.pCode = Application::slangCompiler.getSpirv();
	}
	else
	{
		LOGE("Error compiling shers: %s\n%s\n", shaderSource.string().c_str(),
			Application::slangCompiler.getLastDiagnosticMessage().c_str());
	}
	return shaderCode;
};

void FzbRenderer::saveShaderToFile(const VkShaderModuleCreateInfo& shaderInfo, std::filesystem::path& shaderSource) {
	std::ofstream file(shaderSource, std::ios::binary);
	if (file.is_open()) {
		// codeSize 是字节数，pCode 是 uint32_t*，所以按字节写入
		file.write(reinterpret_cast<const char*>(shaderInfo.pCode), shaderInfo.codeSize);
		file.close();
		printf("Saved shader to : %s\n", shaderSource);
	}
	else {
		printf("Failed to open file: : %s\n", shaderSource);
	}
}

bool FzbRenderer::validateSPIRVFile(std::filesystem::path& shaderSource) {
	std::ifstream file(shaderSource, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		return false;
	}

	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<char> buffer(size);
	if (file.read(buffer.data(), size)) {
		// 检查 SPIR-V 魔数 (0x07230203)
		if (size >= 4) {
			uint32_t magic = *reinterpret_cast<uint32_t*>(buffer.data());
			bool isValid = (magic == 0x07230203);
			std::cout << "SPIR-V Magic: 0x" << std::hex << magic
				<< " (Valid: " << (isValid ? "Yes" : "No") << ")" << std::endl;
			return isValid;
		}
	}
	return false;
}