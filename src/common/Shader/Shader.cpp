#include "common/Application/Application.h"
#include <nvutils/timers.hpp>
#include "common/Shader/Shader.h"


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