#include "./FLIP.h"

#include <common/Application/Application.h>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/default_structs.hpp>
#include <nvgui/property_editor.hpp>

#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

using namespace FzbRenderer;

namespace {

constexpr float FLIP_PI = 3.14159265358979323846f;

// FLIP 论文中的 CSF 高斯参数：Yy / Cx 各一个高斯，Cz 为两个高斯之和
constexpr float CSF_B_Y = 0.0047f;
constexpr float CSF_B_CX = 0.0053f;
constexpr float CSF_A_CZ_1 = 34.1f;
constexpr float CSF_B_CZ_1 = 0.04f;
constexpr float CSF_A_CZ_2 = 13.5f;
constexpr float CSF_B_CZ_2 = 0.025f;

// 特征检测：边缘检测器在视网膜上的宽度为 0.082 度
constexpr float FEATURE_WIDTH_DEGREE = 0.082f;
constexpr float Q_C = 0.7f;
constexpr float Q_F = 0.5f;
constexpr float P_C = 0.4f;
constexpr float P_T = 0.95f;

const glm::vec3 INV_DEFAULT_ILLUMINANT(1.052156925f, 1.000000000f, 0.918357670f);

float sRGBToLinear(float c) {
	return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

glm::vec3 linearRGBToXYZ(glm::vec3 rgb) {
	// D65
	const float a11 = 10135552.0f / 24577794.0f;
	const float a12 = 8788810.0f / 24577794.0f;
	const float a13 = 4435075.0f / 24577794.0f;
	const float a21 = 2613072.0f / 12288897.0f;
	const float a22 = 8788810.0f / 12288897.0f;
	const float a23 = 887015.0f / 12288897.0f;
	const float a31 = 1425312.0f / 73733382.0f;
	const float a32 = 8788810.0f / 73733382.0f;
	const float a33 = 70074185.0f / 73733382.0f;

	return glm::vec3(a11 * rgb.x + a12 * rgb.y + a13 * rgb.z,
		a21 * rgb.x + a22 * rgb.y + a23 * rgb.z,
		a31 * rgb.x + a32 * rgb.y + a33 * rgb.z);
}

glm::vec3 XYZToCIELab(glm::vec3 xyz) {
	const float delta = 6.0f / 29.0f;
	const float deltaSquare = delta * delta;
	const float deltaCube = delta * deltaSquare;
	const float factor = 1.0f / (3.0f * deltaSquare);
	const float term = 4.0f / 29.0f;

	xyz *= INV_DEFAULT_ILLUMINANT;
	const auto f = [&](float v) {
		return v > deltaCube ? std::pow(v, 1.0f / 3.0f) : factor * v + term;
	};

	const float fx = f(xyz.x);
	const float fy = f(xyz.y);
	const float fz = f(xyz.z);

	return glm::vec3(116.0f * fy - 16.0f, 500.0f * (fx - fy), 200.0f * (fy - fz));
}

float hunt(float luminance, float chrominance) {
	return 0.01f * luminance * chrominance;
}

float hyAB(const glm::vec3& ref, const glm::vec3& test) {
	const float dL = std::fabs(ref.x - test.x);
	const float da = ref.y - test.y;
	const float db = ref.z - test.z;
	return dL + std::sqrt(da * da + db * db);
}

float gaussian(float x2, float a, float b) {
	return a * std::sqrt(FLIP_PI / b) * std::exp(-FLIP_PI * FLIP_PI * x2 / b);
}

// 用于把高斯和拆成两个可分离一维核，sqrt 后的幅度在二维乘积中恢复为 a
float gaussianSqrt(float x2, float a, float b) {
	return std::sqrt(a * std::sqrt(FLIP_PI / b)) * std::exp(-FLIP_PI * FLIP_PI * x2 / b);
}

void normalizeKernel(float* kernel, int radius) {
	float sum = 0.0f;
	for (int i = -radius; i <= radius; ++i)
		sum += kernel[FLIP_MAX_KERNEL_RADIUS + i];

	if (sum == 0.0f)
		return;

	const float invSum = 1.0f / sum;
	for (int i = -radius; i <= radius; ++i)
		kernel[FLIP_MAX_KERNEL_RADIUS + i] *= invSum;
}

// FLIP 要求将正权重之和归一化为 1，负权重之和归一化为 -1
void normalizeDerivativeKernel(float* kernel, int radius) {
	float positiveSum = 0.0f;
	float negativeSum = 0.0f;
	for (int i = -radius; i <= radius; ++i) {
		const float w = kernel[FLIP_MAX_KERNEL_RADIUS + i];
		if (w > 0.0f) positiveSum += w;
		else if (w < 0.0f) negativeSum += w;
	}

	if (positiveSum > 0.0f) {
		for (int i = -radius; i <= radius; ++i) {
			float& w = kernel[FLIP_MAX_KERNEL_RADIUS + i];
			if (w > 0.0f) w /= positiveSum;
		}
	}
	if (negativeSum < 0.0f) {
		for (int i = -radius; i <= radius; ++i) {
			float& w = kernel[FLIP_MAX_KERNEL_RADIUS + i];
			if (w < 0.0f) w /= -negativeSum;
		}
	}
}

}

FLIP::FLIP()
	: referenceImageFile("FLIPReference") {
}

void FLIP::init(FLIPCreateInfo createInfo) {
	setting = createInfo;

	// 可选的参考图文件加载。Image 会把上传命令塞进 stagingUploader，
	// 由宿主 Renderer 的 init() 统一提交（和 TAA/SVGF 的纹理一致）。
	if (!setting.referenceImagePath.empty() && setting.referenceImage.image == VK_NULL_HANDLE) {
		std::filesystem::path path = setting.referenceImagePath;
		if (referenceImageFile.init(path) == VK_SUCCESS)
			setting.referenceImage = referenceImageFile.image;
		else
			LOGW("FLIP: failed to load reference image: %s\n", path.string().c_str());
	}

	resolution = { 1, 1 };
	if (setting.referenceImage.image != VK_NULL_HANDLE) {
		resolution = { setting.referenceImage.extent.width, setting.referenceImage.extent.height };
	}
	else if (setting.testImage.image != VK_NULL_HANDLE) {
		resolution = { setting.testImage.extent.width, setting.testImage.extent.height };
	}
	else {
		const VkExtent2D viewportSize = Application::app->getViewportSize();
		if (viewportSize.width != 0 && viewportSize.height != 0)
			resolution = viewportSize;
	}

	inputsValid = setting.referenceImage.image != VK_NULL_HANDLE && setting.testImage.image != VK_NULL_HANDLE;
	if (!inputsValid)
		LOGW("FLIP: reference/test image is not set, call setInputImages() before render().\n");

	Feature::createGBuffer(false, false, (uint32_t)GBuffers_FLIP::bufferCount, resolution);

	createDescriptorSetLayout();
	createPipeline();

	NVVK_CHECK(Application::allocator.createBuffer(kernelBuffer, sizeof(shaderio::FLIPKernelData),
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT));
	NVVK_DBG_NAME(kernelBuffer.buffer);

	createDescriptorSet();
	buildKernels();
	compileAndCreateShaders();

	Feature::init();
}

void FLIP::clean() {
	VkDevice device = Application::app->getDevice();

	vkDestroyShaderEXT(device, computeShader_preprocess, nullptr);
	vkDestroyShaderEXT(device, computeShader_horizontal, nullptr);
	vkDestroyShaderEXT(device, computeShader_evaluate, nullptr);
	computeShader_preprocess = VK_NULL_HANDLE;
	computeShader_horizontal = VK_NULL_HANDLE;
	computeShader_evaluate = VK_NULL_HANDLE;

	if (kernelBuffer.buffer != VK_NULL_HANDLE) {
		Application::allocator.destroyBuffer(kernelBuffer);
		kernelBuffer = {};
	}

	if (referenceImageFile.image.image != VK_NULL_HANDLE)
		referenceImageFile.clean();

	Feature::clean();
}

void FLIP::uiRender() {
	namespace PE = nvgui::PropertyEditor;

	if (ImGui::Begin("FLIP")) {
		if (!inputsValid)
			ImGui::TextColored({ 1.0f, 0.4f, 0.1f, 1.0f }, "Reference/Test image not set");

		PE::begin();
		if (PE::DragFloat("Pixels Per Degree", &setting.pixelsPerDegree, 1.0f, 1.0f, 200.0f, "%.1f",
			ImGuiSliderFlags_AlwaysClamp, "Observer distance and display resolution (PPD)"))
			buildKernels();
		PE::end();

		ImGui::Text("CSF radius: %d px", pushConstant.csfRadius);
		ImGui::Text("Feature radius: %d px", pushConstant.featureRadius);
		ImGui::Text("cMax: %.3f (pc*cMax: %.3f)", pushConstant.cMax, pushConstant.pcCMax);

		ImGui::Separator();
		ImGui::Combo("Debug View", &pushConstant.debugView, "FLIP\0Color Difference\0Feature Difference\0");
		ImGui::Checkbox("Show FLIP Error Map", &showErrorMap);
		if (ImGui::Button("Rebuild Kernels"))
			buildKernels();

		if (showErrorMap)
			Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)GBuffers_FLIP::eFlipResult);
	}
	ImGui::End();
}

void FLIP::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
	if (size.width == 0 || size.height == 0)
		return;

	resolution = size;
	pushConstant.screenSize = { size.width, size.height };

	NVVK_CHECK(gBuffers.update(cmd, size));
	updateGBufferDescriptors();
}

void FLIP::resize(VkCommandBuffer cmd, const VkExtent2D& size, nvvk::Image referenceImage, nvvk::Image testImage) {
	setInputImages(referenceImage, testImage);
	resize(cmd, size);
}

void FLIP::setInputImages(nvvk::Image referenceImage, nvvk::Image testImage) {
	setting.referenceImage = referenceImage;
	setting.testImage = testImage;

	inputsValid = setting.referenceImage.image != VK_NULL_HANDLE && setting.testImage.image != VK_NULL_HANDLE;
	if (!inputsValid) {
		warnedNoInputs = false;
		return;
	}

	if (setting.referenceImage.extent.width != setting.testImage.extent.width ||
		setting.referenceImage.extent.height != setting.testImage.extent.height) {
		LOGW("FLIP: reference and test images have different resolutions (%ux%u vs %ux%u).\n",
			setting.referenceImage.extent.width, setting.referenceImage.extent.height,
			setting.testImage.extent.width, setting.testImage.extent.height);
	}

	updateInputDescriptors();
	updatePushConstant();
}

void FLIP::render(VkCommandBuffer* cmdPtr) {
	evaluate(cmdPtr[0]);
}

void FLIP::render(VkCommandBuffer cmd) {
	evaluate(cmd);
}

void FLIP::evaluate(VkCommandBuffer cmd) {
	if (!inputsValid) {		if (!warnedNoInputs) {
			LOGW("FLIP: skip rendering because reference/test image is not set.\n");
			warnedNoInputs = true;
		}
		return;
	}

	NVVK_DBG_SCOPE(cmd);

	// 宿主 Renderer 可能刚写完 test image，这里先保证输入对 compute 可见
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	if (kernelDirty) {
		vkCmdUpdateBuffer(cmd, kernelBuffer.buffer, 0, sizeof(shaderio::FLIPKernelData), &kernelData);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
		kernelDirty = false;
	}

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	const VkPushConstantsInfo pushInfo{
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::FLIPPushConstant),
		.pValues = &pushConstant,
	};
	vkCmdPushConstants2(cmd, &pushInfo);

	dispatchPreprocess(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	dispatchHorizontal(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	dispatchEvaluate(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
}

void FLIP::evaluate(VkCommandBuffer cmd, nvvk::Image referenceImage, nvvk::Image testImage) {
	setInputImages(referenceImage, testImage);
	evaluate(cmd);
}

void FLIP::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;

	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_FLIP::eReferenceImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_FLIP::eTestImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	const shaderio::BindingPoints_FLIP storageBindings[] = {
		shaderio::BindingPoints_FLIP::eOpponentReference,
		shaderio::BindingPoints_FLIP::eOpponentTest,
		shaderio::BindingPoints_FLIP::eHorizontalReference,
		shaderio::BindingPoints_FLIP::eHorizontalReferenceCz,
		shaderio::BindingPoints_FLIP::eHorizontalTest,
		shaderio::BindingPoints_FLIP::eHorizontalTestCz,
		shaderio::BindingPoints_FLIP::eResultImage,
	};
	for (const auto binding : storageBindings) {
		bindings.addBinding({
			.binding = (uint32_t)binding,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
	}

	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_FLIP::eKernelBuffer,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	LOGI("FLIP static descriptor layout created\n");
	NVVK_DBG_NAME(staticDescPack.getLayout());
	NVVK_DBG_NAME(staticDescPack.getPool());
	NVVK_DBG_NAME(staticDescPack.getSet(0));
}

void FLIP::createDescriptorSet() {
	updateInputDescriptors();
	updateGBufferDescriptors();

	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet kernelWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_FLIP::eKernelBuffer, 0, 0, 1);
	write.append(kernelWrite, kernelBuffer, 0, sizeof(shaderio::FLIPKernelData));
	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}

void FLIP::createPipeline() {
	Feature::createPipelineLayout(sizeof(shaderio::FLIPPushConstant));
}

void FLIP::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	VkDevice device = Application::app->getDevice();

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::FLIPPushConstant),
	};

	std::array<VkDescriptorSetLayout, 1> layouts = { { staticDescPack.getLayout() } };
	VkShaderCreateInfoEXT shaderInfo{
		.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
		.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
		.pName = "main",
		.setLayoutCount = (uint32_t)layouts.size(),
		.pSetLayouts = layouts.data(),
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};

	const auto createShader = [&](const char* fileName, const char* entryPoint, VkShaderEXT& shader) {
		// Use the cached .spv overload. The Debug Slang compiler can take
		// minutes on flipEvaluate.slang, which blocks app startup.
		std::filesystem::path sourcePath = shaderPath / fileName;
		std::vector<uint32_t> shaderBuffer;
		VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(sourcePath, shaderBuffer);
		vkDestroyShaderEXT(device, shader, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = entryPoint;
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		NVVK_CHECK(vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &shader));
		NVVK_DBG_NAME(shader);
	};

	createShader("flipPreprocess.slang", "computeMain_preprocess", computeShader_preprocess);
	createShader("flipHorizontal.slang", "computeMain_horizontal", computeShader_horizontal);
	createShader("flipEvaluate.slang", "computeMain_evaluate", computeShader_evaluate);
}

void FLIP::dispatchPreprocess(VkCommandBuffer cmd) {
	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_preprocess);
	const VkExtent3D groupSize = nvvk::getGroupCounts(
		VkExtent3D{ resolution.width, resolution.height, 1 }, VkExtent3D{ 32, 32, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
}

void FLIP::dispatchHorizontal(VkCommandBuffer cmd) {
	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_horizontal);
	const VkExtent3D groupSize = nvvk::getGroupCounts(
		VkExtent3D{ resolution.width, resolution.height, 1 }, VkExtent3D{ 32, 32, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
}

void FLIP::dispatchEvaluate(VkCommandBuffer cmd) {
	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_evaluate);
	const VkExtent3D groupSize = nvvk::getGroupCounts(
		VkExtent3D{ resolution.width, resolution.height, 1 }, VkExtent3D{ 32, 32, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
}

void FLIP::updateGBufferDescriptors() {
	nvvk::WriteSetContainer write{};

	VkWriteDescriptorSet ws =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_FLIP::eOpponentReference, 0, 0, 1);
	write.append(ws, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_FLIP::eOpponentReference]);

	ws = staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_FLIP::eOpponentTest, 0, 0, 1);
	write.append(ws, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_FLIP::eOpponentTest]);

	ws = staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_FLIP::eHorizontalReference, 0, 0, 1);
	write.append(ws, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_FLIP::eHorizontalReference]);

	ws = staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_FLIP::eHorizontalReferenceCz, 0, 0, 1);
	write.append(ws, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_FLIP::eHorizontalReferenceCz]);

	ws = staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_FLIP::eHorizontalTest, 0, 0, 1);
	write.append(ws, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_FLIP::eHorizontalTest]);

	ws = staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_FLIP::eHorizontalTestCz, 0, 0, 1);
	write.append(ws, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_FLIP::eHorizontalTestCz]);

	ws = staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_FLIP::eResultImage, 0, 0, 1);
	write.append(ws, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_FLIP::eFlipResult]);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}

void FLIP::updateInputDescriptors() {
	if (setting.referenceImage.image == VK_NULL_HANDLE || setting.testImage.image == VK_NULL_HANDLE)
		return;

	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet ws =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_FLIP::eReferenceImage, 0, 0, 1);
	write.append(ws, setting.referenceImage);

	ws = staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_FLIP::eTestImage, 0, 0, 1);
	write.append(ws, setting.testImage);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}

int FLIP::calculateSpatialFilterRadius(float ppd) const {
	const float maxScaleParameter = std::max(std::max(CSF_B_Y, CSF_B_CX), std::max(CSF_B_CZ_1, CSF_B_CZ_2));
	const float sigmaMax = std::sqrt(maxScaleParameter / (2.0f * FLIP_PI * FLIP_PI));
	int radius = (int)std::ceil(3.0f * sigmaMax * std::max(ppd, 1.0f));
	return std::clamp(radius, 1, FLIP_MAX_KERNEL_RADIUS);
}

int FLIP::calculateFeatureFilterRadius(float ppd) const {
	const float sigma = 0.5f * FEATURE_WIDTH_DEGREE * std::max(ppd, 1.0f);
	int radius = (int)std::ceil(3.0f * sigma);
	return std::clamp(radius, 1, FLIP_MAX_KERNEL_RADIUS);
}

float FLIP::computeCMax() const {
	const glm::vec3 greenLab = XYZToCIELab(linearRGBToXYZ(glm::vec3(0.0f, 1.0f, 0.0f)));
	const glm::vec3 blueLab = XYZToCIELab(linearRGBToXYZ(glm::vec3(0.0f, 0.0f, 1.0f)));

	const glm::vec3 greenHunt(greenLab.x, hunt(greenLab.x, greenLab.y), hunt(greenLab.x, greenLab.z));
	const glm::vec3 blueHunt(blueLab.x, hunt(blueLab.x, blueLab.y), hunt(blueLab.x, blueLab.z));

	return std::pow(hyAB(greenHunt, blueHunt), Q_C);
}

void FLIP::buildKernels() {
	std::memset(&kernelData, 0, sizeof(shaderio::FLIPKernelData));

	const float ppd = std::max(setting.pixelsPerDegree, 1.0f);
	const int csfRadius = calculateSpatialFilterRadius(ppd);
	const int featureRadius = calculateFeatureFilterRadius(ppd);

	for (int i = -csfRadius; i <= csfRadius; ++i) {
		const float x = (float)i / ppd;
		const float x2 = x * x;
		const int index = FLIP_MAX_KERNEL_RADIUS + i;

		kernelData.csfY[index] = gaussian(x2, 1.0f, CSF_B_Y);
		kernelData.csfCx[index] = gaussian(x2, 1.0f, CSF_B_CX);
		kernelData.csfCz1[index] = gaussianSqrt(x2, CSF_A_CZ_1, CSF_B_CZ_1);
		kernelData.csfCz2[index] = gaussianSqrt(x2, CSF_A_CZ_2, CSF_B_CZ_2);
	}

	normalizeKernel(kernelData.csfY, csfRadius);
	normalizeKernel(kernelData.csfCx, csfRadius);

	// Cz 是两个不同方差的高斯之和，不能合并成一个一维核（会产生交叉项）。
	// 参考实现归一化的是二维组合核，因此这里用两个一维核的二维和来求缩放系数。
	float czSum1 = 0.0f;
	float czSum2 = 0.0f;
	for (int i = -csfRadius; i <= csfRadius; ++i) {
		const int index = FLIP_MAX_KERNEL_RADIUS + i;
		czSum1 += kernelData.csfCz1[index];
		czSum2 += kernelData.csfCz2[index];
	}
	const float czTotal = czSum1 * czSum1 + czSum2 * czSum2;
	const float czScale = czTotal > 0.0f ? 1.0f / std::sqrt(czTotal) : 0.0f;
	for (int i = -csfRadius; i <= csfRadius; ++i) {
		const int index = FLIP_MAX_KERNEL_RADIUS + i;
		kernelData.csfCz1[index] *= czScale;
		kernelData.csfCz2[index] *= czScale;
	}

	const float sigma = 0.5f * FEATURE_WIDTH_DEGREE * ppd;
	const float sigma2 = sigma * sigma;
	const float sigma4 = sigma2 * sigma2;

	for (int i = -featureRadius; i <= featureRadius; ++i) {
		const int index = FLIP_MAX_KERNEL_RADIUS + i;
		const float g = std::exp(-(float)(i * i) / (2.0f * sigma2));
		kernelData.featureGaussian[index] = g;
		kernelData.featureFirstDerivative[index] = -(float)i / sigma2 * g;
		kernelData.featureSecondDerivative[index] = ((float)(i * i) - sigma2) / sigma4 * g;
	}

	normalizeKernel(kernelData.featureGaussian, featureRadius);
	normalizeDerivativeKernel(kernelData.featureFirstDerivative, featureRadius);
	normalizeDerivativeKernel(kernelData.featureSecondDerivative, featureRadius);

	updatePushConstant();
	kernelDirty = true;
}

void FLIP::updatePushConstant() {
	pushConstant.screenSize = { resolution.width, resolution.height };
	pushConstant.csfRadius = calculateSpatialFilterRadius(setting.pixelsPerDegree);
	pushConstant.featureRadius = calculateFeatureFilterRadius(setting.pixelsPerDegree);
	pushConstant.pixelsPerDegree = setting.pixelsPerDegree;
	pushConstant.cMax = computeCMax();
	pushConstant.pcCMax = P_C * pushConstant.cMax;
	pushConstant.pT = P_T;
	pushConstant.qC = Q_C;
	pushConstant.qF = Q_F;
	pushConstant.referenceIsSRGB = setting.referenceIsSRGB ? 1 : 0;
	pushConstant.testIsSRGB = setting.testIsSRGB ? 1 : 0;
}
