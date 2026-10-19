#pragma once

#ifndef FZBRENDERER_FLIP_H
#define FZBRENDERER_FLIP_H

#include "feature/Feature.h"
#include "common/Image/Image.h"
#include "./FLIPShaderio.h"

#include <filesystem>

namespace FzbRenderer {

// FLIP 中间结果，全部由 Feature 自己管理，外部只需要提供 reference/test 两张输入图
enum class GBuffers_FLIP {
	eOpponentReference = 0,		// YyCxCz 空间下的 reference
	eOpponentTest,				// YyCxCz 空间下的 test
	eHorizontalReference,		// (Y_h, Cx_h, dI/dx, d2I/dx2)
	eHorizontalReferenceCz,		// (Cz1_h, Cz2_h, gaussian(I), 0)
	eHorizontalTest,			// 同 eHorizontalReference，对应 test
	eHorizontalTestCz,			// 同 eHorizontalReferenceCz，对应 test
	eFlipResult,				// (display RGB, FLIP)

	bufferCount,
};

struct FLIPCreateInfo {
	nvvk::Image referenceImage{};			// 参考图（必须与 test 同分辨率）
	nvvk::Image testImage{};				// 被测图
	std::filesystem::path referenceImagePath{};	// 可选：从文件加载参考图（会被 Image 类上传）
	float pixelsPerDegree = 67.0f;			// 观察条件，默认 4K / 0.69m / 0.70m
	bool referenceIsSRGB = true;			// 输入是否为 sRGB 编码（LDR）
	bool testIsSRGB = true;
};

/*
FLIP 使用方式（在任意 Renderer 中）：
	FLIP flip;
	FLIPCreateInfo info{
		.referenceImage = referenceImage,	// 参考图 nvvk::Image
		.testImage = testImage,				// 被测图 nvvk::Image
		.pixelsPerDegree = 67.0f,
	};
	flip.init(info);						// 放在宿主 Renderer::init() 中、提交 staging 之前
	flip.resize(cmd, size);					// 窗口/分辨率变化时调用
	flip.setInputImages(ref, test);			// 输入图变化时调用（可选）
	flip.evaluate(cmd);						// 生成 FLIP error map
	nvvk::Image& result = flip.getResultImage();	// R = FLIP
*/
class FLIP : public Feature {
public:
	FLIP();
	virtual ~FLIP() = default;

	void init(FLIPCreateInfo createInfo);
	void clean() override;
	void uiRender() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void render(VkCommandBuffer* cmdPtr) override;
	void render(VkCommandBuffer cmd);
	void evaluate(VkCommandBuffer cmd);
	void evaluate(VkCommandBuffer cmd, nvvk::Image referenceImage, nvvk::Image testImage);

	// 输入图改变时调用（例如每帧双缓冲切换），只更新描述符，不重建纹理
	void resize(VkCommandBuffer cmd, const VkExtent2D& size, nvvk::Image referenceImage, nvvk::Image testImage);
	void setInputImages(nvvk::Image referenceImage, nvvk::Image testImage);
	bool hasValidInputs() const { return inputsValid; }

	// 结果图：R = FLIP error map，后续 Renderer 可以直接读取/显示
	nvvk::Image& getResultImage() { return gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_FLIP::eFlipResult]; }
	VkDescriptorSet getResultDescriptorSet() { return gBuffers.getDescriptorSet((uint32_t)GBuffers_FLIP::eFlipResult); }
	const VkDescriptorImageInfo& getResultDescriptorImageInfo() { return gBuffers.getDescriptorImageInfo((uint32_t)GBuffers_FLIP::eFlipResult); }

	void createDescriptorSetLayout();
	void createDescriptorSet();
	void createPipeline();
	void compileAndCreateShaders() override;

	FLIPCreateInfo setting;
	nvvk::Buffer kernelBuffer{};
	bool showErrorMap = false;

private:
	void buildKernels();
	void updatePushConstant();
	void updateGBufferDescriptors();
	void updateInputDescriptors();

	void dispatchPreprocess(VkCommandBuffer cmd);
	void dispatchHorizontal(VkCommandBuffer cmd);
	void dispatchEvaluate(VkCommandBuffer cmd);

	int calculateSpatialFilterRadius(float ppd) const;
	int calculateFeatureFilterRadius(float ppd) const;
	float computeCMax() const;

	shaderio::FLIPKernelData kernelData{};
	shaderio::FLIPPushConstant pushConstant{};

	bool kernelDirty = true;
	bool inputsValid = false;
	bool warnedNoInputs = false;
	VkExtent2D resolution{ 0, 0 };

	// 通过 referenceImagePath 加载参考图时，用它持有纹理生命周期
	FzbRenderer::Image referenceImageFile;

	VkShaderEXT computeShader_preprocess{};
	VkShaderEXT computeShader_horizontal{};
	VkShaderEXT computeShader_evaluate{};
};

}

#endif
