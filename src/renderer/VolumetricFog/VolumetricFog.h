#pragma once

#ifndef FZBRENDERER_VOLUMETRIC_FOG_H
#define FZBRENDERER_VOLUMETRIC_FOG_H

#include "renderer/Renderer.h"
#include <common/Image/Image.h>
#include "./VolumetricFogShaderio.h"
#include <feature/ShadowMap/ShadowMap.h>
#include <common/Buffer/Buffer.h>

namespace FzbRenderer {
enum class GBuffers_VolumetricFog{
	eAlbedo = 0,
	eNormal,
	eEmissive,
	eRendered,
	eTonemapping,
};

class VolumetricFog : public Renderer {
public:
	VolumetricFog() = default;
	~VolumetricFog() = default;

	VolumetricFog(pugi::xml_node& rendererNode);

	void init() override;
	void clean() override;
	void uiRender() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender() override;
	void render(VkCommandBuffer* cmd) override;

private:
	void createVolumetricFogImage(FzbRenderer::Image& image, shaderio::uint3 size);
	void createVolumetricFogData();

	void createDescriptorSetLayout() override;
	void createDescriptorSet();
	void createPipelineLayout();
	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	void initVolumetricFogFluid(VkCommandBuffer cmd);
	void createGBuffers(VkCommandBuffer cmd);
	void createVolumetricFog(VkCommandBuffer cmd);
	void deferredRenderring(VkCommandBuffer cmd);

	VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR derivFeatures = {};

	VkPushConstantsInfo pushInfo;

	VkShaderEXT vertexShader_createGBuffer{};
	VkShaderEXT fragmentShader_createGBuffer{};

	VkShaderEXT computeShader_getVisibleVolumetricFog{};

	VkShaderEXT computeShader_initVolumetricFogFluid{};
	VkShaderEXT computeShader_createVolumetricFog_Fluid_A{};
	VkShaderEXT computeShader_createVolumetricFog_Fluid_D{};
	VkShaderEXT computeShader_createVolumetricFog_Fluid_F{};
	VkShaderEXT computeShader_createVolumetricFog_Fluid_P{};
	VkShaderEXT computeShader_createVolumetricFog_Fluid_S{};

	VkShaderEXT computeShader_createLightAttenuationEstimator{};

	VkShaderEXT computeShader_deferredRenderring{};

	shaderio::VolumetricFogPushConstant pushConstant;

	ShadowMap shadowMap;

	FzbRenderer::Buffer GlobalInfoBuffer;
	FzbRenderer::Buffer visibleVolumetricFogIndexBuffer;					//相机范围内的体积雾索引

	uint32_t volumetricFogCount = 1;										//体积雾数量
	std::vector<shaderio::VolumetricFogInfo> volumetricFogInfos;			//体积雾基础信息
	FzbRenderer::Buffer volumetricFogInfosBuffer;							//体积雾基础信息缓冲区
	std::vector<int> volumetricFogInfoModified;								//体积雾数据是否被修改，每帧重置

	uint32_t volumetricFogHeightCount = 0;									//高度雾数量
	std::map<int, int> volumetricFogHeightIndexMap;							//高度雾索引 -> 体积雾索引
	std::vector<shaderio::HeightFogInfo> volumetricFogHeightInfos;			//流体雾基础信息
	FzbRenderer::Buffer volumetricFogHeightInfoBuffer;						//流体雾基础信息缓冲区

	uint32_t volumetricFogFluidCount = 0;									//流体雾数量
	std::map<int, int> volumetricFogFluidIndexMap;							//流体雾索引 -> 体积雾索引
	std::vector<shaderio::FluidFogInfo> volumetricFogFluidInfos;			//流体雾基础信息
	FzbRenderer::Buffer volumetricFogFluidInfoBuffer;						//流体雾基础信息缓冲区
	std::vector<FzbRenderer::Buffer> volumetricFogFluidVoxelInfoBuffers;	//流体雾voxel信息缓冲区
	std::vector<FzbRenderer::Image> volumetricFogFluidVoxelVelocityImages;	//流体雾voxel速度3DTexture
	std::vector<FzbRenderer::Image> volumetricFogFluidExtinctionImages;		//流体雾消光系数3DTexture

	uint32_t volumetricFogNoiseCount = 0;
	std::map<int, int> volumetricFogNoiseIndexMap;							//噪声雾索引 -> 体积雾索引
	std::vector<shaderio::NoiseFogInfo> volumetricFogNoiseInfos;			//流体雾基础信息
	FzbRenderer::Buffer volumetricFogNoiseInfoBuffer;						//流体雾基础信息缓冲区

#ifndef NDEBUG
	void renderVolumetricFogVoxelGrid(VkCommandBuffer cmd);

	VkShaderEXT vertexShader_renderVoxelGrid{};
	VkShaderEXT fragmentShader_renderVoxelGrid{};

	bool showVolumetricFogVoxelGrid = false;
	std::vector<int> showVolumetricFogVoxelGrids;
#endif
};
}

#endif