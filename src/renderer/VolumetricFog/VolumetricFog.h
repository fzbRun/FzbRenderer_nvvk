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

	// mouse force
	glm::vec3 mouseForcePosition = glm::vec3(0.0f);
	float mouseForceStrength = 0.0f;
	float mouseForceRadius = 3.0f;

	FzbRenderer::Buffer GlobalInfoBuffer;
	FzbRenderer::Buffer visibleVolumetricFogIndexBuffer;					//�����Χ�ڵ����������

	uint32_t volumetricFogCount = 1;										//���������
	std::vector<shaderio::VolumetricFogInfo> volumetricFogInfos;			//�����������Ϣ
	FzbRenderer::Buffer volumetricFogInfosBuffer;							//�����������Ϣ������
	std::vector<int> volumetricFogInfoModified;								//����������Ƿ��޸ģ�ÿ֡����

	uint32_t volumetricFogHeightCount = 0;									//�߶�������
	std::map<int, int> volumetricFogHeightIndexMap;							//�߶������� -> ���������
	std::vector<shaderio::HeightFogInfo> volumetricFogHeightInfos;			//������������Ϣ
	FzbRenderer::Buffer volumetricFogHeightInfoBuffer;						//������������Ϣ������

	uint32_t volumetricFogFluidCount = 0;									//����������
	std::map<int, int> volumetricFogFluidIndexMap;							//���������� -> ���������
	std::vector<shaderio::FluidFogInfo> volumetricFogFluidInfos;			//������������Ϣ
	FzbRenderer::Buffer volumetricFogFluidInfoBuffer;						//������������Ϣ������
	std::vector<FzbRenderer::Buffer> volumetricFogFluidVoxelInfoBuffers;	//������voxel��Ϣ������
	std::vector<FzbRenderer::Image> volumetricFogFluidVoxelVelocityImages;	//������voxel�ٶ�3DTexture
	std::vector<FzbRenderer::Image> volumetricFogFluidVoxelInfoImages;		//������ voxel��Ϣ 3DTexture x: ���� y: ɢ�� z: phase

	uint32_t volumetricFogNoiseCount = 0;
	std::map<int, int> volumetricFogNoiseIndexMap;							//���������� -> ���������
	std::vector<shaderio::NoiseFogInfo> volumetricFogNoiseInfos;			//������������Ϣ
	FzbRenderer::Buffer volumetricFogNoiseInfoBuffer;						//������������Ϣ������

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