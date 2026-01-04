#pragma once
/*
sceneManager主要有三个功能
1. 从sceneXML中读取信息，主要包括
	a. mesh信息，如.obj，.glft
	b. material信息，如diffuse roughconductor，roughdielectric
	c. camera信息
	d. light信息
2. 维护上述信息
3. 一些辅助函数，如删除冗余顶点
*/

#include <memory> 
#include <vector>
#include <unordered_set>

#include <common/Mesh/Mesh.h>
#include <nvutils/camera_manipulator.hpp>
#include <common/Mesh/nvvk/gltf_utils.hpp>
#include <common/Shader/shaderStructType.h>
#include <common/Instance/Instance.h>

namespace FzbRenderer {

class Scene {
public:
	Scene() = default;
	~Scene() = default;

	void createSceneFromXML();
	void createSceneInfoBuffer();
	void clean();

	void preRender();
	void UIRender();
	void updateDataPerFrame(VkCommandBuffer cmd);

	std::filesystem::path scenePath;
	std::shared_ptr<nvutils::CameraManipulator> cameraManip{ std::make_shared<nvutils::CameraManipulator>() };
	
	std::vector<FzbRenderer::MeshSet> meshSets;
	std::vector<std::vector<Instance>> instanceInfos;	//static period random
	bool hasDynamicLight = false;
	std::vector<LightInstance> lightInstances;
	//---------------------------------------GPU使用数据---------------------------------------------------
	std::vector<nvvk::Image>     textures{};

	std::vector<shaderio::Mesh> meshes;
	std::vector<shaderio::Instance> instances;
	std::vector<shaderio::Instance> dynamicInstances;
	std::vector<shaderio::BSDFMaterial> materials;
	shaderio::SceneInfo sceneInfo;

	std::vector<nvvk::Buffer> bDatas;	//每个gltf的二进制数据，包含索引和顶点数据
	nvvk::Buffer bMeshes;
	nvvk::Buffer bInstances;
	nvvk::Buffer bMaterials;
	nvvk::Buffer bSceneInfo;
	//-----------------------------------------------------------------------------------------------------
	int loadTexture(const std::filesystem::path& texturePath);
	void addMeshSet(MeshSet& meshSet);

	int getMeshSetIndex(std::string meshSetID) { return meshSetIDToIndex[meshSetID]; };
	int getMaterialIndex(std::string materialID) { return uniqueMaterialIDToIndex[materialID]; };
	int getTextureIndex(std::filesystem::path texturePath) { return texturePathToIndex[texturePath]; }
	int getMeshBufferIndex(uint32_t meshIndex) { return meshToBufferIndex[meshIndex]; };
	int getMeshSetIndex(uint32_t meshIndex) { return meshIndexToMeshSetIndex[meshIndex]; };

	MeshInfo getMeshInfo(uint32_t meshIndex);

	//映射
	std::unordered_map<std::string, uint32_t> uniqueMaterialIDToIndex;
	std::unordered_map<std::filesystem::path, int> texturePathToIndex;
	std::map<std::string, uint32_t> meshSetIDToIndex;	//根据meshSetID获取meshSet数组的索引
	std::vector<uint32_t> meshToBufferIndex;	//meshToBufferIndex[meshIndex] = bufferIndex，前向或延时渲染时按mesh渲染时使用
	std::vector<uint32_t> meshIndexToMeshSetIndex;
	std::map<std::string, std::pair<uint32_t, uint32_t>> instanceIDToInstance;
};

}