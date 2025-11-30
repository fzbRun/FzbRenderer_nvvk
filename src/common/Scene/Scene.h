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

#include <nvutils/camera_manipulator.hpp>
#include <common/Scene/gltf_utils.hpp>


namespace FzbRenderer {

class Scene {
public:
	void createSceneFromXML();
	void createSceneInfBuffer();
	void clean();

	std::filesystem::path scenePath;
	std::shared_ptr<nvutils::CameraManipulator> cameraManip{ std::make_shared<nvutils::CameraManipulator>() };
	std::vector<nvvk::Image>     textures{};

	std::vector<shaderio::GltfMesh> meshes;
	std::vector<shaderio::GltfInstance> instances;
	std::vector<shaderio::GltfMetallicRoughness> materials;
	shaderio::GltfSceneInfo sceneInfo;

	std::vector<nvvk::Buffer> bGltfDatas;	//每个gltf的二进制数据，包含索引和顶点数据
	nvvk::Buffer bMeshes;
	nvvk::Buffer bInstances;
	nvvk::Buffer bMaterials;
	nvvk::Buffer bSceneInfo;

	std::vector<uint32_t> meshToBufferIndex;	//meshToBufferIndex[meshIndex] = bufferIndex
private:
	void loadGltfData(const tinygltf::Model& mode, bool importInstance = false);
};

}