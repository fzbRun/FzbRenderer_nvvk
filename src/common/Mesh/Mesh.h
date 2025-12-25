#pragma once
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "common/Shader/shaderStructType.h"
#include <filesystem>
#include <tinygltf/tiny_gltf.h>
#include <nvutils/primitives.hpp>
#include <nvvk/buffer_suballocator.hpp>


#ifndef FZBRENDERER_MESH_H
#define FZBRENDERER_MESH_H

namespace FzbRenderer {

struct MeshInfo {
	std::string meshID;
	shaderio::Mesh mesh;
	std::string materialID;		//mtl或gltf中的materialID
	shaderio::BSDFMaterial material;	//mtl或gltf中的material
	shaderio::AABB aabb = { { FLT_MAX, FLT_MAX, FLT_MAX }, { -FLT_MAX, -FLT_MAX, -FLT_MAX } };

	uint32_t meshIndex;

	shaderio::AABB getAABB(glm::mat4 transformMatrix = glm::mat4(1.0f));
};

class MeshSet{
public:
	MeshSet() = default;
	MeshSet(std::string meshID, std::string meshType, std::filesystem::path meshPath);
	MeshSet(std::string meshID, nvutils::PrimitiveMesh primitiveMesh);

	nvvk::Buffer createMeshDataBuffer();
	shaderio::AABB getAABB(glm::mat4 transformMatrix = glm::mat4(1.0f));

	static nvutils::PrimitiveMesh createPlane(int steps, float width, float height);
	static nvutils::PrimitiveMesh createCube(bool normal = false, bool texCoords = false, float width = 1.0F, float height = 1.0F, float depth = 1.0F);

	std::string meshID;
	uint32_t meshOffset;
	std::vector<MeshInfo> childMeshInfos;		//当前mesh中的小mesh
	std::vector<uint8_t> meshByteData;
	shaderio::AABB aabb = { { FLT_MAX, FLT_MAX, FLT_MAX }, { -FLT_MAX, -FLT_MAX, -FLT_MAX } };
private:
	void loadGltfData(const tinygltf::Model& model, bool importInstance = false);
	void processMesh(aiMesh* meshData, const aiScene* sceneData);
	void processNode(aiNode* node, const aiScene* sceneData);
	void loadObjData(std::filesystem::path meshPath);
};
}

#endif