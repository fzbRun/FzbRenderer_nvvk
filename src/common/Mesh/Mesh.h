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

struct ChildMesh {
	std::string meshID;
	shaderio::Mesh mesh;
	std::string materialID;
	shaderio::BSDFMaterial material;
};

class Mesh{
public:
	Mesh() = default;
	Mesh(std::string meshID, std::string meshType, std::filesystem::path meshPath);
	Mesh(std::string meshID, nvutils::PrimitiveMesh primitiveMesh);

	nvvk::Buffer createMeshDataBuffer();

	static nvutils::PrimitiveMesh createPlane(int steps, float width, float height);

	std::string meshID;
	std::vector<ChildMesh> childMeshes;		//当前mesh中的小mesh
	std::vector<uint8_t> meshByteData;
private:
	void loadGltfData(const tinygltf::Model& model, bool importInstance = false);
	void processMesh(aiMesh* meshData, const aiScene* sceneData);
	void processNode(aiNode* node, const aiScene* sceneData);
	void loadObjData(std::filesystem::path meshPath);
};
}

#endif