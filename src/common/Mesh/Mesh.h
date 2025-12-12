#pragma once
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "common/Shader/shaderStructType.h"
#include <filesystem>
#include <tinygltf/tiny_gltf.h>


#ifndef FZBRENDERER_MESH_H
#define FZBRENDERER_MESH_H

namespace FzbRenderer {
class Mesh{
public:
	static void loadData(std::string meshType, std::filesystem::path meshPath);

private:
	static void loadGltfData(const tinygltf::Model& model, bool importInstance = false);
	static void processMesh(aiMesh* meshData, const aiScene* sceneData);
	static void processNode(aiNode* node, const aiScene* sceneData);
	static void loadObjData(std::filesystem::path meshPath);
};
}

#endif