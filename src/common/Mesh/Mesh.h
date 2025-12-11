#pragma once
#include "common/Shader/shaderStructType.h"
#include <filesystem>
#include <tinygltf/tiny_gltf.h>

#ifndef FZBRENDERER_MESH_H
#define FZBRENDERER_MESH_H

namespace FzbRenderer {
class Mesh{
public:
	Mesh() = default;
	Mesh(std::string meshType, std::filesystem::path meshPath);

	shaderio::Mesh meshInfo;
private:
	void loadGltfData(const tinygltf::Model& model, bool importInstance = false);
	void loadObjData();
};
}

#endif