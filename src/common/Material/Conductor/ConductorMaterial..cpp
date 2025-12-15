#include <common/utils.hpp>
#include "./ConductorMaterial.h"
#include "common/Application/Application.h"
#include <nvshaders/slang_types.h>

void FzbRenderer::ConductorMaterial::getMaterialInfoFromSceneInfoXML(
	pugi::xml_node& bsdfNode,
	shaderio::BSDFMaterial& material,
	std::unordered_set<std::string>& uniqueTexturePaths) {
	material.type = shaderio::Conductor;
	
	pugi::xml_node etaNode = bsdfNode.child("eta");
	pugi::xml_node kNode = bsdfNode.child("k");
	if (etaNode && kNode) {
		material.eta = FzbRenderer::getRGBFromString(etaNode.attribute("value").value());
		glm::vec3 k = FzbRenderer::getRGBFromString(kNode.attribute("value").value());
		material.albedo = ((material.eta - 1.0f) * (material.eta - 1.0f) + k * k) / ((material.eta + 1.0f) * (material.eta + 1.0f) + k * k);
	}
	else if (pugi::xml_node albedoNode = bsdfNode.child("albedo")) {
		material.albedo = FzbRenderer::getRGBAFromString(albedoNode.attribute("value").value());
	}

	if (pugi::xml_node emissiveNode = bsdfNode.child("emissive"))
		material.emissive = FzbRenderer::getRGBFromString(emissiveNode.attribute("value").value());

	FzbRenderer::Scene& scene = FzbRenderer::Application::sceneResource;
	for (pugi::xml_node mapNode : bsdfNode.children("texture")) {
		std::string mapType = mapNode.attribute("type").value();
		//这里可以用一个enum
		std::string texturePathStr = mapNode.attribute("value").value();
		if (uniqueTexturePaths.count(texturePathStr) == 0) continue;

		std::filesystem::path texturePath = scene.scenePath / "textures" / texturePathStr;
		int textureIndex = scene.textures.size();
		scene.loadTexture(texturePath);
		uniqueTexturePaths.insert(texturePathStr);

		if (mapType == "albedo") material.materialMapIndex.x = textureIndex;
		else if (mapType == "normal") material.materialMapIndex.y = textureIndex;
		else if (mapType == "bsdfParams") material.materialMapIndex.z = textureIndex;
	}
}