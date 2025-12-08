#include <common/utils.hpp>
#include "./DiffuseMaterial.h"
#include "common/Application/Application.h"

void FzbRenderer::DiffuseMaterial::getMaterialInfoFromSceneInfoXML(
	pugi::xml_node& bsdfNode,
	shaderio::BSDFMaterial& material,
	std::unordered_set<std::string>& uniqueTexturePaths) {
	material.type = shaderio::Diffuse;	//diffuse
	if (pugi::xml_node albedoNode = bsdfNode.child("albedo"))
		material.albedo = FzbRenderer::getRGBAFromString(albedoNode.attribute("value").value());
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
	}
}