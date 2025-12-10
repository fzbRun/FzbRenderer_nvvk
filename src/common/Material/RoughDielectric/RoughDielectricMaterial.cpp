#include <common/utils.hpp>
#include "./RoughDielectricMaterial.h"
#include "common/Application/Application.h"

void FzbRenderer::RoughDielectricMaterial::getMaterialInfoFromSceneInfoXML(
	pugi::xml_node& bsdfNode,
	shaderio::BSDFMaterial& material,
	std::unordered_set<std::string>& uniqueTexturePaths) {
	material.type = shaderio::RoughDeielectric;
	if (pugi::xml_node roughnessNode = bsdfNode.child("roughness"))
		material.roughness = std::stof(roughnessNode.attribute("value").value());

	if (pugi::xml_node etaNode = bsdfNode.child("eta"))
		material.eta = FzbRenderer::getRGBFromString(etaNode.attribute("value").value());
	else {
		pugi::xml_node intIorNode = bsdfNode.child("int_ior");
		pugi::xml_node extIorNode = bsdfNode.child("ext_ior");
		if (intIorNode && extIorNode) {
			float intIor = std::stof(intIorNode.attribute("value").value());
			float extIor = std::stof(extIorNode.attribute("value").value());

			if (intIor == extIor) LOGW("不允许两边折射率相同");
			if (intIor == 0 || extIor == 0) LOGW("不允许折射率为0");

			material.eta = glm::vec3(extIor / intIor);
		}		
	}
	material.albedo = glm::vec3((material.eta - 1.0f) * (material.eta - 1.0f)) / ((material.eta + 1.0f) * (material.eta + 1.0f));

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