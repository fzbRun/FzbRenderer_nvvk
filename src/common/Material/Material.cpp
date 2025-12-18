#include "Material.h"
#include "Diffuse/DiffuseMaterial.h"
#include "Conductor/ConductorMaterial.h"
#include "Dielectric/DielectricMaterial.h"
#include "RoughConductor/RoughConductorMaterial.h"
#include "RoughDielectric/RoughDielectricMaterial.h"
#include <common/Scene/Scene.h>
#include <common/Application/Application.h>

const shaderio::BSDFMaterial FzbRenderer::defaultMaterial{
	.albedo = glm::vec4(1.0f),
	.emissive = glm::vec3(0.0f),
	.eta = glm::vec3(1.0f / 1.5f),		//ext_ior(air) / int_ior(glass)
	.roughness = 0.1f,
	.materialMapIndex = { -1, -1, -1 },
};

shaderio::BSDFMaterial  FzbRenderer::getMaterialInfoFromSceneInfoXML(pugi::xml_node& bsdfNode) {
	shaderio::BSDFMaterial material = defaultMaterial;

	std::string materialType = bsdfNode.attribute("type").value();
	if (materialType == "diffuse") FzbRenderer::DiffuseMaterial::getMaterialInfoFromSceneInfoXML(bsdfNode, material);
	else if (materialType == "conductor") FzbRenderer::ConductorMaterial::getMaterialInfoFromSceneInfoXML(bsdfNode, material);
	else if (materialType == "dielectric") FzbRenderer::DielectricMaterial::getMaterialInfoFromSceneInfoXML(bsdfNode, material);
	else if (materialType == "roughConductor") FzbRenderer::RoughConductorMaterial::getMaterialInfoFromSceneInfoXML(bsdfNode, material);
	else if (materialType == "roughDielectric") FzbRenderer::RoughDielectricMaterial::getMaterialInfoFromSceneInfoXML(bsdfNode, material);

	addTexture(bsdfNode, material);
	return material;
}
int addTextureToScene(pugi::xml_node& mapPathNode) {
	FzbRenderer::Scene& scene = FzbRenderer::Application::sceneResource;

	std::string texturePathStr = mapPathNode.attribute("value").value();
	std::filesystem::path texturePath = scene.scenePath / texturePathStr;
	int textureIndex = scene.loadTexture(texturePath);
	return textureIndex;
}
void FzbRenderer::addTexture(pugi::xml_node& bsdfNode, shaderio::BSDFMaterial& material) {
	if (pugi::xml_node normalMapNode = bsdfNode.child("normalMap")) {
		std::string mapType = normalMapNode.attribute("type").value();
		if (mapType == "texture") {
			if (pugi::xml_node mapPathNode = normalMapNode.child("filename"))
				material.materialMapIndex.x = addTextureToScene(mapPathNode);
			else LOGW("normalMap类型为texture，但是没有给出地址");
		}
	}
	if (pugi::xml_node albedoMapNode = bsdfNode.child("albedoMap")) {
		std::string mapType = albedoMapNode.attribute("type").value();
		if (mapType == "texture") {
			if (pugi::xml_node mapPathNode = albedoMapNode.child("filename"))
				material.materialMapIndex.y = addTextureToScene(mapPathNode);
			else LOGW("albedoMap类型为texture，但是没有给出地址");
		}
		else if (mapType == "checkerboard") material.materialMapIndex.y = shaderio::AlbedoMapType::Checkerboard;
	}
	if (pugi::xml_node bsdfParamMapNode = bsdfNode.child("bsdfParamMap")){
		std::string mapType = bsdfParamMapNode.attribute("type").value();
		if (mapType == "texture") {
			if (pugi::xml_node mapPathNode = bsdfParamMapNode.child("filename"))
				material.materialMapIndex.z = addTextureToScene(mapPathNode);
			else LOGW("albedoMap类型为texture，但是没有给出地址");
		}
	}
}