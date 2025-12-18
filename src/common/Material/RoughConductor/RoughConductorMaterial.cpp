#include <common/utils.hpp>
#include "./RoughConductorMaterial.h"
#include "common/Application/Application.h"

void FzbRenderer::RoughConductorMaterial::getMaterialInfoFromSceneInfoXML(
	pugi::xml_node& bsdfNode,
	shaderio::BSDFMaterial& material) {
	material.type = shaderio::RoughConductor;
	if (pugi::xml_node roughnessNode = bsdfNode.child("roughness"))
		material.roughness = std::stof(roughnessNode.attribute("value").value());

	pugi::xml_node etaNode = bsdfNode.child("eta");
	pugi::xml_node kNode = bsdfNode.child("k");
	if (etaNode && kNode) {
		material.eta = FzbRenderer::getRGBFromString(etaNode.attribute("value").value());
		glm::vec3 k = FzbRenderer::getRGBFromString(kNode.attribute("value").value());
		material.albedo = ((material.eta - 1.0f) * (material.eta - 1.0f) + k * k) / ((material.eta + 1.0f) * (material.eta + 1.0f) + k * k);
	}else if(pugi::xml_node albedoNode = bsdfNode.child("albedo")) {
		material.albedo = FzbRenderer::getRGBAFromString(albedoNode.attribute("value").value());
	}
	
	if (pugi::xml_node emissiveNode = bsdfNode.child("emissive"))
		material.emissive = FzbRenderer::getRGBFromString(emissiveNode.attribute("value").value());
}