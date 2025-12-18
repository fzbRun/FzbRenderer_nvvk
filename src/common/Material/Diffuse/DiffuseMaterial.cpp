#include <common/utils.hpp>
#include "./DiffuseMaterial.h"
#include "common/Application/Application.h"

void FzbRenderer::DiffuseMaterial::getMaterialInfoFromSceneInfoXML(
	pugi::xml_node& bsdfNode,
	shaderio::BSDFMaterial& material
) {
	material.type = shaderio::Diffuse;	//diffuse
	if (pugi::xml_node albedoNode = bsdfNode.child("albedo"))
		material.albedo = FzbRenderer::getRGBAFromString(albedoNode.attribute("value").value());
	if (pugi::xml_node emissiveNode = bsdfNode.child("emissive"))
		material.emissive = FzbRenderer::getRGBFromString(emissiveNode.attribute("value").value());
}