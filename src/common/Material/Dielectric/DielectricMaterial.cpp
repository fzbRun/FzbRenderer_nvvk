#include <common/utils.hpp>
#include "./DielectricMaterial.h"
#include "common/Application/Application.h"
#include <nvshaders/slang_types.h>

void FzbRenderer::DielectricMaterial::getMaterialInfoFromSceneInfoXML(
	pugi::xml_node& bsdfNode,
	shaderio::BSDFMaterial& material) {
	material.type = shaderio::Dielectric;

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
}