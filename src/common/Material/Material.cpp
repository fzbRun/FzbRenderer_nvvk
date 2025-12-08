#include "Material.h"
#include "Diffuse/DiffuseMaterial.h"
#include "Conductor/ConductorMaterial.h"
#include "Dielectric/DielectricMaterial.h"

const shaderio::BSDFMaterial FzbRenderer::defaultMaterial{
	.albedo = glm::vec4(1.0f),
	.emissive = glm::vec3(0.0f),
	.eta = glm::vec3(1.0f / 1.5f),		//ext_ior(air) / int_ior(glass)
	.roughness = 0.1f,
	.materialMapIndex = { -1, -1, -1 },
};

shaderio::BSDFMaterial  FzbRenderer::getMaterialInfoFromSceneInfoXML(
	pugi::xml_node& bsdfNode,
	std::unordered_set<std::string>& uniqueTexturePaths
) {
	shaderio::BSDFMaterial material = defaultMaterial;

	std::string materialType = bsdfNode.attribute("type").value();
	if (materialType == "diffuse") FzbRenderer::DiffuseMaterial::getMaterialInfoFromSceneInfoXML(bsdfNode, material, uniqueTexturePaths);
	else if (materialType == "conductor") FzbRenderer::ConductorMaterial::getMaterialInfoFromSceneInfoXML(bsdfNode, material, uniqueTexturePaths);
	else if (materialType == "dielectric") FzbRenderer::DielectricMaterial::getMaterialInfoFromSceneInfoXML(bsdfNode, material, uniqueTexturePaths);

	return material;
}