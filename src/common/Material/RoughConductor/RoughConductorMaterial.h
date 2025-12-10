#pragma once
#include "common/Material/Material.h"

#ifndef FZBRENDERER_ROUGHCONDUCTOR_MATERIAL_H
#define FZBRENDERER_ROUGHCONDUCTOR_MATERIAL_H

namespace FzbRenderer {
class RoughConductorMaterial {
public:
	static void getMaterialInfoFromSceneInfoXML(pugi::xml_node& bsdfNode,
			shaderio::BSDFMaterial& material,
			std::unordered_set<std::string>& uniqueTexturePaths);
};
}

#endif