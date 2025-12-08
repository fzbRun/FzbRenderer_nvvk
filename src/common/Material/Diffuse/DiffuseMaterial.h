#pragma once
#include "common/Material/Material.h"

#ifndef FZBRENDERER_DIFFUSEMATERIAL_H
#define FZBRENDERER_DIFFUSEMATERIAL_H

namespace FzbRenderer {
class DiffuseMaterial {
public:
	static void getMaterialInfoFromSceneInfoXML(pugi::xml_node& bsdfNode,
		shaderio::BSDFMaterial& material,
		std::unordered_set<std::string>& uniqueTexturePaths);
};
}

#endif