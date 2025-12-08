#pragma once

#include "common/Shader/shaderStructType.h"
#include <unordered_set>
#include <pugixml.hpp>

#ifndef FZBRENDERER_MATERIAL_H
#define FZBRENDERER_MATERIAL_H

namespace FzbRenderer {
	extern const shaderio::BSDFMaterial defaultMaterial;

	shaderio::BSDFMaterial getMaterialInfoFromSceneInfoXML(
		pugi::xml_node& bsdfNode,
		std::unordered_set<std::string>& uniqueTexturePaths);
}

#endif