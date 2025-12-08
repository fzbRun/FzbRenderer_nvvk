#pragma once
#include "common/Material/Material.h"

#ifndef FZBRENDERER_CONDUCTOR_MATERIAL_H
#define FZBRENDERER_CONDUCTOR_MATERIAL_H

namespace FzbRenderer {
	class ConductorMaterial {
	public:
		static void getMaterialInfoFromSceneInfoXML(pugi::xml_node& bsdfNode,
			shaderio::BSDFMaterial& material,
			std::unordered_set<std::string>& uniqueTexturePaths);
	};
}

#endif