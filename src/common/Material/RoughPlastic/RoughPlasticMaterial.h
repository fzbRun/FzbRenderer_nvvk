#pragma once
#include "common/Material/Material.h"

#ifndef FZBRENDERER_ROUGHPLASTIC_MATERIAL_H
#define FZBRENDERER_ROUGHPLASTIC_MATERIAL_H

namespace FzbRenderer {
class RoughPlasticMaterial {
public:
	static void getMaterialInfoFromSceneInfoXML(pugi::xml_node& bsdfNode, shaderio::BSDFMaterial& material);
};
}

#endif