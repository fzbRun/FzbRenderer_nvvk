#pragma once
#include <cstdint>
#include <glm/glm.hpp>
#include "pugixml.hpp"
#include <common/Shader/shaderStructType.h>
#include <common/Mesh/Mesh.h>

#ifndef FZBRENDERER_INSTANCE_H
#define FZBRENDERER_INSTANCE_H

namespace FzbRenderer {
enum InstanceType {
	Static,
	PeriodMotion,
	RandomMotion
};
InstanceType getTypeFromString(std::string typeStr);

class Instance {
public:
	std::string instanceID = "defaultInstanceID";
	InstanceType type = Static;
	glm::mat4 baseMatrix = glm::mat4(1.0f);
	uint32_t meshIndex;
	uint32_t materialIndex;

	uint32_t time = 100;
	glm::mat4 startMatrix = glm::mat4(1.0f);
	glm::mat4 endMatrix = glm::mat4(1.0f);

	bool useCustomMeshSet = false;
	MeshSet customMeshSet;

	std::vector<shaderio::Instance> childInstances;

	Instance() = default;
	Instance(pugi::xml_node& instanceNode);

	void getTransformMatrixFromXML(pugi::xml_node& transformNode);
	void getInstance(std::vector<shaderio::Instance>& instance, int offset, float time);
};

class LightInstance : public Instance{
public:
	LightInstance(pugi::xml_node& lightNode);
	void copyInstanceInfo(const Instance& instance);
	shaderio::Light getLight(float time);

	shaderio::Light light;
};
}

#endif