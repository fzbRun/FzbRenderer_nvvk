#include "./Instance.h"
#include <common/Mesh/Mesh.h>
#include <common/Scene/Scene.h>
#include <common/Application/Application.h>
#include <common/utils.hpp>
#include <glm/gtx/matrix_decompose.hpp>

using namespace FzbRenderer;

InstanceType FzbRenderer::getTypeFromString(std::string typeStr) {
	if (typeStr == "static") return InstanceType::Static;
	if (typeStr == "periodMotion") return InstanceType::PeriodMotion;
	if (typeStr == "randomMotion") return InstanceType::RandomMotion;
	return InstanceType::Static;
}
glm::mat4 interpolateTransforms(const glm::mat4& a, const glm::mat4& b, float t) {
	// 确保 t 在 [0, 1] 范围内
	t = glm::clamp(t, 0.0f, 1.0f);

	// 分解矩阵A
	glm::vec3 scaleA, translationA, skewA;
	glm::vec4 perspectiveA;
	glm::quat rotationA;
	glm::decompose(a, scaleA, rotationA, translationA, skewA, perspectiveA);

	// 分解矩阵B
	glm::vec3 scaleB, translationB, skewB;
	glm::vec4 perspectiveB;
	glm::quat rotationB;
	glm::decompose(b, scaleB, rotationB, translationB, skewB, perspectiveB);

	// 对各个分量进行插值
	glm::vec3 scale = glm::mix(scaleA, scaleB, t);
	glm::quat rotation = glm::slerp(rotationA, rotationB, t);  // 球面线性插值
	glm::vec3 translation = glm::mix(translationA, translationB, t);

	// 重新组合矩阵
	glm::mat4 result = glm::translate(glm::mat4(1.0f), translation)
		* glm::mat4_cast(rotation)
		* glm::scale(glm::mat4(1.0f), scale);

	return result;
}

void Instance::getTransformMatrixFromXML(pugi::xml_node& transformNode){
	if (pugi::xml_node matrixNode = transformNode.child("matrix"))
		baseMatrix = FzbRenderer::getMat4FromString(matrixNode.attribute("value").value());
	if (pugi::xml_node translateNode = transformNode.select_node("translate").node()) {
		glm::vec3 translateValue = FzbRenderer::getRGBFromString(translateNode.attribute("value").value());
		baseMatrix = glm::translate(baseMatrix, translateValue);
	}
	if (pugi::xml_node rotateNode = transformNode.child("rotate")) {
		glm::vec3 rotateAngle = glm::radians(FzbRenderer::getRGBFromString(rotateNode.attribute("value").value()));
		if (rotateAngle.x > 0.01f) baseMatrix = glm::rotate(baseMatrix, rotateAngle.x, glm::vec3(1, 0, 0));
		if (rotateAngle.y > 0.01f) baseMatrix = glm::rotate(baseMatrix, rotateAngle.y, glm::vec3(0, 1, 0));
		if (rotateAngle.z > 0.01f) baseMatrix = glm::rotate(baseMatrix, rotateAngle.z, glm::vec3(0, 0, 1));
	}
	if (pugi::xml_node scaleNode = transformNode.child("scale")) {
		glm::vec3 scaleValue = FzbRenderer::getRGBFromString(scaleNode.attribute("value").value());
		baseMatrix = glm::scale(baseMatrix, scaleValue);
	}

	if(type == InstanceType::PeriodMotion) {
		if (pugi::xml_node periodNode = transformNode.child("period")) {
			if (periodNode.attribute("speed")) time = std::stof(periodNode.attribute("time").value());
			if (pugi::xml_node translateNode = periodNode.child("translate")) {
				if (pugi::xml_node translateStartNode = translateNode.child("start")) {
					glm::vec3 translateValue = FzbRenderer::getRGBFromString(translateStartNode.attribute("value").value());
					startMatrix = glm::translate(startMatrix, translateValue);
				}
				if (pugi::xml_node translateEndNode = translateNode.child("end")) {
					glm::vec3 translateValue = FzbRenderer::getRGBFromString(translateEndNode.attribute("value").value());
					endMatrix = glm::translate(endMatrix, translateValue);
				}
			}
			if (pugi::xml_node rotateNode = periodNode.child("rotate")) {
				if (pugi::xml_node rotateStartNode = rotateNode.child("start")) {
					glm::vec3 rotateAngle = glm::radians(FzbRenderer::getRGBFromString(rotateStartNode.attribute("value").value()));
					if (rotateAngle.x > 0.01f) startMatrix = glm::rotate(startMatrix, rotateAngle.x, glm::vec3(1, 0, 0));
					if (rotateAngle.y > 0.01f) startMatrix = glm::rotate(startMatrix, rotateAngle.y, glm::vec3(0, 1, 0));
					if (rotateAngle.z > 0.01f) startMatrix = glm::rotate(startMatrix, rotateAngle.z, glm::vec3(0, 0, 1));
				}
				if (pugi::xml_node rotateEndNode = rotateNode.child("end")) {
					glm::vec3 rotateAngle = glm::radians(FzbRenderer::getRGBFromString(rotateEndNode.attribute("value").value()));
					if (rotateAngle.x > 0.01f) endMatrix = glm::rotate(endMatrix, rotateAngle.x, glm::vec3(1, 0, 0));
					if (rotateAngle.y > 0.01f) endMatrix = glm::rotate(endMatrix, rotateAngle.y, glm::vec3(0, 1, 0));
					if (rotateAngle.z > 0.01f) endMatrix = glm::rotate(endMatrix, rotateAngle.z, glm::vec3(0, 0, 1));
				}
			}
			if (pugi::xml_node scaleNode = periodNode.child("scale")) {
				if (pugi::xml_node scaleStartNode = scaleNode.child("start")) {
					glm::vec3 scaleValue = FzbRenderer::getRGBFromString(scaleStartNode.attribute("value").value());
					startMatrix = glm::translate(startMatrix, scaleValue);
				}
				if (pugi::xml_node scaleEndNode = scaleNode.child("end")) {
					glm::vec3 scaleValue = FzbRenderer::getRGBFromString(scaleEndNode.attribute("value").value());
					endMatrix = glm::translate(endMatrix, scaleValue);
				}
			}
		}
	}
}
Instance::Instance(pugi::xml_node& instanceNode) {
	static int customMeshSetCount = 0;

	Scene& scene = Application::sceneResource;

	if (instanceNode.attribute("type")) type = getTypeFromString(instanceNode.attribute("type").value());
	if (instanceNode.attribute("id")) instanceID = instanceNode.attribute("id").value();

	std::string meshSetID = instanceNode.child("meshRef").attribute("id").value();
	std::vector<MeshInfo> childMeshInfos;
	if (meshSetID == "custom") {
		useCustomMeshSet = true;

		std::string meshType = instanceNode.child("meshRef").attribute("type").value();
		nvutils::PrimitiveMesh primitive;
		if (meshType == "plane") primitive = FzbRenderer::MeshSet::createPlane(1, 1.0f, 1.0f);
		meshSetID = "custom" + meshType + std::to_string(customMeshSetCount++);
		customMeshSet = FzbRenderer::MeshSet(meshSetID, primitive);
		scene.addMeshSet(customMeshSet);
		childMeshInfos = customMeshSet.childMeshInfos;
	}
	else {
		if (!scene.meshSetIDToIndex.count(meshSetID)) LOGW("实例没有对应的mesh：%s\n", meshSetID.c_str());
		FzbRenderer::MeshSet& meshSet = scene.meshSets[scene.meshSetIDToIndex[meshSetID]];
		childMeshInfos = meshSet.childMeshInfos;
	}
	childInstances.resize(childMeshInfos.size());
	for (int i = 0; i < childMeshInfos.size(); i++) {
		MeshInfo& childMesh = childMeshInfos[i];
		shaderio::Instance& instance = childInstances[i];
		instance.meshIndex = childMesh.meshIndex;

		std::string materialID = "defaultMaterial";
		if (pugi::xml_node materialNode = instanceNode.child("materialRef")) materialID = materialNode.attribute("id").value();
		else materialID = childMesh.materialID;
		if (!scene.uniqueMaterialIDToIndex.count(materialID)) materialID = "defaultMaterial";
		instance.materialIndex = scene.uniqueMaterialIDToIndex[materialID];

		//自实例没有单独的transform，只有统一的transform，这样方便一点；如果想要单独的transform，应该单独拿出来
		if (pugi::xml_node transformNode = instanceNode.child("transform")) getTransformMatrixFromXML(transformNode);
		instance.transform = baseMatrix;
	}
}

void Instance::getInstance(std::vector<shaderio::Instance>& instances, int offset, float time) {
	if (type != InstanceType::PeriodMotion) {
		memcpy(instances.data() + offset, childInstances.data(), sizeof(shaderio::Instance) * childInstances.size());
		return;
	}
	for (int i = 0; i < childInstances.size(); ++i) {
		shaderio::Instance instance;
		instance.meshIndex = meshIndex;
		instance.materialIndex = materialIndex;
		instance.transform = ((1.0f - time) * startMatrix + time * endMatrix) * baseMatrix;	//interpolateTransforms(startMatrix, endMatrix, time);
		instances[offset + i] = instance;
	}
}

void LightInstance::copyInstanceInfo(const Instance& instance) {
	this->instanceID = instance.instanceID;
	this->type = instance.type;
	this->baseMatrix = instance.baseMatrix;
	this->meshIndex = instance.meshIndex;
	this->materialIndex = instance.materialIndex;
	this->time = instance.time;
	this->startMatrix = instance.startMatrix;
	this->endMatrix = instance.endMatrix;
	this->useCustomMeshSet = instance.useCustomMeshSet;
	//this->customMeshSet = instance.customMeshSet;
}
LightInstance::LightInstance(pugi::xml_node& lightNode) {
	Scene& scene = Application::sceneResource;

	if (pugi::xml_node instanceRefNode = lightNode.child("instanceRef")) {
		std::string instanceID = instanceRefNode.attribute("id").value();
		if (scene.instanceIDToInstance.count(instanceID)) {
			std::pair<uint32_t, uint32_t> instanceTypeAndIndex = scene.instanceIDToInstance[instanceID];
			type = (InstanceType)instanceTypeAndIndex.first;
			Instance instance = scene.instanceInfos[type][instanceTypeAndIndex.second];
			copyInstanceInfo(instance);
		}
		else printf("光源没有相应的instanceID：%s\n", instanceID);
	}
	else if (pugi::xml_node transformNode = lightNode.child("transform"));
}
shaderio::Light LightInstance::getLight(float time) {
	if (type != InstanceType::PeriodMotion) return light;

	shaderio::Light light_transform = light;
	glm::mat4 transformMatrix = (1.0f - time)* startMatrix + time * endMatrix;
	light_transform.pos = transformMatrix * glm::vec4(light.pos, 1.0f);
	light_transform.edge1 = glm::mat3(transformMatrix) * light.edge1;
	light_transform.edge2 = glm::mat3(transformMatrix) * light.edge2;
	light_transform.direction = glm::normalize(glm::cross(light_transform.edge1, light_transform.edge2));

	return light_transform;
}