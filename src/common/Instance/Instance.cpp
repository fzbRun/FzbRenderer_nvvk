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

void InstanceSet::getTransformMatrixFromXML(pugi::xml_node& transformNode){
	if (pugi::xml_node translateNode = transformNode.select_node("translate").node()) {
		glm::vec3 translateValue = FzbRenderer::getRGBFromString(translateNode.attribute("value").value());
		baseMatrix_translate = glm::translate(baseMatrix_translate, translateValue);
	}
	if (pugi::xml_node rotateNode = transformNode.child("rotate")) {
		glm::vec3 rotateAngle = glm::radians(FzbRenderer::getRGBFromString(rotateNode.attribute("value").value()));
		if (rotateAngle.x != 0.0f) baseMatrix_rotate = glm::rotate(baseMatrix_rotate, rotateAngle.x, glm::vec3(1, 0, 0));
		if (rotateAngle.y != 0.0f) baseMatrix_rotate = glm::rotate(baseMatrix_rotate, rotateAngle.y, glm::vec3(0, 1, 0));
		if (rotateAngle.z != 0.0f) baseMatrix_rotate = glm::rotate(baseMatrix_rotate, rotateAngle.z, glm::vec3(0, 0, 1));
	}
	if (pugi::xml_node scaleNode = transformNode.child("scale")) {
		glm::vec3 scaleValue = FzbRenderer::getRGBFromString(scaleNode.attribute("value").value());
		baseMatrix_scale = glm::scale(baseMatrix_scale, scaleValue);
	}
	baseMatrix = baseMatrix_translate * baseMatrix_rotate * baseMatrix_scale;
	if (pugi::xml_node matrixNode = transformNode.child("matrix"))
		baseMatrix = FzbRenderer::getMat4FromString(matrixNode.attribute("value").value());

	if (pugi::xml_node periodNode = transformNode.child("period")) {
		type = InstanceType::PeriodMotion;
		if (periodNode.attribute("time")) time = std::stof(periodNode.attribute("time").value());
		if (periodNode.attribute("speed")) speed = std::stof(periodNode.attribute("speed").value());
		if (pugi::xml_node translateNode = periodNode.child("translate")) {
			glm::vec3 translateValue = FzbRenderer::getRGBFromString(translateNode.attribute("value").value());
			translateMatrix = glm::translate(translateMatrix, translateValue);
		}
		if (pugi::xml_node rotateNode = periodNode.child("rotate")) {
			glm::vec3 rotateAngle = glm::radians(FzbRenderer::getRGBFromString(rotateNode.attribute("value").value()));
			if (rotateAngle.x > 0.01f) rotateMatrix = glm::rotate(rotateMatrix, rotateAngle.x, glm::vec3(1, 0, 0));
			if (rotateAngle.y > 0.01f) rotateMatrix = glm::rotate(rotateMatrix, rotateAngle.y, glm::vec3(0, 1, 0));
			if (rotateAngle.z > 0.01f) rotateMatrix = glm::rotate(rotateMatrix, rotateAngle.z, glm::vec3(0, 0, 1));
		}
		if (pugi::xml_node scaleNode = periodNode.child("scale")) {
			glm::vec3 scaleValue = FzbRenderer::getRGBFromString(scaleNode.attribute("value").value());
			scaleMatrix = glm::scale(scaleMatrix, scaleValue);
		}
	}

	transform = baseMatrix;
	transform_lastTime = transform;
}
InstanceSet::InstanceSet(pugi::xml_node& instanceNode) {
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
		else if(meshType == "cube") primitive = FzbRenderer::MeshSet::createCube(true, false);
		else if(meshType == "sphere") primitive = FzbRenderer::MeshSet::createSphere(true, false, 36, 18);

		meshSetID = "custom" + meshType + std::to_string(customMeshSetCount++);
		customMeshSet = FzbRenderer::MeshSet(meshSetID, primitive);
		scene.addMeshSet(customMeshSet);
		childMeshInfos = customMeshSet.childMeshInfos;

		meshSetIndex = scene.meshSets.size() - 1;
	}
	else {
		if (!scene.meshSetIDToIndex.count(meshSetID)) LOGW("实例没有对应的mesh：%s\n", meshSetID.c_str());

		meshSetIndex = scene.meshSetIDToIndex[meshSetID];

		FzbRenderer::MeshSet& meshSet = scene.meshSets[meshSetIndex];
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

void InstanceSet::getInstance(std::vector<shaderio::Instance>& instances, int offset, float time) {
	instanceStartIndex = offset;

	if (type != InstanceType::PeriodMotion) {
		memcpy(instances.data() + offset, childInstances.data(), sizeof(shaderio::Instance) * childInstances.size());
		return;
	}

	float phase = time * std::max(speed, 0.0f);
	time = 1.0f - abs(fmod(phase, 2.0f) - 1.0f);

	transform_lastTime = transform;
	if (isStatic == 0) {
		transform = ((1.0f - time) * baseMatrix_translate + time * translateMatrix * baseMatrix_translate) *
			((1.0f - time) * baseMatrix_rotate + time * rotateMatrix * baseMatrix_rotate) *
			((1.0f - time) * baseMatrix_scale + time * scaleMatrix * baseMatrix_scale);
	}

	for (int i = 0; i < childInstances.size(); ++i) {
		shaderio::Instance instance;
		instance.meshIndex = childInstances[i].meshIndex;
		instance.materialIndex = childInstances[i].materialIndex;
		instance.transform = transform;
		instances[offset + i] = instance;
	}
}

void LightInstance::copyInstanceInfo(const InstanceSet& instance) {
	this->instanceID = instance.instanceID;
	this->type = instance.type;
	this->baseMatrix = instance.baseMatrix;
	this->baseMatrix_translate = instance.baseMatrix_translate;
	this->baseMatrix_rotate = instance.baseMatrix_rotate;
	this->baseMatrix_scale = instance.baseMatrix_scale;
	this->time = instance.time;
	this->translateMatrix = instance.translateMatrix;
	this->rotateMatrix = instance.rotateMatrix;
	this->scaleMatrix = instance.scaleMatrix;
	this->useCustomMeshSet = instance.useCustomMeshSet;
	//this->customMeshSet = instance.customMeshSet;
}
LightInstance::LightInstance(pugi::xml_node& lightNode) {
	Scene& scene = Application::sceneResource;

	if (pugi::xml_node instanceRefNode = lightNode.child("instanceRef")) {
		std::string instanceID = instanceRefNode.attribute("id").value();
		if (scene.instanceIDToInstanceSet.count(instanceID)) {
			std::pair<uint32_t, uint32_t> instanceTypeAndIndex = scene.instanceIDToInstanceSet[instanceID];
			type = (InstanceType)instanceTypeAndIndex.first;
			InstanceSet instanceSet = scene.getInstanceSet(type, instanceTypeAndIndex.second);
			copyInstanceInfo(instanceSet);
		}
		else printf("光源没有相应的instanceID：%s\n", instanceID);
	}
	else if (pugi::xml_node transformNode = lightNode.child("transform"));
}
shaderio::Light LightInstance::getLight(float time) {
	if (type != InstanceType::PeriodMotion) return light;

	float phase = time * std::max(speed, 0.0f);
	time = 1.0f - abs(fmod(phase, 2.0f) - 1.0f);

	shaderio::Light light_transform = light;
	glm::mat4 transformMatrix = ((1.0f - time) * glm::mat4(1.0f) + time * translateMatrix) *
		((1.0f - time) * glm::mat4(1.0f) + time * rotateMatrix) *
		((1.0f - time) * glm::mat4(1.0f) + time * scaleMatrix);
	if (light_transform.type == shaderio::LightType::Direction) light_transform.direction = transformMatrix * glm::vec4(light.direction, 1.0f);
	else {
		light_transform.pos = transformMatrix * glm::vec4(light.pos, 1.0f);
		light_transform.edge1 = glm::mat3(transformMatrix) * light.edge1;
		light_transform.edge2 = glm::mat3(transformMatrix) * light.edge2;
		light_transform.direction = glm::normalize(glm::cross(light_transform.edge1, light_transform.edge2));
	}

	return light_transform;
}