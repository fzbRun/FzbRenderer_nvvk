#include "Scene.h"
#include "pugixml.hpp"
#include <common/path_utils.hpp>
#include <common/utils.hpp>
#include <common/Application/Application.h>
#include <unordered_map>
#include <nvgui/sky.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <common/Material/Material.h>
#include <glm/gtx/matrix_decompose.hpp>

int FzbRenderer::Scene::loadTexture(const std::filesystem::path& texturePath) {
	if (texturePathToIndex.count(texturePath)) return texturePathToIndex[texturePath];

	VkCommandBuffer       cmd = Application::app->createTempCmdBuffer();
	nvvk::Image texture = nvsamples::loadAndCreateImage(cmd, Application::stagingUploader, Application::app->getDevice(), texturePath);  // Load the image from the file and create a texture from it
	NVVK_DBG_NAME(texture.image);
	Application::app->submitAndWaitTempCmdBuffer(cmd);
	Application::samplerPool.acquireSampler(texture.descriptor.sampler);
	textures.emplace_back(texture);  // Store the texture in the vector of textures
	texturePathToIndex.insert({ texturePath, textures.size() - 1 });
	return textures.size() - 1;
}
void FzbRenderer::Scene::addMeshSet(MeshSet& meshSet) {
	nvvk::Buffer bData = meshSet.createMeshDataBuffer();
	uint32_t bufferIndex = static_cast<uint32_t>(bDatas.size());
	bDatas.push_back(bData);

	meshSet.meshOffset = meshes.size();
	for (int i = 0; i < meshSet.childMeshInfos.size(); i++)
	{
		MeshInfo& childMeshInfo = meshSet.childMeshInfos[i];
		childMeshInfo.meshIndex = meshes.size();
		childMeshInfo.mesh.dataBuffer = (uint8_t*)bData.address;

		meshIndexToMeshSetIndex.push_back(meshSets.size());
		meshes.emplace_back(childMeshInfo.mesh);
		
		meshToBufferIndex.push_back(bufferIndex);

		if (!uniqueMaterialIDToIndex.count(childMeshInfo.materialID)) {
			uniqueMaterialIDToIndex.insert({ childMeshInfo.materialID, materials.size() });
			materials.push_back(childMeshInfo.material);
		}
	}

	meshSetIDToIndex.insert({ meshSet.meshID, meshSets.size() });
	meshSets.push_back(meshSet);
}
glm::mat4 getTransformMatrixFromXML(pugi::xml_node& transformNode, 
	bool& staticInstance, FzbRenderer::DynamicInstanceInfo* instanceInfo = nullptr) {
	glm::mat4 transformMatrix = glm::mat4(1.0f);
	
	if (pugi::xml_node matrixNode = transformNode.child("matrix"))
		transformMatrix = FzbRenderer::getMat4FromString(matrixNode.attribute("value").value());
	if (pugi::xml_node translateNode = transformNode.select_node("translate").node()) {
		glm::vec3 translateValue = FzbRenderer::getRGBFromString(translateNode.attribute("value").value());
		transformMatrix = glm::translate(transformMatrix, translateValue);
	}
	if (pugi::xml_node rotateNode = transformNode.child("rotate")) {
		glm::vec3 rotateAngle = glm::radians(FzbRenderer::getRGBFromString(rotateNode.attribute("value").value()));
		if (rotateAngle.x > 0.01f) transformMatrix = glm::rotate(transformMatrix, rotateAngle.x, glm::vec3(1, 0, 0));
		if (rotateAngle.y > 0.01f) transformMatrix = glm::rotate(transformMatrix, rotateAngle.y, glm::vec3(0, 1, 0));
		if (rotateAngle.z > 0.01f) transformMatrix = glm::rotate(transformMatrix, rotateAngle.z, glm::vec3(0, 0, 1));
	}
	if (pugi::xml_node scaleNode = transformNode.child("scale")) {
		glm::vec3 scaleValue = FzbRenderer::getRGBFromString(scaleNode.attribute("value").value());
		transformMatrix = glm::scale(transformMatrix, scaleValue);
	}

	if (instanceInfo != nullptr) {
		if (pugi::xml_node dynamicNode = transformNode.child("dynamic")) {
			staticInstance = false;
			if (dynamicNode.attribute("speed")) instanceInfo->speed = std::stof(dynamicNode.attribute("speed").value());
			if (pugi::xml_node translateNode = dynamicNode.child("translate")) {
				if (pugi::xml_node translateStartNode = translateNode.child("start")) {
					glm::vec3 translateValue = FzbRenderer::getRGBFromString(translateStartNode.attribute("value").value());
					instanceInfo->startTransformMatrix = glm::translate(instanceInfo->startTransformMatrix, translateValue);
				}
				if (pugi::xml_node translateEndNode = translateNode.child("end")) {
					glm::vec3 translateValue = FzbRenderer::getRGBFromString(translateEndNode.attribute("value").value());
					instanceInfo->endTransformMatrix = glm::translate(instanceInfo->endTransformMatrix, translateValue);
				}
			}
			if (pugi::xml_node rotateNode = dynamicNode.child("rotate")) {
				if (pugi::xml_node rotateStartNode = rotateNode.child("start")) {
					glm::vec3 rotateAngle = glm::radians(FzbRenderer::getRGBFromString(rotateStartNode.attribute("value").value()));
					if (rotateAngle.x > 0.01f) instanceInfo->startTransformMatrix = glm::rotate(instanceInfo->startTransformMatrix, rotateAngle.x, glm::vec3(1, 0, 0));
					if (rotateAngle.y > 0.01f) instanceInfo->startTransformMatrix = glm::rotate(instanceInfo->startTransformMatrix, rotateAngle.y, glm::vec3(0, 1, 0));
					if (rotateAngle.z > 0.01f) instanceInfo->startTransformMatrix = glm::rotate(instanceInfo->startTransformMatrix, rotateAngle.z, glm::vec3(0, 0, 1));
				}
				if (pugi::xml_node rotateEndNode = rotateNode.child("end")) {
					glm::vec3 rotateAngle = glm::radians(FzbRenderer::getRGBFromString(rotateEndNode.attribute("value").value()));
					if (rotateAngle.x > 0.01f) instanceInfo->endTransformMatrix = glm::rotate(instanceInfo->endTransformMatrix, rotateAngle.x, glm::vec3(1, 0, 0));
					if (rotateAngle.y > 0.01f) instanceInfo->endTransformMatrix = glm::rotate(instanceInfo->endTransformMatrix, rotateAngle.y, glm::vec3(0, 1, 0));
					if (rotateAngle.z > 0.01f) instanceInfo->endTransformMatrix = glm::rotate(instanceInfo->endTransformMatrix, rotateAngle.z, glm::vec3(0, 0, 1));
				}
			}
			if (pugi::xml_node scaleNode = dynamicNode.child("scale")) {
				if (pugi::xml_node scaleStartNode = scaleNode.child("start")) {
					glm::vec3 scaleValue = FzbRenderer::getRGBFromString(scaleStartNode.attribute("value").value());
					instanceInfo->startTransformMatrix = glm::translate(instanceInfo->startTransformMatrix, scaleValue);
				}
				if (pugi::xml_node scaleEndNode = scaleNode.child("end")) {
					glm::vec3 scaleValue = FzbRenderer::getRGBFromString(scaleEndNode.attribute("value").value());
					instanceInfo->endTransformMatrix = glm::translate(instanceInfo->endTransformMatrix, scaleValue);
				}
			}
		}
	}
	return transformMatrix;
}
void FzbRenderer::Scene::createSceneFromXML() {
	scenePath = FzbRenderer::getProjectRootDir() / "resources" / scenePath;
	std::filesystem::path sceneInfoXMLPath = scenePath / "sceneInfo.xml";
	if(sceneInfoXMLPath.empty()) LOGW("\nsceneInfo路径为空\n");
	
	pugi::xml_document doc;
	auto result = doc.load_file(sceneInfoXMLPath.c_str());
	if (!result) LOGW("\nsceneInfoXML不存在\n");
	pugi::xml_node sceneInfoNode = doc.document_element();
	//----------------------------------------------相机参数-------------------------------------------------------------
	for (pugi::xml_node cameraNode : sceneInfoNode.children("sensor")) {
		//float fov = glm::radians(std::stof(cameraNode.select_node(".//float[@name='fov']").node().attribute("value").value()));
		//float aspect = (float)resolution.width / resolution.height;
		//fov = 2.0f * atanf(tanf(fov * 0.5f) / aspect);	//三叶草中给的fov是水平方向的，而glm中要的是垂直方向的
		float fov = std::stof(cameraNode.child("fov").attribute("value").value());
		cameraManip->setFov(fov);

		bool isPerspective = std::string(cameraNode.attribute("type").value()) == "perspective" ? true : false;
		pugi::xml_node tranformNode = cameraNode.child("transform");
		std::string transformNodeType = tranformNode.attribute("type").value();
		if (transformNodeType == "to_world") {
			pugi::xml_node matrixNode = tranformNode.child("matrix");
			glm::mat4 inverseViewMatrix = FzbRenderer::getMat4FromString(matrixNode.attribute("value").value());
			glm::mat4 viewMatrix = glm::inverse(inverseViewMatrix);
			cameraManip->setMatrix(viewMatrix, true);
		}
		else if (transformNodeType == "lookAt") {
			glm::vec3 eye = FzbRenderer::getRGBFromString(tranformNode.child("eye").attribute("value").value());
			glm::vec3 center = FzbRenderer::getRGBFromString(tranformNode.child("center").attribute("value").value());
			glm::vec3 up = FzbRenderer::getRGBFromString(tranformNode.child("up").attribute("value").value());
			cameraManip->setLookat(eye, center, up, true);
		}
		
		VkExtent2D resolution = Application::app->getWindowSize();
		cameraManip->setWindowSize(glm::uvec2(resolution.width, resolution.height));

		cameraManip->setClipPlanes(glm::vec2(0.1f, 100.0f));		
	}
	//------------------------------------------------材质---------------------------------------------------------------
	materials.resize(0);
	uniqueMaterialIDToIndex.clear();
	texturePathToIndex.clear();

	//默认材质
	shaderio::BSDFMaterial defaultMaterial = FzbRenderer::defaultMaterial;
	materials.push_back(defaultMaterial);
	uniqueMaterialIDToIndex.insert({ "defaultMaterial" , 0});

	pugi::xml_node bsdfsNode = sceneInfoNode.child("bsdfs");
	for (pugi::xml_node bsdfNode : bsdfsNode.children("bsdf")) {
		std::string materialID = bsdfNode.attribute("id").value();
		if (uniqueMaterialIDToIndex.count(materialID)) {
			printf("重复material读取: %s\n", materialID);
			continue;
		}
		shaderio::BSDFMaterial material = FzbRenderer::getMaterialInfoFromSceneInfoXML(bsdfNode);
		uniqueMaterialIDToIndex.insert({ materialID, materials.size() });
		materials.push_back(material);
	}
	//------------------------------------------------光源---------------------------------------------------------------
	if (pugi::xml_node lightsNode = sceneInfoNode.child("lights")) {
		sceneInfo.useSky = false;
		if (pugi::xml_node useSkyNode = lightsNode.child("useSky"))
			sceneInfo.useSky = std::string(useSkyNode.attribute("value").value()) == "true";
		sceneInfo.backgroundColor = glm::vec3(0.85f);
		if (pugi::xml_node backgroudColorNode = lightsNode.child("backgroundColor"))
			sceneInfo.backgroundColor = FzbRenderer::getRGBFromString(backgroudColorNode.attribute("value").value());

		sceneInfo.numLights = 0;
		for (pugi::xml_node lightNode : lightsNode.children("light")) {
			shaderio::Light light;

			std::string lightType = lightNode.attribute("type").value();
			std::string lightID = lightNode.attribute("id").value();

			LightInfo lightInfo;
			if (pugi::xml_node transformNode = lightNode.child("transform")) 
				lightInfo.transformMatrix = getTransformMatrixFromXML(transformNode, lightInfo.staticInstance, &lightInfo);
			if (lightInfo.staticInstance) hasDynamicLight = true;

			if(pugi::xml_node emissiveNode = lightNode.child("emissive"))
				light.color = FzbRenderer::getRGBFromString(emissiveNode.attribute("value").value());
			if (pugi::xml_node intensityNode = lightNode.child("intensity"))
				light.intensity = std::stof(intensityNode.attribute("value").value());

			if (lightType == "point") {
				light.type = shaderio::Point;
				lightInfo.light.pos = glm::vec3(0.0f);
				light.pos = glm::vec3(lightInfo.transformMatrix * glm::vec4(lightInfo.light.pos, 1.0f));
			}
			else if (lightType == "spot") {
				light.type = shaderio::Spot;
				lightInfo.light.pos = glm::vec3(0.0f);
				light.pos = glm::vec3(lightInfo.transformMatrix * glm::vec4(lightInfo.light.pos, 1.0f));
				light.direction = glm::normalize(glm::vec3(lightInfo.transformMatrix * glm::vec4(1.0f, 1.0f, 2.0f, 1.0f)) - light.pos);
				light.coneAngle = 60.0f;
				if (pugi::xml_node coneAngleNode = lightNode.child("coneAngle"))
					light.coneAngle = std::stof(coneAngleNode.attribute("value").value());
			}
			else if (lightType == "sun") {
				light.type == shaderio::Directional;
				sceneInfo.useSky = true;
			}
			else if (lightType == "area") {		//默认是矩形光源
				light.type = shaderio::Area;
				if (pugi::xml_node shapeNode = lightNode.child("shape")) {
					if (std::string(shapeNode.attribute("type").value()) == "obj") {
						std::filesystem::path lightMeshPath = scenePath / shapeNode.attribute("value").value();
						FzbRenderer::MeshSet lightMesh("lightMesh", "obj", lightMeshPath);
						shaderio::BufferView posBufferView = lightMesh.childMeshInfos[0].mesh.triMesh.positions;

						std::vector<glm::vec3> lightMeshVertices(posBufferView.count);
						for (int i = 0; i < posBufferView.count; ++i) {
							uint32_t posIndex = posBufferView.offset + posBufferView.byteStride * i;
							memcpy(lightMeshVertices.data() + i, lightMesh.meshByteData.data() + posIndex, sizeof(glm::vec3));
						}
						
						if (lightInfo.staticInstance) {
							light.pos = glm::vec3(lightInfo.transformMatrix * glm::vec4(lightMeshVertices[0], 1.0f));
							light.edge1 = glm::vec3(lightInfo.transformMatrix * glm::vec4(lightMeshVertices[1], 1.0f)) - light.pos;
							light.edge2 = glm::vec3(lightInfo.transformMatrix * glm::vec4(lightMeshVertices[3], 1.0f)) - light.pos;
							light.direction = glm::normalize(glm::cross(light.edge1, light.edge2));
						}
						else {
							lightInfo.light.pos = glm::vec4(lightMeshVertices[0], 1.0f);
							lightInfo.light.edge1 = lightMeshVertices[1] - lightInfo.light.pos;
							lightInfo.light.edge2 = lightMeshVertices[3] - lightInfo.light.pos;
							lightInfo.light.direction = glm::normalize(glm::cross(lightInfo.light.edge1, lightInfo.light.edge2));
						}
					}
				}
				else {
					if (lightInfo.staticInstance) {
						light.pos = glm::vec3(lightInfo.transformMatrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
						light.edge1 = glm::vec3(lightInfo.transformMatrix * glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)) - light.pos;
						light.edge2 = glm::vec3(lightInfo.transformMatrix * glm::vec4(0.0f, 1.0f, 0.0f, 1.0f)) - light.pos;
						light.direction = glm::normalize(glm::cross(light.edge1, light.edge2));
					}
					else {
						lightInfo.light.pos = glm::vec3(0.0f, 0.0f, 0.0f);
						lightInfo.light.edge1 = glm::vec3(1.0f, 0.0f, 0.0f);
						lightInfo.light.edge2 = glm::vec3(0.0f, 1.0f, 0.0f);
						lightInfo.light.direction = glm::vec3(0.0f, 0.0f, 1.0f);
					}
				}

				if (pugi::xml_node SRSNode = lightNode.child("SphericalRectangleSample"))
					light.SphericalRectangleSample = std::string(SRSNode.attribute("value").value()) == "true";
			}

			sceneInfo.lights[sceneInfo.numLights++] = light;
			lightInfos.push_back(lightInfo);
		}
	}
	//------------------------------------------------Mesh---------------------------------------------------------------
	meshSets.resize(0);
	meshes.resize(0);

	pugi::xml_node meshesNode = sceneInfoNode.child("meshes");
	for (pugi::xml_node meshNode : meshesNode.children("mesh")) {
		std::string meshType = meshNode.attribute("type").value();
		std::string meshID = meshNode.attribute("id").value();

		std::filesystem::path meshPath = scenePath / meshNode.child("filename").attribute("value").value();
		FzbRenderer::MeshSet meshSet(meshID, meshType, meshPath);	//创建MeshSet
		addMeshSet(meshSet);
	}
	//------------------------------------------------Instance---------------------------------------------------------------
	/*
	流程是
	1. 根据instanceXML的meshRef的id在meshSetIDToIndexc中找到索引，根据索引找到meshSet
	2. 为meshSet的每一个childMesh创建一个instance，根据childMesh的meshID知道meshes的索引
	3. 根据materialID在uniqueMaterialIDToIndex找到索引
	4. 记录meshes和materials的索引
	*/
	instances.resize(0);
	dynamicInstances.resize(0);
	pugi::xml_node instanceNode = sceneInfoNode.child("instances");
	for (pugi::xml_node instanceNode : instanceNode.children("instance")) {
		std::string meshSetID = instanceNode.child("meshRef").attribute("id").value();
		std::vector<MeshInfo> childMeshInfos;
		if (meshSetID == "custom") {
			FzbRenderer::MeshSet mesh;
			std::string meshType = instanceNode.child("meshRef").attribute("type").value();
			nvutils::PrimitiveMesh primitive;
			if (meshType == "plane") primitive = FzbRenderer::MeshSet::createPlane(1, 1.0f, 1.0f);
			meshSetID = "custom" + meshType + std::to_string(customPrimitiveCount++);
			mesh = FzbRenderer::MeshSet(meshSetID, primitive);
			addMeshSet(mesh);
			childMeshInfos = mesh.childMeshInfos;
		}
		else {
			if (!meshSetIDToIndex.count(meshSetID)) LOGW("实例没有对应的mesh：%s\n", meshSetID.c_str());
			FzbRenderer::MeshSet& meshSet = meshSets[meshSetIDToIndex[meshSetID]];
			childMeshInfos = meshSet.childMeshInfos;
		}
		for (int i = 0; i < childMeshInfos.size(); i++) {
			shaderio::Instance instance;
			MeshInfo& childMesh = childMeshInfos[i];
			instance.meshIndex = childMesh.meshIndex;

			std::string materialID = "defaultMaterial";
			if (pugi::xml_node materialNode = instanceNode.child("materialRef")) materialID = materialNode.attribute("id").value();
			else materialID = childMesh.materialID;
			if (!uniqueMaterialIDToIndex.count(materialID)) materialID = "defaultMaterial";
			instance.materialIndex = uniqueMaterialIDToIndex[materialID];

			instance.transform = glm::mat4(1.0f);
			DynamicInstanceInfo instanceInfo; bool staticInstance = true;
			if (pugi::xml_node transformNode = instanceNode.child("transform")) 
				instance.transform = getTransformMatrixFromXML(transformNode, staticInstance, &instanceInfo);
			if (staticInstance) instances.push_back(instance); 
			else {
				instanceInfo.transformMatrix = instance.transform;
				dynamicInstances.push_back(instance);
				dynamicInstanceInfos.push_back(instanceInfo);
			}
		}
	}

	doc.reset();

	createSceneInfoBuffer();
}
/*
在shader中获取顶点数据的流程是
1. 先获取到当前实例的索引
	a. 正常管线通过pushconstant得到
	b. rt管线通过InstanceIndex函数得到，vulkan在创建tlas时自动添加，所以要保证tlas的创建顺序与instances的vector顺序相同
2. 然后从sceneInfo中拿到具体的gltfInstance数据，知道变换、materialIndex和meshIndex
3. 然后从sceneInfo中拿到具体的gltfMesh数据
	a. 正常管线中，顶点着色器的输入会给顶点索引vertexIndex，所以根据bufferView的offset和vertexIndex，再加上accessor的offset拿到各个顶点数据
	b. rt管线中，PrimitiveIndex函数
	(根据给blas的顶点buffer和索引buffer，就像我cuda床架BVH一样，可以记录三角形中每个顶点在缓冲中的索引)
	会返回打中的图元的索引，然后同理拿到顶点数据
*/
void FzbRenderer::Scene::createSceneInfoBuffer() {
	SCOPED_TIMER(__FUNCTION__);

	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();

	// Create all mesh buffers
	if (meshes.size() > 0) {
		allocator->createBuffer(bMeshes, std::span(meshes).size_bytes(),
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(bMeshes.buffer);
		NVVK_CHECK(stagingUploader.appendBuffer(bMeshes, 0, std::span<const shaderio::Mesh>(meshes)));
	}

	// Create all instance buffers
	if (instances.size() > 0) {
		allocator->createBuffer(bInstances, std::span(instances).size_bytes() + std::span(dynamicInstances).size_bytes(),
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(bInstances.buffer);
		NVVK_CHECK(stagingUploader.appendBuffer(bInstances, 0,
			std::span<const shaderio::Instance>(instances)));
	}

	// Create all material buffers
	if (materials.size() > 0) {
		allocator->createBuffer(bMaterials, std::span(materials).size_bytes(),
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(bMaterials.buffer);
		NVVK_CHECK(stagingUploader.appendBuffer(bMaterials, 0,
			std::span<const shaderio::BSDFMaterial>(materials)));
	}

	// Create the scene info buffer
	NVVK_CHECK(allocator->createBuffer(bSceneInfo,
		std::span<const shaderio::SceneInfo>(&sceneInfo, 1).size_bytes(),
		VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT));
	NVVK_DBG_NAME(bSceneInfo.buffer);
	NVVK_CHECK(stagingUploader.appendBuffer(bSceneInfo, 0,
		std::span<const shaderio::SceneInfo>(&sceneInfo, 1)));

	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();
	Application::stagingUploader.cmdUploadAppended(cmd);
	Application::app->submitAndWaitTempCmdBuffer(cmd);
}

void FzbRenderer::Scene::clean() {
	nvvk::ResourceAllocator& allocator = Application::allocator;

	allocator.destroyBuffer(bSceneInfo);
	allocator.destroyBuffer(bMeshes);
	allocator.destroyBuffer(bMaterials);
	allocator.destroyBuffer(bInstances);
	for (auto& data : bDatas)
		allocator.destroyBuffer(data);
	for (auto& texture : textures)
		allocator.destroyImage(texture);
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
void FzbRenderer::Scene::preRender() {
	static int frameIndex = 0;
	for (int i = 0; i < sceneInfo.numLights; ++i) {
		LightInfo lightInfo = lightInfos[i];
		if (lightInfo.staticInstance) continue;
		shaderio::Light& light = sceneInfo.lights[i];
		float interpolateValue = (float)frameIndex / lightInfo.speed;
		interpolateValue = interpolateValue - std::floor(interpolateValue);
		glm::mat4 interpolateTransformMatrix = interpolateTransforms(lightInfo.startTransformMatrix, lightInfo.endTransformMatrix, interpolateValue);
		
		light.pos = interpolateTransformMatrix * lightInfo.transformMatrix * glm::vec4(lightInfo.light.pos, 1.0f);
		light.edge1 = interpolateTransformMatrix * lightInfo.transformMatrix * glm::vec4(lightInfo.light.edge1, 1.0f);
		light.edge2 = interpolateTransformMatrix * lightInfo.transformMatrix * glm::vec4(lightInfo.light.edge2, 1.0f);
		light.direction = lightInfo.light.direction * glm::inverse(glm::mat3(interpolateTransformMatrix * lightInfo.transformMatrix));
	}

	const glm::mat4& viewMatrix = cameraManip->getViewMatrix();
	const glm::mat4& projMatrix = cameraManip->getPerspectiveMatrix();
	sceneInfo.viewProjMatrix = projMatrix * viewMatrix;
	sceneInfo.projInvMatrix = glm::inverse(projMatrix);
	sceneInfo.viewInvMatrix = glm::inverse(viewMatrix);
	sceneInfo.cameraPosition = cameraManip->getEye();
	sceneInfo.instances = (shaderio::Instance*)bInstances.address;
	sceneInfo.meshes = (shaderio::Mesh*)bMeshes.address;
	sceneInfo.materials = (shaderio::BSDFMaterial*)bMaterials.address;

	if (dynamicInstances.size() > 0) {
		for (int i = 0; i < dynamicInstances.size(); ++i) {
			DynamicInstanceInfo& instanceInfo = dynamicInstanceInfos[i];
			shaderio::Instance& instance = dynamicInstances[i];
			float interpolateValue = (float)frameIndex++ / instanceInfo.speed;
			interpolateValue = interpolateValue - std::floor(interpolateValue);
			glm::mat4 interpolateTransformMatrix = interpolateTransforms(instanceInfo.startTransformMatrix, instanceInfo.endTransformMatrix, interpolateValue);
			instance.transform = interpolateTransformMatrix * instanceInfo.transformMatrix;
		}
	}
}
void FzbRenderer::Scene::UIRender() {
	if (ImGui::Begin("Scene Resources")) {

	}
	ImGui::End();
}

void FzbRenderer::Scene::updateDataPerFrame(VkCommandBuffer cmd) {
	nvvk::cmdBufferMemoryBarrier(cmd, { bSceneInfo.buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
									   VK_PIPELINE_STAGE_2_TRANSFER_BIT });
	vkCmdUpdateBuffer(cmd, bSceneInfo.buffer, 0, sizeof(shaderio::SceneInfo), &sceneInfo);
	nvvk::cmdBufferMemoryBarrier(cmd, { bSceneInfo.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
									   VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT });

	if (dynamicInstances.size() > 0) {
		nvvk::cmdBufferMemoryBarrier(cmd, { bInstances.buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
								   VK_PIPELINE_STAGE_2_TRANSFER_BIT });
		vkCmdUpdateBuffer(cmd, bInstances.buffer, std::span(instances).size_bytes(), std::span(dynamicInstances).size_bytes(), dynamicInstances.data());
		nvvk::cmdBufferMemoryBarrier(cmd, { bInstances.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
									   VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT });
	}
}

FzbRenderer::MeshInfo FzbRenderer::Scene::getMeshInfo(uint32_t meshIndex) {
	uint32_t meshSetIndex = getMeshSetIndex(meshIndex);
	MeshSet& meshSet = meshSets[meshSetIndex];
	return meshSet.childMeshInfos[meshIndex - meshSet.meshOffset];
}