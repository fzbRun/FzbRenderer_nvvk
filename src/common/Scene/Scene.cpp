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
	staticInstanceSets.resize(0); periodInstanceSets.resize(0); randomInstanceSets.resize(0);
	pugi::xml_node instancesNode = sceneInfoNode.child("instances");
	for (pugi::xml_node instanceNode : instancesNode.children("instance")) {
		InstanceSet instanceSet = InstanceSet(instanceNode);
		if(instanceSet.instanceID != "defaultInstanceID")
			instanceIDToInstance.insert({ instanceSet.instanceID, {instanceSet.type, getInstanceSetSize(instanceSet.type)}});
		addInstanceSet(instanceSet);
		if(instanceSet.type == Static) staticInstanceCount += instanceSet.childInstances.size();
		else if(instanceSet.type == PeriodMotion) periodInstanceCount += instanceSet.childInstances.size();
		else if(instanceSet.type == RandomMotion) randomInstanceCount += instanceSet.childInstances.size();
	}

	uint32_t offset = 0;
	instances.resize(staticInstanceCount + periodInstanceCount + randomInstanceCount);
	for (int i = 0; i < staticInstanceSets.size(); ++i) {
		staticInstanceSets[i].getInstance(instances, offset, 0);
		offset += staticInstanceSets[i].childInstances.size();
	}
	for (int i = 0; i < periodInstanceSets.size(); ++i) {
		InstanceSet& instanceSet = periodInstanceSets[i];
		instanceSet.getInstance(instances, offset, 0);

		for (int j = 0; j < periodInstanceSets[i].childInstances.size(); ++j)
			periodInstanceIndexToInstanceSetIndex.insert({ offset, i });

		offset += instanceSet.childInstances.size();
	}
	for (int i = 0; i < randomInstanceSets.size(); ++i) {
		InstanceSet& instanceSet = randomInstanceSets[i];
		instanceSet.getInstance(instances, offset, 0);
		offset += instanceSet.childInstances.size();
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
			++sceneInfo.numLights;

			LightInstance lightInstance = LightInstance(lightNode);
			lightInstances.push_back(lightInstance);

			shaderio::Light& light = lightInstances[lightInstances.size() - 1].light;

			std::string lightType = lightNode.attribute("type").value();
			std::string lightID = lightNode.attribute("id").value();

			if (pugi::xml_node emissiveNode = lightNode.child("emissive"))
				light.color = FzbRenderer::getRGBFromString(emissiveNode.attribute("value").value());
			if (pugi::xml_node intensityNode = lightNode.child("intensity"))
				light.intensity = std::stof(intensityNode.attribute("value").value());

			if (lightType == "point") {
				light.type = shaderio::Point;
				light.pos = glm::vec3(lightInstance.baseMatrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
			}
			else if (lightType == "spot") {
				light.type = shaderio::Spot;
				light.pos = glm::vec3(lightInstance.baseMatrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
				light.direction = glm::normalize(glm::vec3(lightInstance.baseMatrix * glm::vec4(1.0f, 1.0f, 2.0f, 1.0f)));
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
				std::vector<glm::vec3> lightMeshVertices(4);
				if (pugi::xml_node shapeNode = lightNode.child("shape")) {
					if (std::string(shapeNode.attribute("type").value()) == "obj") {
						std::filesystem::path lightMeshPath = scenePath / shapeNode.attribute("value").value();
						FzbRenderer::MeshSet lightMesh("lightMesh", "obj", lightMeshPath);
						shaderio::BufferView posBufferView = lightMesh.childMeshInfos[0].mesh.triMesh.positions;

						for (int i = 0; i < 4; ++i) {
							uint32_t posIndex = posBufferView.offset + posBufferView.byteStride * i;
							memcpy(lightMeshVertices.data() + i, lightMesh.meshByteData.data() + posIndex, sizeof(glm::vec3));
						}
					}
				}
				else {
					lightMeshVertices[0] = glm::vec3(0.0f, 0.0f, 0.0f);
					lightMeshVertices[1] = glm::vec3(1.0f, 0.0f, 0.0f);
					lightMeshVertices[2] = glm::vec3(1.0f, 1.0f, 0.0f);
					lightMeshVertices[3] = glm::vec3(0.0f, 1.0f, 0.0f);
				}

				light.pos = glm::vec3(lightInstance.baseMatrix * glm::vec4(lightMeshVertices[0], 1.0f));
				light.edge1 = glm::vec3(lightInstance.baseMatrix * glm::vec4(lightMeshVertices[1], 1.0f)) - light.pos;
				light.edge2 = glm::vec3(lightInstance.baseMatrix * glm::vec4(lightMeshVertices[3], 1.0f)) - light.pos;
				light.direction = glm::normalize(glm::cross(light.edge1, light.edge2));
				if (lightInstance.type != InstanceType::Static) hasDynamicLight = true;

				if (pugi::xml_node SRSNode = lightNode.child("SphericalRectangleSample"))
					light.SphericalRectangleSample = std::string(SRSNode.attribute("value").value()) == "true";
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
		allocator->createBuffer(bInstances, std::span(instances).size_bytes(),
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(bInstances.buffer);
		if(instances.size() == staticInstanceCount)
			NVVK_CHECK(stagingUploader.appendBuffer(bInstances, 0, std::span<const shaderio::Instance>(instances)));
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

void FzbRenderer::Scene::preRender() {
	for (int i = 0; i < sceneInfo.numLights; ++i) {
		LightInstance lightInstanceInfo = lightInstances[i];		
		float time = frameIndex % (2 * lightInstanceInfo.time);
		if (time < lightInstanceInfo.time) time /= lightInstanceInfo.time;
		else time = 2.0f - (time / lightInstanceInfo.time);
		sceneInfo.lights[i] = lightInstanceInfo.getLight(time);
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

	uint32_t offset = staticInstanceCount;
	for (int i = 0; i < periodInstanceSets.size(); ++i) {
		InstanceSet& instanceSet = periodInstanceSets[i];
		float time = frameIndex % (2 * instanceSet.time);
		if (time < instanceSet.time) time /= instanceSet.time;
		else time = 2.0f - (time / instanceSet.time);
		instanceSet.getInstance(instances, offset, time);
		offset += instanceSet.childInstances.size();
	}
	for (int i = 0; i < randomInstanceSets.size(); ++i) {
		InstanceSet& instanceSet = randomInstanceSets[i];
		instanceSet.getInstance(instances, offset, 0);
		offset += instanceSet.childInstances.size();
	}

	++frameIndex;
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

	if (periodInstanceCount + randomInstanceCount > 0) {
		nvvk::cmdBufferMemoryBarrier(cmd, { bInstances.buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
								   VK_PIPELINE_STAGE_2_TRANSFER_BIT });
		vkCmdUpdateBuffer(cmd, bInstances.buffer, 0, std::span(instances).size_bytes(), instances.data());
		nvvk::cmdBufferMemoryBarrier(cmd, { bInstances.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
									   VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT });
	}
}

FzbRenderer::MeshInfo FzbRenderer::Scene::getMeshInfo(uint32_t meshIndex) {
	uint32_t meshSetIndex = getMeshSetIndex(meshIndex);
	MeshSet& meshSet = meshSets[meshSetIndex];
	return meshSet.childMeshInfos[meshIndex - meshSet.meshOffset];
}
FzbRenderer::InstanceSet FzbRenderer::Scene::getInstanceSet(InstanceType type, uint32_t index) {
	switch (type) {
		case Static: return staticInstanceSets[index]; break;
		case PeriodMotion: return periodInstanceSets[index]; break;
		case RandomMotion: return randomInstanceSets[index]; break;
		default: printf("实例没有相应类型或该类型没有%d索引", index);
	}
	return InstanceSet();
}
uint32_t FzbRenderer::Scene::getInstanceSetSize(InstanceType type) {
	switch (type) {
		case Static: return staticInstanceCount; break;
		case PeriodMotion: return periodInstanceCount; break;
		case RandomMotion: return randomInstanceCount; break;
		default: printf("实例没有相应类型"); return 0;
	}

	return 0;
}
void FzbRenderer::Scene::addInstanceSet(InstanceSet& instanceSet) {
	switch (instanceSet.type) {
		case Static: return staticInstanceSets.push_back(instanceSet); break;
		case PeriodMotion: return periodInstanceSets.push_back(instanceSet); break;
		case RandomMotion: return randomInstanceSets.push_back(instanceSet); break;
		default: printf("实例没有相应类型");
	}
}