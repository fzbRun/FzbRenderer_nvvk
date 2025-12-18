#include "Scene.h"
#include "pugixml.hpp"
#include <common/path_utils.hpp>
#include <common/utils.hpp>
#include <common/Application/Application.h>
#include <unordered_map>
#include <nvgui/sky.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <common/Material/Material.h>

int FzbRenderer::Scene::loadTexture(const std::filesystem::path& texturePath) {
	if (texturePathMap.count(texturePath)) return texturePathMap[texturePath];

	VkCommandBuffer       cmd = Application::app->createTempCmdBuffer();
	nvvk::Image texture = nvsamples::loadAndCreateImage(cmd, Application::stagingUploader, Application::app->getDevice(), texturePath);  // Load the image from the file and create a texture from it
	NVVK_DBG_NAME(texture.image);
	Application::app->submitAndWaitTempCmdBuffer(cmd);
	Application::samplerPool.acquireSampler(texture.descriptor.sampler);
	textures.emplace_back(texture);  // Store the texture in the vector of textures
	texturePathMap.insert({ texturePath, textures.size() - 1 });
	return textures.size() - 1;
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
	std::unordered_map<std::string, uint32_t> uniqueMaterialIDToIndex;
	materials.resize(0);
	texturePathMap.clear();
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

			glm::mat4 transformMatrix = glm::mat4(1.0f);
			if (pugi::xml_node transformNode = lightNode.child("transform")) {
				if (pugi::xml_node matrixNode = transformNode.select_node("matrix").node())
					transformMatrix = FzbRenderer::getMat4FromString(matrixNode.attribute("value").value());
				if (pugi::xml_node translateNode = transformNode.select_node("translate").node()) {
					glm::vec3 translateValue = FzbRenderer::getRGBFromString(translateNode.attribute("value").value());
					transformMatrix = glm::translate(transformMatrix, translateValue);
				}
				if (pugi::xml_node rotateNode = transformNode.select_node("rotate").node()) {
					glm::vec3 rotateAngle = glm::radians(FzbRenderer::getRGBFromString(rotateNode.attribute("value").value()));
					if (rotateAngle.x > 0.01f) transformMatrix = glm::rotate(transformMatrix, rotateAngle.x, glm::vec3(1, 0, 0));
					if (rotateAngle.y > 0.01f) transformMatrix = glm::rotate(transformMatrix, rotateAngle.y, glm::vec3(0, 1, 0));
					if (rotateAngle.z > 0.01f) transformMatrix = glm::rotate(transformMatrix, rotateAngle.z, glm::vec3(0, 0, 1));
				}
			}

			if(pugi::xml_node emissiveNode = lightNode.child("emissive"))
				light.color = FzbRenderer::getRGBFromString(emissiveNode.attribute("value").value());
			if (pugi::xml_node intensityNode = lightNode.child("intensity"))
				light.intensity = std::stof(intensityNode.attribute("value").value());

			if (lightType == "point") {
				light.type = shaderio::Point;
				light.pos = glm::vec3(transformMatrix * glm::vec4(glm::vec3(0.0f), 1.0f));
			}
			else if (lightType == "spot") {
				light.type = shaderio::Spot;
				light.pos = glm::vec3(transformMatrix * glm::vec4(glm::vec3(0.0f), 1.0f));
				light.direction = glm::normalize(glm::vec3(transformMatrix * glm::vec4(1.0f, 1.0f, 2.0f, 1.0f)) - light.pos);
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
						FzbRenderer::Mesh lightMesh("lightMesh", "obj", lightMeshPath);
						shaderio::BufferView posBufferView = lightMesh.childMeshes[0].mesh.triMesh.positions;

						std::vector<glm::vec3> lightMeshVertices(posBufferView.count);
						for (int i = 0; i < posBufferView.count; ++i) {
							uint32_t posIndex = posBufferView.offset + posBufferView.byteStride * i;
							memcpy(lightMeshVertices.data() + i, lightMesh.meshByteData.data() + posIndex, sizeof(glm::vec3));
						}

						light.pos = glm::vec3(transformMatrix * glm::vec4(lightMeshVertices[0], 1.0f));
						light.edge1 = glm::vec3(transformMatrix * glm::vec4(lightMeshVertices[1], 1.0f)) - light.pos;
						light.edge2 = glm::vec3(transformMatrix * glm::vec4(lightMeshVertices[3], 1.0f)) - light.pos;
						light.direction = glm::normalize(glm::cross(light.edge1, light.edge2));
					}
				}
				else {
					light.pos = glm::vec3(transformMatrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
					light.edge1 = glm::vec3(transformMatrix * glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)) - light.pos;
					light.edge2 = glm::vec3(transformMatrix * glm::vec4(0.0f, 1.0f, 0.0f, 1.0f)) - light.pos;
					light.direction = glm::normalize(glm::cross(light.edge1, light.edge2));
				}

				if (pugi::xml_node SRSNode = lightNode.child("SphericalRectangleSample"))
					light.SphericalRectangleSample = std::string(SRSNode.attribute("value").value()) == "true";
			}

			sceneInfo.lights[sceneInfo.numLights++] = light;
		}
	}
	//------------------------------------------------Mesh---------------------------------------------------------------
	std::map<std::string, uint32_t> meshSetIDToIndex;
	meshSets.resize(0);
	std::map<std::string, uint32_t> meshIDToIndex;
	meshes.resize(0);

	pugi::xml_node meshesNode = sceneInfoNode.child("meshes");
	for (pugi::xml_node meshNode : meshesNode.children("mesh")) {
		std::string meshType = meshNode.attribute("type").value();
		std::string meshID = meshNode.attribute("id").value();
		meshSetIDToIndex.insert({ meshID, meshSets.size() });

		std::filesystem::path meshPath = scenePath / meshNode.child("filename").attribute("value").value();
		FzbRenderer::Mesh meshSet(meshID, meshType, meshPath);
		meshSets.push_back(meshSet);

		nvvk::Buffer bData = meshSet.createMeshDataBuffer();
		uint32_t bufferIndex = static_cast<uint32_t>(bDatas.size());
		bDatas.push_back(bData);

		for (int i = 0; i < meshSet.childMeshes.size(); i++)
		{
			ChildMesh& childMesh = meshSet.childMeshes[i];
			childMesh.mesh.dataBuffer = (uint8_t*)bData.address;  //这样address+1只移动1字节，可以按字节偏移寻址
			meshIDToIndex.insert({ childMesh.meshID, meshes.size() });
			meshes.emplace_back(childMesh.mesh);
			meshToBufferIndex.push_back(bufferIndex);

			if (!uniqueMaterialIDToIndex.count(childMesh.materialID)) {
				uniqueMaterialIDToIndex.insert({ childMesh.materialID, materials.size() });
				materials.push_back(childMesh.material);
			}
		}
	}
	//------------------------------------------------Instance---------------------------------------------------------------
	pugi::xml_node instanceNode = sceneInfoNode.child("instances");
	for (pugi::xml_node instanceNode : instanceNode.children("instance")) {
		std::string meshSetID = instanceNode.child("meshRef").attribute("id").value();
		std::vector<ChildMesh> childMeshes;
		if (meshSetID == "custom") {
			FzbRenderer::Mesh mesh;
			std::string meshType = instanceNode.child("meshRef").attribute("type").value();
			nvutils::PrimitiveMesh primitive;
			if (meshType == "plane") primitive = FzbRenderer::Mesh::createPlane(1, 1.0f, 1.0f);
			meshSetID = "custom" + meshType + std::to_string(customPrimitiveCount++);
			mesh = FzbRenderer::Mesh(meshSetID, primitive);
			meshSetIDToIndex.insert({ meshSetID, meshSets.size() });

			nvvk::Buffer bData = mesh.createMeshDataBuffer();
			uint32_t bufferIndex = static_cast<uint32_t>(bDatas.size());
			bDatas.push_back(bData);

			childMeshes = mesh.childMeshes;
			for (int i = 0; i < childMeshes.size(); ++i) {
				ChildMesh& childMesh = mesh.childMeshes[i];
				childMesh.mesh.dataBuffer = (uint8_t*)bData.address;
				meshIDToIndex.insert({ childMesh.meshID, meshes.size() });
				meshes.emplace_back(childMesh.mesh);
				meshToBufferIndex.push_back(bufferIndex);
			}
		}
		else {
			if (!meshSetIDToIndex.count(meshSetID)) LOGW("实例没有对应的mesh：%s\n", meshSetID.c_str());
			FzbRenderer::Mesh& meshSet = meshSets[meshSetIDToIndex[meshSetID]];
			childMeshes = meshSet.childMeshes;
		}
		for (int i = 0; i < childMeshes.size(); i++) {
			shaderio::Instance instance;
			ChildMesh& childMesh = childMeshes[i];
			instance.meshIndex = meshIDToIndex[childMesh.meshID];

			std::string materialID = "defaultMaterial";
			if (pugi::xml_node materialNode = instanceNode.child("materialRef")) materialID = materialNode.attribute("id").value();
			else materialID = childMesh.materialID;
			if (!uniqueMaterialIDToIndex.count(materialID)) materialID = "defaultMaterial";
			instance.materialIndex = uniqueMaterialIDToIndex[materialID];

			glm::mat4 transformMatrix(1.0f);
			if (pugi::xml_node transformNode = instanceNode.select_node("transform").node()) {
				if (pugi::xml_node matrixNode = transformNode.child("matrix"))
					transformMatrix = FzbRenderer::getMat4FromString(matrixNode.attribute("value").value());
				if (pugi::xml_node translateNode = transformNode.select_node("translate").node()) {
					glm::vec3 translateValue = FzbRenderer::getRGBFromString(translateNode.attribute("value").value());
					transformMatrix = glm::translate(transformMatrix, translateValue);
				}
				if (pugi::xml_node rotateNode = transformNode.select_node("rotate").node()) {
					glm::vec3 rotateAngle = glm::radians(FzbRenderer::getRGBFromString(rotateNode.attribute("value").value()));
					if (rotateAngle.x > 0.01f) transformMatrix = glm::rotate(transformMatrix, rotateAngle.x, glm::vec3(1, 0, 0));
					if (rotateAngle.y > 0.01f) transformMatrix = glm::rotate(transformMatrix, rotateAngle.y, glm::vec3(0, 1, 0));
					if (rotateAngle.z > 0.01f) transformMatrix = glm::rotate(transformMatrix, rotateAngle.z, glm::vec3(0, 0, 1));
				}
			}
			instance.transform = transformMatrix;

			instances.push_back(instance);
		}
	}

	doc.reset();

	createSceneInfBuffer();
	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();
	Application::stagingUploader.cmdUploadAppended(cmd);
	Application::app->submitAndWaitTempCmdBuffer(cmd);
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
void FzbRenderer::Scene::createSceneInfBuffer() {
	SCOPED_TIMER(__FUNCTION__);

	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();

	// Create all mesh buffers
	allocator->createBuffer(bMeshes, std::span(meshes).size_bytes(),
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(bMeshes.buffer);
	NVVK_CHECK(stagingUploader.appendBuffer(bMeshes, 0, std::span<const shaderio::Mesh>(meshes)));

	// Create all instance buffers
	allocator->createBuffer(bInstances, std::span(instances).size_bytes(),
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(bInstances.buffer);
	NVVK_CHECK(stagingUploader.appendBuffer(bInstances, 0,
		std::span<const shaderio::Instance>(instances)));

	// Create all material buffers
	allocator->createBuffer(bMaterials, std::span(materials).size_bytes(),
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(bMaterials.buffer);
	NVVK_CHECK(stagingUploader.appendBuffer(bMaterials, 0,
		std::span<const shaderio::BSDFMaterial>(materials)));

	// Create the scene info buffer
	NVVK_CHECK(allocator->createBuffer(bSceneInfo,
		std::span<const shaderio::SceneInfo>(&sceneInfo, 1).size_bytes(),
		VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT));
	NVVK_DBG_NAME(bSceneInfo.buffer);
	NVVK_CHECK(stagingUploader.appendBuffer(bSceneInfo, 0,
		std::span<const shaderio::SceneInfo>(&sceneInfo, 1)));
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

void FzbRenderer::Scene::UIRender() {
	if (ImGui::Begin("Scene Resources")) {

	}
	ImGui::End();
}