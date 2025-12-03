#include "Scene.h"
#include "pugixml.hpp"
#include <common/path_utils.hpp>
#include <common/utils.hpp>
#include <common/Application/Application.h>
#include <unordered_map>
#include <nvgui/sky.hpp>
#include <glm/gtc/type_ptr.hpp>

void FzbRenderer::Scene::loadGltfData(const tinygltf::Model& model, bool importInstance) {
	SCOPED_TIMER(__FUNCTION__);

	const uint32_t meshOffset = uint32_t(meshes.size());

	auto getElementByteSize = [](int type) -> uint32_t {	//最小数据单元的大小
		return  type == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ? 2U :
				type == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT ? 4U :
				type == TINYGLTF_COMPONENT_TYPE_FLOAT ? 4U :
				0U;
	};
	auto getTypeSize = [](int type) -> uint32_t {			//顶点属性一个分量的数据单元数量
		return  type == TINYGLTF_TYPE_VEC2 ? 2U :
				type == TINYGLTF_TYPE_VEC3 ? 3U :
				type == TINYGLTF_TYPE_VEC4 ? 4U :
				type == TINYGLTF_TYPE_MAT2 ? 4U * 2U :
				type == TINYGLTF_TYPE_MAT3 ? 4U * 3U :
				type == TINYGLTF_TYPE_MAT4 ? 4U * 4U :
				0U;
	};
	auto extractAttribute = [&](const std::string& name, shaderio::BufferView& attr, const tinygltf::Primitive& primitive) {
		if (!primitive.attributes.contains(name)) {
			attr.offset = -1;
			return;
		}
		const tinygltf::Accessor& acc = model.accessors[primitive.attributes.at(name)];	//Accessor知道如何解读一个bufferView
		const tinygltf::BufferView& bv = model.bufferViews[acc.bufferView];				//bufferView知道buffer的某一段的信息
		assert((acc.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) && "Should be floats");
		attr = {
			//bv.byteOffset相当于在buffer中的起点;acc.byteOffset表面顶点某个属性在顶点数据中的偏移，比方说normal的offset为3*4=12
			.offset = uint32_t(bv.byteOffset + acc.byteOffset),
			.count = uint32_t(acc.count),
			.byteStride = uint32_t(bv.byteStride ? uint32_t(bv.byteStride) : getTypeSize(acc.type) * getElementByteSize(acc.componentType)),
		};
	};

	nvvk::Buffer bGltfData;
	uint32_t bufferIndex{};
	{
		nvvk::ResourceAllocator* allocator = Application::stagingUploader.getResourceAllocator();
		NVVK_CHECK(allocator->createBuffer(bGltfData, std::span<const unsigned char>(model.buffers[0].data).size_bytes(),
			VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT
			| VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR));
		NVVK_CHECK(Application::stagingUploader.appendBuffer(bGltfData, 0, std::span<const unsigned char>(model.buffers[0].data)));
		NVVK_DBG_NAME(bGltfData.buffer);
		bufferIndex = static_cast<uint32_t>(bGltfDatas.size());
		bGltfDatas.push_back(bGltfData);
	}

	for (size_t meshIdx = 0; meshIdx < model.meshes.size(); ++meshIdx) {
		shaderio::GltfMesh mesh{};

		const tinygltf::Mesh& tinyMesh = model.meshes[meshIdx];
		const tinygltf::Primitive& primitive = tinyMesh.primitives.front();
		assert((tinyMesh.primitives.size() == 1 && primitive.mode == TINYGLTF_MODE_TRIANGLES) && "Must have one triangle primitive");

		auto& accessor = model.accessors[primitive.indices];
		auto& bufferView = model.bufferViews[accessor.bufferView];
		assert((accessor.count % 3 == 0) && "Should be a multiple of 3");
		mesh.triMesh.indices = {
			.offset = uint32_t(bufferView.byteOffset + accessor.byteOffset),
			.count = uint32_t(accessor.count),
			.byteStride = uint32_t(bufferView.byteStride ? bufferView.byteStride : getElementByteSize(accessor.componentType)),
		};
		mesh.indexType = accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;

		mesh.gltfBuffer = (uint8_t*)bGltfData.address;  //这样address+1只移动1字节，可以按字节偏移寻址

		extractAttribute("POSITION", mesh.triMesh.positions, primitive);
		extractAttribute("NORMAL", mesh.triMesh.normals, primitive);
		extractAttribute("COLOR_0", mesh.triMesh.colorVert, primitive);
		extractAttribute("TEXCOORD_0", mesh.triMesh.texCoords, primitive);
		extractAttribute("TANGENT", mesh.triMesh.tangents, primitive);

		meshes.emplace_back(mesh);
		meshToBufferIndex.push_back(bufferIndex);
	}
	
	if (importInstance) {
		std::function<void(const tinygltf::Node&, const glm::mat4&)> processNode = [&](const tinygltf::Node& node, const glm::mat4& parentTransform) {
			glm::mat4 nodeTransform = parentTransform;	//当前node的变换矩阵与父node的变换依赖
			if (!node.matrix.empty()) {
				glm::mat4 matrix = glm::make_mat4(node.matrix.data());
				nodeTransform = parentTransform * matrix;
			}
			else {
				if (!node.translation.empty()) {
					glm::vec3 translation = glm::make_vec3(node.translation.data());
					nodeTransform = glm::translate(nodeTransform, translation);
				}
				if (!node.rotation.empty())
				{
					glm::quat rotation = glm::make_quat(node.rotation.data());
					nodeTransform = nodeTransform * glm::mat4_cast(rotation);
				}
				if (!node.scale.empty())
				{
					glm::vec3 scale = glm::make_vec3(node.scale.data());
					nodeTransform = glm::scale(nodeTransform, scale);
				}
			}

			if (node.mesh != -1) {
				const tinygltf::Mesh& tinyMesh = model.meshes[node.mesh];
				const tinygltf::Primitive& primitive = tinyMesh.primitives.front();
				assert((tinyMesh.primitives.size() == 1 && primitive.mode == TINYGLTF_MODE_TRIANGLES) && "Must have one triangle primitive");
				shaderio::GltfInstance instance{};
				instance.meshIndex = node.mesh + meshOffset;
				instance.transform = nodeTransform;
				instances.push_back(instance);
			}

			for (int childIdx : node.children)
			{
				if (childIdx >= 0 && childIdx < static_cast<int>(model.nodes.size()))
					processNode(model.nodes[childIdx], nodeTransform);
			}
		};
	
		for (size_t nodeIdx = 0; nodeIdx < model.nodes.size(); ++nodeIdx) {
			const tinygltf::Node& node = model.nodes[nodeIdx];
			bool isRootNode = true;
			for (const auto& otherNode : model.nodes) {
				for (int childIdx : otherNode.children) {
					if (childIdx == static_cast<int>(nodeIdx)) {
						isRootNode = false;
						break;
					}
				}
				if (!isRootNode) break;
			}

			if (isRootNode) processNode(node, glm::mat4(1.0f));
		}
	}
};
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
	std::unordered_map<std::string, uint32_t> uniqueMaterials;
	materials.resize(0);

	//默认材质
	shaderio::GltfMetallicRoughness defaultMaterial = {
		.baseColorFactor = glm::vec4(1.0f),
		.metallicFactor = 0.1f,
		.roughnessFactor = 0.8f,
		.baseColorTextureIndex = -1
	};
	materials.push_back(defaultMaterial);
	uniqueMaterials.insert({ "defaultMaterial", 0 });

	pugi::xml_node bsdfsNode = sceneInfoNode.child("bsdfs");
	for (pugi::xml_node bsdfNode : bsdfsNode.children("bsdf")) {
		std::string materialID = bsdfNode.attribute("id").value();
		if (uniqueMaterials.count(materialID)) {
			printf("重复material读取: %s\n", materialID);
			continue;
		}
		shaderio::GltfMetallicRoughness material = defaultMaterial;
		if (pugi::xml_node baseColorNode = bsdfNode.child("baseColorFactor"))
			material.baseColorFactor = FzbRenderer::getRGBAFromString(baseColorNode.attribute("value").value());

		if (pugi::xml_node metallicNode = bsdfNode.child("metallicFactor"))
			material.metallicFactor = std::stof(metallicNode.attribute("value").value());

		if (pugi::xml_node roughnessNode = bsdfNode.child("roughnessFactor"))
			material.roughnessFactor = std::stof(roughnessNode.attribute("value").value());

		if (pugi::xml_node textureIndexNode = bsdfNode.child("baseColorTextureIndex"))
			material.baseColorTextureIndex = std::stoi(textureIndexNode.attribute("value").value());

		uniqueMaterials.insert({ materialID, materials.size()});
		materials.push_back(material);
	}
	//------------------------------------------------光源---------------------------------------------------------------
	if (pugi::xml_node lightsNode = sceneInfoNode.child("lights")) {
		sceneInfo.useSky = false;
		sceneInfo.backgroundColor = glm::vec3(0.85f);
		if (pugi::xml_node backgroudColorNode = lightsNode.child("backgroundColor"))
			sceneInfo.backgroundColor = FzbRenderer::getRGBFromString(backgroudColorNode.attribute("value").value());

		sceneInfo.numLights = 0;
		for (pugi::xml_node lightNode : lightsNode.children("light")) {
			shaderio::GltfPunctual light;

			std::string lightType = lightNode.attribute("type").value();
			std::string lightID = lightNode.attribute("id").value();

			glm::mat4 transformMatrix = glm::mat4(1.0f);
			if (pugi::xml_node tranformNode = lightNode.child("transform"))
				transformMatrix = FzbRenderer::getMat4FromString(tranformNode.child("matrix").attribute("value").value());

			if(pugi::xml_node emissiveNode = lightNode.child("emissive"))
				light.color = FzbRenderer::getRGBFromString(emissiveNode.attribute("value").value());
			if (pugi::xml_node intensityNode = lightNode.child("intensity"))
				light.intensity = std::stof(intensityNode.attribute("value").value());

			if (lightType == "point") {
				light.type = shaderio::ePoint;
				light.position = glm::vec3(transformMatrix * glm::vec4(1.0f));
			}
			else if (lightType == "spot") {
				light.type = shaderio::eSpot;
				light.position = glm::vec3(transformMatrix * glm::vec4(1.0f));
				light.direction = glm::normalize(glm::vec3(transformMatrix * glm::vec4(1.0f, 1.0f, 2.0f, 1.0f)) - light.position);
				light.coneAngle = 60.0f;
				if (pugi::xml_node coneAngleNode = lightNode.child("coneAngle"))
					light.coneAngle = std::stof(coneAngleNode.attribute("value").value());
			}
			else if (lightType == "sun") {
				light.type == shaderio::eDirectional;
				sceneInfo.useSky = true;
			}
			else if (lightType == "area") {
				light.type = shaderio::eArea;
			}

			sceneInfo.punctualLights[sceneInfo.numLights++] = light;
		}
	}
	//------------------------------------------------Mesh---------------------------------------------------------------
	std::map<std::string, uint32_t> meshIDToIndex;
	meshes.resize(0);
	pugi::xml_node meshesNode = sceneInfoNode.child("meshes");
	for (pugi::xml_node meshNode : meshesNode.children("mesh")) {
		shaderio::GltfMesh mesh;

		std::string meshType = meshNode.attribute("type").value();
		std::string meshID = meshNode.attribute("id").value();
		meshIDToIndex.insert({ meshID, meshes.size() });

		if (meshType == "gltf") {
			std::filesystem::path meshPath = FzbRenderer::getProjectRootDir() / "resources";
			meshPath = meshPath / (meshNode.child("filename").attribute("value").value());
			tinygltf::Model gltfModel = nvsamples::loadGltfResources(nvutils::findFile(meshPath, { meshPath }));
			loadGltfData(gltfModel);
		}
	}
	//------------------------------------------------Instance---------------------------------------------------------------
	pugi::xml_node instanceNode = sceneInfoNode.child("instances");
	for (pugi::xml_node instanceNode : instanceNode.children("instance")) {
		shaderio::GltfInstance instance;

		std::string meshID = instanceNode.child("meshRef").attribute("id").value();
		instance.meshIndex = meshIDToIndex[meshID];

		std::string materialID = "defaultMaterial";
		if (pugi::xml_node materialNode = instanceNode.child("materialRef")) materialID = materialNode.attribute("id").value();
		if (!uniqueMaterials.count(materialID)) materialID = "defaultMaterial";
		instance.materialIndex = uniqueMaterials[materialID];

		glm::mat4 transformMatrix(1.0f);
		if (pugi::xml_node transformNode = instanceNode.select_node("transform").node()) {
			transformMatrix = FzbRenderer::getMat4FromString(transformNode.child("matrix").attribute("value").value());
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
	NVVK_CHECK(stagingUploader.appendBuffer(bMeshes, 0, std::span<const shaderio::GltfMesh>(meshes)));

	// Create all instance buffers
	allocator->createBuffer(bInstances, std::span(instances).size_bytes(),
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(bInstances.buffer);
	NVVK_CHECK(stagingUploader.appendBuffer(bInstances, 0,
		std::span<const shaderio::GltfInstance>(instances)));

	// Create all material buffers
	allocator->createBuffer(bMaterials, std::span(materials).size_bytes(),
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(bMaterials.buffer);
	NVVK_CHECK(stagingUploader.appendBuffer(bMaterials, 0,
		std::span<const shaderio::GltfMetallicRoughness>(materials)));

	// Create the scene info buffer
	NVVK_CHECK(allocator->createBuffer(bSceneInfo,
		std::span<const shaderio::GltfSceneInfo>(&sceneInfo, 1).size_bytes(),
		VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT));
	NVVK_DBG_NAME(bSceneInfo.buffer);
	NVVK_CHECK(stagingUploader.appendBuffer(bSceneInfo, 0,
		std::span<const shaderio::GltfSceneInfo>(&sceneInfo, 1)));
}

void FzbRenderer::Scene::clean() {
	nvvk::ResourceAllocator& allocator = Application::allocator;

	allocator.destroyBuffer(bSceneInfo);
	allocator.destroyBuffer(bMeshes);
	allocator.destroyBuffer(bMaterials);
	allocator.destroyBuffer(bInstances);
	for (auto& gltfData : bGltfDatas)
		allocator.destroyBuffer(gltfData);
	for (auto& texture : textures)
		allocator.destroyImage(texture);
}