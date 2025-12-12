#include <nvutils/timers.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/resource_allocator.hpp>
#include <common/Application/Application.h>
#include <glm/gtc/type_ptr.hpp>
#include <memory>
#include "./Mesh.h"

void FzbRenderer::Mesh::loadData(std::string meshType, std::filesystem::path meshPath)
{
	Scene& scene = Application::sceneResource;
	if (meshType == "gltf" || meshType == "glb") {
		tinygltf::Model gltfModel = nvsamples::loadGltfResources(nvutils::findFile(meshPath, { meshPath }));
		loadGltfData(gltfModel);
	}
	else if (meshType == "obj") {
		loadObjData(meshPath);
	}
}

void FzbRenderer::Mesh::loadGltfData(const tinygltf::Model& model, bool importInstance) {
	SCOPED_TIMER(__FUNCTION__);

	Scene& scene = Application::sceneResource;
	const uint32_t meshOffset = uint32_t(scene.meshes.size());

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
		bufferIndex = static_cast<uint32_t>(scene.bDatas.size());
		scene.bDatas.push_back(bGltfData);
	}

	for (size_t meshIdx = 0; meshIdx < model.meshes.size(); ++meshIdx) {
		shaderio::Mesh mesh{};

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

		mesh.dataBuffer = (uint8_t*)bGltfData.address;  //这样address+1只移动1字节，可以按字节偏移寻址

		extractAttribute("POSITION", mesh.triMesh.positions, primitive);
		extractAttribute("NORMAL", mesh.triMesh.normals, primitive);
		extractAttribute("COLOR_0", mesh.triMesh.colorVert, primitive);
		extractAttribute("TEXCOORD_0", mesh.triMesh.texCoords, primitive);
		extractAttribute("TANGENT", mesh.triMesh.tangents, primitive);

		scene.meshes.emplace_back(mesh);
		scene.meshToBufferIndex.push_back(bufferIndex);
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
				shaderio::Instance instance{};
				instance.meshIndex = node.mesh + meshOffset;
				instance.transform = nodeTransform;
				scene.instances.push_back(instance);
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

uint32_t addData(std::vector<uint8_t> data, std::vector<uint8_t> newData, uint32_t alignment) {
	uint32_t dataSize = data.size();
	uint32_t padding = alignment - dataSize % alignment;
	data.reserve(dataSize + padding + newData.size());
	
	if (padding > 0) {
		std::vector<uint8_t> paddingData(padding);
		data.insert(data.end(), paddingData.begin(), paddingData.end());
	}
	data.insert(data.end(), newData.begin(), newData.end());

	return padding;
}
void FzbRenderer::Mesh::processMesh(aiMesh* meshData, const aiScene* sceneData) {
	shaderio::Mesh mesh;
	std::vector<uint8_t> meshByteData;

	uint32_t faceNum = meshData->mNumFaces;
	uint32_t indexNum = 0;
	for (uint32_t i = 0; i < faceNum; ++i) indexNum += meshData->mFaces[i].mNumIndices;
	uint32_t maxIndex = 0;
	for (uint32_t i = 0; i < faceNum; i++) {
		aiFace& face = meshData->mFaces[i];
		for (uint32_t j = 0; j < face.mNumIndices; j++) {
			if (face.mIndices[j] > maxIndex)
				maxIndex = face.mIndices[j];
		}
	}
	std::vector<uint8_t> indexByteData;
	if (maxIndex <= 0xFFFF) {  // 65535
		mesh.indexType = VK_INDEX_TYPE_UINT16;
		std::vector<uint16_t> indexData;
		indexData.resize(indexNum);

		uint32_t offset = 0;
		for (uint32_t i = 0; i < faceNum; i++) {
			aiFace& face = meshData->mFaces[i];
			for (uint32_t j = 0; j < face.mNumIndices; j++) {
				indexData[offset + j] = static_cast<uint16_t>(face.mIndices[j]);
			}
			offset += face.mNumIndices;
		}

		memcpy(indexByteData.data(), indexData.data(), sizeof(uint16_t) * indexNum);
		addData(meshByteData, indexByteData, 2);
	}
	else {
		mesh.indexType = VK_INDEX_TYPE_UINT32;

		uint32_t offset = 0;
		for (uint32_t i = 0; i < faceNum; i++) {
			aiFace& face = meshData->mFaces[i];
			std::memcpy(
				indexByteData.data() + offset * sizeof(uint32_t),
				face.mIndices,
				sizeof(uint32_t) * face.mNumIndices
			);
			offset += face.mNumIndices;
		}

		addData(meshByteData, indexByteData, 4);
	}
	mesh.triMesh.indices = {
		.offset = 0,
		.count = indexNum,
		.byteStride = mesh.indexType == VK_INDEX_TYPE_UINT16 ? 2u : 4u
	};

	uint32_t vertexNum = meshData->mNumVertices;
	if (true) {
		mesh.triMesh.positions = {
			.offset = uint32_t(meshByteData.size()),
			.count = vertexNum,
			.byteStride = sizeof(glm::vec3)
		};
		std::vector<float> posData; posData.reserve(vertexNum * sizeof(glm::vec3));
		for (uint32_t i = 0; i < vertexNum; i++) {
			posData.emplace_back(meshData->mVertices[i].x);
			posData.emplace_back(meshData->mVertices[i].y);
			posData.emplace_back(meshData->mVertices[i].z);
		}
		std::vector<uint8_t> posByteData(sizeof(float) * posData.size());
		std::memcpy(posByteData.data(), posData.data(), sizeof(float) * posData.size());
		uint32_t padding = addData(meshByteData, posByteData, sizeof(glm::vec3));
		mesh.triMesh.positions.offset += padding;
	}
	if (meshData->HasNormals()) {
		mesh.triMesh.normals = {
			.offset = uint32_t(meshByteData.size()),
			.count = vertexNum,
			.byteStride = sizeof(glm::vec3)
		};
		std::vector<float> normalData;
		normalData.reserve(vertexNum * sizeof(glm::vec3));
		for (uint32_t i = 0; i < vertexNum; i++) {
			normalData.emplace_back(meshData->mNormals[i].x);
			normalData.emplace_back(meshData->mNormals[i].y);
			normalData.emplace_back(meshData->mNormals[i].z);
		}

		std::vector<uint8_t> normalByteData(sizeof(float) * normalData.size());
		std::memcpy(normalByteData.data(), normalData.data(), sizeof(float) * normalData.size());
		uint32_t padding = addData(meshByteData, normalByteData, sizeof(glm::vec3));
		mesh.triMesh.normals.offset += padding;
	}	
	if (meshData->mTextureCoords[0]) {
		mesh.triMesh.texCoords = {
			.offset = uint32_t(meshByteData.size()),
			.count = vertexNum,
			.byteStride = sizeof(glm::vec2)
		};
		std::vector<float> texCoordData;
		texCoordData.reserve(vertexNum * sizeof(glm::vec2));
		for (uint32_t i = 0; i < vertexNum; i++) {
			texCoordData.emplace_back(meshData->mTextureCoords[0][i].x);
			texCoordData.emplace_back(meshData->mTextureCoords[0][i].y);
		}

		std::vector<uint8_t> texCoordByteData(sizeof(float) * texCoordData.size());
		std::memcpy(texCoordByteData.data(), texCoordData.data(), sizeof(float) * texCoordData.size());
		uint32_t padding = addData(meshByteData, texCoordByteData, sizeof(glm::vec2));
		mesh.triMesh.texCoords.offset += padding;
	}
	if (meshData->HasTangentsAndBitangents()) {
		mesh.triMesh.tangents = {
			.offset = uint32_t(meshByteData.size()),
			.count = vertexNum,
			.byteStride = sizeof(glm::vec4)
		};
		std::vector<float> tangentData;
		tangentData.reserve(vertexNum * sizeof(glm::vec4));
		for (uint32_t i = 0; i < vertexNum; i++) {
			glm::vec3 T(meshData->mTangents[i].x, meshData->mTangents[i].y, meshData->mTangents[i].z);
			glm::vec3 B(meshData->mBitangents[i].x, meshData->mBitangents[i].y, meshData->mBitangents[i].z);
			glm::vec3 N(meshData->mNormals[i].x, meshData->mNormals[i].y, meshData->mNormals[i].z);

			T = glm::normalize(T);
			N = glm::normalize(N);
			float handed = (glm::dot(glm::cross(N, T), B) < 0.0f) ? -1.0f : 1.0f;

			tangentData.emplace_back(meshData->mTangents[i].x);
			tangentData.emplace_back(meshData->mTangents[i].y);
			tangentData.emplace_back(meshData->mTangents[i].z);
			tangentData.emplace_back(handed);
		}

		std::vector<uint8_t> tangentByteData(sizeof(float)* tangentData.size());
		std::memcpy(tangentByteData.data(), tangentData.data(), sizeof(float)* tangentData.size());
		uint32_t padding = addData(meshByteData, tangentByteData, sizeof(glm::vec4));
		mesh.triMesh.tangents.offset += padding;
	}
	
	nvvk::Buffer bObjData;
	uint32_t bufferIndex{};
	{
		nvvk::ResourceAllocator* allocator = Application::stagingUploader.getResourceAllocator();
		NVVK_CHECK(allocator->createBuffer(bObjData, std::span<const unsigned char>(meshByteData).size_bytes(),
			VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT
			| VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR));
		NVVK_CHECK(Application::stagingUploader.appendBuffer(bObjData, 0, std::span<const unsigned char>(meshByteData)));
		NVVK_DBG_NAME(bObjData.buffer);
		bufferIndex = static_cast<uint32_t>(Application::sceneResource.bDatas.size());
		Application::sceneResource.bDatas.push_back(bObjData);
	}

	Application::sceneResource.meshes.emplace_back(mesh);
	Application::sceneResource.meshToBufferIndex.push_back(bufferIndex);
}
void FzbRenderer::Mesh::processNode(aiNode* node, const aiScene* sceneData) {
	std::vector<shaderio::Mesh> meshes;
	for (uint32_t i = 0; i < node->mNumMeshes; i++) {
		aiMesh* meshData = sceneData->mMeshes[node->mMeshes[i]];
		processMesh(meshData, sceneData);
	}

	for (uint32_t i = 0; i < node->mNumChildren; i++) processNode(node->mChildren[i], sceneData);
}
void FzbRenderer::Mesh::loadObjData(std::filesystem::path meshPath) {
	Assimp::Importer import;
	uint32_t needs = aiProcess_Triangulate;// |
		//(vertexFormat.useTexCoord ? aiProcess_FlipUVs : aiPostProcessSteps(0u)) |
		//(vertexFormat.useNormal ? aiProcess_GenSmoothNormals : aiPostProcessSteps(0u)) |
		//(vertexFormat.useTangent ? aiProcess_CalcTangentSpace : aiPostProcessSteps(0u));
	const aiScene* sceneData = import.ReadFile(meshPath.string(), needs);

	if (!sceneData || sceneData->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !sceneData->mRootNode) 
		LOGW(import.GetErrorString());

	processNode(sceneData->mRootNode, sceneData);
}
