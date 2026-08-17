#include <nvutils/timers.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/resource_allocator.hpp>
#include <common/Application/Application.h>
#include <glm/gtc/type_ptr.hpp>
#include <algorithm>
#include <memory>
#include <future>
#include <thread>
#include <vector>
#include "./Mesh.h"
#include <common/Material/Material.h>

//#define USE_DEFAULT_MATERIAL

shaderio::AABB FzbRenderer::MeshInfo::getAABB(glm::mat4 transformMatrix, bool isStatic) {
	if (isStatic) {
		glm::vec3 maximum = { FLT_MAX, FLT_MAX, FLT_MAX };
		if (aabb.minimum != maximum && aabb.maximum != -maximum) return aabb;
	}

	Scene& sceneRsource = Application::sceneResource;
	std::vector<uint8_t>& meshByteData = sceneRsource.meshSets[sceneRsource.getMeshSetIndex(meshIndex)].meshByteData;
	const auto& positions = mesh.triMesh.positions;
	const glm::vec3* vertexData = reinterpret_cast<const glm::vec3*>(meshByteData.data() + positions.offset);
	const uint32_t vertexCount = positions.count;

	aabb = { { FLT_MAX, FLT_MAX, FLT_MAX }, { -FLT_MAX, -FLT_MAX, -FLT_MAX } };
	if (vertexCount == 0) return aabb;

	auto mergeAABB = [](shaderio::AABB& dst, const shaderio::AABB& src) {
		dst.minimum.x = std::min(src.minimum.x, dst.minimum.x);
		dst.minimum.y = std::min(src.minimum.y, dst.minimum.y);
		dst.minimum.z = std::min(src.minimum.z, dst.minimum.z);
		dst.maximum.x = std::max(src.maximum.x, dst.maximum.x);
		dst.maximum.y = std::max(src.maximum.y, dst.maximum.y);
		dst.maximum.z = std::max(src.maximum.z, dst.maximum.z);
	};

	auto computeRangeAABB = [vertexData, transformMatrix](uint32_t begin, uint32_t end) {
		shaderio::AABB localAABB = { { FLT_MAX, FLT_MAX, FLT_MAX }, { -FLT_MAX, -FLT_MAX, -FLT_MAX } };
		for (uint32_t i = begin; i < end; ++i) {
			const glm::vec3& pos = vertexData[i];
			glm::vec3 pos_transform = transformMatrix * glm::vec4(pos, 1.0f);
			localAABB.minimum.x = std::min(pos_transform.x, localAABB.minimum.x);
			localAABB.minimum.y = std::min(pos_transform.y, localAABB.minimum.y);
			localAABB.minimum.z = std::min(pos_transform.z, localAABB.minimum.z);
			localAABB.maximum.x = std::max(pos_transform.x, localAABB.maximum.x);
			localAABB.maximum.y = std::max(pos_transform.y, localAABB.maximum.y);
			localAABB.maximum.z = std::max(pos_transform.z, localAABB.maximum.z);
		}
		return localAABB;
	};

	constexpr uint32_t minVerticesPerTask = 4096;
	uint32_t taskCount = std::thread::hardware_concurrency();
	if (taskCount == 0) taskCount = 4;
	taskCount = std::min(taskCount, (vertexCount + minVerticesPerTask - 1) / minVerticesPerTask);
	taskCount = std::max(taskCount, 1u);

	if (taskCount == 1) {
		aabb = computeRangeAABB(0, vertexCount);
		return aabb;
	}

	std::vector<std::future<shaderio::AABB>> futures;
	futures.reserve(taskCount);
	const uint32_t chunkSize = (vertexCount + taskCount - 1) / taskCount;
	for (uint32_t taskIndex = 0; taskIndex < taskCount; ++taskIndex) {
		const uint32_t begin = taskIndex * chunkSize;
		const uint32_t end = std::min(begin + chunkSize, vertexCount);
		if (begin >= end) break;
		futures.push_back(std::async(std::launch::async, computeRangeAABB, begin, end));
	}

	for (auto& future : futures) {
		mergeAABB(aabb, future.get());
	}

	return aabb;
}
//-----------------------------------------------------MeshSet---------------------------------------------------
FzbRenderer::MeshSet::MeshSet(std::string meshID, std::string meshType, std::filesystem::path meshPath)
{
	this->meshID = meshID;

	Scene& scene = Application::sceneResource;
	if (meshType == "gltf" || meshType == "glb") {
		tinygltf::Model gltfModel = nvsamples::loadGltfResources(nvutils::findFile(meshPath, { meshPath }));
		loadGltfData(gltfModel);
	}
	else if (meshType == "obj") {
		loadObjData(meshPath);
	}
	else if (meshType == "plane") {
		nvutils::PrimitiveMesh primitive = FzbRenderer::MeshSet::createPlane(1, 1.0f, 1.0f);
		createCustomMeshSet(meshID, primitive);
	}
	else if (meshType == "cube") {
		nvutils::PrimitiveMesh primitive = FzbRenderer::MeshSet::createCube(true);
		createCustomMeshSet(meshID, primitive);
	}

	aabb.minimum = { FLT_MAX, FLT_MAX, FLT_MAX };
	aabb.maximum = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
}

/*
gltf��material���ý���-�ֲڶ�ģ�ͣ���˺���֪���ǲ���diffuse�ģ����Ǳ��ʲô���ʵ�
���ԣ����ǰ��գ�����ֲڶ�ΪΪ0�����ǹ⻬�ģ���֮Ϊ�ֲڶ�
���alphaModeΪOPAQUE,���ǵ��壻��֮Ϊ�����
���Ϊ����ʣ���ôeta��Ϊ͸����
*/
shaderio::BSDFMaterial loadGltfMaterial(const tinygltf::Material& gltfMaterial) {
	glm::vec3 albedo = glm::make_vec3(gltfMaterial.pbrMetallicRoughness.baseColorFactor.data());
	return {
		.type = gltfMaterial.alphaMode == "OPAQUE" ? shaderio::MaterialType::Conductor : shaderio::MaterialType::Dielectric,
		.albedo = albedo,
		.emissive = glm::make_vec3(gltfMaterial.emissiveFactor.data()),
		.eta = (glm::vec3(1.0f) + albedo) / (glm::vec3(1.0f) - albedo),
		.roughness = float(gltfMaterial.pbrMetallicRoughness.roughnessFactor),
	};
}
void FzbRenderer::MeshSet::loadGltfData(const tinygltf::Model& model, bool importInstance) {
	SCOPED_TIMER(__FUNCTION__);

	auto getElementByteSize = [](int type) -> uint32_t {	//��С���ݵ�Ԫ�Ĵ�С
		return  type == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ? 2U :
			type == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT ? 4U :
			type == TINYGLTF_COMPONENT_TYPE_FLOAT ? 4U :
			0U;
		};
	auto getTypeSize = [](int type) -> uint32_t {			//��������һ�����������ݵ�Ԫ����
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
		const tinygltf::Accessor& acc = model.accessors[primitive.attributes.at(name)];	//Accessor֪����ν��һ��bufferView
		const tinygltf::BufferView& bv = model.bufferViews[acc.bufferView];				//bufferView֪��buffer��ĳһ�ε���Ϣ
		assert((acc.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) && "Should be floats");
		attr = {
			//bv.byteOffset�൱����buffer�е����;acc.byteOffset���涥��ĳ�������ڶ��������е�ƫ�ƣ��ȷ�˵normal��offsetΪ3*4=12
			.offset = uint32_t(bv.byteOffset + acc.byteOffset),
			.count = uint32_t(acc.count),
			.byteStride = uint32_t(bv.byteStride ? uint32_t(bv.byteStride) : getTypeSize(acc.type) * getElementByteSize(acc.componentType)),
		};
		};

	meshByteData = model.buffers[0].data;
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

		extractAttribute("POSITION", mesh.triMesh.positions, primitive);
		extractAttribute("NORMAL", mesh.triMesh.normals, primitive);
		extractAttribute("COLOR_0", mesh.triMesh.colorVert, primitive);
		extractAttribute("TEXCOORD_0", mesh.triMesh.texCoords, primitive);
		extractAttribute("TANGENT", mesh.triMesh.tangents, primitive);

		std::string materialID = "defaultMaterial";
		shaderio::BSDFMaterial material = FzbRenderer::defaultMaterial;
		if (primitive.material >= 0) {
			const tinygltf::Material& gltfMaterial = model.materials[primitive.material];
			materialID = gltfMaterial.name;
			material = loadGltfMaterial(gltfMaterial);
		}

		FzbRenderer::MeshInfo childMesh = {
			.meshID = meshID + model.meshes[meshIdx].name,
			.mesh = mesh,
			.materialID = materialID,
			.material = material
		};
		childMeshInfos.push_back(childMesh);
	}

	/*
	if (importInstance) {
		std::function<void(const tinygltf::Node&, const glm::mat4&)> processNode = [&](const tinygltf::Node& node, const glm::mat4& parentTransform) {
			glm::mat4 nodeTransform = parentTransform;	//��ǰnode�ı任�����븸node�ı任����
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
	*/
};

uint32_t addData(std::vector<uint8_t>& data, std::vector<uint8_t>& newData, uint32_t alignment) {
	uint32_t dataSize = data.size();
	uint32_t padding = (alignment - dataSize % alignment) % alignment;
	data.reserve(dataSize + padding + newData.size());
	
	if (padding > 0) {
		std::vector<uint8_t> paddingData(padding);
		data.insert(data.end(), paddingData.begin(), paddingData.end());
	}
	data.insert(data.end(), newData.begin(), newData.end());

	return padding;
}
shaderio::BSDFMaterial loadMtlMaterial(aiMaterial* mtlMaterial) {
	shaderio::BSDFMaterial material;

	int illumModel = 0;
	mtlMaterial->Get("$mat.illum", 0, 0, illumModel);
	material.type = shaderio::MaterialType(illumModel);

	aiColor3D color;
	mtlMaterial->Get(AI_MATKEY_COLOR_DIFFUSE, color);
	material.albedo = glm::vec3(color.r, color.g, color.b);

	mtlMaterial->Get(AI_MATKEY_COLOR_EMISSIVE, color);
	material.emissive = glm::vec3(color.r, color.g, color.b);

	float roughness = 0.1f;
	mtlMaterial->Get(AI_MATKEY_OPACITY, roughness);
	material.roughness = roughness;

	float eta = 1.5f;
	mtlMaterial->Get(AI_MATKEY_REFRACTI, eta);
	material.eta = glm::vec3(1.0f / eta);

	return material;
}

// ---------------------------------------------------------------
// Parallel OBJ loading helpers
// ---------------------------------------------------------------
namespace {

	// Holds all processed data for a single mesh — built in parallel, merged sequentially.
	struct ProcessedMeshData {
		std::vector<uint8_t> indexByteData;
		VkIndexType           indexType = VK_INDEX_TYPE_UINT32;
		uint32_t              indexCount = 0;

		std::vector<uint8_t> posByteData;
		std::vector<uint8_t> normalByteData;
		std::vector<uint8_t> texCoordByteData;
		std::vector<uint8_t> tangentByteData;

		uint32_t vertexNum = 0;

		std::string              materialID = "defaultMaterial";
		shaderio::BSDFMaterial   material;
		std::string              meshName;
	};

	// Flat work item: (index into aiScene::mMeshes, owning node name)
	struct MeshWorkItem {
		uint32_t    meshIndex;
		std::string nodeName;
	};

	// Depth-first collection of all mesh indices from the Assimp node tree.
	void collectMeshIndices(aiNode* node, std::vector<MeshWorkItem>& outItems) {
		for (uint32_t i = 0; i < node->mNumMeshes; i++) {
			outItems.push_back({ node->mMeshes[i], std::string(node->mName.C_Str()) });
		}
		for (uint32_t i = 0; i < node->mNumChildren; i++) {
			collectMeshIndices(node->mChildren[i], outItems);
		}
	}

	// Process one aiMesh entirely into a local ProcessedMeshData — no shared state.
	// This is the function that runs in parallel for every mesh.
	ProcessedMeshData processMeshLocal(const aiMesh* meshData, const aiScene* sceneData)
	{
		ProcessedMeshData out;

		uint32_t faceNum = meshData->mNumFaces;
		uint32_t indexNum = 0;
		for (uint32_t i = 0; i < faceNum; ++i)
			indexNum += meshData->mFaces[i].mNumIndices;

		uint32_t maxIndex = 0;
		for (uint32_t i = 0; i < faceNum; i++) {
			const aiFace& face = meshData->mFaces[i];
			for (uint32_t j = 0; j < face.mNumIndices; j++) {
				if (face.mIndices[j] > maxIndex)
					maxIndex = face.mIndices[j];
			}
		}

		// ---- indices ----
		if (maxIndex <= 0xFFFF) {
			out.indexType = VK_INDEX_TYPE_UINT16;
			std::vector<uint16_t> indexData;
			indexData.resize(indexNum);
			uint32_t offset = 0;
			for (uint32_t i = 0; i < faceNum; i++) {
				const aiFace& face = meshData->mFaces[i];
				for (uint32_t j = 0; j < face.mNumIndices; j++) {
					indexData[offset + j] = static_cast<uint16_t>(face.mIndices[j]);
				}
				offset += face.mNumIndices;
			}
			out.indexByteData.resize(indexNum * sizeof(uint16_t));
			std::memcpy(out.indexByteData.data(), indexData.data(), sizeof(uint16_t) * indexNum);
		}
		else {
			out.indexType = VK_INDEX_TYPE_UINT32;
			out.indexByteData.resize(indexNum * sizeof(uint32_t));
			uint32_t offset = 0;
			for (uint32_t i = 0; i < faceNum; i++) {
				const aiFace& face = meshData->mFaces[i];
				std::memcpy(
					out.indexByteData.data() + offset * sizeof(uint32_t),
					face.mIndices,
					sizeof(uint32_t) * face.mNumIndices
				);
				offset += face.mNumIndices;
			}
		}
		out.indexCount = indexNum;

		// ---- positions ----
		uint32_t vertexNum = meshData->mNumVertices;
		out.vertexNum = vertexNum;
		if (meshData->HasPositions()) {
			out.posByteData.resize(vertexNum * sizeof(glm::vec3));
			float* dst = reinterpret_cast<float*>(out.posByteData.data());
			for (uint32_t i = 0; i < vertexNum; i++) {
				dst[i * 3 + 0] = meshData->mVertices[i].x;
				dst[i * 3 + 1] = meshData->mVertices[i].y;
				dst[i * 3 + 2] = meshData->mVertices[i].z;
			}
		}

		// ---- normals ----
		if (meshData->HasNormals()) {
			out.normalByteData.resize(vertexNum * sizeof(glm::vec3));
			float* dst = reinterpret_cast<float*>(out.normalByteData.data());
			for (uint32_t i = 0; i < vertexNum; i++) {
				dst[i * 3 + 0] = meshData->mNormals[i].x;
				dst[i * 3 + 1] = meshData->mNormals[i].y;
				dst[i * 3 + 2] = meshData->mNormals[i].z;
			}
		}

		// ---- texCoords ----
		if (meshData->mTextureCoords[0]) {
			out.texCoordByteData.resize(vertexNum * sizeof(glm::vec2));
			float* dst = reinterpret_cast<float*>(out.texCoordByteData.data());
			for (uint32_t i = 0; i < vertexNum; i++) {
				dst[i * 2 + 0] = meshData->mTextureCoords[0][i].x;
				dst[i * 2 + 1] = meshData->mTextureCoords[0][i].y;
			}
		}

		// ---- tangents ----
		if (meshData->HasTangentsAndBitangents()) {
			out.tangentByteData.resize(vertexNum * sizeof(glm::vec4));
			float* dst = reinterpret_cast<float*>(out.tangentByteData.data());
			for (uint32_t i = 0; i < vertexNum; i++) {
				glm::vec3 T(meshData->mTangents[i].x, meshData->mTangents[i].y, meshData->mTangents[i].z);
				glm::vec3 B(meshData->mBitangents[i].x, meshData->mBitangents[i].y, meshData->mBitangents[i].z);
				glm::vec3 N(meshData->mNormals[i].x, meshData->mNormals[i].y, meshData->mNormals[i].z);
				T = glm::normalize(T);
				N = glm::normalize(N);
				float handed = (glm::dot(glm::cross(N, T), B) < 0.0f) ? -1.0f : 1.0f;
				dst[i * 4 + 0] = meshData->mTangents[i].x;
				dst[i * 4 + 1] = meshData->mTangents[i].y;
				dst[i * 4 + 2] = meshData->mTangents[i].z;
				dst[i * 4 + 3] = handed;
			}
		}

		// ---- material ----
		out.meshName = std::string(meshData->mName.C_Str());
		out.material = FzbRenderer::defaultMaterial;
		out.materialID = "defaultMaterial";
#ifndef USE_DEFAULT_MATERIAL
		if (sceneData->mNumMaterials > 1) {
			aiMaterial* mtlMaterial = sceneData->mMaterials[meshData->mMaterialIndex];
			out.materialID = std::string(mtlMaterial->GetName().data);
			out.material = loadMtlMaterial(mtlMaterial);
		}
#endif
		return out;
	}

} // anonymous namespace
void FzbRenderer::MeshSet::processMesh(aiMesh* meshData, const aiScene* sceneData) {
	shaderio::Mesh mesh{};

	uint32_t padding = 0;

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
	
	mesh.triMesh.indices = {
		.offset = uint32_t(meshByteData.size()),
		.count = indexNum
	};
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

		indexByteData.resize(indexNum * 2);
		memcpy(indexByteData.data(), indexData.data(), sizeof(uint16_t) * indexNum);
		padding = addData(meshByteData, indexByteData, 2);
	}
	else {
		mesh.indexType = VK_INDEX_TYPE_UINT32;

		indexByteData.resize(indexNum * 4);
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

		padding = addData(meshByteData, indexByteData, 4);
	}
	mesh.triMesh.indices.offset += padding;
	mesh.triMesh.indices.byteStride = mesh.indexType == VK_INDEX_TYPE_UINT16 ? 2u : 4u;

	uint32_t vertexNum = meshData->mNumVertices;
	if (meshData->HasPositions()) {
		mesh.triMesh.positions = {
			.offset = uint32_t(meshByteData.size()),
			.count = vertexNum,
			.byteStride = sizeof(glm::vec3)
		};
		std::vector<float> posData; posData.reserve(vertexNum * 3);
		for (uint32_t i = 0; i < vertexNum; i++) {
			posData.emplace_back(meshData->mVertices[i].x);
			posData.emplace_back(meshData->mVertices[i].y);
			posData.emplace_back(meshData->mVertices[i].z);
		}
		std::vector<uint8_t> posByteData(sizeof(float) * posData.size());
		std::memcpy(posByteData.data(), posData.data(), sizeof(float) * posData.size());
		padding = addData(meshByteData, posByteData, sizeof(glm::vec3));
		mesh.triMesh.positions.offset += padding;
	}
	if (meshData->HasNormals()) {
		mesh.triMesh.normals = {
			.offset = uint32_t(meshByteData.size()),
			.count = vertexNum,
			.byteStride = sizeof(glm::vec3)
		};
		std::vector<float> normalData;
		normalData.reserve(vertexNum * 3);
		for (uint32_t i = 0; i < vertexNum; i++) {
			normalData.emplace_back(meshData->mNormals[i].x);
			normalData.emplace_back(meshData->mNormals[i].y);
			normalData.emplace_back(meshData->mNormals[i].z);
		}
	
		std::vector<uint8_t> normalByteData(sizeof(float) * normalData.size());
		std::memcpy(normalByteData.data(), normalData.data(), sizeof(float) * normalData.size());
		padding = addData(meshByteData, normalByteData, sizeof(glm::vec3));
		mesh.triMesh.normals.offset += padding;
	}	
	if (meshData->mTextureCoords[0]) {
		mesh.triMesh.texCoords = {
			.offset = uint32_t(meshByteData.size()),
			.count = vertexNum,
			.byteStride = sizeof(glm::vec2)
		};
		std::vector<float> texCoordData;
		texCoordData.reserve(vertexNum * 2);
		for (uint32_t i = 0; i < vertexNum; i++) {
			texCoordData.emplace_back(meshData->mTextureCoords[0][i].x);
			texCoordData.emplace_back(meshData->mTextureCoords[0][i].y);
		}

		std::vector<uint8_t> texCoordByteData(sizeof(float) * texCoordData.size());
		std::memcpy(texCoordByteData.data(), texCoordData.data(), sizeof(float) * texCoordData.size());
		padding = addData(meshByteData, texCoordByteData, sizeof(glm::vec2));
		mesh.triMesh.texCoords.offset += padding;
	}
	if (meshData->HasTangentsAndBitangents()) {
		mesh.triMesh.tangents = {
			.offset = uint32_t(meshByteData.size()),
			.count = vertexNum,
			.byteStride = sizeof(glm::vec4)
		};
		std::vector<float> tangentData;
		tangentData.reserve(vertexNum * 4);
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
		padding = addData(meshByteData, tangentByteData, sizeof(glm::vec4));
		mesh.triMesh.tangents.offset += padding;
	}
	
	std::string materialID = "defaultMaterial";
	shaderio::BSDFMaterial material = FzbRenderer::defaultMaterial;
	#ifndef USE_DEFAULT_MATERIAL
	if (sceneData->mNumMaterials > 1) {		
		aiMaterial* mtlMaterial = sceneData->mMaterials[meshData->mMaterialIndex];
		materialID = std::string(mtlMaterial->GetName().data);
		material = loadMtlMaterial(mtlMaterial);
	}
	#endif

	MeshInfo childMeshInfo = {
		.meshID = meshID + meshData->mName.C_Str(),
		.mesh = mesh,
		.materialID = materialID,
		.material = material
	};
	childMeshInfos.emplace_back(childMeshInfo);
}
void FzbRenderer::MeshSet::processNode(aiNode* node, const aiScene* sceneData) {
	std::vector<shaderio::Mesh> meshes;
	for (uint32_t i = 0; i < node->mNumMeshes; i++) {
		aiMesh* meshData = sceneData->mMeshes[node->mMeshes[i]];
		processMesh(meshData, sceneData);
	}

	for (uint32_t i = 0; i < node->mNumChildren; i++) processNode(node->mChildren[i], sceneData);
}
void FzbRenderer::MeshSet::loadObjData(std::filesystem::path meshPath) {
	SCOPED_TIMER(__FUNCTION__);

	Assimp::Importer import;
	uint32_t needs = aiProcess_Triangulate | aiProcess_GenSmoothNormals;
	const aiScene* sceneData = import.ReadFile(meshPath.string(), needs);

	if (!sceneData || sceneData->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !sceneData->mRootNode) {
		LOGW(import.GetErrorString());
		return;
	}

	// --- Phase 1: collect all mesh indices (preserves depth-first order) ---
	std::vector<MeshWorkItem> workItems;
	collectMeshIndices(sceneData->mRootNode, workItems);

	if (workItems.empty()) return;

	// Pre-reserve meshByteData: sum up all per-mesh buffer sizes plus alignment slop.
	{
		size_t totalBytes = 0;
		for (const auto& item : workItems) {
			const aiMesh* md = sceneData->mMeshes[item.meshIndex];
			// Rough estimate — accurate enough for reserve
			totalBytes += md->mNumFaces * 3 * sizeof(uint32_t);   // indices (worst case)
			totalBytes += md->mNumVertices * sizeof(glm::vec3);   // positions
			if (md->HasNormals())         totalBytes += md->mNumVertices * sizeof(glm::vec3);
			if (md->mTextureCoords[0])    totalBytes += md->mNumVertices * sizeof(glm::vec2);
			if (md->HasTangentsAndBitangents()) totalBytes += md->mNumVertices * sizeof(glm::vec4);
			totalBytes += 16;  // padding ceiling per buffer (worst case per attribute)
		}
		meshByteData.reserve(totalBytes);
	}

	// Launch parallel work
	std::vector<std::future<ProcessedMeshData>> futures;
	futures.reserve(workItems.size());
	for (const auto& item : workItems) {
		const aiMesh* meshData = sceneData->mMeshes[item.meshIndex];
		futures.push_back(std::async(
			std::launch::async,
			processMeshLocal,
			meshData,
			sceneData
		));
	}

	// --- Phase 3: sequential merge (preserves ordering, no locks needed) ---
	for (size_t i = 0; i < workItems.size(); ++i) {
		ProcessedMeshData result = futures[i].get();
		uint32_t padding = 0;

		shaderio::Mesh mesh{};

		// Indices
		if (!result.indexByteData.empty()) {
			mesh.triMesh.indices.offset = uint32_t(meshByteData.size());
			mesh.triMesh.indices.count = result.indexCount;
			mesh.triMesh.indices.byteStride = (result.indexType == VK_INDEX_TYPE_UINT16) ? 2u : 4u;
			mesh.indexType = result.indexType;

			uint32_t alignment = (result.indexType == VK_INDEX_TYPE_UINT16) ? 2u : 4u;
			uint32_t pad = (alignment - meshByteData.size() % alignment) % alignment;
			if (pad > 0) meshByteData.insert(meshByteData.end(), pad, 0);
			mesh.triMesh.indices.offset += pad;
			meshByteData.insert(meshByteData.end(), result.indexByteData.begin(), result.indexByteData.end());
		}

		// Positions
		if (!result.posByteData.empty()) {
			mesh.triMesh.positions.offset = uint32_t(meshByteData.size());
			mesh.triMesh.positions.count = result.vertexNum;
			mesh.triMesh.positions.byteStride = sizeof(glm::vec3);

			uint32_t pad = (sizeof(glm::vec3) - meshByteData.size() % sizeof(glm::vec3)) % sizeof(glm::vec3);
			if (pad > 0) meshByteData.insert(meshByteData.end(), pad, 0);
			mesh.triMesh.positions.offset += pad;
			meshByteData.insert(meshByteData.end(), result.posByteData.begin(), result.posByteData.end());
		}

		// Normals
		if (!result.normalByteData.empty()) {
			mesh.triMesh.normals.offset = uint32_t(meshByteData.size());
			mesh.triMesh.normals.count = result.vertexNum;
			mesh.triMesh.normals.byteStride = sizeof(glm::vec3);

			uint32_t pad = (sizeof(glm::vec3) - meshByteData.size() % sizeof(glm::vec3)) % sizeof(glm::vec3);
			if (pad > 0) meshByteData.insert(meshByteData.end(), pad, 0);
			mesh.triMesh.normals.offset += pad;
			meshByteData.insert(meshByteData.end(), result.normalByteData.begin(), result.normalByteData.end());
		}

		// TexCoords
		if (!result.texCoordByteData.empty()) {
			mesh.triMesh.texCoords.offset = uint32_t(meshByteData.size());
			mesh.triMesh.texCoords.count = result.vertexNum;
			mesh.triMesh.texCoords.byteStride = sizeof(glm::vec2);

			uint32_t pad = (sizeof(glm::vec2) - meshByteData.size() % sizeof(glm::vec2)) % sizeof(glm::vec2);
			if (pad > 0) meshByteData.insert(meshByteData.end(), pad, 0);
			mesh.triMesh.texCoords.offset += pad;
			meshByteData.insert(meshByteData.end(), result.texCoordByteData.begin(), result.texCoordByteData.end());
		}

		// Tangents
		if (!result.tangentByteData.empty()) {
			mesh.triMesh.tangents.offset = uint32_t(meshByteData.size());
			mesh.triMesh.tangents.count = result.vertexNum;
			mesh.triMesh.tangents.byteStride = sizeof(glm::vec4);

			uint32_t pad = (sizeof(glm::vec4) - meshByteData.size() % sizeof(glm::vec4)) % sizeof(glm::vec4);
			if (pad > 0) meshByteData.insert(meshByteData.end(), pad, 0);
			mesh.triMesh.tangents.offset += pad;
			meshByteData.insert(meshByteData.end(), result.tangentByteData.begin(), result.tangentByteData.end());
		}

		MeshInfo childMeshInfo = {
			.meshID = meshID + result.meshName,
			.mesh = mesh,
			.materialID = result.materialID,
			.material = result.material
		};
		childMeshInfos.emplace_back(childMeshInfo);
	}
}

void FzbRenderer::MeshSet::createCustomMeshSet(std::string meshID, nvutils::PrimitiveMesh primitiveMesh) {
	this->meshID = meshID;
	shaderio::Mesh mesh{};

	uint32_t indexCount = primitiveMesh.triangles.size() * 3;
	uint32_t maxIndex = 0;
	for (const auto& tri : primitiveMesh.triangles)
		maxIndex = std::max(maxIndex, std::max(tri.indices.x, std::max(tri.indices.y, tri.indices.z)));
	if (maxIndex <= 0xFFFF) {
		mesh.indexType = VK_INDEX_TYPE_UINT16;
		std::vector<uint16_t> indexData;
		indexData.reserve(indexCount);
		for (const auto& tri : primitiveMesh.triangles) {
			indexData.emplace_back(static_cast<uint16_t>(tri.indices.x));
			indexData.emplace_back(static_cast<uint16_t>(tri.indices.y));
			indexData.emplace_back(static_cast<uint16_t>(tri.indices.z));
		}
		meshByteData.resize(sizeof(uint16_t) * indexCount);
		memcpy(meshByteData.data(), indexData.data(), sizeof(uint16_t) * indexCount);
	}
	else {
		mesh.indexType = VK_INDEX_TYPE_UINT32;
		memcpy(meshByteData.data(), primitiveMesh.triangles.data(), sizeof(uint32_t) * indexCount);
	}
	mesh.triMesh.indices = {
		.offset = 0,
		.count = indexCount,
		.byteStride = mesh.indexType == VK_INDEX_TYPE_UINT16 ? 2u : 4u
	};

	uint32_t vertexNum = primitiveMesh.vertices.size();
	mesh.triMesh.positions = {
		.offset = uint32_t(meshByteData.size()),
		.count = vertexNum,
		.byteStride = sizeof(glm::vec3)
	};
	std::vector<float> posData; posData.reserve(vertexNum * sizeof(glm::vec3));
	for (uint32_t i = 0; i < vertexNum; i++) {
		posData.emplace_back(primitiveMesh.vertices[i].pos.x);
		posData.emplace_back(primitiveMesh.vertices[i].pos.y);
		posData.emplace_back(primitiveMesh.vertices[i].pos.z);
	}
	std::vector<uint8_t> posByteData(sizeof(float) * posData.size());
	std::memcpy(posByteData.data(), posData.data(), sizeof(float) * posData.size());
	uint32_t padding = addData(meshByteData, posByteData, sizeof(glm::vec3));
	mesh.triMesh.positions.offset += padding;

	mesh.triMesh.normals = {
		.offset = uint32_t(meshByteData.size()),
		.count = vertexNum,
		.byteStride = sizeof(glm::vec3)
	};
	std::vector<float> normalData; normalData.reserve(vertexNum * sizeof(glm::vec3));
	for (uint32_t i = 0; i < vertexNum; i++) {
		normalData.emplace_back(primitiveMesh.vertices[i].nrm.x);
		normalData.emplace_back(primitiveMesh.vertices[i].nrm.y);
		normalData.emplace_back(primitiveMesh.vertices[i].nrm.z);
	}
	std::vector<uint8_t> normalByteData(sizeof(float) * normalData.size());
	std::memcpy(normalByteData.data(), normalData.data(), sizeof(float) * normalData.size());
	padding = addData(meshByteData, normalByteData, sizeof(glm::vec3));
	mesh.triMesh.normals.offset += padding;

	MeshInfo childMeshInfo = {
		.meshID = meshID,
		.mesh = mesh,
	};
	childMeshInfos.emplace_back(childMeshInfo);
}
FzbRenderer::MeshSet::MeshSet(std::string meshID, nvutils::PrimitiveMesh primitiveMesh)
{
	createCustomMeshSet(meshID, primitiveMesh);
}

nvvk::Buffer FzbRenderer::MeshSet::createMeshDataBuffer() {
	nvvk::Buffer bData;
	nvvk::ResourceAllocator* allocator = Application::stagingUploader.getResourceAllocator();
	NVVK_CHECK(allocator->createBuffer(bData, std::span<const unsigned char>(meshByteData).size_bytes(),
		VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT
		| VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR));
	NVVK_CHECK(Application::stagingUploader.appendBuffer(bData, 0, std::span<const unsigned char>(meshByteData)));
	NVVK_DBG_NAME(bData.buffer);
	return bData;
}

shaderio::AABB FzbRenderer::MeshSet::getAABB(glm::mat4 transformMatrix) {
	glm::vec3 maxmum = { FLT_MAX, FLT_MAX, FLT_MAX };
	if (aabb.minimum != maxmum && aabb.maximum != -maxmum) return aabb;

	for (auto& childMeshInfo : childMeshInfos) {
		shaderio::AABB childMeshAABB = childMeshInfo.getAABB(transformMatrix);

		aabb.minimum.x = std::min(childMeshAABB.minimum.x, aabb.minimum.x);
		aabb.minimum.y = std::min(childMeshAABB.minimum.y, aabb.minimum.y);
		aabb.minimum.z = std::min(childMeshAABB.minimum.z, aabb.minimum.z);
		aabb.maximum.x = std::max(childMeshAABB.maximum.x, aabb.maximum.x);
		aabb.maximum.y = std::max(childMeshAABB.maximum.y, aabb.maximum.y);
		aabb.maximum.z = std::max(childMeshAABB.maximum.z, aabb.maximum.z);
	}

	return aabb;
}
//---------------------------------------------------------------------------------------------------------
static uint32_t addPos(nvutils::PrimitiveMesh& mesh, glm::vec3 p)
{
	nvutils::PrimitiveVertex v{};
	v.pos = p;
	mesh.vertices.emplace_back(v);
	return static_cast<uint32_t>(mesh.vertices.size()) - 1;
}
static void addTriangle(nvutils::PrimitiveMesh& mesh, uint32_t a, uint32_t b, uint32_t c)
{
	mesh.triangles.push_back({ {a, b, c} });
}
static void addTriangle(nvutils::PrimitiveMesh& mesh, glm::vec3 a, glm::vec3 b, glm::vec3 c)
{
	mesh.triangles.push_back({ {addPos(mesh, a), addPos(mesh, b), addPos(mesh, c)} });
}
nvutils::PrimitiveMesh FzbRenderer::MeshSet::createPlane(int steps, float width, float height) {
	nvutils::PrimitiveMesh mesh;

	if (steps <= 0) return mesh;

	const float invSteps = 1.0f / static_cast<float>(steps);

	// ���ɶ��㣺�� (0,0,0) ��ʼ������Ϊ X(0..width)������Ϊ Y(0..depth)��Z = 0
	for (int sy = 0; sy <= steps; ++sy)
	{
		for (int sx = 0; sx <= steps; ++sx)
		{
			nvutils::PrimitiveVertex v{};

			float u = static_cast<float>(sx) * invSteps; // 0..1
			float v_t = static_cast<float>(sy) * invSteps; // 0..1

			v.pos = glm::vec3(u * width, v_t * height, 0.0f); // XY ƽ�棬��� (0,0,0)
			v.nrm = glm::vec3(0.0f, 0.0f, 1.0f);               // ָ�� +Z
			v.tex = glm::vec2(u, v_t);                        // (0,0) -> (1,1)

			mesh.vertices.emplace_back(v);
		}
	}

	// ���������������Σ���ȷ������˳���� XY ƽ����Ϊ CCW���泯 +Z��
	const int rowStride = steps + 1;
	for (int sy = 0; sy < steps; ++sy)
	{
		for (int sx = 0; sx < steps; ++sx)
		{
			int a = sx + sy * rowStride;                 // (sx, sy)
			int c = (sx + 1) + sy * rowStride;           // (sx+1, sy)
			int b = (sx + 1) + (sy + 1) * rowStride;     // (sx+1, sy+1)
			int d = sx + (sy + 1) * rowStride;           // (sx, sy+1)

			// ���������Σ� (a, c, b) �� (a, b, d) -> ��֤���� +Z��CCW��
			addTriangle(mesh, a, c, b);
			addTriangle(mesh, a, b, d);
		}
	}

	return mesh;
}
nvutils::PrimitiveMesh FzbRenderer::MeshSet::createCube(bool normal, bool texCoords,float width, float height , float depth)
{
	nvutils::PrimitiveMesh mesh;
	if (normal == false && texCoords == false) {
		std::vector<glm::vec3> pos = { {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
									   {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 1.0f} };
		for (int i = 0; i < 8; ++i) {
			mesh.vertices.push_back({ pos[i] });
		}
		mesh.triangles = {
			{{1, 0, 3}}, {{1, 3, 2}},
			{{4, 5, 6}}, {{4, 6, 7}},
			{{5, 1, 2}}, {{5, 2, 6}},
			{{0, 4, 7}}, {{0, 7, 3}},
			{{7, 6, 2}}, {{7, 2, 3}},
			{{0, 1, 5}}, {{0, 5, 4}}
		};
	}
	else {
		// ÿ�������ĸ�ԭʼ����������ɣ�����ԭ������˳��
		const uint32_t faces[6][4] = {
			{1, 0, 3, 2}, // z = 0 �棨back��
			{4, 5, 6, 7}, // z = 1 �棨front��
			{5, 1, 2, 6}, // x = 1 �棨right��
			{0, 4, 7, 3}, // x = 0 �棨left��
			{7, 6, 2, 3}, // y = 1 �棨top��
			{0, 1, 5, 4}  // y = 0 �棨bottom��
		};
		const glm::vec3 pos[8] = {
				{0.0f, 0.0f, 0.0f}, // 0
				{1.0f, 0.0f, 0.0f}, // 1
				{1.0f, 1.0f, 0.0f}, // 2
				{0.0f, 1.0f, 0.0f}, // 3
				{0.0f, 0.0f, 1.0f}, // 4
				{1.0f, 0.0f, 1.0f}, // 5
				{1.0f, 1.0f, 1.0f}, // 6
				{0.0f, 1.0f, 1.0f}  // 7
		};
		// ��Ӧÿ����ķ��ߣ������� faces ˳��һһ��Ӧ��
		const glm::vec3 normals[6] = {
			{ 0.0f,  0.0f, -1.0f}, // back
			{ 0.0f,  0.0f,  1.0f}, // front
			{ 1.0f,  0.0f,  0.0f}, // right
			{-1.0f,  0.0f,  0.0f}, // left
			{ 0.0f,  1.0f,  0.0f}, // top
			{ 0.0f, -1.0f,  0.0f}  // bottom
		};
		const glm::vec2 texcoords[4] = {
			{0.0f, 0.0f},
			{1.0f, 0.0f},
			{1.0f, 1.0f},
			{0.0f, 1.0f}
		};
		mesh.vertices.clear();
		mesh.triangles.clear();

		// Ϊÿ���� push 4 ������
		for (int f = 0; f < 6; ++f) {
			for (int v = 0; v < 4; ++v) {
				uint32_t pi = faces[f][v];
				nvutils::PrimitiveVertex vertexData;
				vertexData.pos = pos[pi];
				if (normal) vertexData.nrm = normals[f];
				if (texCoords) vertexData.tex = texcoords[v];
				mesh.vertices.push_back(vertexData);
			}
			uint32_t base = f * 4;
			mesh.triangles.push_back({ {base + 0, base + 1, base + 2} });
			mesh.triangles.push_back({ {base + 0, base + 2, base + 3} });
		}
	}

	return mesh;
}
nvutils::PrimitiveMesh FzbRenderer::MeshSet::createWireframe(float width, float height, float depth) {
	nvutils::PrimitiveMesh mesh;
	std::vector<glm::vec3> pos = { {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
									   {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 1.0f} };
	for (int i = 0; i < 8; ++i) {
		mesh.vertices.push_back({ pos[i] });
	}
	mesh.triangles = {
		{{0, 1, 1}}, {{2, 2, 3}},
		{{3, 0, 4}}, {{5, 5, 6}},
		{{6, 7, 7}}, {{4, 0, 4}},
		{{1, 5, 2}}, {{6, 3, 7}},
	};

	return mesh;
}
nvutils::PrimitiveMesh FzbRenderer::MeshSet::createSphere(bool normal, bool texCoords, uint32_t sectorCount, uint32_t stackCount){
	// ��֤�����ܹ���һ������
	assert(sectorCount >= 3 && stackCount >= 2);

	nvutils::PrimitiveMesh mesh;
	const float radius = 1.0f;

	// Ԥ�������ж����λ�á����ߡ��������꣨��λ��
	std::vector<glm::vec3> posArray;
	std::vector<glm::vec3> nrmArray;
	std::vector<glm::vec2> texArray;

	for (uint32_t i = 0; i <= stackCount; ++i)
	{
		float theta = (float)i * glm::pi<float>() / stackCount; // 0 �� PI
		float sinTheta = sin(theta);
		float cosTheta = cos(theta);

		for (uint32_t j = 0; j <= sectorCount; ++j)
		{
			float phi = (float)j * 2.0f * glm::pi<float>() / sectorCount; // 0 �� 2*PI
			float sinPhi = sin(phi);
			float cosPhi = cos(phi);

			// ����λ��
			glm::vec3 pos = glm::vec3(
				radius * sinTheta * cosPhi,
				radius * cosTheta,
				radius * sinTheta * sinPhi);
			posArray.push_back(pos);
			nrmArray.push_back(glm::normalize(pos));   // ��λ���߼�λ��
			texArray.push_back(glm::vec2((float)j / sectorCount, (float)i / stackCount));
		}
	}

	// ��䶥�����ݣ���ԭ���߼�һ�£���ֱ��ʹ�� posArray �ȣ�
	mesh.vertices.clear();
	mesh.vertices.reserve(posArray.size());
	for (size_t i = 0; i < posArray.size(); ++i)
	{
		nvutils::PrimitiveVertex v;
		v.pos = posArray[i];
		if (normal)    v.nrm = nrmArray[i];
		if (texCoords) v.tex = texArray[i];
		mesh.vertices.push_back(v);
	}

	// �������ɣ�����/�ײ����� + �м��ı����������������������˻������Σ�
	mesh.triangles.clear();
	mesh.triangles.reserve(sectorCount * stackCount * 2); // Ԥ��

	const uint32_t stride = sectorCount + 1;          // ÿ�ж�����
	const uint32_t northPoleIdx = 0;                  // �����㣨ʹ�õ�һ�����㣩
	const uint32_t southPoleIdx = stackCount * stride; // �ϼ��㣨ʹ�����һ�еĵ�һ�����㣩

	// �������Σ�i = 0 ~ 1��
	for (uint32_t j = 0; j < sectorCount; ++j)
	{
		uint32_t nextRowCur = stride + j;
		uint32_t nextRowNext = stride + j + 1;
		mesh.triangles.push_back({ { northPoleIdx, nextRowCur, nextRowNext } });
	}

	// �м��ı��λ���i = 1 ~ stackCount-1��
	for (uint32_t i = 1; i < stackCount - 1; ++i)
	{
		uint32_t curRow = i * stride;
		uint32_t nextRow = (i + 1) * stride;
		for (uint32_t j = 0; j < sectorCount; ++j)
		{
			uint32_t cur = curRow + j;
			uint32_t next = cur + 1;
			uint32_t curUp = nextRow + j;
			uint32_t nextUp = nextRow + j + 1;

			// ���������Σ�������ʱ�����������ɼ���
			mesh.triangles.push_back({ { cur, next, curUp } });
			mesh.triangles.push_back({ { next, nextUp, curUp } });
		}
	}

	// �ײ����Σ�i = stackCount-1 ~ stackCount��
	{
		uint32_t lastRow = (stackCount - 1) * stride;
		for (uint32_t j = 0; j < sectorCount; ++j)
		{
			uint32_t cur = lastRow + j;
			uint32_t next = lastRow + j + 1;
			mesh.triangles.push_back({ { cur, next, southPoleIdx } });
		}
	}

	return mesh;
}
//---------------------------------------------------------------------------------------------------------
void FzbRenderer::MeshSet::createMeshLets() {
	//const size_t kMaxVertices = 64;
	//const size_t kMaxTriangles = 124;
	//const float kConeWeight = 0.0f;
	//
	//for (int childMeshIndex = 0; childMeshIndex < childMeshInfos.size(); ++childMeshIndex) {
	//	MeshInfo& childMeshInfo = childMeshInfos[childMeshIndex];
	//	uint32_t indexCount = childMeshInfo.mesh.triMesh.indices.count;
	//
	//	const size_t maxMeshlets = meshopt_buildMeshletsBound(indexCount, kMaxVertices, kMaxTriangles);
	//	meshlets.resize(maxMeshlets);
	//	meshletVertices.resize(maxMeshlets * kMaxVertices);
	//	meshletTriangles.resize(maxMeshlets * kMaxTriangles * 3);
	//
	//	size_t meshletCount = meshopt_buildMeshlets(
	//		meshlets.data(),
	//		meshletVertices.data(),
	//		meshletTriangles.data(),
	//		reinterpret_cast<const uint32_t*>(indices->data()),
	//		indices->size(),
	//		reinterpret_cast<const float*>(vertices->data()),
	//		vertices->size(),
	//		sizeof(T),
	//		kMaxVertices,
	//		kMaxTriangles,
	//		kConeWeight
	//	);
	//}
	//
	//size_t meshletCount = meshopt_buildMeshlets(
	//	meshlets.data(),
	//	meshletVertices.data(),
	//	meshletTriangles.data(),
	//	reinterpret_cast<const uint32_t*>(indices->data()),
	//	indices->size(),
	//	reinterpret_cast<const float*>(vertices->data()),
	//	vertices->size(),
	//	sizeof(T),
	//	kMaxVertices,
	//	kMaxTriangles,
	//	kConeWeight
	//);
	//
	//auto& last = meshlets[meshletCount - 1];
	//meshletVertices.resize(last.vertex_offset + last.vertex_count);
	//meshletTriangles.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3));	//保证一定是4的倍数
	//meshlets.resize(meshletCount);
}
//---------------------------------------------------------------------------------------------------------
void FzbRenderer::MeshSet::createLowPoly(float ratio) {
	meshByteData_LowPoly.clear();
	childMeshInfos_LowPoly.clear();
	if (!(ratio > 0.0f) || meshByteData.empty() || childMeshInfos.empty()) return;

	const float simplificationRatio = std::min(ratio, 1.0f);

	struct ProcessedLowPolyData {
		std::vector<uint32_t> indices;
		std::vector<uint8_t> positions;
	};

	auto processChild = [this, simplificationRatio](const MeshInfo& childMeshInfo) {
		ProcessedLowPolyData result;

		const shaderio::BufferView& indices = childMeshInfo.mesh.triMesh.indices;
		const shaderio::BufferView& positions = childMeshInfo.mesh.triMesh.positions;
		const uint32_t indexCount = indices.count;
		const uint32_t vertexCount = positions.count;

		if (indexCount < 3 || vertexCount == 0) return result;
		if (indices.offset == static_cast<uint32_t>(-1) || positions.offset == static_cast<uint32_t>(-1)) return result;
		if (indices.offset > meshByteData.size() || positions.offset > meshByteData.size()) return result;

		const size_t indexByteStride = indices.byteStride ? indices.byteStride : (childMeshInfo.mesh.indexType == VK_INDEX_TYPE_UINT16 ? sizeof(uint16_t) : sizeof(uint32_t));
		const size_t indexElementSize = childMeshInfo.mesh.indexType == VK_INDEX_TYPE_UINT16 ? sizeof(uint16_t) : sizeof(uint32_t);
		if (indexElementSize + indexByteStride * (indexCount - 1) > meshByteData.size() - indices.offset) return result;

		const size_t positionByteStride = positions.byteStride ? positions.byteStride : sizeof(glm::vec3);
		if (positionByteStride < sizeof(glm::vec3)) return result;
		if (sizeof(glm::vec3) + positionByteStride * (vertexCount - 1) > meshByteData.size() - positions.offset) return result;

		std::vector<uint32_t> meshIndices(indexCount);
		const uint8_t* indexData = meshByteData.data() + indices.offset;
		if (childMeshInfo.mesh.indexType == VK_INDEX_TYPE_UINT16) {
			for (uint32_t i = 0; i < indexCount; ++i) {
				const uint16_t* index = reinterpret_cast<const uint16_t*>(indexData + i * indexByteStride);
				if (*index >= vertexCount) return result;
				meshIndices[i] = *index;
			}
		}
		else {
			for (uint32_t i = 0; i < indexCount; ++i) {
				const uint32_t* index = reinterpret_cast<const uint32_t*>(indexData + i * indexByteStride);
				if (*index >= vertexCount) return result;
				meshIndices[i] = *index;
			}
		}

		if (indexCount / 3 < 64) {
			result.indices = meshIndices;
			result.positions.resize(vertexCount * sizeof(glm::vec3));
			float* dstPositions = reinterpret_cast<float*>(result.positions.data());
			const uint8_t* srcPositions = meshByteData.data() + positions.offset;
			for (uint32_t i = 0; i < vertexCount; ++i) {
				const float* srcPosition = reinterpret_cast<const float*>(srcPositions + i * positionByteStride);
				dstPositions[i * 3 + 0] = srcPosition[0];
				dstPositions[i * 3 + 1] = srcPosition[1];
				dstPositions[i * 3 + 2] = srcPosition[2];
			}
			return result;
		}

		size_t targetIndexCount = static_cast<size_t>(indexCount * simplificationRatio);
		targetIndexCount = std::max<size_t>(targetIndexCount, 3);
		targetIndexCount -= targetIndexCount % 3;
		targetIndexCount = std::min<size_t>(targetIndexCount, indexCount);

		std::vector<uint32_t> simplifiedIndices(indexCount);
		const float* vertexPositions = reinterpret_cast<const float*>(meshByteData.data() + positions.offset);
		size_t simplifiedIndexCount = 0;
		if (simplificationRatio < 0.25f) {
			simplifiedIndexCount = meshopt_simplifySloppy(
				simplifiedIndices.data(),
				meshIndices.data(),
				indexCount,
				vertexPositions,
				vertexCount,
				positionByteStride,
				nullptr,
				targetIndexCount,
				1.0f,
				nullptr
			);
		}
		else {
			simplifiedIndexCount = meshopt_simplify(
				simplifiedIndices.data(),
				meshIndices.data(),
				indexCount,
				vertexPositions,
				vertexCount,
				positionByteStride,
				targetIndexCount,
				0.01f,
				0,
				nullptr
			);
		}
		simplifiedIndices.resize(simplifiedIndexCount);

		std::vector<uint32_t> remap(vertexCount, static_cast<uint32_t>(-1));
		result.indices.resize(simplifiedIndexCount);
		result.positions.reserve(simplifiedIndexCount * sizeof(glm::vec3));

		const uint8_t* srcPositions = meshByteData.data() + positions.offset;

		for (size_t i = 0; i < simplifiedIndexCount; ++i) {
			const uint32_t vertexIndex = simplifiedIndices[i];
			if (vertexIndex >= vertexCount) return ProcessedLowPolyData{};

			uint32_t newIndex = remap[vertexIndex];
			if (newIndex == static_cast<uint32_t>(-1)) {
				newIndex = static_cast<uint32_t>(result.positions.size() / sizeof(glm::vec3));
				remap[vertexIndex] = newIndex;

				const float* srcPosition = reinterpret_cast<const float*>(srcPositions + vertexIndex * positionByteStride);
				const size_t oldSize = result.positions.size();
				result.positions.resize(oldSize + sizeof(glm::vec3));
				float* dstPosition = reinterpret_cast<float*>(result.positions.data() + oldSize);
				dstPosition[0] = srcPosition[0];
				dstPosition[1] = srcPosition[1];
				dstPosition[2] = srcPosition[2];
			}
			result.indices[i] = newIndex;
		}

		return result;
	};

	size_t reserveSize = 0;
	for (const auto& childMeshInfo : childMeshInfos) {
		const auto& indices = childMeshInfo.mesh.triMesh.indices;
		reserveSize += static_cast<size_t>(indices.count * simplificationRatio) * (sizeof(uint32_t) + sizeof(glm::vec3));
	}
	meshByteData_LowPoly.reserve(reserveSize);
	childMeshInfos_LowPoly.reserve(childMeshInfos.size());

	std::vector<std::future<ProcessedLowPolyData>> futures;
	futures.reserve(childMeshInfos.size());
	for (const auto& childMeshInfo : childMeshInfos) {
		futures.push_back(std::async(std::launch::async, processChild, childMeshInfo));
	}

	for (size_t i = 0; i < futures.size(); ++i) {
		ProcessedLowPolyData lowPolyData = futures[i].get();

		const uint32_t indexOffset = static_cast<uint32_t>(meshByteData_LowPoly.size());
		const size_t indexByteCount = lowPolyData.indices.size() * sizeof(uint32_t);
		if (indexByteCount > 0) {
			const uint8_t* indexBytes = reinterpret_cast<const uint8_t*>(lowPolyData.indices.data());
			meshByteData_LowPoly.insert(meshByteData_LowPoly.end(), indexBytes, indexBytes + indexByteCount);
		}

		const uint32_t positionOffset = static_cast<uint32_t>(meshByteData_LowPoly.size());
		const uint32_t positionPadding = (sizeof(glm::vec3) - meshByteData_LowPoly.size() % sizeof(glm::vec3)) % sizeof(glm::vec3);
		if (positionPadding > 0) meshByteData_LowPoly.insert(meshByteData_LowPoly.end(), positionPadding, 0);

		MeshInfo lowPolyInfo = childMeshInfos[i];
		lowPolyInfo.mesh = {};
		lowPolyInfo.mesh.indexType = VK_INDEX_TYPE_UINT32;
		lowPolyInfo.mesh.triMesh.indices = {
			.offset = indexOffset,
			.count = static_cast<uint32_t>(lowPolyData.indices.size()),
			.byteStride = sizeof(uint32_t)
		};
		lowPolyInfo.mesh.triMesh.positions = {
			.offset = positionOffset + positionPadding,
			.count = static_cast<uint32_t>(lowPolyData.positions.size() / sizeof(glm::vec3)),
			.byteStride = sizeof(glm::vec3)
		};
		childMeshInfos_LowPoly.push_back(lowPolyInfo);

		if (!lowPolyData.positions.empty()) {
			meshByteData_LowPoly.insert(meshByteData_LowPoly.end(), lowPolyData.positions.begin(), lowPolyData.positions.end());
		}
	}
}
