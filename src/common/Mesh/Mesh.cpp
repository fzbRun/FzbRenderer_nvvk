#include <nvutils/timers.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/resource_allocator.hpp>
#include <common/Application/Application.h>
#include <glm/gtc/type_ptr.hpp>
#include <memory>
#include "./Mesh.h"
#include <common/Material/Material.h>

shaderio::AABB FzbRenderer::MeshInfo::getAABB(glm::mat4 transformMatrix) {
	glm::vec3 maximum = { FLT_MAX, FLT_MAX, FLT_MAX };
	if (aabb.minimum != maximum && aabb.maximum != -maximum) return aabb;

	Scene& sceneRsource = Application::sceneResource;
	std::vector<uint8_t>& meshByteData = sceneRsource.meshSets[sceneRsource.getMeshSetIndex(meshIndex)].meshByteData;
	const auto& positions = mesh.triMesh.positions;
	const glm::vec3* vertexData = reinterpret_cast<const glm::vec3*>(meshByteData.data() + positions.offset);

	for (uint32_t i = 0; i < positions.count; ++i) {
		const glm::vec3& pos = vertexData[i];
		glm::vec3 pos_transform = transformMatrix * glm::vec4(pos, 1.0f);
		aabb.minimum.x = std::min(pos_transform.x, aabb.minimum.x);
		aabb.minimum.y = std::min(pos_transform.y, aabb.minimum.y);
		aabb.minimum.z = std::min(pos_transform.z, aabb.minimum.z);
		aabb.maximum.x = std::max(pos_transform.x, aabb.maximum.x);
		aabb.maximum.y = std::max(pos_transform.y, aabb.maximum.y);
		aabb.maximum.z = std::max(pos_transform.z, aabb.maximum.z);
	}

	return aabb;
}
//-----------------------------------------------------MeshSet---------------------------------------------------
//所有的数据都不是交错的，即pos全放在一起，normal全放在一起……。这也影响这其他所有读取数据的地方，需要注意!!!!!
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

	aabb.minimum = { FLT_MAX, FLT_MAX, FLT_MAX };
	aabb.maximum = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
}

/*
gltf的material采用金属-粗糙度模型，因此很难知道是不是diffuse的，还是别的什么材质的
所以，我们按照，如果粗糙度为为0，则是光滑的；反之为粗糙度
如果alphaMode为OPAQUE,则是导体；反之为电介质
如果为电介质，那么eta就为透明度
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
	if (sceneData->mNumMaterials > 1) {		//有一个默认材质
		aiMaterial* mtlMaterial = sceneData->mMaterials[meshData->mMaterialIndex];
		materialID = std::string(mtlMaterial->GetName().data);
		material = loadMtlMaterial(mtlMaterial);
	}

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
	Assimp::Importer import;
	uint32_t needs = aiProcess_Triangulate | aiProcess_GenSmoothNormals;// |
		//(vertexFormat.useTexCoord ? aiProcess_FlipUVs : aiPostProcessSteps(0u)) |
		//(vertexFormat.useNormal ? aiProcess_GenSmoothNormals : aiPostProcessSteps(0u)) |
		//(vertexFormat.useTangent ? aiProcess_CalcTangentSpace : aiPostProcessSteps(0u));
	const aiScene* sceneData = import.ReadFile(meshPath.string(), needs);

	if (!sceneData || sceneData->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !sceneData->mRootNode) 
		LOGW(import.GetErrorString());

	processNode(sceneData->mRootNode, sceneData);
}

FzbRenderer::MeshSet::MeshSet(std::string meshID, nvutils::PrimitiveMesh primitiveMesh)
{
	this->meshID = meshID;
	shaderio::Mesh mesh{};

	uint32_t indexCount = primitiveMesh.triangles.size() * 3;
	uint32_t maxIndex = 0;
	for (const auto& tri : primitiveMesh.triangles)
		maxIndex = std::max(maxIndex, std::max(tri.indices.x, std::max(tri.indices.y, tri.indices.z)));
	if(maxIndex <= 0xFFFF) {
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
//-----------------------------------------------------生成图元----------------------------------------------------
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

	// 生成顶点：从 (0,0,0) 开始，横向为 X(0..width)，纵向为 Y(0..depth)，Z = 0
	for (int sy = 0; sy <= steps; ++sy)
	{
		for (int sx = 0; sx <= steps; ++sx)
		{
			nvutils::PrimitiveVertex v{};

			float u = static_cast<float>(sx) * invSteps; // 0..1
			float v_t = static_cast<float>(sy) * invSteps; // 0..1

			v.pos = glm::vec3(u * width, v_t * height, 0.0f); // XY 平面，起点 (0,0,0)
			v.nrm = glm::vec3(0.0f, 0.0f, 1.0f);               // 指向 +Z
			v.tex = glm::vec2(u, v_t);                        // (0,0) -> (1,1)

			mesh.vertices.emplace_back(v);
		}
	}

	// 生成索引（三角形），确保顶点顺序在 XY 平面上为 CCW（面朝 +Z）
	const int rowStride = steps + 1;
	for (int sy = 0; sy < steps; ++sy)
	{
		for (int sx = 0; sx < steps; ++sx)
		{
			int a = sx + sy * rowStride;                 // (sx, sy)
			int c = (sx + 1) + sy * rowStride;           // (sx+1, sy)
			int b = (sx + 1) + (sy + 1) * rowStride;     // (sx+1, sy+1)
			int d = sx + (sy + 1) * rowStride;           // (sx, sy+1)

			// 两个三角形： (a, c, b) 和 (a, b, d) -> 保证朝向 +Z（CCW）
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
		// 每个面由四个原始顶点索引组成（按你原来的面顺序）
		const uint32_t faces[6][4] = {
			{1, 0, 3, 2}, // z = 0 面（back）
			{4, 5, 6, 7}, // z = 1 面（front）
			{5, 1, 2, 6}, // x = 1 面（right）
			{0, 4, 7, 3}, // x = 0 面（left）
			{7, 6, 2, 3}, // y = 1 面（top）
			{0, 1, 5, 4}  // y = 0 面（bottom）
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
		// 对应每个面的法线（与上面 faces 顺序一一对应）
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

		// 为每个面 push 4 个顶点
		for (int f = 0; f < 6; ++f) {
			for (int v = 0; v < 4; ++v) {
				uint32_t pi = faces[f][v];
				nvutils::PrimitiveVertex vertexData;
				vertexData.pos = pos[pi];
				if (normal) vertexData.nrm = normals[f];
				if (texCoords) vertexData.tex = texcoords[v];
				mesh.vertices.push_back(vertexData);
			}
			glm::uvec3 indices0 = glm::uvec3(faces[f][0], faces[f][1], faces[f][2]);
			glm::uvec3 indices1 = glm::uvec3(faces[f][1], faces[f][2], faces[f][3]);
			mesh.triangles.push_back({ indices0 });
			mesh.triangles.push_back({ indices1 });
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