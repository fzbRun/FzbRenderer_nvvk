#include "./Mesh.h"
#include <nvutils/timers.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/resource_allocator.hpp>
#include <common/Application/Application.h>
#include <glm/gtc/type_ptr.hpp>

FzbRenderer::Mesh::Mesh(std::string meshType, std::filesystem::path meshPath)
{
	Scene& scene = Application::sceneResource;
	if (meshType == "gltf" || meshType == "glb") {
		tinygltf::Model gltfModel = nvsamples::loadGltfResources(nvutils::findFile(meshPath, { meshPath }));
		loadGltfData(gltfModel);
	}
	else if (meshType == "obj") {

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