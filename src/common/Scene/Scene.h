/*
sceneManager主要有三个功能
1. 从sceneXML中读取信息，主要包括
	a. mesh信息，如.obj，.glft
	b. material信息，如diffuse roughconductor，roughdielectric
	c. camera信息
	d. light信息
2. 维护上述信息
3. 一些辅助函数，如删除冗余顶点
*/

#include <memory> 
#include <vector>

#include <nvutils/camera_manipulator.hpp>
#include <gltf_utils.hpp>


namespace FzbRenderer {

class SceneSourceManager {
public:

private:
	std::shared_ptr<nvutils::CameraManipulator> m_cameraManip{ std::make_shared<nvutils::CameraManipulator>() };

	nvsamples::GltfSceneResource m_sceneResource{};
	std::vector<nvvk::Image>     m_textures{};
};

}