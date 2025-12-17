#pragma once

#include "./nvvk/shaderio.h"
#include "./nvvk/io_gltf.h"
#include <nvshaders/slang_types.h>

#ifndef FZBRENDERER_SHADERSTRUCTTYPE_H
#define FZBRENDERER_SHADERSTRUCTTYPE_H

NAMESPACE_SHADERIO_BEGIN()
//-------------------------------------------------------实例------------------------------------------------------------
struct Instance {
	float4x4 transform;      // Transform matrix for the instance (local to world)
	uint32_t materialIndex;  // Material properties for the instance
	uint32_t meshIndex;      // Index of the mesh in the GltfMesh vector
};
CHECK_STRUCT_ALIGNMENT(Instance)
//-------------------------------------------------------材质------------------------------------------------------------
enum MaterialType {
	Diffuse = 0,
	Conductor = 1,
	Deielectric = 2,
	RoughConductor = 3,
	RoughDeielectric = 4
};
struct BSDFMaterial {
	MaterialType type;

	float3 albedo;
	float3 emissive;
	float3 eta;
	float roughness;

	int3 materialMapIndex; // 0:albedo, 2:normal, 3:bsdfPara
};
//--------------------------------------------------------Mesh-------------------------------------------------------------
struct Mesh
{
	uint8_t* dataBuffer = nullptr;  // Buffer to the data (index, position, normal, ...)
	TriangleMesh triMesh;               // Mesh data
	int          indexType;             // Index type (uint16_t or uint32_t)
};
//---------------------------------------------------------光源--------------------------------------------------------------
enum LightType {
	Point = 0,
	Spot = 1,
	Directional = 2,
	Area = 3
};;
struct Light {
	float3 pos;
	int type;
	float3 direction;
	float intensity;
	float3 color;
	float coneAngle;

	float3 edge1 = float3(1.0f);
	uint32_t SphericalRectangleSample = false;
	float3 edge2 = float3(1.0f);
	float padding;
};
CHECK_STRUCT_ALIGNMENT(Light)
//-------------------------------------------------------场景信息------------------------------------------------------------
struct SceneInfo
{
	float4x4               viewProjMatrix;     // View projection matrix for the scene
	float4x4               projInvMatrix;      // Inverse projection matrix for the scene
	float4x4               viewInvMatrix;      // Inverse view matrix for the scene
	float3                 cameraPosition;     // Camera position in world space
	int                    useSky;             // Whether to use the sky rendering
	float3                 backgroundColor;    // Background color of the scene (used when not using sky)
	int                    numLights;          // Number of punctual lights in the scene (up to 2)
	Instance* instances;					// Address of the instance buffer containing GltfInstance data
	Mesh* meshes;							// Address of the mesh buffer containing GltfMesh data
	BSDFMaterial* materials;					// Material properties for the instance
	Light           lights[2];			// Array of punctual lights in the scene (up to 2)
	SkySimpleParameters    skySimpleParam;
};
CHECK_STRUCT_ALIGNMENT(SceneInfo)
//-------------------------------------------------------推送常量------------------------------------------------------------
struct PushConstant
{
	float3x3       normalMatrix;
	int            instanceIndex;              // Instance index for the current draw call
	SceneInfo*     sceneInfoAddress;           // Address of the scene information buffer
	int NEEShaderIndex = -1;
	int frameIndex = 0;
	int maxDepth = 3;
};
NAMESPACE_SHADERIO_END()
#endif