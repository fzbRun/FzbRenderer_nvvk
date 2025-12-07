#ifndef SHADERIO_H
#define SHADERIO_H

#include "./io_gltf.h"
NAMESPACE_SHADERIO_BEGIN()
// Binding Points
enum BindingPoints
{
	eTextures = 0,
	eTlas = 1,
	eOutImage = 2,
	eHeatmap = 3,
	eHeatStates = 4
};

enum ImplicitObjectKind {
	eSphere = 0,
	eCube = 1
};
struct Sphere {
	float3 center;
	float radius;
};
struct AABB {
	float3 minimum;
	float3 maximum;
};

struct TutoPushConstant
{
  float3x3       normalMatrix;
  int            instanceIndex;              // Instance index for the current draw call
  GltfSceneInfo* sceneInfoAddress;           // Address of the scene information buffer
  float2         metallicRoughnessOverride;  // Metallic and roughness override values


  int frameIndex = 0;
  int maxDepth = 3;
};

struct HeatState {
	uint32_t maxDuration[2];
};



NAMESPACE_SHADERIO_END()
#endif  // SHADERIO_H
