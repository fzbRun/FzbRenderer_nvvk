#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_RESTIR_GI_SHADER_IO_H
#define FZBRENDERER_RESTIR_GI_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

struct ReSTIR_GIConstant{
	int mode = 0;
	int spatialReuseRadius = 8;

	int spp = 1;
	int frameIndex = 0;
	int maxFrameCount;
	int areaLightCount;
	SceneInfo* sceneInfoAddress;
	uint2 sceneSize;
};
struct AreaLight_ReSTIR_GI {
	float3 startPos;
	float3 edge1;
	float3 edge2;
	float3 normal;
	float3 color;
};

struct Reservoir {
	float3 samplePos;
	float3 sampleEmissive;
	int M;
	float W;
};
struct PixelData_ReSTIR_GI {
	Reservoir reservoir;
	
	int materialIndex;
	float3 hitPos;
	float3 hitNormal;
	float3 outgoing;
	float3 emissive;
	float p_hat;
};

enum class StaticBindingPoints_ReSTIR_GI {
	eAreaLights = 2,
	ePixelData = 3,
};

NAMESPACE_SHADERIO_END()
#endif