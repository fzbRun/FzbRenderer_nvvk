#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_RESTIR_GI_SHADER_IO_H
#define FZBRENDERER_RESTIR_GI_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define Spatial_Reuse_GroupSize 16

struct CreateGBufferPushConstant_ReSTIR_GI {
	float4x4 vpMatrix_lastFrame;
	float4x4 tansfromMatrix_lastFrame;
	int instanceIndex;
	int padding0;
	SceneInfo* sceneInfoAddress;

	float4x4 padding1;
	float4 padding2;
	float4 padding3;
	float4 padding4;
};
struct ReSTIR_GIConstant {
	int mode = 0;
	int spatialReuseRadius = 8;

	int spp = 1;
	int frameIndex = 0;
	int maxFrameCount;
	int areaLightCount;
	SceneInfo* sceneInfoAddress;
	uint2 screenSize;

	float4x4 padding0;
	float4x4 padding1;
	float4x4 padding2;
	float4 padding3;
	float2 padding4;
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
	float2 texCoords;
	float3 hitPos;
	float3 hitNormal;
	float3 outgoing;
	float3 emissive;
	float3 f;
};

enum class StaticBindingPoints_ReSTIR_GI {
	eAreaLights = 2,
	ePixelData,
	eVelocityImage,
	ePixelData_lastFrame,
};

NAMESPACE_SHADERIO_END()
#endif