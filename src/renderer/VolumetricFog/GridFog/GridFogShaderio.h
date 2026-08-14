#pragma once

#ifndef FZBRENDERER_GRID_FOG_SHADER_IO_H
#define FZBRENDERER_GRID_FOG_SHADER_IO_H

#include <common/Shader/shaderStructType.h>
#include <renderer/VolumetricFog/VolumetricFogCommonShaderio.h>

NAMESPACE_SHADERIO_BEGIN()

#define MAX_GRID_FOG_COUNT 3

enum class GridFogGenerateMode {
	Cloud,
};

struct GridFogInfo {
	uint3 gridSize;
	float3 voxelSize;

	float absorption;
	float scattering;
	float phase;
	float ambientIntensity;
	float3 color;

	float transmittance = 1.0f;
};

struct CloudFogInfo {
	float3 cloudScale;
	float cloudFlowSpeed;
	float2 cloudCoverage;
	float2 cloudTypePreference;
	float weatherScale;
};
struct GridFogGenerationInfo {
	CloudFogInfo cloudInfo;
};

struct GridFogPushConstant {
	int randomSeed = 0;

	int index;

	GridFogGenerateMode mode;
	AABB fogRange;
	uint3 gridSize;
	float3 voxelSize;

	float absorption;
	float scattering;
	float phase;
	float3 color;
	float ambientIntensity;

	GridFogGenerationInfo generationInfo;
};

enum class StaticBindingPoints_GridFogSet {
	eGridFogImages,
};

NAMESPACE_SHADERIO_END()
#endif