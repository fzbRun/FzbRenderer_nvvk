#pragma once

#ifndef FZBRENDERER_FLUID_FOG_SHADER_IO_H
#define FZBRENDERER_FLUID_FOG_SHADER_IO_H

#include <common/Shader/shaderStructType.h>
#include <renderer/VolumetricFog/VolumetricFogCommonShaderio.h>

NAMESPACE_SHADERIO_BEGIN()

#define MAX_FLUID_FOG_COUNT 1

struct FluidFogInfo {
	int startUp;

	uint3 gridSize;
	float3 voxelSize;

	float viscosity;
	float FIntensity;
	float restoreSpeed;
	float3 fogStartPos_lastTime;

	float3 attenuation;
};

struct FluidFogVoxelInfo {
	float4 dirtyVelocity_pressure[2];
	bool isBoundary;
	float4 voxelFogInfo;
};

NAMESPACE_SHADERIO_END()
#endif