#pragma once

#ifndef FZBRENDERER_FLUID_FOG_SHADER_IO_H
#define FZBRENDERER_FLUID_FOG_SHADER_IO_H

#include <common/Shader/shaderStructType.h>
#include <renderer/VolumetricFog/VolumetricFogCommonShaderio.h>

NAMESPACE_SHADERIO_BEGIN()

#define MAX_FLUID_FOG_COUNT 1

#define FLUID_A_MACCORMACK

struct FluidFogInfo {
	int startUp;			//是否开启流体

	uint3 gridSize;
	float3 voxelSize;

	float3 magicNumber;				//不用管
	float viscosity;				//粘性系数
	float FIntensity;				//力的强度
	float restoreSpeed;				//与环境烟雾的还原速度
	float3 fogStartPos_lastTime;	//上一帧的雾的起始位置

	float3 attenuation;				//入射光透射率，默认为1
};

struct FluidFogVoxelInfo {
	float4 dirtyVelocity_p[2];		//xyz为脏速度，w为标量场值
	bool isBoundary;				//是否为边界
	float4 voxelFogInfo;			//雾数据，就是流体image中存储的值
};

NAMESPACE_SHADERIO_END()
#endif