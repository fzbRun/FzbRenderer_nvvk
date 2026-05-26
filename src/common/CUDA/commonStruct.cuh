#pragma once

#include "./commonCudaFunction.cuh"

#ifndef FZB_COMMON_STRUCT_H
#define FZB_COMMON_STRUCT_H

struct FzbAABB {
	float leftX = FLT_MAX;
	float rightX = -FLT_MAX;
	float leftY = FLT_MAX;
	float rightY = -FLT_MAX;
	float leftZ = FLT_MAX;
	float rightZ = -FLT_MAX;

	__host__ __device__ FzbAABB();;
	__host__ __device__ FzbAABB(float leftX, float rightX, float leftY, float rightY, float leftZ, float rightZ);
};

__host__ __device__ struct FzbAABBUint {	//在glsl中只能对uint和int进行atomicMin和Max
	uint32_t leftX;
	uint32_t rightX;
	uint32_t leftY;
	uint32_t rightY;
	uint32_t leftZ;
	uint32_t rightZ;
};

__host__ __device__ struct FzbOBB {
	glm::vec3 minXPos;
	glm::vec3 minYPos;
	glm::vec3 minZPos;
	glm::vec3 maxXPos;
	glm::vec3 maxYPos;
	glm::vec3 maxZPos;
};
__host__ __device__ struct FzbOBBUint {
	glm::ivec3 minXPos;
	glm::ivec3 minYPos;
	glm::ivec3 minZPos;
	glm::ivec3 maxXPos;
	glm::ivec3 maxYPos;
	glm::ivec3 maxZPos;
};

#endif