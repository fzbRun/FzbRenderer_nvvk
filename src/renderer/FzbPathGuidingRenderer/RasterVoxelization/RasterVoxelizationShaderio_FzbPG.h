#pragma once

#include <common/Shader/shaderStructType.h>
#include <feature/SceneDivision/RasterVoxelization/shaderio.h>

#ifndef FZBRENDERER_RASTERVOXELIZATION_FZBPG_SHADER_IO_H
#define FZBRENDERER_RASTERVOXELIZATION_FZBPG_SHADER_IO_H

NAMESPACE_SHADERIO_BEGIN()
enum class RasterVoxelizationBindingPoints_FzbPG
{
	eTextures = 0,
	eVGB,
#ifndef NDEBUG
	eFragmentCountBuffer,
	eWireframeMap,
	eBaseMap,
#endif
};

struct VGBVoxelData_FzbPG {
	float4 irradiance;
	float4 sumNormal_G;
	AABBI aabbI;
};
NAMESPACE_SHADERIO_END()
#endif