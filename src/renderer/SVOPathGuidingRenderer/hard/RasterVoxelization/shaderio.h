#pragma once

#include <common/Shader/shaderStructType.h>
#include <feature/SceneDivision/RasterVoxelization/shaderio.h>
#include "renderer/SVOPathGuidingRenderer/hard/shaderio.h"

#ifndef FZBRENDERER_RASTERVOXELIZATION_SVOPG_SHADER_IO_H
#define FZBRENDERER_RASTERVOXELIZATION_SVOPG_SHADER_IO_H

NAMESPACE_SHADERIO_BEGIN()
enum RasterVoxelizationBindingPoints_SVOPG
{
	eTextures_SVOPG = 0,
	eVGB_SVOPG,
	#ifdef CLUSTER_WITH_MATERIAL
	eVGBMaterialInfo_SVOPG,
	#endif

	#ifndef NDEBUG
	eFragmentCountBuffer_SVOPG,
	eWireframeMap_SVOPG,
	eBaseMap_SVOPG,
	#endif
};

struct VGBMaterialInfo_SVOPG {
	uint materialCount[MAX_MATERIAL_COUNT];
};

struct AABBI {
	int4 minimum;
	int4 maximum;
};
struct VGBVoxelData_SVOPG {
	float4 irradiance;
	float4 sumNormal_G;
	float4 sumNormal_E;
	AABBI aabbI;
};
NAMESPACE_SHADERIO_END()
#endif