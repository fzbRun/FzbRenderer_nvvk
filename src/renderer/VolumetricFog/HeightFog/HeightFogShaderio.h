#pragma once

#ifndef FZBRENDERER_HEIGHT_FOG_SHADER_IO_H
#define FZBRENDERER_HEIGHT_FOG_SHADER_IO_H

#include <common/Shader/shaderStructType.h>
#include <renderer/VolumetricFog/VolumetricFogCommonShaderio.h>

NAMESPACE_SHADERIO_BEGIN()

#define MAX_HEIGHT_FOG_COUNT 3

struct HeightFogInfo {
    float3 color;
    float ambientIntensity;
    float absorption;
    float scattering;
    float phase;

	float heightScale;

    float3 attenuation;
};

NAMESPACE_SHADERIO_END()
#endif