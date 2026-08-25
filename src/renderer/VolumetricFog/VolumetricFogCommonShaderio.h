#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_VOLUMETRIC_FOG_COMMON_SHADER_IO_H
#define FZBRENDERER_VOLUMETRIC_FOG_COMMON_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

enum class VolumetricFogType {
    Height,
    Fluid,
    Noise,
    Grid,
};

struct VolumetricFogInfo {
    AABB aabb;                  //范围

    VolumetricFogType type;     //类型
    int typeFogIndex;           //在相应类型的体积雾中的索引
};

NAMESPACE_SHADERIO_END()
#endif