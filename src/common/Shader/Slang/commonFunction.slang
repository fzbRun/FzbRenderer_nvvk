#include "common/Shader/shaderStructType.h"

float3x3 Inverse3x3(float3x3 m) {
	float3 c0 = m[0];
	float3 c1 = m[1];
	float3 c2 = m[2];
	
	float3 c0_c1 = cross(c0, c1);
	float3 c1_c2 = cross(c1, c2);
	float3 c2_c0 = cross(c2, c0);

    float det = dot(c2, c0_c1);
    if (fabs(det) < 1e-12f) return float3x3(1.0f);
    float invDet = 1.0f / det;

    float3 r0 = c1_c2 * invDet;
    float3 r1 = c2_c0 * invDet;
    float3 r2 = c0_c1 * invDet;

    return transpose(float3x3(r0, r1, r2));
}