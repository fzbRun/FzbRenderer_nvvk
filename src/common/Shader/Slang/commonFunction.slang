#include "common/Shader/shaderStructType.h"
#include "nvshaders/constants.h.slang"

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

float3 fibSpherePoint(uint i, uint n) {
    float f_i = float(i);
    float f_n = float(n);

    float cosTheta = 1.0 - (2.0 * f_i + 1.0) / f_n;
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    //float PHI = 1.618033988749895f;
    float GOLDEN_ANGLE = 2.0 * M_PI * (1.618033988749895f - 1.0);
    float phi = GOLDEN_ANGLE * f_i;

    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}
uint inverseSF(float3 p, uint n) {
    float f_n = float(n);
    float cosTheta = p.z;

    //    cosTheta = 1 - (2*i+1)/n  => i ¡Ö (n*(1-cosTheta)-1)/2
    float i_est = (f_n * (1.0 - cosTheta) - 1.0) * 0.5;
    int i0 = int(floor(i_est));

    int searchRadius = 2;
    int start = max(0, i0 - searchRadius);
    int end = min(int(n) - 1, i0 + searchRadius);

    uint bestIdx = 0;
    float bestDistSq = 1e10;

    for (int idx = start; idx <= end; ++idx) {
        float3 q = fibSpherePoint(uint(idx), n);
        float3 d = q - p;
        float distSq = dot(d, d);
        if (distSq < bestDistSq) {
            bestDistSq = distSq;
            bestIdx = uint(idx);
        }
    }

    return bestIdx;
}