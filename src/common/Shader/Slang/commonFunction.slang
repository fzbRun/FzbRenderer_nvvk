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

    // 纬度相关：cosTheta 从 1 线性递减到 -1
    float cosTheta = 1.0 - (2.0 * f_i + 1.0) / f_n;
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    // 经度：使用黄金角旋转
    //float PHI = 1.618033988749895f;
    float GOLDEN_ANGLE = 2.0 * M_PI * (1.618033988749895f - 1.0);
    float phi = GOLDEN_ANGLE * f_i;

    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}
// 逆映射：返回离 p 最近的点在 fibSpherePoint 中的索引
uint inverseSF(float3 p, uint n) {
    float f_n = float(n);
    float cosTheta = p.z;

    // 1. 根据 cosTheta 粗略估计索引（忽略 phi 的影响）
    //    cosTheta = 1 - (2*i+1)/n  => i ≈ (n*(1-cosTheta)-1)/2
    float i_est = (f_n * (1.0 - cosTheta) - 1.0) * 0.5;
    int i0 = int(floor(i_est));

    // 2. 在 i0 附近搜索几个候选（考虑 phi 的偏差）
    int searchRadius = 2; // 通常 ±2 足够覆盖误差
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