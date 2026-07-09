#ifndef FZBRENDERER_NOISE_SLANG
#define FZBRENDERER_NOISE_SLANG

#include "common/Shader/shaderStructType.h"
#include "nvshaders/constants.h.slang"
#include "nvshaders/random.h.slang"

float hash(float3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float3 randomGradient(float3 p) {
    float cosTheta = hash(p) * 2 - 1;       //theta = arccos(1 - 2u)
    float a = hash(p + 17.0) * 6.2831853;   //phi = 2Pi * u
    float sinTheta = sqrt(1 - cosTheta * cosTheta);
    return float3(sinTheta * cos(a), sinTheta * sin(a), cosTheta);
}
float perlinNoise(float3 p) {
    float3 i = floor(p);
    float3 f = fract(p);
    float3 u = f * f * f * (f * (f * 6 - 15) + 10);

    return lerp(
        lerp(
            lerp(dot(randomGradient(i + float3(0, 0, 0)), f - float3(0, 0, 0)),
                          dot(randomGradient(i + float3(1, 0, 0)), f - float3(1, 0, 0)), u.x),
                     lerp(dot(randomGradient(i + float3(0, 1, 0)), f - float3(0, 1, 0)),
                          dot(randomGradient(i + float3(1, 1, 0)), f - float3(1, 1, 0)), u.x), u.y),
                lerp(
            lerp(dot(randomGradient(i + float3(0, 0, 1)), f - float3(0, 0, 1)),
                          dot(randomGradient(i + float3(1, 0, 1)), f - float3(1, 0, 1)), u.x),
                     lerp(dot(randomGradient(i + float3(0, 1, 1)), f - float3(0, 1, 1)),
                          dot(randomGradient(i + float3(1, 1, 1)), f - float3(1, 1, 1)), u.x), u.y), u.z);
}

float3 mod289(float3 x) {
    return x - floor(x / 289.0) * 289.0;
}
float4 mod289(float4 x) {
    return x - floor(x / 289.0) * 289.0;
}
float4 permute(float4 x) {
    return mod289((x * 34.0 + 1.0) * x);
}
float4 taylorInvSqrt(float4 r) {
    return 1.79284291400159 - 0.85373472095314 * r;
}

float snoise(float3 v) {
    const float2 C = float2(1.0 / 6.0, 1.0 / 3.0);
    const float4 D = float4(0.0, 0.5, 1.0, 2.0);

    float3 i = floor(v + dot(v, C.yyy));
    float3 x0 = v - i + dot(i, C.xxx);

    float3 g = step(x0.yzx, x0.xyz);
    float3 l = 1.0 - g;
    float3 i1 = min(g.xyz, l.zxy);
    float3 i2 = max(g.xyz, l.zxy);

    float3 x1 = x0 - i1 + C.xxx;
    float3 x2 = x0 - i2 + C.yyy;
    float3 x3 = x0 - D.yyy;

    i = mod289(i);
    float4 p = permute(permute(permute(
                i.z + float4(0.0, i1.z, i2.z, 1.0))
              + i.y + float4(0.0, i1.y, i2.y, 1.0))
              + i.x + float4(0.0, i1.x, i2.x, 1.0));

    float n_ = 0.142857142857; // 1/7
    float3 ns = n_ * D.wyz - D.xzx;

    float4 j = p - 49.0 * floor(p / 49.0); // mod(p, 7*7)

    float4 x_ = floor(j / 7.0);
    float4 y_ = floor(j - 7.0 * x_); // mod(j, N)

    float4 x = x_ * ns.x + ns.yyyy;
    float4 y = y_ * ns.x + ns.yyyy;
    float4 h = 1.0 - abs(x) - abs(y);

    float4 b0 = float4(x.xy, y.xy);
    float4 b1 = float4(x.zw, y.zw);

    float4 s0 = floor(b0) * 2.0 + 1.0;
    float4 s1 = floor(b1) * 2.0 + 1.0;
    float4 sh = -step(h, float4(0.0, 0.0, 0.0, 0.0));

    float4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    float4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

    float3 p0 = float3(a0.xy, h.x);
    float3 p1 = float3(a0.zw, h.y);
    float3 p2 = float3(a1.xy, h.z);
    float3 p3 = float3(a1.zw, h.w);

    float4 norm = taylorInvSqrt(float4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    float4 m = max(0.6 - float4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m * m, float4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

float vnoise(float3 p) {
    float3 i = floor(p);
    float3 f = frac(p);
    float3 u = f * f * f * (f * (f * 6 - 15) + 10);

    return lerp(
        lerp(
            lerp(hash(i + float3(0.0, 0.0, 0.0)), hash(i + float3(1.0, 0.0, 0.0)), u.x),
                     lerp(hash(i + float3(0.0, 1.0, 0.0)), hash(i + float3(1.0, 1.0, 0.0)), u.x),
            u.y
        ),
                lerp(
            lerp(hash(i + float3(0.0, 0.0, 1.0)), hash(i + float3(1.0, 0.0, 1.0)), u.x),
                     lerp(hash(i + float3(0.0, 1.0, 1.0)), hash(i + float3(1.0, 1.0, 1.0)), u.x),
            u.y
        ),
                u.z
    );
}

float fbm(float3 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    float persistence = 0.5;

    for (int i = 0; i < 6; i++) {
        value += amplitude * perlinNoise(p * frequency);
        frequency *= 2.0;
        amplitude *= persistence;
    }
    return value;
}
float worleyNoise(float3 p) {
    float3 i = floor(p);
    float3 f = fract(p);
    float minDist = 1.0;

    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            for (int z = -1; z <= 1; z++) {
                float3 neighbor = float3(float(x), float(y), float(z));
                float3 point = float3(
                    hash(i + neighbor),
                    hash(i + neighbor + 10.0),
                    hash(i + neighbor + 20.0)
                );
                float3 diff = neighbor + point - f;
                float d = dot(diff, diff);
                minDist = min(minDist, d);
            }
        }
    }
    return sqrt(minDist);
}
float turbulenceAbs(float3 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    for (int i = 0; i < 6; i++) {
        value += amplitude * abs(perlinNoise(p * frequency));
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}
float turbulenceWarp(float3 p) {
    float offsetX = fbm(p);
    float offsetY = fbm(p + float3(1.3, 5.2, 3.1));
    float offsetZ = fbm(p + float3(3.1, 1.7, 4.3));

    float3 warped = p + float3(offsetX, offsetY, offsetZ) * 2.0; // can be change strength
    return fbm(warped);
}

#endif