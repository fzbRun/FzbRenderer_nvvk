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

//---------------------------------------------------------------------clouad-----------------------------------------------------------------------
float3 hash33(float3 p) {
    uint3 q = uint3(int3(p)) * 2798796415U;
    q = (q.x ^ q.y ^ q.z) * 2798796415U;
    return -1.0 + 2.0 * float3(q) * 2.3283064365386963e-10; // 1/2^32
}
float3 Mod(float3 x, float y) { return x - y * floor(x / y); }
float gradientNoise(float3 x, float freq) {
    float3 p = floor(x);
    float3 w = frac(x);
    float3 u = w * w * w * (w * (w * 6.0 - 15.0) + 10.0);
    float3 ga = hash33(Mod(p + float3(0, 0, 0), freq));
    float3 gb = hash33(Mod(p + float3(1, 0, 0), freq));
    float3 gc = hash33(Mod(p + float3(0, 1, 0), freq));
    float3 gd = hash33(Mod(p + float3(1, 1, 0), freq));
    float3 ge = hash33(Mod(p + float3(0, 0, 1), freq));
    float3 gf = hash33(Mod(p + float3(1, 0, 1), freq));
    float3 gg = hash33(Mod(p + float3(0, 1, 1), freq));
    float3 gh = hash33(Mod(p + float3(1, 1, 1), freq));
    float va = dot(ga, w - float3(0, 0, 0));
    float vb = dot(gb, w - float3(1, 0, 0));
    float vc = dot(gc, w - float3(0, 1, 0));
    float vd = dot(gd, w - float3(1, 1, 0));
    float ve = dot(ge, w - float3(0, 0, 1));
    float vf = dot(gf, w - float3(1, 0, 1));
    float vg = dot(gg, w - float3(0, 1, 1));
    float vh = dot(gh, w - float3(1, 1, 1));
    return va + u.x * (vb - va) + u.y * (vc - va) + u.z * (ve - va)
         + u.x * u.y * (va - vb - vc + vd) + u.y * u.z * (va - vc - ve + vg) + u.z * u.x * (va - vb - ve + vf)
         + u.x * u.y * u.z * (-va + vb + vc - vd + ve - vf - vg + vh);
}
float perlinfbm(float3 p, float freq, int octaves) {
    float G = exp2(-0.85);
    float amp = 1.0;
    float noise = 0.0;
    for (int i = 0; i < octaves; ++i) {
        noise += amp * gradientNoise(p * freq, freq);
        freq *= 2.0;
        amp *= G;
    }
    return noise;
}
float worleyNoise(float3 uv, float freq) {
    float3 id = floor(uv);
    float3 p = frac(uv);
    float minDist = 10000.0;
    for (float x = -1.0; x <= 1.0; x += 1.0) {
        for (float y = -1.0; y <= 1.0; y += 1.0) {
            for (float z = -1.0; z <= 1.0; z += 1.0) {
                float3 offset = float3(x, y, z);
                float3 h = hash33(Mod(id + offset, freq)) * 0.5 + 0.5;
                h += offset;
                float3 d = p - h;
                minDist = min(minDist, dot(d, d));
            }
        }
    }
    return 1.0 - minDist; // inverted Worley for cloud shapes
}
float worleyFbm(float3 p, float freq) {
    return worleyNoise(p * freq, freq) * 0.625
         + worleyNoise(p * freq * 2.0, freq * 2.0) * 0.25
         + worleyNoise(p * freq * 4.0, freq * 4.0) * 0.125;
}

// ----- Remap utility -----
float Remap(float v, float oldMin, float oldMax, float newMin, float newMax) {
    return newMin + ((v - oldMin) / (oldMax - oldMin)) * (newMax - newMin);
}

// ============================================================
// Original cloud helper functions (preserved, added weather gen)
// ============================================================
float GetHeightFractionForPoint(float3 inPosition, float2 inCloudMinMax) {
    float h = (inPosition.y - inCloudMinMax.x) / (inCloudMinMax.y - inCloudMinMax.x);
    return saturate(h);
}

float2 GetXZFractionForPoint(float3 inPosition, float4 inCloudMinMax) {
    float2 xz;
    xz.x = (inPosition.x - inCloudMinMax.x) / (inCloudMinMax.z - inCloudMinMax.x);
    xz.y = (inPosition.z - inCloudMinMax.y) / (inCloudMinMax.w - inCloudMinMax.y);
    return xz;
}

float GetGradientHeightFactor(float height, int cloudType) {
    float timeWidthUp, startTimeUp, startTimeDown;
    if (cloudType == 0) { // Stratus
        timeWidthUp = 0.08; startTimeUp = 0.08; startTimeDown = 0.3;
    } else if (cloudType == 1) { // Cumulus
        timeWidthUp = 0.14; startTimeUp = 0.1; startTimeDown = 0.8; // startTimeDown = 0.5
    } else { // Cumulonimbus (cloudType == 2)
        timeWidthUp = 0.2; startTimeUp = 0.1; startTimeDown = 0.7;
    }
    float factor = (2.0 * 3.14159265) / (2.0 * timeWidthUp);
    float density = 0.0;
    if (height < startTimeUp) {
        density = 0.0;
    } else if (height < startTimeUp + timeWidthUp) {
        density = 0.5 * sin(factor * height - 1.5707963 - factor * startTimeUp) + 0.5;
    } else if (height < startTimeDown) {
        density = 1.0;
    } else if (height < startTimeDown + timeWidthUp) {
        density = 0.5 * sin(factor * height - 1.5707963 - factor * (startTimeDown + timeWidthUp)) + 0.5;
    } else {
        density = 0.0;
    }
    return density;
}

float GetDensityHeightGradientForPoint(float3 positionWS, float3 weatherData, float heightFraction) {
    float cloudTypeValue = weatherData.b; // B channel now carries cloud type
    int cloudType = 1;
    if (cloudTypeValue < 0.1) cloudType = 0;
    else if (cloudTypeValue > 0.9) cloudType = 2;
    return GetGradientHeightFactor(heightFraction, cloudType);
}

// ----- Procedural weather data (replaces _WeatherDataTexture) -----
float3 SampleWeatherData(float2 xzFraction, float weatherScale = 0.01f) {
    // Use a scale factor to control pattern size (tune externally)
    float2 uv = xzFraction * weatherScale;
    // Generate coverage, precipitation (unused), and cloud type from layered noise
    float r = snoise(float3(uv, 1.0)) * 0.5 + 0.8;
    float g = snoise(float3(uv + 13.0, 4.5)) * 0.5 + 0.5;
    float b = snoise(float3(uv - 7.3, 12.1)) * 0.5 + 0.5;
    return float3(r, g, b);
}

// ----- Curl noise (2D on XZ plane, derived from snoise gradients) -----
float2 ComputeCurl(float3 p) {
    float eps = 0.1;
    float dx = snoise(p + float3(eps, 0.0, 0.0)) - snoise(p - float3(eps, 0.0, 0.0));
    float dz = snoise(p + float3(0.0, 0.0, eps)) - snoise(p - float3(0.0, 0.0, eps));
    // Return normalized curl vector (original implementation normalizes)
    return normalize(float2(-dz, dx));
}

// ============================================================
// Main cloud density function ¨C fully procedural
// cloudScale contorl cloud size
// weatherData: r contorl cloud solid; g: Rainwater (currently unused); b: Cloud type
// weatherScale control cloud change speed
// ============================================================
float SampleCloudDensity(float3 positionWS, float3 cloudMin, float3 cloadMax, float time, float cloadFlowSpeed = 1.0f, float3 cloudScale = 1.0f) {
    float heightFraction = GetHeightFractionForPoint(positionWS, float2(cloudMin.y, cloadMax.y));
    float2 xzFraction = GetXZFractionForPoint(positionWS, float4(cloudMin.xz, cloadMax.xz));
    float3 uvw = float3(xzFraction.x, heightFraction, xzFraction.y) * cloudScale;
    uvw += float3(time * cloadFlowSpeed);

    // Procedural low-frequency noises (replaces _CloudNoise3DTextureA)
    float perlinWorley = perlinfbm(uvw, 4.0, 7) - (1.0 - worleyFbm(uvw, 4.0));
    float worleyG = worleyFbm(uvw, 4.0);
    float worleyB = worleyFbm(uvw, 8.0);
    float worleyA = worleyFbm(uvw, 16.0);
    float lowFrequencyFbm = 0.625 * worleyG + 0.25 * worleyB + 0.125 * worleyA;
    float baseCloud = Remap(perlinWorley, -(1.0 - lowFrequencyFbm), 1.0, 0.0, 1.0);

    // Procedural weather (replaces _WeatherDataTexture)
    float3 weatherData = SampleWeatherData(xzFraction);
    // Apply original perturbations and clip
    heightFraction += (2.0 * weatherData.r - 1.0) * 0.03;
    //weatherData.r += 0.25;
    weatherData.r = weatherData.r * 0.3 + 0.2;

    float densityHeightGradient = GetDensityHeightGradientForPoint(positionWS, weatherData, heightFraction);
    baseCloud *= densityHeightGradient;
    return saturate(baseCloud);

    //float cloudCoverage = clamp(weatherData.r, 0.0, 0.999);
    //float baseCloudWithCoverage = Remap(baseCloud, cloudCoverage, 1.0, 0.0, 1.0);
    //baseCloudWithCoverage *= cloudCoverage;
    //
    //float finalCloud = baseCloudWithCoverage;
    //
    //if (baseCloudWithCoverage > 0.0) {
    //    // High-frequency erosion (replaces _CloudNoise3DTextureB)
    //    float2 curl = ComputeCurl(uvw);
    //    float3 uvwHigh = uvw;
    //    uvwHigh.xz += curl * (1.0 - heightFraction);
    //
    //    float hfR = worleyFbm(uvwHigh, 4.0);
    //    float hfG = worleyFbm(uvwHigh, 8.0);
    //    float hfB = worleyFbm(uvwHigh, 16.0);
    //    float highFrequencyFbm = 0.625 * hfR + 0.25 * hfG + 0.125 * hfB;
    //
    //    float highFreqNoiseModifier = lerp(highFrequencyFbm, 1.0 - highFrequencyFbm, saturate(heightFraction * 10.0));
    //    finalCloud = Remap(baseCloudWithCoverage, highFreqNoiseModifier * 0.2, 1.0, 0.0, 1.0);
    //}
    //
    //return saturate(finalCloud);
}
#endif