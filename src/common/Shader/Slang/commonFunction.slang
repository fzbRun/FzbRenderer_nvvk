#ifndef FZBRENDERER_COMMON_FUNCTION_SLANG
#define FZBRENDERER_COMMON_FUNCTION_SLANG

#include "common/Shader/shaderStructType.h"
#include "nvshaders/constants.h.slang"
#include "nvshaders/random.h.slang"

//---------------------------------------------------debug---------------------------------------------------
void printfAABB(AABB aabb) {
    printf("aabb: %f %f %f %f %f %f\n",
        aabb.minimum.x, aabb.minimum.y, aabb.minimum.z,
        aabb.maximum.x, aabb.maximum.y, aabb.maximum.z,
    );
}
//---------------------------------------------------Number---------------------------------------------------
int FloatToOrderedInt(float v)
{
    uint bits = asuint(v);
    uint ordered = (bits & 0x80000000u) != 0
        ? ~bits
        : (bits ^ 0x80000000u);
    return asint(ordered ^ 0x80000000u);
}
float OrderedIntToFloat(int orderedInt)
{
    uint ordered = asuint(orderedInt) ^ 0x80000000u;
    uint bits = (ordered & 0x80000000u) != 0
        ? (ordered ^ 0x80000000u)
        : ~ordered;
    return asfloat(bits);
}
//---------------------------------------------------Matrix---------------------------------------------------
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
float2x2 Inverse2x2(float2x2 m) {
    // 计算行列式：det = a*d - b*c
    float det = m[0][0] * m[1][1] - m[0][1] * m[1][0];

    // 假设 det != 0（实际使用时需保证）
    float invDet = 1.0f / det;

    // 使用向量运算一次性构造两行
    float2x2 result;
    result[0] = float2(m[1][1], -m[0][1]) * invDet;
    result[1] = float2(-m[1][0], m[0][0]) * invDet;

    return result;
}
//---------------------------------------------------AABB---------------------------------------------------
inline bool checkPointInAABB(float3 pos, AABB aabb) {
    return all((pos - aabb.minimum >= 0.0f) && (aabb.maximum - pos >= 0.0f));
}
inline float getDistanceToAABB(float3 pos, AABB aabb) {
    float3 center = (aabb.maximum + aabb.minimum) * 0.5f;
    float3 centerToPos = abs(center - pos) + 1e-16f;
    return length(centerToPos);
}
inline float3 getAABBPoint(float3 sampleDir, float3 aabbCenter, float3 aabbHalfSize) {
    float3 invDir = 1.0f / max(abs(sampleDir), 1e-6f);

    float3 signHalfSize = sign(aabbHalfSize);
    float3 aabbHalfSize2 = (signHalfSize * -1.0f + 1.0f) * 3.402823466e+36F + aabbHalfSize;

    float3 tCandidate = aabbHalfSize2 * invDir;
    float t = min(min(tCandidate.x, tCandidate.y), tCandidate.z);
    return aabbCenter + sampleDir * t * signHalfSize;
}
float3 getAABBPoint(AABB aabb, float3 x, float3 sampleDir) {
    float tEnter = -1e30;
    float tExit = 1e30;

    for (int i = 0; i < 3; ++i)
    {
        float d = sampleDir[i];
        float o = x[i];

        if (abs(d) < 1e-12)
        {
            if (o < aabb.minimum[i] || o > aabb.maximum[i]) return x;
            continue;
        }

        float t0 = (aabb.minimum[i] - o) / d;
        float t1 = (aabb.maximum[i] - o) / d;
        float tMin = min(t0, t1);
        float tMax = max(t0, t1);

        tEnter = max(tEnter, tMin);
        tExit = min(tExit, tMax);

        if (tEnter > tExit) return x;
    }

    return x + sampleDir * tExit;
}
float PointToAABBDistance(float3 pos, AABB aabb)
{
    float dx = max(aabb.minimum.x - pos.x, max(0.0f, pos.x - aabb.maximum.x));
    float dy = max(aabb.minimum.y - pos.y, max(0.0f, pos.y - aabb.maximum.y));
    float dz = max(aabb.minimum.z - pos.z, max(0.0f, pos.z - aabb.maximum.z));
    return sqrt(max(dx * dx + dy * dy + dz * dz, 1e-6f));
}
float AABBDistance(AABB aabb1, AABB aabb2)
{
    float3 d1 = aabb2.minimum - aabb1.maximum;
    float3 d2 = aabb1.minimum - aabb2.maximum;
    float3 gap = max(d1, d2);
    gap = max(gap, 0.0);
    return length(gap);
}
float AABBContainmentRatio(inout AABB x, AABB y) {
    float3 origLen = x.maximum - x.minimum;
    float3 yLen = y.maximum - y.minimum;

    x.minimum = max(x.minimum, y.minimum);
    x.maximum = min(x.maximum, y.maximum);

    float3 intersectLen = x.maximum - x.minimum;
    if (intersectLen.x < -0.001f || intersectLen.y < -0.001f || intersectLen.z < -0.001f) return 0.0f;

    intersectLen = abs(intersectLen);

    float ratio = 1.0f;
    for (int i = 0; i < 3; ++i) {
        if (abs(yLen[i]) < 0.0001f) continue;

        if (origLen[i] > 0.0f) ratio *= intersectLen[i] / origLen[i];
        else ratio *= 1.0f;
    }
    return ratio;
}
float3 RandomPointOnAABBSurface(inout uint seed, AABB aabb)
{
    float3 extent = aabb.maximum - aabb.minimum;

    float3 axisArea;
    axisArea.x = 2.0f * extent.y * extent.z;
    axisArea.y = 2.0f * extent.x * extent.z;
    axisArea.z = 2.0f * extent.x * extent.y;
    float totalArea = axisArea.x + axisArea.y + axisArea.z;

    if (totalArea <= 0.0f) return aabb.minimum;

    float r = rand(seed) * totalArea;
    int axis;
    if (r < axisArea.x) axis = 0;
    else if (r < axisArea.x + axisArea.y) axis = 1;
    else axis = 2;

    bool useMax = rand(seed) >= 0.5f;

    float u = rand(seed);
    float v = rand(seed);
    float3 pt;

    if (axis == 0)
    {
        float y = aabb.minimum.y + u * extent.y;
        float z = aabb.minimum.z + v * extent.z;
        pt = useMax ? float3(aabb.maximum.x, y, z)
                    : float3(aabb.minimum.x, y, z);
    }
    else if (axis == 1)
    {
        float x = aabb.minimum.x + u * extent.x;
        float z = aabb.minimum.z + v * extent.z;
        pt = useMax ? float3(x, aabb.maximum.y, z)
                    : float3(x, aabb.minimum.y, z);
    }
    else
    {
        float x = aabb.minimum.x + u * extent.x;
        float y = aabb.minimum.y + v * extent.y;
        pt = useMax ? float3(x, y, aabb.maximum.z)
                    : float3(x, y, aabb.minimum.z);
    }
    return pt;
}
//------------------------------------------------SphericalFibonacci---------------------------------------------
#define M_PHI 1.618033988749895
//y axis is up, get worldPos
inline float3 fibSpherePoint(float i, float n) {
    float phi = 2.0 * M_PI * frac(i / M_PHI);
    float cosTheta = 1.0 - (2.0 * i + 1.0) / n;
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    return float3(cos(phi) * sinTheta, cosTheta, sin(phi) * sinTheta);
}

#define madfrac(A, B) mad((A), (B), -floor((A) * (B)))
uint inverseSF(float3 p, float n) {
    float phi = min(atan2(p.z, p.x), M_PI), cosTheta = p.y;
    float k = max(2, floor(
        log(n * M_PI * sqrt(5) * (1 - cosTheta * cosTheta))
        / log(M_PHI * M_PHI)));
    float Fk = pow(M_PHI, k) / sqrt(5);
    float F0 = round(Fk), F1 = round(Fk * M_PHI);
    float2x2 B = float2x2(
                2 * M_PI * madfrac(F0 + 1, M_PHI - 1) - 2 * M_PI * (M_PHI - 1),
                2 * M_PI * madfrac(F1 + 1, M_PHI - 1) - 2 * M_PI * (M_PHI - 1),
                -2 * F0 / n,
                -2 * F1 / n);
    float2x2 invB = Inverse2x2(B);
    float2 c = floor(mul(invB, float2(phi, cosTheta - (1 - 1 / n))));
    float d = INFINITE, j = 0;
    for (uint s = 0; s < 4; ++s) {
        float cosTheta = dot(B[1], float2(s % 2, s / 2) + c) + (1 - 1 / n);
        cosTheta = clamp(cosTheta, -1, +1) * 2 - cosTheta;
        float i = floor(n * 0.5 - cosTheta * n * 0.5);
        float3 q = fibSpherePoint(i, n);
        float squaredDistance = dot(q - p, q - p);
        if (squaredDistance < d) {
            d = squaredDistance;
            j = i;
        }
    }
    return (uint)j;
}

uint inverseSF(float3 p, float n, inout float probability) {
    float phi = min(atan2(p.z, p.x), M_PI), cosTheta = p.y;
    float k = max(2, floor(
        log(n * M_PI * sqrt(5) * (1 - cosTheta * cosTheta))
        / log(M_PHI * M_PHI)));
    float Fk = pow(M_PHI, k) / sqrt(5);
    float F0 = round(Fk), F1 = round(Fk * M_PHI);
    float2x2 B = float2x2(
                2 * M_PI * madfrac(F0 + 1, M_PHI - 1) - 2 * M_PI * (M_PHI - 1),
                2 * M_PI * madfrac(F1 + 1, M_PHI - 1) - 2 * M_PI * (M_PHI - 1),
                -2 * F0 / n,
                -2 * F1 / n);
    float2x2 invB = Inverse2x2(B);

    if (k <= 4.0f) {    //range = 90 - arcsin(sqrt(6.68 / n))
        float2 c = floor(mul(invB, float2(phi, cosTheta - (1 - 1 / n))));
        float d = INFINITE, j = 0;
        for (uint s = 0; s < 4; ++s) {
            float cosTheta = dot(B[1], float2(s % 2, s / 2) + c) + (1 - 1 / n);
            cosTheta = clamp(cosTheta, -1, +1) * 2 - cosTheta;
            float i = floor(n * 0.5 - cosTheta * n * 0.5);
            float3 q = fibSpherePoint(i, n);
            float squaredDistance = dot(q - p, q - p);
            if (squaredDistance < d) {
                d = squaredDistance;
                j = i;
            }
        }
        probability = 1.0f;
        return (uint)j;
    } else {
        float2 c = mul(invB, float2(phi, cosTheta - (1.0 - 1.0 / n)));
        int2 center = int2((int)round(c.x), (int)round(c.y));

        float h = sqrt(4.0 * M_PI / n);
        float totalWeight = 0.0;
        float weights[9]; uint idxs[9];
        uint idx_result = 0;
        for (int di = -1; di <= 1; ++di) {
            for (int dj = -1; dj <= 1; ++dj) {
                int idx_index = (di + 1) * 3 + (dj + 1);
                int2 gridCoord = center + int2(di, dj);

                float cosCorner = B[1][0] * gridCoord.x + B[1][1] * gridCoord.y + (1.0 - 1.0 / n);

                if (cosCorner > 1.0) cosCorner = 2.0 - cosCorner;
                else if (cosCorner < -1.0) cosCorner = -2.0 - cosCorner;

                float iEst = floor(n * 0.5 - cosCorner * n * 0.5);
                int idx = (int)iEst;
                idx = clamp(idx, 0, (int)n - 1);
                idxs[idx_index] = idx;

                float3 q = fibSpherePoint(idx, n);

                float delta = distance(q, p);
                float t = max(0.0, 1.0 - delta / h);
                float w = 3.0 * t * t - 2.0 * t * t * t;

                //float w = max(dot(q, p), 0.0f);

                if (w > 0.0) {
                    totalWeight += w;
                    weights[idx_index] = w;
                } else weights[idx_index] = 0.0f;
            }
        }
        if (totalWeight == 0) {
            probability = 1.0f;
            return 0;
        }
        for (int i = 0; i < 9; ++i) {
            float probability_i = weights[i] / totalWeight;
            if (probability <= probability_i) {
                idx_result = idxs[i];
                break;
            }
            probability -= probability_i;
        }
        probability = 0.0f;
        for (int i = 0; i < 9; ++i) {
            if (idxs[i] == idx_result) probability += weights[i] / totalWeight;
        }
        return idx_result;
    }

    probability = 1.0f;
    return 0;
}

uint inverseSF(float3 p, float n, inout float probability, float3 normal) {
    float phi = min(atan2(p.z, p.x), M_PI), cosTheta = p.y;
    float k = max(2, floor(
        log(n * M_PI * sqrt(5) * (1 - cosTheta * cosTheta))
        / log(M_PHI * M_PHI)));
    float Fk = pow(M_PHI, k) / sqrt(5);
    float F0 = round(Fk), F1 = round(Fk * M_PHI);
    float2x2 B = float2x2(
                2 * M_PI * madfrac(F0 + 1, M_PHI - 1) - 2 * M_PI * (M_PHI - 1),
                          2 * M_PI * madfrac(F1 + 1, M_PHI - 1) - 2 * M_PI * (M_PHI - 1),
                -2 * F0 / n,
                -2 * F1 / n);
    float2x2 invB = Inverse2x2(B);

    if (k <= 4.0f) {
        float2 c = floor(mul(invB, float2(phi, cosTheta - (1 - 1 / n))));
        float d = INFINITE, j = 0;
        for (uint s = 0; s < 4; ++s) {
            float cosTheta = dot(B[1], float2(s % 2, s / 2) + c) + (1 - 1 / n);
            cosTheta = clamp(cosTheta, -1, +1) * 2 - cosTheta;
            float i = floor(n * 0.5 - cosTheta * n * 0.5);
            float3 q = fibSpherePoint(i, n);

            if (dot(q, normal) <= 0.0) continue;

            float squaredDistance = dot(q - p, q - p);
            if (squaredDistance < d) {
                d = squaredDistance;
                j = i;
            }
        }

        probability = 1.0f;
        return (uint)j;
    } else {
        float2 c = mul(invB, float2(phi, cosTheta - (1.0 - 1.0 / n)));
        int2 center = int2((int)round(c.x), (int)round(c.y));

        float h = sqrt(4.0 * M_PI / n);
        float totalWeight = 0.0;
        float weights[9]; uint idxs[9];
        uint idx_result = 0;
        for (int di = -1; di <= 1; ++di) {
            for (int dj = -1; dj <= 1; ++dj) {
                int idx_index = (di + 1) * 3 + (dj + 1);
                int2 gridCoord = center + int2(di, dj);

                float cosCorner = B[1][0] * gridCoord.x + B[1][1] * gridCoord.y + (1.0 - 1.0 / n);
                if (cosCorner > 1.0) cosCorner = 2.0 - cosCorner;
                else if (cosCorner < -1.0) cosCorner = -2.0 - cosCorner;

                float iEst = floor(n * 0.5 - cosCorner * n * 0.5);
                int idx = (int)iEst;
                idx = clamp(idx, 0, (int)n - 1);
                idxs[idx_index] = idx;

                float3 q = fibSpherePoint(idx, n);

                float w = 0.0;
                if (dot(q, normal) > 0.0) {
                    float delta = distance(q, p);
                    float t = max(0.0, 1.0 - delta / h);
                    w = 3.0 * t * t - 2.0 * t * t * t;
                }

                if (w > 0.0) {
                    totalWeight += w;
                    weights[idx_index] = w;
                } else {
                    weights[idx_index] = 0.0;
                }
            }
        }
        if (totalWeight == 0) {
            probability = 1.0f;
            return 0; // fallback
        }
        for (int i = 0; i < 9; ++i) {
            float probability_i = weights[i] / totalWeight;
            if (probability <= probability_i) {
                idx_result = idxs[i];
                break;
            }
            probability -= probability_i;
        }
        probability = 0.0f;
        for (int i = 0; i < 9; ++i) {
            if (idxs[i] == idx_result) probability += weights[i] / totalWeight;
        }
        return idx_result;
    }

    probability = 1.0f;
    return 0;
}
//------------------------------------------------Low difference sequence---------------------------------------------
float RadicalInverse_VdC(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}
float2 Hammersley(uint i, uint N)
{
    return float2(float(i) / float(N), RadicalInverse_VdC(i));
}
float3 hammersleySpherePoint(int i, int n) {
    float2 randomNumber = Hammersley(i, n);
    float phi = M_TWO_PI * randomNumber.x;
    float cosTheta = 1.0f - 2.0f * randomNumber.y;
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

uint reverseBits(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return bits;
}
int inverseSH(float3 dir, int n)
{
    dir = normalize(dir);

    float phi = atan2(dir.y, dir.x);
    if (phi < 0.0f) phi += M_TWO_PI;
    int i_phi = int(round((phi / M_TWO_PI) * n)) % n;

    float cosTheta = clamp(dir.z, -1.0f, 1.0f);
    float v = (1.0f - cosTheta) * 0.5f;
    uint bits = uint(v * 4294967296.0f); // 2^32
    int i_theta = int(reverseBits(bits)) % n;

    int best_i = 0;
    float best_dot = -2.0f;

    int candidates[12];
    int count = 0;
    for (int d = -2; d <= 2; ++d)
    {
        int c1 = (i_phi + d + n) % n;
        int c2 = (i_theta + d + n) % n;
        candidates[count++] = c1;
        candidates[count++] = c2;
    }

    for (int j = 0; j < count; ++j)
    {
        int i = candidates[j];
        bool dup = false;
        for (int k = 0; k < j; ++k) if (candidates[k] == i) { dup = true; break; }
        if (dup) continue;

        float2 uv = Hammersley(i, n);
        float p_phi = M_TWO_PI * uv.x;
        float p_cos = 1.0f - 2.0f * uv.y;
        float p_sin = sqrt(1.0f - p_cos * p_cos);
        float3 p = float3(p_sin * cos(p_phi), p_sin * sin(p_phi), p_cos);

        float dot_p = dot(dir, p);
        if (dot_p > best_dot)
        {
            best_dot = dot_p;
            best_i = i;
        }
    }

    return best_i;
}

int inverseSH(float3 dir, int n, inout float probability) {
    dir = normalize(dir);

    float phi = atan2(dir.y, dir.x);
    if (phi < 0.0f) phi += M_TWO_PI;
    int i_phi = int(round((phi / M_TWO_PI) * n)) % n;

    float cosTheta = clamp(dir.z, -1.0f, 1.0f);
    float v = (1.0f - cosTheta) * 0.5f;
    uint bits = uint(v * 4294967296.0f); // 2^32
    int i_theta = int(reverseBits(bits)) % n;

    int best_i = 0;
    float best_dot = -2.0f;

    int candidates[6];
    int count = 0;
    for (int d = -1; d <= 1; ++d)
    {
        int c1 = (i_phi + d + n) % n;
        int c2 = (i_theta + d + n) % n;
        candidates[count++] = c1;
        candidates[count++] = c2;
    }

    float h = sqrt(4.0 * M_PI / n);
    float weights[6]; float weightSum = 0.0f;
    for (int j = 0; j < count; ++j) {
        weights[j] = 0.0f;

        int i = candidates[j];
        bool dup = false;
        for (int k = 0; k < j; ++k) {
            if (candidates[k] == i) {
                weightSum += weights[k];
                weights[k] *= 2.0f;
                dup = true; break;
            }
        }
        if (dup) continue;

        float2 uv = Hammersley(i, n);
        float p_phi = M_TWO_PI * uv.x;
        float p_cos = 1.0f - 2.0f * uv.y;
        float p_sin = sqrt(1.0f - p_cos * p_cos);
        float3 p = float3(p_sin * cos(p_phi), p_sin * sin(p_phi), p_cos);

        float delta = distance(dir, p);
        float t = max(0.0, 1.0 - delta / h);
        weights[j] = 3.0 * t * t - 2.0 * t * t * t;

        //weights[j] = max(dot(dir, p), 0.0f);

        weightSum += weights[j];
    }
    if (weightSum == 0.0f) {
        probability = 1.0f;
        return 0;
    }
    for (int j = 0; j < count; ++j) {
        float probability_j = weights[j] / weightSum;
        if (probability <= probability_j) {
            best_i = candidates[j];
            probability = probability_j;
            break;
        }
        probability -= probability_j;
    }

    return best_i;
}

//------------------------------------------------sphereMeanSample---------------------------------------------
float3 sample_hemisphere_mean(inout uint seed) {
    float r1 = rand(seed); float r2 = rand(seed); // mean sample hemisphere
    float cosTheta = r1; float phi = 2.0f * M_PI * r2;
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}
float3 sample_sphere_mean(inout uint seed) {
    float r1 = rand(seed); float r2 = rand(seed); // mean sample hemisphere
    float cosTheta = 1 - 2 * r1; float phi = 2.0f * M_PI * r2;
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}
//------------------------------------------------sphereRectangleSample---------------------------------------------
struct Quadrilateral {
    float3 startPos;
    float3 normal;
    float3 edge1;
    float3 edge2;
};
float3 sphereRectangleSample(Quadrilateral quad, float3 hitPos, float u, float v, inout float pdf) {
    float exl = length(quad.edge1);
    float eyl = length(quad.edge2);
    float3 axisX = quad.edge1 / exl;
    float3 axisY = quad.edge2 / eyl;
    float3 axisZ = quad.normal;

    float3 d = quad.startPos - hitPos;
    float z0 = dot(d, axisZ);
    if (z0 > 0) {
        axisZ *= -1.0f;
        z0 *= -1.0f;
    }
    float z0sq = z0 * z0;

    float x0 = dot(d, axisX);
    float y0 = dot(d, axisY);
    float x1 = x0 + exl;
    float y1 = y0 + eyl;
    float y0sq = y0 * y0;
    float y1sq = y1 * y1;

    float3 v00 = float3(x0, y0, z0);
    float3 v01 = float3(x0, y1, z0);
    float3 v10 = float3(x1, y0, z0);
    float3 v11 = float3(x1, y1, z0);

    float3 n0 = float3(0.0f, z0, -y0); // normalize(cross(v00, v10));
    n0.z /= sqrt(z0sq + y0sq);         // 后续只有z需要被用到，x，y会乘以0，不用管
    float3 n1 = float3(-z0, 0.0f, x1); // normalize(cross(v10, v11));
    n1.z /= sqrt(z0sq + x1 * x1);
    float3 n2 = float3(0.0f, -z0, y1); // normalize(cross(v11, v01));
    n2.z /= sqrt(z0sq + y1sq);
    float3 n3 = float3(z0, 0.0f, -x0); // normalize(cross(v01, v00));
    n3.z /= sqrt(z0sq + x0 * x0);

    float g0 = acos(-n0.z * n1.z); // acos(-dot(n0, n1));
    float g1 = acos(-n1.z * n2.z); // acos(-dot(n1, n2));
    float g2 = acos(-n2.z * n3.z); // acos(-dot(n2, n3));
    float g3 = acos(-n3.z * n0.z); // acos(-dot(n3, n0));

    float b0 = n0.z;
    float b1 = n2.z;
    float b0sq = b0 * b0;
    float k = 2 * M_PI - g2 - g3;
    float S = g0 + g1 - k;

    pdf = 1.0f / S;

    float au = u * S + k;
    float fu = (cos(au) * b0 - b1) / sin(au);
    float cu = 1.0f / sqrt(fu * fu + b0sq) * (fu > 0.0f ? 1.0f : -1.0f);
    cu = clamp(cu, -1.0f, 1.0f);
    float xu = -(cu * z0) / sqrt(1.0f - cu * cu);
    xu = clamp(xu, x0, x1);

    float dl = sqrt(xu * xu + z0sq);
    float h0 = y0 / sqrt(dl * dl + y0sq);
    float h1 = y1 / sqrt(dl * dl + y1sq);
    float hv = h0 + v * (h1 - h0);
    float hv2 = hv * hv;
    float yv = hv2 < 1 - 1e-6 ? (hv * dl) / sqrt(1.0f - hv2) : y1;

    return xu * axisX + yv * axisY + z0 * axisZ; // 未归一化
}

struct Quadrilateral_initialization { // For a surface that needs to be sampled multiple times, it can be initialized once and reused
    float3 o, x, y, z;
    float z0, z0sq;
    float x0, y0, y0sq;    // rectangle coords in ’R’
    float x1, y1, y1sq;    //
    float b0, b1, b0sq, k; // misc precomputed constants
    float S;               // solid angle of ’Q’
};
Quadrilateral_initialization initQuadrilateral(float3 hitPos, float3 quadRangleStartPos, float3 ex, float3 ey) {
    Quadrilateral_initialization quadrangle;
    quadrangle.o = hitPos;
    float exl = length(ex);
    float eyl = length(ey);
    quadrangle.x = ex / exl;
    quadrangle.y = ey / eyl;
    quadrangle.z = cross(quadrangle.x, quadrangle.y);

    float3 d = quadRangleStartPos - quadrangle.o;
    quadrangle.z0 = dot(d, quadrangle.z);
    if (quadrangle.z0 > 0) {
        quadrangle.z *= -1.0f;
        quadrangle.z0 *= -1.0f;
    }
    quadrangle.z0sq = quadrangle.z0 * quadrangle.z0;

    quadrangle.x0 = dot(d, quadrangle.x);
    quadrangle.y0 = dot(d, quadrangle.y);
    quadrangle.x1 = quadrangle.x0 + exl;
    quadrangle.y1 = quadrangle.y0 + eyl;
    quadrangle.y0sq = quadrangle.y0 * quadrangle.y0;
    quadrangle.y1sq = quadrangle.y1 * quadrangle.y1;

    float3 v00 = float3(quadrangle.x0, quadrangle.y0, quadrangle.z0);
    float3 v01 = float3(quadrangle.x0, quadrangle.y1, quadrangle.z0);
    float3 v10 = float3(quadrangle.x1, quadrangle.y0, quadrangle.z0);
    float3 v11 = float3(quadrangle.x1, quadrangle.y1, quadrangle.z0);

    float3 n0 = float3(0.0f, quadrangle.z0, -quadrangle.y0); // glm::normalize(cross(v00, v10));
    n0.z /= sqrt(quadrangle.z0sq + quadrangle.y0sq);         // 后续只有z需要被用到，x，y会乘以0，不用管
    float3 n1 = float3(-quadrangle.z0, 0.0f, quadrangle.x1); // glm::normalize(cross(v10, v11));
    n1.z /= sqrt(quadrangle.z0sq + quadrangle.x1 * quadrangle.x1);
    float3 n2 = float3(0.0f, -quadrangle.z0, quadrangle.y1); // glm::normalize(cross(v11, v01));
    n2.z /= sqrt(quadrangle.z0sq + quadrangle.y1sq);
    float3 n3 = float3(quadrangle.z0, 0.0f, -quadrangle.x0); // glm::normalize(cross(v01, v00));
    n3.z /= sqrt(quadrangle.z0sq + quadrangle.x0 * quadrangle.x0);

    float g0 = acos(-n0.z * n1.z); // glm::acos(-glm::dot(n0, n1));
    float g1 = acos(-n1.z * n2.z); // glm::acos(-glm::dot(n1, n2));
    float g2 = acos(-n2.z * n3.z); // glm::acos(-glm::dot(n2, n3));
    float g3 = acos(-n3.z * n0.z); // glm::acos(-glm::dot(n3, n0));

    quadrangle.b0 = n0.z;
    quadrangle.b1 = n2.z;
    quadrangle.b0sq = quadrangle.b0 * quadrangle.b0;
    quadrangle.k = 2 * M_PI - g2 - g3;
    quadrangle.S = g0 + g1 - quadrangle.k;

    return quadrangle;
}
float3 sphericalRectangleSample(Quadrilateral_initialization quadrangle, float u, float v, inout float pdf) {
    pdf *= 1.0f / quadrangle.S;

    float au = u * quadrangle.S + quadrangle.k;
    float fu = (cos(au) * quadrangle.b0 - quadrangle.b1) / sin(au);
    float cu = 1.0f / sqrt(fu * fu + quadrangle.b0sq) * (fu > 0.0f ? 1.0f : -1.0f);
    cu = clamp(cu, -1.0f, 1.0f);
    float xu = -(cu * quadrangle.z0) / sqrt(1.0f - cu * cu);
    xu = clamp(xu, quadrangle.x0, quadrangle.x1);

    float dl = sqrt(xu * xu + quadrangle.z0sq);
    float h0 = quadrangle.y0 / sqrt(dl * dl + quadrangle.y0sq);
    float h1 = quadrangle.y1 / sqrt(dl * dl + quadrangle.y1sq);
    float hv = h0 + v * (h1 - h0);
    float hv2 = hv * hv;
    float yv = hv2 < 1 - 1e-6 ? (hv * dl) / sqrt(1.0f - hv2) : quadrangle.y1;

    return xu * quadrangle.x + yv * quadrangle.y + quadrangle.z0 * quadrangle.z; // 未归一化
}
#endif