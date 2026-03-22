#include "common/Shader/shaderStructType.h"
#include "nvshaders/constants.h.slang"
#include "nvshaders/random.h.slang"

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
    return all((pos - aabb.minimum >= -0.001f) && (aabb.maximum - pos >= -0.001f));
}
inline float3 getAABBPoint(float3 sampleDir, float3 aabbCenter, float3 aabbHalfSize) {
    float3 invDir = 1.0f / max(abs(sampleDir), 1e-6f);

    float3 signHalfSize = sign(aabbHalfSize);
    float3 aabbHalfSize2 = (signHalfSize * -1.0f + 1.0f) * 3.402823466e+36F + aabbHalfSize;

    float3 tCandidate = aabbHalfSize2 * invDir;
    float t = min(min(tCandidate.x, tCandidate.y), tCandidate.z);
    return aabbCenter + sampleDir * t * signHalfSize;
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
        for (int i = 0; i < 9; ++i) weights[i] /= totalWeight;
        for (int i = 0; i < 9; ++i) {
            if (probability <= weights[i]) {
                idx_result = idxs[i];
                break;
            }
            probability -= weights[i];
        }
        probability = 0.0f;
        for (int i = 0; i < 9; ++i) {
            if (idxs[i] == idx_result) probability += weights[i];
        }
        return idx_result;
    }

    probability = 1.0f;
    return 0;
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

