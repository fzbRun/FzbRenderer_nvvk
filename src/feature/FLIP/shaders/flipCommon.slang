#ifndef FLIP_COMMON_SLANG
#define FLIP_COMMON_SLANG

// FLIP color spaces and color metric (paper Section 4.1).

static const float FLIP_PI = 3.14159265358979323846f;
static const float3 FLIP_D65 = float3(0.950428545f, 1.000000000f, 1.088900371f);
static const float3 FLIP_INV_D65 = float3(1.052156925f, 1.000000000f, 0.918357670f);

float flipSRGBToLinear(float c) {
    return c <= 0.04045f ? c / 12.92f : pow((c + 0.055f) / 1.055f, 2.4f);
}

float3 flipSRGBToLinear(float3 c) {
    return float3(flipSRGBToLinear(c.x), flipSRGBToLinear(c.y), flipSRGBToLinear(c.z));
}

float3 flipLinearRGBToXYZ(float3 rgb) {
    const float a11 = 10135552.0f / 24577794.0f;
    const float a12 = 8788810.0f / 24577794.0f;
    const float a13 = 4435075.0f / 24577794.0f;
    const float a21 = 2613072.0f / 12288897.0f;
    const float a22 = 8788810.0f / 12288897.0f;
    const float a23 = 887015.0f / 12288897.0f;
    const float a31 = 1425312.0f / 73733382.0f;
    const float a32 = 8788810.0f / 73733382.0f;
    const float a33 = 70074185.0f / 73733382.0f;

    return float3(a11 * rgb.x + a12 * rgb.y + a13 * rgb.z,
                  a21 * rgb.x + a22 * rgb.y + a23 * rgb.z,
                  a31 * rgb.x + a32 * rgb.y + a33 * rgb.z);
}

float3 flipXYZToLinearRGB(float3 xyz) {
    const float a11 = 3.241003275f;
    const float a12 = -1.537398934f;
    const float a13 = -0.498615861f;
    const float a21 = -0.969224334f;
    const float a22 = 1.875930071f;
    const float a23 = 0.041554224f;
    const float a31 = 0.055639423f;
    const float a32 = -0.204011202f;
    const float a33 = 1.057148933f;

    return float3(a11 * xyz.x + a12 * xyz.y + a13 * xyz.z,
                  a21 * xyz.x + a22 * xyz.y + a23 * xyz.z,
                  a31 * xyz.x + a32 * xyz.y + a33 * xyz.z);
}

float3 flipXYZToCIELab(float3 xyz) {
    const float delta = 6.0f / 29.0f;
    const float deltaSquare = delta * delta;
    const float deltaCube = delta * deltaSquare;
    const float factor = 1.0f / (3.0f * deltaSquare);
    const float term = 4.0f / 29.0f;

    xyz *= FLIP_INV_D65;
    const float fx = xyz.x > deltaCube ? pow(xyz.x, 1.0f / 3.0f) : factor * xyz.x + term;
    const float fy = xyz.y > deltaCube ? pow(xyz.y, 1.0f / 3.0f) : factor * xyz.y + term;
    const float fz = xyz.z > deltaCube ? pow(xyz.z, 1.0f / 3.0f) : factor * xyz.z + term;

    return float3(116.0f * fy - 16.0f, 500.0f * (fx - fy), 200.0f * (fy - fz));
}

// YyCxCz used by FLIP is a linearized version of CIELAB.
float3 flipXYZToYCxCz(float3 xyz) {
    xyz *= FLIP_INV_D65;
    const float y = 116.0f * xyz.y - 16.0f;
    const float cx = 500.0f * (xyz.x - xyz.y);
    const float cz = 200.0f * (xyz.y - xyz.z);
    return float3(y, cx, cz);
}

float3 flipYCxCzToXYZ(float3 ycxcz) {
    const float y = (ycxcz.x + 16.0f) / 116.0f;
    const float cx = ycxcz.y / 500.0f;
    const float cz = ycxcz.z / 200.0f;
    return float3(y + cx, y, y - cz) * FLIP_D65;
}

float flipHunt(float luminance, float chrominance) {
    return 0.01f * luminance * chrominance;
}

float flipHyAB(float3 refColor, float3 testColor) {
    const float dL = abs(refColor.x - testColor.x);
    const float da = refColor.y - testColor.y;
    const float db = refColor.z - testColor.z;
    return dL + sqrt(da * da + db * db);
}

#endif
