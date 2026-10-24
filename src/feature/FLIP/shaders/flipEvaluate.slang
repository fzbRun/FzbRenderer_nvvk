#include "feature/FLIP/FLIPShaderio.h"
#include "feature/FLIP/shaders/flipCommon.slang"

[[vk::push_constant]] ConstantBuffer<FLIPPushConstant> pushConst;

[[vk::binding(BindingPoints_FLIP::eHorizontalReference)]] RWTexture2D<float4> horizontalReference;
[[vk::binding(BindingPoints_FLIP::eHorizontalReferenceCz)]] RWTexture2D<float4> horizontalReferenceCz;
[[vk::binding(BindingPoints_FLIP::eHorizontalTest)]] RWTexture2D<float4> horizontalTest;
[[vk::binding(BindingPoints_FLIP::eHorizontalTestCz)]] RWTexture2D<float4> horizontalTestCz;

[[vk::binding(BindingPoints_FLIP::eResultImage)]] RWTexture2D<float4> resultImage;

[[vk::binding(BindingPoints_FLIP::eKernelBuffer)]] StructuredBuffer<FLIPKernelData> flipKernel;

// Second dimension (vertical): finish CSF/feature filtering and compute the final FLIP value.
[numthreads(32, 32, 1)]
[shader("compute")]
void computeMain_evaluate(uint2 threadIndex : SV_DispatchThreadID) {
    if (any(threadIndex >= pushConst.screenSize))
        return;

    const int height = int(pushConst.screenSize.y);
    const int2 pixel = int2(threadIndex);

    float yRef = 0.0f, cxRef = 0.0f, cz1Ref = 0.0f, cz2Ref = 0.0f;
    float yTest = 0.0f, cxTest = 0.0f, cz1Test = 0.0f, cz2Test = 0.0f;

    for (int i = -pushConst.csfRadius; i <= pushConst.csfRadius; ++i) {
        const int sampleY = clamp(pixel.y + i, 0, height - 1);
        const uint2 sampleIndex = uint2(pixel.x, sampleY);
        const int kernelIndex = FLIP_MAX_KERNEL_RADIUS + i;

        const float4 refColor = horizontalReference[sampleIndex];
        const float4 refCz = horizontalReferenceCz[sampleIndex];
        const float4 testColor = horizontalTest[sampleIndex];
        const float4 testCz = horizontalTestCz[sampleIndex];

        yRef += flipKernel[0].csfY[kernelIndex] * refColor.x;
        cxRef += flipKernel[0].csfCx[kernelIndex] * refColor.y;
        cz1Ref += flipKernel[0].csfCz1[kernelIndex] * refCz.x;
        cz2Ref += flipKernel[0].csfCz2[kernelIndex] * refCz.y;

        yTest += flipKernel[0].csfY[kernelIndex] * testColor.x;
        cxTest += flipKernel[0].csfCx[kernelIndex] * testColor.y;
        cz1Test += flipKernel[0].csfCz1[kernelIndex] * testCz.x;
        cz2Test += flipKernel[0].csfCz2[kernelIndex] * testCz.y;
    }

    const float czRef = cz1Ref + cz2Ref;
    const float czTest = cz1Test + cz2Test;

    float dxRef = 0.0f, ddxRef = 0.0f, dyRef = 0.0f, ddyRef = 0.0f;
    float dxTest = 0.0f, ddxTest = 0.0f, dyTest = 0.0f, ddyTest = 0.0f;

    for (int i = -pushConst.featureRadius; i <= pushConst.featureRadius; ++i) {
        const int sampleY = clamp(pixel.y + i, 0, height - 1);
        const uint2 sampleIndex = uint2(pixel.x, sampleY);
        const int kernelIndex = FLIP_MAX_KERNEL_RADIUS + i;

        const float4 refColor = horizontalReference[sampleIndex];
        const float4 refCz = horizontalReferenceCz[sampleIndex];
        const float4 testColor = horizontalTest[sampleIndex];
        const float4 testCz = horizontalTestCz[sampleIndex];

        // Horizontal pass already applied the 1st/2nd derivative; only apply the vertical Gaussian here.
        dxRef += flipKernel[0].featureGaussian[kernelIndex] * refColor.z;
        ddxRef += flipKernel[0].featureGaussian[kernelIndex] * refColor.w;
        dxTest += flipKernel[0].featureGaussian[kernelIndex] * testColor.z;
        ddxTest += flipKernel[0].featureGaussian[kernelIndex] * testColor.w;

        // Horizontal pass already applied the Gaussian; apply the vertical 1st/2nd derivative here.
        dyRef += flipKernel[0].featureFirstDerivative[kernelIndex] * refCz.z;
        ddyRef += flipKernel[0].featureSecondDerivative[kernelIndex] * refCz.z;
        dyTest += flipKernel[0].featureFirstDerivative[kernelIndex] * testCz.z;
        ddyTest += flipKernel[0].featureSecondDerivative[kernelIndex] * testCz.z;
    }

    // Color pipeline: filtered YyCxCz -> linear RGB -> clamp -> CIELAB -> Hunt adjustment.
    float3 filteredRef = flipYCxCzToXYZ(float3(yRef, cxRef, czRef));
    float3 filteredTest = flipYCxCzToXYZ(float3(yTest, cxTest, czTest));
    filteredRef = clamp(flipXYZToLinearRGB(filteredRef), 0.0f, 1.0f);
    filteredTest = clamp(flipXYZToLinearRGB(filteredTest), 0.0f, 1.0f);

    float3 labRef = flipXYZToCIELab(flipLinearRGBToXYZ(filteredRef));
    float3 labTest = flipXYZToCIELab(flipLinearRGBToXYZ(filteredTest));
    labRef.y = flipHunt(labRef.x, labRef.y);
    labRef.z = flipHunt(labRef.x, labRef.z);
    labTest.y = flipHunt(labTest.x, labTest.y);
    labTest.z = flipHunt(labTest.x, labTest.z);

    float colorDifference = pow(max(flipHyAB(labRef, labTest), 0.0f), pushConst.qC);
    if (colorDifference < pushConst.pcCMax) {
        colorDifference *= pushConst.pT / max(pushConst.pcCMax, 1.0e-6f);
    }
    else {
        colorDifference = pushConst.pT +
            ((colorDifference - pushConst.pcCMax) / max(pushConst.cMax - pushConst.pcCMax, 1.0e-6f)) *
            (1.0f - pushConst.pT);
    }
    colorDifference = clamp(colorDifference, 0.0f, 1.0f);

    // Feature pipeline: edge = ||grad I||, point = ||laplacian I||.
    const float edgeRef = sqrt(dxRef * dxRef + dyRef * dyRef);
    const float edgeTest = sqrt(dxTest * dxTest + dyTest * dyTest);
    const float pointRef = sqrt(ddxRef * ddxRef + ddyRef * ddyRef);
    const float pointTest = sqrt(ddxTest * ddxTest + ddyTest * ddyTest);

    const float edgeDifference = abs(edgeRef - edgeTest);
    const float pointDifference = abs(pointRef - pointTest);
    const float normalization = 1.0f / sqrt(2.0f);
    float featureDifference = pow(max(normalization * max(edgeDifference, pointDifference), 0.0f), pushConst.qF);
    featureDifference = clamp(featureDifference, 0.0f, 1.0f);

    const float flipValue = colorDifference > 0.0f
        ? pow(colorDifference, max(0.0f, 1.0f - featureDifference))
        : 0.0f;

    float3 displayColor = float3(flipValue, flipValue, flipValue);
    if (pushConst.debugView == 1)
        displayColor = float3(colorDifference, 0.0f, 0.0f);
    else if (pushConst.debugView == 2)
        displayColor = float3(0.0f, featureDifference, 0.0f);

    // In the default view R is the final FLIP value; A always stores the final value.
    resultImage[threadIndex] = float4(displayColor, flipValue);
}
