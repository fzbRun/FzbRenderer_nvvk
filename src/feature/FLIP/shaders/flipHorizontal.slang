#include "feature/FLIP/FLIPShaderio.h"
#include "feature/FLIP/shaders/flipCommon.slang"

[[vk::push_constant]] ConstantBuffer<FLIPPushConstant> pushConst;

[[vk::binding(BindingPoints_FLIP::eOpponentReference)]] RWTexture2D<float4> opponentReference;
[[vk::binding(BindingPoints_FLIP::eOpponentTest)]] RWTexture2D<float4> opponentTest;

[[vk::binding(BindingPoints_FLIP::eHorizontalReference)]] RWTexture2D<float4> horizontalReference;
[[vk::binding(BindingPoints_FLIP::eHorizontalReferenceCz)]] RWTexture2D<float4> horizontalReferenceCz;
[[vk::binding(BindingPoints_FLIP::eHorizontalTest)]] RWTexture2D<float4> horizontalTest;
[[vk::binding(BindingPoints_FLIP::eHorizontalTestCz)]] RWTexture2D<float4> horizontalTestCz;

[[vk::binding(BindingPoints_FLIP::eKernelBuffer)]] StructuredBuffer<FLIPKernelData> flipKernel;

// First dimension (horizontal): CSF filtering and horizontal edge/point convolution.
[numthreads(32, 32, 1)]
[shader("compute")]
void computeMain_horizontal(uint2 threadIndex : SV_DispatchThreadID) {
    if (any(threadIndex >= pushConst.screenSize))
        return;

    const int width = int(pushConst.screenSize.x);
    const int2 pixel = int2(threadIndex);

    float yRef = 0.0f, cxRef = 0.0f, cz1Ref = 0.0f, cz2Ref = 0.0f;
    float yTest = 0.0f, cxTest = 0.0f, cz1Test = 0.0f, cz2Test = 0.0f;
    float dxRef = 0.0f, ddxRef = 0.0f, gaussianRef = 0.0f;
    float dxTest = 0.0f, ddxTest = 0.0f, gaussianTest = 0.0f;

    for (int i = -pushConst.csfRadius; i <= pushConst.csfRadius; ++i) {
        const int sampleX = clamp(pixel.x + i, 0, width - 1);
        const uint2 sampleIndex = uint2(sampleX, pixel.y);
        const int kernelIndex = FLIP_MAX_KERNEL_RADIUS + i;

        const float4 refColor = opponentReference[sampleIndex];
        const float4 testColor = opponentTest[sampleIndex];

        yRef += flipKernel[0].csfY[kernelIndex] * refColor.x;
        cxRef += flipKernel[0].csfCx[kernelIndex] * refColor.y;
        cz1Ref += flipKernel[0].csfCz1[kernelIndex] * refColor.z;
        cz2Ref += flipKernel[0].csfCz2[kernelIndex] * refColor.z;

        yTest += flipKernel[0].csfY[kernelIndex] * testColor.x;
        cxTest += flipKernel[0].csfCx[kernelIndex] * testColor.y;
        cz1Test += flipKernel[0].csfCz1[kernelIndex] * testColor.z;
        cz2Test += flipKernel[0].csfCz2[kernelIndex] * testColor.z;
    }

    for (int i = -pushConst.featureRadius; i <= pushConst.featureRadius; ++i) {
        const int sampleX = clamp(pixel.x + i, 0, width - 1);
        const uint2 sampleIndex = uint2(sampleX, pixel.y);
        const int kernelIndex = FLIP_MAX_KERNEL_RADIUS + i;

        const float refY = (opponentReference[sampleIndex].x + 16.0f) / 116.0f;
        const float testY = (opponentTest[sampleIndex].x + 16.0f) / 116.0f;

        dxRef += flipKernel[0].featureFirstDerivative[kernelIndex] * refY;
        ddxRef += flipKernel[0].featureSecondDerivative[kernelIndex] * refY;
        gaussianRef += flipKernel[0].featureGaussian[kernelIndex] * refY;

        dxTest += flipKernel[0].featureFirstDerivative[kernelIndex] * testY;
        ddxTest += flipKernel[0].featureSecondDerivative[kernelIndex] * testY;
        gaussianTest += flipKernel[0].featureGaussian[kernelIndex] * testY;
    }

    horizontalReference[threadIndex] = float4(yRef, cxRef, dxRef, ddxRef);
    horizontalTest[threadIndex] = float4(yTest, cxTest, dxTest, ddxTest);
    // Cz is stored as two separable Gaussian components to avoid cross terms.
    horizontalReferenceCz[threadIndex] = float4(cz1Ref, cz2Ref, gaussianRef, 0.0f);
    horizontalTestCz[threadIndex] = float4(cz1Test, cz2Test, gaussianTest, 0.0f);
}
