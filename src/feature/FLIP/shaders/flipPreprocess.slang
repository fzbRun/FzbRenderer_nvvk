#include "feature/FLIP/FLIPShaderio.h"
#include "feature/FLIP/shaders/flipCommon.slang"

[[vk::push_constant]] ConstantBuffer<FLIPPushConstant> pushConst;

[[vk::binding(BindingPoints_FLIP::eReferenceImage)]] Texture2D<float4> referenceImage;
[[vk::binding(BindingPoints_FLIP::eTestImage)]] Texture2D<float4> testImage;

[[vk::binding(BindingPoints_FLIP::eOpponentReference)]] RWTexture2D<float4> opponentReference;
[[vk::binding(BindingPoints_FLIP::eOpponentTest)]] RWTexture2D<float4> opponentTest;

[numthreads(32, 32, 1)]
[shader("compute")]
void computeMain_preprocess(uint2 threadIndex : SV_DispatchThreadID) {
    if (any(threadIndex >= pushConst.screenSize))
        return;

    float3 refColor = referenceImage[threadIndex].rgb;
    float3 testColor = testImage[threadIndex].rgb;

    if (pushConst.referenceIsSRGB != 0)
        refColor = flipSRGBToLinear(refColor);
    if (pushConst.testIsSRGB != 0)
        testColor = flipSRGBToLinear(testColor);

    const float3 refOpponent = flipXYZToYCxCz(flipLinearRGBToXYZ(refColor));
    const float3 testOpponent = flipXYZToYCxCz(flipLinearRGBToXYZ(testColor));

    opponentReference[threadIndex] = float4(refOpponent.x, refOpponent.y, refOpponent.z, 1.0f);
    opponentTest[threadIndex] = float4(testOpponent.x, testOpponent.y, testOpponent.z, 1.0f);
}
