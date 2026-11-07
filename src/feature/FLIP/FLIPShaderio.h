#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_FLIP_SHADERIO_H
#define FZBRENDERER_FLIP_SHADERIO_H

// Maximum 1D kernel radius (in pixels). Larger kernels are clamped.
#define FLIP_MAX_KERNEL_RADIUS 32
#define FLIP_MAX_KERNEL_SIZE (FLIP_MAX_KERNEL_RADIUS * 2 + 1)

NAMESPACE_SHADERIO_BEGIN()

// 1D kernels precomputed on the CPU, so the GPU only samples and accumulates.
struct FLIPKernelData {
	float csfY[FLIP_MAX_KERNEL_SIZE];					// luminance CSF
	float csfCx[FLIP_MAX_KERNEL_SIZE];					// red-green CSF
	float csfCz1[FLIP_MAX_KERNEL_SIZE];					// blue-yellow CSF, first Gaussian
	float csfCz2[FLIP_MAX_KERNEL_SIZE];					// blue-yellow CSF, second Gaussian
	float featureGaussian[FLIP_MAX_KERNEL_SIZE];		// Gaussian for edge/point detection
	float featureFirstDerivative[FLIP_MAX_KERNEL_SIZE];	// first derivative of the Gaussian
	float featureSecondDerivative[FLIP_MAX_KERNEL_SIZE];	// second derivative of the Gaussian
};

struct FLIPPushConstant {
	uint2 screenSize;
	int csfRadius;
	int featureRadius;
	float pixelsPerDegree;
	float cMax;
	float pcCMax;
	float pT;
	float qC;
	float qF;
	int referenceIsSRGB;
	int testIsSRGB;
	int debugView;
	int padding[3];
};

enum class BindingPoints_FLIP {
	eReferenceImage = 0,
	eTestImage,
	eOpponentReference,
	eOpponentTest,
	eHorizontalReference,
	eHorizontalReferenceCz,
	eHorizontalTest,
	eHorizontalTestCz,
	eResultImage,
	eKernelBuffer,
};

NAMESPACE_SHADERIO_END()

#endif
