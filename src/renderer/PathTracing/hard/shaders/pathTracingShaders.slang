#include "shaderio.h"
#include "nvshaders/constants.h.slang"

[[vk::push_constant]] ConstantBuffer<TutoPushConstant> pushConst;
[[vk::binding(BindingPoints::eTextures, 0)]] Sampler2D textures[];
[[vk::binding(BindingPoints::eTlas, 1)]] RayTracingAccelerationStructure topLevelAS;
[[vk::binding(BindingPoints::eOutImage, 1)]] RWTexture2D<flaot4> outImage;

struct HitPayLoad{
	float3 color;
	float weight;
	int depth;
};

[shader("raygeneration")]
void raygenMain(){
	float2 launchID = (float2)DispatchRaysIndex().xy;
	float2 launchSize = (float2)DispatchRaysDimensions().xy;

	GltfSceneInfo sceneInfo = pushConst.sceneInfoAddress[0];

	const uint rayFlags = 0;

	const float2 clipCoords = launchID / launchSize * 2.0f - 1.0f;

	const float4 viewCoords = mul(float4(clipCoords, 1.0f, 1.0f), sceneInfo,projInvMatrix);	//这里不除以w的原因是归一化后没有影响

	RayDesc ray;
	ray.Origin = mul(float4(0.0f, 0.0f, 0.0f, 1.0f), sceneInfo.viewInvMatrix).xyz;
	ray.Direction = mul(float4(normalize(viewCoords.xyz), 0.0f), sceneInfo.viewInvMatrix).xyz;
	ray.TMin = 0.001f;
	ray.TMax = INFINITE;

	HitPayload payLoad;
	payload.color = float3(0.0f);
	payload.weight = 1.0f;
	payload.depth = 0.0f;

	TraceRay(topLevelAs, rayFlags, 0xff, 0, 0, 0, ray, payload);

	float3 color = payload.color;
	outImage[int2(launchID)] = float4(color, 1.0f);
}

[shader("closesthit")]
void rayClosestHitMain(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr){
	payload.color = float3(1.0f, 0.0f, 0.0f);
}

[shader("miss")]
void rayMissMain(inout HitPayload payload){
	payload.color = float3(0.5f);
}
