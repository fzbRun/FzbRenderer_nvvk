#include "shaderio.h"
#include "nvshaders/constants.h.slang"
#include "Slang/pbr.h.slang"
#include "nvshaders/sky_functions.h.slang"
#include "nvshaders/random.h.slang"
#define MISS_DEPTH 1000

[[vk::push_constant]] ConstantBuffer<TutoPushConstant> pushConst;
[[vk::binding(BindingPoints::eTextures, 0)]] Sampler2D textures[];
[[vk::binding(BindingPoints::eTlas, 1)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(BindingPoints::eOutImage, 1)]] RWTexture2D<float4> outImage;

struct HitPayload{
	float3 color;
	float weight;
	int depth;		//这个不是光线的hitDepth，而是boucne的深度
};

[shader("raygeneration")]
void raygenMain(){
	float2 launchID = (float2)DispatchRaysIndex().xy;
	float2 launchSize = (float2)DispatchRaysDimensions().xy;

	GltfSceneInfo sceneInfo = pushConst.sceneInfoAddress[0];

	const uint rayFlags = 0;

	const float2 clipCoords = launchID / launchSize * 2.0f - 1.0f;

	const float4 viewCoords = mul(float4(clipCoords, 1.0f, 1.0f), sceneInfo.projInvMatrix);	//这里不除以w的原因是归一化后没有影响

	RayDesc ray;
	ray.Origin = mul(float4(0.0f, 0.0f, 0.0f, 1.0f), sceneInfo.viewInvMatrix).xyz;
	ray.Direction = mul(float4(normalize(viewCoords.xyz), 0.0f), sceneInfo.viewInvMatrix).xyz;
	ray.TMin = 0.001f;
	ray.TMax = INFINITE;

	HitPayload payload;
	payload.color = float3(0.0f);
	payload.weight = 1.0f;
	payload.depth = 0;

	TraceRay(topLevelAS, rayFlags, 0xff, 0, 0, 0, ray, payload);

	float3 color = payload.color;
	outImage[int2(launchID)] = float4(color, 1.0f);
}

__generic<T : IFloat> T getAttribute(uint8_t* dataBufferAddress, BufferView bufferView, uint attributeIndex)
{
	if(bufferView.count > 0)
	{
	  T* ptr = (T*)(dataBufferAddress + bufferView.offset + attributeIndex * bufferView.byteStride);
	  return ptr[0];
	}
	
	return T(1);  // Error case
}
int3 getTriangleIndices(uint8_t* dataBufferAddress, const TriangleMesh mesh, int primitiveID)
{
	if(mesh.indices.byteStride == sizeof(int16_t))
	{
	  int16_t3* indices = (int16_t3*)(dataBufferAddress + mesh.indices.offset);
	  return indices[primitiveID];
	}
	else if(mesh.indices.byteStride == sizeof(int32_t))
	{
	  int3* indices = (int3*)(dataBufferAddress + mesh.indices.offset);
	  return indices[primitiveID];
	}
	
	return int3(-1);  // Error case
}
__generic<T : IFloat> T getTriangleAttribute(uint8_t* dataBufferAddress, BufferView bufferView, uint3 attributeIndex, float3 barycentrics)
{
	T attr0 = getAttribute<T>(dataBufferAddress, bufferView, attributeIndex.x);
	T attr1 = getAttribute<T>(dataBufferAddress, bufferView, attributeIndex.y);
	T attr2 = getAttribute<T>(dataBufferAddress, bufferView, attributeIndex.z);
	return T(barycentrics.x) * attr0 + T(barycentrics.y) * attr1 + T(barycentrics.z) * attr2;
}
float testShadow(float3 worldPos, float3 worldNormal, float3 lightDirection, GltfPunctual light){
RayDesc shadowRay;
	shadowRay.Origin    = worldPos + worldNormal * 0.001;  // Offset to avoid self-intersection
	shadowRay.Direction = normalize(lightDirection);
	shadowRay.TMin      = 0.001;
	shadowRay.TMax      = light.type == GltfLightType::eDirectional ? INFINITE : length(lightDirection);
  
	HitPayload shadowPayload;
	shadowPayload.depth = 0;

	TraceRay(topLevelAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0xff, 0, 0, 0,
           shadowRay, shadowPayload);
	 return shadowPayload.depth != MISS_DEPTH ? 0.0 : 1.0;
}
GltfPunctual processLight(GltfSceneInfo sceneInfo, float3 worldPos){
	GltfPunctual light = sceneInfo.punctualLights[0];

	if(sceneInfo.useSky == 1)
	{
		light.direction = sceneInfo.skySimpleParam.sunDirection;
		light.color     = sceneInfo.skySimpleParam.sunColor;
		light.intensity = sceneInfo.skySimpleParam.sunIntensity;
		light.type      = GltfLightType::eDirectional;
	}

	if(light.type == GltfLightType::ePoint)
	{
		light.direction = light.position - worldPos;
		float d         = length(light.direction);
		light.intensity /= (d * d);  // Inverse square law
	}
	else if(light.type == GltfLightType::eSpot){
		float3 lightDir = light.position - worldPos;
		float  d        = length(lightDir);
		light.intensity /= (d * d);  // Distance attenuation

		float theta         = dot(normalize(lightDir), normalize(light.direction));
		float spotIntensity = clamp((theta - cos(light.coneAngle)) / (1.0 - cos(light.coneAngle)), 0.0, 1.0);
		light.intensity *= spotIntensity;
		light.direction = lightDir;
	}

	return light;
}
[shader("closesthit")]
void rayClosestHitMain(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr){
	//这里好奇怪，既然得到了重心坐标，那么不就知道了三角形的信息吗，为什么还要我再去访存呢？
	float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

	uint instanceID = InstanceIndex();
    uint meshID = InstanceID();
    uint triID = PrimitiveIndex();

    GltfSceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    GltfInstance instance = sceneInfo.instances[instanceID];
    GltfMesh mesh = sceneInfo.meshes[instance.meshIndex];
    GltfMetallicRoughness material = sceneInfo.materials[instance.materialIndex];

    int3 indices = getTriangleIndices(mesh.gltfBuffer, mesh.triMesh, triID);
    float3 pos = getTriangleAttribute<float3>(mesh.gltfBuffer, mesh.triMesh.positions, indices, barycentrics);
    float3 normal = getTriangleAttribute<float3>(mesh.gltfBuffer, mesh.triMesh.normals, indices, barycentrics);
    float3 worldPos = float3(mul(float4(pos, 1.0), ObjectToWorld4x3()));
    float3 worldNormal = normalize(mul(WorldToObject4x3(), normal).xyz);

    float3 albedo = material.baseColorFactor.xyz;
    float metallic = material.metallicFactor;
    float roughness = material.roughnessFactor;

	GltfPunctual light = processLight(sceneInfo, worldPos);

    float3 N = worldNormal;
    float3 V = -WorldRayDirection();
    float3 L = normalize(light.direction);

	float shadowFactor = testShadow(worldPos, N, light.direction, light);

    float3 color = pbrMetallicRoughness(albedo, metallic, roughness, N, V, L);
    color *= light.color * light.intensity * shadowFactor;
    
    payload.color = color;
}

[shader("miss")]
void rayMissMain(inout HitPayload payload){
	GltfSceneInfo sceneInfo = pushConst.sceneInfoAddress[0];

	if(sceneInfo.useSky == 1)
    {
        float3 skyColor = evalSimpleSky(sceneInfo.skySimpleParam, WorldRayDirection());
        payload.color += skyColor * payload.weight;
    }else payload.color += sceneInfo.backgroundColor * payload.weight;

	payload.depth = MISS_DEPTH;
}

[shader("anyhit")]
void rayAnyHitMain(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr){
	float3 hitPos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();		//RayTCurrent返回命中点到起点的长度

	uint instanceID = InstanceIndex();
	GltfSceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
	GltfInstance instance = sceneInfo.instances[instanceID];
	GltfMetallicRoughness material = sceneInfo.materials[instance.materialIndex];

	float opacity = material.baseColorFactor.a == 1 ? 1 : material.baseColorFactor.a * pushConst.opacity;

	if(pushConst.transparencyMode == 0){
		float3 uv = hitPos * 10.1f;
		if((frac(uv.x) >= 0.5f != frac(uv.y) >= 0.5f) && hitPos.y > 0.01f) return;

		if(length(hitPos) > pushConst.radius){
			IgnoreHit();
			return;
		}
	}
	else if(pushConst.transparencyMode == 1){
		if(opacity == 1) return;

		float2 launchID = (float2)DispatchRaysIndex().xy;
		uint3 seedInput = uint3(uint2(launchID.xy), pushConst.frameNumber);
		uint seed = xxhash32(seedInput);
		float randValue = rand(seed);
		if(randValue > opacity) IgnoreHit();
	}
	else if(pushConst.transparencyMode == 2){
		if(opacity < 1.0f){
			float3 tintColor = material.baseColorFactor.rgb;
			payload.color += tintColor * payload.weight * opacity;
			payload.weight *= (1.0f - opacity);
			IgnoreHit();
		}
	}

}