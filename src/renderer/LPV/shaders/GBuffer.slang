//#define LPV_DEBUG
#include "renderer/LPV/shaders/LPVCommon.slang"

#ifndef LPV_DEBUG
float3 LPV_DirectLighting(LPVHit hit, BSDFMaterial material, float3x3 TBN, bool isExt, float3 outgoing){
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    float3 result = float3(0.0, 0.0, 0.0);

    int lightCount = min(sceneInfo.numLights, LIGHT_COUNT);
    for (int lightIndex = 0; lightIndex < lightCount; ++lightIndex){
        Light light = sceneInfo.lights[lightIndex];
        float3 direction;
        float3 radiance;
        float distanceToLight;
        if (!LPV_LightDirection(light, hit.position, direction, radiance, distanceToLight)) continue;

        float cosine = dot(direction, hit.normal);
        if (cosine <= 0.0) continue;

        // Bit 6 of the flags drops the shadow ray (debug / correctness inspection only). It used to
        // be bit 5, which collides with debug view 4 and quietly unshadowed the Reference GI view.
        if ((pushConst.flags & int(LPVFlags::eLPVFlagNoShadow)) == 0 && LPV_Occluded(hit.position, direction, distanceToLight)) continue;

        float3 bsdf = getBSDF(material, direction, hit.normal, outgoing, TBN, isExt);
        result += radiance * bsdf * cosine;
    }
    return result;
}
void LPV_MakePayload(inout CallablePayload payload, LPVHit hit, float3 outgoing, uint seed, SceneInfo sceneInfo){
    payload.material = sceneInfo.materials[hit.materialIndex];
    payload.hitPos = hit.position;
    payload.hitNormal = hit.normal;
    payload.outgoing = outgoing;
    payload.isExt = hit.frontFacing;
    payload.randomSeed = seed;
    payload.bsdf = float3(0.0, 0.0, 0.0);
    payload.pdf = 1.0;
    payload.incidence = float3(0.0, 0.0, 0.0);
    payload.directionLightDir = float3(0.0, 0.0, 0.0);
    payload.directionLightRadiance = float3(0.0, 0.0, 0.0);

    float3 tangent, bitangent;
    orthonormalBasis(payload.hitNormal, tangent, bitangent);
    payload.TBN = float3x3(tangent, bitangent, payload.hitNormal);

    if (hit.hasTexCoords) getTextueData(payload, hit.texCoords);
}
//---------------------------------------------------------------------------------------------
[numthreads(LPV_PIXEL_THREADGROUP_X, LPV_PIXEL_THREADGROUP_Y, 1)]
[shader("compute")] 
void lpvGBufferMain(uint2 threadIndex: SV_DispatchThreadID){
    if (any(threadIndex >= pushConst.screenSize)) return;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    uint pixelIndex = threadIndex.y * pushConst.screenSize.x + threadIndex.x;

    RayDesc cameraRay = LPV_CameraRay(threadIndex, float2(0.5, 0.5));
    LPVHit hit = LPV_TraceRay(cameraRay);

    LPVPixelData data;
    if (!hit.hit){
        data.positionValid = float4(0.0, 0.0, 0.0, 0.0);
        data.normal = float4(0.0, 1.0, 0.0, 0.0);
        data.albedo = float4(0.0, 0.0, 0.0, 0.0);
        data.directColor = float4(0.0, 0.0, 0.0, 0.0);
        data.specular = float4(0.0, 0.0, 0.0, 0.0);
        pixelData[pixelIndex] = data;
        return;
    }

    float3 outgoing = -cameraRay.Direction;

    // Material evaluation, exactly like the other compute renderers of this project.
    CallablePayload payload;
    LPV_MakePayload(payload, hit, outgoing, 0u, sceneInfo);
    BSDFMaterial material = payload.material;
    float3 normal = payload.hitNormal;

    // Direct lighting (shadow ray per punctual light) + emissive.
    float3 direct = material.emissive + LPV_DirectLighting(hit, material, payload.TBN, hit.frontFacing, outgoing);

    // Diffuse albedo for the indirect term: LPV models the diffuse bounce only, purely specular
    // BSDFs (conductor / dielectric) must not receive an extra diffuse lobe.
    float3 diffuseAlbedo = float3(0.0, 0.0, 0.0);
    if (material.type == MaterialType::Diffuse || material.type == MaterialType::RoughPlastic) diffuseAlbedo = material.albedo;

    // Specular reflectance, used by the glossy LPV lookup of the gather pass (paper section 7).
    float3 specular = float3(0.0, 0.0, 0.0);
    if (material.type == MaterialType::RoughPlastic) specular = material.albedo_specular;
    else if (material.type == MaterialType::Conductor || material.type == MaterialType::RoughConductor ||
             material.type == MaterialType::Dielectric || material.type == MaterialType::RoughDielectric)
        specular = material.albedo;

    data.positionValid = float4(hit.position, 1.0);
    data.normal = float4(normal, 1.0);
    data.albedo = float4(diffuseAlbedo, 0.0);
    data.directColor = float4(direct, 0.0);
    data.specular = float4(specular, 0.0);
    pixelData[pixelIndex] = data;

    /* Blocking potential of the camera surfel (paper section 3.2). One pixel covers an area
       perpendicular to the *view ray*; dividing by the observation cosine |n.v| undoes that
       projection and yields the true area of the surface patch, which is what the blocking potential
       needs (the projected footprint would be wrong by |n.v|). The clamp bounds the magnification at
       grazing angles (~87 degrees). */
    float distance = max(hit.distance, 1e-4);
    float3 viewDir = normalize(hit.position - sceneInfo.cameraPosition);
    float cosView = max(abs(dot(normal, viewDir)), 0.05);
    float pixelFootprint = distance * distance * LPV_PixelSolidAngle(threadIndex);
    float surfelArea = pixelFootprint / cosView;

    int cascadeCount = LPV_CascadeCount();
    for (int cascade = 0; cascade < cascadeCount; ++cascade)
        LPV_InjectBlocking(LPVGeometrySource::eLPVGeometryFromCamera, cascade, hit.position, normal, surfelArea);
}
#endif
//---------------------------------------------------------------------------------------------
#ifdef LPV_DEBUG
[numthreads(16, 16, 1)]
[shader("compute")]
void showGBufferMain(uint2 threadIndex: SV_DispatchThreadID) {
    if (any(threadIndex >= pushConst.screenSize)) return;

    uint pixelIndex = threadIndex.y * pushConst.screenSize.x + threadIndex.x;
    LPVPixelData pixelData = pixelData[pixelIndex];

    float4 result = 0.0f;
    if (pushConst.GBufferIndex == 0) result = float4(pixelData.normal.xyz, 1.0f);
    else if (pushConst.GBufferIndex == 1) result = float4(pixelData.albedo.xyz, 1.0f);
    else if (pushConst.GBufferIndex == 2) result = float4(pixelData.specular.xyz, 1.0f);
    else if (pushConst.GBufferIndex == 3) result = float4(pixelData.directColor.xyz, 1.0f);
    outImage[threadIndex] = result;
}
#endif