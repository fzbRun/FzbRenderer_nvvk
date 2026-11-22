#include "renderer/VRPG2/PathGuiding/shaders/guiding.slang"

//---------------------------------------------PathGuiding-----------------------------------------
// Path sampling techniques. The BSDF and the guide are sampled by dedicated helpers and
// VRPG_PATH_SAMPLE_MODE only decides how the two are combined:
//   0 (default) : two sample balance MIS (original behaviour: guide ray + BSDF ray, one sample is
//                 chosen and both contributions are divided by the selection probability)
//   1           : one sample 50/50 (only the chosen technique is traced)
//------------------------------------------------------------------------------------------
[numthreads(FZB_PATHGUIDING_THREADGROUP_SIZE_X, FZB_PATHGUIDING_THREADGROUP_SIZE_Y, 1)]
[shader("compute")]
void computeMain_VRPG(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    threadIndex += pushConst.dispatchOffset;
    uint groupThreadIndex_linear = groupThreadIndex.y * FZB_PATHGUIDING_THREADGROUP_SIZE_X + groupThreadIndex.x;
    uint threadGroupThreadTotalCount = FZB_PATHGUIDING_THREADGROUP_SIZE_X * FZB_PATHGUIDING_THREADGROUP_SIZE_Y;

    float2 launchID = (float2)threadIndex;
    float2 launchSize = (float2)pushConst.sceneSize;
    if (threadIndex.x >= pushConst.sceneSize.x || threadIndex.y >= pushConst.sceneSize.y) return;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];

    const uint rayFlags = 0;
    RayDesc ray;
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    HitPayload_VRPG payload;
    HitPayload_VRPG vrPayload;

    uint pgSeed = xxhash32(uint3(threadIndex, pushConst.randomFrameIndex));

    float3 accumulatedRadiance = float3(0.0f);
    float RR = 0.8f;

    float3 firstHitPos = float3(0.0f);
    bool firstHit = false;
    for (int sampleIndex = 0; sampleIndex < pushConst.spp; ++sampleIndex) {
        float r1 = rand(pgSeed);
        float r2 = rand(pgSeed);
        float2 subpixel_jitter = pushConst.frameIndex == 0 && sampleIndex == 0 ? float2(0.5f, 0.5f) : float2(r1, r2);

        const float2 pixelCenter = launchID + subpixel_jitter;
        const float2 clipCoords = pixelCenter / launchSize * 2.0 - 1.0;
        float4 viewCoords = mul(float4(clipCoords, 1.0, 1.0), sceneInfo.projInvMatrix);
        viewCoords /= viewCoords.w;

        ray.Origin = sceneInfo.cameraPosition;
        ray.Direction = normalize(mul(float4(normalize(viewCoords.xyz), 0.0), sceneInfo.viewInvMatrix).xyz);

        payload.bsdf = float3(1.0f);
        payload.pdf = 1.0f;
        payload.outgoing = -ray.Direction;
        payload.isExt = true;
        payload.randomSeed = pgSeed;

        float3 bsdf_cosine_pdf = float3(1.0f);
        uint bounceDepth = 0;
        bool hit = traceRay(ray, payload);
        firstHitPos = payload.hitPos;
        firstHit = hit;

        while (bounceDepth < pushConst.maxDepth) {
            if (!hit) {
                accumulatedRadiance += RayMiss(ray.Direction) * bsdf_cosine_pdf;
                break;
            }
            if (bounceDepth == 0) {
                getDirectionLight2(payload, sceneInfo);
                accumulatedRadiance += (payload.material.emissive + payload.directionLightRadiance) * bsdf_cosine_pdf;
            }
            //-------------------------------------------------- technique samples --------------------------------------------------
            // The two techniques are sampled independently; the mode below only decides how they
            // are combined. The direct lighting term is produced as a side effect of the BSDF
            // sample (getDirectionLight/BSDF_SAMPLE fold the BRDF into directionLightRadiance) and
            // is independent of the outgoing direction that ends up being used.
            float3 previousHitPos = payload.hitPos;

            VRSampleData vrSampleData = {};
            payload.randomSeed = pgSeed;
            bool getVrSample = getNodeAABB_E_VRPG(vrSampleData, payload, threadIndex);
            pgSeed = payload.randomSeed;

            HitPayload_VRPG bsdfPayload = payload;
            bsdfPayload.randomSeed = pgSeed;
            VRPGTechniqueSample bsdfSample = vrpgSampleBsdf(bsdfPayload, sceneInfo);
            pgSeed = bsdfPayload.randomSeed;

            VRPGTechniqueSample guideSample = vrpgSampleGuide(payload, vrSampleData, getVrSample);
            float randomNumber = 0.0f;

            #if VRPG_PATH_SAMPLE_MODE == 0
            //-------------------------------------------------- two sample balance MIS (default) --------------------------------------------------
            {
                VRPGGuideHit guideHit;
                bool vrHit = false;
                bool validSample = false;
                if (guideSample.valid) {
                    ray.Origin = offsetRay(previousHitPos, payload.hitNormal);
                    ray.Direction = guideSample.dir;
                    guideHit = traceGuideRay(ray);
                    vrHit = guideHit.hit;
                    if (vrHit) {
                        validSample = checkPointInAABB(guideHit.hitPos, vrSampleData.aabb);
                        validSample = validSample && (getNormalIndex(guideHit.hitNormal) == vrSampleData.normalIndex);
                        validSample = validSample && (guideSample.guidePdf > 0.0f);
                    }
                }
                const float pgSample_weight = validSample ? guideSample.guidePdf / max(guideSample.guidePdf + guideSample.bsdfPdf, 1e-16f) : 0.0f;
                const bool pgAvailable = validSample && (pgSample_weight > 0.0f) && any(guideSample.bsdfCosine > 0.0f);
                float3 bsdf_cosine_pdf_pgSample = bsdf_cosine_pdf * guideSample.bsdfCosine / max(guideSample.guidePdf, 1e-16f) * pgSample_weight;
                if (pgAvailable) {
                    float3 directionLight = getDirectionLight2(guideHit, -guideSample.dir, guideHit.frontFacing, sceneInfo);
                    accumulatedRadiance += (guideHit.material.emissive + directionLight) * bsdf_cosine_pdf_pgSample;
                }

                ray.Origin = offsetRay(previousHitPos, payload.hitNormal);
                ray.Direction = bsdfSample.dir;
                bool bsdfHit = traceRay(ray, payload);

                const float bsdfSample_pgPdf = (validSample && bsdfHit) ? vrpgGuidePdfAt(payload, previousHitPos, vrSampleData, bsdfSample.dir, bsdfSample.bsdfCosine, threadIndex) : 0.0f;
                const float bsdfSample_weight = bsdfSample.bsdfPdf / max(bsdfSample_pgPdf + bsdfSample.bsdfPdf, 1e-16f);
                const bool bsdfAvailable = (bsdfSample.bsdfPdf > 0.0f) && (bsdfSample_weight > 0.0f) && any(bsdfSample.bsdfCosine > 0.0f);
                float3 bsdf_cosine_pdf_bsdfSample = bsdf_cosine_pdf * bsdfSample.bsdfCosine / max(bsdfSample.bsdfPdf, 1e-16f) * bsdfSample_weight;
                if (bsdfAvailable) {
                    getDirectionLight2(payload, sceneInfo);
                    accumulatedRadiance += (payload.material.emissive + payload.directionLightRadiance) * bsdf_cosine_pdf_bsdfSample;
                }
                if (!pgAvailable && !bsdfAvailable) break;

                const float selectProbability = pgAvailable ? (bsdfAvailable ? 0.5f : 1.0f) : 0.0f;
                pgSeed = payload.randomSeed;
                randomNumber = rand(pgSeed);
                if (randomNumber < selectProbability) {
                    payload = makeGuidePayload(guideHit, guideSample.dir, pgSeed, sceneInfo);
                    bsdf_cosine_pdf = bsdf_cosine_pdf_pgSample / max(selectProbability, 1e-16f);
                    hit = vrHit;
                }
                else {
                    payload.incidence = bsdfSample.dir;
                    bsdf_cosine_pdf = bsdf_cosine_pdf_bsdfSample / max(1.0f - selectProbability, 1e-16f);
                    hit = bsdfHit;
                }
            }
            #else
            //-------------------------------------------------- one sample 50/50 with balance weights (mode 1) --------------------------------------------------
            // Picks ONE technique with probability q and traces only that ray. The chosen
            // technique's contribution is divided by q (one sample MIS), so its expectation equals
            // the sum of the two weighted terms of mode 0.
            {
                const bool guideOk = guideSample.valid;
                const bool bsdfOk = bsdfSample.valid;
                if (!guideOk && !bsdfOk) break;

                const float q = guideOk ? (bsdfOk ? 0.5f : 1.0f) : 0.0f;
                const bool chooseGuide = guideOk && (rand(pgSeed) < q);

                if (chooseGuide){
                    ray.Origin = offsetRay(previousHitPos, payload.hitNormal);
                    ray.Direction = guideSample.dir;
                    VRPGGuideHit guideHit = traceGuideRay(ray);

                    const bool hitInVoxel = guideHit.hit
                        && checkPointInAABB(guideHit.hitPos, vrSampleData.aabb)
                        && (getNormalIndex(guideHit.hitNormal) == vrSampleData.normalIndex)
                        && (guideSample.guidePdf > 0.0f);

                    const float wg = bsdfOk ? guideSample.guidePdf / max(guideSample.guidePdf + guideSample.bsdfPdf, 1e-16f) : 1.0f;
                    float3 sampleThroughput = bsdf_cosine_pdf * guideSample.bsdfCosine / max(guideSample.guidePdf, 1e-16f) * wg;
                    if (hitInVoxel){
                        float3 directionLight = getDirectionLight2(guideHit, -guideSample.dir, guideHit.frontFacing, sceneInfo);
                        accumulatedRadiance += (guideHit.material.emissive + directionLight) * sampleThroughput / max(q, 1e-16f);
                    }
                    payload = makeGuidePayload(guideHit, guideSample.dir, pgSeed, sceneInfo);
                    bsdf_cosine_pdf = sampleThroughput / max(q, 1e-16f);
                    hit = hitInVoxel;
                } else {
                    ray.Origin = offsetRay(previousHitPos, payload.hitNormal);
                    ray.Direction = bsdfSample.dir;
                    const bool bsdfHit = traceRay(ray, payload);
                    payload.incidence = bsdfSample.dir; // traceRay leaves incidence untouched

                    const float bsdfSample_pgPdf = bsdfHit
                        ? vrpgGuidePdfAt(payload, previousHitPos, vrSampleData, bsdfSample.dir, bsdfSample.bsdfCosine, threadIndex) : 0.0f;
                    const float wb = guideOk ? bsdfSample.bsdfPdf / max(bsdfSample.bsdfPdf + bsdfSample_pgPdf, 1e-16f) : 1.0f;
                    float3 sampleThroughput = bsdf_cosine_pdf * bsdfSample.bsdfCosine / max(bsdfSample.bsdfPdf, 1e-16f) * wb;
                    if (bsdfHit){
                        getDirectionLight2(payload, sceneInfo);
                        accumulatedRadiance += (payload.material.emissive + payload.directionLightRadiance) * sampleThroughput / max(1.0f - q, 1e-16f);
                    }
                    bsdf_cosine_pdf = sampleThroughput / max(1.0f - q, 1e-16f);
                    hit = bsdfHit;
                }
            }
            #endif
            if (!hit) ray.Direction = payload.incidence;

            bool end = any(bsdf_cosine_pdf < 0.0f) || all(bsdf_cosine_pdf == 0.0f) || any(isnan(bsdf_cosine_pdf)) || any(isinf(bsdf_cosine_pdf));
            if (end) break;

            randomNumber = rand(pgSeed);
            if (randomNumber >= RR) break;
            bsdf_cosine_pdf /= RR;

            payload.randomSeed = pgSeed;
            payload.outgoing = -payload.incidence;
            ++bounceDepth;
        }
    }

    accumulatedRadiance /= pushConst.spp;

    if (pushConst.maxFrameCount == 2 && pushConst.maxFrameCount > 1) {
        float a = 0.5f;
        float3 old_color = outImage[int2(launchID)].xyz;
        outImage[int2(launchID)] = float4(lerp(old_color, accumulatedRadiance, a), 1.0f);
    }
    else if (pushConst.maxFrameCount > 1 && pushConst.frameIndex > 0) {
        float a = 1.0f / float(pushConst.frameIndex + 1);
        float3 old_color = outImage[int2(launchID)].xyz;
        outImage[int2(launchID)] = float4(lerp(old_color, accumulatedRadiance, a), 1.0f);
    }
    else outImage[int2(launchID)] = float4(accumulatedRadiance, 1.0f);
}
