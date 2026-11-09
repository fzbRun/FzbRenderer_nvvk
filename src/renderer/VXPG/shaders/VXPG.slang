// VXPG - the integrator. This is the port of the reference "Guide + BSDF" sample mode
// (sample_stragegy == 4 of vxguiding-gi.slang): one BSDF sampled ray and one guided ray are
// traced from the primary vertex, both are combined with the balance (or power) MIS
// heuristic, and the guided sample is only accepted when its hit lies inside the sampled
// voxel ("semi NEE" of the reference). Deeper bounces are plain BSDF sampling + NEE.
#include "renderer/VXPG/shaders/VXPGCommon.slang"

[numthreads(VXPG_THREADGROUP_SIZE_X, VXPG_THREADGROUP_SIZE_Y, 1)]
[shader("compute")]
void computeMain_VXPG(uint2 threadIndex : SV_DispatchThreadID)
{
    threadIndex += pushConst.dispatchOffset;
    if (any(threadIndex >= pushConst.screenSize))
        return;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    uint seed = xxhash32(uint3(threadIndex, uint(pushConst.randomFrameIndex)));
    int2 pixel = int2(threadIndex);
    float2 launchSize = float2(pushConst.screenSize);

    float3 accumulated = float3(0.0);
    int spp = max(pushConst.spp, 1);

    for (int sampleIndex = 0; sampleIndex < spp; ++sampleIndex)
    {
        float rnd0 = rand(seed);
        float rnd1 = rand(seed);
        float2 jitter = (pushConst.frameIndex == 0 && sampleIndex == 0) ? float2(0.5, 0.5) : float2(rnd0, rnd1);

        float2 clipCoords = (float2(threadIndex) + jitter) / launchSize * 2.0 - 1.0;
        float4 viewCoords = mul(float4(clipCoords, 1.0, 1.0), sceneInfo.projInvMatrix);
        viewCoords /= viewCoords.w;

        RayDesc cameraRay;
        cameraRay.Origin = sceneInfo.cameraPosition;
        cameraRay.Direction = normalize(mul(float4(normalize(viewCoords.xyz), 0.0), sceneInfo.viewInvMatrix).xyz);
        cameraRay.TMin = 0.001;
        cameraRay.TMax = INFINITE;

        //if (pushConst.frameIndex == 0 && all(threadIndex >= uint2(635, 355)) && all(threadIndex <= uint2(645, 365))) {
        //    printf("pixel: %d %d\n", pixel.x, pixel.y);
        //}

        VXPGHit primary = VXPG_TraceRay(cameraRay);
        if (!primary.hit){
            accumulated += VXPG_RayMiss(cameraRay.Direction);
            continue;
        }

        CallablePayload primaryPayload;
        VXPG_MakePayload(primaryPayload, primary, -cameraRay.Direction, seed, sceneInfo);

        // Debug view of the guiding structure (VXPG_DEBUG=16):
        //   red   : VPL count of the primary hit voxel (0.25 -> 1)
        //   green : superpixel assigned for this pixel
        //   blue  : fraction of the superpixel's clusters that are usable (weight and a valid root)
        if ((pushConst.flags & 32) != 0){
            // "Is guiding data available at this pixel?" view:
            //   red   : VPLs in the visible voxel (light injection reached this region)
            //   green : accumulated weight sum(T * Phi) of the pixel's superpixel
            //   blue  : the superpixel's dominant cluster has a valid light tree root
            const int3 voxel = VXPG_WorldToVoxelID(primary.position);
            const float r = saturate(float(voxelVplCount[VXPG_FlatIndex(voxel)]) * 0.25);
            const int sp = VXPG_SelectSpixel(pixel);   // -1 for background pixels

            float g = 0.0;
            float b = 0.0;
            if (sp >= 0)   // never index the top level tree with the -1 sentinel
            {
                const int offset = sp * VXPG_TOP_LEVEL_STRIDE;
                // Log scale: the accumulated weights are fixed point (x1000), so a linear 1e-3 scale
                // saturates to white as soon as any guiding data exists.
                const float total = max(topLevelTree[offset + 1], 0.0);
                g = saturate(log2(1.0 + total) * 0.0625);   // 0 at 0, ~0.6 at 1e3, 1 at 6.4e4

                // Blue: fraction of the superpixel's clusters that hold weight and own a valid light
                // tree root, i.e. how much of the 32 cluster distribution is actually usable.
                int usableClusters = 0;
                for (int c = 0; c < VXPG_CLUSTER_COUNT; ++c)
                {
                    const float w = topLevelTree[offset + VXPG_TOP_LEVEL_LEAF_OFFSET + c];
                    if (w > 0.0 && clusterRoots[c] >= 0)
                        ++usableClusters;
                }
                b = float(usableClusters) / float(VXPG_CLUSTER_COUNT);
            }

            outImage[pixel] = float4(r, g, b, 1.0);
            continue;
        }

        // Debug view of the guide sampler (VXPG_DEBUG=32):
        //   red   : SampleVoxelGuiding produced a valid sample
        //   green : the top level tree returned a cluster
        //   blue  : the light tree traversal reached a leaf
        if ((pushConst.flags & 64) != 0)
        {
            float topProb = 0.0f;
            int sp = VXPG_SelectSpixel(pixel);
            int cluster = VXPG_SampleTopLevelTree(sp, rand(seed), topProb);
            float blue = 0.0;
            if (cluster >= 0 && clusterRoots[cluster] >= 0)
            {
                float bottomProb = 0.0f;
                uint leaf = VXPG_TraverseTree(uint(clusterRoots[cluster]), rand(seed), bottomProb);
                blue = (leaf != VXPG_INVALID_INDEX && bottomProb > 0.0f) ? 1.0 : 0.0;
            }
            VXPGGuideSample guide = VXPG_SampleVoxelGuiding(pixel, primary.position, primaryPayload.hitNormal, VXPG_WorldToVoxelID(primary.position), seed);
            outImage[pixel] = float4(guide.valid ? 1.0 : 0.0, (cluster >= 0) ? 1.0 : 0.0, blue, 1.0);
            continue;
        }

        // Direct lighting of the camera visible vertex.
        {
            CallablePayload neePayload = primaryPayload;
            getDirectionLight(neePayload, sceneInfo);
            BSDF_SAMPLE(neePayload);
            accumulated += neePayload.material.emissive + neePayload.directionLightRadiance;
        }

        int3 primaryVoxel = VXPG_WorldToVoxelID(primary.position);
        const bool useBsdfBranch = (pushConst.flags & 8) == 0;
        const bool useGuideBranch = (pushConst.flags & 16) == 0;

        // Path length in vertices (1 = primary only). The first indirect vertex is traced by
        // both techniques of mode 4, every additional bounce only by the BSDF technique, which
        // is why its contribution is not MIS weighted (the guided technique cannot generate
        // paths of length >= 3).
        const int maxPathVertices = max(pushConst.maxDepth, 1);
        const int extraBounces = clamp(maxPathVertices - 2, 0, 8);
        const bool traceIndirect = maxPathVertices >= 2;
        // Guiding only makes sense for shading points inside the camera centred voxel domain.
        const bool guideUsable = ((pushConst.flags & 1) != 0) && traceIndirect && VXPG_PositionInDomain(primary.position);

        // Guide pdf breakdown (VXPG_DEBUG=512): which factor of guide.pdf is off.
        //   red   = log2(1 + guide.pdf) / 25
        //   green = p(leaf | cluster)          (0..1)
        //   blue  = p(cluster | pixel) * 32    (1 = uniform over the 32 clusters)
        if ((pushConst.flags & 1024) != 0)
        {
            VXPGGuideSample g = VXPG_SampleVoxelGuiding(pixel, primary.position, primaryPayload.hitNormal, primaryVoxel, seed);
            float leafProb = 0.0;
            float clusterProb = 0.0;
            if (g.valid)
            {
                int3 gv = g.voxel;
                uint gflat = VXPG_FlatIndex(gv);
                uint gslot = voxelLeaf[gflat];
                if (gslot != VXPG_INVALID_INDEX)
                {
                    uint gleaf = VXPG_LeafNodeIndex(gslot);
                    int gcluster = VXPG_ClusterOfLeafSlot(gslot);
                    if (gcluster >= 0 && clusterRoots[gcluster] >= 0)
                        leafProb = VXPG_TreePathProbability(uint(clusterRoots[gcluster]), gleaf);
                    int ids[4];
                    float wts[4];
                    if (VXPG_SpixelMixture(pixel, primary.position, primaryPayload.hitNormal, ids, wts) != 0)
                        clusterProb = VXPG_TopLevelProbabilityMixture(ids, wts, gcluster);
                }
            }
            // Write the breakdown into the pixel record so it can be read back on the CPU through
            // the dump (the image goes through the tonemapper and cannot be read quantitatively).
            //   vplPos.x = p_sph (= 1 / sum S), .y = p(leaf|cluster), .z = p(cluster|pixel),
            //   .w = guide.pdf
            {
                const float denom = max(leafProb * clusterProb, 1e-30);
                uint dumpIndex = pixel.y * pushConst.screenSize.x + pixel.x;
                VXPGPixelData dumpRecord = pixelData[dumpIndex];
                dumpRecord.vplPos = float4(g.valid ? g.pdf / denom : 0.0, leafProb, clusterProb, g.pdf);
                pixelData[dumpIndex] = dumpRecord;
            }
            float pdfView = clamp(log2(1.0 + g.pdf) / 25.0, 0.0, 1.0);
            outImage[pixel] = float4(g.valid ? pdfView : 0.0,
                                     clamp(leafProb, 0.0, 1.0),
                                     clamp(clusterProb * 32.0, 0.0, 1.0), 1.0);
            continue;
        }

        // MIS mode: 0 = one sample MIS (pick one technique with probability 1/2 and scale by 2).
        // Its weights contain no per voxel pdf ratio, which is what painted the voxel grid onto
        // the shaded surfaces as blocks. 1 = balance, 2 = power heuristic.
        const bool oneSampleMis = (pushConst.misMode == 0);
        const float selectProb = 0.5;
        bool pickBsdf = true;
        if (oneSampleMis) pickBsdf = rand(seed) < selectProb;
        const float3 indirectBefore = accumulated;

        // Debug view of the guided branch (VXPG_DEBUG=128):
        //   red: valid sample, green: hit inside the sampled voxel, blue: non zero BSDF weight
        if ((pushConst.flags & 256) != 0)
        {
            VXPGGuideSample guideDbg = VXPG_SampleVoxelGuiding(pixel, primary.position, primaryPayload.hitNormal, primaryVoxel, seed);
            float r = 0.0;
            float g = 0.0;
            float b = 0.0;
            if (guideDbg.valid)
            {
                // red: 1 = sane direction, 0.75 = NaN/zero direction, 0.5 = direction points
                // away from the sampled voxel (the sampler disagrees with the pdf)
                r = 1.0;
                if (any(isnan(guideDbg.direction)) || all(guideDbg.direction == float3(0.0)))
                    r = 0.75;
                RayDesc rayDbg = VXPG_SpawnRay(primary.position, primary.normal, guideDbg.direction);
                VXPGHit hitDbg = VXPG_TraceRay(rayDbg);
                if (hitDbg.hit)
                {
                    g = (all(hitDbg.position >= guideDbg.aabbMin) && all(hitDbg.position <= guideDbg.aabbMax)) ? 1.0 : 0.5;
                    if (g == 1.0)
                    {
                        float3 bsdfDbg = getBSDF(primaryPayload.material, guideDbg.direction, primaryPayload.hitNormal,
                                                 primaryPayload.outgoing, primaryPayload.TBN, primaryPayload.isExt);
                        float3 wDbg = bsdfDbg * abs(dot(guideDbg.direction, primaryPayload.hitNormal));
                        b = any(wDbg > 0.0) ? 1.0 : 0.25;
                    }
                }
            }
            outImage[pixel] = float4(r, g, b, 1.0);
            continue;
        }
        //----------------------------------------------------------------------------------
        // Technique A: BSDF sampling. Contribution of the first indirect vertex combined
        // with the MIS weight; deeper bounces are added unweighted because the guided
        // technique cannot generate paths of length >= 3.
        //----------------------------------------------------------------------------------
        if (useBsdfBranch && traceIndirect && (!oneSampleMis || pickBsdf))
        {
            CallablePayload bsdfPayload = primaryPayload;
            BSDF_SAMPLE(bsdfPayload);
            float bsdfPdf = bsdfPayload.pdf;
            float3 bsdfWeight = bsdfPayload.bsdf * abs(dot(bsdfPayload.incidence, bsdfPayload.hitNormal));

            if (bsdfPdf > 0.0 && any(bsdfWeight > 0.0))
            {
                RayDesc rayA = VXPG_SpawnRay(primary.position, primary.normal, bsdfPayload.incidence);
                VXPGHit hitA = VXPG_TraceRay(rayA);

                if (!hitA.hit) accumulated += VXPG_RayMiss(rayA.Direction) * bsdfWeight / bsdfPdf;
                else{
                    CallablePayload payloadA;
                    VXPG_MakePayload(payloadA, hitA, -bsdfPayload.incidence, seed, sceneInfo);
                    float3 radianceA = VXPG_DirectLight(payloadA, sceneInfo);
                    float3 contributionA = radianceA * bsdfWeight / bsdfPdf;
                    if (any(isnan(contributionA)) || any(isinf(contributionA))) contributionA = float3(0.0);
                    if (pushConst.fireflyClamp > 0.0)
                    {
                        const float peakA = max(max(contributionA.x, contributionA.y), contributionA.z);
                        if (peakA > pushConst.fireflyClamp) contributionA *= pushConst.fireflyClamp / peakA;
                    }

                    int3 voxelA = VXPG_WorldToVoxelID(hitA.position);
                    float guidePdfA = VXPG_PositionInDomain(hitA.position)
                        ? VXPG_PdfVoxelGuidingSmooth(pixel, voxelA, primaryVoxel, primary.position, primaryPayload.hitNormal)
                        : 0.0;
                    float w1 = bsdfPdf;
                    float w2 = guidePdfA;
                    if (pushConst.misMode == int(VXPG_MisMode::ePower))
                    {
                        w1 *= w1;
                        w2 *= w2;
                    }

                    // One-sample MIS (Veach): every pixel evaluates a single technique, so the
                    // balance weight has to be divided by the selection probability:
                    //     F = f(X_i) / (c_i * (p1(X_i) + p2(X_i)))
                    // Without the c_i division (and with the weight pinned to 1) the estimator was
                    // only valid when both techniques cover the whole hemisphere; the guided
                    // technique only covers directions towards injected voxels, so it converged too
                    // dark. VXPG_DEBUG=64 keeps the unweighted variant for energy experiments.
                    const float balanceA = w1 / max(w1 + w2, 1e-8);
                    const float misA = (pushConst.flags & 128) != 0 ? 1.0f
                                     : (oneSampleMis ? balanceA / selectProb : balanceA);
                    accumulated += contributionA * misA;
                }
            }
        }

        //----------------------------------------------------------------------------------
        // Technique B: voxel guided sampling. The sample is only accepted when the traced
        // hit lies inside the sampled voxel (the reference's semi NEE acceptance test).
        //----------------------------------------------------------------------------------
        if (useGuideBranch && guideUsable && (!oneSampleMis || !pickBsdf))
        {
            VXPGGuideSample guide = VXPG_SampleVoxelGuiding(pixel, primary.position, primaryPayload.hitNormal, primaryVoxel, seed);
            if (guide.valid && guide.pdf > 0.0)
            {
                RayDesc rayB = VXPG_SpawnRay(primary.position, primary.normal, guide.direction);
                VXPGHit hitB = VXPG_TraceRay(rayB);

                // The hit has to land in the *sampled voxel itself*, not merely inside its AABB: at wall
                // corners the wall surface lies exactly on the voxel boundary, so an AABB test accepts the
                // occluder as if it were the VPL and light leaks along every edge (the bright hairlines).
                const bool hitInVoxel = hitB.hit && all(hitB.position >= guide.aabbMin) && all(hitB.position <= guide.aabbMax)
                                     && all(VXPG_WorldToVoxelID(hitB.position) == guide.voxel);

                if (hitInVoxel)
                {
                    float3 bsdfB = getBSDF(primaryPayload.material, guide.direction, primaryPayload.hitNormal,
                                           primaryPayload.outgoing, primaryPayload.TBN, primaryPayload.isExt);
                    float3 bsdfWeightB = bsdfB * abs(dot(guide.direction, primaryPayload.hitNormal));
                    if (any(bsdfWeightB > 0.0))
                    {
                        CallablePayload payloadB;
                        VXPG_MakePayload(payloadB, hitB, -guide.direction, seed, sceneInfo);
                        float3 radianceB = VXPG_DirectLight(payloadB, sceneInfo);
                        float3 contributionB = radianceB * bsdfWeightB / guide.pdf;
                        if (any(isnan(contributionB)) || any(isinf(contributionB))) contributionB = float3(0.0);
                        if (pushConst.fireflyClamp > 0.0)
                        {
                            const float peakB = max(max(contributionB.x, contributionB.y), contributionB.z);
                            if (peakB > pushConst.fireflyClamp) contributionB *= pushConst.fireflyClamp / peakB;
                        }

                        float w1 = getPdf(primaryPayload.material, guide.direction, primaryPayload.hitNormal,
                                          primaryPayload.outgoing, primaryPayload.TBN, primaryPayload.isExt);
                        // Use the density the sample was actually drawn from (the sampler's own pdf),
                        // not the smoothed approximation: the MIS weight has to see the same density
                        // that divides the contribution, otherwise the one-sample form
                        // f / (c_i (p1 + p2)) is no longer the density ratio it assumes.
                        float w2 = guide.pdf;
                        if (pushConst.misMode == int(VXPG_MisMode::ePower))
                        {
                            w1 *= w1;
                            w2 *= w2;
                        }
                        const float balanceB = w2 / max(w1 + w2, 1e-8);
                        const float misB = (pushConst.flags & 128) != 0 ? 1.0f
                                         : (oneSampleMis ? balanceB / selectProb : balanceB);
                        accumulated += contributionB * misB;
                    }
                }
            }
        }

        // Contribution of paths other than those with length 3 in path space
        if (extraBounces > 0)
        {
            CallablePayload bouncePayload = primaryPayload;
            BSDF_SAMPLE(bouncePayload);
            float bsdfPdf = bouncePayload.pdf;
            float3 bsdfWeight = bouncePayload.bsdf * abs(dot(bouncePayload.incidence, bouncePayload.hitNormal));
            float3 throughput = bsdfWeight / bsdfPdf;

            RayDesc bounceRay = VXPG_SpawnRay(bouncePayload.hitPos, bouncePayload.hitNormal, bouncePayload.incidence);
            VXPGHit bounceHit = VXPG_TraceRay(bounceRay);
            if (!bounceHit.hit){
                accumulated += VXPG_RayMiss(bounceRay.Direction) * throughput;
                break;
            }
            VXPG_MakePayload(bouncePayload, bounceHit, -bouncePayload.incidence, seed, sceneInfo);
            BSDF_SAMPLE(bouncePayload);

            for (int b = 0; b < extraBounces; ++b){
                float stepPdf = bouncePayload.pdf;
                float3 stepWeight = bouncePayload.bsdf * abs(dot(bouncePayload.incidence, bouncePayload.hitNormal));
                if (!(stepPdf > 0.0) || all(stepWeight <= 0.0)) break;

                throughput *= stepWeight / stepPdf;
                bounceRay = VXPG_SpawnRay(bouncePayload.hitPos, bouncePayload.hitNormal, bouncePayload.incidence);
                bounceHit = VXPG_TraceRay(bounceRay);
                if (!bounceHit.hit){
                    accumulated += VXPG_RayMiss(bounceRay.Direction) * throughput;
                    break;
                }
                VXPG_MakePayload(bouncePayload, bounceHit, -bouncePayload.incidence, seed, sceneInfo);
                getDirectionLight(bouncePayload, sceneInfo);
                BSDF_SAMPLE(bouncePayload);
                accumulated += (bouncePayload.material.emissive + bouncePayload.directionLightRadiance) * throughput;
            }
        }

        // One sample MIS: both techniques are unbiased estimators of the same indirect integral, so
        // randomly picking ONE of them (with probability selectProb) and taking its raw estimate is
        // already unbiased: E = selectProb * I + (1 - selectProb) * I = I. Scaling the picked
        // estimate by 1/selectProb would double count and was removed.
        accumulated = indirectBefore + (accumulated - indirectBefore);
    }

    accumulated /= float(spp);

    // Debug outputs (VXPG_DEBUG environment variable of the renderer):
    //   bit1: constant grey, to check the tonemapper / image plumbing
    //   bit2: tonemapped radiance without temporal accumulation
    if ((pushConst.flags & 2) != 0)
    {
        outImage[pixel] = float4(0.5f, 0.5f, 0.5f, 1.0f);
        return;
    }
    // Self check (VXPG_DEBUG=512): what this shader actually sees for the estimator switch.
    //   red   = 1 when one sample MIS is active (misMode == 0)
    //   green = misMode / 4
    //   blue  = (flags & 255) / 255  (bit0 guiding, bit1 fuzzy, bit3/4 disable branches)
    if ((pushConst.flags & 1024) != 0)
    {
        const float oneSample = (pushConst.misMode == 0) ? 1.0 : 0.0;
        outImage[pixel] = float4(oneSample,
                                 float(pushConst.misMode) * 0.25,
                                 float(pushConst.flags & 255) / 255.0, 1.0);
        return;
    }

    if ((pushConst.flags & 4) != 0)
    {
        float3 mapped = accumulated / (1.0f + accumulated);
        outImage[pixel] = float4(mapped, 1.0f);
        return;
    }

    if (pushConst.frameIndex == 0 && all(threadIndex >= uint2(635, 355)) && all(threadIndex <= uint2(645, 365))) accumulated = float3(10.0f, 0.0f, 0.0f);

    if (any(isnan(accumulated)) || any(isinf(accumulated))) accumulated = float3(0.0);

    // Temporal accumulation, identical to the VRPG renderer:
    //   maxFrames == 1 : current frame only (no accumulation)
    //   maxFrames == 2 : rolling average of the current and the previous frame (alpha 0.5)
    //   maxFrames >  2 : progressive average, alpha = 1 / (frameIndex + 1)
    if (pushConst.maxFrameCount == 2)
    {
        const float a = 0.5f;
        float3 oldColor = outImage[pixel].xyz;
        if (any(isnan(oldColor)) || any(isinf(oldColor)))
            oldColor = float3(0.0);
        outImage[pixel] = float4(lerp(oldColor, accumulated, a), 1.0f);
    }
    else if (pushConst.maxFrameCount > 1 && pushConst.frameIndex > 0)
    {
        float a = 1.0f / float(pushConst.frameIndex + 1);
        float3 oldColor = outImage[pixel].xyz;
        if (any(isnan(oldColor)) || any(isinf(oldColor)))
            oldColor = float3(0.0);
        outImage[pixel] = float4(lerp(oldColor, accumulated, a), 1.0f);
    }
    else
    {
        outImage[pixel] = float4(accumulated, 1.0f);
    }
}
