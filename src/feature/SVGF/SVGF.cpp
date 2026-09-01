#include "./SVGF.h"
#include <common/Application/Application.h>
#include <nvvk/formats.hpp>
#include <common/Image/Image.h>
#include <common/Shader/Shader.h>
#include <nvvk/default_structs.hpp>
#include <nvgui/property_editor.hpp>
#include <nvvk/compute_pipeline.hpp>

using namespace FzbRenderer;

SVGF::SVGF() {}

void SVGF::init(SVGFCreateInfo createInfo) {
    this->setting = createInfo;

    Feature::createGBuffer(false, false, (uint32_t)GBuffers_SVGF::bufferCount);

    createDescriptorSetLayout();
    createDescriptorSet();
    createPipeline();
    compileAndCreateShaders();

    Feature::init();

    pushConstant.frameIndex = 0;
}

void SVGF::clean() {
    VkDevice device = Application::app->getDevice();
    vkDestroyShaderEXT(device, computeShader_getDepthGradient, nullptr);
    vkDestroyShaderEXT(device, computeShader_getEffectiveIrradiance, nullptr);
    vkDestroyShaderEXT(device, computeShader_varianceEstimate, nullptr);
    vkDestroyShaderEXT(device, computeShader_varianceConvolution, nullptr);
    vkDestroyShaderEXT(device, computeShader_Filter_X, nullptr);
    vkDestroyShaderEXT(device, computeShader_Filter_Y, nullptr);

    Feature::clean();
}
void SVGF::uiRender() {}
void SVGF::resize(VkCommandBuffer cmd, const VkExtent2D& size, nvvk::Image images[6]) {
    NVVK_CHECK(gBuffers.update(cmd, size));

    nvvk::WriteSetContainer write{};

    VkWriteDescriptorSet    renderTargetWrite =
        staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_SVGF::eAlbedoImage, 0, 0, 1);
    setting.albedoImage = images[0];
    write.append(renderTargetWrite, setting.albedoImage);
        
    renderTargetWrite =
        staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_SVGF::eDepthImage, 0, 0, 1);
    setting.depthImage = images[1];
    write.append(renderTargetWrite, setting.depthImage);

    renderTargetWrite =
        staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_SVGF::eNormalImage, 0, 0, 1);
    setting.normalImage = images[2];
    write.append(renderTargetWrite, setting.normalImage);

    renderTargetWrite =
        staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_SVGF::eVelocityImage, 0, 0, 1);
    setting.velocityImage = images[3];
    write.append(renderTargetWrite, setting.velocityImage);

    renderTargetWrite =
        staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_SVGF::eVertexInfoImage, 0, 0, 1);
    setting.vertexInfoImage = images[4];
    write.append(renderTargetWrite, setting.vertexInfoImage);

    renderTargetWrite =
        staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_SVGF::eResultImage, 0, 0, 1);
    setting.renderTarget = images[5];
    write.append(renderTargetWrite, setting.renderTarget);

    renderTargetWrite =
        staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_SVGF::eDepthGradientImage, 0, 0, 1);
    write.append(renderTargetWrite, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_SVGF::DepthGradient]);

    renderTargetWrite =
        staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_SVGF::eIrradianceImage, 0, 0, 1);
    write.append(renderTargetWrite, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_SVGF::Irradiance]);

    renderTargetWrite =
        staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_SVGF::eMoment1Image, 0, 0, 1);
    write.append(renderTargetWrite, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_SVGF::moment1]);

    renderTargetWrite =
        staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_SVGF::eMoment2Image, 0, 0, 1);
    write.append(renderTargetWrite, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_SVGF::moment2]);

    renderTargetWrite =
        staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_SVGF::eVarianceImage, 0, 0, 2);
    std::vector<nvvk::Image> varianceImages = { gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_SVGF::Variance0], gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_SVGF::Variance1] };
    write.append(renderTargetWrite, varianceImages.data());

    renderTargetWrite =
        staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_SVGF::eFilterImages, 0, 0, 2);
    std::vector<nvvk::Image> filterImages = { gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_SVGF::Filter0], gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_SVGF::Filter1] };
    write.append(renderTargetWrite, filterImages.data());

    {
        renderTargetWrite =
            staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_SVGF::eHistoryDepthImage, 0, 0, 1);
        write.append(renderTargetWrite, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_SVGF::historyDepth]);

        renderTargetWrite =
            staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_SVGF::eHistoryNormalImage, 0, 0, 1);
        write.append(renderTargetWrite, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_SVGF::historyNormal]);

        renderTargetWrite =
            staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_SVGF::eHistoryVertexInfoImage, 0, 0, 1);
        write.append(renderTargetWrite, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_SVGF::historyVertexInfo]);
    }

    vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);

    pushConstant.screenSize = { size.width, size.height };
    pushConstant.frameIndex = 0;
}
void SVGF::preRender() {
    Scene& scene = Application::sceneResource;
    pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;

    pushConstant.nearPlane = Application::sceneResource.cameraManip->getClipPlanes().x;
    pushConstant.farPlane = Application::sceneResource.cameraManip->getClipPlanes().y;

    pushConstant.projMatrix = Application::sceneResource.cameraManip->getPerspectiveMatrix();
}
void SVGF::render(VkCommandBuffer cmd) {
    NVVK_DBG_SCOPE(cmd);

    updateDataPerFrame(cmd);

    pushInfo = {
        .sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
        .layout = pipelineLayout,
        .stageFlags = VK_SHADER_STAGE_ALL,
        .offset = 0,
        .size = sizeof(shaderio::SVGFPushConstant),
        .pValues = &pushConstant,
    };

    getFeatures(cmd);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    filter(cmd);

    ++pushConstant.frameIndex;
}

void SVGF::createDescriptorSetLayout() {
    SCOPED_TIMER(__FUNCTION__);
    nvvk::DescriptorBindings bindings;

    bindings.addBinding({
            .binding = (uint32_t)shaderio::BindingPoints_SVGF::eAlbedoImage,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_ALL });

    bindings.addBinding({
        .binding = (uint32_t)shaderio::BindingPoints_SVGF::eDepthImage,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_ALL });

    bindings.addBinding({
        .binding = (uint32_t)shaderio::BindingPoints_SVGF::eNormalImage,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_ALL });

    bindings.addBinding({
        .binding = (uint32_t)shaderio::BindingPoints_SVGF::eVelocityImage,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_ALL });

    bindings.addBinding({
        .binding = (uint32_t)shaderio::BindingPoints_SVGF::eVertexInfoImage,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_ALL });

    bindings.addBinding({
        .binding = (uint32_t)shaderio::BindingPoints_SVGF::eResultImage,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_ALL });

    bindings.addBinding({
        .binding = (uint32_t)shaderio::BindingPoints_SVGF::eDepthGradientImage,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_ALL });

    bindings.addBinding({
        .binding = (uint32_t)shaderio::BindingPoints_SVGF::eIrradianceImage,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_ALL });

    bindings.addBinding({
        .binding = (uint32_t)shaderio::BindingPoints_SVGF::eMoment1Image,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_ALL });

    bindings.addBinding({
        .binding = (uint32_t)shaderio::BindingPoints_SVGF::eMoment2Image,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_ALL });

    bindings.addBinding({
        .binding = (uint32_t)shaderio::BindingPoints_SVGF::eVarianceImage,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 2,
        .stageFlags = VK_SHADER_STAGE_ALL });

    bindings.addBinding({
        .binding = (uint32_t)shaderio::BindingPoints_SVGF::eFilterImages,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 2,
        .stageFlags = VK_SHADER_STAGE_ALL });

    {
        bindings.addBinding({
            .binding = (uint32_t)shaderio::BindingPoints_SVGF::eHistoryDepthImage,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_ALL });

        bindings.addBinding({
            .binding = (uint32_t)shaderio::BindingPoints_SVGF::eHistoryNormalImage,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_ALL });

        bindings.addBinding({
            .binding = (uint32_t)shaderio::BindingPoints_SVGF::eHistoryVertexInfoImage,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_ALL });
    }

    staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

    LOGI("Fzb PathGuiding static descriptor layout created\n");
    NVVK_DBG_NAME(staticDescPack.getLayout());
    NVVK_DBG_NAME(staticDescPack.getPool());
    NVVK_DBG_NAME(staticDescPack.getSet(0));
}
void SVGF::createDescriptorSet() {}
void SVGF::createPipeline() {
    const VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_ALL,
        .offset = 0,
        .size = sizeof(shaderio::SVGFPushConstant)
    };

    std::array<VkDescriptorSetLayout, 1> layouts = { {staticDescPack.getLayout()} };
    const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = layouts.size(),
        .pSetLayouts = layouts.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange,
    };
    NVVK_CHECK(vkCreatePipelineLayout(Application::app->getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout));
    NVVK_DBG_NAME(pipelineLayout);
}
void SVGF::compileAndCreateShaders() {
    SCOPED_TIMER(__FUNCTION__);

    std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
    std::filesystem::path shaderSource;
    VkShaderModuleCreateInfo shaderCode;

    const VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_ALL ,
        .offset = 0,
        .size = sizeof(shaderio::SVGFPushConstant),
    };

    std::array<VkDescriptorSetLayout, 1> layouts = { {staticDescPack.getLayout()} };
    VkShaderCreateInfoEXT shaderInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
        .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
        .pName = "main",
        .setLayoutCount = layouts.size(),
        .pSetLayouts = layouts.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange,
    };
    VkDevice device = Application::app->getDevice();
    //--------------------------------------------------------------------------------------
    {
        shaderSource = shaderPath / "getFeatures.slang";
        shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

        vkDestroyShaderEXT(device, computeShader_getDepthGradient, nullptr);
        shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderInfo.nextStage = 0;
        shaderInfo.pName = "computeMain_getDepthGradient";
        shaderInfo.codeSize = shaderCode.codeSize;
        shaderInfo.pCode = shaderCode.pCode;
        vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getDepthGradient);
        NVVK_DBG_NAME(computeShader_getDepthGradient);

        vkDestroyShaderEXT(device, computeShader_getEffectiveIrradiance, nullptr);
        shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderInfo.nextStage = 0;
        shaderInfo.pName = "computeMain_getEffectiveIrradiance";
        shaderInfo.codeSize = shaderCode.codeSize;
        shaderInfo.pCode = shaderCode.pCode;
        vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getEffectiveIrradiance);
        NVVK_DBG_NAME(computeShader_getEffectiveIrradiance);

        vkDestroyShaderEXT(device, computeShader_varianceEstimate, nullptr);
        shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderInfo.nextStage = 0;
        shaderInfo.pName = "computeMain_varianceEstimate";
        shaderInfo.codeSize = shaderCode.codeSize;
        shaderInfo.pCode = shaderCode.pCode;
        vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_varianceEstimate);
        NVVK_DBG_NAME(computeShader_varianceEstimate);
    }
    //--------------------------------------------------------------------------------------
    {
        shaderSource = shaderPath / "filter.slang";
        shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

        vkDestroyShaderEXT(device, computeShader_Filter_X, nullptr);
        shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderInfo.nextStage = 0;
        shaderInfo.pName = "computeMain_Filter_X";
        shaderInfo.codeSize = shaderCode.codeSize;
        shaderInfo.pCode = shaderCode.pCode;
        vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_Filter_X);
        NVVK_DBG_NAME(computeShader_Filter_X);

        vkDestroyShaderEXT(device, computeShader_Filter_Y, nullptr);
        shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderInfo.nextStage = 0;
        shaderInfo.pName = "computeMain_Filter_Y";
        shaderInfo.codeSize = shaderCode.codeSize;
        shaderInfo.pCode = shaderCode.pCode;
        vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_Filter_Y);
        NVVK_DBG_NAME(computeShader_Filter_Y);

        vkDestroyShaderEXT(device, computeShader_varianceConvolution, nullptr);
        shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderInfo.nextStage = 0;
        shaderInfo.pName = "computeMain_varianceConvolution";
        shaderInfo.codeSize = shaderCode.codeSize;
        shaderInfo.pCode = shaderCode.pCode;
        vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_varianceConvolution);
        NVVK_DBG_NAME(computeShader_varianceConvolution);
    }
}

void SVGF::getFeatures(VkCommandBuffer cmd) {
    NVVK_DBG_SCOPE(cmd);

    VkExtent3D groupSize = nvvk::getGroupCounts(VkExtent3D{ pushConstant.screenSize.x, pushConstant.screenSize.y, 1 }, VkExtent3D{ 32, 32, 1 });
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);
    VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;

    vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getDepthGradient);
    vkCmdPushConstants2(cmd, &pushInfo);
    vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);

    vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getEffectiveIrradiance);
    vkCmdPushConstants2(cmd, &pushInfo);
    vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);

    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_varianceEstimate);
    vkCmdPushConstants2(cmd, &pushInfo);
    vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
}
void SVGF::filter(VkCommandBuffer cmd) {
    NVVK_DBG_SCOPE(cmd);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

    VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
    for (int i = 0; i < SVGF_FILTER_COUNT; ++i) {
        pushConstant.filterIndex = i;
        vkCmdPushConstants2(cmd, &pushInfo);
        VkExtent3D groupSize = nvvk::getGroupCounts(VkExtent3D{ pushConstant.screenSize.x, pushConstant.screenSize.y, 1 }, VkExtent3D{ 32, 32, 1 });

        vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_varianceConvolution);
        vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
        nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

        vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_Filter_X);
        vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
        nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

        vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_varianceConvolution);
        vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
        nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

        vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_Filter_Y);
        vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
        nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    }
}