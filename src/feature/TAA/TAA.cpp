#include "./TAA.h"
#include <common/Application/Application.h>
#include <nvvk/formats.hpp>
#include <common/Image/Image.h>
#include <common/Shader/Shader.h>
#include <nvvk/default_structs.hpp>
#include <nvgui/property_editor.hpp>
#include <nvvk/compute_pipeline.hpp>

using namespace FzbRenderer;

TAA::TAA() {

}

void TAA::init(TAACreateInfo createInfo) {
	this->setting = createInfo;

    pushConstant.Halton_2_3[0] = { 0.0f, -1.0f / 3.0f };
    pushConstant.Halton_2_3[1] = { -1.0f / 2.0f, 1.0f / 3.0f };
    pushConstant.Halton_2_3[2] = { 1.0f / 2.0f, -7.0f / 9.0f };
    pushConstant.Halton_2_3[3] = { -3.0f / 4.0f, -1.0f / 9.0f };
    pushConstant.Halton_2_3[4] = { 1.0f / 4.0f, 5.0f / 9.0f };
    pushConstant.Halton_2_3[5] = { -1.0f / 4.0f, -5.0f / 9.0f };
    pushConstant.Halton_2_3[6] = { 3.0f / 4.0f, 1.0f / 9.0f };
    pushConstant.Halton_2_3[7] = { -7.0f / 8.0f, 7.0f / 9.0f };

    Feature::createGBuffer(false, false, 2);

    createDescriptorSetLayout();
    createDescriptorSet();
    createPipeline();
    compileAndCreateShaders();

    Feature::init();
}

void TAA::clean() {
    VkDevice device = Application::app->getDevice();
    vkDestroyShaderEXT(device, computeShader_mergeRenderTarget, nullptr);

    Feature::clean();
}
void TAA::uiRender() {

}
void TAA::resize(VkCommandBuffer cmd, const VkExtent2D& size, nvvk::Image images[3]){
    NVVK_CHECK(gBuffers.update(cmd, size));

    nvvk::WriteSetContainer write{};

    VkWriteDescriptorSet	renderTargetWrite =
        staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_TAA::eRendereTarget_Current, 0, 0, 1);
    write.append(renderTargetWrite, gBuffers.m_res.gBufferColor[0]);

    renderTargetWrite =
        staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_TAA::eRenderTarget_Last, 0, 0, 1);
    write.append(renderTargetWrite, gBuffers.m_res.gBufferColor[1]);

    renderTargetWrite =
        staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_TAA::eDepthImage, 0, 0, 1);
    setting.depthImage = images[0];
    write.append(renderTargetWrite, setting.depthImage);

    renderTargetWrite =
        staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_TAA::eVelocityImage, 0, 0, 1);
    setting.velocityImage = images[1];
    write.append(renderTargetWrite, setting.velocityImage);

    renderTargetWrite =
        staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_TAA::eRenderTarget_Final, 0, 0, 1);
    setting.renderTarget = images[2];
    write.append(renderTargetWrite, setting.renderTarget);

    vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);

    pushConstant.screenSize = { size.width, size.height };
}
void TAA::preRender() {
    static int frameIndex = 0;
    pushConstant.frameIndex = frameIndex;
    ++frameIndex;

    Scene& scene = Application::sceneResource;
    pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;

    nvvk::Buffer& bSceneInfo = Application::sceneResource.bSceneInfo;
    shaderio::SceneInfo& sceneInfo = Application::sceneResource.sceneInfo;
    std::shared_ptr<nvutils::CameraManipulator> cameraManip = Application::sceneResource.cameraManip;

    //glm::mat4 projMatrix = cameraManip->getPerspectiveMatrix();
    //shaderio::float2 sampleOffset = pushConstant.Halton_2_3[pushConstant.frameIndex % 8];
    //projMatrix[2][0] += sampleOffset.x / setting.renderTarget.extent.width;
    //projMatrix[2][1] += sampleOffset.y / setting.renderTarget.extent.height;
    //
    //const glm::mat4 viewMatrix = cameraManip->getViewMatrix();
    //sceneInfo.viewProjMatrix = projMatrix * viewMatrix;
    //sceneInfo.projInvMatrix = glm::inverse(projMatrix);

    pushConstant.mergeRatio = setting.mergeRatio;
}

void TAA::createDescriptorSetLayout() {
    SCOPED_TIMER(__FUNCTION__);
    nvvk::DescriptorBindings bindings;

    bindings.addBinding({
            .binding = (uint32_t)shaderio::BindingPoints_TAA::eRendereTarget_Current,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_ALL });

    bindings.addBinding({
        .binding = (uint32_t)shaderio::BindingPoints_TAA::eRenderTarget_Last,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_ALL });

    bindings.addBinding({
        .binding = (uint32_t)shaderio::BindingPoints_TAA::eRenderTarget_Final,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_ALL });

    bindings.addBinding({
        .binding = (uint32_t)shaderio::BindingPoints_TAA::eVelocityImage,
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_ALL });

    bindings.addBinding({
        .binding = (uint32_t)shaderio::BindingPoints_TAA::eDepthImage,
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_ALL });

    staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

    LOGI("Fzb PathGuiding static descriptor layout created\n");
    NVVK_DBG_NAME(staticDescPack.getLayout());
    NVVK_DBG_NAME(staticDescPack.getPool());
    NVVK_DBG_NAME(staticDescPack.getSet(0));
}
void TAA::createDescriptorSet() {}
void TAA::createPipeline() {
    const VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_ALL,
        .offset = 0,
        .size = sizeof(shaderio::TAAPushConstant)
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
void TAA::compileAndCreateShaders() {
    SCOPED_TIMER(__FUNCTION__);

    std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
    std::filesystem::path shaderSource;
    VkShaderModuleCreateInfo shaderCode;

    const VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_ALL ,
        .offset = 0,
        .size = sizeof(shaderio::TAAPushConstant),
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
        shaderSource = shaderPath / "mergeRenderTarget.slang";
        shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

        vkDestroyShaderEXT(device, computeShader_mergeRenderTarget, nullptr);
        shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderInfo.nextStage = 0;
        shaderInfo.pName = "computeMain_mergeRenderTarget";
        shaderInfo.codeSize = shaderCode.codeSize;
        shaderInfo.pCode = shaderCode.pCode;
        vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_mergeRenderTarget);
        NVVK_DBG_NAME(computeShader_mergeRenderTarget);
    }
}

void TAA::mergeResult(VkCommandBuffer cmd) {
    NVVK_DBG_SCOPE(cmd);

    VkPushConstantsInfo pushInfo = {
        .sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
        .layout = pipelineLayout,
        .stageFlags = VK_SHADER_STAGE_ALL,
        .offset = 0,
        .size = sizeof(shaderio::TAAPushConstant),
        .pValues = &pushConstant,
    };
    
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

    {
        VkImageCopy2 copyRegion = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
            .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .srcOffset = {0, 0, 0},
            .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .dstOffset = {0, 0, 0},
            .extent = {gBuffers.getSize().width, gBuffers.getSize().height, 1}
        };
        VkCopyImageInfo2 copyInfo = {
            .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2,
            .srcImage = setting.renderTarget.image,
            .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
            .dstImage = gBuffers.getColorImage(0),
            .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
            .regionCount = 1,
            .pRegions = &copyRegion
        };
        vkCmdCopyImage2(cmd, &copyInfo);
    }
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT_KHR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
    vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_mergeRenderTarget);
    vkCmdPushConstants2(cmd, &pushInfo);
    VkExtent3D groupSize = nvvk::getGroupCounts(setting.renderTarget.extent, VkExtent3D{ 32, 32, 1 });
    vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT_KHR);

    {
        VkImageCopy2 copyRegion = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
            .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .srcOffset = {0, 0, 0},
            .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .dstOffset = {0, 0, 0},
            .extent = {gBuffers.getSize().width, gBuffers.getSize().height, 1}
        };
        VkCopyImageInfo2 copyInfo = {
            .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2,
            .srcImage = setting.renderTarget.image,
            .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
            .dstImage = gBuffers.getColorImage(1),
            .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
            .regionCount = 1,
            .pRegions = &copyRegion
        };
        vkCmdCopyImage2(cmd, &copyInfo);
    }
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
}

