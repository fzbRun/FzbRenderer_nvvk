#include "./DeferredRenderer.h"
#include <common/Application/Application.h>
#include <nvvk/formats.hpp>
#include "common/Shader/nvvk/shaderio.h"
#include "./shaders/spv/deferredShaders.h"
#include <nvgui/sky.hpp>
#include <nvvk/default_structs.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "common/Shader/Shader.h"

FzbRenderer::DeferredRenderer::DeferredRenderer(RendererCreateInfo& createInfo) {

}

void FzbRenderer::DeferredRenderer::createImage() {
    VkSampler linearSampler{};
    NVVK_CHECK(Application::samplerPool.acquireSampler(linearSampler));
    NVVK_DBG_NAME(linearSampler);

    nvvk::GBufferInitInfo gBufferInit{
        .allocator = &Application::allocator,
        .colorFormats = {VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM},
        .depthFormat = nvvk::findDepthFormat(Application::app->getPhysicalDevice()),
        .imageSampler = linearSampler,
        .descriptorPool = Application::app->getTextureDescriptorPool(),
    };
    gBuffers.init(gBufferInit);
}
void FzbRenderer::DeferredRenderer::createGraphicsDescriptorSetLayout() {
    nvvk::DescriptorBindings bindings;
    bindings.addBinding({ .binding = shaderio::BindingPoints::eTextures,
                         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                         .descriptorCount = 10,
                         .stageFlags = VK_SHADER_STAGE_ALL },
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
        | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
    descPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

    NVVK_DBG_NAME(descPack.getLayout());
    NVVK_DBG_NAME(descPack.getPool());
    NVVK_DBG_NAME(descPack.getSet(0));
}
void FzbRenderer::DeferredRenderer::createGraphicsPipelineLayout() {
    const VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
        .offset = 0,
        .size = sizeof(shaderio::PushConstant)
    };

    const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = descPack.getLayoutPtr(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange,
    };
    NVVK_CHECK(vkCreatePipelineLayout(Application::app->getDevice(), &pipelineLayoutInfo, nullptr, &graphicPipelineLayout));
    NVVK_DBG_NAME(graphicPipelineLayout);
}
void FzbRenderer::DeferredRenderer::compileAndCreateShaders() {
    SCOPED_TIMER(__FUNCTION__);

    //编译后的数据放在了slangCompiler中
    std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
    std::filesystem::path shaderSource = shaderPath / "deferredShaders.slang";
    VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, deferredShaders_slang);

    vkDestroyShaderEXT(Application::app->getDevice(), vertexShader, nullptr);
    vkDestroyShaderEXT(Application::app->getDevice(), fragmentShader, nullptr);

    const VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
        .offset = 0,
        .size = sizeof(shaderio::PushConstant),
    };

    VkShaderCreateInfoEXT shaderInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
        .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
        .pName = "main",
        .setLayoutCount = 1,
        .pSetLayouts = descPack.getLayoutPtr(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange,
    };

    shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderInfo.pName = "vertexMain";
    shaderInfo.codeSize = shaderCode.codeSize;
    shaderInfo.pCode = shaderCode.pCode;
    vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &vertexShader);
    NVVK_DBG_NAME(vertexShader);

    shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderInfo.nextStage = 0;
    shaderInfo.pName = "fragmentMain";
    shaderInfo.codeSize = shaderCode.codeSize;
    shaderInfo.pCode = shaderCode.pCode;
    vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &fragmentShader);
    NVVK_DBG_NAME(fragmentShader);
}
void FzbRenderer::DeferredRenderer::updateTextures() {
    if (Application::sceneResource.textures.empty())
        return;

    nvvk::WriteSetContainer write{};
    VkWriteDescriptorSet    allTextures =
        descPack.makeWrite(shaderio::BindingPoints::eTextures, 0, 1, uint32_t(Application::sceneResource.textures.size()));
    nvvk::Image* allImages = Application::sceneResource.textures.data();
    write.append(allTextures, allImages);
    vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void FzbRenderer::DeferredRenderer::init() {
    createImage();
    createGraphicsDescriptorSetLayout();
    createGraphicsPipelineLayout();
    compileAndCreateShaders();
    updateTextures();
}

void FzbRenderer::DeferredRenderer::clean() {
    VkDevice device = Application::app->getDevice();

    descPack.deinit();
    vkDestroyPipelineLayout(device, graphicPipelineLayout, nullptr);
    vkDestroyShaderEXT(device, vertexShader, nullptr);
    vkDestroyShaderEXT(device, fragmentShader, nullptr);

    gBuffers.deinit();
}

void FzbRenderer::DeferredRenderer::uiRender() {
    namespace PE = nvgui::PropertyEditor;
    if (ImGui::Begin("Viewport"))
        ImGui::Image(ImTextureID(gBuffers.getDescriptorSet(eImgTonemapped)), ImGui::GetContentRegionAvail());
    ImGui::End();
}

void FzbRenderer::DeferredRenderer::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
    NVVK_CHECK(gBuffers.update(cmd, size));
}

void FzbRenderer::DeferredRenderer::render(VkCommandBuffer cmd) {
    NVVK_DBG_SCOPE(cmd);
    shaderio::PushConstant pushValues{
        .sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address,
        //.metallicRoughnessOverride = metallicRoughnessOverride,
    };

    const VkPushConstantsInfo pushInfo{
        .sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
        .layout = graphicPipelineLayout,
        .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
        .offset = 0,
        .size = sizeof(shaderio::PushConstant),
        .pValues = &pushValues,
    };

    if (Application::sceneResource.sceneInfo.useSky)
    {
        const glm::mat4& viewMatrix = Application::sceneResource.cameraManip->getViewMatrix();
        const glm::mat4& projMatrix = Application::sceneResource.cameraManip->getPerspectiveMatrix();
        Application::skySimple.runCompute(cmd, Application::app->getViewportSize(), viewMatrix, projMatrix,
            Application::sceneResource.sceneInfo.skySimpleParam, gBuffers.getDescriptorImageInfo(eImgRendered));
    }

    VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
    colorAttachment.loadOp = Application::sceneResource.sceneInfo.useSky ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.imageView = gBuffers.getColorImageView(eImgRendered);
    colorAttachment.clearValue = { .color = {Application::sceneResource.sceneInfo.backgroundColor.x,
                                            Application::sceneResource.sceneInfo.backgroundColor.y,
                                            Application::sceneResource.sceneInfo.backgroundColor.z, 1.0f} };
    VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
    depthAttachment.imageView = gBuffers.getDepthImageView();
    depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };

    VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
    renderingInfo.renderArea = DEFAULT_VkRect2D(gBuffers.getSize());
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthAttachment;

    nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(eImgRendered), VK_IMAGE_LAYOUT_GENERAL,
                                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
    const VkBindDescriptorSetsInfo bindDescriptorSetsInfo{
        .sType = VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO,
        .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
        .layout = graphicPipelineLayout,
        .firstSet = 0,
        .descriptorSetCount = 1,
        .pDescriptorSets = descPack.getSetPtr(),
    };
    //vkCmdBindDescriptorSets2(cmd, &bindDescriptorSetsInfo);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicPipelineLayout, 0, 1,
        descPack.getSetPtr(), 0, nullptr);

    vkCmdBeginRendering(cmd, &renderingInfo);

    //使用VK_EXT_SHADER_OBJECT_EXTENSION_NAME后可以不需要pipeline，直接通过命令设置渲染设置和着色器
    dynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
    dynamicPipeline.cmdApplyAllStates(cmd);
    dynamicPipeline.cmdSetViewportAndScissor(cmd, Application::app->getViewportSize());
    vkCmdSetDepthTestEnable(cmd, VK_TRUE);
    dynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader, .fragment = fragmentShader });

    VkVertexInputBindingDescription2EXT bindingDescription{};
    VkVertexInputAttributeDescription2EXT attributeDescription = {};
    vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

    for (size_t i = 0; i < Application::sceneResource.instances.size(); ++i)
    {
        uint32_t meshIndex = Application::sceneResource.instances[i].meshIndex;
        const shaderio::Mesh& mesh = Application::sceneResource.meshes[meshIndex];
        const shaderio::TriangleMesh& triMesh = mesh.triMesh;

        pushValues.normalMatrix = glm::transpose(glm::inverse(glm::mat3(Application::sceneResource.instances[i].transform)));
        pushValues.instanceIndex = int(i);
        vkCmdPushConstants2(cmd, &pushInfo);

        uint32_t bufferIndex = Application::sceneResource.meshToBufferIndex[meshIndex];
        const nvvk::Buffer& v = Application::sceneResource.bDatas[bufferIndex];

        vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

        vkCmdDrawIndexed(cmd, triMesh.indices.count, 1, 0, 0, 0);
    }

    vkCmdEndRendering(cmd);
    nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(eImgRendered),
                                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });

    postProcess(cmd);
}
void FzbRenderer::DeferredRenderer::postProcess(VkCommandBuffer cmd) {
    NVVK_DBG_SCOPE(cmd);
    Application::tonemapper.runCompute(cmd, gBuffers.getSize(), Application::tonemapperData, gBuffers.getDescriptorImageInfo(eImgRendered),
        gBuffers.getDescriptorImageInfo(eImgTonemapped));
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);
}

void FzbRenderer::DeferredRenderer::onLastHeadlessFrame() {
    Application::app->saveImageToFile(gBuffers.getColorImage(eImgTonemapped), gBuffers.getSize(),
        nvutils::getExecutablePath().replace_extension(".jpg").string());
}