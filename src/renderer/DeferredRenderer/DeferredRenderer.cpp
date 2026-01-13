#include "./DeferredRenderer.h"
#include <common/Application/Application.h>
#include <nvvk/formats.hpp>
#include "common/Shader/nvvk/shaderio.h"
#include "./shaders/spv/deferredShaders.h"
#include <nvgui/sky.hpp>
#include <nvvk/default_structs.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "common/Shader/Shader.h"

FzbRenderer::DeferredRenderer::DeferredRenderer(pugi::xml_node& rendererNode) {

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
        .size = sizeof(shaderio::DefaultPushConstant),
    };

    VkShaderCreateInfoEXT shaderInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
        .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
        .pName = "main",
        .setLayoutCount = 1,
        .pSetLayouts = staticDescPack.getLayoutPtr(),
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
void FzbRenderer::DeferredRenderer::init() {
    FzbRenderer::Renderer::createGBuffer(true);
    Renderer::createDescriptorSetLayout();
    Renderer::createPipelineLayout();
    compileAndCreateShaders();
    Renderer::addTextureArrayDescriptor();
}

void FzbRenderer::DeferredRenderer::clean() {
    Renderer::clean();
    VkDevice device = Application::app->getDevice();

    vkDestroyShaderEXT(device, vertexShader, nullptr);
    vkDestroyShaderEXT(device, fragmentShader, nullptr);
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
    shaderio::DefaultPushConstant pushValues{
        .sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address,
        //.metallicRoughnessOverride = metallicRoughnessOverride,
    };

    const VkPushConstantsInfo pushInfo{
        .sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
        .layout = pipelineLayout,
        .stageFlags = VK_SHADER_STAGE_ALL,
        .offset = 0,
        .size = sizeof(shaderio::DefaultPushConstant),
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
        .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT,
        .layout = pipelineLayout,
        .firstSet = 0,
        .descriptorSetCount = 1,
        .pDescriptorSets = staticDescPack.getSetPtr(),
    };
    //vkCmdBindDescriptorSets2(cmd, &bindDescriptorSetsInfo);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
        staticDescPack.getSetPtr(), 0, nullptr);

    vkCmdBeginRendering(cmd, &renderingInfo);

    //使用VK_EXT_SHADER_OBJECT_EXTENSION_NAME后可以不需要pipeline，直接通过命令设置渲染设置和着色器
    graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
    graphicsDynamicPipeline.cmdApplyAllStates(cmd);
    graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, Application::app->getViewportSize());
    vkCmdSetDepthTestEnable(cmd, VK_TRUE);
    graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader, .fragment = fragmentShader });

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

        uint32_t bufferIndex = Application::sceneResource.getMeshBufferIndex(meshIndex);
        const nvvk::Buffer& v = Application::sceneResource.bDatas[bufferIndex];

        vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

        vkCmdDrawIndexed(cmd, triMesh.indices.count, 1, 0, 0, 0);
    }

    vkCmdEndRendering(cmd);
    //nvvk的cmdImageMemoryBarrier函数可以根据image的old和new layout判断前后的可能的所有stage和access
    nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(eImgRendered),
                                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
    Renderer::postProcess(cmd);
}