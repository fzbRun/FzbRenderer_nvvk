#define TINYGLTF_IMPLEMENTATION         // Implementation of the GLTF loader library
#define STB_IMAGE_IMPLEMENTATION        // Implementation of the image loading library
#define STB_IMAGE_WRITE_IMPLEMENTATION  // Implementation of the image writing library
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1  // Use dynamic Vulkan functions for VMA (Vulkan Memory Allocator)
#define VMA_IMPLEMENTATION              // Implementation of the Vulkan Memory Allocator
#define VMA_LEAK_LOG_FORMAT(format, ...)                                                                               \
  {                                                                                                                    \
    printf((format), __VA_ARGS__);                                                                                     \
    printf("\n");                                                                                                      \
  }

#include <nvapp/application.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvutils/file_operations.hpp>
#include <nvvk/context.hpp>
#include <nvvk/validation_settings.hpp>
#include <nvaftermath/aftermath.hpp>
#include <nvvk/check_error.hpp>
#include <nvutils/logger.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvslang/slang.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/descriptors.hpp>

#include "gltf_utils.hpp"  // GLTF utilities for loading and importing GLTF models
#include "utils.hpp"       // Common utilities for the sample application
#include "path_utils.hpp"  // Path utilities for handling resources file paths

#include <nvshaders_host/sky.hpp>
#include <nvshaders_host/tonemapper.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/formats.hpp>

#include "shaderio.h"
#include "./_autogen/sky_simple.slang.h"  // from nvpro_core2
#include "./_autogen/tonemapper.slang.h"  //   "    "
#include "./_autogen/foundation.slang.h"  // Local shader

#include <nvgui/sky.hpp>
#include <nvgui/camera.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <nvgui/tonemapper.hpp>
#include <nvvk/default_structs.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvapp/elem_default_menu.hpp>

class FzbRenderer_nvvk : public nvapp::IAppElement
{
    enum
    {
        eImgRendered,
        eImgTonemapped
    };

public:
    FzbRenderer_nvvk() = default;
    ~FzbRenderer_nvvk() override = default;

    //app初始化时调用
    void onAttach(nvapp::Application* app) override
    {
        m_app = app;
        VmaAllocatorCreateInfo allocatorInfo = {
            .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
            .physicalDevice = app->getPhysicalDevice(),
            .device = app->getDevice(),
            .instance = app->getInstance(),
            .vulkanApiVersion = VK_API_VERSION_1_4,
        };
        m_allocator.init(allocatorInfo);

        m_stagingUploader.init(&m_allocator, true);   //所有的CPU、GPU只一方可见的缓冲的交互都要经过暂存缓冲区

        m_slangCompiler.addSearchPaths(nvsamples::getShaderDirs());
        m_slangCompiler.defaultTarget();
        m_slangCompiler.defaultOptions();
        m_slangCompiler.addOption({ slang::CompilerOptionName::DebugInformation,
            {slang::CompilerOptionValueKind::Int, SLANG_DEBUG_INFO_LEVEL_MAXIMAL} });

//#if defined(AFTERMATH_AVAILABLE)
//        m_slangCompiler.setCompileCallback([&](const std::filesystem::path& sourceFile, const uint32_t* spirvCode, size_t spirvSize) {
//            std::span<const uint32_t> data(spirvCode, spirvSize / sizeof(uint32_t));
//            AftermathCrashTracker::getInstance().addShaderBinary(data);
//        }
//#endif

            m_samplerPool.init(app->getDevice());
            VkSampler linearSampler{};
            NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
            NVVK_DBG_NAME(linearSampler);

            nvvk::GBufferInitInfo gBufferInit{
                .allocator = &m_allocator,
                .colorFormats = {VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM},
                .depthFormat = nvvk::findDepthFormat(m_app->getPhysicalDevice()),
                .imageSampler = linearSampler,
                .descriptorPool = m_app->getTextureDescriptorPool(),
            };
            m_gBuffers.init(gBufferInit);

            createScene();
            createGraphicsDescriptorSetLayout();
            createGraphicsPipelineLayout();
            compileAndCreateGraphicsShaders();
            updateTextures();

            m_skySimple.init(&m_allocator, std::span(sky_simple_slang));
            m_tonemapper.init(&m_allocator, std::span(tonemapper_slang));
            }
            void createScene() {
                SCOPED_TIMER(__FUNCTION__);

                VkCommandBuffer cmd = m_app->createTempCmdBuffer();
                {
                    tinygltf::Model teapotModel =
                        nvsamples::loadGltfResources(nvutils::findFile("nvvk/teapot.gltf", nvsamples::getResourcesDirs()));
                    tinygltf::Model planeModel =
                        nvsamples::loadGltfResources(nvutils::findFile("nvvk/plane.gltf", nvsamples::getResourcesDirs()));

                    {
                        std::filesystem::path imageFilename = nvutils::findFile("nvvk/tiled_floor.png", nvsamples::getResourcesDirs());
                        nvvk::Image texture = nvsamples::loadAndCreateImage(cmd, m_stagingUploader, m_app->getDevice(), imageFilename);
                        NVVK_DBG_NAME(texture.image);
                        m_samplerPool.acquireSampler(texture.descriptor.sampler);
                        m_textures.emplace_back(texture);
                    }

                    {
                        nvsamples::importGltfData(m_sceneResource, teapotModel, m_stagingUploader);
                        nvsamples::importGltfData(m_sceneResource, planeModel, m_stagingUploader);
                    }
                }

                m_sceneResource.materials = {
                    {.baseColorFactor = glm::vec4(0.8f, 1.0f, 0.6f, 1.0f), .metallicFactor = 0.5f, .roughnessFactor = 0.5f},
                    {.baseColorFactor = glm::vec4(1.0f), .metallicFactor = 0.1f, .roughnessFactor = 0.8f, .baseColorTextureIndex = 0},
                };

                m_sceneResource.instances = {
                    {.transform = glm::translate(glm::mat4(1.0f), glm::vec3(0)) * glm::scale(glm::mat4(1.0f), glm::vec3(0.5f)),
                     .materialIndex = 0,
                     .meshIndex = 0},
                    {.transform = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.9f, 0.0f)), glm::vec3(2.0f)),
                     .materialIndex = 1,
                     .meshIndex = 1},
                };

                nvsamples::createGltfSceneInfoBuffer(m_sceneResource, m_stagingUploader);
                m_stagingUploader.cmdUploadAppended(cmd);

                shaderio::GltfSceneInfo& sceneInfo = m_sceneResource.sceneInfo;
                sceneInfo.useSky = false;
                sceneInfo.instances = (shaderio::GltfInstance*)m_sceneResource.bInstances.address;
                sceneInfo.meshes = (shaderio::GltfMesh*)m_sceneResource.bMeshes.address;
                sceneInfo.materials = (shaderio::GltfMetallicRoughness*)m_sceneResource.bMaterials.address;
                sceneInfo.backgroundColor = { 0.85f, 0.85f, 0.85f };
                sceneInfo.numLights = 1;
                sceneInfo.punctualLights[0].color = glm::vec3(1.0f);
                sceneInfo.punctualLights[0].intensity = 4.0f;
                sceneInfo.punctualLights[0].position = glm::vec3(1.0f);
                sceneInfo.punctualLights[0].direction = glm::vec3(1.0f);
                sceneInfo.punctualLights[0].type = shaderio::GltfLightType::ePoint;
                sceneInfo.punctualLights[0].coneAngle = 0.9f;

                m_app->submitAndWaitTempCmdBuffer(cmd);

                m_cameraManip->setClipPlanes({ 0.01f, 100.0f });
                m_cameraManip->setLookat({ 0.0f, 0.5f, 5.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f });
            }
            void createGraphicsDescriptorSetLayout() {
                nvvk::DescriptorBindings bindings;
                bindings.addBinding({ .binding = shaderio::BindingPoints::eTextures,
                                     .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                     .descriptorCount = 10,
                                     .stageFlags = VK_SHADER_STAGE_ALL },
                    VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
                    | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
                m_descPack.init(bindings, m_app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
                    VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

                NVVK_DBG_NAME(m_descPack.getLayout());
                NVVK_DBG_NAME(m_descPack.getPool());
                NVVK_DBG_NAME(m_descPack.getSet(0));
            }
            void createGraphicsPipelineLayout() {
                const VkPushConstantRange pushConstantRange{
                    .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
                    .offset = 0,
                    .size = sizeof(shaderio::TutoPushConstant)
                };

                const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    .setLayoutCount = 1,
                    .pSetLayouts = m_descPack.getLayoutPtr(),
                    .pushConstantRangeCount = 1,
                    .pPushConstantRanges = &pushConstantRange,
                };
                NVVK_CHECK(vkCreatePipelineLayout(m_app->getDevice(), &pipelineLayoutInfo, nullptr, &m_graphicPipelineLayout));
                NVVK_DBG_NAME(m_graphicPipelineLayout);
            }
            VkShaderModuleCreateInfo compileSlangShader(const std::filesystem::path & filename, const std::span<const uint32_t>&spirv) {
                SCOPED_TIMER(__FUNCTION__);

                VkShaderModuleCreateInfo shaderCode = nvsamples::getShaderModuleCreateInfo(spirv);

                std::filesystem::path shaderSource = nvutils::findFile(filename, nvsamples::getShaderDirs());
                if (m_slangCompiler.compileFile(shaderSource))
                {
                    shaderCode.codeSize = m_slangCompiler.getSpirvSize();
                    shaderCode.pCode = m_slangCompiler.getSpirv();
                }
                else
                {
                    LOGE("Error compiling shers: %s\n%s\n", shaderSource.string().c_str(),
                        m_slangCompiler.getLastDiagnosticMessage().c_str());
                }
                return shaderCode;
            }
            void compileAndCreateGraphicsShaders() {
                SCOPED_TIMER(__FUNCTION__);

                //编译后的数据放在了m_slangCompiler中
                VkShaderModuleCreateInfo shaderCode = compileSlangShader("foundation.slang", foundation_slang);

                vkDestroyShaderEXT(m_app->getDevice(), m_vertexShader, nullptr);
                vkDestroyShaderEXT(m_app->getDevice(), m_fragmentShader, nullptr);

                const VkPushConstantRange pushConstantRange{
                    .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
                    .offset = 0,
                    .size = sizeof(shaderio::TutoPushConstant),
                };

                VkShaderCreateInfoEXT shaderInfo{
                    .sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
                    .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
                    .pName = "main",
                    .setLayoutCount = 1,
                    .pSetLayouts = m_descPack.getLayoutPtr(),
                    .pushConstantRangeCount = 1,
                    .pPushConstantRanges = &pushConstantRange,
                };

                shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
                shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
                shaderInfo.pName = "vertexMain";
                shaderInfo.codeSize = shaderCode.codeSize;
                shaderInfo.pCode = shaderCode.pCode;
                vkCreateShadersEXT(m_app->getDevice(), 1U, &shaderInfo, nullptr, &m_vertexShader);
                NVVK_DBG_NAME(m_vertexShader);

                shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
                shaderInfo.nextStage = 0;
                shaderInfo.pName = "fragmentMain";
                shaderInfo.codeSize = shaderCode.codeSize;
                shaderInfo.pCode = shaderCode.pCode;
                vkCreateShadersEXT(m_app->getDevice(), 1U, &shaderInfo, nullptr, &m_fragmentShader);
                NVVK_DBG_NAME(m_fragmentShader);
            }
            void updateTextures() {
                if (m_textures.empty())
                    return;

                nvvk::WriteSetContainer write{};
                VkWriteDescriptorSet    allTextures =
                    m_descPack.makeWrite(shaderio::BindingPoints::eTextures, 0, 1, uint32_t(m_textures.size()));
                nvvk::Image* allImages = m_textures.data();
                write.append(allTextures, allImages);
                vkUpdateDescriptorSets(m_app->getDevice(), write.size(), write.data(), 0, nullptr);
            }

            void onDetach() override {
                NVVK_CHECK(vkQueueWaitIdle(m_app->getQueue(0).queue));

                VkDevice device = m_app->getDevice();

                m_descPack.deinit();
                vkDestroyPipelineLayout(device, m_graphicPipelineLayout, nullptr);
                vkDestroyShaderEXT(device, m_vertexShader, nullptr);
                vkDestroyShaderEXT(device, m_fragmentShader, nullptr);

                m_allocator.destroyBuffer(m_sceneResource.bSceneInfo);
                m_allocator.destroyBuffer(m_sceneResource.bMeshes);
                m_allocator.destroyBuffer(m_sceneResource.bMaterials);
                m_allocator.destroyBuffer(m_sceneResource.bInstances);
                for (auto& gltfData : m_sceneResource.bGltfDatas)
                    m_allocator.destroyBuffer(gltfData);
                for (auto& texture : m_textures)
                    m_allocator.destroyImage(texture);

                m_gBuffers.deinit();
                m_stagingUploader.deinit();
                m_skySimple.deinit();
                m_tonemapper.deinit();
                m_samplerPool.deinit();
                m_allocator.deinit();
            }

            void onUIRender() override {
                namespace PE = nvgui::PropertyEditor;
                if (ImGui::Begin("Viewport"))
                    ImGui::Image(ImTextureID(m_gBuffers.getDescriptorSet(eImgTonemapped)), ImGui::GetContentRegionAvail());
                ImGui::End();

                if (ImGui::Begin("Settings"))
                {
                    if (ImGui::CollapsingHeader("Camera"))
                        nvgui::CameraWidget(m_cameraManip);
                    if (ImGui::CollapsingHeader("Environment"))
                    {
                        ImGui::Checkbox("USE Sky", (bool*)&m_sceneResource.sceneInfo.useSky);
                        if (m_sceneResource.sceneInfo.useSky)
                            nvgui::skySimpleParametersUI(m_sceneResource.sceneInfo.skySimpleParam);
                        else
                        {
                            PE::begin();
                            PE::ColorEdit3("Background", (float*)&m_sceneResource.sceneInfo.backgroundColor);
                            PE::end();
                            // Light
                            PE::begin();
                            if (m_sceneResource.sceneInfo.punctualLights[0].type == shaderio::GltfLightType::ePoint
                                || m_sceneResource.sceneInfo.punctualLights[0].type == shaderio::GltfLightType::eSpot)
                                PE::DragFloat3("Light Position", glm::value_ptr(m_sceneResource.sceneInfo.punctualLights[0].position),
                                    1.0f, -20.0f, 20.0f, "%.2f", ImGuiSliderFlags_None, "Position of the light");
                            if (m_sceneResource.sceneInfo.punctualLights[0].type == shaderio::GltfLightType::eDirectional
                                || m_sceneResource.sceneInfo.punctualLights[0].type == shaderio::GltfLightType::eSpot)
                                PE::SliderFloat3("Light Direction", glm::value_ptr(m_sceneResource.sceneInfo.punctualLights[0].direction),
                                    -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_None, "Direction of the light");

                            PE::SliderFloat("Light Intensity", &m_sceneResource.sceneInfo.punctualLights[0].intensity, 0.0f, 1000.0f,
                                "%.2f", ImGuiSliderFlags_Logarithmic, "Intensity of the light");
                            PE::ColorEdit3("Light Color", glm::value_ptr(m_sceneResource.sceneInfo.punctualLights[0].color),
                                ImGuiColorEditFlags_NoInputs, "Color of the light");
                            PE::Combo("Light Type", (int*)&m_sceneResource.sceneInfo.punctualLights[0].type,
                                "Point\0Spot\0Directional\0", 3, "Type of the light (Point, Spot, Directional)");
                            if (m_sceneResource.sceneInfo.punctualLights[0].type == shaderio::GltfLightType::eSpot)
                                PE::SliderAngle("Cone Angle", &m_sceneResource.sceneInfo.punctualLights[0].coneAngle, 0.f, 90.f, "%.2f",
                                    ImGuiSliderFlags_AlwaysClamp, "Cone angle of the spot light");
                            PE::end();
                        }
                    }
                    if (ImGui::CollapsingHeader("Tonemapper"))
                        nvgui::tonemapperWidget(m_tonemapperData);

                    ImGui::Separator();
                    PE::begin();
                    PE::SliderFloat2("Metallic/Roughness Override", glm::value_ptr(m_metallicRoughnessOverride), -0.01f, 1.0f,
                        "%.2f", ImGuiSliderFlags_AlwaysClamp, "Override all material metallic and roughness");
                    PE::end();
                }
                ImGui::End();
            }

            void onResize(VkCommandBuffer cmd, const VkExtent2D & size) { NVVK_CHECK(m_gBuffers.update(cmd, size)); }

            void onRender(VkCommandBuffer cmd) {
                NVVK_DBG_SCOPE(cmd);
                updateSceneBuffer(cmd);
                rasterScene(cmd);
                postProcess(cmd);
            }
            void updateSceneBuffer(VkCommandBuffer cmd) {
                NVVK_DBG_SCOPE(cmd);
                const glm::mat4& viewMatrix = m_cameraManip->getViewMatrix();
                const glm::mat4& projMatrix = m_cameraManip->getPerspectiveMatrix();

                m_sceneResource.sceneInfo.viewProjMatrix = projMatrix * viewMatrix;
                m_sceneResource.sceneInfo.cameraPosition = m_cameraManip->getEye();
                m_sceneResource.sceneInfo.instances = (shaderio::GltfInstance*)m_sceneResource.bInstances.address;
                m_sceneResource.sceneInfo.meshes = (shaderio::GltfMesh*)m_sceneResource.bMeshes.address;
                m_sceneResource.sceneInfo.materials = (shaderio::GltfMetallicRoughness*)m_sceneResource.bMaterials.address;

                nvvk::cmdBufferMemoryBarrier(cmd, { m_sceneResource.bSceneInfo.buffer, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                                   VK_PIPELINE_STAGE_2_TRANSFER_BIT });
                vkCmdUpdateBuffer(cmd, m_sceneResource.bSceneInfo.buffer, 0, sizeof(shaderio::GltfSceneInfo), &m_sceneResource.sceneInfo);
                nvvk::cmdBufferMemoryBarrier(cmd, { m_sceneResource.bSceneInfo.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT });
            }
            void rasterScene(VkCommandBuffer cmd) {
                NVVK_DBG_SCOPE(cmd);

                shaderio::TutoPushConstant pushValues{
                    .sceneInfoAddress = (shaderio::GltfSceneInfo*)m_sceneResource.bSceneInfo.address,
                    .metallicRoughnessOverride = m_metallicRoughnessOverride,
                };
                const VkPushConstantsInfo pushInfo{
                    .sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
                    .layout = m_graphicPipelineLayout,
                    .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
                    .offset = 0,
                    .size = sizeof(shaderio::TutoPushConstant),
                    .pValues = &pushValues,
                };

                if (m_sceneResource.sceneInfo.useSky)
                {
                    const glm::mat4& viewMatrix = m_cameraManip->getViewMatrix();
                    const glm::mat4& projMatrix = m_cameraManip->getPerspectiveMatrix();
                    m_skySimple.runCompute(cmd, m_app->getViewportSize(), viewMatrix, projMatrix,
                        m_sceneResource.sceneInfo.skySimpleParam, m_gBuffers.getDescriptorImageInfo(eImgRendered));
                }

                VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
                colorAttachment.loadOp = m_sceneResource.sceneInfo.useSky ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
                colorAttachment.imageView = m_gBuffers.getColorImageView(eImgRendered);
                colorAttachment.clearValue = { .color = {m_sceneResource.sceneInfo.backgroundColor.x,
                                                        m_sceneResource.sceneInfo.backgroundColor.y,
                                                        m_sceneResource.sceneInfo.backgroundColor.z, 1.0f} };
                VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
                depthAttachment.imageView = m_gBuffers.getDepthImageView();
                depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };

                VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
                renderingInfo.renderArea = DEFAULT_VkRect2D(m_gBuffers.getSize());
                renderingInfo.colorAttachmentCount = 1;
                renderingInfo.pColorAttachments = &colorAttachment;
                renderingInfo.pDepthAttachment = &depthAttachment;

                nvvk::cmdImageMemoryBarrier(cmd, { m_gBuffers.getColorImage(eImgRendered), VK_IMAGE_LAYOUT_GENERAL,
                                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
                const VkBindDescriptorSetsInfo bindDescriptorSetsInfo{
                    .sType = VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO,
                    .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
                    .layout = m_graphicPipelineLayout,
                    .firstSet = 0,
                    .descriptorSetCount = 1,
                    .pDescriptorSets = m_descPack.getSetPtr(),
                };
                vkCmdBindDescriptorSets2(cmd, &bindDescriptorSetsInfo);

                vkCmdBeginRendering(cmd, &renderingInfo);

                m_dynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
                m_dynamicPipeline.cmdApplyAllStates(cmd);
                m_dynamicPipeline.cmdSetViewportAndScissor(cmd, m_app->getViewportSize());
                vkCmdSetDepthTestEnable(cmd, VK_TRUE);

                m_dynamicPipeline.cmdBindShaders(cmd, { .vertex = m_vertexShader, .fragment = m_fragmentShader });

                VkVertexInputBindingDescription2EXT bindingDescription{};
                VkVertexInputAttributeDescription2EXT attributeDescription = {};
                vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

                for (size_t i = 0; i < m_sceneResource.instances.size(); ++i)
                {
                    uint32_t meshIndex = m_sceneResource.instances[i].meshIndex;
                    const shaderio::GltfMesh& gltfMesh = m_sceneResource.meshes[meshIndex];
                    const shaderio::TriangleMesh& triMesh = gltfMesh.triMesh;

                    pushValues.normalMatrix = glm::transpose(glm::inverse(glm::mat3(m_sceneResource.instances[i].transform)));
                    pushValues.instanceIndex = int(i);
                    vkCmdPushConstants2(cmd, &pushInfo);

                    uint32_t bufferIndex = m_sceneResource.meshToBufferIndex[meshIndex];
                    const nvvk::Buffer& v = m_sceneResource.bGltfDatas[bufferIndex];

                    vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(gltfMesh.indexType));

                    vkCmdDrawIndexed(cmd, triMesh.indices.count, 1, 0, 0, 0);
                }

                vkCmdEndRendering(cmd);
                nvvk::cmdImageMemoryBarrier(cmd, { m_gBuffers.getColorImage(eImgRendered),
                                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
            }
            void postProcess(VkCommandBuffer cmd) {
                NVVK_DBG_SCOPE(cmd);
                m_tonemapper.runCompute(cmd, m_gBuffers.getSize(), m_tonemapperData, m_gBuffers.getDescriptorImageInfo(eImgRendered),
                    m_gBuffers.getDescriptorImageInfo(eImgTonemapped));
                nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);
            }

            void onUIMenu() override {
                bool reload = false;
                if (ImGui::BeginMenu("Tools"))
                {
                    reload |= ImGui::MenuItem("Reload shaders", "F5");
                    ImGui::EndMenu();
                }
                reload |= ImGui::IsKeyPressed(ImGuiKey_F5);
                if (reload)
                {
                    vkQueueWaitIdle(m_app->getQueue(0).queue);
                    compileAndCreateGraphicsShaders();
                }
            }

            void onLastHeadlessFrame() override {
                m_app->saveImageToFile(m_gBuffers.getColorImage(eImgTonemapped), m_gBuffers.getSize(),
                    nvutils::getExecutablePath().replace_extension(".jpg").string());
            }

            std::shared_ptr<nvutils::CameraManipulator> getCameraManipulator() const { return m_cameraManip; };

    private:
        nvapp::Application* m_app{};
        nvvk::ResourceAllocator m_allocator{};
        nvvk::StagingUploader   m_stagingUploader{};
        nvvk::SamplerPool       m_samplerPool{};
        nvvk::GBuffer            m_gBuffers{};
        nvslang::SlangCompiler     m_slangCompiler{};

        std::shared_ptr<nvutils::CameraManipulator> m_cameraManip{ std::make_shared<nvutils::CameraManipulator>() };

        nvvk::GraphicsPipelineState m_dynamicPipeline;
        nvvk::DescriptorPack        m_descPack;
        VkPipelineLayout            m_graphicPipelineLayout{};

        VkShaderEXT m_vertexShader{};
        VkShaderEXT m_fragmentShader{};

        nvsamples::GltfSceneResource m_sceneResource{};
        std::vector<nvvk::Image>     m_textures{};

        nvshaders::SkySimple m_skySimple{};
        nvshaders::Tonemapper m_tonemapper{};
        shaderio::TonemapperData m_tonemapperData{};
        glm::vec2 m_metallicRoughnessOverride{ -0.01f, -0.01f };
    };

    int main(int argc, char** argv) {
        nvapp::ApplicationCreateInfo appInfo{};

        nvutils::ParameterParser cli(nvutils::getExecutablePath().stem().string());   //将可执行文件名作为参数传入
        nvutils::ParameterRegistry reg;
        reg.add({ "headless", "Run in headless mode" }, &appInfo.headless, true);   //headless表示是否不要窗口
        cli.add(reg);
        cli.parse(argc, argv);

        VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT };
        nvvk::ContextInitInfo vkSetup{
            .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
            .deviceExtensions =
                {
                    {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME},
                    {VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjectFeatures},
                },
        };
        if (!appInfo.headless)
        {
            nvvk::addSurfaceExtensions(vkSetup.instanceExtensions, &vkSetup.deviceExtensions);
        }

        nvvk::ValidationSettings validationSettings;
        validationSettings.setPreset(nvvk::ValidationSettings::LayerPresets::eStandard);
        vkSetup.instanceCreateInfoExt = validationSettings.buildPNextChain();

#if defined(USE_NSIGHT_AFTERMATH)   //GPU崩溃之后的调试工具
        auto& aftermath = AftermathCrashTracker::getInstance();
        aftermath.initialize();
        aftermath.addExtensions(vkSetup.deviceExtensions);
        nvvk::CheckError::getInstance().setCallbackFunction([&](VkResult result) { aftermath.errorCallback(result); });
#endif

        nvvk::Context vkContext;
        if (vkContext.init(vkSetup) != VK_SUCCESS)
        {
            LOGE("Error in Vulkan context creation\n");
            return 1;
        }

        appInfo.name = "FzbRenderer_nvvk";
        appInfo.instance = vkContext.getInstance();
        appInfo.device = vkContext.getDevice();
        appInfo.physicalDevice = vkContext.getPhysicalDevice();
        appInfo.queues = vkContext.getQueueInfos();

        nvapp::Application application;
        application.init(appInfo);

        auto fzbRenderer_nvvk = std::make_shared<FzbRenderer_nvvk>();
        auto elemCamera = std::make_shared<nvapp::ElementCamera>();
        auto windowTitle = std::make_shared<nvapp::ElementDefaultWindowTitle>();
        auto windowMenu = std::make_shared<nvapp::ElementDefaultMenu>();
        auto camManip = fzbRenderer_nvvk->getCameraManipulator();
        elemCamera->setCameraManipulator(camManip);

        application.addElement(windowMenu);
        application.addElement(windowTitle);
        application.addElement(elemCamera);
        application.addElement(fzbRenderer_nvvk);

        application.run();
        application.deinit();
        vkContext.deinit();

        return 0;
    }
