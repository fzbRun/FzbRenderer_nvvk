/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <nvutils/file_operations.hpp>
#include <vulkan/vulkan_core.h>
#include <glm/glm.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/staging.hpp>

namespace nvsamples {

inline static VkShaderModuleCreateInfo getShaderModuleCreateInfo(const std::span<const uint32_t>& spirv)
{
  return VkShaderModuleCreateInfo{
      .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spirv.size_bytes(),
      .pCode    = spirv.data(),
  };
}

nvvk::Image loadAndCreateImage(VkCommandBuffer              cmd,
                               nvvk::StagingUploader&       staging,
                               VkDevice                     device,
                               const std::filesystem::path& filename,
                               bool                         sRgb = true);

}  // namespace nvsamples

namespace FzbRenderer {
    glm::vec3 getRGBFromString(std::string str);
    glm::mat4 getMat4FromString(std::string str);
    glm::vec2 getfloat2FromString(std::string str);
    glm::vec4 getRGBAFromString(std::string str);

    void printfMat4(glm::mat4* matrix);

    nvvk::Buffer createStagingBuffer(size_t bufferSize, size_t dataSize, const void* data);
}