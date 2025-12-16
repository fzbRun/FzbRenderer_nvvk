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

#include <nvvk/staging.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/check_error.hpp>
#include <nvutils/timers.hpp>
#include <nvutils/file_operations.hpp>

#include <stb/stb_image.h>

#include "utils.hpp"


namespace nvsamples {

nvvk::Image loadAndCreateImage(VkCommandBuffer cmd, nvvk::StagingUploader& staging, VkDevice device, const std::filesystem::path& filename, bool sRgb)
{
  // Load the image from disk
  int            w, h, comp, req_comp{4};
  std::string    filenameUtf8 = nvutils::utf8FromPath(filename);
  const stbi_uc* data         = stbi_load(filenameUtf8.c_str(), &w, &h, &comp, req_comp);
  assert((data != nullptr) && "Could not load texture image!");

  // Define how to create the image
  VkImageCreateInfo imageInfo = DEFAULT_VkImageCreateInfo;
  imageInfo.format            = sRgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
  imageInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT;
  imageInfo.extent            = {uint32_t(w), uint32_t(h), 1};

  nvvk::ResourceAllocator* allocator = staging.getResourceAllocator();

  // Use the VMA allocator to create the image
  const std::span dataSpan(data, w * h * req_comp);
  nvvk::Image     texture;
  NVVK_CHECK(allocator->createImage(texture, imageInfo, DEFAULT_VkImageViewCreateInfo));
  NVVK_CHECK(staging.appendImage(texture, dataSpan, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));

  return texture;
}

}  // namespace nvsamples

namespace FzbRenderer {
	glm::vec3 getRGBFromString(std::string str) {
		std::vector<float> float3_array(0);
		std::stringstream ss(str);
		std::string token;
		while (std::getline(ss, token, ',')) {
			float3_array.push_back(std::stof(token));
		}
		if (float3_array.size() == 1) return glm::vec3(float3_array[0]);
		return glm::vec3(float3_array[0], float3_array[1], float3_array[2]);
	}
	glm::mat4 getMat4FromString(std::string str) {
		std::vector<float> mat4_array;
		std::stringstream ss(str);
		std::string token;
		while (std::getline(ss, token, ' ')) {
			mat4_array.push_back(std::stof(token));
		}
		return glm::mat4(mat4_array[0], mat4_array[4], mat4_array[8], mat4_array[12],
			mat4_array[1], mat4_array[5], mat4_array[9], mat4_array[13],
			mat4_array[2], mat4_array[6], mat4_array[10], mat4_array[14],
			mat4_array[3], mat4_array[7], mat4_array[11], mat4_array[15]);
	}
	glm::vec2 getfloat2FromString(std::string str) {
		std::vector<float> float2_array;
		std::stringstream ss(str);
		std::string token;
		while (std::getline(ss, token, ' ')) {
			float2_array.push_back(std::stof(token));
		}
		return glm::vec2(float2_array[0], float2_array[1]);
	}
	glm::vec4 getRGBAFromString(std::string str) {
		std::vector<float> float4_array;
		std::stringstream ss(str);
		std::string token;
		while (std::getline(ss, token, ',')) {
			float4_array.push_back(std::stof(token));
		}
		if (float4_array.size() == 3) return glm::vec4(float4_array[0], float4_array[1], float4_array[2], 1.0f);
		return glm::vec4(float4_array[0], float4_array[1], float4_array[2], float4_array[3]);
	}
}