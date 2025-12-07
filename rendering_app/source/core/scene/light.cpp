#include "light.hpp"
#include "context.hpp"

LightBuilder::LightBuilder()
{
  ubo.lightCount = 0;
}

void LightBuilder::build(const Light& light)
{
  if (ubo.lightCount >= MAX_LIGHTS) return;
  GPULight& gpu = ubo.lights[ubo.lightCount];
  ubo.lightCount += 1;
  gpu.position = glm::vec4(light.transform.position, static_cast<float>(light.type));
  glm::vec3 dir = glm::normalize(light.transform.rotation * glm::vec3(0, 0, -1));
  gpu.direction = glm::vec4(dir, light.angle);
  gpu.color = glm::vec4(light.color, light.intensity);
  gpu.view = glm::lookAt(light.transform.position,
                         dir,
                         glm::vec3(0, 1, 0));
  gpu.proj = glm::ortho(
                        0.0f,
                        static_cast<float>(gpu::ctx__->pSwapChainContext->extent__.width),
                        0.0f, // bottom
                        static_cast<float>(gpu::ctx__->pSwapChainContext->extent__.height),
                        -1.0f, // near
                        1.0f   // far
                       );
}

void LightBuilder::uploadData()
{
  buffer.data_ = &ubo;
  buffer.size_ = sizeof(lightUBO);
  buffer.uploadData();
}

void LightBuilder::drawUI()
{
  if (ImGui::Button("light window")) this->uiState = !uiState;
  if (uiState)
  {
    ImGui::Begin("light state:", &this->uiState);
    if (ImGui::Button("build"))
    {
      Light light;
      this->build(light);
    };
    ImGui::Separator();
    ImGui::Text("light state :");
    for (uint32_t i = 0; i < ubo.lightCount; i++)
    {
      ImGui::SliderFloat("   pos x:"+ i, &this->ubo.lights[i].position.x, -10.0f, 10.0f);
      ImGui::SliderFloat("   pos y:"+ i, &this->ubo.lights[i].position.y, -10.0f, 10.0f);
      ImGui::SliderFloat("   pos z:"+ i, &this->ubo.lights[i].position.z, -10.0f, 10.0f);

      ImGui::Separator();
      ImGui::SliderFloat("    dir x:"+ i, &this->ubo.lights[i].direction.x, -10.0f, 10.0f);
      ImGui::SliderFloat("    dir y:"+ i, &this->ubo.lights[i].direction.y, -10.0f, 10.0f);
      ImGui::SliderFloat("    dir z:"+ i, &this->ubo.lights[i].direction.z, -10.0f, 10.0f);
      ImGui::Separator();
      ImGui::ColorPicker4("    albedo" + i,
                    reinterpret_cast<float*>(&ubo.lights[i].color),
                    ImGuiColorEditFlags_NoSmallPreview |
                    ImGuiColorEditFlags_NoLabel |
                    ImGuiColorEditFlags_AlphaNoBg |
                    ImGuiColorEditFlags_NoSidePreview |
                    ImGuiColorEditFlags_NoBorder);
      ImGui::Separator();
    }
    ImGui::End();
  }
}
