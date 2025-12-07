#include <filesystem>
#include <fstream>
#include "unique.hpp"
#include "imgui_internal.h"
#include "../../core/GPU/gpu_context.hpp"
#include "io.hpp"
#include "ui.hpp"
#include "Render/RenderPassPool.hpp"
UI::UI() = default;

void UI::init()
{
  auto io = ImGui::GetIO();
  io.ConfigFlags = 0;  // ConfigFlags 초기화
  io.BackendFlags = 0; // BackendFlags 초기화
  io.MouseDrawCursor = false;
  offscreenTagets.albedoTargets.resize(gpu::ctx__->renderingContext.maxInflight__);
  offscreenTagets.depthTargets.resize(gpu::ctx__->renderingContext.maxInflight__);
  offscreenTagets.normalTargets.resize(gpu::ctx__->renderingContext.maxInflight__);
  offscreenTagets.positionTargets.resize(gpu::ctx__->renderingContext.maxInflight__);
  offscreenTagets.specularTargets.resize(gpu::ctx__->renderingContext.maxInflight__);
  offscreenTagets.updated.resize(gpu::ctx__->renderingContext.maxInflight__, false);
};

void UI::update()
{
  rec();
  drawcall();
  ImDrawList* draw_list = ImGui::GetForegroundDrawList();
  ImVec2 mouse_pos = ImGui::GetIO().MousePos;
  ImU32 mouseColor = IM_COL32(255, 255, 255, 255);
  float thickness = 2.0f;
  draw_list->AddCircle(mouse_pos,
                       mns::io__.mouseState__.radius,
                       mouseColor,
                       32,
                       thickness);
}

void UI::rec()
{
  gpu::newFrameApiCall();
  gpu::newFrameSurfaceCall();
  ImGui::NewFrame();
}

void UI::uploadImageToUI()
{
  //pPassBuilder->fragGBufferAlbedoRender
  //for (auto& texture : backgroundTextures_)
  //{
  //  if (texture->waitFrame > 0)
  //  {
  //    texture->waitFrame--;
  //    continue;
  //  }
  //  VkDescriptorSet textureDesc = ImGui_ImplVulkan_AddTexture(texture->sampler,
  //                                                            texture->textureImageView,
  //                                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  //  bindingIndex index = texture->bindigIndex;
  //  UITexture uiTexture;
  //  uiTexture.descriptorSet = textureDesc;
  //  uiTexture.index = index;
  //  uiTextures_.push_back(uiTexture);
  //  backgroundTextures_.pop_back();
  //}
}


void UI::render()
{
  ImGui::Render();
}

void UI::drawcall()
{
  const ImVec2 smallSize = smallUi_ ? ImVec2(160, 80) : ImVec2(440, 220);
  drawFramebufferState();
  drawStateWindow(smallSize);
  drawToolBoxLeft(smallSize);
  drawToolBoxUnder(smallSize);
  drawToolBoxUnderTexture(smallSize);
}


void UI::drawStateWindow(ImVec2 size)
{
  {
    ImVec2 dispSize = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos(ImVec2(dispSize.x / 9.0f, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(dispSize.x - dispSize.x * 2 / 9, (dispSize.y / 9)));
    if (ImGui::Begin("box", nullptr))
    {
      // drawFramebufferState();
      // drawVertexState(size);
      // drawIndexState(size);
      // drawLightState(size);
      // drawTextureState(size);
      // drawCameraState(size);
      // drawMaterialState(size);
      // drawShaderState(size);
    }
    ImGui::End();
  }
}


void UI::drawToolBoxLeft(ImVec2 size)
{
  {
    ImVec2 dispSize = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(dispSize.x / 9.0f, dispSize.y / 6 * 5));
    if (ImGui::Begin("setting Tool Box",
                     nullptr))
    {
      {
        ImGui::Text("Texture Binding");
        if (ImGui::Button("albedo binding"))
        {
          pResourceManager_->selectedModel.constant.albedoTextureIndex = selectedTextureBinding;
        }

        if (ImGui::Button("normal binding: "))
        {
          pResourceManager_->selectedModel.constant.normalTextureIndex = selectedTextureBinding;
        }

        if (ImGui::Button("metallic binding: "))
        {
          pResourceManager_->selectedModel.constant.metalicTextureIndex = selectedTextureBinding;
        }
        ImGui::Separator();
        ImGui::Text("light setting:");
        std::string s = "light";
        ImGui::Separator();
      }
    }
  }

  ImGui::End();
}

void UI::drawToolBoxUnder(ImVec2 size)
{
  ImVec2 dispSize = ImGui::GetIO().DisplaySize;
  ImGui::SetNextWindowPos(ImVec2(0, (dispSize.y / 6) * 5));
  ImGui::SetNextWindowSize(ImVec2(dispSize.x / 80, dispSize.x / 80));
  if (ImGui::Begin("log",
                   nullptr,
                   ImGuiWindowFlags_NoScrollbar))
  {
    ImGui::SetNextWindowPos(ImVec2(dispSize.x / 80,
                                   (dispSize.y / 6) * 5),
                            ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(dispSize.x
                                    - dispSize.x / 80,
                                    (dispSize.y / 6)),
                             ImGuiCond_Always);
    if (ImGui::Begin("system Log:",
                     nullptr))
    {
      ImVec2 minPos(0, (dispSize.y / 6) * 5);
      ImVec2 maxPos(dispSize.x - dispSize.x / 5, (dispSize.y / 6));
      float lineSpacing = ImGui::GetTextLineHeightWithSpacing();
      ImGui::SetNextWindowPos(ImVec2(10, (dispSize.y / 6) * 5));
      ImGui::SetNextWindowSize(ImVec2(dispSize.x, dispSize.y - lineSpacing));
      static bool autoScroll = true;
      for (uint32_t i = 0; i < sink_->buffer_.size(); i++)
      {
        const std::string& line = sink_->buffer_[i];
        ImGui::Text("%s", line.c_str());
        if (autoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
          ImGui::SetScrollHereY(1.0f);
      }
    }
    ImGui::End();
  }
  ImGui::End();
}


void UI::drawToolBoxUnder3(ImVec2 size)
{
  ImVec2 dispSize = ImGui::GetIO().DisplaySize;
  ImGui::SetNextWindowPos(ImVec2(0, (dispSize.y / 6) * 5), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(dispSize.x / 80, dispSize.x / 80), ImGuiCond_Once);
  if (ImGui::Begin("tool3",
                   nullptr,
                   ImGuiWindowFlags_NoScrollbar))
  {
    ImGui::SetNextWindowPos(ImVec2(dispSize.x / 80,
                                   (dispSize.y / 6) * 5),
                            ImGuiCond_Always);

    ImGui::SetNextWindowSize(ImVec2(dispSize.x
                                    - dispSize.x / 80,
                                    (dispSize.y / 6)),
                             ImGuiCond_Always);
    if (ImGui::Begin("system Log:",
                     nullptr))
    {
      ImVec2 minPos(0, (dispSize.y / 6) * 5);
      ImVec2 maxPos(dispSize.x - dispSize.x / 5, (dispSize.y / 6));
      float lineSpacing = ImGui::GetTextLineHeightWithSpacing();
      ImGui::SetNextWindowPos(ImVec2(10, (dispSize.y / 6) * 5));
      ImGui::SetNextWindowSize(ImVec2(dispSize.x, dispSize.y - lineSpacing));
      static bool autoScroll = true;
      for (uint32_t i = 0; i < sink_->buffer_.size(); i++)
      {
        const std::string& line = sink_->buffer_[i];
        ImGui::Text("%s", line.c_str());
        if (autoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
          ImGui::SetScrollHereY(1.0f);
      }
    }
    ImGui::End();
  }
  ImGui::End();
}

void UI::drawToolBoxUnderTexture(ImVec2 size)
{
  ImVec2 dispSize = ImGui::GetIO().DisplaySize;
  ImGui::SetNextWindowPos(ImVec2(0, (dispSize.y / 6) * 5 + dispSize.x / 80), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(dispSize.x / 80, dispSize.x / 80), ImGuiCond_Once);

  if (ImGui::Begin("txt",
                   nullptr,
                   ImGuiWindowFlags_NoScrollbar))
  {
    ImGui::SetNextWindowPos(ImVec2(dispSize.x / 80,
                                   (dispSize.y / 6) * 5),
                            ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(dispSize.x
                                    - dispSize.x / 80,
                                    (dispSize.y / 6)),
                             ImGuiCond_Always);

    if (ImGui::Begin("alocated Textures :",
                     nullptr))
    {
      for (int i = 0; i < uiTextures_.size(); i++)
      {
        std::string id = "text" + std::to_string(i);
        //if (ImGui::ImageButton(id.c_str(),
        //                       (ImTextureID)(intptr_t)uiTextures_[i].descriptorSet,
        //                       ImVec2(64, 64)))
        //{
        //  spdlog::info("alocated textures {}", uiTextures_[i].index);
        //  selectedTextureBinding = uiTextures_[i].index;
        //}
        ImGui::SameLine();
      }
      //  draw_list->AddImage((ImTextureID) (intptr_t) uiTextures_[0].descriptorSet,
      //                      ImVec2(dispSize.x - 180, dispSize.y - 160),
      //                      ImVec2(dispSize.x - 40, dispSize.y - 10));
    }
    ImGui::End();
  }
  ImGui::End();
}




void UI::drawFramebufferState()
{
  {
    ImGui::SetNextWindowBgAlpha(0.35f);
    if (ImGui::Begin("Overlay",
                     nullptr))
    {
      ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
      ImGui::Text("Frame Time: %.2f ms", 1000.0f / ImGui::GetIO().Framerate);
      ImGui::Separator();
    }
    ImGui::End();
  }
}


void UI::setupStyle()
{
  auto& io = ImGui::GetIO();
  io.Fonts->Clear();
  ImGui::StyleColorsDark(); // 다크 테마 기반
  ImGuiStyle& style = ImGui::GetStyle();
  ImVec4* colors = style.Colors;
  colors[ImGuiCol_WindowBg] = ImVec4(0.010f, 0.010f, 0.010f, 1.0f);
  colors[ImGuiCol_ChildBg] = ImVec4(0.010f, 0.010f, 0.010f, 1.0f);
  colors[ImGuiCol_PopupBg] = ImVec4(0.010f, 0.010f, 0.010f, 1.0f);

  colors[ImGuiCol_TitleBg] = ImVec4(0.03f, 0.03f, 0.03f, 1.0f);
  colors[ImGuiCol_TitleBgActive] = ImVec4(0.03f, 0.03f, 0.03f, 1.0f);
  colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.03f, 0.03f, 0.03f, 1.0f);

  colors[ImGuiCol_Button] = ImVec4(0.2f, 0.2f, 0.2f, 1.0f);
  colors[ImGuiCol_ButtonHovered] = ImVec4(0.25f, 0.25f, 0.25f, 1.0f);
  colors[ImGuiCol_ButtonActive] = ImVec4(0.15f, 0.15f, 0.15f, 1.0f);

  colors[ImGuiCol_FrameBg] = ImVec4(0.15f, 0.15f, 0.18f, 1.0f);
  colors[ImGuiCol_FrameBgHovered] = ImVec4(0.25f, 0.45f, 0.75f, 1.0f);
  colors[ImGuiCol_FrameBgActive] = ImVec4(0.20f, 0.40f, 0.70f, 1.0f);

  colors[ImGuiCol_Text] = ImVec4(0.90f, 0.90f, 0.90f, 1.0f);
  colors[ImGuiCol_TextDisabled] = ImVec4(0.40f, 0.40f, 0.40f, 1.0f);

  colors[ImGuiCol_Border] = ImVec4(0.30f, 0.30f, 0.35f, 1.0f);
  colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

  colors[ImGuiCol_ScrollbarBg] = ImVec4(0.10f, 0.10f, 0.12f, 1.0f);
  colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.30f, 0.30f, 0.35f, 1.0f);
  colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.35f, 0.35f, 0.40f, 1.0f);
  colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.40f, 0.40f, 0.45f, 1.0f);

  colors[ImGuiCol_CheckMark] = ImVec4(0.20f, 0.45f, 0.85f, 1.0f);
  colors[ImGuiCol_SliderGrab] = ImVec4(0.20f, 0.45f, 0.85f, 1.0f);
  colors[ImGuiCol_SliderGrabActive] = ImVec4(0.25f, 0.55f, 0.95f, 1.0f);

  colors[ImGuiCol_Tab] = ImVec4(0.15f, 0.15f, 0.18f, 1.0f);
  colors[ImGuiCol_TabHovered] = ImVec4(0.25f, 0.45f, 0.75f, 1.0f);
  colors[ImGuiCol_TabActive] = ImVec4(0.20f, 0.40f, 0.70f, 1.0f);
  colors[ImGuiCol_TabUnfocused] = ImVec4(0.12f, 0.12f, 0.14f, 1.0f);
  colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.15f, 0.15f, 0.18f, 1.0f);

  style.WindowRounding = 1.0f;
  style.FrameRounding = 1.0f;
  style.ScrollbarRounding = 1.0f;
  style.GrabRounding = 1.0f;
  style.PopupRounding = 1.0f;

  style.WindowPadding = ImVec2(15, 15);
  style.FramePadding = ImVec2(10, 6);
  style.ItemSpacing = ImVec2(10, 8);
  style.ItemInnerSpacing = ImVec2(6, 6);
  style.ScrollbarSize = 14.0f;
  style.GrabMinSize = 14.0f;
  style.FontSizeBase = 11;
  io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
}
