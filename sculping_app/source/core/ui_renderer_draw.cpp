#include <ui_renderer.hpp>
#include <sculptor/brush.hpp>
#include "imgui_internal.h"
#include<fstream>
#include <filesystem>

UIRenderer::UIRenderer(UIRendererCreateInfo info)
  : device_h(info.device_h),
    instance_h(info.instance_h),
    physical_device_h(info.physical_device_h),
    window_h(info.window_h),
    renderpass_h(info.renderpass_h),
    graphics_family(info.graphics_family),
    present_family(info.present_family),
    graphics_q(info.graphics_q)
{
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  ImGui::StyleColorsDark();
  createPool();

  VkPhysicalDeviceProperties properties;
  vkGetPhysicalDeviceProperties(physical_device_h, &properties);
  ImGui_ImplVulkan_InitInfo UIinfo = {};
  UIinfo.Instance                  = instance_h;
  UIinfo.PhysicalDevice            = physical_device_h;
  UIinfo.Device                    = device_h;
  UIinfo.QueueFamily               = graphics_family;
  UIinfo.Queue                     = graphics_q;
  UIinfo.PipelineCache             = VK_NULL_HANDLE;
  UIinfo.DescriptorPool            = imguiPool;
  UIinfo.Allocator                 = nullptr;
  UIinfo.MinImageCount             = 2;
  UIinfo.ImageCount                = 3;
  UIinfo.CheckVkResultFn           = nullptr; // 필요하면 콜백 등록
  UIinfo.RenderPass                = renderpass_h;
  UIinfo.ApiVersion                = properties.apiVersion;
  UIinfo.Subpass                   = 1;
  UIinfo.UseDynamicRendering       = info.useDynamic;

  ImGui_ImplGlfw_InitForVulkan(window_h, true);
  ImGui_ImplVulkan_Init(&UIinfo);
  spdlog::info("start imgui");
  setupStyle();
}

void UIRenderer::createPool()
{
  VkDescriptorPoolSize pool_sizes[] = {
  {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
  {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
  {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
  {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
  {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
  {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
  {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
  {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
  {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
  {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}
  };
  VkDescriptorPoolCreateInfo pool_info = {};
  pool_info.sType                      = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags                      = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets                    = 1000 * IM_ARRAYSIZE(pool_sizes);
  pool_info.poolSizeCount              = (uint32_t) IM_ARRAYSIZE(pool_sizes);
  pool_info.pPoolSizes                 = pool_sizes;
  if (vkCreateDescriptorPool(device_h, &pool_info, nullptr, &imguiPool) != VK_SUCCESS)
  {
    throw std::runtime_error("fail to create Imgui Pool");
  }
}

void UIRenderer::setupStyle()
{
  ImGui::StyleColorsDark(); // 다크 테마 기반
  ImGuiStyle &style         = ImGui::GetStyle();
  ImVec4 *colors            = style.Colors;
  colors[ImGuiCol_WindowBg] = ImVec4(0.010f, 0.010f, 0.010f, 1.0f);
  colors[ImGuiCol_ChildBg]  = ImVec4(0.010f, 0.010f, 0.010f, 1.0f);
  colors[ImGuiCol_PopupBg]  = ImVec4(0.010f, 0.010f, 0.010f, 1.0f);

  colors[ImGuiCol_TitleBg]          = ImVec4(0.03f, 0.03f, 0.03f, 1.0f);
  colors[ImGuiCol_TitleBgActive]    = ImVec4(0.03f, 0.03f, 0.03f, 1.0f);
  colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.03f, 0.03f, 0.03f, 1.0f);

  colors[ImGuiCol_Button]        = ImVec4(0.2f, 0.2f, 0.2f, 1.0f);
  colors[ImGuiCol_ButtonHovered] = ImVec4(0.25f, 0.25f, 0.25f, 1.0f);
  colors[ImGuiCol_ButtonActive]  = ImVec4(0.15f, 0.15f, 0.15f, 1.0f);

  colors[ImGuiCol_FrameBg]        = ImVec4(0.15f, 0.15f, 0.18f, 1.0f);
  colors[ImGuiCol_FrameBgHovered] = ImVec4(0.25f, 0.45f, 0.75f, 1.0f);
  colors[ImGuiCol_FrameBgActive]  = ImVec4(0.20f, 0.40f, 0.70f, 1.0f);

  colors[ImGuiCol_Text]         = ImVec4(0.90f, 0.90f, 0.90f, 1.0f);
  colors[ImGuiCol_TextDisabled] = ImVec4(0.40f, 0.40f, 0.40f, 1.0f);

  colors[ImGuiCol_Border]       = ImVec4(0.30f, 0.30f, 0.35f, 1.0f);
  colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

  colors[ImGuiCol_ScrollbarBg]          = ImVec4(0.10f, 0.10f, 0.12f, 1.0f);
  colors[ImGuiCol_ScrollbarGrab]        = ImVec4(0.30f, 0.30f, 0.35f, 1.0f);
  colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.35f, 0.35f, 0.40f, 1.0f);
  colors[ImGuiCol_ScrollbarGrabActive]  = ImVec4(0.40f, 0.40f, 0.45f, 1.0f);

  colors[ImGuiCol_CheckMark]        = ImVec4(0.20f, 0.45f, 0.85f, 1.0f);
  colors[ImGuiCol_SliderGrab]       = ImVec4(0.20f, 0.45f, 0.85f, 1.0f);
  colors[ImGuiCol_SliderGrabActive] = ImVec4(0.25f, 0.55f, 0.95f, 1.0f);

  colors[ImGuiCol_Tab]                = ImVec4(0.15f, 0.15f, 0.18f, 1.0f);
  colors[ImGuiCol_TabHovered]         = ImVec4(0.25f, 0.45f, 0.75f, 1.0f);
  colors[ImGuiCol_TabActive]          = ImVec4(0.20f, 0.40f, 0.70f, 1.0f);
  colors[ImGuiCol_TabUnfocused]       = ImVec4(0.12f, 0.12f, 0.14f, 1.0f);
  colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.15f, 0.15f, 0.18f, 1.0f);

  style.WindowRounding    = 1.0f;
  style.FrameRounding     = 1.0f;
  style.ScrollbarRounding = 1.0f;
  style.GrabRounding      = 1.0f;
  style.PopupRounding     = 1.0f;

  style.WindowPadding    = ImVec2(15, 15);
  style.FramePadding     = ImVec2(10, 6);
  style.ItemSpacing      = ImVec2(10, 8);
  style.ItemInnerSpacing = ImVec2(6, 6);
  style.ScrollbarSize    = 14.0f;
  style.GrabMinSize      = 14.0f;
  style.FontSizeBase     = 11;
}

void UIRenderer::uploadImageToUI()
{
  auto &backgroundTextures_ = pResourceManager_->uiNeedTextures;
  for (auto &texture: backgroundTextures_)
  {
    if (texture->waitFrame > 0)
    {
      texture->waitFrame--;
      continue;
    }
    VkDescriptorSet textureDesc = ImGui_ImplVulkan_AddTexture(texture->sampler,
                                                              texture->textureImageView,
                                                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    bindingIndex index = texture->bindigIndex;
    UITexture uiTexture;
    uiTexture.descriptorSet = textureDesc;
    uiTexture.index         = index;
    uiTextures_.push_back(uiTexture);
    backgroundTextures_.pop_back();
  }
}

void UIRenderer::rec(VkCommandBuffer command)
{
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
}

void UIRenderer::render(VkCommandBuffer command)
{
  ImGui::Render();
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), command);
}

void UIRenderer::draw(VkCommandBuffer command)
{
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  const ImVec2 smallSize = smallUi_ ? ImVec2(60, 40) : ImVec2(200, 120);
  drawFramebufferState();
  drawStateWindow(smallSize);
  drawMouseState(smallSize);
  drawToolBoxRight(smallSize);
  drawToolBoxLeft(smallSize);

  ImGui::Render();
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), command);
}

void UIRenderer::drawcall(VkCommandBuffer command)
{
  const ImVec2 smallSize = smallUi_ ? ImVec2(160, 80) : ImVec2(440, 220);
  drawFramebufferState();
  drawStateWindow(smallSize);
  drawMouseState(smallSize);
  drawToolBoxLeft(smallSize);
  drawToolBoxRight(smallSize);
  drawToolBoxUnder(smallSize);
  drawToolBoxUnderTexture(smallSize);
}

void UIRenderer::drawTransition(VkCommandBuffer command)
{
  ImVec2 dispSize       = ImGui::GetIO().DisplaySize;
  ImDrawList *draw_list = ImGui::GetForegroundDrawList();
  //if (uiTextures_.size()!= 0)
  //{
  //  draw_list->AddImage((ImTextureID) (intptr_t) uiTextures_[0].descriptorSet,
  //                      ImVec2(dispSize.x - 180, dispSize.y - 160),
  //                      ImVec2(dispSize.x - 40, dispSize.y - 10));
  //}
}

void UIRenderer::drawStateWindow(ImVec2 size)
{
  {
    ImVec2 dispSize = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos(ImVec2(dispSize.x / 9.0f, 0), ImGuiCond_Always);
    ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(dispSize.x - dispSize.x * 2 / 9, (dispSize.y / 9)), ImGuiCond_Once);
    if (ImGui::Begin("box", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove))
    {
      ImGui::BeginChild("Texture folder");
      {
        {
          for (const auto &entry:
               std::filesystem::directory_iterator("/home/ljh/CLionProjects/VkMain/extern/vTuber/TUBASA_014/TUBASA_014.4096"))
          {
            if (entry.is_regular_file())
            {
              if (ImGui::Button(entry.path().filename().c_str()))
              {
                UICall call;
                call.path = entry.path().filename();
                call.type = CallType::Texture;
                callStack_.push_back(call);
                spdlog::info("mesh call {}", call.path);
              }
            }
          }
        }
        ImGui::EndChild();
      }
      ImGui::End();
    }
  }
}

void UIRenderer::drawToolBoxRight(ImVec2 size)
{
  {
    ImVec2 dispSize = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos(ImVec2(dispSize.x - dispSize.x / 9, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(dispSize.x / 9, dispSize.y / 6 * 5), ImGuiCond_Once);
    if (ImGui::Begin("Model Tool Box",
                     nullptr,
                     ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove))
    {
      ImGui::Text("MODEL FOLDER : ");
      ImGui::BeginChild("Model folder");
      {
        for (const auto &entry: std::filesystem::directory_iterator(ASSET_MODELS_DIR))
        {
          if (entry.is_regular_file())
          {
            if (ImGui::Button(entry.path().filename().c_str()))
            {
              UICall call;
              call.path = entry.path().filename();
              call.type = CallType::Mesh;
              callStack_.push_back(call);
              spdlog::info("mesh call {}", call.path);
            }
          }
        }
      }
      ImGui::Separator();
      ImGui::Text("TEXTURE FOLDER : ");
      ImGui::BeginChild("Texture folder");
      {
        {
          for (const auto &entry: std::filesystem::directory_iterator(ASSET_TEXTURES_DIR))
          {
            if (entry.is_regular_file())
            {
              if (ImGui::Button(entry.path().filename().c_str()))
              {
                UICall call;
                call.path = entry.path().filename();
                call.type = CallType::Texture;
                callStack_.push_back(call);
                spdlog::info("mesh call {}", call.path);
              }
            }
          }
          ImGui::EndChild();
        }
      }
      ImGui::EndChild();
    }
  }
  ImGui::End();
}

void UIRenderer::drawToolBoxLeft(ImVec2 size)
{
  {
    ImVec2 dispSize = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(dispSize.x / 9.0f, dispSize.y / 6 * 5), ImGuiCond_Once);
    if (ImGui::Begin("setting Tool Box",
                     nullptr,
                     ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove))
    {
      {
        ImGui::Text("Material Parameter");
        if (ImGui::Button("picked color To materail:"))
        {
          pResourceManager_->selectedModel.constant.color = glm::vec4(color[0], color[1], color[2], color[3]);
        }
        if (ImGui::SliderFloat(" metallic: ", &pResourceManager_->selectedModel.constant.metallic, 0, 1))
        {
          spdlog::info("material pramater: {}", pResourceManager_->selectedModel.constant.metallic);
        }
        ImGui::SliderFloat(" roughness: ", &pResourceManager_->selectedModel.constant.roughness, 0, 1);
        ImGui::SliderFloat(" ao: ", &pResourceManager_->selectedModel.constant.ao, 0, 1);
        ImGui::SliderFloat(" emission: ", &pResourceManager_->selectedModel.constant.emission, 0, 1);
        ImGui::SliderFloat(" N scale : ", &pResourceManager_->selectedModel.constant.normalScale, 0, 1);
        ImGui::SliderFloat(" alpha: ", &pResourceManager_->selectedModel.constant.alphaCutoff, 0, 1);

        ImGui::Separator();
        if (sculpting != nullptr)
        {
          ImGui::Text("Sculpting Brush");
          ImGui::SliderFloat("Strength ", &sculpting->sculptor->brush->strength, -1, 1);
          if (ImGui::Button("Standard:"))
          {
            sculpting->sculptor->brush = &sculpting->sculptor->basicBrush_;
          }
          if (ImGui::Button("Smooth :"))
          {
            sculpting->sculptor->brush = &sculpting->sculptor->smoothBrush_;
          }

          if (ImGui::Button("Grab : "))
          {
            sculpting->sculptor->brush = &sculpting->sculptor->grabBrush_;
          }

          if (ImGui::Button("inflate: "))
          {
            sculpting->sculptor->brush = &sculpting->sculptor->inflateBrush_;
          }
        }
        ImGui::Separator();

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
        ///todo :
        /// material setting
        for (uint32_t i = 0; i < pResourceManager_->lightBuilder.ubo.lightCount; i++)
        {
          s = s + std::to_string(i);

          std::string poss = s + " pos";
          if (ImGui::SliderFloat3(s.c_str(), pos, -100, 100))
          {
            pResourceManager_->lightBuilder.uploadData();
          }
          std::string iten = s + " itensity";
          if (ImGui::SliderFloat(s.c_str(),
                                 &pResourceManager_->lightBuilder.ubo.lights[i].color.w,
                                 -100,
                                 100))
          {
            pResourceManager_->lightBuilder.uploadData();
          }
          s = s + " color update";
          if (ImGui::Button(s.c_str()))
          {
            pResourceManager_->lightBuilder.ubo.lights[i].color = glm::vec4(color[0],
                                                                            color[1],
                                                                            color[2],
                                                                            color[3]);
            pResourceManager_->lightBuilder.uploadData();
          }
        }
        //qif (ImGui::Begin("setting Tool Box",
        //q                 nullptr,
        //q                 ImGuiWindowFlags_NoResize |
        //q                 ImGuiWindowFlags_NoMove))
        //std::vector<cpu::light> light_state = controler.checkLightState();
        //for (int i = 0; i < light_state.size i++)
        //{
        //  ImGui::Text(" light[%d] position:  %f, %f, %f",
        //              i,
        //              light_state[i].position.x,
        //              light_state[i].position.y,
        //              light_state[i].position.z);
        //
        //  ImGui::Text(" light[%d] color:  %f, %f, %f",
        //              i,
        //              light_state[i].color.r,
        //              light_state[i].color.g,
        //              light_state[i].color.b);
        //  if (light_state[i].type == LightType::POINT)
        //  {
        //    ImGui::Text(" light[%d]: light type : POINT", i);
        //  }
        //  if (light_state[i].type == LightType::DIRECTIONAL)
        //  {
        //    ImGui::Text(" light[%d]: light type : DIRENCTION", i);
        //    ImGui::Text(" light[%d] direction:  %f, %f, %f",
        //                i,
        //                light_state[i].direction.x,
        //                light_state[i].direction.y,
        //                light_state[i].direction.z);
        //  }
        //  if (light_state[i].type == LightType::SPOT)
        //  {
        //    ImGui::Text(" light[%d]: light type : SPOT", i);
        //  }
        //  ImGui::Text(" light[%d] range:  %f,", i, light_state[i].range);
        //  ImGui::Text(" light[%d] inner degree:  %f,", i, light_state[i].innerDeg);
        //  ImGui::Text(" light[%d] intensity:  %f,", i, light_state[i].intensity);
        ImGui::Separator();
        ImGui::ColorPicker4("albedo",
                            color,
                            ImGuiColorEditFlags_NoSmallPreview |
                            ImGuiColorEditFlags_NoLabel |
                            ImGuiColorEditFlags_AlphaNoBg |
                            ImGuiColorEditFlags_NoSidePreview |
                            ImGuiColorEditFlags_NoBorder);

        ImGui::Separator();
      }
    }
  }

  ImGui::End();
}

void UIRenderer::drawToolBoxUnder(ImVec2 size)
{
  ImVec2 dispSize = ImGui::GetIO().DisplaySize;
  ImGui::SetNextWindowPos(ImVec2(0, (dispSize.y / 6) * 5), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(dispSize.x / 80, dispSize.x / 80), ImGuiCond_Once);
  if (ImGui::Begin("log",
                   nullptr,
                   ImGuiWindowFlags_NoResize |
                   ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoScrollbar))
  {
    ImGui::SetNextWindowPos(ImVec2(dispSize.x / 80,
                                   (dispSize.y / 6) * 5),
                            ImGuiCond_Always);
    ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(dispSize.x
                                    - dispSize.x / 80,
                                    (dispSize.y / 6)),
                             ImGuiCond_Always);
    if (ImGui::Begin("system Log:",
                     nullptr,
                     ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoResize))
    {
      ImVec2 minPos(0, (dispSize.y / 6) * 5);
      ImVec2 maxPos(dispSize.x - dispSize.x / 5, (dispSize.y / 6));
      float lineSpacing = ImGui::GetTextLineHeightWithSpacing();
      ImGui::SetNextWindowPos(ImVec2(10, (dispSize.y / 6) * 5));
      ImGui::SetNextWindowSize(ImVec2(dispSize.x, dispSize.y - lineSpacing));
      static bool autoScroll = true;
      for (uint32_t i = 0; i < sink_->buffer_.size(); i++)
      {
        const std::string &line = sink_->buffer_[i];
        ImGui::Text("%s", line.c_str());
        if (autoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
          ImGui::SetScrollHereY(1.0f);
      }
    }
    ImGui::End();
  }
  ImGui::End();
}

void UIRenderer::drawToolBoxUnderTexture(ImVec2 size)
{
  ImVec2 dispSize = ImGui::GetIO().DisplaySize;
  ImGui::SetNextWindowPos(ImVec2(0, (dispSize.y / 6) * 5 + dispSize.x / 80), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(dispSize.x / 80, dispSize.x / 80), ImGuiCond_Once);
  if (ImGui::Begin("txt",
                   nullptr,
                   ImGuiWindowFlags_NoResize |
                   ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoScrollbar))
  {
    ImGui::SetNextWindowPos(ImVec2(dispSize.x / 80,
                                   (dispSize.y / 6) * 5),
                            ImGuiCond_Always);
    ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(dispSize.x
                                    - dispSize.x / 80,
                                    (dispSize.y / 6)),
                             ImGuiCond_Always);

    if (ImGui::Begin("alocated Textures :",
                     nullptr,
                     ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoResize))
    {
      for (int i = 0; i < uiTextures_.size(); i++)
      {
        std::string id = "text" + std::to_string(i);
        if (ImGui::ImageButton(id.c_str(),
                               (ImTextureID) (intptr_t) uiTextures_[i].descriptorSet,
                               ImVec2(64, 64)))
        {
          spdlog::info("alocated textures {}", uiTextures_[i].index);
          selectedTextureBinding = uiTextures_[i].index;
        }
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

/// todo : imple this options
void UIRenderer::drawLightUI(ImVec2 size)
{
  ImGui::SetNextWindowPos(ImVec2(20, 300), ImGuiCond_Always);
  ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(size, ImGuiCond_Once);
  if (ImGui::Begin("Light setting", nullptr, ImGuiWindowFlags_NoMove))
  {
    ImGui::Button("light : Position ");
    ImGui::Button("light : Direction");
    ImGui::Button("light : Color ");
  }
  ImGui::End();
}

void UIRenderer::drawCameraUI()
{
  if (ImGui::Begin("Light setting", nullptr, ImGuiWindowFlags_NoMove))
  {
    ImGui::Button("light : Position ");
    ImGui::Button("light : Direction");
    ImGui::Button("light : Color ");
  }
  ImGui::End();
}

void UIRenderer::drawMaterialUI()
{
  if (ImGui::Begin("Light setting", nullptr, ImGuiWindowFlags_NoMove))
  {
    ImGui::Button("light : Position ");
    ImGui::Button("light : Direction");
    ImGui::Button("light : Color ");
  }
  ImGui::End();
}

void UIRenderer::drawShaderUI()
{
  if (ImGui::Begin("Light setting", nullptr, ImGuiWindowFlags_NoMove))
  {
    ImGui::Button("light : Position ");
    ImGui::Button("light : Direction");
    ImGui::Button("light : Color ");
  }
  ImGui::End();
}

void UIRenderer::setResourceManager(ResourceManager *resourceManager)
{
  pResourceManager_ = resourceManager;
}



/// todo:
///  vertex and index :
///  check height and index size current state draw call is need

void UIRenderer::drawVertexState(ImVec2 size)
{
  {
    ImGui::SetNextWindowPos(ImVec2(20, 55), ImGuiCond_Always);
    ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(size);
  }
}

void UIRenderer::drawIndexState(ImVec2 size)
{
  {
    ImGui::SetNextWindowPos(ImVec2(20, 80), ImGuiCond_Always);
    ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(size);
    if (ImGui::Begin("Index State:", nullptr, ImGuiWindowFlags_NoMove))
    {
      ImGui::BeginChild("ScrollRegion", ImVec2(0, 300), true); // 높이 300
      ImGui::Text("Indices:");
      //auto i_state = controler.checkIndexState();
      //for (int i = 0; i < i_state.size i++)
      //{
      //  ImGui::Text(" [%d] : %d", i, i_state[i]);
      //}
      ImGui::EndChild();
    }
    ImGui::End();
  }
}



void UIRenderer::drawTextureState(ImVec2 size)
{
  {
    ImGui::SetNextWindowPos(ImVec2(20, 130), ImGuiCond_Always);
    ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(size);
    if (ImGui::Begin("texture State:", nullptr, ImGuiWindowFlags_NoMove))
    {
      ImGui::BeginChild("ScrollRegion", ImVec2(0, 300), true); // 높이 300
      ImGui::Text("light :");
      //auto i_state = controler.checkLightState();
      // for (int i = 0; i < i_state.size i++)
      // {
      //   ImGui::Text(" [] light state :");
      // }
      ImGui::EndChild();
    }
    ImGui::End();
  }
}

void UIRenderer::drawMaterialState(ImVec2 size)
{
  {
    ImGui::SetNextWindowPos(ImVec2(20, 155), ImGuiCond_Always);
    ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(size);
    if (ImGui::Begin("Material State:", nullptr, ImGuiWindowFlags_NoMove))
    {
      ImGui::BeginChild("ScrollRegion", ImVec2(0, 300), true); // 높이 300
      ImGui::Text("Material:: need to set up material structure");
      ///todo :
      /// set upm maaterial -> hot load text
      ImGui::EndChild();
    }
    ImGui::End();
  }
}

void UIRenderer::drawCameraState(ImVec2 size)
{
  {
    ImGui::SetNextWindowPos(ImVec2(20, 180), ImGuiCond_Always);
    ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(size);
    if (ImGui::Begin("Camera State:", nullptr, ImGuiWindowFlags_NoMove))
    {
      //auto temp = controler.checkCameraState();
      ImGui::BeginChild("ScrollRegion", ImVec2(0, 300), true); // 높이 300
      ImGui::Text("camera state :");
      //todo:
      //ImGui::Text("current cam: %s :", controler.camera.pos_);
      //write current state setting value
      // for (int i = 0; i < temp.size i++
      // )
      // {
      //   ImGui::Text(" [%d] camera name : %s", i, temp[i].cfg.name.c_str());
      // }
      ImGui::EndChild();
    }
    ImGui::End();
  }
}


void UIRenderer::drawMouseState(ImVec2 size)
{
  if (sculpting != nullptr)
  {
    ImDrawList *draw_list = ImGui::GetForegroundDrawList();
    ImVec2 mouse_pos      = ImGui::GetIO().MousePos;
    float radius          = sculpting->sculptor->brush->radius;
    ImU32 mouseColor      = IM_COL32(255, 0, 0, 255);
    float thickness       = 2.0f;
    draw_list->AddCircle(mouse_pos, radius, mouseColor, 32, thickness);
    if (sculpting->symmetry_)
    {
      // show symmetry
      ImVec2 point = ImGui::GetIO().DisplaySize;
      point.x -= mouse_pos.x;
      point.y = mouse_pos.y;
      draw_list->AddCircle(point, radius, mouseColor, 32, thickness);
    }
  }
}

void UIRenderer::drawFramebufferState()
{
  {
    ImGui::SetNextWindowBgAlpha(0.35f);
    if (ImGui::Begin("Overlay",
                     nullptr,
                     ImGuiWindowFlags_NoDecoration |
                     ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoFocusOnAppearing |
                     ImGuiWindowFlags_NoNav))
    {
      ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
      ImGui::Text("Frame Time: %.2f ms", 1000.0f / ImGui::GetIO().Framerate);
      ImGui::Separator();
    }
    ImGui::End();
  }
}

//void UIRenderer::drawFrame(ImVec2 size)
//{
//  int numWindows = 6; // 몇 개의 UI 창을 반복할지
//  float spacingX = size.x ; // 창 사이 가로 간격
//
//  for (int i = 1; i < numWindows; i++)
//  {
//    // 창 위치: 오른쪽으로 간격만큼 이동
//    ImGui::SetNextWindowPos(ImVec2(spacingX * i, 0), ImGuiCond_Always);
//    ImGui::SetNextWindowSize(size, ImGuiCond_Always);
//    ImGui::SetNextWindowCollapsed(false, ImGuiCond_Always);
//
//    std::string windowName = "Frame " + std::to_string(i);
//    if (ImGui::Begin(windowName.c_str(), nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize))
//    {
//      // 스크롤 영역
//      ImGui::BeginChild(("ScrollRegion" + std::to_string(i)).c_str(),
//                        ImVec2(0, 300), true, ImGuiWindowFlags_HorizontalScrollbar);
//
//      ImGui::Text("UI Content for window %d", i);
//      ImGui::Separator();
//
//      // 예시: 칸 2개로 나누기
//      ImGui::Columns(2, nullptr, false);
//      for (int j = 0; j < 4; j++)
//      {
//        ImGui::Text("Item %d.%d", i, j);
//        ImGui::NextColumn();
//      }
//      ImGui::Columns(1); // 칸 초기화
//
//      ImGui::EndChild();
//    }
//    ImGui::End();
//  }
//}