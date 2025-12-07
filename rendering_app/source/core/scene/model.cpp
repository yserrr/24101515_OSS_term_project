#include "model.hpp"

void Model::drawUIState()
{
  if (ImGui::Button(name.c_str())) this->uiState = !this->uiState;
  if (uiState)
  {
    ImGui::Begin(name.c_str(), &this->uiState);
    ImGui::Text("Model transform Parameter");

    if (ImGui::SliderFloat((name + std::string("point x:")).c_str(), &transform.position.x, -10, 10))
    {
      this->transform.update();
      this->constant.modelMatrix = this->transform.matrix;
    }
    if (ImGui::SliderFloat((name + std::string("point y:")).c_str(), &transform.position.y, -10, 10))
    {
      this->transform.update();
      this->constant.modelMatrix = this->transform.matrix;
    }
    if (ImGui::SliderFloat((name + std::string("point z:")).c_str(), &transform.position.z, -10, 10))
    {
      this->transform.update();
      this->constant.modelMatrix = this->transform.matrix;
    }
    ImGui::Separator();
    if (ImGui::SliderFloat((name + std::string("rotate x")).c_str(), &rotateX, -1, 1))
    {
      glm::vec3 pivot = this->transform.position;
      glm::mat4 model = this->constant.modelMatrix;
      model = glm::translate(model, pivot);
      this->transform.matrix = model;
      this->transform.rotate(rotateX, 0);
      this->transform.update();
      this->transform.matrix = glm::translate(transform.matrix, pivot);
      this->constant.modelMatrix = this->transform.matrix;
    }
    if (ImGui::SliderFloat((name + std::string("rotate Y")).c_str(), &rotateY, -1, 1))
    {
      glm::vec3 pivot = this->transform.position;
      glm::mat4 model = this->constant.modelMatrix;
      model = glm::translate(model, pivot);
      this->transform.matrix = model;
      this->transform.rotate(0, rotateY);
      this->transform.update();
      this->transform.matrix = glm::translate(transform.matrix, pivot);
      this->constant.modelMatrix = this->transform.matrix;
    }
    //if (ImGui::SliderFloat((name + std::string("rotate x")).c_str(), &rotateX, -1, 1))
    //{
    //  this->transform.rotate( rotateX ,0 );
    //}
    ImGui::Separator();

    ImGui::SliderFloat((name + std::string("scale x:")).c_str(), &transform.scale.x, -10, 10);
    ImGui::SliderFloat((name + std::string("scale y:")).c_str(), &transform.scale.y, -10, 10);
    ImGui::SliderFloat((name + std::string("scale z:")).c_str(), &transform.scale.z, -10, 10);
    ImGui::Separator();
    ImGui::Text("Model Material Parameter");
    //ImGui::SliderFloat((name + std::string("metallic: ")).c_str(), &material->metallic, 0, 1);
    //ImGui::SliderFloat((name + std::string("roughness: ")).c_str(), &material->metallic, 0, 1);
    //ImGui::Separator();
    //ImGui::SliderFloat((name + std::string("color x:")).c_str(), &material->baseColor.x, 0, 1);
    //ImGui::SliderFloat((name + std::string("color y:")).c_str(), &material->baseColor.y, 0, 1);
    //ImGui::SliderFloat((name + std::string("color z:")).c_str(), &material->baseColor.z, 0, 1);
    //ImGui::SliderFloat((name + std::string("color a:")).c_str(), &material->baseColor.a, 0, 1);
    //ImGui::Separator();
    //ImGui::SliderFloat((name + std::string("ao")).c_str(), &material->ao, 0, 1);
    //ImGui::SliderFloat((name + std::string("emission")).c_str(), &material->emission, 0, 1);
    //ImGui::SliderFloat((name + std::string("alpha cut")).c_str(), &material->alphaCutoff, 0, 1);
    //ImGui::SliderFloat((name + std::string("normal scale")).c_str(), &material->normalScale, 0, 1);

    ImGui::End();
  }
}
