//
// Created by dlwog on 25. 10. 23..
//

#include "vk_pass.hpp"

gpu::VkPass::VkPass()
{
  this->dependency__ = {};
  this->dependent__ = {};
  this->linkCount = 0;
}

void gpu::VkPass::clear()
{
  this->dependency__.clear();
  this->dependent__.clear();
  this->linkCount = 0;
  this->execute = nullptr;
  this->read__ = {};
  this->write__ = {};
  this->passParameter__ = {};
}
