#include "frame_graph_viewer.hpp"
#include "context.hpp"
#include "RenderPassPool.hpp"
#include "unordered_map"

namespace dag
{
  FrameGraphViewer::FrameGraphViewer()
  {
  }

  void FrameGraphViewer::init()
  {
    ctx = ImNodes::CreateContext();
    ImNodes::SetCurrentContext(ctx);
    ImNodes::StyleColorsDark();
    posReset = false;
  }

  void FrameGraphViewer::capture(FrameGraph* frame)
  {
    this->nodes.clear();
    this->links.clear();
    this->next_node_id = 1;
    this->next_link_id = 1;
    float distance = 400;
    uint32_t lev = 0;
    for (auto pass : frame->uploadPasses_)
    {
      RenderNode passNode;
      passNode.id = next_node_id++;
      passNode.name = pass->name.c_str();
      passNode.pos = ImVec2(distance * lev, 200);
      passNode.size = ImVec2(10, 10);
      passNode.posSeted = false;

      for (uint32_t i = 0; i < pass->read__.size(); i++)
      {
        passNode.input_attr_id.push_back(next_link_id++);
        if (resources.find(pass->read__[i]) == resources.end())
        {
          RenderNode readNode;
          readNode.name = "Frame Image";
          readNode.id = next_node_id++;
          readNode.output_attr_id.push_back(next_link_id++);
          readNode.size = ImVec2(40, 40);
          readNode.pos = ImVec2(distance * lev - 150, i * 120);
          readNode.posSeted = false;
          resources[pass->read__[i]] = readNode;
          readNode.bindings = pass->read__[i]->descriptorBindingId;
          readNode.frame = true;

          Link link;
          link.id = next_link_id++;
          link.end_attr_id = passNode.input_attr_id.back();
          link.start_attr_id = readNode.output_attr_id.back();
          links.push_back(link);
        }
        else
        {
          RenderNode& readNode = resources[pass->read__[i]];
          readNode.output_attr_id.push_back(next_link_id++);

          Link link;
          link.id = next_link_id++;
          link.end_attr_id = passNode.input_attr_id.back();
          link.start_attr_id = readNode.output_attr_id.back();
          links.push_back(link);
        }
      }
      for (uint32_t i = 0; i < pass->write__.size(); i++)
      {
        passNode.output_attr_id.push_back(next_link_id++);
        if (resources.find(pass->write__[i]) == resources.end())
        {
          RenderNode writeNode;
          writeNode.id = next_node_id++;
          writeNode.name = "Frame Image";
          writeNode.input_attr_id.push_back(next_link_id++);
          writeNode.size = ImVec2(40, 40);
          writeNode.pos = ImVec2(distance * lev + 150, i * 120);
          writeNode.posSeted = false;
          writeNode.bindings = pass->write__[i]->descriptorBindingId;
          writeNode.frame = true;
          resources[pass->write__[i]] = writeNode;

          Link link;
          link.id = next_link_id++;
          link.end_attr_id = passNode.output_attr_id.back();
          link.start_attr_id = writeNode.input_attr_id.back();
          links.push_back(link);
        }
        else
        {
          RenderNode& writeNode = resources[pass->write__[i]];
          writeNode.input_attr_id.push_back(next_link_id++);
          Link link;
          link.id = next_link_id++;
          link.end_attr_id = passNode.output_attr_id.back();
          link.start_attr_id = writeNode.input_attr_id.back();
          links.push_back(link);
        }
      }
      nodes.push_back(passNode);
      lev++;
    }
    for (auto& rec : resources)
    {
      nodes.push_back(rec.second);
    }
  }

  void FrameGraphViewer::show()
  {
    if (ImGui::Begin("frame viewer", &pOpen))
    {
      ImNodes::BeginNodeEditor();
      ImNodes::SetCurrentContext(ctx);
      ImNodes::MiniMap();
      uint32_t lev = 0;

      for (auto& node : nodes)
      {
        ImNodes::BeginNode(node.id);
        if (!node.posSeted)
        {
          ImNodes::SetNodeGridSpacePos(node.id, ImVec2(node.pos.x,
                                                       node.pos.y));
          node.posSeted = true;
        }

        ImNodes::BeginNodeTitleBar();
        ImGui::Text("%s %d", node.name, node.id);
        ImNodes::EndNodeTitleBar();
        for (auto input : node.input_attr_id)
        {
          ImNodes::BeginInputAttribute(input);
          ImGui::Text("input");
          ImNodes::EndInputAttribute();
        }
        for (auto output : node.output_attr_id)
        {
          ImNodes::BeginOutputAttribute(output);
          ImGui::Text("       output");
          ImNodes::EndOutputAttribute();
        }
        ImNodes::EndNode();
      }
      for (auto& link : links)
      {
        ImNodes::Link(link.id,
                      link.start_attr_id,
                      link.end_attr_id
                     );
      }
      ImNodes::EndNodeEditor();
    }
    ImGui::End();
  }
}
