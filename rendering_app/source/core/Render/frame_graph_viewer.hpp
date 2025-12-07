#ifndef MYPROJECT_FRAME_VIEWER_HPP
#define MYPROJECT_FRAME_VIEWER_HPP
#include <vector>
#include <cmath>
#include <algorithm>
#include "imnodes.h"
#include "context.hpp"
#include "frame_graph.hpp"

namespace dag
{
  // Force-directed layout parameters
  const float REPULSION = 5000.0f;
  const float ATTRACTION = 0.1f;
  const float DAMPING = 0.85f;


  struct Link
  {
    int id;
    int start_attr_id;
    int end_attr_id;
  };

  class FrameGraphViewer
  {
    public:
    FrameGraphViewer();
    void init();
    void computeSort();
    void capture(FrameGraph* frame);
    void show();

    std::unordered_map<gpu::VkResource*, RenderNode> resources{};
    std::vector<RenderNode> nodes;
    std::vector<Link> links;
    int next_node_id = 1;
    int next_link_id = 1;
    dag::FrameGraph* graph;
    ImNodesContext* ctx;
    bool pOpen = false;
    bool posReset = false;
  };
}


#endif //MYPROJECT_FRAME_VIEWER_HPP
