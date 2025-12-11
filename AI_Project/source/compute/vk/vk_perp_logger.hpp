//
// Created by dlwog on 25. 12. 7..
//

#ifndef MYPROJECT_VK_PERP_LOGGER_HPP
#define MYPROJECT_VK_PERP_LOGGER_HPP
class vk_perf_logger
{
public:
  void print_timings()
  {
    if (timings.empty())
    {
      return;
    }
    uint64_t total_all_op_times = 0;
    std::cerr << "----------------\nVulkan Timings:" << std::endl;
    for (const auto& t : timings)
    {
      uint64_t total_op_times = 0;
      for (const auto& time : t.second)
      {
        total_op_times += time;
      }
      std::cerr << t.first << ": " << t.second.size() << " x " << (total_op_times / t.second.size() / 1000.0)
        << " us";

      // If we have as many flops entries as timing entries for the op, then compute and log the flops/S.
      auto it = flops.find(t.first);
      if (it != flops.end() && (it->second).size() == t.second.size())
      {
        uint64_t total_op_flops = 0;
        for (const auto& elem : it->second)
        {
          total_op_flops += elem;
        }
        std::cerr << " ("
          << (double(total_op_flops) / (1000.0 * 1000.0 * 1000.0)) /
          (double(total_op_times) / (1000.0 * 1000.0 * 1000.0))
          << " GFLOPS/s)";
      }

      total_all_op_times += total_op_times;

      std::cerr << std::endl;
    }

    if (timings.size() > 0)
    {
      std::cerr << "Total time: " << total_all_op_times / 1000.0 << " us." << std::endl;
    }

    timings.clear();
    flops.clear();
  }

  void log_timing(const v_tensor* node, uint64_t time)
  {
    if (node->op == v_OP_UNARY)
    {
      timings[v_unary_op_name(v_get_unary_op(node))].push_back(time);
      return;
    }
    if (node->op == V_OP_MUL_MAT || node->op == v_OP_MUL_MAT_ID)
    {
      const uint64_t m = node->src[0]->ne[1];
      const uint64_t n = node->ne[1];
      const uint64_t k = node->src[1]->ne[0];
      const uint64_t batch = node->src[1]->ne[2] * node->src[1]->ne[3];
      std::string name = v_op_name(node->op);
      if ((node->op == V_OP_MUL_MAT && n <= mul_mat_vec_max_cols) ||
          (node->op == v_OP_MUL_MAT_ID && node->src[2]->ne[1] == 1))
      {
        name += "_VEC";
      }
      name += " ";
      name += v_type_name(node->src[0]->type);
      name += " m=" + std::to_string(m) + " n=" + std::to_string(n) + " k=" + std::to_string(k);
      if (batch > 1)
      {
        name += " batch=" + std::to_string(batch);
      }
      timings[name].push_back(time);
      flops[name].push_back(m * n * (k + (k - 1)) * batch);
      return;
    }
    if (node->op == v_OP_CONV_2D || node->op == v_OP_CONV_TRANSPOSE_2D)
    {
      std::string name = v_op_name(node->op);
      v_tensor* knl = node->src[0];
      uint64_t OW = node->ne[0];
      uint64_t OH = node->ne[1];
      uint64_t N = node->ne[3];
      uint64_t Cout = node->ne[2];
      uint64_t KW = knl->ne[0];
      uint64_t KH = knl->ne[1];
      uint64_t Cin = node->src[1]->ne[2];
      // KxCRS @ CRSxNPQ = KxNPQ -> M=K, K=CRS, N=NPQ
      uint64_t size_M = Cout;
      uint64_t size_K = Cin * KW * KH;
      uint64_t size_N = N * OW * OH;
      uint64_t n_flops = size_M * size_N * (size_K + (size_K - 1));
      name += " M=Cout=" + std::to_string(size_M) + ", K=Cin*KW*KH=" + std::to_string(size_K) +
        ", N=N*OW*OH=" + std::to_string(size_N);
      flops[name].push_back(n_flops);
      timings[name].push_back(time);
      return;
    }
    if (node->op == v_OP_RMS_NORM)
    {
      std::string name = v_op_name(node->op);
      name += "(" + std::to_string(node->ne[0]) + "," + std::to_string(node->ne[1]) + "," + std::to_string(node->ne[2])
        + "," + std::to_string(node->ne[3]) + ")";
      timings[name].push_back(time);
      return;
    }
    timings[v_op_name(node->op)].push_back(time);
  }

private:
  std::map<std::string, std::vector<uint64_t>> timings;
  std::map<std::string, std::vector<uint64_t>> flops;
};

#endif //MYPROJECT_VK_PERP_LOGGER_HPP