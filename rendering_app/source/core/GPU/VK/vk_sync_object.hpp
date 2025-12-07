#ifndef MYPROJECT_RENDER_GRAPH_SYNC_HPP
#define MYPROJECT_RENDER_GRAPH_SYNC_HPP

#include <cstdint>
#include <vector>
#include "vulkan/vulkan.h"

namespace gpu
{
  class VkContext;

  struct VkSemaphorePool{
  public:
    VkSemaphorePool(VkContext* pCtxt);
    ~VkSemaphorePool();
    void recreate();
    VkContext* pCtxt;
    uint32_t MAX_FRAMES_IN_FLIGHT__;
    std::vector<VkSemaphore> semaphores__;
  };

  struct VkFencePool{
    VkFencePool(VkContext* pCtxt, bool signaled = true);

    ~VkFencePool();

    void recreate();

    VkContext* pCtxt;
    uint32_t MAX_FRAMES_IN_FLIGHT__;
    std::vector<VkFence> fences;
  };

  struct Query{
    VkQueryPool queryPool = VK_NULL_HANDLE;
    uint32_t queryCount   = 0;
    // query type: timestamp, occlusion, pipeline statistics
    VkQueryType type;

    // 생성
    void create(VkDevice device, VkQueryType queryType, uint32_t count)
    {
      type       = queryType;
      queryCount = count;

      VkQueryPoolCreateInfo info{};
      info.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
      info.queryType  = queryType;
      info.queryCount = count;

      vkCreateQueryPool(device, &info, nullptr, &queryPool);
      //vkCmdBeginQuery()
    }

    void destroy(VkDevice device)
    {
      if (queryPool != VK_NULL_HANDLE)
      {
        vkDestroyQueryPool(device, queryPool, nullptr);
        queryPool = VK_NULL_HANDLE;
      }
    }

    void
    getResults(VkDevice device, uint32_t firstQuery, uint32_t queryCount, void* data, VkDeviceSize stride,
               VkQueryResultFlags flags)
    {
      vkGetQueryPoolResults(device, queryPool, firstQuery, queryCount, stride, data, stride, flags);
    }
  };
}
// VkQueryPool createTimestampQueryPool(VkDevice device, uint32_t queryCount = 2) {
//   VkQueryPoolCreateInfo queryPoolInfo{};
//   queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
//   queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
//   queryPoolInfo.queryCount = queryCount;
//
//   VkQueryPool queryPool;
//   if (vkCreateQueryPool(device, &queryPoolInfo, nullptr, &queryPool) != VK_SUCCESS) {
//     throw std::runtime_error("failed to create query pool!");
//   }
//   return queryPool;
// }
//
// // Command Buffer에서 Timestamp 기록
// void recordTimestamp(VkCommandBuffer cmd, VkQueryPool queryPool) {
//   // 쿼리 리셋
//   vkCmdResetQueryPool(cmd, queryPool, 0, 2);
//
//   // GPU 파이프라인 시작 시점
//   vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
//
//   // === 여기에 실제 draw / dispatch 호출 들어감 ===
//   // vkCmdDraw(...);
//   // vkCmdDispatch(...);
//
//   // GPU 파이프라인 끝 시점
//   vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
// }
//
// // 결과 읽기
// double getGpuTime(VkDevice device, VkQueryPool queryPool, VkPhysicalDevice physicalDevice) {
//   uint64_t timestamps[2] = {};
//
//   VkResult res = vkGetQueryPoolResults(
//       device,
//       queryPool,
//       0, 2,                           // 0번~1번까지 읽음
//       sizeof(timestamps),
//       timestamps,
//       sizeof(uint64_t),
//       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
//   );
//
//   if (res != VK_SUCCESS) {
//     throw std::runtime_error("failed to get query results!");
//   }
//
//   // GPU 타임스탬프 주기(나노초 단위 환산용)
//   VkPhysicalDeviceProperties props{};
//   vkGetPhysicalDeviceProperties(physicalDevice, &props);
//
//   double time_ns = double(timestamps[1] - timestamps[0]) * props.limits.timestampPeriod;
//   return time_ns / 1e6; // ms 단위로 변환
// }

// vkCmdResetQueryPool(cmd, queryPool, 0, 2);
// vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
// // ... draw calls ...
// vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
//
// // CPU 쪽에서 결과 읽기
// uint64_t timestamps[2];
// vkGetQueryPoolResults(..., timestamps, ...);
// double time_ns = (timestamps[1] - timestamps[0]) * props.limits.timestampPeriod;

#endif //MYPROJECT_RENDER_GRAPH_SYNC_HPP
