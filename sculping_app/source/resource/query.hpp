//
// Created by ljh on 25. 9. 19..
//

#ifndef MYPROJECT_QUERY_HPP
#define MYPROJECT_QUERY_HPP
#include <vulkan/vulkan.h>
#include <stdexcept>

VkQueryPool createTimestampQueryPool(VkDevice device, uint32_t queryCount = 2) {
  VkQueryPoolCreateInfo queryPoolInfo{};
  queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
  queryPoolInfo.queryCount = queryCount;

  VkQueryPool queryPool;
  if (vkCreateQueryPool(device, &queryPoolInfo, nullptr, &queryPool) != VK_SUCCESS) {
    throw std::runtime_error("failed to create query pool!");
  }
  return queryPool;
}

// Command Buffer에서 Timestamp 기록
void recordTimestamp(VkCommandBuffer cmd, VkQueryPool queryPool) {
  // 쿼리 리셋
  vkCmdResetQueryPool(cmd, queryPool, 0, 2);

  // GPU 파이프라인 시작 시점
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);

  // === 여기에 실제 draw / dispatch 호출 들어감 ===
  // vkCmdDraw(...);
  // vkCmdDispatch(...);

  // GPU 파이프라인 끝 시점
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
}

// 결과 읽기
double getGpuTime(VkDevice device, VkQueryPool queryPool, VkPhysicalDevice physicalDevice) {
  uint64_t timestamps[2] = {};

  VkResult res = vkGetQueryPoolResults(
      device,
      queryPool,
      0, 2,                           // 0번~1번까지 읽음
      sizeof(timestamps),
      timestamps,
      sizeof(uint64_t),
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
  );

  if (res != VK_SUCCESS) {
    throw std::runtime_error("failed to get query results!");
  }

  // GPU 타임스탬프 주기(나노초 단위 환산용)
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(physicalDevice, &props);

  double time_ns = double(timestamps[1] - timestamps[0]) * props.limits.timestampPeriod;
  return time_ns / 1e6; // ms 단위로 변환
}

// vkCmdResetQueryPool(cmd, queryPool, 0, 2);
// vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
// // ... draw calls ...
// vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
//
// // CPU 쪽에서 결과 읽기
// uint64_t timestamps[2];
// vkGetQueryPoolResults(..., timestamps, ...);
// double time_ns = (timestamps[1] - timestamps[0]) * props.limits.timestampPeriod;


#endif //MYPROJECT_QUERY_HPP