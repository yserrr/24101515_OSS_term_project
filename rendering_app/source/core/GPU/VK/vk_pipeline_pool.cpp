#include "vk_pipeline_pool.hpp"
#include "vk_context.hpp"
#include "vk_vertex_attr.hpp"

namespace gpu
{
  VkPipelinePool::VkPipelinePool(VkContext* pCtxt) :
    pCtxt_(pCtxt)

  {
    device_ = pCtxt->deviceh__;
    std::vector<uint8_t> oldCacheData = loadPipelineCache("check this");

    VkPipelineCacheCreateInfo cacheInfo{};
    cacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cacheInfo.initialDataSize = oldCacheData.size();
    cacheInfo.pInitialData = oldCacheData.empty() ? nullptr : oldCacheData.data();
    vkCreatePipelineCache(device_, &cacheInfo, nullptr, &oldPipelineCache_);
  }

  VkPipelinePool::~VkPipelinePool()
  {
    size_t dataSize = 0;
    vkGetPipelineCacheData(device_, oldPipelineCache_, &dataSize, nullptr);
    std::vector<uint8_t> cacheData(dataSize);
    vkGetPipelineCacheData(device_, oldPipelineCache_, &dataSize, cacheData.data());
    std::ofstream outFile(std::string(PIPELINE_CACHE_DIR),
                          std::ios::binary);

    outFile.write(reinterpret_cast<char*>(cacheData.data()), cacheData.size());
    for (auto pipeline : pipelineHash_)
    {
      if (pipeline.second != VK_NULL_HANDLE)
      {
        vkDestroyPipeline(device_, pipeline.second, nullptr);
      }
    }
    spdlog::info("destroy pipeline");
  }

  VkPipeline VkPipelinePool::createPipeline(VkPipelineProgram program)
  {
    VkPipeline pipeline = getPipeline(program);
    if (pipeline == VK_NULL_HANDLE)
    {
      return createPipeline(program.vertexType,
                            program.vertShaderModule,
                            program.fragShaderModule,
                            program.renderingType,
                            program.pipelineLayout);
    }
    return pipeline;
  }

  VkPipeline VkPipelinePool::getPipeline(
    VkPipelineProgram program
    ) const
  {
    auto it = pipelineHash_.find(program);
    if (it != pipelineHash_.end())
      return it->second;
    return VK_NULL_HANDLE;
  }

  // VkPipeline VkPipelinePool::buildRTPipeline()
  //{
  //  //    // Instead of a simple triangle, we'll be loading a more complex scene for this example
  //  //    // The shaders are accessing the vertex and index buffers of the scene, so the proper usage flag has to be set on the vertex and index buffers for the scene
  //  //    vkglTF::memoryPropertyFlags = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
  //  //      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  //  //    const uint32_t glTFLoadingFlags = vkglTF::FileLoadingFlags::PreTransformVertices |
  //  //      vkglTF::FileLoadingFlags::PreMultiplyVertexColors | vkglTF::FileLoadingFlags::FlipY;
  //  //    scene.loadFromFile(getAssetPath() + "models/reflection_scene.gltf", vulkanDevice, queue, glTFLoadingFlags);
  //  //
  //  //    VkDeviceOrHostAddressConstKHR vertexBufferDeviceAddress{};
  //  //    VkDeviceOrHostAddressConstKHR indexBufferDeviceAddress{};
  //  //
  //  //    vertexBufferDeviceAddress.deviceAddress = getBufferDeviceAddress(scene.vertices.buffer);
  //  //    indexBufferDeviceAddress.deviceAddress = getBufferDeviceAddress(scene.indices.buffer);
  //  //
  //  //    uint32_t numTriangles = static_cast<uint32_t>(scene.indices.count) / 3;
  //  //
  //  //    // Build
  //  //    VkAccelerationStructureGeometryKHR accelerationStructureGeometry =
  //  //      vks::initializers::accelerationStructureGeometryKHR();
  //  //    accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  //  //    accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  //  //    accelerationStructureGeometry.geometry.triangles.sType =
  //  //      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  //  //    accelerationStructureGeometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  //  //    accelerationStructureGeometry.geometry.triangles.vertexData = vertexBufferDeviceAddress;
  //  //    accelerationStructureGeometry.geometry.triangles.maxVertex = scene.vertices.count - 1;
  //  //    accelerationStructureGeometry.geometry.triangles.vertexStride = sizeof(vkglTF::Vertex);
  //  //    accelerationStructureGeometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
  //  //    accelerationStructureGeometry.geometry.triangles.indexData = indexBufferDeviceAddress;
  //  //    accelerationStructureGeometry.geometry.triangles.transformData.deviceAddress = 0;
  //  //    accelerationStructureGeometry.geometry.triangles.transformData.hostAddress = nullptr;
  //  //
  //  //    // Get size info
  //  //    VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo =
  //  //      vks::initializers::accelerationStructureBuildGeometryInfoKHR();
  //  //    accelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  //  //    accelerationStructureBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  //  //    accelerationStructureBuildGeometryInfo.geometryCount = 1;
  //  //    accelerationStructureBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;
  //  //
  //  //    VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo =
  //  //      vks::initializers::accelerationStructureBuildSizesInfoKHR();
  //  //    vkGetAccelerationStructureBuildSizesKHR(
  //  //                                            device,
  //  //                                            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
  //  //                                            &accelerationStructureBuildGeometryInfo,
  //  //                                            &numTriangles,
  //  //                                            &accelerationStructureBuildSizesInfo);
  //  //
  //  //    createAccelerationStructure(bottomLevelAS, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
  //  //                                accelerationStructureBuildSizesInfo);
  //  //
  //  //    // Create a small scratch buffer used during build of the bottom level acceleration structure
  //  //    ScratchBuffer scratchBuffer = createScratchBuffer(accelerationStructureBuildSizesInfo.buildScratchSize);
  //  //
  //  //    VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo =
  //  //      vks::initializers::accelerationStructureBuildGeometryInfoKHR();
  //  //    accelerationBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  //  //    accelerationBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  //  //    accelerationBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  //  //    accelerationBuildGeometryInfo.dstAccelerationStructure = bottomLevelAS.handle;
  //  //    accelerationBuildGeometryInfo.geometryCount = 1;
  //  //    accelerationBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;
  //  //    accelerationBuildGeometryInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;
  //  //
  //  //    VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
  //  //    accelerationStructureBuildRangeInfo.primitiveCount = numTriangles;
  //  //    accelerationStructureBuildRangeInfo.primitiveOffset = 0;
  //  //    accelerationStructureBuildRangeInfo.firstVertex = 0;
  //  //    accelerationStructureBuildRangeInfo.transformOffset = 0;
  //  //    std::vector<VkAccelerationStructureBuildRangeInfoKHR*> accelerationBuildStructureRangeInfos = {
  //  //      &accelerationStructureBuildRangeInfo
  //  //    };
  //  //
  //  //    // Build the acceleration structure on the device via a one-time command buffer submission
  //  //    // Some implementations may support acceleration structure building on the host (VkPhysicalDeviceAccelerationStructureFeaturesKHR->accelerationStructureHostCommands), but we prefer device builds
  //  //    VkCommandBuffer commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
  //  //    vkCmdBuildAccelerationStructuresKHR(
  //  //                                        commandBuffer,
  //  //                                        1,
  //  //                                        &accelerationBuildGeometryInfo,
  //  //                                        accelerationBuildStructureRangeInfos.data());
  //  //    vulkanDevice->flushCommandBuffer(commandBuffer, queue);
  //  //
  //  //    deleteScratchBuffer(scratchBuffer);
  //  //  }
  //  //
  //  //  VkTransformMatrixKHR transformMatrix = {
  //  //    1.0f, 0.0f, 0.0f, 0.0f,
  //  //    0.0f, 1.0f, 0.0f, 0.0f,
  //  //    0.0f, 0.0f, 1.0f, 0.0f
  //  //  };
  //  //
  //  //  VkAccelerationStructureInstanceKHR instance{};
  //  //  instance
  //  //  .
  //  //  transform= transformMatrix;
  //  //  instance
  //  //  .
  //  //  instanceCustomIndex=
  //  //  0;
  //  //  instance
  //  //  .
  //  //  mask=
  //  //  0xFF;
  //  //  instance
  //  //  .
  //  //  instanceShaderBindingTableRecordOffset=
  //  //  0;
  //  //  instance
  //  //  .
  //  //  flags= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
  //  //  instance
  //  //  .
  //  //  accelerationStructureReference= bottomLevelAS
  //  //  .
  //  //  deviceAddress;
  //  //
  //  //  // Buffer for instance data
  //  //  vks::Buffer instancesBuffer;
  //  //  VK_CHECK_RESULT (vulkanDevice
  //  //  ->
  //  //  createBuffer (
  //  //    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
  //  //  |
  //  //  VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
  //  //  ,
  //  //  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
  //  //  |
  //  //  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
  //  //  ,
  //  //  &
  //  //  instancesBuffer
  //  //  ,
  //  //  sizeof
  //  //  (VkAccelerationStructureInstanceKHR),
  //  //    &instance
  //  //  )
  //  //  );
  //  //
  //  //  VkDeviceOrHostAddressConstKHR instanceDataDeviceAddress{};
  //  //  instanceDataDeviceAddress
  //  //  .
  //  //  deviceAddress= getBufferDeviceAddress
  //  //  (instancesBuffer
  //  //  .
  //  //  buffer
  //  //  );
  //  //
  //  //  VkAccelerationStructureGeometryKHR accelerationStructureGeometry =
  //  //    vks::initializers::accelerationStructureGeometryKHR();
  //  //  accelerationStructureGeometry
  //  //  .
  //  //  geometryType= VK_GEOMETRY_TYPE_INSTANCES_KHR;
  //  //  accelerationStructureGeometry
  //  //  .
  //  //  flags= VK_GEOMETRY_OPAQUE_BIT_KHR;
  //  //  accelerationStructureGeometry
  //  //  .
  //  //  geometry
  //  //  .
  //  //  instances
  //  //  .
  //  //  sType= VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  //  //  accelerationStructureGeometry
  //  //  .
  //  //  geometry
  //  //  .
  //  //  instances
  //  //  .
  //  //  arrayOfPointers=
  //  //  VK_FALSE;
  //  //  accelerationStructureGeometry
  //  //  .
  //  //  geometry
  //  //  .
  //  //  instances
  //  //  .
  //  //  data= instanceDataDeviceAddress;
  //  //
  //  //  // Get size info
  //  //  VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo =
  //  //    vks::initializers::accelerationStructureBuildGeometryInfoKHR();
  //  //  accelerationStructureBuildGeometryInfo
  //  //  .
  //  //  type= VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  //  //  accelerationStructureBuildGeometryInfo
  //  //  .
  //  //  flags= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  //  //  accelerationStructureBuildGeometryInfo
  //  //  .
  //  //  geometryCount=
  //  //  1;
  //  //  accelerationStructureBuildGeometryInfo
  //  //  .
  //  //  pGeometries=
  //  //  &
  //  //  accelerationStructureGeometry;
  //  //
  //  //  uint32_t primitive_count = 1;
  //  //
  //  //  VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo =
  //  //    vks::initializers::accelerationStructureBuildSizesInfoKHR();
  //  //  vkGetAccelerationStructureBuildSizesKHR(
  //  //                                          device,
  //  //                                          VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
  //  //                                          &accelerationStructureBuildGeometryInfo,
  //  //                                          &primitive_count,
  //  //                                          &accelerationStructureBuildSizesInfo);
  //  //
  //  //  createAccelerationStructure(topLevelAS, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
  //  //                              accelerationStructureBuildSizesInfo);
  //  //
  //  //  // Create a small scratch buffer used during build of the top level acceleration structure
  //  //  ScratchBuffer scratchBuffer = createScratchBuffer(accelerationStructureBuildSizesInfo.buildScratchSize);
  //  //
  //  //  VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo =
  //  //    vks::initializers::accelerationStructureBuildGeometryInfoKHR();
  //  //  accelerationBuildGeometryInfo
  //  //  .
  //  //  type= VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  //  //  accelerationBuildGeometryInfo
  //  //  .
  //  //  flags= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  //  //  accelerationBuildGeometryInfo
  //  //  .
  //  //  mode= VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  //  //  accelerationBuildGeometryInfo
  //  //  .
  //  //  dstAccelerationStructure= topLevelAS
  //  //  .
  //  //  handle;
  //  //  accelerationBuildGeometryInfo
  //  //  .
  //  //  geometryCount=
  //  //  1;
  //  //  accelerationBuildGeometryInfo
  //  //  .
  //  //  pGeometries=
  //  //  &
  //  //  accelerationStructureGeometry;
  //  //  accelerationBuildGeometryInfo
  //  //  .
  //  //  scratchData
  //  //  .
  //  //  deviceAddress= scratchBuffer
  //  //  .
  //  //  deviceAddress;
  //  //
  //  //  VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
  //  //  accelerationStructureBuildRangeInfo
  //  //  .
  //  //  primitiveCount=
  //  //  1;
  //  //  accelerationStructureBuildRangeInfo
  //  //  .
  //  //  primitiveOffset=
  //  //  0;
  //  //  accelerationStructureBuildRangeInfo
  //  //  .
  //  //  firstVertex=
  //  //  0;
  //  //  accelerationStructureBuildRangeInfo
  //  //  .
  //  //  transformOffset=
  //  //  0;
  //  //  std::vector<VkAccelerationStructureBuildRangeInfoKHR*> accelerationBuildStructureRangeInfos = {
  //  //    &accelerationStructureBuildRangeInfo
  //  //  };
  //  //
  //  //  // Build the acceleration structure on the device via a one-time command buffer submission
  //  //  // Some implementations may support acceleration structure building on the host (VkPhysicalDeviceAccelerationStructureFeaturesKHR->accelerationStructureHostCommands), but we prefer device builds
  //  //  VkCommandBuffer commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
  //  //  vkCmdBuildAccelerationStructuresKHR(
  //  //                                      commandBuffer,
  //  //                                      1,
  //  //                                      &accelerationBuildGeometryInfo,
  //  //                                      accelerationBuildStructureRangeInfos.data());
  //  //  vulkanDevice
  //  //  ->
  //  //  flushCommandBuffer(commandBuffer, queue);
  //  //
  //  //  deleteScratchBuffer (scratchBuffer);
  //  //  instancesBuffer
  //  //  .
  //  //  destroy();
  //  //  const uint32_t handleSize = rayTracingPipelineProperties.shaderGroupHandleSize;
  //  //  const uint32_t handleSizeAligned = vks::tools::alignedSize(rayTracingPipelineProperties.shaderGroupHandleSize,
  //  //                                                             rayTracingPipelineProperties.shaderGroupHandleAlignment);
  //  //  const uint32_t groupCount = static_cast<uint32_t>(shaderGroups.size());
  //  //  const uint32_t sbtSize = groupCount * handleSizeAligned;
  //  //
  //  //  std::vector<uint8_t> shaderHandleStorage(sbtSize);
  //  //  VK_CHECK_RESULT (vkGetRayTracingShaderGroupHandlesKHR
  //  //  (device
  //  //  ,
  //  //  pipeline
  //  //  ,
  //  //  0
  //  //  ,
  //  //  groupCount
  //  //  ,
  //  //  sbtSize
  //  //  ,
  //  //  shaderHandleStorage
  //  //  .
  //  //  data()
  //  //  )
  //  //  );
  //  //
  //  //  createShaderBindingTable (shaderBindingTables
  //  //  .
  //  //  raygen
  //  //  ,
  //  //  1
  //  //  );
  //  //  createShaderBindingTable (shaderBindingTables
  //  //  .
  //  //  miss
  //  //  ,
  //  //  1
  //  //  );
  //  //  createShaderBindingTable (shaderBindingTables
  //  //  .
  //  //  hit
  //  //  ,
  //  //  1
  //  //  );
  //  //
  //  //  // Copy handles
  //  //  memcpy (shaderBindingTables
  //  //  .
  //  //  raygen
  //  //  .
  //  //  mapped
  //  //  ,
  //  //  shaderHandleStorage
  //  //  .
  //  //  data(), handleSize
  //  //  );
  //  //  memcpy (shaderBindingTables
  //  //  .
  //  //  miss
  //  //  .
  //  //  mapped
  //  //  ,
  //  //  shaderHandleStorage
  //  //  .
  //  //  data()
  //  //  +
  //  //  handleSizeAligned
  //  //  ,
  //  //  handleSize
  //  //  );
  //  //  memcpy (shaderBindingTables
  //  //  .
  //  //  hit
  //  //  .
  //  //  mapped
  //  //  ,
  //  //  shaderHandleStorage
  //  //  .
  //  //  data()
  //  //  +
  //  //  handleSizeAligned*
  //  //  2
  //  //  ,
  //  //  handleSize
  //  //  );
  //  //
  //  //  std::vector<VkDescriptorPoolSize> poolSizes = {
  //  //    {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, maxConcurrentFrames},
  //  //    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, maxConcurrentFrames},
  //  //    {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, maxConcurrentFrames},
  //  //    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, maxConcurrentFrames * 2}
  //  //  };
  //  //  VkDescriptorPoolCreateInfo descriptorPoolCreateInfo =
  //  //    vks::initializers::descriptorPoolCreateInfo(poolSizes, maxConcurrentFrames);
  //  //  VK_CHECK_RESULT (vkCreateDescriptorPool
  //  //  (device
  //  //  ,
  //  //  &
  //  //  descriptorPoolCreateInfo
  //  //  ,
  //  //  nullptr
  //  //  ,
  //  //  &
  //  //  descriptorPool
  //  //  )
  //  //  );
  //  //
  //  //  // Sets per frame, just like the buffers themselves
  //  //  // Acceleration structure, vertex and index buffers and images do not need to be duplicated per frame, we use the same for each descriptor to keep things simple
  //  //  VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool,
  //  //    &descriptorSetLayout, 1);
  //  //  for
  //  //  (
  //  //  auto i = 0;
  //  //  i<maxConcurrentFrames;
  //  //  i
  //  //  ++
  //  //  )
  //  // {
  //  //			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets[i]));
  //  //
  //  //			// The fragment shader needs access to the ray tracing acceleration structure, so we pass it as a descriptor
  //  //
  //  //			VkWriteDescriptorSetAccelerationStructureKHR descriptorAccelerationStructureInfo{};
  //  //			descriptorAccelerationStructureInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
  //  //			descriptorAccelerationStructureInfo.accelerationStructureCount = 1;
  //  //			descriptorAccelerationStructureInfo.pAccelerationStructures = &topLevelAS.handle;
  //  //
  //  //			VkWriteDescriptorSet accelerationStructureWrite{};
  //  //			accelerationStructureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  //  //			// The specialized acceleration structure descriptor has to be chained
  //  //			accelerationStructureWrite.pNext = &descriptorAccelerationStructureInfo;
  //  //			accelerationStructureWrite.dstSet = descriptorSets[i];
  //  //			accelerationStructureWrite.dstBinding = 0;
  //  //			accelerationStructureWrite.descriptorCount = 1;
  //  //			accelerationStructureWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  //  //
  //  //			VkDescriptorImageInfo storageImageDescriptor{ VK_NULL_HANDLE, storageImage.view, VK_IMAGE_LAYOUT_GENERAL };
  //  //			VkDescriptorBufferInfo vertexBufferDescriptor{ scene.vertices.buffer, 0, VK_WHOLE_SIZE };
  //  //			VkDescriptorBufferInfo indexBufferDescriptor{ scene.indices.buffer, 0, VK_WHOLE_SIZE };
  //  //
  //  //			std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
  //  //				// Binding 0: Top level acceleration structure
  //  //				accelerationStructureWrite,
  //  //				// Binding 1: Ray tracing result image
  //  //				vks::initializers::writeDescriptorSet(descriptorSets[i], VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, &storageImageDescriptor),
  //  //				// Binding 2: Uniform data
  //  //				vks::initializers::writeDescriptorSet(descriptorSets[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2, &uniformBuffers[i].descriptor),
  //  //				// Binding 3: Scene vertex buffer
  //  //				vks::initializers::writeDescriptorSet(descriptorSets[i], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3, &vertexBufferDescriptor),
  //  //				// Binding 4: Scene index buffer
  //  //				vks::initializers::writeDescriptorSet(descriptorSets[i], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, &indexBufferDescriptor),
  //  //			};
  //  //			vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, VK_NULL_HANDLE);
  //  //
  //  //		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
  //  //			// Binding 0: Acceleration structure
  //  //			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0),
  //  //			// Binding 1: Storage image
  //  //			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 1),
  //  //			// Binding 2: Uniform buffer
  //  //			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR, 2),
  //  //			// Binding 3: Vertex buffer
  //  //			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 3),
  //  //			// Binding 4: Index buffer
  //  //			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 4),
  //  //		};
  //  //
  //  //		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
  //  //		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayout));
  //  //
  //  //		VkPipelineLayoutCreateInfo pPipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
  //  //		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCI, nullptr, &pipelineLayout));
  //  //
  //  //		/*
  //  //			Setup ray tracing shader groups
  //  //		*/
  //  //		std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
  //  //
  //  //		VkSpecializationMapEntry specializationMapEntry = vks::initializers::specializationMapEntry(0, 0, sizeof(uint32_t));
  //  //		uint32_t maxRecursion = 4;
  //  //		VkSpecializationInfo specializationInfo = vks::initializers::specializationInfo(1, &specializationMapEntry, sizeof(maxRecursion), &maxRecursion);
  //  //
  //  //			shaderStages.push_back(loadShader(getShadersPath() + "raytracingreflections/raygen.rgen.spv", VK_SHADER_STAGE_RAYGEN_BIT_KHR));
  //  //			// Pass recursion depth for reflections to ray generation shader via specialization constant
  //  //			shaderStages.back().pSpecializationInfo = &specializationInfo;
  //  //			VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
  //  //			shaderGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
  //  //			shaderGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  //  //			shaderGroup.generalShader = static_cast<uint32_t>(shaderStages.size()) - 1;
  //  //			shaderGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
  //  //			shaderGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
  //  //			shaderGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
  //  //			shaderGroups.push_back(shaderGroup);
  //  //
  //  //			shaderStages.push_back(loadShader(getShadersPath() + "raytracingreflections/miss.rmiss.spv", VK_SHADER_STAGE_MISS_BIT_KHR));
  //  //			VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
  //  //			shaderGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
  //  //			shaderGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  //  //			shaderGroup.generalShader = static_cast<uint32_t>(shaderStages.size()) - 1;
  //  //			shaderGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
  //  //			shaderGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
  //  //			shaderGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
  //  //			shaderGroups.push_back(shaderGroup);
  //  //
  //  //			shaderStages.push_back(loadShader(getShadersPath() + "raytracingreflections/closesthit.rchit.spv", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR));
  //  //			VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
  //  //			shaderGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
  //  //			shaderGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  //  //			shaderGroup.generalShader = VK_SHADER_UNUSED_KHR;
  //  //			shaderGroup.closestHitShader = static_cast<uint32_t>(shaderStages.size()) - 1;
  //  //			shaderGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
  //  //			shaderGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
  //  //			shaderGroups.push_back(shaderGroup);
  //  //
  //  //
  //  //		VkRayTracingPipelineCreateInfoKHR rayTracingPipelineCI = vks::initializers::rayTracingPipelineCreateInfoKHR();
  //  //		rayTracingPipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
  //  //		rayTracingPipelineCI.pStages = shaderStages.data();
  //  //		rayTracingPipelineCI.groupCount = static_cast<uint32_t>(shaderGroups.size());
  //  //		rayTracingPipelineCI.pGroups = shaderGroups.data();
  //  //		rayTracingPipelineCI.maxPipelineRayRecursionDepth = std::min(uint32_t(4), rayTracingPipelineProperties.maxRayRecursionDepth);
  //  //		rayTracingPipelineCI.layout = pipelineLayout;
  //  //		VK_CHECK_RESULT(vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rayTracingPipelineCI, nullptr, &pipeline));
  //    //  }]
  //}

  void VkPipelinePool::createComputePipeline(
    const std::vector<VkDescriptorSetLayout>& descriptorSetLayouts,
    VkShaderModule computeShader
    )
  {
    //push constant setting
    VkPushConstantRange pushConstant{};
    pushConstant.offset = 0;
    //pushConstant.size       = sizeof(gpu::constant);
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    //if need -> create and layout setting
    VkPipelineLayoutCreateInfo computeLayout{};
    computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    computeLayout.pNext = nullptr;
    computeLayout.pSetLayouts = descriptorSetLayouts.data();
    computeLayout.setLayoutCount = descriptorSetLayouts.size();
    computeLayout.pPushConstantRanges = &pushConstant;
    computeLayout.pushConstantRangeCount = 1;
    computeLayout.pPushConstantRanges = &pushConstant;
    computeLayout.pushConstantRangeCount = 1;

    if (vkCreatePipelineLayout(device_, &computeLayout, nullptr, &computePipelineLayout_) != VK_SUCCESS)
    {
      throw std::runtime_error("devicecreate pipeline layout!");
    }
    VkPipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = computeShader;
    shaderStageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = computePipelineLayout_;
    pipelineInfo.flags = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;
  }

  void VkPipelinePool::buildVertexDescriptor(VertexType type,
                                             VkVertexInputBindingDescription& vertexBindingDesc,
                                             std::vector<VkVertexInputAttributeDescription>&
                                             vertexAttributeDescriptions,
                                             VkPipelineVertexInputStateCreateInfo& vertexInputInfo,
                                             uint32_t vertexBinding)
  {
    switch (type)
    {
      case (gpu::VertexType::ALL):
      {
        vertexBindingDesc.binding = 0;
        vertexBindingDesc.stride = sizeof(gpu::VertexAll);
        vertexBindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        vertexAttributeDescriptions.resize(8);
        vertexAttributeDescriptions[0].binding = vertexBinding;
        vertexAttributeDescriptions[0].location = 0;
        vertexAttributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        vertexAttributeDescriptions[0].offset = offsetof(gpu::VertexAll, position);

        vertexAttributeDescriptions[1].binding = vertexBinding;
        vertexAttributeDescriptions[1].location = 1;
        vertexAttributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        vertexAttributeDescriptions[1].offset = offsetof(gpu::VertexAll, normal);

        vertexAttributeDescriptions[2].binding = vertexBinding;
        vertexAttributeDescriptions[2].location = 2;
        vertexAttributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        vertexAttributeDescriptions[2].offset = offsetof(gpu::VertexAll, uv);

        vertexAttributeDescriptions[3].binding = vertexBinding;
        vertexAttributeDescriptions[3].location = 3;
        vertexAttributeDescriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
        vertexAttributeDescriptions[3].offset = offsetof(gpu::VertexAll, tangent);

        vertexAttributeDescriptions[4].binding = vertexBinding;
        vertexAttributeDescriptions[4].location = 4;
        vertexAttributeDescriptions[4].format = VK_FORMAT_R32G32B32_SFLOAT;
        vertexAttributeDescriptions[4].offset = offsetof(gpu::VertexAll, bitangent);

        vertexAttributeDescriptions[5].binding = vertexBinding;
        vertexAttributeDescriptions[5].location = 5;
        vertexAttributeDescriptions[5].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        vertexAttributeDescriptions[5].offset = offsetof(gpu::VertexAll, color);

        vertexAttributeDescriptions[6].binding = vertexBinding;
        vertexAttributeDescriptions[6].location = 6;
        vertexAttributeDescriptions[6].format = VK_FORMAT_R32G32B32A32_SINT;
        vertexAttributeDescriptions[6].offset = offsetof(gpu::VertexAll, boneIndices);

        vertexAttributeDescriptions[7].binding = vertexBinding;
        vertexAttributeDescriptions[7].location = 7;
        vertexAttributeDescriptions[7].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        vertexAttributeDescriptions[7].offset = offsetof(gpu::VertexAll, boneWeights);
        break;
      }
      case (VertexType::PC):
      {
        vertexBindingDesc.binding = vertexBinding;
        vertexBindingDesc.stride = sizeof(VertexPC);
        vertexBindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        vertexAttributeDescriptions.resize(2);
        vertexAttributeDescriptions[0].binding = vertexBinding;
        vertexAttributeDescriptions[0].location = 0;
        vertexAttributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        vertexAttributeDescriptions[0].offset = offsetof(VertexPC, position);

        vertexAttributeDescriptions[1].binding = vertexBinding;
        vertexAttributeDescriptions[1].location = 1;
        vertexAttributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        vertexAttributeDescriptions[1].offset = offsetof(VertexPC, color);
        break;
      }
      case (VertexType::PUVN):
      {
        vertexBindingDesc.binding = vertexBinding;
        vertexBindingDesc.stride = sizeof(VertexPUVN);
        vertexBindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        vertexAttributeDescriptions.resize(3);
        vertexAttributeDescriptions[0].binding = vertexBinding;
        vertexAttributeDescriptions[0].location = 0;
        vertexAttributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        vertexAttributeDescriptions[0].offset = offsetof(VertexPUVN, position);

        vertexAttributeDescriptions[1].binding = vertexBinding;
        vertexAttributeDescriptions[1].location = 1;
        vertexAttributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
        vertexAttributeDescriptions[1].offset = offsetof(VertexPUVN, uv);

        vertexAttributeDescriptions[2].binding = vertexBinding;
        vertexAttributeDescriptions[2].location = 2;
        vertexAttributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
        vertexAttributeDescriptions[2].offset = offsetof(VertexPUVN, normal);
        break;
      }
      case (VertexType::PUVNTC):
      {
        vertexBindingDesc.binding = vertexBinding;
        vertexBindingDesc.stride = sizeof(VertexPUVNTC);
        vertexBindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        vertexAttributeDescriptions.resize(5);

        vertexAttributeDescriptions[0].binding = vertexBinding;
        vertexAttributeDescriptions[0].location = 0;
        vertexAttributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        vertexAttributeDescriptions[0].offset = offsetof(VertexPUVNTC, position);

        vertexAttributeDescriptions[1].binding = vertexBinding;
        vertexAttributeDescriptions[1].location = 1;
        vertexAttributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
        vertexAttributeDescriptions[1].offset = offsetof(VertexPUVNTC, uv);

        vertexAttributeDescriptions[2].binding = vertexBinding;
        vertexAttributeDescriptions[2].location = 2;
        vertexAttributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
        vertexAttributeDescriptions[2].offset = offsetof(VertexPUVNTC, normal);

        vertexAttributeDescriptions[3].binding = vertexBinding;
        vertexAttributeDescriptions[3].location = 3;
        vertexAttributeDescriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
        vertexAttributeDescriptions[3].offset = offsetof(VertexPUVNTC, tangent);

        vertexAttributeDescriptions[4].binding = vertexBinding;
        vertexAttributeDescriptions[4].location = 4;
        vertexAttributeDescriptions[4].format = VK_FORMAT_R32G32B32_SFLOAT;
        vertexAttributeDescriptions[4].offset = offsetof(VertexPUVNTC, color);
        break;
      }
      case (VertexType::QUAD):
      {
        vertexBindingDesc.binding = vertexBinding;
        vertexBindingDesc.stride = sizeof(Quad);
        vertexBindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        vertexAttributeDescriptions.resize(2);

        vertexAttributeDescriptions[0].binding = vertexBinding;
        vertexAttributeDescriptions[0].location = 0;
        vertexAttributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        vertexAttributeDescriptions[0].offset = offsetof(Quad, point1);

        vertexAttributeDescriptions[1].binding = vertexBinding;
        vertexAttributeDescriptions[1].location = 0;
        vertexAttributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
        vertexAttributeDescriptions[1].offset = offsetof(Quad, point2);
      }
      case (VertexType::BACKGROUND):
      {
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr;
        return;
      }
      default:
      {
        break;
      }
    }
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &vertexBindingDesc;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = vertexAttributeDescriptions.data();
  }

  void VkPipelinePool::buildDepthStencilPipeline(VkPipelineDepthStencilStateCreateInfo& depthStencilCi,
                                                 VkBool32 depthTestEnable,
                                                 VkBool32 depthWriteEnable,
                                                 VkBool32 stencilTestEnable,
                                                 VkCompareOp depthCompareOp)
  {
    depthStencilCi.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencilCi.depthTestEnable = depthTestEnable;
    depthStencilCi.depthWriteEnable = depthWriteEnable;
    depthStencilCi.depthCompareOp = depthCompareOp;
    depthStencilCi.depthBoundsTestEnable = VK_FALSE;
    depthStencilCi.stencilTestEnable = VK_FALSE;
    //todo : check need the stencil setting
  }

  void VkPipelinePool::buildFragmentPipeline(VkPipelineShaderStageCreateInfo& shaderStateCi,
                                             VkShaderModule fragModule)
  {
    shaderStateCi.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStateCi.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStateCi.module = fragModule;
    shaderStateCi.pName = "main";
  }

  void VkPipelinePool::buildVertexPipeline(VkPipelineShaderStageCreateInfo& shaderStateCi,
                                           VkShaderModule vertexModule)
  {
    shaderStateCi.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStateCi.stage = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStateCi.module = vertexModule;
    shaderStateCi.pName = "main";
  }

  void VkPipelinePool::buildAssemblyPipeline(VkPipelineInputAssemblyStateCreateInfo& inputAssembly,
                                             VkPrimitiveTopology topology,
                                             VkBool32 primitiveRestartEnable,
                                             VkPipelineInputAssemblyStateCreateFlags
                                             flags)
  {
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = topology;
    inputAssembly.primitiveRestartEnable = primitiveRestartEnable;
    //inputAssembly.flags                  = flags;
  }

  void VkPipelinePool::buildDynamicPipelineDyscriptor(VkPipelineDynamicStateCreateInfo& dynamicStateCi,
                                                      std::vector<VkDynamicState>& dynamicStates,
                                                      VkPipelineViewportStateCreateInfo& viewportStateCi,
                                                      uint32_t viewCount,
                                                      VkBool32 dynamicStencilTestEnable,
                                                      VkBool32 dynamicStateDepthCompare,
                                                      VkBool32 dynamicStateVetexStride,
                                                      VkPipelineDynamicStateCreateFlags flags)
  {
    ///todo:
    /// if need -> create View and Scissor with array
    /// update all and create Mulit view mode
    /// temp-> just single view and multi draw call

    viewportStateCi.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportStateCi.viewportCount = viewCount;
    viewportStateCi.scissorCount = viewCount;

    dynamicStates.push_back(VK_DYNAMIC_STATE_VIEWPORT);
    dynamicStates.push_back(VK_DYNAMIC_STATE_SCISSOR);
    dynamicStates.push_back(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);
    dynamicStates.push_back(VK_DYNAMIC_STATE_POLYGON_MODE_EXT);

    if (dynamicStencilTestEnable) dynamicStates.push_back(VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE);
    if (dynamicStateDepthCompare) dynamicStates.push_back(VK_DYNAMIC_STATE_DEPTH_COMPARE_OP);
    if (dynamicStateVetexStride) dynamicStates.push_back(VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE);

    dynamicStateCi.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateCi.dynamicStateCount = dynamicStates.size();
    dynamicStateCi.pDynamicStates = dynamicStates.data();
  }

  void VkPipelinePool::buildRasterizationPipeline(VkPipelineRasterizationStateCreateInfo& rasterizaCi,
                                                  VkCullModeFlags cullMode,
                                                  VkPolygonMode mode,
                                                  VkFrontFace front,
                                                  VkPipelineRasterizationStateCreateFlags flags)
  {
    rasterizaCi.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizaCi.depthClampEnable = VK_FALSE;
    rasterizaCi.rasterizerDiscardEnable = VK_FALSE;
    rasterizaCi.lineWidth = 1.0f;
    rasterizaCi.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizaCi.cullMode = cullMode;
    rasterizaCi.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizaCi.depthBiasEnable = VK_FALSE;
    //rasterizaCi.flags                   = flags;
  }

  void VkPipelinePool::buildMultiSamplingPipeline(VkPipelineMultisampleStateCreateInfo& multiSamplingCi,
                                                  VkSampleCountFlagBits samples,
                                                  VkPipelineMultisampleStateCreateFlags flags)
  {
    multiSamplingCi.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multiSamplingCi.rasterizationSamples = samples;
    //multiSamplingCi.flags                = flags;
  }

  VkPipelineColorBlendAttachmentState VkPipelinePool::buildColorBlendingAttachment(uint32_t flags,
    VkBool32 blendEnable)
  {
    //this can be vector
    VkPipelineColorBlendAttachmentState state;
    state.colorWriteMask = flags;
    state.blendEnable = VK_FALSE;
    return state;
  }

  void VkPipelinePool::buildColorBlendingPipeline(VkPipelineColorBlendStateCreateInfo& colorBlendingCi,
                                                  VkPipelineColorBlendAttachmentState* attachment,
                                                  uint32_t attachmentCount)
  {
    colorBlendingCi.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlendingCi.attachmentCount = attachmentCount;
    colorBlendingCi.pAttachments = attachment;
  }

  void VkPipelinePool::buildDynamicRenderingPipeline(VkPipelineRenderingCreateInfo& dynamicRendering,
                                                     VkFormat* colorAttachmentFormats,
                                                     uint32_t colorAttachmentCount,
                                                     uint32_t viewMask,
                                                     VkFormat depthFormat,
                                                     VkFormat stencilAttachment)
  {
    dynamicRendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    dynamicRendering.pColorAttachmentFormats = colorAttachmentFormats;
    dynamicRendering.colorAttachmentCount = colorAttachmentCount;
    dynamicRendering.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
    dynamicRendering.viewMask = viewMask;
    dynamicRendering.stencilAttachmentFormat = stencilAttachment;
  }

  VkPipelineLayout VkPipelinePool::createPipelineLayout(
    VkDescriptorSetLayout* descriptorLayoutData,
    uint32_t descriptorSetCount
    )
  {
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(gpu::VkConstant);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = descriptorSetCount;
    pipelineLayoutInfo.pSetLayouts = descriptorLayoutData;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
    {
      throw std::runtime_error("devicecreate pipeline layout!");
    }
    return pipelineLayout;
  }

  VkPipeline VkPipelinePool::createPipeline(VertexType type,
                                            VkShaderModule vertexModule,
                                            VkShaderModule fragModule,
                                            RenderingAttachmentType renderingType,
                                            VkPipelineLayout pipelineLayout,
                                            uint32_t viewMask,
                                            VkFormat depthFormat,
                                            VkFormat stencilAttachment,

                                            VkPrimitiveTopology topology,
                                            VkCullModeFlags cullMode,
                                            VkBool32 depthTestEnable,
                                            VkBool32 depthWriteEnable,
                                            VkBool32 stencilTestEnable,
                                            VkCompareOp depthCompareOp)
  {
    VkPipelineShaderStageCreateInfo vertexInputInfo{};
    buildVertexPipeline(vertexInputInfo, vertexModule);

    VkVertexInputBindingDescription vertexInputBinding{};
    std::vector<VkVertexInputAttributeDescription> vertexInputAttribute{};
    VkPipelineVertexInputStateCreateInfo vertexStateInfo{};

    buildVertexDescriptor(type, vertexInputBinding, vertexInputAttribute, vertexStateInfo, 0);

    VkPipelineShaderStageCreateInfo fragShaderInputInfo{};
    buildFragmentPipeline(fragShaderInputInfo, fragModule);

    VkPipelineShaderStageCreateInfo shaderStage[] = {vertexInputInfo, fragShaderInputInfo};

    VkPipelineDepthStencilStateCreateInfo depthStencilState{};
    buildDepthStencilPipeline(depthStencilState, depthTestEnable, depthWriteEnable);

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState{};
    buildAssemblyPipeline(inputAssemblyState, topology, false);

    VkPipelineDynamicStateCreateInfo dynamicStateInfo{};
    std::vector<VkDynamicState> dynamicStates{};
    VkPipelineViewportStateCreateInfo viewportState{};

    buildDynamicPipelineDyscriptor(dynamicStateInfo,
                                   dynamicStates,
                                   viewportState);

    VkPipelineRasterizationStateCreateInfo rasterizerState{};
    buildRasterizationPipeline(rasterizerState, cullMode);

    VkPipelineMultisampleStateCreateInfo multisampleState{};
    buildMultiSamplingPipeline(multisampleState);

    std::vector<VkFormat> formats;
    std::vector<VkPipelineColorBlendAttachmentState> colorBlendStates{};

    VkPipelineRenderingCreateInfo dRenderingInfo{};

    switch (renderingType)
    {
      case (RenderingAttachmentType::DEPTH):
      {
        buildDynamicRenderingPipeline(dRenderingInfo,
                                      nullptr,
                                      0);

        colorBlendStates.push_back(buildColorBlendingAttachment());
        break;
      }
      case (RenderingAttachmentType::SWAPCHAIN):
      {
        formats.push_back(VK_FORMAT_B8G8R8A8_SRGB);
        break;
      }
      case (RenderingAttachmentType::G_BUFFER):
      {
        formats.push_back(VK_FORMAT_R16G16B16A16_SFLOAT);
        formats.push_back(VK_FORMAT_B8G8R8A8_SRGB);
        formats.push_back(VK_FORMAT_R16G16B16A16_SNORM);
        formats.push_back(VK_FORMAT_R16_UNORM);
        break;
      }
      case(RenderingAttachmentType::LIGHTNING):
      {
        formats.push_back(VK_FORMAT_R16G16B16A16_SFLOAT);
        break;
      }

      case(RenderingAttachmentType::BLOOMING):
      {
        formats.push_back(VK_FORMAT_R32G32B32A32_SFLOAT);
        break;
      }

      case(RenderingAttachmentType::TONEMAP):
      {
        formats.push_back(VK_FORMAT_R16G16B16A16_SFLOAT);
        break;
      }
      case(RenderingAttachmentType::GAMMA_CORRECTION):
      {
        formats.push_back(VK_FORMAT_B8G8R8A8_SRGB);
        break;
      }
    }
    buildDynamicRenderingPipeline(dRenderingInfo,
                                  formats.data(),
                                  formats.size());
    for (uint32_t i = 0; i < formats.size(); i++)
    {
      colorBlendStates.push_back(buildColorBlendingAttachment());
    }
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    buildColorBlendingPipeline(colorBlending,
                               colorBlendStates.data(),
                               colorBlendStates.size());

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStage;
    pipelineInfo.pVertexInputState = &vertexStateInfo;
    pipelineInfo.pInputAssemblyState = &inputAssemblyState;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizerState;
    pipelineInfo.pMultisampleState = &multisampleState;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicStateInfo;
    pipelineInfo.pDepthStencilState = &depthStencilState;
    pipelineInfo.pNext = &dRenderingInfo;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = VK_NULL_HANDLE;
    pipelineInfo.subpass = 0;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_ASSERT(vkCreateGraphicsPipelines(device_,
                VK_NULL_HANDLE,
                1,
                &pipelineInfo,
                nullptr,
                &pipeline));

    VkPipelineProgram program;
    program.vertShaderModule = vertexModule;
    program.fragShaderModule = fragModule;
    program.topology = topology;
    pipelineHash_[program] = pipeline;

    return pipeline;
  }

  std::vector<uint8_t> VkPipelinePool::loadPipelineCache(
    const std::string& filename
    )
  {
    std::ifstream inFile(filename, std::ios::binary | std::ios::ate);
    if (!inFile.is_open())
    {
      spdlog::info("no pipeline cache data");
      return {};
    }
    std::streamsize size = inFile.tellg();
    inFile.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    if (!inFile.read(reinterpret_cast<char*>(buffer.data()), size))
    {
      spdlog::info("failed to read pipeline cache");
      return {};
    }
    return buffer;
  }
}
