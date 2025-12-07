
#ifndef MYPROJECT_ANIMATION_HPP
#define MYPROJECT_ANIMATION_HPP
#include <string>
#include <vector>

#include "glm/glm.hpp"


// Morph target: 정점 오프셋(positions, normals) 집합.
// positions/normals 는 target vertex count 에 맞춰져야 함.
struct MorphTarget{
  std::string name;
  std::vector<glm::vec3> positions; // 오프셋(또는 절대 위치) - size == vertexCount
  std::vector<glm::vec3> normals; // optional - size == vertexCount or empty

  MorphTarget() = default;

  MorphTarget(std::string n, std::vector<glm::vec3> pos, std::vector<glm::vec3> nrm = {})
  : name(std::move(n)), positions(std::move(pos)), normals(std::move(nrm)) {}
};

// 애니메이션 처리 모드
enum class AnimationMode : uint8_t{
  CPU_MORPH = 0, // CPU에서 블렌드 계산 후 버퍼 업로드
  VAT_TEXTURE // VAT: 애니메이션을 텍스처에 베이크하여 GPU 샘플링
};

// VAT 관련 설정(베이킹/포맷) — 필요 시 확장
struct VATConfig{
  int textureWidth;     // 텍스처 가로 (vertex 수)
  int textureHeight;    // 텍스처 세로 (frame 수)
  int frameCount;       // 애니메이션 총 프레임 수
  float frameRate;      // FPS
  int vertexCount;      // 모델 vertex 수
  //Format format;
  // 텍스처 포맷 등은 구현체에서 결정
};
//Skeleton hierarchy → Parent → Child 관계 유지, bone matrix 전파
//Timeline / Clip / Blending → 어떤 애니메이션을 재생할지 관리
//Multi-instance / VAT Texture → GPU 효율 위해 구조 설계
//Morph / BlendShape → 여러 delta를 linear blend, weight 관리
// MeshAnimator: morph targets 관리, 프레임 업데이트, GPU 업로드 또는 VAT 관리
class MeshAnimator{
public:
  MeshAnimator(std::shared_ptr<Buffer> baseVertexBuffer, uint32_t vertexCount) noexcept;
  ~MeshAnimator();

  // MorphTarget 관리
  void addMorphTarget(const MorphTarget &mt);
  std::size_t morphTargetCount() const noexcept;
  const MorphTarget &morphTargetAt(std::size_t idx) const;

  // 가중치 설정 (targetCount 크기 벡터 사용 권장)
  void setWeights(const std::vector<float> &weights);
  void setWeight(std::size_t index, float weight);

  // VAT 베이크(실제 베이킹은 소스에서 구현)
  // 반환: VAT 텍스처 핸들(shared_ptr) — 실패 시 nullptr
  std::shared_ptr<TextureManager> bakeVAT(const VATConfig &cfg);

  // 프레임 갱신: mode 에 따라 CPU 블렌드 후 업로드 또는 VAT 파라미터 설정
  void updateFrame(float timeSeconds, AnimationMode mode);

  // 렌더러가 바인드할 때 사용할 현재 동적 버퍼(정점 포지션 포함)
  // bind 단계에서 이 버퍼를 정점 버퍼로 바인드하면 됨
  std::shared_ptr<Buffer> currentDynamicVertexBuffer() const noexcept;

  // 렌더 바인드용 유틸: 커맨드 버퍼/디스크립터 세트에 필요한 셋업을 수행
  // (구체적 파라미터는 엔진/렌더러 규약에 맞춰 확장)
  void bindForRender(CommandBuffer &cmd, DescriptorSet &desc, PipelineLayout &layout) const;

  // 성능/동기화 옵션
  void setDoubleBuffering(bool enable) noexcept;
  void setPersistentMapping(bool enable) noexcept;

  // 정보 접근자
  uint32_t vertexCount() const noexcept
  {
    return m_vertexCount;
  }

  bool usesVAT() const noexcept
  {
    return m_vatTexture != nullptr;
  }

private:
  uint32_t m_vertexCount;
  std::shared_ptr<Buffer> m_vertexBase; // 원본(immutable) 정점 버퍼
  std::vector<std::shared_ptr<Buffer> > m_dynamicVB; // 더블/트리플 버퍼 지원
  std::shared_ptr<TextureManager> m_vatTexture; // VAT 텍스처(있을 경우)
  VATConfig m_vatConfig;

  std::vector<MorphTarget> m_morphTargets;
  std::vector<float> m_weights;

  bool m_doubleBuffering = true;
  bool m_persistentMapping = false;
  std::size_t m_currentBufferIdx = 0;

  // 내부: CPU에서 블렌드된 positions/normals 를 계산하여 반환
  // (소스에서 구현)
  std::vector<glm::vec3> computeBlendedPositions() const;
  std::vector<glm::vec3> computeBlendedNormals() const;

  // 내부: blended 결과를 현재 dynamic buffer 로 업로드 (소스에서 구현)
  void uploadBlendedToGPU(const std::vector<glm::vec3> &positions,
                          const std::vector<glm::vec3> &normals);
};

// 인스턴싱 지원: 여러 모델 매트릭스를 보관할 경우 사용
void setInstanceMatrices(std::vector<glm::mat4> matrices); // MeshAnimator 멤버로 추가 가능
const std::vector<glm::mat4> &instanceMatrices() const noexcept;

// 단일 모델 매트릭스(렌더러에서 주로 다루지만 animator에 보관 가능)
void setModelMatrix(const glm::mat4 &m) noexcept;
const glm::mat4 &modelMatrix() const noexcept;

// 스켈레톤/스킨 데이터 구조
struct Joint{
  int parentIndex; // 계층
  glm::mat4 inverseBindMatrix; // Inverse bind pose
};

void setSkeleton(const std::vector<Joint> &joints);
void setJointMatrices(const std::vector<glm::mat4> &jointMatrices); // GPU로 업로드할 행렬 배열

// 스킨 + 모프 혼합 방식 제어 (정책)
enum class SkinMorphOrder{ MORPH_THEN_SKIN, SKIN_THEN_MORPH };

void setSkinMorphOrder(SkinMorphOrder order) noexcept;

// RendererBinding 확장: 모델 매트릭스 / 인스턴스 버퍼 / 조인트 UBO/SSBO 바인드
void bindModelMatrix(CommandBuffer &cmd, const glm::mat4 &model, DescriptorSet &desc, PipelineLayout &layout);
void bindInstanceBuffer(CommandBuffer &cmd, /* instance buffer handle */);
void bindSkeleton(Buffer &jointMatrixBuffer, DescriptorSet &desc);


#endif //MYPROJECT_ANIMATION_HPP