# 고급 LLM 서빙 최적화와 프레임워크 선택

> 이 글의 목표는 단일 GPU 최적화를 넘어서는 대규모 LLM 서빙 기법을 이해하고, **Speculative Decoding·분산 병렬화·Prefill/Decode 분리·계층형 KV Cache**가 각각 어떤 병목을 해결하는지 설명할 수 있게 되는 것이다. 이어서 vLLM 내부 구조를 기준으로 전문 LLM 서빙 프레임워크가 요청·토큰·KV Cache·GPU 실행을 어떻게 연결하는지 살펴보고, 환경에 맞는 프레임워크를 선택하는 기준을 정리한다.
>
> **작성 상태:** 고급 LLM 최적화와 서빙 프레임워크의 핵심 내용을 복습하기 쉽도록 하나의 문서로 재구성했다. 이번에는 별도 실습을 수행하지 않았으며 측정 결과도 포함하지 않는다. 관련 실습은 문서 말미에 추후 진행 후보 목록으로만 남겼다.

## 먼저 보는 핵심 요약

1. **대규모 LLM 최적화는 전체 시스템 문제다.** Kernel, 실행 engine, KV Cache, routing과 orchestration 중 한 계층의 병목이 다른 계층의 개선 효과를 상쇄할 수 있다.
2. **모든 최적화는 새로운 비용을 만든다.** Speculative Decoding은 잘못 추측한 token, Tensor Parallelism은 collective 통신, P/D 분리는 KV 전송, 계층형 cache는 저장·복원 overhead를 지불한다.
3. **Speculative Decoding은 작은 draft가 여러 token을 제안하고 target이 병렬 검증한다.** 검증 알고리즘이 올바르면 target 분포를 보존하면서 Decode iteration 수를 줄일 수 있다.
4. **Speculative Decoding은 낮은 batch의 memory-bound Decode에서 유리하다.** 이미 큰 batch로 GPU가 compute-bound라면 draft 실행과 검증 overhead 때문에 효과가 줄거나 역효과가 날 수 있다.
5. **Acceptance rate와 accepted tokens per target step이 핵심 지표다.** draft가 빠르더라도 target과 분포가 맞지 않으면 거부가 많아져 이득이 사라진다.
6. **Data Parallelism(DP)은 모델 replica를 늘린다.** 각 replica에 전체 모델이 들어가야 하지만 요청 처리량과 장애 격리에 유리하다.
7. **Tensor Parallelism(TP)은 한 layer의 tensor를 여러 GPU에 나눈다.** 단일 replica의 메모리 한계와 latency를 줄일 수 있지만 layer마다 collective 통신이 발생한다.
8. **Pipeline Parallelism(PP)은 layer 구간을 stage로 나눈다.** TP보다 통신 빈도가 낮을 수 있지만 pipeline bubble과 stage 불균형을 관리해야 한다.
9. **Expert Parallelism(EP)은 MoE expert를 여러 GPU에 배치한다.** token dispatch/combine의 All-to-All 통신과 hot expert 불균형이 새로운 병목이다.
10. **병렬화는 GPU 수가 아니라 topology에서 시작한다.** NVLink 같은 빠른 node 내부 연결과 PCIe·InfiniBand·RoCE 같은 경계를 구분하지 않으면 GPU를 늘리고도 느려질 수 있다.
11. **Prefill과 Decode는 자원 특성이 다르다.** Prefill은 긴 입력에서 compute-bound가 되기 쉽고, Decode는 작은 batch에서 memory-bandwidth-bound가 되기 쉽다.
12. **P/D Disaggregation은 두 단계를 서로 다른 worker pool로 분리한다.** TTFT와 ITL, hardware와 autoscaling을 독립적으로 튜닝할 수 있지만 KV Cache를 빠르게 전달해야 한다.
13. **P/D 분리의 손익분기점은 KV 전송 비용이다.** routing·동기화·전송 시간이 분리로 줄인 간섭보다 크면 통합 배포보다 느리다.
14. **KV Cache는 일급 자원이다.** GPU에만 머무는 임시 buffer가 아니라 CPU·SSD·원격 저장소로 이동하고, 압축·축출·복원·공유되는 관리 대상이다.
15. **계층형 KV Cache는 용량과 latency를 교환한다.** GPU는 빠르지만 작고, CPU·SSD·remote tier로 갈수록 커지지만 복원 시간이 늘어난다.
16. **RAG와 CAG는 대체 관계가 아니다.** RAG는 최신성·검색 범위·인용에 유리하고, CAG는 반복되는 큰 context의 KV를 재사용해 TTFT를 줄이는 데 유리하다.
17. **전문 LLM serving framework는 단순한 inference wrapper가 아니다.** Scheduler, KV Cache manager, model executor, worker, distributed runtime과 optimized kernel을 포함한다.
18. **vLLM은 scheduling과 model execution을 분리한다.** Scheduler는 무엇을 언제 실행할지 결정하고, ModelExecutor·GPUWorker·ModelRunner는 실제 연산을 수행한다.
19. **vLLM Scheduler는 request가 아니라 token budget을 중심으로 동작한다.** 이 구조가 Continuous Batching, Chunked Prefill, Prefix Caching과 Speculative Decoding을 하나의 실행 loop에 통합한다.
20. **프레임워크 선택은 benchmark 1위가 아니라 적합성 문제다.** SLO, prompt/output 분포, hardware, structured output, 운영성, portability와 lock-in을 함께 평가해야 한다.

---

## 학습 범위

이 문서는 다음 내용을 하나의 흐름으로 통합했다.

- Speculative Decoding의 원리와 적용 조건
- DP·TP·PP·EP의 분할 단위와 통신 특성
- multi-GPU·multi-node topology와 병렬화 선택
- Prefill-Decode Disaggregation과 KV Cache 전송
- 긴 context에서 RAG·CAG의 trade-off
- KV Cache offloading·compression·blending
- kernel → engine → cache → orchestration으로 이어지는 serving stack
- 전문 LLM serving framework가 필요한 이유
- vLLM의 초기화·요청 실행·token scheduling 구조
- vLLM의 계층화된 최적화 전략
- TensorRT-LLM·SGLang·llama.cpp의 설계 방향
- workload와 운영 조건에 따른 framework 선택 기준
- R9700 32GB 환경에서의 적용 가능 범위
- 추후 검토할 실습 후보 목록

compute-bound·memory-bound, batching, KV Cache와 Prefix Caching의 기본 개념을 알고 있으면 이번 고급 기법이 어떤 문제를 확장해 해결하는지 연결하기 쉽다.

## 1. 고급 최적화를 관통하는 공통 원리

### 1.1 단일 GPU 최적화에서 시스템 최적화로

작은 모델을 단일 GPU에 배포할 때는 batching, quantization, PagedAttention과 kernel optimization만으로도 큰 효과를 얻을 수 있다. 그러나 모델이 한 GPU에 들어가지 않거나, 여러 tenant와 긴 context를 높은 처리량으로 서비스해야 하면 최적화 범위가 넓어진다.

```text
Application / Gateway
  → Routing & Orchestration
  → KV Cache Management
  → LLM Serving Engine
  → Model / Distributed Executor
  → Kernel
  → Accelerator & Interconnect
```

이 계층은 독립적이지 않다.

- 빠른 kernel이 있어도 scheduler가 작은 batch만 만들면 GPU를 채우지 못한다.
- KV Cache를 재사용해도 router가 매번 다른 replica로 보내면 cache hit가 사라진다.
- P/D를 분리해도 network가 KV 전송량을 감당하지 못하면 오히려 느려진다.
- GPU를 늘려도 collective 통신이 느리면 TP의 이득이 사라진다.

### 1.2 모든 최적화에는 handoff가 있다

| 최적화 | 얻는 것 | 새로 지불하는 것 |
| --- | --- | --- |
| Speculative Decoding | target Decode step 감소 | draft 실행, 검증, rejected token 낭비 |
| Tensor Parallelism | model shard, 단일 replica 연산 분산 | layer별 All-Reduce/All-Gather |
| Pipeline Parallelism | layer를 node/GPU에 분할 | activation 전달, pipeline bubble |
| Expert Parallelism | MoE expert 분산 | token dispatch/combine All-to-All |
| P/D Disaggregation | Prefill·Decode 독립 최적화 | KV Cache transfer와 routing |
| KV offloading | cache capacity 확대 | lower tier 저장·복원 latency |
| KV compression | 저장·전송 byte 감소 | encode/decode와 품질 검증 |

따라서 기능을 켰다는 사실보다 **절약한 시간과 새 overhead 중 어느 쪽이 큰지**를 측정해야 한다.

### 1.3 병목 분류가 먼저다

| 관찰된 병목 | 우선 검토할 방향 |
| --- | --- |
| 작은 batch Decode의 높은 ITL | batching, Speculative Decoding, weight/KV byte 축소 |
| 모델이 한 GPU에 들어가지 않음 | TP, PP, EP, quantization |
| replica 하나의 처리량 부족 | DP, routing, batching |
| 긴 Prefill이 Decode를 방해 | Chunked Prefill, P/D Disaggregation |
| 반복되는 긴 context의 TTFT | Prefix Caching, CAG, hierarchical KV Cache |
| KV Cache가 GPU에서 계속 축출 | CPU/SSD offloading, cache-aware routing |
| multi-GPU 확장 효율 저하 | topology, collective volume, load imbalance 분석 |

## 2. Speculative Decoding

### 2.1 해결하려는 문제

일반적인 Decode는 token 하나를 생성할 때마다 큰 target model의 forward pass를 실행한다.

```text
Target pass → 1 token
Target pass → 1 token
Target pass → 1 token
```

작은 batch에서는 매 step마다 weight를 읽으면서 token 하나만 생성하므로 memory bandwidth 활용 효율이 낮다. Speculative Decoding은 작은 draft가 여러 token을 먼저 제안하고 target이 한 번에 검증하도록 바꾼다.

```text
Draft:  t1, t2, t3, t4 제안
Target: t1~t4를 한 번의 병렬 pass로 검증
Result: 연속으로 수락된 token + 필요 시 target의 교정 token
```

### 2.2 동작 흐름

1. Draft model 또는 draft mechanism이 `K`개의 candidate token을 생성한다.
2. Target model이 candidate prefix 전체를 한 번에 평가한다.
3. 앞에서부터 token을 수락 또는 거부한다.
4. 첫 거부 이후의 candidate는 자기회귀 의존성이 깨지므로 폐기한다.
5. Target 분포에서 교정 token을 선택하고 다음 iteration을 시작한다.

표준 speculative sampling은 rejection sampling을 올바르게 구현하면 **target model의 확률 분포를 보존**한다. 이는 draft output을 그대로 믿는 근사 생성과 다르다. 다만 floating-point 비결정성, kernel과 sampling seed 차이 때문에 실행 결과 문자열이 매번 byte 단위로 동일하다는 뜻은 아니다.

### 2.3 Draft를 만드는 방법

| 방법 | 원리 | 장점 | 한계 |
| --- | --- | --- | --- |
| 별도 소형 model | 같은 계열의 작은 모델로 candidate 생성 | 이해와 적용이 단순 | 두 모델의 VRAM·실행 비용, 낮은 정렬 시 acceptance 감소 |
| Distilled draft | target 행동을 작은 model에 학습 | target과 높은 정렬 가능 | 별도 학습·artifact 관리 필요 |
| Self-drafting | target에 추가 prediction head/module 사용 | 별도 전체 model 부담 감소 | architecture·runtime 지원과 추가 학습 필요 |
| N-gram | prompt/output의 반복 token pattern 재사용 | 매우 낮은 overhead | 반복성이 낮은 자유 생성에는 약함 |

### 2.4 핵심 지표

- **Acceptance rate:** 제안 token 중 수락된 비율
- **Accepted tokens per target step:** target pass 한 번에 확정한 평균 token 수
- **Draft latency:** `K`개 candidate를 만드는 시간
- **Verification latency:** target이 candidate를 검증하는 시간
- **ITL/TPOT:** 사용자가 실제로 체감하는 token 간 지연
- **Output TPS:** 동시 요청 전체 처리량

단순한 손익 관계는 다음과 같이 생각할 수 있다.

```text
이득
≈ 줄어든 target Decode pass 비용
 - draft 생성 비용
 - target 검증 추가 비용
 - rejected candidate 낭비
```

### 2.5 `K`와 acceptance의 trade-off

- `K`가 너무 작으면 한 target pass에서 확정할 수 있는 token 수가 적다.
- `K`가 너무 크면 뒤쪽 candidate의 acceptance가 낮아지고 폐기 계산이 늘 수 있다.
- 최적 `K`는 model pair, prompt 유형, sampling parameter와 workload에 따라 다르다.
- 평균 acceptance만 보지 말고 candidate 위치별 acceptance를 확인해야 한다.

### 2.6 언제 유리한가

**유리할 가능성이 큰 조건**

- concurrency와 batch가 낮은 interactive serving
- output이 길고 Decode 비중이 큰 workload
- target과 draft의 분포가 잘 맞는 domain
- code, structured output, 반복 문장처럼 예측 가능한 생성
- target pass가 memory-bound이고 draft overhead가 작은 경우

**효과가 작거나 역효과가 날 수 있는 조건**

- 긴 prompt 중심으로 Prefill 비용이 지배하는 workload
- 이미 큰 batch로 target GPU가 compute-bound인 경우
- draft와 target의 tokenizer 또는 분포가 잘 맞지 않는 경우
- 매우 짧은 output
- draft까지 올리느라 KV Cache와 batch 공간이 줄어드는 경우

즉, Speculative Decoding은 항상 throughput을 높이는 기능이 아니라 **특정 Decode latency 문제를 해결하는 조건부 최적화**다.

## 3. Multi-GPU·Multi-Node 병렬화

### 3.1 먼저 구분할 두 가지 목표

병렬화 선택 전에 다음 질문을 분리해야 한다.

1. **모델이 한 GPU에 들어가지 않는가?**
   - model parallelism인 TP·PP·EP가 필요할 수 있다.
2. **모델은 들어가지만 요청 처리량이 부족한가?**
   - replica를 늘리는 DP가 우선일 수 있다.

모델이 한 GPU에 충분히 들어가는데 무조건 TP를 늘리면 collective 통신 때문에 replica 수를 늘리는 것보다 처리량이 낮아질 수 있다.

### 3.2 Data Parallelism

DP는 동일한 model replica를 여러 GPU에 복제하고 요청을 나눠 보낸다.

```text
Router
 ├─ Replica 0: full model
 ├─ Replica 1: full model
 └─ Replica 2: full model
```

**장점**

- replica끼리 token generation을 위해 매 layer 동기화할 필요가 없다.
- 요청 처리량을 수평 확장하기 쉽다.
- 장애가 난 replica를 routing에서 제외할 수 있다.
- 서로 다른 tenant나 priority pool을 격리하기 쉽다.

**한계**

- GPU마다 전체 model weight가 들어가야 한다.
- 각 replica가 독립 KV Cache를 가지므로 cache-aware routing이 필요하다.
- 부하가 고르게 분배되지 않으면 일부 replica는 queue가 쌓이고 다른 replica는 유휴 상태가 된다.
- API server나 load balancer 자체가 병목이 될 수 있다.

### 3.3 Tensor Parallelism

TP는 한 layer의 weight tensor를 여러 GPU에 나눠 각 GPU가 부분 연산을 수행하고 결과를 collective 통신으로 합친다.

```text
Layer N weight
 ├─ GPU 0 shard ┐
 ├─ GPU 1 shard ├─ partial compute → collective → full layer output
 └─ GPU 2 shard ┘
```

**장점**

- 하나의 GPU에 들어가지 않는 layer와 model을 분할할 수 있다.
- 충분히 빠른 연결에서는 단일 request latency를 줄일 가능성이 있다.
- 모든 GPU가 각 layer 연산에 참여하므로 pipeline bubble이 없다.

**한계**

- Transformer layer마다 All-Reduce 또는 All-Gather가 반복될 수 있다.
- TP degree가 커질수록 GPU당 연산은 줄지만 communication 비중은 커진다.
- node를 넘어가면 network latency와 bandwidth가 크게 영향을 준다.
- 작은 batch나 작은 model에서는 communication overhead가 연산 절감보다 클 수 있다.

일반적으로 TP는 빠른 node 내부 interconnect 범위에서 먼저 검토한다. “TP는 반드시 NVLink에서만 가능하다”는 절대 규칙은 아니지만, 통신량이 많은 특성 때문에 느린 PCIe나 node 간 network에서는 효율이 급격히 떨어질 수 있다.

### 3.4 Pipeline Parallelism

PP는 연속된 layer block을 stage로 나누고 activation을 다음 stage로 전달한다.

```text
GPU 0: Layers 0~N
   ↓ activation
GPU 1: Layers N+1~M
   ↓ activation
GPU 2: Layers M+1~Last
```

**장점**

- layer 전체를 한 GPU에 두므로 TP보다 collective 통신 빈도가 낮을 수 있다.
- 상대적으로 느린 interconnect 경계를 넘을 때 선택지가 될 수 있다.
- layer 단위로 model memory를 분산한다.

**한계**

- 요청 또는 microbatch가 stage를 채우기 전후에 pipeline bubble이 생긴다.
- stage별 연산량이 다르면 가장 느린 stage가 전체 속도를 제한한다.
- activation 전달과 scheduling이 복잡하다.
- 작은 online batch에서는 pipeline을 충분히 채우기 어렵다.

### 3.5 Expert Parallelism

MoE model은 모든 token에 전체 FFN을 실행하지 않고 router가 일부 expert만 선택한다. EP는 expert를 여러 GPU에 나눠 배치하고 token을 선택된 expert가 있는 GPU로 보낸다.

```text
Token batch
  → Router
  → All-to-All dispatch
  → Selected experts execute
  → All-to-All combine
```

**장점**

- 전체 expert parameter가 한 GPU에 들어가지 않는 문제를 해결한다.
- 선택되지 않은 expert의 연산을 피하면서 큰 model capacity를 유지한다.
- attention과 MoE layer에 서로 다른 병렬화 방식을 적용할 수 있다.

**한계**

- token dispatch/combine의 All-to-All 통신량이 크다.
- 특정 expert에 token이 몰리는 hot expert 문제가 생긴다.
- expert별 token 수가 너무 적으면 작은 matrix 연산이 되어 GPU 효율이 낮다.
- expert 배치, redundant expert와 load balancing이 추가로 필요할 수 있다.

MoE의 sparse activation이 곧바로 높은 효율을 보장하지 않는다. 충분한 batch가 있어야 각 expert가 처리하는 token 수가 커지고 연산 효율이 올라간다.

### 3.6 비교표

| 방식 | 무엇을 나누는가 | 각 GPU에 full model 필요 | 주요 통신 | 주된 목적 |
| --- | --- | --- | --- | --- |
| DP | Request와 replica | 예 | 일반적으로 request routing | 처리량·가용성 |
| TP | Layer 내부 tensor | 아니오 | 빈번한 collective | model fit·single-replica latency |
| PP | Layer stage | 아니오 | stage 간 activation | model fit·node 경계 확장 |
| EP | MoE expert | 아니오 | token All-to-All | 큰 MoE model 실행 |

실제 대형 배포는 DP×TP×PP와 EP를 조합한다. 단, “GPU 수를 모두 곱해 채우는 것”이 목표가 아니라 model architecture와 topology에 맞게 **통신량이 가장 큰 경로를 가장 빠른 연결 안에 가두는 것**이 목표다.

### 3.7 Topology를 먼저 기록해야 하는 이유

같은 GPU 8개라도 다음 구성은 전혀 다른 결과를 낼 수 있다.

- 하나의 switch로 연결된 8 GPU
- 두 개의 4-GPU island
- PCIe root complex가 나뉜 구성
- 두 node에 4개씩 배치되고 node 사이는 Ethernet인 구성
- node 사이는 InfiniBand/RoCE지만 oversubscription이 있는 구성

병렬화 benchmark에는 GPU model뿐 아니라 다음 정보를 함께 기록해야 한다.

- GPU 간 topology와 link 종류
- node 내부·node 간 실측 bandwidth와 latency
- NUMA·CPU affinity
- collective backend와 algorithm
- TP·PP·DP·EP degree
- model weight·KV Cache 배치

### 3.8 Cloud GPU 환경을 사용할 때

로컬에 multi-GPU가 없다면 Runpod 같은 GPU provider의 Pod·Serverless·Cluster 또는 AWS GPU EC2를 검증 환경으로 사용할 수 있다. 여기서 중요한 것은 provider 이름보다 **실제 할당된 hardware와 운영 경계**다.

- model weight와 KV Cache를 담을 VRAM이 충분한가
- GPU끼리 NVLink인지 PCIe인지, node 사이는 어떤 network인지
- container 안에서 topology·collective·profiler를 어디까지 관찰할 수 있는가
- model download와 container image를 보존할 volume이 있는가
- spot/preemptible 종료가 benchmark와 cache warm 상태에 어떤 영향을 주는가
- API key·Hugging Face token을 image나 shell history에 남기지 않는가
- instance, volume, public endpoint와 load balancer를 종료했는가

AWS의 GPU quota와 image ID처럼 region·시점에 따라 달라지는 값은 문서에서 복사해 고정하지 않고 현재 console/API로 확인한다. Serverless 환경은 scale-to-zero와 cold start를 포함해 평가하고, multi-node cluster는 provider가 표시한 GPU 수만이 아니라 실제 collective bandwidth를 확인해야 한다.

## 4. Prefill-Decode Disaggregation

### 4.1 왜 분리하는가

Prefill과 Decode는 서로 다른 hardware profile을 가진다.

| 단계 | 주된 작업 | 일반적 성격 | 주요 SLO |
| --- | --- | --- | --- |
| Prefill | prompt 전체를 병렬 처리하고 초기 KV 생성 | 긴 입력에서 compute-bound | TTFT |
| Decode | 기존 KV를 읽으며 token을 순차 생성 | 작은 batch에서 memory-bandwidth-bound | ITL/TPOT |

같은 GPU pool에서 두 단계를 처리하면 긴 Prefill이 Decode를 방해하거나, Decode 중심 설정 때문에 Prefill compute를 충분히 활용하지 못할 수 있다. Chunked Prefill은 간섭을 완화하지만 hardware와 scaling policy를 단계별로 완전히 분리하지는 못한다.

### 4.2 기본 구조

```text
Client
  → Gateway / Scheduler
  → Prefill Worker Pool
  → KV Cache Transfer
  → Decode Worker Pool
  → Streaming Response
```

1. Gateway가 request를 받고 Prefill worker를 선택한다.
2. Prefill worker가 prompt를 처리해 KV Cache를 만든다.
3. 생성된 KV block과 metadata를 Decode worker로 전달한다.
4. Decode worker가 이어서 token을 생성한다.

### 4.3 얻는 이점

- **TTFT와 ITL 독립 최적화:** input-heavy와 output-heavy workload에 서로 다른 자원 비율 적용
- **독립 batching:** Prefill은 compute utilization, Decode는 ITL과 memory bandwidth에 맞춰 batch 조정
- **이종 hardware:** compute 중심 GPU와 memory capacity/bandwidth 중심 GPU를 다르게 선택 가능
- **비대칭 scaling:** burst가 큰 Prefill pool과 지속 시간이 긴 Decode pool을 별도 확장
- **간섭 감소:** 긴 Prefill이 진행 중인 Decode의 tail latency를 밀어내는 현상 완화

### 4.4 가장 큰 비용: KV Cache transfer

분리 후에는 request마다 큰 KV state를 다른 worker로 옮겨야 한다.

```text
required KV bandwidth
≈ average KV bytes per request × prefill completions per second
```

여기에 serialization, metadata 교환, queueing, synchronization과 retry가 추가된다. 다음 조건을 만족하지 못하면 분리가 손해일 수 있다.

```text
줄어든 Prefill/Decode 간섭과 독립 scaling 이득
>
routing + KV transfer + synchronization + 추가 운영 비용
```

고성능 구현은 GPU Direct와 RDMA 같은 경로로 CPU copy를 줄이고 필요한 KV block만 전송한다. 그러나 특정 library 이름보다 중요한 것은 **실제 end-to-end transfer path와 fallback 여부를 확인하는 것**이다. 빠른 interconnect가 없으면 통합 배포가 더 단순하고 빠를 수 있다.

NIXL은 이런 inference data transfer를 위한 계층의 한 예로, GPU memory·host memory·storage와 UCX/RDMA 계열 backend를 추상화해 P/D worker 사이의 KV block 이동에 사용될 수 있다. 다만 library를 설정했다는 사실만으로 zero-copy가 보장되는 것은 아니다. 실제 선택 backend, GPU Direct 가능 여부와 TCP fallback을 log와 profiler로 검증해야 한다.

### 4.5 언제 고려하는가

**고려할 조건**

- 큰 model과 많은 replica를 운영하는 경우
- input/output token 비율이 workload별로 크게 다른 경우
- TTFT와 ITL을 독립적으로 엄격하게 관리해야 하는 경우
- Prefill·Decode pool을 각각 충분히 채울 traffic이 있는 경우
- KV transfer를 감당할 빠른 fabric과 observability가 있는 경우

**단순한 통합 배포가 나은 조건**

- 단일 GPU 또는 소규모 replica
- traffic이 적어 각 pool이 자주 유휴 상태가 되는 경우
- 짧은 context로 KV 전송보다 scheduling overhead가 큰 경우
- network가 느리거나 안정적인 GPU-to-GPU 전송 경로가 없는 경우

## 5. Advanced KV Caching

### 5.1 KV Cache를 일급 자원으로 보기

긴 context와 multi-tenant serving에서는 KV Cache가 weight만큼 중요한 운영 자원이 된다.

- 얼마나 생성되었는가
- 어느 tier에 저장되어 있는가
- 어떤 request·tenant가 소유하는가
- 언제 eviction되는가
- 다른 replica로 이동할 수 있는가
- 압축·복원했을 때 품질과 latency가 어떤가
- cache hit가 재계산보다 실제로 싼가

### 5.2 계층형 저장 구조

```text
GPU VRAM
  ↕ fastest / smallest
CPU DRAM
  ↕
Local SSD / NVMe
  ↕
Remote cache / distributed storage
  ↕ slowest / largest
```

상위 tier는 빠르지만 작고 비싸며, 하위 tier는 크지만 이동 시간이 길다. Cache manager는 access frequency, context size, tenant, TTL과 recomputation cost를 바탕으로 placement와 eviction을 결정해야 한다.

### 5.3 KV Cache Offloading

GPU에 모두 담지 못하는 KV block을 CPU나 SSD로 내리고 필요할 때 다시 올린다.

**이점**

- 더 많은 긴 context와 tenant의 cache 유지
- GPU eviction 후 전체 Prefill을 다시 하는 비용 감소
- GPU VRAM을 active sequence와 batch에 더 많이 사용

**비용**

- cache miss 때 lower tier에서 복원하는 latency
- PCIe·storage·network bandwidth 경쟁
- 비동기 prefetch와 eviction policy의 복잡도
- cold request에는 저장 overhead만 추가될 가능성

따라서 cache hit가 드문 random prompt에는 계층형 cache가 오히려 느릴 수 있다. **recompute cost, transfer cost, hit probability**를 함께 봐야 한다.

### 5.4 KV Cache Compression

KV를 낮은 precision으로 quantize하거나 전송용 encoding을 사용하면 저장 공간과 이동 byte를 줄일 수 있다.

```text
net benefit
≈ transfer time saved + capacity benefit
 - compression/decompression overhead
 - quality risk
```

압축률만 높아도 decoding kernel이 해당 format을 효율적으로 소비하지 못하면 전체 latency가 개선되지 않을 수 있다. 저장 format, wire format과 compute format을 구분한다.

### 5.5 KV Cache Blending

서로 다른 RAG chunk의 KV Cache를 단순히 이어 붙이면 self-attention의 chunk 간 관계가 계산되지 않아 올바른 전체 Prefill과 같지 않다. Blending 계열 기법은 기존 cache를 최대한 재사용하면서 일부 token이나 layer를 선택적으로 재계산해 cross-token dependency를 복원한다.

핵심 trade-off는 다음과 같다.

- 재계산 비율이 낮으면 빠르지만 품질 위험이 커질 수 있다.
- 재계산 비율이 높으면 품질은 안정되지만 전체 Prefill과 차이가 줄어든다.
- chunk 순서, query와 model architecture에 따라 필요한 재계산량이 달라진다.

### 5.6 LMCache가 담당하는 위치

LMCache는 serving engine 바깥 또는 connector 경계에서 KV Cache를 저장·이동·재사용하는 cache management layer의 예다. Engine의 token scheduler를 대체하는 것이 아니라 vLLM·SGLang 같은 engine이 GPU 밖의 KV tier를 활용할 수 있도록 연결한다.

```text
Serving Engine
  ↕ KV connector
LMCache / Cache Service
  ├─ CPU memory tier
  ├─ local SSD tier
  └─ distributed/remote tier
```

도입 효과는 workload에 따라 달라진다.

- cold miss에서는 cache 저장·metadata 처리 때문에 기본 engine보다 느릴 수 있다.
- GPU cache가 포화된 뒤 반복되는 긴 prefix에서는 CPU/SSD 복원이 전체 Prefill 재계산보다 빠를 수 있다.
- concurrency가 높을수록 active KV와 장기 cache가 같은 VRAM을 경쟁하므로 offloading 가치가 커질 수 있다.
- engine version, connector와 cache server protocol이 맞지 않으면 기능이 동작하지 않거나 fallback될 수 있다.

CacheBlend 같은 기능은 비연속 RAG chunk의 KV를 재사용하려는 고급 기법이며, 일반적인 exact prefix hit와 같은 correctness 조건으로 보아서는 안 된다. 선택적 재계산 비율과 품질 검증이 필요하다.

## 6. RAG와 CAG

### 6.1 기본 차이

| 항목 | RAG | CAG |
| --- | --- | --- |
| Context 구성 | query마다 관련 chunk 검색 | 큰 정적 context를 cache하고 재사용 |
| 최신성 | index 갱신으로 반영하기 쉬움 | cache invalidation과 재생성 필요 |
| Context 크기 | 선택된 일부 문서 | 전체 또는 매우 큰 지식 묶음 |
| TTFT | 검색 + 동적 Prefill 비용 | warm cache hit이면 매우 낮을 수 있음 |
| 설명 가능성 | 검색 문서와 citation 추적에 유리 | 전체 context 안의 근거 추적이 어려울 수 있음 |
| 비용 요인 | embedding, index, retrieval, Prefill | cache build·storage·cached-token 처리 |

CAG는 Cache-Augmented Generation을 뜻하며, 정적 지식을 큰 prefix로 두고 KV를 반복 재사용하는 방식은 가장 단순한 형태다.

### 6.2 CAG가 유리할 수 있는 조건

- 동일한 긴 context를 많은 request가 반복 사용
- 지식 변경 빈도가 낮음
- TTFT가 비용보다 중요함
- model의 long-context 품질이 충분함
- cache를 오래 유지할 memory tier와 routing이 있음

### 6.3 RAG가 계속 중요한 이유

- 지식 base가 model context window보다 훨씬 큼
- 정보가 자주 바뀌어 cache invalidation이 비쌈
- tenant별 접근 제어와 문서 citation이 중요함
- query마다 필요한 정보가 전체 중 일부에 불과함
- cache hit가 낮아 긴 context build 비용을 회수하기 어려움

“CAG가 RAG보다 빠르다”와 “CAG가 더 싸다”는 다른 주장이다. Cache된 token도 저장·전송·attention 비용이 있으며, 매우 큰 context를 항상 유지하면 TTFT는 줄어도 총 비용이 커질 수 있다.

### 6.4 Hybrid 접근

실제 시스템은 둘을 조합할 수 있다.

```text
Stable system/tool context → long-lived KV cache
Tenant common context      → tenant-scoped cache
Frequently changing data   → RAG retrieval
User-specific question     → dynamic suffix
```

이때 cache key, document ordering, tenant isolation, TTL과 invalidation이 correctness의 일부가 된다.

## 7. Serving Stack의 계층

### 7.1 Engine과 Orchestrator는 다르다

```text
Orchestration tier
  - fleet routing
  - replica/pool selection
  - P/D placement
  - autoscaling and global policy

Engine tier
  - token scheduling
  - batching
  - KV block management
  - model execution
  - token generation
```

Orchestrator는 요청을 올바른 engine으로 보내지만 직접 token을 생성하지 않는다. 반대로 engine은 하나의 replica나 worker group 내부 실행을 책임지지만 전체 fleet의 배치와 scaling을 모두 해결하지는 않는다.

NVIDIA Dynamo와 Kubernetes-native llm-d는 orchestration tier의 예이며, vLLM·SGLang 같은 engine과 조합해 routing, P/D pool 선택과 fleet-level policy를 구성할 수 있다. 이름이 비슷한 기능을 제공하더라도 **engine이 token을 계산하고 orchestrator가 배치·경로를 결정한다는 책임 경계**는 유지된다.

### 7.2 네 계층으로 보는 최적화

| 계층 | 책임 | 예시 |
| --- | --- | --- |
| Kernel | hardware 연산·메모리 접근 | fused op, attention kernel, quantized GEMM |
| Serving Engine | token scheduling·model execution | vLLM, SGLang, TensorRT-LLM, llama.cpp server |
| Cache Management | KV placement·transfer·reuse | local block manager, hierarchical cache |
| Orchestration | fleet routing·pool scaling | gateway, cache-aware router, P/D controller |

한 계층에서 해결할 문제를 다른 계층에 억지로 넣으면 결합도가 높아진다. 예를 들어 GPU별 kernel 선택은 router가 아니라 engine/kernel 계층이 담당하고, replica queue와 cache locality를 이용한 요청 배치는 fleet router가 담당하는 편이 자연스럽다.

## 8. 전문 LLM Serving Framework가 필요한 이유

전통적인 inference server는 주로 고정 shape 입력을 batch로 묶고 한 번의 forward pass로 결과를 반환하는 workload에 맞춰졌다. LLM은 다음 특성이 다르다.

1. **Autoregressive generation:** request가 여러 Decode step 동안 살아 있다.
2. **Variable context:** 입력과 출력 길이 편차가 매우 크다.
3. **KV Cache:** request마다 증가하는 상태를 GPU memory에서 관리해야 한다.
4. **Continuous Batching:** 완료된 sequence를 제거하고 새 request를 iteration 중 투입해야 한다.
5. **Streaming:** token을 생성 즉시 올바른 client에 전달해야 한다.
6. **Token-level fairness:** 긴 request가 짧은 request를 과도하게 막지 않게 해야 한다.
7. **Distributed execution:** model shard, replica와 cache를 여러 GPU/node에서 조율해야 한다.

따라서 전문 framework는 단순한 `model.generate()` wrapper가 아니라 다음 runtime 기능을 포함한다.

```text
API / Input Processor
→ Request lifecycle
→ Scheduler
→ KV Cache Manager
→ Model Executor
→ Worker / Model Runner
→ Optimized Kernels
→ Output Processor / Streaming
```

## 9. vLLM 내부 구조

> vLLM 내부 class 이름과 세부 동작은 version에 따라 바뀔 수 있다. 여기서는 책임 경계를 이해하는 데 집중하며, 실제 구현을 확인할 때는 설치한 version의 코드와 공식 설계 문서를 기준으로 삼는다.

### 9.1 두 가지 사용 방식

| 방식 | 특징 | 적합한 용도 |
| --- | --- | --- |
| Python `LLM` class | application process 안에서 직접 호출 | offline batch, 평가, 단순 pipeline |
| API server | OpenAI-compatible HTTP, multi-client, streaming | online serving, 독립 배포 |

두 방식 모두 내부 engine을 사용하지만 network layer와 request lifecycle의 범위가 다르다.

### 9.2 핵심 component

```text
LLMEngine
  └─ EngineCore
       └─ Scheduler
            └─ SchedulerOutput
                 └─ ModelExecutor
                      └─ GPUWorker
                           └─ GPUModelRunner
```

| Component | 책임 |
| --- | --- |
| LLMEngine | 공개 API와 request lifecycle 관리 |
| EngineCore | scheduling과 execution loop 조율 |
| Scheduler | request 우선순위, token budget, KV block 할당 |
| SchedulerOutput | 다음 forward pass의 실행 계획과 metadata |
| ModelExecutor | worker process와 distributed execution 조율 |
| GPUWorker | device·process·model lifecycle 관리 |
| GPUModelRunner | input 준비와 실제 model forward 실행 |

핵심은 **Scheduler가 model layer의 세부 연산을 몰라도 실행 계획을 만들고, Worker는 queue policy를 몰라도 주어진 계획을 실행**한다는 점이다.

### 9.3 초기화 흐름

```text
1. Main process
   → config, Engine, Scheduler, KV manager, Executor 초기화
2. Executor
   → worker process/group 생성
3. Worker
   → device 설정, distributed communication 초기화
4. Model Runner
   → model implementation 선택, weight load, runtime 준비
```

단일 node multi-GPU는 local multiprocess 실행이 자연스러울 수 있고, multi-node에서는 별도 distributed backend가 필요하다. 중요한 점은 backend 이름을 외우는 것이 아니라 **control process, worker group, device와 communication 초기화의 경계**를 이해하는 것이다.

### 9.4 생성 request 실행 흐름

```text
Raw input
  → Processor: 검증·tokenization·Request 생성
  → LLMEngine / EngineCore loop
  → Scheduler: 이번 step의 request·token·KV block 결정
  → ModelExecutor / Worker: forward pass
  → Output Processor: sampling·detokenization·stream routing
  → Client
```

한 번의 request가 이 흐름을 한 번만 통과하는 것이 아니다. Decode가 끝날 때까지 scheduling과 execution loop를 반복한다.

## 10. vLLM Scheduler

### 10.1 중앙 교통 관제 역할

Scheduler는 다음 자원을 함께 계산한다.

- WAITING과 RUNNING request
- scheduling iteration의 token budget
- 사용 가능한 KV Cache block
- maximum active sequence와 model length
- cached prefix와 external KV availability
- speculative token
- multimodal encoder budget
- priority, fairness와 preemption

### 10.2 Request 순서와 token 수를 분리한다

두 결정은 서로 다르다.

1. **어느 request를 먼저 볼 것인가**
   - FCFS, priority와 lifecycle state를 이용한다.
2. **선택한 request에서 이번 step에 몇 token을 처리할 것인가**
   - token budget, KV Cache와 이미 계산된 token 수를 이용한다.

개념적으로 다음 간극을 줄인다.

```text
remaining work
≈ tokens that should exist
 - tokens already computed or restored from cache
```

이 구조 덕분에 같은 queue policy 위에 Chunked Prefill, Prefix Caching과 Speculative Decoding을 독립적으로 조합할 수 있다.

### 10.3 Scheduling cycle

1. 새 request, 재개 request와 현재 RUNNING state를 수집한다.
2. token·KV·encoder budget을 갱신한다.
3. RUNNING request가 계속 진행할 수 있는지 계산한다.
4. 남은 budget으로 WAITING request를 admission한다.
5. cache hit, chunk limit, speculative token과 adapter 정보를 반영한다.
6. request별 token 수와 KV metadata를 `SchedulerOutput`으로 만든다.
7. Executor가 계획대로 forward pass를 실행한다.
8. 완료·선점·추가 output을 반영해 다음 cycle로 돌아간다.

RUNNING request를 무조건 영구 우선하면 WAITING starvation이 생길 수 있고, WAITING request를 너무 공격적으로 넣으면 active KV Cache가 부족해진다. 따라서 throughput뿐 아니라 fairness, preemption cost와 tail latency가 scheduler 품질을 결정한다.

### 10.4 최적화가 Scheduler에 연결되는 방식

- **Chunked Prefill:** 긴 Prefill의 이번 iteration token 수를 제한한다.
- **Prefix Caching:** 이미 계산된 block만큼 remaining Prefill을 줄인다.
- **Speculative Decoding:** 검증할 speculative token을 token plan에 포함한다.
- **P/D Disaggregation:** 외부 worker의 KV availability와 transfer state를 반영한다.
- **Guided/Structured Decoding:** 허용 token 집합을 sampling 단계와 연결한다.

Scheduler가 model-agnostic해야 새로운 model architecture가 추가될 때 queue와 token accounting 전체를 다시 만들지 않아도 된다.

## 11. vLLM의 계층화된 최적화

| 계층 | 범위 | 대표 책임 |
| --- | --- | --- |
| Scheduler | 시스템 전체, model-agnostic | batching, fairness, token·KV budget |
| ModelExecutor | model architecture·distributed execution | worker 조율, architecture별 실행 경로 |
| Model layer | attention·FFN 등 component | KV reuse, fused layer, attention implementation |
| CustomOp / Kernel | hardware-specific | GPU kernel, matrix core, quantized operator |

위로 갈수록 범용 policy에 가깝고 아래로 갈수록 model과 hardware에 특화된다.

이 분리의 장점은 다음과 같다.

- 새 GPU 지원 시 상위 scheduler를 바꾸지 않고 kernel/backend를 추가할 수 있다.
- 새 model architecture 지원 시 global request policy와 분리해 구현할 수 있다.
- Prefix Caching 같은 시스템 기능과 특정 attention kernel을 독립적으로 개선할 수 있다.
- 같은 API와 scheduler 아래에서 여러 device backend를 선택할 수 있다.

다만 실제 성능 문제는 계층을 가로질러 나타난다. 예를 들어 scheduler가 큰 batch를 만들더라도 kernel이 해당 shape에 비효율적일 수 있으므로 profiling은 end-to-end와 layer/kernel 수준을 함께 봐야 한다.

## 12. 다른 Serving Framework

프레임워크의 지원 hardware, model, quantization과 API는 빠르게 변한다. 아래 내용은 설계 방향을 비교하기 위한 것이며 실제 도입 전에는 **사용할 version의 공식 문서와 compatibility matrix**를 확인해야 한다.

### 12.1 TensorRT-LLM

TensorRT-LLM은 NVIDIA GPU에서 높은 inference 효율을 목표로 하는 최적화 stack이다.

**주요 방향**

- NVIDIA hardware와 CUDA/TensorRT 생태계에 깊게 최적화
- model을 최적화된 engine/runtime path로 변환·실행
- in-flight batching, paged KV Cache, quantization과 distributed parallelism 지원
- NVIDIA의 serving·orchestration 제품군과 통합

**적합한 경우**

- hardware fleet가 NVIDIA로 표준화됨
- compile/build와 engine artifact 운영을 감수할 수 있음
- 특정 model·shape에서 최대 효율이 중요함
- NVIDIA 전용 kernel과 precision 기능을 적극 사용함

**주의점**

- AMD·CPU 같은 다른 backend로의 portability가 목표가 아니다.
- model 변경 시 build·compatibility 검증 비용이 클 수 있다.
- 최고 peak 성능뿐 아니라 build time, cold start와 upgrade 운영성을 함께 봐야 한다.

### 12.2 SGLang

SGLang은 빠른 backend runtime과 generation workflow·structured output을 함께 설계한 serving framework다.

**주요 방향**

- Radix tree 기반 prefix/KV reuse
- Continuous Batching, Paged KV와 Speculative Decoding
- JSON, regex, grammar 등 constrained/structured generation
- agent·multi-turn·multi-stage workflow에 유용한 frontend와 API
- 여러 형태의 distributed parallelism과 routing

**적합한 경우**

- 반복 prefix가 많은 agent 또는 multi-turn workload
- 구조화된 output의 correctness와 속도가 중요함
- workflow 표현과 inference runtime을 함께 최적화하고 싶음
- vLLM과 다른 scheduling·cache 전략을 평가하려는 경우

**주의점**

- 지원 model·hardware·kernel은 version마다 다르다.
- RadixAttention이라는 구현명과 일반적인 prefix caching 개념을 구분해야 한다.
- 구조화 출력 기능이 많아도 실제 workload에서 compile·FSM overhead와 token 절감을 함께 측정해야 한다.

### 12.3 llama.cpp

llama.cpp는 가벼운 C/C++ runtime과 GGUF ecosystem을 중심으로 다양한 local·edge hardware에서 LLM을 실행하는 데 강점이 있다.

**주요 방향**

- 적은 dependency와 빠른 시작
- GGUF model format과 다양한 integer quantization
- CPU SIMD를 기본으로 Metal, CUDA, ROCm, Vulkan 등 선택적 backend
- CLI와 OpenAI-compatible local server
- 낮은 동시성의 local·offline·private inference

**적합한 경우**

- laptop, workstation, edge 또는 on-premise 소형 server
- cloud로 data를 보내기 어려운 private workload
- GPU가 없거나 CPU/GPU hybrid offload가 필요한 경우
- 최대 fleet throughput보다 단순성과 footprint가 중요한 경우

**주의점**

- GGUF quantization level별 품질과 memory를 확인해야 한다.
- high-concurrency data-center serving에서는 전문 GPU engine과 목표가 다르다.
- GPU offload layer 수와 backend에 따라 성능 특성이 크게 달라진다.

## 13. 프레임워크 비교

| 관점 | vLLM | TensorRT-LLM | SGLang | llama.cpp |
| --- | --- | --- | --- | --- |
| 주된 목표 | 범용 high-throughput LLM serving | NVIDIA GPU 최대 효율 | high-performance serving + structured/agent workflow | portable local·edge inference |
| 실행 중심 | Python ecosystem, scheduler·KV engine | TensorRT/CUDA 최적화 runtime | runtime와 generation frontend 공동 설계 | 경량 C/C++ runtime |
| 대표 cache 관점 | Paged KV, hash 기반 prefix reuse | Paged KV와 NVIDIA 최적화 | Radix tree 기반 prefix reuse | context cache와 local memory 관리 |
| Hardware 방향 | 여러 accelerator backend 지향 | NVIDIA 중심 | 여러 backend 지향, version별 확인 필요 | CPU와 다양한 선택 backend |
| 구조화 출력 | 지원 기능을 version별 확인 | 지원 기능을 version별 확인 | 주요 강점 중 하나 | 기본 server/runtime 범위에서 확인 |
| 대규모 분산 | TP·PP·DP·EP 및 ecosystem | NVIDIA stack과 강한 통합 | 병렬화·routing·P/D 기능 | 주목표가 아님 |
| Local/edge | 가능하지만 주목표는 server | 부적합 | 가능 여부를 환경별 확인 | 핵심 강점 |
| 주요 trade-off | 빠른 변화와 version별 복잡성 | vendor lock-in·build 운영 | 생태계 성숙도·호환성 확인 | 대규모 multi-tenant throughput 한계 |

이 표는 승자를 정하기 위한 것이 아니다. 같은 “OpenAI-compatible API”를 제공해도 내부 cache, scheduler, model format과 hardware dependency는 다르다.

## 14. 프레임워크 선택 절차

### 14.1 SLO부터 적는다

- TTFT p95/p99
- ITL/TPOT p95/p99
- input/output TPS 또는 request rate
- error·timeout·OOM 허용치
- model quality와 structured output correctness
- token당 또는 request당 비용
- availability와 recovery time

### 14.2 실제 workload를 분류한다

| 질문 | 선택에 미치는 영향 |
| --- | --- |
| Prefill과 Decode 중 무엇이 지배하는가 | cache/P-D/speculative 기능의 가치 |
| input/output 길이 분포는 어떤가 | scheduler와 memory pressure |
| prefix 반복률은 높은가 | prefix cache와 routing 전략 |
| 동시성은 어느 수준인가 | batching과 replica 구조 |
| JSON·grammar 제약이 필요한가 | structured decoding 성숙도 |
| vision·audio·LoRA를 사용하는가 | model feature compatibility |
| local/edge인가 data center인가 | framework와 model format 선택 |

### 14.3 동일 조건으로 비교한다

- 동일 model과 revision
- 동일 dtype·quantization artifact
- 동일 max context와 sampling parameter
- 동일 input/output dataset
- 동일 concurrency 또는 arrival rate
- 동일 warm/cold cache 조건
- 동일 streaming·client behavior
- 같은 hardware partition과 power state

### 14.4 운영성을 측정한다

성능 숫자 외에 다음을 확인한다.

- model load·compile·warm-up·server ready 시간
- logging, metrics, tracing과 profiling
- cancellation, timeout과 backpressure
- OOM·worker crash·node loss 복구
- rolling update와 model 교체 방식
- multi-tenant isolation과 fairness
- container image 크기와 dependency 관리
- security patch와 upgrade 주기

### 14.5 Lock-in과 exit plan

API compatibility만으로 framework 교체가 쉬워지는 것은 아니다. 다음 항목을 application에서 분리해야 한다.

- prompt template와 tokenizer assumption
- sampling·structured output option
- model/adapter lifecycle API
- metrics name과 autoscaling signal
- cache key와 routing metadata
- vendor-specific quantization artifact
- distributed deployment manifest

가능하면 공통 request contract 뒤에 framework adapter를 두고, benchmark와 correctness test를 교체 검증에 재사용한다.

## 15. R9700 32GB 환경에서의 해석

현재 환경은 AMD Radeon AI PRO R9700 32GB 단일 GPU이므로 학습 내용의 적용 범위가 나뉜다.

### 직접 검토하기 좋은 범위

- vLLM의 token scheduling과 cache behavior 관찰
- 낮은 concurrency에서 Speculative Decoding 지원 여부와 효과 비교
- Prefix Caching과 긴 context의 TTFT
- CPU memory를 이용한 KV offloading의 cold/warm 차이
- SGLang의 현재 ROCm·model 지원 여부 확인 후 단일 GPU 실행
- llama.cpp의 GGUF와 ROCm/Vulkan/CPU 경로 비교
- framework별 model load, TTFT, ITL, TPS와 VRAM 비교

### 현재 단일 GPU로 검증하기 어려운 범위

- TP·PP·DP의 multi-GPU 비교
- MoE Expert Parallelism과 All-to-All
- node 간 network topology와 collective 성능
- 독립 Prefill·Decode worker pool
- GPU Direct/RDMA 기반 KV transfer
- multi-replica cache-aware routing

TensorRT-LLM은 NVIDIA GPU 중심이므로 R9700에서 직접 실행할 대상이 아니다. 필요하다면 NVIDIA 환경에서 별도 검증하되, local vLLM 결과와 hardware 차이를 섞어 framework 차이로 해석하지 않아야 한다.

## 16. 측정할 때 주의할 점

### 16.1 Speculative Decoding

- vanilla와 speculative의 target model·sampling을 동일하게 유지한다.
- draft overhead, acceptance와 accepted token 수를 함께 기록한다.
- concurrency를 단계별로 높여 이득이 사라지는 지점을 찾는다.
- ITL뿐 아니라 total TPS와 VRAM 감소 여부도 확인한다.

### 16.2 병렬화

- GPU 수뿐 아니라 topology와 link traffic을 기록한다.
- TP와 DP를 비교할 때 replica당 batch와 global arrival rate를 구분한다.
- TTFT, ITL, total TPS와 GPU별 utilization 불균형을 함께 본다.
- model fit을 위한 병렬화와 성능 향상을 위한 병렬화를 구분한다.

### 16.3 KV Cache

- cold miss, GPU hit, CPU/SSD hit를 별도로 분리한다.
- cache build/store 시간도 포함한다.
- hit rate만이 아니라 재사용 token 수를 본다.
- eviction, restore와 recomputation 횟수를 기록한다.
- multi-tenant data isolation과 stale cache invalidation을 확인한다.

### 16.4 Framework 비교

- framework가 자동 선택한 kernel/backend를 startup log에서 확인한다.
- unsupported 기능의 fallback을 성능 결과와 함께 기록한다.
- client load generator가 병목이 아닌지 확인한다.
- 평균뿐 아니라 p95/p99와 error rate를 본다.
- peak TPS 직전의 latency cliff와 안정적인 운영점을 구분한다.

### 16.5 Monitoring과 profiling

관측 도구는 보는 계층이 다르다.

| 계층 | 확인할 내용 | 도구 유형 |
| --- | --- | --- |
| Service | request rate, TTFT, ITL, queue, error | application metrics, Prometheus/Grafana |
| Engine | scheduled token, cache hit, preemption, batch | engine log·metrics |
| GPU | utilization, memory, power, link traffic | vendor telemetry |
| Runtime timeline | CPU thread, kernel launch, communication overlap | system profiler |
| Kernel | occupancy, memory transaction, stall, tensor/matrix unit | kernel profiler |

NVIDIA Nsight Systems는 전체 timeline과 CPU/GPU overlap을, Nsight Compute는 개별 CUDA kernel의 세부 지표를 보는 대표적인 예다. AMD 환경에서는 해당 ROCm profiler 계열을 사용해야 한다. Service p99 문제를 곧바로 kernel 하나의 문제로 단정하지 말고, 상위 metric에서 병목 구간을 좁힌 뒤 더 낮은 계층으로 내려간다.

## 17. 추후 검증 계획

아래 항목은 이번 문서에서 수행하지 않았으며 결과도 포함하지 않는다. 실행 명령이나 구체적인 설정값은 정하지 않고 비교 기준, 핵심 지표와 필요한 환경만 기록한다.

| 검증 주제 | 비교 기준 | 핵심 지표 | 전제 조건 |
| --- | --- | --- | --- |
| Speculative Decoding | Vanilla vs 활성화 | ITL, output TPS, acceptance rate | 현재 runtime에서 지원되는 draft 방식 |
| 계층형 KV Cache | Cold miss vs GPU hit vs lower-tier hit | TTFT, cache hit, 저장·복원 시간 | R9700과 CPU memory에서 검토 가능 |
| Serving Framework | vLLM vs SGLang vs llama.cpp | TTFT, ITL, TPS, VRAM, 시작 시간 | 공통 model·dtype·workload 확보 |
| TP와 DP | 동일 GPU 수의 TP 구성 vs DP 구성 | TTFT, TPOT, total TPS, link traffic | Multi-GPU와 확인 가능한 topology |
| P/D Disaggregation | 통합 serving vs Prefill·Decode 분리 | TTFT, ITL, KV 전송량, 오류율 | 여러 worker와 충분히 빠른 interconnect |
| TensorRT-LLM | 동일 NVIDIA 환경의 vLLM vs TensorRT-LLM | 성능, build 시간, 시작 시간, 운영 복잡도 | NVIDIA GPU와 동일 model artifact |

## 18. 스스로 답해볼 핵심 질문

1. Speculative Decoding이 target model의 확률 분포를 유지하면서 빨라질 수 있는 이유는 무엇인가?
2. Acceptance rate가 높아도 Speculative Decoding이 느려질 수 있는 조건은 무엇인가?
3. model이 GPU 한 장에 들어갈 때 TP보다 DP가 처리량에 유리할 수 있는 이유는 무엇인가?
4. TP degree를 늘릴수록 GPU당 연산은 줄지만 전체 속도가 계속 좋아지지 않는 이유는 무엇인가?
5. PP의 통신 빈도가 TP보다 낮아도 online serving에서 느릴 수 있는 이유는 무엇인가?
6. MoE는 token당 일부 expert만 실행하는데도 큰 batch가 필요한 이유는 무엇인가?
7. EP에서 hot expert가 발생하면 어떤 GPU와 collective가 병목이 되는가?
8. P/D 분리가 Chunked Prefill과 근본적으로 다른 점은 무엇인가?
9. KV Cache 전송량으로 P/D 분리의 network 요구량을 어떻게 추정할 수 있는가?
10. KV를 CPU에 offload했을 때 GPU hit보다 느려도 전체 Prefill 재계산보다 유리할 수 있는 이유는 무엇인가?
11. 서로 다른 RAG chunk의 KV를 단순 concat하면 안 되는 이유는 무엇인가?
12. CAG가 RAG보다 TTFT는 낮지만 비용은 더 높을 수 있는 이유는 무엇인가?
13. Orchestrator와 serving engine은 각각 무엇을 책임지며 무엇을 대신하지 못하는가?
14. vLLM Scheduler와 GPUWorker를 분리한 설계가 새 model·hardware 지원에 어떤 이점을 주는가?
15. Request 우선순위와 token budget을 분리하면 어떤 최적화를 독립적으로 조합할 수 있는가?
16. 같은 OpenAI-compatible API를 제공해도 framework 교체가 어려울 수 있는 이유는 무엇인가?
17. R9700과 NVIDIA GPU에서 얻은 결과를 framework 차이로 바로 비교하면 안 되는 이유는 무엇인가?
18. 최고의 peak TPS보다 낮은 지점이 production 운영점으로 더 적합할 수 있는 이유는 무엇인가?

## 마무리

고급 LLM 서빙 최적화는 “GPU를 더 붙이면 빨라진다”거나 “새 기능을 켜면 지연이 줄어든다”는 단순한 문제가 아니다. Draft와 target 사이의 검증, GPU 사이의 collective, Prefill과 Decode 사이의 KV transfer, GPU와 CPU·SSD 사이의 cache 이동처럼 성능을 얻는 모든 지점에 새로운 handoff가 생긴다.

따라서 최적화 순서는 다음처럼 정리할 수 있다.

```text
SLO와 실제 workload 정의
→ compute·memory·capacity·communication·cache 병목 분류
→ 가장 작은 범위의 최적화 선택
→ 새 handoff 비용 측정
→ tail latency·처리량·품질·비용 함께 검증
→ 필요할 때만 분산·계층형 구조로 확장
```

프레임워크도 같은 기준으로 선택해야 한다. vLLM, TensorRT-LLM, SGLang과 llama.cpp는 모두 LLM을 실행하지만 목표 hardware, scheduler, cache 전략과 운영 철학이 다르다. 중요한 것은 영구적인 승자를 정하는 것이 아니라 application과 serving engine 사이의 경계를 유지하고, 현재 SLO와 hardware에 가장 잘 맞는 runtime을 재현 가능한 benchmark로 선택하는 것이다.
