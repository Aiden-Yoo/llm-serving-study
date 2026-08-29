# LLM 서빙 병목과 핵심 최적화: 하드웨어·스케줄링·압축·캐시

> 이 글의 목표는 LLM 서빙이 느리거나 GPU 메모리가 부족할 때 병목을 추측으로 판단하지 않고, **모델 로딩·GPU 메모리·연산·스케줄링·어텐션·모델 압축·프리픽스 캐시**의 관점에서 원인을 좁혀 적절한 최적화 방법을 선택할 수 있게 되는 것이다.
>
> **작성 상태:** LLM 서빙 병목과 필수 최적화 기법을 복습하기 쉽도록 하나의 문서로 재구성했다. 이번에는 별도 실습을 수행하지 않았으며, 문서에는 실험 결과를 포함하지 않는다. 관련 실습은 추후 진행할 후보 목록으로만 남겼다.

## 먼저 보는 핵심 요약

1. **최적화의 목표는 무조건 가장 빠른 응답이 아니다.** 사용자 경험을 만족하는 지연시간 안에서 처리량, 안정성, 모델 품질과 비용을 함께 최적화해야 한다.
2. **GPU의 TFLOPS만으로 서빙 성능을 설명할 수 없다.** VRAM 용량, 메모리 대역폭, 지원 정밀도, GPU 간 인터커넥트와 전력까지 함께 봐야 한다.
3. **모델 로딩과 모델 실행은 서로 다른 병목이다.** 로딩은 storage → CPU memory → GPU memory 경로와 초기화·컴파일의 영향을 받고, 실행은 GPU 메모리와 연산 유닛 사이의 데이터 이동에 크게 좌우된다.
4. **모델 가중치가 VRAM에 들어간다고 서빙 가능한 것은 아니다.** KV Cache, activation, 임시 buffer, runtime context와 메모리 단편화를 위한 여유가 추가로 필요하다.
5. **KV Cache는 동시 요청 수와 context 길이에 따라 커진다.** 특히 `num_key_value_heads`가 크고 긴 요청이 많을수록 GPU 메모리 압박이 커진다.
6. **산술 강도는 연산량을 데이터 이동량으로 나눈 값이다.** 이를 GPU의 peak compute와 memory bandwidth에 대입하면 compute-bound인지 memory-bound인지 판단할 수 있다.
7. **Prefill과 Decode는 성격이 다르다.** 긴 prompt의 Prefill은 compute-bound가 될 수 있지만, 작은 batch의 Decode는 일반적으로 memory-bandwidth-bound다.
8. **배칭은 Decode의 낮은 산술 강도를 높이는 핵심 수단이다.** 여러 요청이 같은 weight를 함께 사용하게 해 GPU 활용률과 처리량을 높인다.
9. **연속 배칭은 완료된 sequence 자리에 새 요청을 넣는다.** 길이가 다른 요청을 고정된 batch로 끝까지 묶는 낭비를 줄인다.
10. **Chunked Prefill은 긴 Prefill을 나눠 Decode와 함께 스케줄링한다.** 긴 prompt가 진행 중인 요청의 다음 토큰 생성을 오래 막는 현상을 줄이지만, TTFT·ITL·처리량 사이의 새로운 trade-off를 만든다.
11. **MHA → GQA/MQA/MLA의 흐름은 KV Cache를 줄이려는 모델 아키텍처의 진화다.** 이것은 보통 서버 실행 옵션이 아니라 모델을 선택할 때 함께 결정되는 특성이다.
12. **Kernel Fusion, FlashAttention, PagedAttention은 서로 다른 문제를 푼다.** 각각 중간 데이터 왕복, attention의 I/O, KV Cache 메모리 단편화를 줄인다.
13. **양자화는 모델을 작게 만들지만 항상 빨라지는 것은 아니다.** 저정밀 연산을 실제로 가속하는 kernel과 hardware가 있어야 지연시간·처리량 개선으로 이어진다.
14. **증류는 작은 모델을 새로 학습하는 방법이고, 가지치기는 불필요한 구조나 weight를 제거하는 방법이다.** 둘 다 양자화와 목적은 비슷하지만 비용과 적용 방식이 다르다.
15. **Prefix Caching은 반복되는 prompt 앞부분의 KV Cache를 재사용한다.** 주로 Prefill 연산과 TTFT를 줄이며, 서로 다른 prefix가 많은 workload에는 효과가 작다.
16. **복제본이 여러 개면 cache locality와 load balancing을 함께 고려해야 한다.** cache hit만 좇으면 특정 replica가 과부하될 수 있고, 부하만 균등화하면 prefix cache를 재사용하지 못한다.
17. **최적화는 반드시 동일 workload로 측정해야 한다.** 모델 revision, 정밀도, input/output 길이, 동시성, cache 상태와 runtime version이 다르면 숫자를 직접 비교하기 어렵다.

---

## 학습 범위

이 문서는 실제로 성능 문제를 진단하는 순서에 맞춰 다음 내용을 통합했다.

- 최적화가 필요한 이유와 SLO 관점의 목표 설정
- GPU 사양을 compute·memory·interconnect 관점에서 읽는 법
- 모델 로딩 경로와 cold start 병목
- weight와 KV Cache의 GPU 메모리 요구량 계산
- 산술 강도와 Roofline Model
- Prefill·Decode의 병목 차이
- Dynamic Batching, Continuous Batching, Chunked Prefill
- MHA·MQA·GQA·MLA와 KV Cache 절감
- Kernel Fusion, FlashAttention, PagedAttention
- Quantization, Distillation, Pruning
- Prefix Caching과 Cache-aware Routing
- Hugging Face 기본 추론에서 vLLM API·부하 시험·모니터링으로 이어지는 실습 흐름
- 추후 검토할 실습 후보 목록

Transformer, KV Cache, Prefill·Decode와 기본적인 서빙 시스템 구조를 이해하고 있으면 각 최적화가 어떤 병목을 해결하는지 연결하기 쉽다.

## 1. 무엇을 최적화할 것인가

LLM 서빙 최적화는 하나의 숫자를 최대화하는 문제가 아니다. 다음 목표는 서로 충돌할 수 있다.

| 목표 | 대표 지표 | 지나치게 우선했을 때의 문제 |
| --- | --- | --- |
| 사용자 경험 | E2E, TTFT, ITL/TPOT의 p95·p99 | 낮은 동시성만 허용해 GPU와 비용을 낭비할 수 있음 |
| 처리량 | request/s, input/output token/s | queueing과 요청 간 간섭으로 tail latency가 증가할 수 있음 |
| 비용 효율 | request당·token당 비용, GPU utilization | 품질이나 latency SLO를 희생할 수 있음 |
| 모델 품질 | task accuracy, pass rate, perplexity 등 | 더 큰 모델과 높은 정밀도로 비용·지연이 증가할 수 있음 |
| 확장성 | peak traffic 수용량, scale-out 시간 | 복잡도와 유휴 자원이 증가할 수 있음 |
| 안정성 | 오류율, timeout, OOM, availability | 지나치게 보수적인 설정으로 처리량이 낮아질 수 있음 |

중요한 지표의 의미는 다음과 같다.

- **E2E latency:** 요청 전송부터 마지막 응답 수신까지 걸린 전체 시간
- **TTFT(Time To First Token):** 요청 후 첫 토큰을 받기까지의 시간
- **ITL(Inter-Token Latency):** 스트리밍 중 인접한 출력 토큰 사이의 시간
- **TPOT(Time Per Output Token):** 첫 토큰 이후 출력 토큰 하나당 평균 시간
- **TPS:** 초당 처리한 token 수. input, output, total 중 무엇인지 구분해야 한다.
- **RPS:** 초당 완료한 request 수. 요청 길이가 다르면 RPS만으로 GPU 효율을 비교하기 어렵다.

평균만 보면 일부 사용자가 겪는 긴 지연을 숨길 수 있으므로 p50과 함께 p95·p99를 확인한다. 또한 TTFT를 사람이 구분하기 어려운 수준까지 낮추기 위해 처리량과 비용을 크게 희생하는 것이 항상 좋은 선택은 아니다. 먼저 목표 SLO를 정하고 그 범위 안에서 비용과 처리량을 최적화해야 한다.

## 2. GPU 사양을 서빙 관점에서 읽는 법

GPU를 비교할 때는 적어도 네 축을 함께 본다.

### 2.1 Compute

- FP32, FP16, BF16, FP8, INT8 등 **실제 사용할 precision의 peak performance**
- Tensor Core 또는 Matrix Core와 같은 행렬 연산 장치 지원 여부
- serving runtime과 kernel이 해당 장치를 실제로 사용할 수 있는지

서로 다른 precision의 peak 수치를 섞어 비교하면 안 된다. FP8 수치가 높아도 모델·runtime·kernel이 FP8을 지원하지 않으면 실제 workload에는 사용할 수 없다.

### 2.2 Memory capacity

VRAM에는 weight만 올라가는 것이 아니다.

```text
GPU memory
≈ model weights
 + KV Cache
 + activations
 + temporary/workspace buffers
 + runtime context and graph/compile artifacts
 + fragmentation and safety margin
```

용량은 **모델을 띄울 수 있는가**, **얼마나 긴 context와 많은 동시 요청을 받을 수 있는가**를 결정한다.

### 2.3 Memory bandwidth

모델 실행 중에는 GPU global memory에 있는 weight와 KV Cache를 on-chip cache·shared memory·register로 계속 옮긴다. 이 경로가 연산 속도를 따라가지 못하면 compute unit이 놀게 된다.

데이터센터 GPU는 주로 HBM을 사용하지만 R9700은 GDDR6 VRAM을 사용한다. 따라서 이 문서에서는 특정 메모리 기술을 가리킬 때를 제외하고 **GPU VRAM 또는 global memory**라는 표현을 사용한다. 핵심은 메모리 종류의 이름보다 **연산 유닛에 데이터를 공급할 수 있는 대역폭**이다.

### 2.4 Interconnect와 topology

모델을 여러 GPU 또는 여러 node에 나누면 통신이 새 병목이 될 수 있다.

- **Node 내부:** PCIe, 전용 GPU interconnect, switch topology
- **Node 사이:** InfiniBand, RoCE/RDMA와 같은 network fabric
- **확인할 점:** bandwidth뿐 아니라 hop, contention, collective 통신 패턴과 NUMA 배치

GPU 수를 늘렸다고 성능이 선형으로 증가하지 않는다. tensor parallel처럼 매 layer에서 자주 통신하는 방식은 느린 interconnect의 영향을 크게 받는다. 가능하다면 먼저 한 node 안에서 통신을 끝낼 수 있는지 검토하고, scale-out이 필요하면 compute와 communication이 겹칠 수 있는지도 측정한다.

### 2.5 Power와 cooling

전력과 냉각은 단순한 운영비 항목이 아니다. rack당 공급 가능한 전력과 발열 한계가 배치 가능한 GPU 수, 지속 clock, 전체 cluster capacity를 제한할 수 있다.

### 2.6 다른 accelerator와 Memory Wall

LLM inference 장치는 NVIDIA GPU만 있는 것이 아니다. AMD GPU, Google TPU, Intel Gaudi, AWS Inferentia와 여러 전용 accelerator도 compute·memory·interconnect를 서로 다른 방식으로 구성한다. 장치를 선택할 때는 peak 연산량뿐 아니라 다음 항목을 함께 비교해야 한다.

- 필요한 dtype과 model architecture를 지원하는가
- serving framework와 compiler·kernel 생태계가 충분한가
- VRAM/accelerator memory 용량과 bandwidth가 workload에 맞는가
- 단일 장치에서 끝나는지, 여러 장치의 collective 통신이 필요한지
- 공급 가능성, 전력, 운영 난도와 전체 비용은 어떤가

최근 accelerator의 compute 성능은 빠르게 증가하지만 memory capacity와 bandwidth, 장치 간 통신은 같은 비율로 늘기 어렵다. 연산 장치가 데이터를 기다리는 **Memory Wall**이 중요한 이유다. 이를 완화하는 방향은 크게 세 가지다.

1. 더 많은 on-chip SRAM/cache와 tiling으로 data locality를 높인다.
2. 고대역폭 memory와 빠른 interconnect로 데이터 공급 속도를 높인다.
3. quantization, GQA·MLA, fusion처럼 이동해야 할 byte 자체를 줄인다.

결국 accelerator 종류가 달라져도 “어디에서 byte가 이동하고, 그 이동량에 비해 얼마나 계산하는가”라는 진단 원리는 같다.

## 3. 모델 로딩과 cold start

모델이 요청을 처리할 준비가 되기까지의 경로는 다음과 같다.

```text
원격 저장소 또는 NFS
  → local storage/page cache
  → CPU memory
  → CPU-to-GPU transfer
  → GPU memory allocation
  → runtime initialization
  → kernel compile / graph capture / warm-up
  → Ready
```

각 단계가 서로 다른 병목을 만든다.

| 단계 | 가능한 병목 | 관찰 방법 |
| --- | --- | --- |
| 원격/NFS 읽기 | network bandwidth, metadata I/O, 공유 storage 경쟁 | 파일 읽기 시간, network throughput, cold/warm 차이 |
| local disk 읽기 | SSD bandwidth와 page cache 상태 | cache를 구분한 반복 측정 |
| CPU memory | 역직렬화, shard 병합, host RAM 부족 | CPU·RAM 사용량과 단계별 로그 |
| CPU → GPU | PCIe bandwidth, NUMA, pinned memory | transfer 구간과 GPU activity |
| GPU allocation | VRAM 부족과 단편화 | 시작 시 free/allocated/reserved memory |
| runtime 준비 | kernel compile, graph capture, tokenizer·engine init | server ready 전 단계별 timestamp |
| 첫 요청 | lazy compile, cache 생성, warm-up | 첫 요청과 이후 요청의 latency 차이 |

따라서 **model load time**, **server ready time**, **first-request latency**는 분리해 기록하는 편이 좋다. 모델 파일을 읽은 시간만 재고 cold start 전체라고 부르면 compile과 warm-up 비용을 놓친다.

현재 환경처럼 모델 원본을 NFS에 보관한다면 다음 세 조건이 의미 있다.

1. NFS cache가 비어 있는 첫 시작
2. OS page cache 또는 NFS client cache가 따뜻한 반복 시작
3. 동일 파일을 local SSD에 staging한 뒤 시작

세 조건을 나누면 storage가 느린지, runtime 초기화가 느린지 구분할 수 있다. 공개 문서에는 실제 mount 절대경로 대신 `./models/...` 같은 저장소 기준 상대경로만 기록한다.

## 4. GPU 메모리 요구량 계산

### 4.1 Weight memory

가장 단순한 이론적 추정식은 다음과 같다.

```text
weight bytes ≈ parameter count × bytes per parameter
```

| Precision | parameter당 이론적 크기 |
| --- | ---: |
| FP32 | 4 bytes |
| FP16 / BF16 | 2 bytes |
| FP8 / INT8 | 1 byte |
| INT4 / FP4 | 0.5 byte |

예를 들어 7B 모델을 BF16으로 저장하면 다음과 같다.

```text
7,000,000,000 × 2 bytes = 14 GB ≈ 13.0 GiB
```

이 값은 순수 weight의 근사치다. scale·zero point 같은 quantization metadata, tensor 정렬, 일부 고정밀 layer, tokenizer, runtime buffer는 별도다. 또한 제조사 표기의 GB와 운영체제가 흔히 표시하는 GiB를 구분해야 한다.

### 4.2 KV Cache memory

일반적인 Transformer에서 token 하나에 필요한 KV Cache의 근사치는 다음과 같다.

```text
KV bytes per token
≈ 2 × number of layers × number of KV heads × head dimension × bytes per element
```

- 앞의 `2`는 Key와 Value를 뜻한다.
- MHA에서는 `number of KV heads = number of attention heads`다.
- GQA/MQA에서는 반드시 `num_key_value_heads`를 사용해야 한다.
- MLA는 latent representation을 저장하므로 위 단순식만으로 직접 비교하기 어렵다.

Llama 2 7B MHA를 예로 들면 다음과 같다.

```text
2 × 32 layers × 32 KV heads × 128 head dimension × 2 bytes
= 524,288 bytes/token
= 0.5 MiB/token
```

활성 sequence 전체가 65,536 token을 cache한다면 이론상 약 32GiB가 필요하다.

```text
0.5 MiB/token × 4,096 tokens × 16 sequences ≈ 32 GiB
```

이 예시는 **weight보다 KV Cache가 더 커질 수 있음**을 보여준다. 다만 실제 사용량은 모든 요청이 최대 길이에 동시에 도달하는지, block size, prefix sharing, KV dtype과 cache eviction 정책에 따라 달라진다.

### 4.3 메모리 여유를 판단하는 법

“weight가 14GB이고 VRAM이 16GB이므로 된다”는 판단은 위험하다. 다음 순서로 계산해야 한다.

1. weight와 quantization metadata의 크기를 구한다.
2. runtime 시작 후 고정 overhead를 측정한다.
3. 목표 동시성과 input/output 길이 분포로 최대 cached token 수를 추정한다.
4. KV Cache와 temporary buffer를 더한다.
5. OOM과 fragmentation을 피할 safety margin을 둔다.

“weight의 두 배 VRAM부터 검토한다”는 경험칙은 초기 하드웨어 후보를 거르는 데는 유용하지만 보편 법칙은 아니다. GQA·MQA, 짧은 context, 작은 동시성에서는 여유가 클 수 있고, MHA·긴 context·prefix caching에서는 두 배도 부족할 수 있다.

R9700 32GB에서는 3B~7B급 BF16 모델로 scheduling·KV Cache 실험을 하기에 비교적 여유가 있지만, 최대 context와 동시성을 동시에 크게 올리면 32GB도 빠르게 소진될 수 있다. `max_model_len`을 지원 가능한 최대값 그대로 두기보다 실제 workload에 필요한 범위로 제한하는 것이 안전하다.

## 5. 산술 강도와 Roofline Model

### 5.1 산술 강도

산술 강도(Arithmetic Intensity, AI)는 데이터를 한 byte 옮길 때 얼마나 많은 연산을 수행하는지를 나타낸다.

```text
AI = FLOPs / bytes moved
```

- AI가 낮으면 데이터를 기다리는 시간이 커져 **memory-bandwidth-bound**가 되기 쉽다.
- AI가 높으면 연산 장치가 포화되어 **compute-bound**가 되기 쉽다.

### 5.2 Roofline Model

단순화된 Roofline Model에서 달성 가능한 성능은 다음 상한을 가진다.

```text
attainable performance
≤ min(peak compute, memory bandwidth × arithmetic intensity)
```

memory-bound와 compute-bound가 바뀌는 지점은 다음과 같다.

```text
ridge point = peak compute / memory bandwidth
```

같은 workload도 GPU가 달라지면 ridge point가 달라져 병목 판정이 바뀔 수 있다. 또한 이 모델은 peak 수치와 이상적인 데이터 재사용을 이용한 상한 모델이므로 실제 kernel launch overhead, cache miss, synchronization과 통신은 별도로 확인해야 한다.

### 5.3 행렬 곱셈의 산술 강도

`[M, K] × [K, N] → [M, N]` 행렬 곱셈에서 연산량과 최소 데이터 이동을 단순화하면 다음과 같다.

```text
FLOPs ≈ 2MKN
bytes moved ≈ b(MK + KN + MN)

AI ≈ 2MKN / b(MK + KN + MN)
```

`b`는 element당 byte 수다. 실제 구현에서는 cache reuse, tiling, fusion, intermediate write와 quantization/dequantization 때문에 값이 달라질 수 있지만, 행렬의 row 수가 커질수록 weight를 더 많이 재사용해 AI를 높일 수 있다는 직관은 유효하다.

### 5.4 Prefill과 Decode

| 구분 | 입력 형태의 직관 | 일반적인 병목 | 중요한 지표 |
| --- | --- | --- | --- |
| Prefill | prompt token 여러 개를 병렬 처리 | 짧으면 memory-bound, 길거나 batch가 크면 compute-bound 가능 | TTFT, input TPS |
| Decode | sequence마다 한 step에 token 하나 생성 | 작은 batch에서는 대체로 memory-bandwidth-bound | ITL/TPOT, output TPS |

Decode에서 batch가 1이면 큰 weight를 읽어 token 하나만 만든다. 반면 여러 sequence를 함께 decode하면 같은 weight를 여러 row 계산에 재사용해 산술 강도를 높일 수 있다. 따라서 “Decode는 항상 memory-bound”라기보다 **작은 batch의 Decode가 강하게 memory-bound이고, batching이 이를 완화한다**고 이해하는 편이 정확하다.

병목에 따라 최적화 방향도 달라진다.

| 병목 | 우선 검토할 방향 |
| --- | --- |
| Compute-bound | FLOPs 절감, 낮은 precision, 더 효율적인 kernel, model architecture 변경, compute parallelism |
| Memory-bandwidth-bound | weight/KV byte 축소, batching으로 재사용 증가, kernel fusion, I/O-aware attention, cache locality |
| Memory-capacity-bound | quantization, GQA/MQA/MLA 모델, PagedAttention, context·동시성 제한, KV dtype 조정 |
| Communication-bound | parallelism 축소·변경, topology-aware placement, 통신/연산 overlap, 더 빠른 interconnect |
| Storage/startup-bound | local staging, shard·format 최적화, image/model cache, preloading, warm pool |

## 6. Batching과 scheduling

### 6.1 Static, Dynamic, Continuous Batching

| 방식 | 동작 | 장점 | 한계 |
| --- | --- | --- | --- |
| Static batching | 미리 만든 batch를 끝까지 함께 처리 | 단순하고 offline 처리에 적합 | 길이 차이가 크면 완료된 slot이 낭비됨 |
| Dynamic batching | 최대 batch 크기 또는 최대 대기시간까지 도착 요청을 모음 | 일반적인 online inference에 적용 가능 | 생성 길이가 다른 LLM에서는 batch 내부 낭비가 남음 |
| Continuous batching | 매 iteration에 완료 sequence를 빼고 새 sequence를 투입 | GPU 유휴 시간과 head-of-line blocking을 줄임 | scheduler와 KV Cache 관리가 복잡함 |

Dynamic batching의 핵심은 **batch를 키우기 위해 얼마나 기다릴 것인가**다. 요청이 적을 때 무조건 최대 batch를 채우려 하면 latency가 커지므로 `max batch size`와 `max delay`를 함께 사용한다.

Continuous batching은 iteration-level 또는 inflight batching이라고도 한다. LLM은 출력 길이를 미리 알 수 없으므로 요청 A가 끝난 뒤에도 B와 C가 계속 생성할 수 있다. 완료된 A의 자리에 D를 넣어 batch를 다시 구성하면 GPU를 더 꾸준히 사용할 수 있다.

### 6.2 vLLM의 주요 scheduler 한도

정확한 기본값과 flag 지원 여부는 vLLM version마다 달라질 수 있으므로 현재 환경의 `vllm serve --help`와 server startup log를 기준으로 기록한다.

| 설정 | 의미 | 너무 작을 때 | 너무 클 때 |
| --- | --- | --- | --- |
| `max_num_seqs` | 동시에 scheduler가 다룰 sequence 수의 상한 | concurrency와 batch 효과가 제한됨 | KV Cache 압박과 요청 간 간섭 증가 |
| `max_model_len` | sequence 하나의 최대 context 길이 | 필요한 긴 요청을 거부·절단 | 실제로 쓰지 않는 최대 길이를 위해 운영 여유가 줄 수 있음 |
| `max_num_batched_tokens` | 한 scheduling iteration에서 처리할 token 예산 | GPU 활용률과 Prefill 진척이 낮아질 수 있음 | 긴 Prefill이 decode를 방해하거나 메모리 압박 증가 가능 |

세 값은 독립적인 knob가 아니다.

```text
scheduled sequences ≤ max_num_seqs
tokens in one sequence ≤ max_model_len
scheduled tokens per iteration ≤ max_num_batched_tokens
```

실제 동시 수용량은 이 한도뿐 아니라 남은 KV block, 각 요청의 현재 길이, 예상하지 못한 output 증가와 scheduling policy가 함께 결정한다.

## 7. Chunked Prefill

긴 prompt의 Prefill은 한 번에 많은 token을 처리한다. 이를 큰 단위로 독점 실행하면 이미 생성 중인 요청의 Decode가 기다려 ITL이 급격히 나빠질 수 있다.

Chunked Prefill은 긴 Prefill을 여러 token chunk로 나누고 Decode와 같은 iteration들에 섞는다.

```text
Without chunking
Long Prefill ──────────────────→ Decode requests wait

With chunking
Prefill chunk → Decode → Prefill chunk → Decode → ...
```

기대 효과는 다음과 같다.

- 긴 prompt가 다른 요청을 오래 막는 현상 완화
- Decode의 ITL과 tail latency 안정화
- token budget의 남는 부분에 Prefill을 채워 활용률 개선 가능

반면 chunk가 너무 작으면 scheduling과 kernel launch overhead가 늘고, 하나의 prompt를 끝내는 시간이 길어져 TTFT가 악화될 수 있다. 너무 크면 chunking하지 않은 것과 비슷해져 Decode 간섭이 커진다. 따라서 `max_num_batched_tokens`를 포함해 **짧은 대화형 요청과 긴 prompt가 섞인 workload**에서 측정해야 의미가 있다.

## 8. KV Cache를 줄이는 attention architecture

### 8.1 MHA, MQA, GQA, MLA

| 방식 | KV 구성 | 장점 | 주의점 |
| --- | --- | --- | --- |
| MHA | Query head마다 K/V head 사용 | 표현력의 전통적 기준 | KV Cache가 가장 큼 |
| MQA | 모든 Query head가 하나의 K/V head 공유 | KV 용량과 bandwidth를 크게 절감 | 품질 trade-off 가능, 모델 학습 단계에서 결정 |
| GQA | Query head group마다 K/V head 공유 | MHA와 MQA 사이의 균형 | group 수에 따라 용량·품질이 달라짐 |
| MLA | K/V 정보를 저차원 latent로 압축해 cache | 매우 작은 KV 표현 가능 | architecture와 kernel 지원에 종속, 단순 head 수 비교가 어려움 |

Hugging Face 모델의 `config.json`에서 다음 항목을 확인하면 MHA·GQA·MQA 계열을 빠르게 구분할 수 있다.

```text
num_attention_heads == num_key_value_heads  → 보통 MHA
num_key_value_heads == 1                    → 보통 MQA
1 < num_key_value_heads < attention_heads  → 보통 GQA
```

MLA는 별도의 latent·projection 설정을 사용하므로 모델 문서와 architecture implementation을 함께 확인해야 한다.

중요한 점은 MHA를 실행 옵션 하나로 GQA로 바꾸는 것이 아니라는 것이다. 이미 학습된 model architecture와 weight가 다르므로 비교할 때는 **모델 크기·학습 데이터·품질이 함께 달라지는 confounder**가 생긴다. 속도만 보고 attention 방식의 우열이라고 단정하면 안 된다.

### 8.2 KV Cache 관점의 의미

KV head 수가 `H`에서 `H_kv`로 줄면 다른 조건이 같을 때 KV Cache는 대략 `H_kv / H` 비율로 감소한다. 그 결과는 단순한 메모리 절약에 그치지 않는다.

- 더 많은 active sequence를 수용할 수 있다.
- 더 긴 context를 처리할 여유가 생긴다.
- Decode에서 읽을 KV data가 줄어 memory bandwidth 압박이 완화된다.
- Prefix Cache에 더 많은 block을 남길 수 있다.

## 9. Kernel과 KV 메모리 관리 최적화

### 9.1 Kernel Fusion

여러 연산을 각각 실행하면 intermediate result를 global memory에 썼다가 다음 kernel이 다시 읽는다. Kernel Fusion은 가능한 연산을 한 kernel에 묶어 register와 shared memory의 값을 바로 재사용한다.

```text
Unfused: read → op A → write → read → op B → write
Fused:   read → op A → op B → write
```

이 방식은 FLOPs 자체보다 kernel launch와 memory round trip을 줄이는 최적화다. 다만 큰 fused kernel은 register pressure를 높여 occupancy를 낮출 수 있으므로 항상 많이 합칠수록 좋은 것은 아니다.

### 9.2 FlashAttention

일반 attention은 큰 score matrix를 global memory에 저장하고 다시 읽는 과정이 비싸다. FlashAttention은 tiling과 online softmax를 이용해 전체 score matrix를 materialize하지 않고 작은 block을 빠른 on-chip memory에서 처리한다.

- exact attention의 결과를 유지하면서 I/O를 줄이는 algorithm/kernel 계열이다.
- model quantization이나 KV paging과는 다른 기술이다.
- version별로 scheduling, work partitioning과 특정 GPU architecture 활용 방식이 개선된다.
- 실제 사용 가능 version과 성능은 GPU, ROCm/CUDA, head dimension, dtype, model과 runtime build에 따라 달라진다.

**FlashInfer는 FlashAttention의 한 version이 아니라**, LLM inference를 위한 여러 kernel을 제공하는 library다. vLLM·SGLang에서 attention backend를 비교할 때는 “FlashInfer 대 FlashAttention 2/3/4”처럼 backend/library와 algorithm version을 구분해 기록해야 한다.

R9700은 AMD RDNA 계열이므로 NVIDIA 특정 기능을 전제로 한 backend 이름을 그대로 강제하지 않는다. 먼저 runtime이 자동 선택한 backend와 startup log를 기록하고, ROCm build에서 명시적으로 지원되는 후보만 비교한다. 지원되지 않는 backend의 실패도 version·error log·hardware 제약을 남기면 유효한 실험 결과다.

### 9.3 PagedAttention

요청마다 input과 output 길이가 다르고 output 길이는 미리 알 수 없다. sequence 최대 길이만큼 연속 메모리를 선점하면 내부·외부 단편화가 커진다.

PagedAttention은 운영체제의 virtual memory와 비슷하게 KV Cache를 고정 크기 block으로 나누고, logical block을 block table을 통해 물리 block에 연결한다.

```text
Logical KV blocks:   0 → 1 → 2
                       │   │   │
Block table:           ▼   ▼   ▼
Physical GPU blocks:  7   1   3
```

효과는 다음과 같다.

- sequence마다 거대한 연속 공간을 미리 예약하지 않아도 된다.
- 마지막 block을 제외한 공간 낭비와 memory fragmentation을 줄인다.
- 완료된 sequence의 block을 회수해 새 요청에 사용할 수 있다.
- copy-on-write와 block sharing을 통해 beam/prefix 계열 최적화의 기반이 될 수 있다.

vLLM에서 PagedAttention은 KV Cache 관리의 핵심 내부 구조이므로 일반적인 기능 flag처럼 ON/OFF 비교하는 대상이 아니다.

## 10. 모델 압축

모델 압축은 서빙 방법이 아니라 모델 데이터나 architecture 자체를 줄이는 최적화 층위다.

| 방법 | 무엇을 바꾸는가 | 장점 | 주요 비용·위험 |
| --- | --- | --- | --- |
| Quantization | 숫자의 precision | 즉시 적용하기 쉽고 memory·bandwidth 절감 | quantization error, kernel/hardware 종속성 |
| Distillation | 큰 teacher 지식을 작은 student에 학습 | model size와 연산량을 크게 줄일 잠재력 | 학습 비용, 품질 손실, 데이터·teacher 접근 필요 |
| Pruning | weight, channel, neuron, head 또는 block 제거 | parameter·연산 감소 가능 | sparsity 지원 없으면 실제 가속이 작음, 재학습 필요 가능 |

### 10.1 Quantization

Quantization은 weight, activation 또는 KV Cache를 더 낮은 bit 형식으로 표현한다.

같은 bit 수라고 숫자 특성이 같은 것은 아니다. 부동소수점은 부호, 지수부와 가수부에 bit를 나눠 사용한다.

| Format | 총 bit | 특징 |
| --- | ---: | --- |
| FP32 | 32 | 넓은 범위와 높은 정밀도, serving weight에는 비용이 큼 |
| FP16 | 16 | BF16보다 가수부가 길어 세밀하지만 표현 범위가 좁음 |
| BF16 | 16 | FP32와 같은 크기의 지수부를 가져 범위가 넓고 가수 정밀도는 낮음 |
| FP8 계열 | 8 | 지수·가수 배분 방식이 여러 가지이며 hardware·kernel 지원이 중요 |
| INT8/INT4 | 8/4 | scale과 zero point 등을 이용해 실수 범위를 정수에 mapping |

따라서 “16-bit” 또는 “8-bit”라는 이름만으로 정확도와 속도를 판단할 수 없다. format, scaling granularity, outlier 처리와 실제 accumulation precision까지 확인해야 한다.

#### 얻을 수 있는 이점

1. weight가 작아져 더 작은 VRAM에 모델을 올릴 수 있다.
2. global memory에서 읽는 byte가 줄어 memory-bound Decode가 빨라질 수 있다.
3. 지원되는 저정밀 matrix unit과 kernel이 있으면 compute throughput도 높아질 수 있다.
4. 남은 VRAM을 더 큰 batch와 KV Cache에 사용할 수 있다.

#### 발생하는 오차

- **Rounding error:** 원래 값을 낮은 precision의 가장 가까운 값으로 바꾸며 생긴다.
- **Clipping/clamping error:** 표현 가능한 범위를 벗어난 outlier가 최대·최소값으로 잘리며 생긴다.
- **Scaling:** 원래 값의 범위를 low-bit format에 배치하는 방법이다. per-tensor, per-channel, group-wise 방식에 따라 정확도와 kernel 효율이 달라진다.

#### 대표 구분

| 구분 | 예 | 특징 |
| --- | --- | --- |
| Weight-only | W4A16, GPTQ, AWQ 계열 | weight bandwidth·capacity를 크게 줄이고 activation은 높은 precision 유지 |
| Weight-and-activation | W8A8, 일부 FP8 방식 | weight와 activation 모두 줄여 지원 hardware에서 더 큰 가속 가능 |
| KV Cache quantization | FP8/저비트 KV | 긴 context·높은 concurrency의 KV memory 절감, attention kernel 지원 필요 |
| PTQ | 학습 후 calibration/변환 | 비교적 저렴하고 실무 적용이 쉬움 |
| QAT | 학습 중 quantization error를 모사 | 공격적 저비트에서 품질 보전에 유리하지만 학습 비용이 큼 |

파일이 절반으로 줄었다고 latency가 자동으로 절반이 되지는 않는다. runtime이 dequantization을 비효율적으로 수행하거나 GPU가 해당 format을 native로 가속하지 못하면 memory만 줄고 속도는 같거나 느려질 수 있다. 따라서 quantized model은 반드시 다음을 함께 비교한다.

- 실제 GPU memory 사용량과 최대 concurrency
- TTFT, ITL/TPOT, output TPS와 saturation point
- task별 품질과 생성 안정성
- fallback 또는 dequantization kernel 사용 여부
- model load time과 artifact 크기

GGUF는 다양한 저비트 형식을 제공하며 주로 llama.cpp 계열의 CPU·Apple Silicon·CPU/GPU hybrid 실행에서 널리 사용된다. vLLM용 GPU quantization artifact와 동일한 성능 경로라고 가정하면 안 된다.

### 10.2 Distillation

Distillation은 큰 teacher의 행동이나 분포를 작은 student가 모방하도록 학습한다. 원본 weight의 bit만 줄이는 양자화와 달리 **더 작은 architecture의 새로운 모델**을 만든다.

- teacher의 token·sequence output만 이용하는 sequence-level 또는 black-box distillation도 가능하다.
- teacher의 logits, hidden state와 loss를 이용하는 고전적 knowledge distillation은 더 풍부한 신호를 주지만 teacher 내부 접근이 필요하다.
- 이미 공개된 distilled model이 요구 품질을 만족하면 직접 distillation하는 것보다 먼저 평가하는 편이 경제적이다.
- distilled model 위에 quantization을 추가할 수도 있으므로 두 기법은 배타적이지 않다.

### 10.3 Pruning

- **Unstructured pruning:** 개별 weight를 제거해 높은 sparsity를 만들 수 있지만 일반 dense kernel에서는 0도 그대로 계산할 수 있다.
- **Structured pruning:** channel, neuron, head, block 또는 정해진 N:M pattern을 제거해 hardware/kernel이 활용하기 쉽다.

따라서 parameter의 50%를 0으로 만들었다고 실제 inference가 2배 빨라지는 것은 아니다. sparse format, sparse kernel과 hardware support가 일치하고 정확도 회복 과정까지 검증되어야 한다.

## 11. Prefix Caching

### 11.1 무엇을 재사용하는가

다음처럼 여러 요청이 같은 앞부분을 공유한다고 가정한다.

```text
[공통 system prompt][공통 문서][질문 A]
[공통 system prompt][공통 문서][질문 B]
[공통 system prompt][공통 문서][질문 C]
```

Prefix Caching은 공통 token prefix를 Prefill하면서 만든 KV block을 저장하고, 다음 요청이 동일한 token prefix를 사용할 때 재사용한다.

```text
첫 요청:  공통 prefix Prefill + 질문 A Prefill + Decode
다음 요청: cached KV 재사용 + 질문 B Prefill + Decode
```

주된 효과는 Prefill 계산과 TTFT 감소다. 이후 output token을 생성하는 Decode 자체를 없애는 것은 아니다. exact token prefix의 계산 결과를 재사용하므로 올바르게 구현된 cache는 모델 품질을 바꾸는 최적화가 아니다.

### 11.2 효과적인 workload

- 동일 system prompt를 쓰는 다수 요청
- 같은 긴 문서를 대상으로 여러 질문을 하는 RAG
- 이전 대화 전체가 prefix로 반복되는 multi-turn chat
- 공통 tool definition이나 few-shot example이 긴 agent 요청

효과가 작을 수 있는 경우는 다음과 같다.

- 요청마다 prompt 앞부분부터 다른 경우
- 문서 순서, whitespace, serialization이 매번 달라 token prefix가 일치하지 않는 경우
- cache보다 요청 종류가 많아 block이 자주 eviction되는 경우
- Prefill이 전체 latency에서 차지하는 비중이 작은 짧은 prompt

### 11.3 Prompt를 cache-friendly하게 만드는 법

1. 변경되지 않는 system prompt와 tool schema를 앞에 둔다.
2. RAG document의 정렬과 serialization을 결정적으로 유지한다.
3. 사용자별 동적 정보는 공유와 격리 요구를 고려해 배치한다.
4. tokenization 후 실제로 같은 prefix인지 확인한다.
5. hit rate뿐 아니라 TTFT 절감과 GPU memory pressure를 함께 측정한다.

Multi-tenant 환경에서는 latency 차이를 이용해 다른 tenant의 cache 존재를 추론하는 side channel과 의도하지 않은 공유를 고려해야 한다. tenant/session salt를 prefix에 포함하면 격리는 강해지지만 tenant 간 cache sharing은 줄어든다. 보안 경계가 hit rate보다 우선이다.

### 11.4 Cache-aware Routing

여러 replica에 분산하면 prefix KV Cache도 각 replica의 GPU memory에 따로 존재한다. 단순 round-robin은 같은 prefix를 매번 다른 replica로 보내 cache hit를 낮출 수 있다.

Cache-aware router는 request prefix와 replica affinity를 이용해 해당 cache를 가진 replica를 우선한다. 그러나 locality만 보면 hot prefix가 있는 replica에 요청이 몰린다. 실전 routing score에는 다음 요소가 함께 필요하다.

- prefix cache hit 가능성 또는 재사용 가능한 token 수
- queue length와 active sequence 수
- KV Cache 여유와 eviction 가능성
- 예상 Prefill·Decode 비용
- tenant quota와 장애 상태

즉, **cache locality와 load balancing의 균형**이 핵심이다.

SGLang의 RadixAttention은 radix tree를 이용하는 prefix reuse 구현이고, vLLM은 block hash 기반의 자체 prefix caching을 사용한다. 같은 목적을 풀지만 내부 자료구조와 설정이 같지는 않다.

## 12. 최적화 기법을 구분하는 한 장 표

| 기법 | 적용 층위 | 주로 줄이는 것 | 가장 직접적인 기대 효과 | 대표 trade-off |
| --- | --- | --- | --- | --- |
| Continuous Batching | Scheduler | 비어 있는 batch slot | output TPS, GPU utilization | queueing·간섭·KV 압박 |
| Chunked Prefill | Scheduler | 긴 Prefill의 독점 시간 | Decode ITL과 fairness | chunk overhead, TTFT 변화 |
| GQA/MQA/MLA | Model architecture | KV head/representation | KV capacity·bandwidth | model 선택·품질·kernel 지원 |
| Kernel Fusion | Kernel | intermediate memory traffic | latency·throughput | register pressure, 구현 종속 |
| FlashAttention | Attention kernel | attention HBM/global-memory I/O | attention latency·memory | hardware·shape·backend 종속 |
| PagedAttention | KV memory manager | fragmentation·preallocation 낭비 | 동시성·memory utilization | block 관리 overhead |
| Weight quantization | Model representation | weight bytes | memory·Decode bandwidth | 품질·dequant/kernel 종속 |
| KV quantization | Cache representation | KV bytes | 긴 context·동시성 | 품질·attention kernel 종속 |
| Distillation | Model training | model size/FLOPs | 큰 latency·cost 절감 잠재력 | 학습 비용과 품질 손실 |
| Pruning | Model structure/weight | 불필요 parameter | memory·compute | sparse hardware/kernel 필요 |
| Prefix Caching | Request cache | 반복 Prefill | TTFT·input compute | VRAM, hit rate, tenant isolation |
| Cache-aware Routing | Distributed routing | replica 간 cache miss | cluster Prefill 비용 | load imbalance와 복잡도 |

## 13. vLLM 실습 흐름에서 알아야 할 것

vLLM 학습 흐름은 명령을 외우는 것이 아니라 다음 변화가 왜 필요한지 확인하는 과정으로 볼 수 있다.

1. **Naive Hugging Face inference로 baseline을 만든다.**
   - 같은 모델이라도 일반 training-oriented runtime과 serving engine의 scheduling·KV 관리가 다름을 확인한다.
2. **vLLM offline inference를 비교한다.**
   - PagedAttention과 batching을 활용하는 engine이 동일 workload를 어떻게 처리하는지 본다.
3. **KV Cache 문제를 관찰한다.**
   - context 길이와 동시성이 GPU memory를 어떻게 소모하는지 확인한다.
4. **PagedAttention의 효과를 이해한다.**
   - OS paging과 유사한 block 관리가 variable-length sequence의 단편화를 줄이는 이유를 연결한다.
5. **OpenAI-compatible API server를 실행한다.**
   - offline batch가 아니라 실제 online request, streaming과 client compatibility를 확인한다.
6. **여러 client로 부하를 준다.**
   - concurrency가 증가할 때 throughput은 어느 지점까지 오르고 latency는 언제 급증하는지 찾는다.
7. **scheduler parameter를 조정한다.**
   - `max_num_seqs`, token budget, context limit을 하나씩 바꿔 capacity와 tail latency를 비교한다.
8. **production monitoring 관점을 추가한다.**
   - request 수만 보지 않고 token throughput, queue, TTFT, ITL, GPU memory와 오류를 함께 본다.

핵심은 “vLLM이 항상 빠르다”는 문장을 외우는 것이 아니라, **어떤 workload에서 어떤 engine 기능이 병목을 줄였는지 증명하는 것**이다.

## 14. 추후 진행 예정

아래 항목은 이번 문서에서 수행하지 않았으며, 결과도 포함하지 않는다. 이후 필요에 따라 검토할 실습 후보만 남긴다.

- Continuous Batching 활성화 전후의 처리량·지연시간 비교
- `max_num_seqs`, `max_model_len`, `max_num_batched_tokens` 변경 비교
- Chunked Prefill 활성화 전후 비교
- MHA·MQA·GQA·MLA 모델의 추론 성능과 품질 비교
- R9700·ROCm에서 지원되는 attention backend 비교
- 동일 모델의 BF16·quantized artifact 비교
- 공개된 teacher·student 모델의 Distillation 효과 비교
- Prefix Caching 활성화 전후 비교
- multi-replica 환경의 Cache-aware Routing 비교
- NFS cold/warm 상태와 local SSD staging의 model loading 시간 비교

## 15. 신뢰할 수 있는 benchmark 설계

### 15.1 먼저 가설을 쓴다

예시는 다음과 같다.

> 동일한 mixed workload에서 Chunked Prefill을 활성화하면 output TPS를 크게 해치지 않으면서 진행 중인 Decode 요청의 ITL p95를 낮출 것이다.

가설이 없으면 숫자가 바뀐 이유를 사후에 임의로 설명하기 쉽다.

### 15.2 한 번에 한 변수만 바꾼다

모델, dtype, scheduler, prompt 길이와 concurrency를 동시에 바꾸면 어느 변화가 결과를 만들었는지 알 수 없다. 비교할 configuration을 표로 먼저 고정한다.

| 항목 | Baseline | Variant |
| --- | --- | --- |
| Model/revision | 동일 | 동일 |
| Dtype/quantization | 동일 | 동일 |
| Input/output distribution | 동일 | 동일 |
| Concurrency | 동일 | 동일 |
| Prefix cache state | 동일 | 동일 |
| 변경 변수 | OFF 또는 기준값 | ON 또는 비교값 |

### 15.3 workload를 숫자로 공개한다

- input token의 min/median/p95/max
- output token의 목표와 실제 분포
- 동시 client 수와 총 request 수
- open-loop arrival rate인지 closed-loop concurrency인지
- streaming 여부
- shared prefix의 token 길이와 반복 비율
- warm-up 횟수와 측정 반복 수

Open-loop는 정해진 arrival rate로 요청을 보내 queueing까지 포함한 capacity를 보기 좋고, closed-loop는 일정 concurrency를 유지해 처리량을 비교하기 좋다. 서로 다른 방식의 결과를 같은 그래프에서 직접 비교하지 않는다.

### 15.4 saturation point를 찾는다

concurrency를 1, 2, 4, 8, 16처럼 올리며 다음 변화를 본다.

1. 처음에는 GPU utilization과 TPS가 함께 증가한다.
2. 어느 지점부터 TPS 증가가 둔화한다.
3. 그 이후 queueing과 p95/p99 latency가 급증한다.

보통 운영점은 최대 TPS 한계보다 조금 낮은 곳에서 SLO와 headroom을 확보한다. peak benchmark의 최대 숫자를 그대로 production 설정으로 사용하면 작은 traffic burst에도 tail latency와 timeout이 커질 수 있다.

### 15.5 결과 표의 최소 형태

| Profile | Concurrency | Input p50/p95 | Output p50/p95 | TTFT p95 | ITL p95 | Output TPS | Peak VRAM | Error rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline |  |  |  |  |  |  |  |  |
| variant |  |  |  |  |  |  |  |  |

결론에는 다음 네 항목을 반드시 함께 쓴다.

1. 어떤 조건에서 무엇이 개선되었는가
2. 무엇이 악화되었는가
3. 어떤 병목이 원인이었다고 해석하는가
4. 이 결과를 다른 model·GPU·workload에 일반화할 수 없는 이유는 무엇인가

## 16. 자주 혼동하는 개념

### 모델 로딩 bandwidth와 실행 memory bandwidth

- 모델 로딩: storage·CPU memory·PCIe를 거쳐 weight를 GPU에 올리는 일
- 모델 실행: 이미 GPU에 있는 weight·KV를 on-chip memory와 compute unit으로 공급하는 일

첫 번째는 cold start, 두 번째는 token generation 성능에 직접 연결된다.

### FlashAttention과 PagedAttention

- FlashAttention: attention 계산 중 global memory I/O를 줄이는 algorithm/kernel
- PagedAttention: KV Cache를 block으로 할당해 fragmentation을 줄이는 memory manager

이름은 비슷하지만 해결 계층이 다르다.

### Prefix Caching과 일반 response caching

- Prefix Caching: prompt 앞부분의 KV state를 재사용하고 나머지 Prefill과 Decode는 실행
- Response caching: 동일 요청의 최종 응답 자체를 반환해 model inference를 생략

sampling이나 사용자 상태가 달라질 수 있는 생성 요청에서 두 기술의 correctness 조건은 다르다.

### Quantization과 Distillation

- Quantization: 같은 model weight의 숫자 표현을 줄이는 경우가 많음
- Distillation: 작은 student model을 새로 학습

둘 다 model을 가볍게 하지만 비용, 품질 위험과 배포 artifact가 다르다.

### GPU utilization과 유효 처리량

GPU utilization이 높아도 불필요한 padding, 재계산 또는 SLO를 넘긴 요청을 처리하고 있을 수 있다. utilization은 원인 진단용 지표이며 사용자에게 전달된 정상 token 처리량과 함께 봐야 한다.

## 17. 스스로 답해볼 핵심 질문

1. peak TFLOPS가 더 높은 GPU가 Decode TPS에서는 더 느릴 수 있는 이유는 무엇인가?
2. model weight가 VRAM의 절반만 차지해도 OOM이 발생할 수 있는 이유는 무엇인가?
3. KV Cache 공식에서 `num_attention_heads` 대신 `num_key_value_heads`를 봐야 하는 경우는 언제인가?
4. 긴 Prefill과 작은 batch Decode가 서로 다른 병목을 갖는 이유를 산술 강도로 설명할 수 있는가?
5. `max_num_seqs`를 늘렸을 때 throughput과 TTFT·ITL·VRAM에는 각각 어떤 변화가 생길 수 있는가?
6. Chunked Prefill의 효과를 짧은 prompt만으로 측정하면 왜 결론을 내리기 어려운가?
7. MHA 모델을 runtime flag만으로 GQA 모델로 바꿀 수 없는 이유는 무엇인가?
8. Kernel Fusion, FlashAttention, PagedAttention이 줄이는 memory traffic은 각각 어떻게 다른가?
9. 4-bit checkpoint가 BF16보다 작아도 실제 GPU에서 더 느릴 수 있는 이유는 무엇인가?
10. Distillation과 quantization을 함께 사용할 수 있는 이유는 무엇인가?
11. Prefix Cache hit rate는 높은데 일부 replica의 p99가 악화된다면 routing을 어떻게 바꿔야 하는가?
12. NFS에서 model을 읽는 시간이 줄었는데 server ready time이 거의 같다면 다음에 어느 구간을 측정해야 하는가?
13. 평균 TTFT가 개선됐지만 p99와 error rate가 악화됐다면 그 설정을 production에 적용해도 되는가?
14. benchmark 결과를 재현하려면 model과 hardware 외에 어떤 조건을 공개해야 하는가?

## 마무리

LLM 서빙 성능은 하나의 component가 결정하지 않는다. storage에서 model을 읽는 순간부터 GPU memory에 weight와 KV Cache를 배치하고, Prefill과 Decode를 scheduling하고, kernel이 data를 이동하며, 여러 replica가 cache를 공유하지 못한 채 요청을 처리하는 과정 전체에서 병목이 생길 수 있다.

따라서 가장 중요한 습관은 최적화 기법을 무작정 켜는 것이 아니라 다음 순서를 지키는 것이다.

```text
SLO와 workload 정의
→ 단계별 측정
→ capacity/compute/bandwidth/communication 병목 분류
→ 병목에 맞는 최적화 하나 적용
→ 동일 조건 재측정
→ 품질·비용·tail latency까지 판단
```

현재 R9700 32GB 환경에서는 Continuous Batching, scheduler 한도, Chunked Prefill과 Prefix Caching을 먼저 비교하는 것이 가장 직접적이다. 이후 ROCm에서 실제로 지원되는 quantization과 attention backend를 확인하고 범위를 넓히는 편이 좋다. 이 순서를 따르면 성공한 설정뿐 아니라 지원되지 않거나 성능이 나빠진 결과도 **어떤 계층이 병목인지 보여주는 재현 가능한 근거**로 남길 수 있다.
