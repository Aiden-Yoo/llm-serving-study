# LLM 서빙 핵심 정리: Transformer에서 vLLM까지

> 이 글의 목표는 모델 서빙의 전체 그림을 잡고, LLM 추론에서 왜 KV Cache·PagedAttention·배칭·스트리밍이 필요한지 이해한 뒤 실제 GPU에서 검증하는 것이다.
>
> **작성 상태:** 개념 정리와 R9700의 첫 기준 측정을 완료했다. 아래 수치는 특정 단일 시스템에서 얻은 초기 결과이며, 보편적인 성능 수치가 아니라 이후 설정 비교를 위한 baseline으로 사용한다.

## 먼저 보는 핵심 요약

1. **모델은 가중치 파일 하나가 아니다.** 모델 데이터, 아키텍처, 실행 코드가 함께 있어야 실행 가능한 소프트웨어가 된다.
2. **모델 서빙은 시스템 엔지니어링 문제다.** 정확도뿐 아니라 지연시간, 처리량, 안정성, 보안, 확장성, 비용을 함께 관리해야 한다.
3. **LLM은 한 번의 추론으로 문장 전체를 만들지 않는다.** 이전 토큰을 문맥으로 사용해 다음 토큰을 하나씩 생성한다.
4. **Self-Attention은 시퀀스가 길어질수록 비싸진다.** 모든 토큰 쌍의 관계를 계산하므로 일반적인 full attention의 계산량은 시퀀스 길이에 대해 제곱으로 증가한다.
5. **KV Cache는 이미 계산한 Key와 Value를 저장해 재사용한다.** 중복 계산을 크게 줄이는 대신 GPU 메모리를 소비한다.
6. **LLM 추론은 Prefill과 Decode로 나뉜다.** Prefill은 주로 연산 집약적이고, Decode는 주로 메모리 대역폭과 KV Cache 관리의 영향을 크게 받는다.
7. **PagedAttention은 KV Cache를 고정 크기 블록으로 관리한다.** 메모리 단편화와 과도한 선점 문제를 줄여 더 많은 동시 요청을 수용할 수 있게 한다.
8. **스트리밍은 체감 지연시간을 개선한다.** 그러나 실제 총 연산량이나 전체 생성 시간 자체를 없애는 기술은 아니다.
9. **배칭은 GPU 활용률과 처리량을 높인다.** 반면 배치 대기시간, 요청 간 간섭, 공정성 문제로 개별 요청의 지연시간은 나빠질 수 있다.
10. **최적 설정은 하나로 정해져 있지 않다.** 모델 크기, 입력·출력 길이, 트래픽 패턴, SLA, 하드웨어와 예산에 맞춰 측정하며 결정해야 한다.

---

## 과제 수행 환경과 범위

이 글은 개념 요약으로 끝내지 않고 다음 단일 GPU 환경에서 과제와 연계 실험을 수행하는 것을 목표로 한다.

| 항목 | 현재 환경 |
| --- | --- |
| GPU | AMD Radeon AI PRO R9700 |
| GPU 아키텍처 | RDNA 4, `gfx1201` |
| VRAM | 32GB GDDR6 |
| OS | Ubuntu 24.04.4 LTS |
| ROCm | 7.2.4 |
| CPU / RAM | AMD EPYC 7551, 할당된 16 vCPU / RAM 94GiB |

[AMD의 ROCm 호환성 문서](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html)는 R9700(`gfx1201`)을 지원 대상으로 명시하고, [vLLM의 AMD GPU 요구사항](https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html)은 ROCm과 `gfx1200/1201` 계열 Radeon 9000 GPU를 지원 범위로 안내한다. 32GB VRAM이면 3B·7B급 모델의 FP16/BF16 추론과 여러 KV Cache·배칭 실험을 수행하기에 충분하다. 14B 이상은 모델 가중치만으로 VRAM 여유가 빠르게 줄어들기 때문에 양자화와 짧은 context부터 검증하는 편이 안전하다.

로컬 루트 저장 장치의 여유 공간이 작아, 실험 기준 모델인 공식 `Qwen/Qwen3-4B-Instruct-2507` BF16 safetensors는 NFS에 저장했다. 공개 문서와 실행 명령에서는 실제 마운트 경로 대신 저장소 기준 상대경로로 참조한다.

```text
./models/Qwen3-4B-Instruct-2507
```

모델은 revision `cdbee75f17c01a7cc42f958dc650907174af0554`로 고정했다. 세 개의 safetensors shard와 tokenizer/config 파일을 포함하며 총 약 7.6GB다. 반면 vLLM 가상환경과 컴파일·커널 캐시는 작은 파일 I/O가 많은 점을 고려해 `./.runtime/vllm`에 대응하는 로컬 SSD 영역에 두었다. 즉, **용량이 큰 모델 원본은 NFS, 반복 접근이 많은 런타임 캐시는 로컬 SSD**로 역할을 나눴다. 실제 경로가 다른 환경에서는 `VLLM_MODEL_PATH`와 `VLLM_RUNTIME_ROOT` 환경변수로 재지정할 수 있다.

실험은 다음 순서로 진행한다.

1. Qwen3-4B BF16으로 ROCm·PyTorch·vLLM 동작 확인
2. 같은 모델과 고정된 prompt 집합으로 기준 성능 측정
3. KV Cache, 입력·출력 길이, 배칭, prefix caching 설정을 하나씩 변경
4. 결과가 안정되면 기존 Q8 GGUF 또는 추가 양자화 모델로 범위 확장

이 가운데 1단계와 기준 성능 측정은 완료했다. 설정별 반복 측정과 GGUF·양자화 비교는 baseline이 흔들리지 않는지 확인한 뒤 진행한다.

단일 GPU에서 의미가 작은 텐서 병렬과 P/D disaggregation은 구현 과제에서 제외하고 개념과 트레이드오프만 다룬다. 반면 KV Cache, Prefill/Decode, 스트리밍, 배칭은 이 환경에서 직접 측정할 수 있으므로 글의 핵심 실험으로 삼는다.

## 1. 모델 서빙이란 무엇인가

모델 서빙(model serving)은 학습이 완료된 모델을 실제 요청을 받을 수 있는 형태로 배포하고, 입력을 받아 추론한 뒤 결과를 반환하는 운영 단계다.

```text
요청 수신 → 전처리 → 모델 추론 → 후처리 → 응답 반환
```

학습 단계에서는 정확도, 손실, 학습 속도가 중요하다. 반면 서빙 단계에서는 다음 항목이 중요하다.

- **Latency**: 요청 하나를 얼마나 빨리 처리하는가
- **Throughput**: 단위 시간에 얼마나 많은 요청 또는 토큰을 처리하는가
- **Availability / Reliability**: 장애 상황에서도 안정적으로 동작하는가
- **Scalability**: 트래픽 변화에 맞게 확장·축소할 수 있는가
- **Resource efficiency**: GPU, CPU, 메모리를 효율적으로 사용하는가
- **Cost**: 요청당 또는 토큰당 비용을 지속 가능한 수준으로 유지하는가
- **Security / Observability**: 접근 제어, 데이터 보호, 로그·메트릭·추적이 가능한가

즉, 좋은 모델을 보유하는 것과 좋은 AI 서비스를 운영하는 것은 다른 문제다. 프로덕션에서는 모델의 정확도뿐 아니라 **SLA를 만족하면서 비용을 통제하는 능력**이 필요하다.

## 2. 모델은 무엇으로 구성되는가

모델은 크게 세 요소로 볼 수 있다.

### 2.1 모델 데이터

- 학습된 weight와 bias
- 모델 설정값(config)
- 토크나이저와 vocabulary
- 입출력 텐서 정의, 라벨, 임베딩 등 실행 메타데이터

Hugging Face 모델을 예로 들면 `config.json`, `generation_config.json`, `model.safetensors`, `tokenizer.json` 등이 함께 필요하다.

### 2.2 모델 아키텍처

레이어 종류와 개수, 연결 방식, 연산 순서 등 모델의 설계도다. LLM이라면 일반적으로 다음과 같은 흐름을 가진다.

```text
Token Embedding
→ Transformer Blocks
→ Attention + FFN
→ Final Normalization
→ LM Head
```

### 2.3 모델 실행 코드

아키텍처를 초기화하고, 가중치를 로드하고, 추론 모드에서 입력을 실행하는 코드다. 같은 가중치가 있어도 이를 해석할 아키텍처와 실행 환경이 없으면 모델을 사용할 수 없다.

아키텍처와 가중치를 분리하면 버전 관리, 부분 로딩, 파인튜닝, 레이어 변경 같은 운영 시나리오에 유연하게 대응할 수 있다.

## 3. 대표적인 모델 서빙 방식

### 3.1 On-device / Edge Serving

스마트폰, 카메라, 차량, 로봇 등 사용자 기기에서 모델을 직접 실행한다.

**장점**

- 네트워크 왕복이 없어 초저지연 구현 가능
- 오프라인 환경에서도 동작
- 원본 데이터를 외부로 보내지 않아 프라이버시에 유리

**한계**

- 연산 능력, 메모리, 저장 공간, 전력의 제약
- 기기별 하드웨어와 런타임 호환성 차이
- 모델 업데이트와 배포 관리의 어려움

### 3.2 Single-model Service

모델 하나 또는 모델 버전 하나를 전용 서비스로 배포한다. 보통 API 서버, 모델 관리 계층, 추론 백엔드로 구성한다.

**장점**

- 모델별 독립 확장과 장애 격리가 쉽다.
- 리소스 경쟁이 적어 높은 성능과 낮은 지연시간을 얻기 쉽다.
- 배포, 관측, 디버깅이 비교적 단순하다.

**한계**

- 모델 수가 많으면 서비스와 유휴 자원이 함께 늘어난다.
- 수백·수천 개의 저트래픽 모델을 각각 띄우면 비용과 운영 부담이 커진다.

### 3.3 Multi-model Service

하나의 서빙 컨테이너가 여러 모델을 호스팅하고, 요청에 따라 모델을 동적으로 로드하거나 언로드한다.

**장점**

- 여러 저트래픽 모델이 자원을 공유해 비용을 줄일 수 있다.
- LRU 같은 캐시 정책으로 자주 쓰는 모델만 메모리에 유지할 수 있다.

**한계**

- 모델이 로드되지 않은 경우 콜드 스타트가 발생한다.
- 어떤 인스턴스에 어떤 모델이 있는지 아는 cache-aware routing이 필요하다.
- 모델별 트래픽에 따른 replica 관리와 오토스케일링이 복잡하다.
- 의존성 충돌과 모델별 보안 정책 차이도 관리해야 한다.

### 3.4 Model Serving Platform

Gateway, routing, 모델 서비스, 리소스 그룹, 다단계 추론 워크플로, 보안과 관측 기능을 통합한 플랫폼이다.

예를 들어 하나의 요청이 다음 파이프라인을 거칠 수 있다.

```text
Intent Classification
→ Embedding
→ Retrieval
→ LLM
→ Safety Filter
```

중요한 점은 네 방식 중 항상 우월한 하나가 있는 것이 아니라는 것이다. **모델 크기, 모델 수, 트래픽, 지연시간 목표, 보안, 비용에 따라 선택하거나 조합해야 한다.**

## 4. Decoder-only Transformer의 생성 과정

GPT, Llama, Qwen 같은 생성형 LLM은 주로 decoder-only Transformer 구조를 사용한다.

하나의 다음 토큰을 생성하는 과정은 다음과 같다.

```text
텍스트 입력
→ Tokenizer가 토큰 ID로 변환
→ Embedding 벡터로 변환
→ N개의 Decoder Block 통과
   ├─ Causal Self-Attention
   └─ Feed-Forward Network
→ LM Head가 vocabulary 크기의 logits 생성
→ 샘플링 또는 선택으로 다음 토큰 결정
```

생성된 토큰은 기존 입력 뒤에 붙고 같은 과정이 반복된다.

```text
prompt → token 1
prompt + token 1 → token 2
prompt + token 1 + token 2 → token 3
...
```

이를 **자기회귀적(autoregressive) 생성**이라고 한다. 요청 하나가 단 한 번의 forward pass로 끝나지 않고, 출력 길이만큼 decode step을 반복한다는 사실이 LLM 서빙을 어렵게 만드는 출발점이다.

## 5. Self-Attention과 Q, K, V

Self-Attention은 각 토큰이 다른 토큰을 얼마나 참고할지 계산하여 문맥을 반영한다.

각 토큰 표현으로부터 다음 세 벡터를 만든다.

- **Query(Q)**: 현재 토큰이 찾고자 하는 정보
- **Key(K)**: 각 토큰이 어떤 정보를 가지고 있는지를 나타내는 색인
- **Value(V)**: 실제로 결합할 정보

Scaled Dot-Product Attention의 핵심 흐름은 다음과 같다.

1. Query와 Key를 내적해 관련도 점수를 계산한다.
2. 점수를 스케일링하고 causal mask를 적용한다.
3. softmax로 attention weight를 만든다.
4. weight에 따라 Value를 가중합한다.

Full attention에서는 길이 `L`인 시퀀스의 토큰 쌍을 비교하므로 attention score 행렬의 크기가 `L × L`이 된다. 따라서 대표적인 attention 연산 비용은 `O(L²D)`로 설명할 수 있고, 긴 문맥일수록 prefill 비용이 빠르게 커진다.

### Multi-Head Attention과 GQA

Multi-Head Attention은 여러 attention head가 서로 다른 관계를 병렬로 학습하도록 한다. 그러나 모든 Query head마다 별도의 Key/Value head를 유지하면 KV Cache가 커진다.

GQA(Grouped Query Attention)는 여러 Query head가 더 적은 수의 Key/Value head를 공유한다. 예를 들어 Qwen2.5-0.5B는 Query head 14개와 KV head 2개를 사용한다. Query head 7개가 KV head 하나를 공유하므로, Query head마다 K/V를 두는 일반 MHA보다 KV Cache 저장량을 크게 줄일 수 있다.

### 과제 1: GQA·MQA 동작 원리 정리

GQA·MQA가 KV head를 공유해 메모리 사용량을 줄이는 원리를 정리한 뒤, 실제 모델 설정을 이용해 KV Cache 크기를 계산한다.

#### 확장 실험: 모델 설정에서 KV Cache 크기 계산

**가설:** Query head 수가 같아도 KV head가 적은 GQA/MQA 모델은 토큰당 KV Cache 사용량이 더 작다.

1. 후보 모델의 `config.json` 또는 `model.config`에서 `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `hidden_size`를 확인한다.
2. `head_dim = hidden_size / num_attention_heads`를 계산한다.
3. 정밀도별 토큰당 KV Cache 크기를 아래 식으로 추정한다.

```text
token당 KV Cache bytes
= 2(K와 V)
  × layer 수
  × KV head 수
  × head dimension
  × dtype bytes
```

4. 동일한 context 길이와 동시 요청 수에서 MHA·GQA 모델의 이론값을 비교한다.

이 계산은 모델 가중치와 activation을 제외한 KV Cache 자체의 근사치다. 실제 엔진에서는 블록 할당, 정렬, 메타데이터 등으로 측정값이 달라질 수 있다.

기준 모델인 Qwen3-4B는 36 layers, 32 Query heads, 8 KV heads, head dimension 128, BF16 구성을 사용한다. 위 식에 대입하면 토큰당 KV Cache 이론값은 약 144KiB다. 따라서 한 요청이 16,384 tokens를 모두 사용하면 KV Cache만 약 2.25GiB가 필요하다. 모델이 지원하는 최대 context를 그대로 설정하는 대신 실제 요구 길이에 맞춰 `max_model_len`을 제한해야 하는 이유를 보여준다.

## 6. KV Cache: 계산을 메모리와 교환하기

### KV Cache가 없을 때

다음 토큰을 만들 때마다 지금까지의 전체 시퀀스를 모델에 다시 넣으면, 이전 토큰의 Key와 Value까지 매번 다시 계산한다.

```text
1번째 생성: prompt 전체 계산
2번째 생성: prompt + token 1 전체 재계산
3번째 생성: prompt + token 1 + token 2 전체 재계산
```

시퀀스가 길어질수록 이미 수행한 연산을 반복하므로 토큰 생성 시간이 계속 늘어난다.

### KV Cache를 사용할 때

이전에 계산한 각 레이어의 Key와 Value를 GPU 메모리에 저장한다. 다음 decode step에서는 새 토큰의 Q/K/V만 계산하고, 새 Query가 캐시된 과거 Key/Value를 참조한다.

```text
최초 prompt 처리 → 과거 K/V 저장
새 토큰 입력 → 새 K/V를 cache에 추가 → 다음 토큰 생성
```

이때 한 decode step의 attention 계산은 새 Query와 이전 Key들을 비교하는 형태가 되어 대표적으로 `O(LD)` 수준으로 줄어든다. 얼마나 빨라지는지는 모델, prompt 길이, 출력 길이, 커널과 하드웨어에 따라 달라지므로 직접 측정해야 한다.

### KV Cache의 트레이드오프

KV Cache는 공짜 최적화가 아니다.

- 계산량과 decode latency는 줄어든다.
- 대신 레이어 수, KV head 수, head dimension, 시퀀스 길이, 동시 요청 수에 비례해 GPU 메모리를 소비한다.
- 출력 길이는 요청 전에 정확히 알기 어려우므로 메모리 예약과 회수가 복잡하다.

따라서 LLM 서빙에서 KV Cache 용량과 관리 방식은 **동시성, 처리량, 지연시간을 결정하는 핵심 요소**다.

### 과제 2: KV Cache, Prefill·Decode, P/D Disaggregation 학습 및 vLLM/SGLang 실습

> **이번 수행 범위:** KV Cache와 Prefill·Decode를 정리하고 vLLM에서 직접 측정한다. P/D Disaggregation은 개념과 트레이드오프만 다루며, SGLang 비교 실습은 범위에서 제외한다.

#### 확장 실험: KV Cache와 Prefix Cache 효과 측정

**가설:** KV Cache를 사용하면 출력이 길어져도 decode step 시간이 비교적 안정적으로 유지되지만, peak VRAM은 증가한다.

- 같은 모델, prompt, 출력 토큰 수, dtype, sampling 설정을 사용한다.
- `use_cache=False`와 `use_cache=True`만 바꾼다.
- 최초 실행의 커널 준비 시간을 제외하기 위해 warm-up 후 측정한다.
- GPU 연산은 비동기이므로 타이머 전후에 `torch.cuda.synchronize()`를 호출한다. PyTorch는 ROCm에서도 `torch.cuda` API를 사용한다.
- 다음 값을 기록한다.

| 측정값 | 확인하려는 내용 |
| --- | --- |
| TTFT | prompt 처리와 첫 토큰 비용 |
| 토큰별 decode 시간 | 출력 길이에 따라 시간이 증가하는지 |
| E2E latency | 전체 생성 시간 차이 |
| output tokens/s | 실제 생성 처리량 |
| peak VRAM | 계산 절감과 메모리 증가의 교환 관계 |

한 번의 결과만 비교하지 않고 동일 조건을 여러 번 실행해 중앙값과 P95를 기록한다.

#### 보조 측정: 반복 prefix의 KV 재사용

vLLM의 prefix cache는 일반적인 `use_cache=True/False` 비교와 동일한 실험은 아니다. 하지만 여러 요청이 공통 prompt의 KV block을 실제로 재사용할 때 어떤 효과가 생기는지 확인할 수 있다. 약 2K tokens의 공통 prefix 뒤에 서로 다른 질문을 붙여 차례로 요청한 결과는 다음과 같았다.

| 요청 | 실제 prompt tokens | Prefix cache hit | TTFT | E2E |
| --- | ---: | ---: | ---: | ---: |
| 최초 요청(cold) | 2,093 | 16 tokens | 0.210초 | 0.673초 |
| 반복 요청(warm) | 2,093 | 2,064 tokens | 0.058초 | 0.523초 |

공통 prefix가 재사용된 두 번째 요청의 TTFT는 이 한 번의 측정에서 약 72% 감소했다. 이는 prefix cache가 모델 연산 전체를 없앤다는 뜻이 아니라, **공통 입력 구간의 prefill 계산을 재사용해 첫 토큰까지의 시간을 줄였다**는 뜻이다. 정확한 효과는 prefix 길이, cache block 크기, 동시 부하와 eviction 여부에 따라 달라진다.

## 7. Prefill과 Decode를 구분해야 하는 이유

LLM 추론은 성격이 다른 두 단계로 나뉜다.

| 구분 | Prefill | Decode |
| --- | --- | --- |
| 역할 | 입력 prompt 전체 처리, 첫 토큰 생성, KV Cache 구성 | 이전 결과를 이용해 새 토큰을 하나씩 반복 생성 |
| 병렬성 | prompt 토큰을 병렬로 처리하기 쉬움 | 토큰 간 선행 의존성 때문에 순차적 |
| 대표 병목 | GPU 연산 성능, compute-bound 경향 | 메모리 대역폭과 KV Cache, memory-bound 경향 |
| 주요 지표 | TTFT, input token throughput | ITL/TPOT, output token throughput |
| 영향을 크게 받는 것 | 입력 길이 | 출력 길이와 동시 요청 수 |

### 주요 지연시간 지표

- **TTFT(Time To First Token)**: 요청을 보낸 뒤 첫 토큰을 받기까지의 시간
- **ITL(Inter-Token Latency)**: 연속한 출력 토큰 사이의 지연시간
- **TPOT(Time Per Output Token)**: 첫 토큰 이후 출력 토큰 하나를 만드는 평균 시간
- **E2E Latency**: 요청부터 전체 응답 완료까지 걸린 시간
- **Tail Latency(P95/P99)**: 느린 요청 구간의 지연시간

긴 문서처럼 입력이 매우 길면 Prefill이 병목이 되기 쉽다. 반대로 짧은 질문에 긴 답을 생성하는 챗봇이나 스토리 생성은 Decode 비용이 커지기 쉽다.

GPU 연산 유닛 사용률이 낮다고 해서 반드시 GPU가 남는 것은 아니다. Decode가 메모리 대역폭에 막혀 있다면 연산 유닛이 완전히 사용되지 않아도 전체 성능은 제한될 수 있다.

Prefill과 Decode가 서로 간섭하는 문제를 줄이기 위해 두 단계를 별도 워커나 하드웨어에 배치하는 **P/D disaggregation**도 사용할 수 있다. 다만 KV 전송 비용, 스케줄링 복잡도와 운영 비용이 추가되므로 항상 유리한 것은 아니다.

### 과제 2 연계 실험: Prefill과 Decode 병목 분리

**가설:** 입력 길이를 늘리면 TTFT가 크게 증가하고, 출력 길이를 늘리면 전체 생성 시간과 KV Cache 사용량이 증가한다.

두 실험에서 동시에 여러 변수를 바꾸지 않는다.

#### A. Prefill 실험

- 출력 길이를 32 tokens로 고정한다.
- 실제 tokenizer 기준 입력 길이를 128, 512, 2,048, 4,096 tokens로 변경한다.
- TTFT와 input tokens/s를 측정한다.

#### B. Decode 실험

- 입력 길이를 128 tokens로 고정한다.
- 출력 길이를 32, 128, 256 tokens로 변경한다.
- TPOT, output tokens/s, E2E latency, peak VRAM을 측정한다.

입력 문자열의 글자 수가 아니라 **토크나이저가 만든 실제 토큰 수**를 기준으로 조건을 맞춰야 한다. 또한 랜덤한 출력 조기 종료가 결과를 흔들지 않도록 EOS 처리와 생성 길이를 통제한다.

#### R9700 첫 측정 결과

워밍업 뒤 출력 길이를 32 tokens로 고정하고 입력만 늘렸다. chat template와 system prompt가 추가되어 실제 prompt tokens는 목표값보다 36개 많았다.

| 목표 user tokens | 실제 prompt tokens | TTFT | E2E |
| ---: | ---: | ---: | ---: |
| 128 | 164 | 0.085초 | 0.533초 |
| 512 | 548 | 0.088초 | 0.539초 |
| 2,048 | 2,084 | 0.210초 | 0.674초 |
| 4,096 | 4,132 | 0.445초 | 0.924초 |

짧은 두 입력은 고정 오버헤드와 측정 오차가 상대적으로 커 차이가 거의 없었다. 하지만 실제 prompt가 164 tokens에서 4,132 tokens로 늘었을 때 TTFT는 약 5.2배 증가했다. 입력 길이가 prefill 지연에 직접 영향을 준다는 가설과 일치한다.

다음으로 짧은 입력을 유지하고 출력 길이만 바꿨다. EOS를 무시해 요청한 길이만큼 생성하도록 통제했다.

| 출력 tokens | TTFT | TPOT | E2E |
| ---: | ---: | ---: | ---: |
| 32 | 0.086초 | 14.40ms | 0.533초 |
| 128 | 0.115초 | 14.50ms | 1.953초 |
| 256 | 0.103초 | 14.50ms | 3.804초 |

출력 길이가 늘어도 TPOT는 약 14.5ms로 안정적이었고, E2E는 생성한 토큰 수에 비례해 증가했다. 즉, 이 조건에서는 첫 토큰 이후의 decode 속도는 일정했지만 **자기회귀 생성 횟수 자체가 늘어 전체 응답 시간은 길어졌다.**

## 8. PagedAttention은 무엇을 해결하는가

전통적으로 요청마다 최대 시퀀스 길이를 기준으로 연속된 KV Cache 메모리를 미리 예약하면 문제가 생긴다.

- 실제 출력이 짧으면 예약한 공간 대부분이 낭비된다.
- 요청별 길이가 달라 메모리 사이에 사용하기 어려운 빈 공간이 생긴다.
- 전체 빈 메모리가 충분해도 연속 공간이 부족해 새 요청을 받지 못할 수 있다.

PagedAttention은 운영체제의 가상 메모리와 유사하게 KV Cache를 고정 크기 블록으로 나눈다. 각 요청은 논리 블록과 실제 GPU 메모리의 물리 블록을 연결하는 block table을 가진다.

**효과**

- 필요한 만큼 블록을 점진적으로 할당한다.
- 연속된 물리 메모리가 없어도 요청을 수용할 수 있다.
- 내부 단편화는 마지막 부분 블록 수준으로 제한된다.
- beam search나 공통 prefix가 있는 시퀀스가 물리 블록을 공유하고, 수정 시점에만 복사하는 방식도 가능하다.
- 절약한 메모리로 더 많은 KV Cache와 동시 요청을 수용할 수 있다.

중요한 구분은 다음과 같다.

- **FlashAttention**: attention 계산 중 HBM과 온칩 메모리 사이의 데이터 이동을 줄이는 연산 커널 최적화
- **PagedAttention**: 요청별 KV Cache의 할당, 저장, 공유를 효율화하는 메모리 관리 기법

둘 다 attention과 관련 있지만 해결하려는 병목이 다르다.

## 9. 왜 vLLM 같은 서빙 엔진을 사용하는가

Hugging Face `pipeline()`은 실험과 프로토타이핑에 편리하다. 하지만 여러 동시 요청을 효율적으로 스케줄링하고 GPU 메모리를 관리하는 프로덕션 서빙에는 전용 엔진이 유리하다.

vLLM은 대표적으로 다음 기능을 제공한다.

- PagedAttention 기반 KV Cache 관리
- continuous batching
- prefix caching
- chunked prefill
- 최적화된 attention kernel
- graph capture 등 실행 오버헤드 최적화
- tensor parallelism
- OpenAI-compatible API server
- 비동기 스트리밍

vLLM의 개념적인 실행 구조는 다음과 같다.

```text
API / LLMEngine
→ EngineCore
   ├─ Scheduler
   ├─ KV Cache Manager
   └─ Model Executor
       └─ GPU Worker / Model Runner
```

성능을 좌우하는 대표 설정은 `dtype`, `gpu_memory_utilization`, `max_num_seqs`, `max_num_batched_tokens`, `max_model_len`, `tensor_parallel_size` 등이다.

설정값을 크게 잡는다고 항상 좋아지는 것은 아니다. 예를 들어 동시 시퀀스와 배치 토큰 한도를 늘리면 처리량은 좋아질 수 있지만 KV Cache와 activation 메모리가 늘고, OOM 또는 요청별 지연시간 증가로 이어질 수 있다.

### 과제 2 연계 실험: R9700에서 vLLM 서빙 성능 측정

**가설:** `max_num_seqs`와 `max_num_batched_tokens`를 늘리면 어느 구간까지는 처리량이 증가하지만, 이후에는 queueing·메모리 압박·요청 간 간섭으로 이득이 줄어든다.

첫 기준 모델은 NFS에 저장한 Qwen3-4B BF16으로 정한다. 기존 `Qwen3-4B-Instruct-2507-Q8_0.gguf`와 같은 모델 계열이어서 llama.cpp 실험과도 연결할 수 있다. 다만 BF16과 Q8은 정밀도가 다르므로 두 결과의 차이를 서빙 엔진 차이만으로 해석하면 안 된다. 32GB 전체를 모델 가중치로 채우기보다 KV Cache와 동시 요청을 위한 공간을 남기는 것이 서빙 실험에 적합하다.

다음 순서로 한 번에 하나의 설정만 변경한다.

1. `gpu_memory_utilization`을 보수적인 값에서 시작한다.
2. `max_num_seqs`를 1, 4, 8, 16으로 늘린다.
3. `max_num_batched_tokens`를 단계적으로 늘린다.
4. 반복되는 system prompt를 사용해 prefix caching 활성화 전후를 비교한다.
5. OOM, latency 급증 또는 처리량 정체가 발생하는 지점을 기록한다.

`tensor_parallel_size`는 GPU가 한 장이므로 1로 유지한다. 모델을 간신히 메모리에 넣는 최대 크기보다, 여러 동시 요청과 충분한 context를 안정적으로 처리하는 크기를 기준 모델로 선택한다.

#### 첫 baseline 구성과 자원 할당

첫 서버는 다음 값으로 실행했다.

```text
vLLM 0.26.0+rocm723 / PyTorch 2.11.0+gitd0c8b1f
dtype=bfloat16
max_model_len=16384
gpu_memory_utilization=0.80
max_num_seqs=16
max_num_batched_tokens=8192
prefix_caching=true
attention backend=ROCM_ATTN
```

서버 로그에서 모델 weight는 7.61GiB, 사용 가능한 KV Cache는 16.64GiB로 계산되었다. KV Cache 수용량은 121,168 tokens였고, 16,384-token 요청을 기준으로 표시된 최대 동시성은 7.40배였다. 첫 실행에서는 NFS의 세 shard를 읽는 weight load에 79.63초가 걸렸으므로, 이 시간은 온라인 요청 latency와 분리해 봐야 한다.

#### 동시 요청 baseline

각 요청이 128 output tokens를 생성하도록 하고, 동시성마다 최소 두 묶음이 실행되도록 총 요청 수를 정했다. 모든 값은 streaming 응답의 첫 content delta를 기준으로 계산했다.

| 동시성 | 요청 수 | 총 output tok/s | TTFT P50/P95 | TPOT P50/P95 | E2E P50/P95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 66.3 | 0.092/0.100초 | 14.47/14.48ms | 1.930/1.938초 |
| 4 | 8 | 243.1 | 0.183/0.189초 | 15.11/15.12ms | 2.102/2.105초 |
| 8 | 16 | 371.2 | 0.201/0.213초 | 20.15/20.16ms | 2.746/2.772초 |
| 16 | 32 | 702.5 | 0.175/0.192초 | 21.55/21.59ms | 2.900/2.929초 |

동시성 1에서 16으로 늘리자 총 output throughput은 약 10.6배 증가했다. 그러나 요청별 TPOT P50은 14.47ms에서 21.55ms, E2E P50은 1.93초에서 2.90초로 악화됐다. **배칭은 전체 GPU 효율을 높이지만 개별 요청을 항상 더 빠르게 만들지는 않는다**는 트레이드오프가 드러난다. TTFT가 동시성에 따라 단조 증가하지 않은 것은 짧은 prompt, continuous batching의 스케줄링, 단일 실행의 변동성이 함께 섞인 결과로 보고 반복 측정이 필요하다.

이 측정에서는 클라이언트 동시성만 바꿨다. `max_num_seqs`와 `max_num_batched_tokens` 자체를 바꾼 비교는 다음 실험으로 남아 있으므로, 현재 표만으로 설정값 16과 8192가 최적이라고 결론내릴 수 없다.

## 10. Streaming과 Batching

두 기술은 서로 다른 목표를 가진다.

### 10.1 Streaming: 사용자 체감 지연시간 개선

LLM은 내부적으로 토큰을 하나씩 생성한다. 동기 API는 모든 토큰 생성이 끝난 뒤 전체 결과를 반환하지만, 스트리밍은 생성된 토큰 또는 작은 chunk를 즉시 클라이언트에 전달한다.

**장점**

- 사용자가 첫 토큰을 빠르게 확인해 응답이 시작되었다고 느낀다.
- 긴 응답을 기다리는 UX가 개선된다.
- 잘못된 방향의 생성을 중간에 취소해 불필요한 연산과 비용을 줄일 수 있다.

**주의점**

- TTFT 이후 출력을 즉시 보여줄 뿐, 전체 Decode 비용이 사라지지는 않는다.
- 연결 유지, backpressure, 취소 처리, 오류 전달을 구현해야 한다.
- 화면에 같은 누적 문자열을 반복 전송하는지, 새 delta만 전송하는지 API 계약을 명확히 해야 한다.

### 10.2 Batching: GPU 처리량 개선

여러 요청을 묶어 GPU에서 함께 처리하면 큰 행렬 연산을 만들 수 있고, 모든 요청이 모델 가중치를 공유하므로 GPU 활용률이 높아진다.

**장점**

- 요청당 반복되는 스케줄링과 커널 실행 오버헤드 감소
- GPU 병렬성 활용
- 단위 시간당 처리 요청 또는 토큰 증가

**트레이드오프**

- 배치를 모으는 대기시간이 생길 수 있다.
- 입력과 출력 길이가 서로 다르면 패딩 또는 스케줄링 비효율이 발생한다.
- 긴 요청이 짧은 요청을 지연시키는 head-of-line blocking과 fairness 문제가 생길 수 있다.
- 배치가 커질수록 메모리 사용량과 개별 요청 latency가 증가할 수 있다.

배칭의 개선 폭은 요청 길이 분포, 배치 크기, 커널과 GPU 포화도에 따라 달라진다. 따라서 다른 환경의 배수 수치를 인용하기보다 현재 하드웨어에서 기준값을 다시 측정한다.

### 10.3 Static, Dynamic, Continuous Batching

| 방식 | 동작 | 특징 |
| --- | --- | --- |
| Static batching | 고정된 요청 묶음을 모두 모아 함께 실행 | 단순하지만 배치 대기와 긴 요청의 영향이 큼 |
| Dynamic batching | 크기 한도 또는 짧은 timeout까지 모은 뒤 실행 | 처리량과 대기시간의 균형을 조정 |
| Continuous batching | decode iteration마다 완료된 요청을 빼고 새 요청을 투입 | GPU 슬롯 낭비를 줄이고 온라인 서빙에 적합 |

Continuous batching은 요청 단위가 아니라 **iteration/token 수준에서 실행 묶음을 재구성**한다. 먼저 끝난 시퀀스의 자리를 즉시 새 요청이 채우므로, 출력 길이가 다양한 온라인 트래픽에서 GPU를 더 지속적으로 활용할 수 있다.

### 과제 3: Streaming Serving과 Batch Serving의 동작 원리 정리

Streaming과 Batching이 각각 사용자 체감 지연시간과 전체 처리량에 어떤 영향을 주는지 정리하고, continuous batching의 효과를 실제 동시 요청으로 확인한다.

#### 확장 실험: Streaming과 Batching의 목표 분리

**가설:** 스트리밍은 사용자가 결과를 보기 시작하는 시점을 앞당기지만 E2E 연산량을 줄이지 않고, 동시 요청 배칭은 총 처리량을 높이지만 일부 요청의 tail latency를 악화시킬 수 있다.

동일한 prompt 집합을 다음 세 방식으로 실행한다.

1. 요청을 하나씩 순차 실행
2. 같은 요청을 한 번에 제출하는 offline batch
3. 도착 시간을 분산시킨 online concurrent requests

추가로 온라인 요청은 non-streaming과 streaming으로 각각 호출한다.

| 비교 | 핵심 지표 |
| --- | --- |
| 순차 실행 vs batch | 전체 소요 시간, output tokens/s, peak VRAM |
| 동시성 1/4/8/16 | requests/s, P50/P95 TTFT, P50/P95 TPOT |
| non-streaming vs streaming | 첫 화면 출력 시점, E2E latency |
| 정상 완료 vs 중간 취소 | 생성 토큰 수, 취소 반영 시간, 절약된 실행 시간 |

성능 결과에는 모델, dtype, prompt/output token 분포, warm-up 횟수, 동시성, 엔진 설정을 함께 기록한다. 그래야 단순히 “몇 배 빨랐다”가 아니라 왜 차이가 났는지 설명할 수 있다.

첫 baseline에서는 streaming으로 모든 요청의 TTFT를 수집했고, 위 동시성 표에서 continuous batching의 처리량 이득과 요청별 지연 증가를 함께 확인했다. 다만 non-streaming은 응답이 완료된 뒤 한꺼번에 전달되므로 서버 내부 TTFT와 사용자가 실제 화면에서 첫 글자를 보는 시간을 구분해야 한다. 브라우저 렌더링까지 포함한 streaming/non-streaming UX 비교와 중간 취소 실험은 별도의 클라이언트 측정으로 남겨 둔다.

## 11. 반드시 이해해야 할 트레이드오프

### Latency와 Throughput

- 작은 배치: 개별 요청은 빨리 시작할 수 있지만 GPU 활용률이 낮을 수 있다.
- 큰 배치: 전체 처리량은 높아질 수 있지만 queueing과 요청 간 간섭으로 latency가 늘 수 있다.

### 계산량과 메모리

- KV Cache: 중복 계산을 줄이지만 GPU 메모리를 사용한다.
- GQA/MQA: KV head 수를 줄여 cache 용량과 메모리 대역폭 부담을 낮춘다.
- 양자화: 가중치와 경우에 따라 KV Cache 메모리를 줄일 수 있지만 품질, 지원 커널, 하드웨어 호환성을 검증해야 한다.

### 수평 확장과 수직 확장

- **Scale out**: 같은 모델 replica를 늘려 더 많은 요청을 처리한다.
- **Scale up / model parallelism**: 한 장에 들어가지 않는 모델을 더 큰 GPU 또는 여러 GPU에 분산한다.

가능하면 통신 대역폭이 높은 단일 노드 내 여러 GPU를 우선 고려한다. 노드 간 분산은 네트워크 통신과 동기화 비용이 추가되므로 모델이 커질수록 interconnect가 중요한 병목이 된다.

### Single-model과 Multi-model

- 저지연·고트래픽·강한 격리가 필요하면 single-model이 유리하다.
- 모델 수가 많고 각 모델의 트래픽이 낮거나 불규칙하면 multi-model이 비용 면에서 유리할 수 있다.

## 12. 실무에서는 무엇을 측정해야 하는가

최적화는 추측이 아니라 측정에서 시작해야 한다.

### 요청과 사용자 경험

- TTFT
- ITL 또는 TPOT
- E2E latency
- P50/P95/P99 latency
- timeout, cancel, error rate

### 처리량과 스케줄링

- requests per second
- input/output tokens per second
- 현재 실행 중인 sequence 수
- queue length와 queue time
- batch당 sequence 수와 token 수

### GPU와 메모리

- GPU compute utilization
- HBM 사용량과 memory bandwidth
- 모델 weight, activation, KV Cache가 차지하는 메모리
- KV Cache utilization과 eviction
- OOM 횟수

### 비용

- request당 비용
- input/output token당 비용
- 유휴 GPU 시간
- SLA를 만족하는 최소 replica 수

평균값만 보면 일부 사용자의 심각한 지연을 놓칠 수 있으므로 P95/P99 같은 tail latency를 반드시 함께 봐야 한다.

## 13. 실험 결과를 기록하는 방법

모든 실험은 아래 공통 정보를 남긴다.

```text
실행 시각:
GPU / ROCm / PyTorch / vLLM 버전:
모델과 revision:
dtype / quantization:
max_model_len:
gpu_memory_utilization:
max_num_seqs / max_num_batched_tokens:
prompt token 분포:
output token 분포:
warm-up 횟수 / 측정 횟수:
TTFT P50/P95:
TPOT P50/P95:
input/output tokens per second:
peak VRAM:
오류 또는 OOM:
```

실험 결과는 다음 순서로 해석한다.

1. **관찰:** 실제로 측정된 숫자
2. **해석:** 왜 그런 결과가 나왔다고 보는지
3. **한계:** 통제하지 못한 변수와 일반화할 수 없는 범위
4. **다음 실험:** 가설을 확인하기 위해 바꿀 단 하나의 변수

이 구조를 사용하면 다른 환경의 벤치마크를 재인용하는 글이 아니라, R9700에서 직접 재현한 분석 글이 된다.

### 측정 코드의 핵심

전체 벤치마크 코드를 본문에 옮기기보다 streaming 응답에서 첫 content delta를 찾는 부분만 단순화해 보면 다음과 같다.

```python
started = time.perf_counter()
first_content_at = None

for event in streaming_response:
    if first_content_at is None and event.content:
        first_content_at = time.perf_counter()

finished = time.perf_counter()

ttft = first_content_at - started
tpot = (finished - first_content_at) / (output_tokens - 1)
throughput = total_output_tokens / total_wall_time
```

- **TTFT**는 요청을 시작한 시점부터 첫 실제 content를 받은 시점까지다. role만 전달하는 빈 첫 event는 제외한다.
- **TPOT**는 첫 토큰 이후 완료까지 걸린 시간을 나머지 output token 수로 나눈 값이다.
- **Throughput**은 동시 요청 전체가 생성한 output tokens를 전체 wall time으로 나눈 값이다.
- 여러 요청의 TTFT·TPOT·E2E를 각각 수집한 뒤 P50과 P95를 계산한다.

실제 구현은 OpenAI 호환 API의 SSE `data:` event를 JSON으로 해석하고, 마지막 usage event에서 prompt·completion token 수를 읽는다. 위 코드는 측정 기준을 보여주기 위한 축약본이며 예외 처리와 동시 요청 코드는 별도 스크립트에 둔다.

이번 baseline은 다음 명령으로 재현할 수 있다.

```bash
# 저장소 루트에서 1주차 디렉터리로 이동
cd week01

# 터미널 1: 서버 시작
./experiments/start_vllm_r9700.sh

# 터미널 2: 측정 실행
./.runtime/vllm/venvs/vllm-rocm723/bin/python \
  ./experiments/benchmark_vllm.py
```

위 명령은 모델을 `./models/Qwen3-4B-Instruct-2507`, 런타임을 `./.runtime/vllm`에 둔 공개용 기본 구성을 가정한다. NFS나 별도 SSD를 사용한다면 실제 경로는 공개 문서에 적지 않고 환경변수로만 전달한다.

전체 구현은 [서버 실행 스크립트](./experiments/start_vllm_r9700.sh)와 [벤치마크 스크립트](./experiments/benchmark_vllm.py)에서 확인할 수 있다. 원시 측정값과 요약은 각각 [latest.json](./experiments/results/latest.json), [latest.md](./experiments/results/latest.md)에 저장된다. JSON에는 각 sweep의 조건과 vLLM·PyTorch 버전, 서버 설정, prefix cache counter 변화가 함께 들어 있다.

## 마무리

중요한 것은 개별 최적화 기법의 이름을 외우는 것이 아니다. **LLM이 실제로 어떻게 실행되고, 어느 단계에서 어떤 자원이 병목이 되는지 연결해서 이해하는 것**이다.

```text
자기회귀 생성
→ 토큰 단위 반복 실행
→ 과거 문맥의 반복 계산 문제
→ KV Cache로 계산 재사용
→ KV Cache 메모리 관리 문제
→ PagedAttention
→ 여러 요청을 효율적으로 섞는 continuous batching
```

그리고 사용자 경험과 시스템 효율을 분리해서 봐야 한다.

```text
Streaming → 사용자가 응답을 빨리 보기 시작하게 함
Batching  → GPU가 더 많은 요청을 효율적으로 처리하게 함
```

결국 좋은 LLM 서빙 시스템은 가장 빠른 단일 요청만 만드는 시스템이 아니다. **목표 latency와 안정성을 지키면서, 제한된 GPU로 최대한 많은 유효 토큰을 지속 가능한 비용에 처리하는 시스템**이다.
