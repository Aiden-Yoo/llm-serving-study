# LLM 서빙 시스템 설계와 운영: 단일·멀티 모델에서 엔터프라이즈 아키텍처까지

> 이 글의 목표는 단순한 `model.generate()` 호출을 실제 서빙 시스템으로 확장할 때 필요한 구성 요소를 이해하고, 단일 모델·멀티 모델·에이전트·엔터프라이즈 환경에서 어떤 설계 선택을 해야 하는지 설명할 수 있게 되는 것이다.
>
> **작성 상태:** 이번 학습 범위의 핵심 내용을 하나의 문서로 재구성하고, R9700에서 비동기 호출·vLLM 스케줄링 비교와 멀티 모델 lazy loading·LRU 실험을 완료했다. RayService·AWS EKS 배포는 후속 과제로 구분한다.

## 먼저 보는 핵심 요약

1. **모델 서빙은 `generate()`를 API로 감싸는 작업보다 훨씬 크다.** 요청 추적, 큐잉, 배칭, 스트리밍, 프로세스 격리, 모델 수명주기, 라우팅, 확장, 장애 복구를 함께 설계해야 한다.
2. **API 처리와 모델 실행은 분리하는 편이 좋다.** CPU 중심의 네트워크·전후처리 작업과 GPU 중심의 추론 작업을 격리하면 자원 활용, 장애 격리, 독립 확장이 쉬워진다.
3. **배칭은 처리량을 높이지만 대기시간을 만든다.** 처리량만 최대화하면 개별 요청의 지연시간과 공정성이 악화될 수 있다.
4. **스트리밍은 전체 계산을 없애지 않는다.** 첫 토큰을 먼저 전달해 체감 응답성을 높이는 기술이며, 서버 내부에서는 요청별 상태와 출력 라우팅이 필요하다.
5. **정적 배칭과 연속 배칭은 다르다.** 정적 배칭은 배치 전체가 끝날 때까지 기다리지만, 연속 배칭은 완료된 요청의 자리에 새 요청을 넣어 GPU 유휴 시간을 줄인다.
6. **비동기 API만 사용한다고 서버가 비동기가 되는 것은 아니다.** `async` 핸들러 안에서 동기 추론 함수를 직접 호출하면 이벤트 루프가 막혀 동시 요청이 직렬화될 수 있다.
7. **멀티 모델 서빙의 핵심은 모델 수명주기 관리다.** 요청한 모델이 메모리에 없으면 로드하고, 자원이 부족하면 LRU 같은 정책으로 제거하며, 실제 GPU 메모리까지 회수해야 한다.
8. **비용 최적화와 지연시간 최적화는 서로 다른 구조를 요구한다.** 공유 인스턴스와 동적 로딩은 비용에 유리하고, 모델별 전용 인스턴스와 사전 로딩은 지연시간에 유리하다.
9. **에이전트는 한 요청에서 모델을 여러 번 호출한다.** 단일 호출의 작은 지연도 계획·도구 실행·검색·후속 추론이 이어지는 제어 루프에서는 크게 증폭된다.
10. **엔터프라이즈 서빙은 모델 실행보다 넓은 문제다.** Public API, 자원 관리, 모델 선택, 분산 서빙, 추론 엔진, 모델 최적화, 모델 수명주기를 계층별로 나눠야 한다.
11. **직접 구축과 관리형 서비스는 이분법이 아니라 스펙트럼이다.** 요구사항과 팀 역량에 맞춰 Bedrock 같은 완전 관리형부터 Kubernetes 기반 자체 플랫폼까지 통제 범위를 선택한다.
12. **성능은 지연시간과 처리량을 함께 봐야 한다.** E2E, TTFT, ITL/TPOT, RPS/RPM, TPS는 서로 다른 현상을 측정한다.
13. **벤치마크 숫자에는 측정 조건이 포함되어야 한다.** 모델·정밀도·하드웨어뿐 아니라 입력/출력 길이 분포, 동시성, 배치 정책, 캐시 상태를 공개해야 비교가 가능하다.
14. **좋은 설계는 목표 SLO에서 출발한다.** 가장 빠르거나 가장 복잡한 시스템이 아니라, 필요한 사용자 경험과 비용 목표를 안정적으로 만족하는 시스템이 좋은 시스템이다.

---

## 학습 범위

CH3·CH4에서 다룬 내용을 장별로 분리하지 않고, 실제 서빙 시스템의 흐름에 맞춰 하나의 글로 통합했다.

- 단일 모델 온라인 서비스의 최소 구성
- 배칭과 스트리밍의 내부 동작
- 일반화된 단일 모델 서빙 구조
- 멀티 모델의 동적 로딩·캐싱·제거
- 비용 최적화와 지연시간 최적화 설계 비교
- 에이전트 환경의 모델 서빙
- 계층형 엔터프라이즈 아키텍처
- Kubernetes·Ray Serve·vLLM 기반 오픈소스 스택
- 클라우드 관리형부터 자체 인프라까지의 선택지
- 지연시간과 처리량 측정 방법

예제 구현은 원리를 드러내기 위해 의도적으로 단순화되어 있다. 프로덕션 시스템이라면 여기에 인증, 입력 검증, 취소와 타임아웃, backpressure, 재시도, 장애 복구, 텔레메트리, 배포 전략, 보안 정책 등이 추가되어야 한다.

## 1. `generate()` 호출과 서빙 시스템의 차이

로컬 코드에서 다음과 같이 모델을 실행하는 것은 추론의 핵심 연산만 보여준다.

```python
output = model.generate(input_ids)
```

온라인 서비스는 그 앞뒤에 여러 책임을 추가해야 한다.

```text
Client
  → 인증·요청 검증
  → 큐잉·스케줄링
  → 토크나이징·배치 구성
  → 모델 실행
  → 출력 추적·디토크나이징
  → 스트리밍 또는 최종 응답
  → 메트릭·로그·과금
```

특히 LLM 요청은 다음 특성 때문에 일반적인 짧은 예측 API보다 다루기 어렵다.

- 입력과 출력 길이가 요청마다 크게 다르다.
- 출력 토큰을 순차적으로 생성하므로 한 요청이 여러 decode step을 점유한다.
- 각 요청의 KV Cache가 GPU 메모리를 계속 사용한다.
- 스트리밍 중인 연결을 오래 유지해야 한다.
- 긴 요청 하나가 짧은 요청 여러 개의 처리를 지연시킬 수 있다.
- 모델 하나가 크기 때문에 로드, 복제, 교체 비용이 높다.

따라서 서빙 시스템의 핵심 질문은 단순히 “모델이 동작하는가?”가 아니다.

> **어떤 요청을 언제, 어느 모델과 어느 자원에서 실행해야 목표 지연시간·처리량·비용·안정성을 만족하는가?**

## 2. 단일 모델 서빙 시스템의 기본 구조

교육용 단일 모델 서비스는 여섯 가지 구성 요소로 나눌 수 있다.

| 구성 요소 | 주요 책임 | 핵심 관심사 |
| --- | --- | --- |
| API Server | HTTP 요청 수신, 검증, 응답·스트리밍 | 연결 관리, 인증, 오류 처리 |
| LLM Engine | 전체 요청 흐름 오케스트레이션 | 컴포넌트 초기화, 상태 조율 |
| Workload Manager | 대기열, 스케줄링, 배치 구성 | 공정성, 배치 크기, 대기시간 |
| Model Executor | 워커 생성·관리와 IPC | 프로세스 수명주기, 장애 처리 |
| Model Worker | 실제 모델 추론 | GPU 실행, 토큰 생성 |
| Model Manager | 모델·토크나이저 로드와 캐싱 | 메모리, 버전, load/unload |

전체 요청 흐름은 다음과 같다.

```text
Client
  → API Server
  → LLM Engine
  → Workload Manager
  → Model Executor
  → Model Worker
  → Model Manager / Model
  → 생성 결과를 역방향으로 반환
```

### 2.1 각 계층을 나누는 이유

책임을 나누면 다음 효과를 얻는다.

- **대체 가능성:** Hugging Face 기반 워커를 vLLM이나 Triton 기반 워커로 교체하기 쉽다.
- **독립 확장:** API 연결 수와 GPU 추론량을 서로 다른 기준으로 확장할 수 있다.
- **장애 격리:** 모델 프로세스의 오류가 API 프로세스 전체로 전파되는 범위를 줄인다.
- **관측 가능성:** 큐 대기시간, 모델 실행시간, 응답 전송시간을 나눠 측정할 수 있다.
- **테스트 가능성:** 스케줄러, 모델 로더, API 계약을 각자 검증할 수 있다.

컴포넌트를 나누는 것 자체가 목적은 아니다. 요청량이 적은 개인용 서비스라면 한 프로세스가 더 단순할 수 있다. 다만 부하와 팀 규모가 커질수록 **책임 경계가 확장성과 운영 안정성을 결정한다.**

### 2.2 CPU 작업과 GPU 작업의 프로세스 격리

API 처리, JSON 파싱, 토크나이징, 전후처리는 CPU 작업이다. 모델 forward는 GPU 작업이다. 이들을 하나의 실행 흐름에 강하게 묶으면 CPU 작업이나 Python 런타임 문제가 GPU 실행을 기다리게 할 수 있다.

```text
API / Orchestrator process              Model Worker process
──────────────────────────              ────────────────────
HTTP 연결                               모델 로드
요청 검증               IPC Queue       GPU forward
스케줄링             ───────────────→    토큰 생성
응답 라우팅           ←───────────────    결과 반환
```

프로세스 격리의 장점은 다음과 같다.

- API와 모델의 서로 다른 동시성 모델을 독립적으로 운영할 수 있다.
- 모델 크래시나 메모리 오류의 영향을 제한할 수 있다.
- 여러 워커를 띄우거나 다른 실행 백엔드로 교체하기 쉽다.
- GPU는 모델 연산에 집중하고 CPU는 요청 관리에 집중할 수 있다.

반대로 IPC 직렬화 비용, 프로세스 관리 복잡도, 디버깅 난도가 추가된다. 그러므로 분리는 **무료 최적화가 아니라 운영을 위한 트레이드오프**다.

## 3. 단일 요청에서 배칭으로

단일 요청만 순차 처리하면 구조는 단순하지만 GPU 병렬성을 충분히 사용하지 못할 수 있다.

```text
Request A 실행 완료 → Request B 실행 완료 → Request C 실행 완료
```

배칭은 여러 입력을 하나의 tensor로 묶어 한 번에 실행한다.

```text
Request A ┐
Request B ├─→ Batch → Model → A/B/C 결과 분리
Request C ┘
```

### 3.1 배칭 처리 흐름

1. 각 요청에 고유한 sequence ID를 부여한다.
2. 요청을 prompt queue에 넣는다.
3. 스케줄러가 최대 배치 크기와 대기시간을 기준으로 요청을 선택한다.
4. 길이가 다른 입력을 padding하고 batch tensor를 만든다.
5. 워커가 배치 추론을 실행한다.
6. 결과를 sequence ID에 따라 원래 요청으로 돌려보낸다.

sequence ID가 중요한 이유는 **GPU가 배치를 처리하더라도 클라이언트는 자기 요청의 결과만 받아야 하기 때문**이다.

### 3.2 정적 배칭과 연속 배칭

| 구분 | 정적 배칭 | 연속 배칭 |
| --- | --- | --- |
| 배치 구성 | 시작할 때 요청 묶음을 고정 | 매 decode step에서 활성 요청을 재구성 가능 |
| 완료 요청 처리 | 배치 전체가 끝날 때까지 함께 관리 | 완료된 슬롯에 새 요청 투입 |
| 구현 난도 | 낮음 | 높음 |
| GPU 활용률 | 길이 편차가 크면 낮아짐 | 상대적으로 높음 |
| 대표 용도 | 교육용 구현, 균일한 offline batch | vLLM 같은 온라인 LLM 서빙 엔진 |

정적 배치에서 짧은 요청은 먼저 끝나도 긴 요청이 끝날 때까지 배치 자원을 효율적으로 넘겨주기 어렵다. 연속 배칭은 끝난 요청을 제거하고 새 요청을 투입해 이 낭비를 줄인다.

### 3.3 배칭의 트레이드오프

배치 크기를 늘리면 보통 처리량이 좋아지지만 다음 비용이 생긴다.

- 배치를 채우기 위한 queueing delay
- 입력·출력 길이 차이로 인한 padding 또는 실행 낭비
- KV Cache 증가에 따른 메모리 압박
- 긴 요청이 짧은 요청에 주는 간섭
- 특정 테넌트가 배치를 독점하는 공정성 문제

따라서 `batch_size` 하나만 크게 잡는 방식으로는 충분하지 않다. 실제 시스템에서는 다음 조건을 함께 고려한다.

- 최대 동시 sequence 수
- 토큰 예산과 KV Cache 여유
- 최대 대기시간
- 입력 길이와 예상 출력 길이
- 우선순위와 테넌트별 quota
- deadline 또는 SLO

## 4. 스트리밍과 요청별 상태 관리

비스트리밍 방식은 전체 결과가 완성된 뒤 응답한다.

```text
요청 ───────── 전체 생성 완료 ─────────→ 한 번에 응답
```

스트리밍 방식은 생성된 토큰 또는 텍스트 조각을 순차적으로 전송한다.

```text
요청 → token 1 → token 2 → token 3 → ... → done
```

HTTP 환경에서는 SSE(Server-Sent Events)를 사용할 수 있다. 서버 내부에는 보통 요청마다 결과를 받을 수 있는 이벤트 큐나 future가 필요하다.

```python
request_id = create_sequence(prompt)
event_queue = queues[request_id]

while True:
    event = await event_queue.get()
    yield encode_sse(event)
    if event.is_finished:
        break
```

핵심은 모델 워커가 여러 요청을 한 배치로 처리해도, 생성 결과를 다시 올바른 `request_id`의 연결로 전달하는 것이다.

### 4.1 스트리밍이 해결하는 문제와 해결하지 않는 문제

**해결하는 문제**

- 사용자가 첫 결과를 더 빨리 볼 수 있다.
- 긴 응답의 체감 대기시간이 줄어든다.
- 클라이언트가 출력을 점진적으로 표시하거나 중간에 취소할 수 있다.

**자동으로 해결하지 않는 문제**

- 모델의 총 연산량
- 최종 토큰까지의 전체 시간
- GPU 메모리 사용량
- 서버 처리량
- 느린 클라이언트에 대한 backpressure

### 4.2 교육용 스트리밍 구현에서 배울 점

원문의 단순 구현은 매 step에서 지금까지의 전체 prompt를 다시 넣고 `use_cache=False`로 forward한다. 생성 길이가 늘수록 이미 처리한 prefix를 반복 계산하므로 비효율적이다.

```text
step 1: prompt 계산
step 2: prompt + token 1 전체 재계산
step 3: prompt + token 1 + token 2 전체 재계산
...
```

이 구현은 프로덕션 해법이 아니라 **KV Cache와 증분 디코딩이 왜 필요한지 드러내는 반례**로 이해해야 한다.

실제 구현에서는 다음을 고려해야 한다.

- 요청별 KV Cache와 sampling state
- 클라이언트 연결 종료 시 generation 취소
- timeout과 최대 출력 길이
- 느린 소비자에 대한 bounded queue와 backpressure
- 오류·종료 이벤트의 명시적 전송
- UTF-8 경계와 토큰 단위/문자열 단위 chunk 처리

## 5. 직접 구현과 vLLM의 차이

교육용 서비스에는 다음 네 실행 경로가 등장한다.

| 엔드포인트 | 실행 방식 | 학습 포인트 |
| --- | --- | --- |
| `/basic_generate` | 단일 Hugging Face `generate()` | 가장 단순한 요청 흐름 |
| `/generate` | 큐에 모아 고정 크기 배치 실행 | 수동 정적 배칭 |
| `/generate_stream` | 토큰 단위 반복 실행과 SSE | 요청 상태와 출력 라우팅 |
| `/generate_vllm` | vLLM 엔진 호출 | PagedAttention·연속 배칭 추상화 |

직접 구현은 큐와 상태 전이를 눈으로 확인하기 좋다. 반면 vLLM은 다음과 같은 복잡성을 엔진 내부에서 처리한다.

- KV Cache 블록 관리
- 연속 배칭
- 요청별 generation state
- 메모리 예산에 따른 스케줄링
- 최적화된 attention kernel

하지만 프레임워크를 사용해도 모든 운영 문제가 사라지는 것은 아니다.

- 어떤 engine 설정이 workload에 맞는가?
- API와 엔진을 어떻게 비동기로 연결할 것인가?
- overload 시 어디에서 거절하거나 대기시킬 것인가?
- timeout, cancellation, retry를 어떻게 전파할 것인가?
- 어떤 지표로 autoscaling할 것인가?

### 5.1 비동기 API의 함정

다음 코드는 함수 선언만 비동기일 뿐, 동기 추론이 실행되는 동안 이벤트 루프를 막을 수 있다.

```python
@app.post("/generate")
async def generate(request: Request):
    return llm.generate(request.prompt)  # 동기·장시간 호출
```

이 경우 동시 HTTP 요청을 받아도 실제 추론 호출이 직렬로 진행될 수 있다. 해결 방향은 다음과 같다.

- 프레임워크가 제공하는 비동기 엔진 사용
- 별도 worker process와 비동기 IPC 사용
- 제한된 thread/process executor로 blocking 작업 격리
- 서버의 admission control과 동시성 한도 설정

중요한 것은 `async`라는 문법이 아니라 **호출 경로 전체가 non-blocking인지** 확인하는 것이다.

## 6. 프로덕션 단일 모델 서비스의 일반화

단일 모델 서비스의 목표는 보통 다음 여섯 항목으로 정리할 수 있다.

- 낮은 지연시간
- 높은 처리량
- 트래픽 변화에 대한 확장성
- 신뢰성과 가용성
- 자원 효율과 비용 통제
- 관측성과 디버깅 가능성

이를 세 책임 영역으로 나누면 설계가 선명해진다.

### 6.1 Infrastructure Management

- 로드 밸런서와 서비스 디스커버리
- replica 배치와 autoscaling
- health check, restart, failover
- CPU·GPU·메모리 자원 할당
- 로그·메트릭·트레이스 수집

### 6.2 Serving Frontend / Business Logic

- 인증·인가와 tenant 처리
- 요청 검증과 입력 정규화
- rate limit과 quota
- 모델 alias·version 선택
- 요청 큐잉, 우선순위, 라우팅
- 사용량 기록과 과금 정보 생성

### 6.3 Serving Backend / Model Inference

- 모델 로드와 warm-up
- 토크나이징과 generation
- 배칭과 스케줄링
- KV Cache 관리
- 양자화·병렬화·kernel 최적화
- GPU 메모리와 실행 오류 관리

이 분리는 조직 구조에도 영향을 준다. 플랫폼 팀, API 팀, 모델 최적화 팀이 안정된 계약을 통해 독립적으로 개선할 수 있어야 한다.

## 7. 멀티 모델 서빙

단일 모델 서비스는 모델마다 전용 replica를 유지하기 쉽지만, 모델이 수백 개이고 각 모델의 트래픽이 낮다면 유휴 GPU 비용이 커진다. 멀티 모델 서비스는 하나의 인스턴스 또는 자원 풀에서 여러 모델을 필요할 때 로드해 이 문제를 줄인다.

### 7.1 주요 구성 요소

| 구성 요소 | 역할 |
| --- | --- |
| API Server | `model_id`와 입력을 받아 통합 API 제공 |
| Model Manager | 어떤 모델이 로드되어 있는지 추적하고 load/unload 결정 |
| Model Store | 로컬 디스크나 객체 저장소에서 모델 artifact 제공 |
| Model Engine / Worker Factory | 모델 종류에 맞는 worker 생성 |
| Model Worker | Transformers, TorchVision, Triton 등 실제 실행 백엔드 |

요청 흐름은 다음과 같다.

```text
요청(model_id)
  → Model Manager에서 cache hit 확인
  ├─ hit: 기존 worker 사용
  └─ miss: Model Store에서 artifact 확인
           → Worker Factory로 적절한 worker 생성
           → 모델 로드·warm-up
           → 필요하면 기존 모델 제거
  → 추론
  → 결과 반환
```

### 7.2 Lazy Loading과 LRU

**Lazy loading**은 모델 요청이 처음 들어올 때 모델을 로드한다. 모든 모델을 미리 올리지 않아 메모리를 절약하지만 첫 요청에는 cold start가 생긴다.

메모리 한도에 도달하면 LRU(Least Recently Used) 같은 정책으로 오래 사용되지 않은 모델을 제거할 수 있다.

```python
def get_or_load(model_id):
    if model_id in cache:
        cache.touch(model_id)
        return cache[model_id]

    if cache.is_full():
        victim = cache.least_recently_used()
        unload(victim)

    worker = worker_factory.create(model_id)
    worker.load()
    cache.put(model_id, worker)
    return worker
```

실제 시스템에서 LRU만으로 충분하지 않은 이유도 알아야 한다.

- 모델마다 크기와 로드 시간이 다르다.
- 최근 사용되지 않았어도 곧 다시 호출될 수 있다.
- 특정 모델은 SLO 때문에 항상 warm 상태여야 한다.
- Python 객체 삭제만으로 GPU 메모리가 즉시 운영체제에 반환되지 않을 수 있다.
- 진행 중인 요청이 있는 모델은 안전하게 제거할 수 없다.

따라서 실무 정책은 모델 크기, 요청 빈도, 로드 비용, 우선순위, pin 여부, 진행 중 요청 수를 함께 고려하는 경우가 많다.

### 7.3 Worker Factory가 필요한 이유

멀티 모델 서비스는 텍스트 생성 모델만 다루지 않을 수 있다.

```text
model metadata
  ├─ transformers → TransformersWorker
  ├─ torchvision  → TorchVisionWorker
  └─ triton       → TritonWorker
```

factory는 프레임워크별 생성 방식을 한곳에 모으고, API와 Model Manager가 구체적인 런타임에 의존하지 않도록 한다. 다만 서로 다른 프레임워크를 같은 프로세스에 무리하게 설치하면 dependency와 CUDA/ROCm 버전 충돌이 생길 수 있으므로, 컨테이너 또는 프로세스 단위 격리도 고려해야 한다.

## 8. Triton을 백엔드로 사용하는 의미

NVIDIA Triton Inference Server를 사용하면 애플리케이션의 worker가 모델을 직접 실행하는 대신 Triton의 관리 API와 추론 API를 호출하는 wrapper가 될 수 있다.

```text
Application / Model Manager
  → TritonWorker
     ├─ model load API
     ├─ inference API
     └─ model unload API
  → Triton Server
  → Model Repository
```

이 방식의 장점은 모델 로딩, 런타임 선택, 배칭, 메트릭 같은 일부 책임을 전용 추론 서버에 위임할 수 있다는 점이다. 반대로 Triton 운영, 모델 repository 규약, 네트워크 hop, 버전 호환성이라는 새로운 책임이 생긴다.

특히 다음 두 계층을 혼동하지 않는 것이 중요하다.

- **Control plane:** 어떤 모델을 어디에 배치하고 load/unload할지 결정
- **Data plane:** 실제 추론 요청과 tensor가 오가는 경로

Triton은 강력한 data plane과 모델 관리 기능을 제공하지만, 전체 서비스의 tenant 정책이나 fleet 전체의 모델 배치 결정을 자동으로 대신해 주는 것은 아니다.

## 9. 멀티 모델 설계의 두 방향

### 9.1 비용 최적화: 공유 인스턴스와 동적 로딩

여러 모델이 인스턴스 풀을 공유하고, 라우터가 현재 모델이 올라간 replica를 찾는다. 없으면 여유 자원에 모델을 로드하고 필요하면 다른 모델을 제거한다.

**장점**

- 저트래픽 모델의 유휴 자원을 공유할 수 있다.
- bin packing으로 필요한 GPU 수를 줄일 수 있다.
- 모델 수가 많고 접근 빈도가 낮은 환경에 유리하다.

**단점**

- cache miss 시 cold start가 발생한다.
- 모델 위치를 추적하는 라우팅 맵이 필요하다.
- load/unload와 eviction 경쟁 조건이 복잡하다.
- hot model이 갑자기 늘면 반응형 확장이 늦을 수 있다.

### 9.2 지연시간 최적화: 모델별 전용 인스턴스 그룹

각 모델 또는 모델 그룹을 전용 replica에 미리 로드하고 독립적으로 확장한다.

**장점**

- 요청 시 모델이 이미 메모리에 있어 cold start를 줄인다.
- 모델별 autoscaling과 성능 튜닝이 쉽다.
- 장애와 noisy neighbor의 영향을 격리하기 쉽다.
- 런타임 라우팅이 상대적으로 단순하다.

**단점**

- 유휴 replica가 생겨 GPU 활용률과 비용 효율이 낮아질 수 있다.
- 모델 수가 늘수록 배포 객체와 운영 대상이 증가한다.
- 용량 계획이 보수적으로 변해 과다 프로비저닝하기 쉽다.

### 9.3 비교

| 판단 기준 | 공유·동적 로딩 | 전용·사전 로딩 |
| --- | --- | --- |
| 우선 목표 | 비용·자원 활용률 | 지연시간·예측 가능성 |
| cold start | 있음 | 거의 없음 |
| 모델 수 | 매우 많고 저트래픽일 때 유리 | 제한된 hot model에 유리 |
| 확장 단위 | 공용 자원 풀 | 모델별 deployment |
| 운영 복잡도 | 캐시·배치·라우팅이 복잡 | 배포 수·용량 관리가 복잡 |
| 장애 격리 | 상대적으로 약함 | 상대적으로 강함 |

실제 플랫폼은 두 방식을 섞을 수 있다. 예를 들어 핵심 모델은 전용 replica로 유지하고, 롱테일 모델만 공유 풀에 lazy loading할 수 있다.

LLM에서는 다음 최적화가 라우팅에 추가 영향을 준다.

- **Prefix-cache-aware routing:** 같은 prefix의 KV Cache가 있는 replica를 우선 선택
- **Multi-LoRA serving:** base model은 공유하고 요청별 adapter만 선택
- **Session affinity:** 대화 세션을 cache가 남아 있는 replica로 유지

## 10. 에이전트 환경에서 서빙이 달라지는 이유

일반적인 요청은 모델을 한 번 호출할 수 있지만 에이전트는 목표가 끝날 때까지 반복한다.

```text
사용자 목표
  → 계획 수립
  → 도구 선택
  → 도구 실행
  → 결과 해석
  → 추가 검색 또는 모델 호출
  → 최종 응답
```

한 번의 사용자 요청이 embedding, retrieval, reranking, 여러 chat completion, 외부 API 호출을 연쇄적으로 만들 수 있다. 따라서 다음 문제가 중요해진다.

- 단계별 지연이 전체 E2E 지연으로 누적된다.
- 한 단계의 실패가 전체 workflow를 실패시킬 수 있다.
- 중간 결과와 tool call을 추적해야 한다.
- 호출 수 증가로 토큰 비용과 rate limit 부담이 커진다.
- 같은 세션의 context와 cache를 재사용할 가치가 커진다.

MCP(Model Context Protocol) 같은 표준은 모델이 사용할 도구의 발견과 호출 인터페이스를 정규화할 수 있다. 그러나 표준 인터페이스가 도구의 정확성, 권한 통제, timeout, 재시도, 감사 로그까지 자동으로 해결하지는 않는다.

### 10.1 RAG와 CAG

에이전트에 지식을 제공하는 대표적인 두 접근은 다음과 같다.

| 구분 | RAG | CAG |
| --- | --- | --- |
| 핵심 방식 | 요청 시 관련 문서를 검색해 context에 추가 | 큰 context 또는 재사용 가능한 KV Cache에 지식을 미리 반영 |
| 적합한 지식 | 자주 바뀌고 최신성이 중요한 지식 | 비교적 고정되고 반복 조회되는 지식 |
| 장점 | 필요한 정보만 선택, 업데이트 반영이 쉬움 | 반복 요청의 검색 비용과 지연을 줄일 수 있음 |
| 비용 | embedding, vector search, reranking, 검색 오류 | 긴 context 처리, KV Cache 메모리, 갱신 비용 |
| 주요 위험 | 관련 문서 누락·오검색, 파이프라인 복잡도 | 오래된 지식, 큰 메모리 점유, context 한계 |

RAG의 offline 경로에서는 문서를 수집하고 chunking·embedding·indexing한다. online 경로에서는 query embedding, retrieval, 선택된 context와 함께 generation을 수행한다.

chunk가 너무 작으면 의미가 끊기고, 너무 크면 관련 없는 정보와 토큰 비용이 늘어난다. 따라서 chunking도 단순 전처리가 아니라 검색 정확도, 지연시간, context 비용 사이의 설계 변수다.

## 11. 엔터프라이즈 모델 서빙의 7개 계층

엔터프라이즈 서빙에서는 여러 팀이 시스템을 동시에 바꾸기 때문에 계층별 책임과 인터페이스가 특히 중요하다.

### 11.1 Public API

고객·개발자·내부 서비스가 만나는 외부 인터페이스다.

- 네트워킹과 글로벌 라우팅
- 인증·인가와 tenant 격리
- quota, rate limit, abuse 방지
- 가격 정책과 사용량 기록
- 요청 검증과 API 호환성

### 11.2 Resource Management

여러 리전의 CPU, GPU, 메모리, 디스크, 네트워크를 관리한다.

- 수요 예측과 capacity planning
- 이기종 GPU 배치와 활용률
- 우선순위, reservation, preemption
- 예산과 비용 배분
- 장애 도메인과 가용성 관리

### 11.3 Model Selection & Orchestration

요청에 사용할 모델 또는 모델 조합을 선택한다.

- 품질·지연시간·비용에 따른 small/large model routing
- canary, fallback, tenant override
- 모델 family 간 load balancing
- speculative decoding을 위한 모델 조합

항상 가장 큰 모델을 사용하는 것은 품질이 아니라 비용 낭비가 될 수 있다. 요청 난도와 SLO에 맞는 모델을 선택하는 것이 중요하다.

### 11.4 Distributed Serving

단일 GPU를 넘어 모델과 cache를 여러 GPU·노드에 배치한다.

- tensor/pipeline parallelism
- multi-node 통신과 동기화
- 분산 KV·prompt·semantic cache
- cache-aware routing
- replica 간 상태 관리

### 11.5 Core Inference

모델의 실제 계산이 실행되는 계층이다.

- vLLM, Triton, TensorRT-LLM, SGLang 같은 serving runtime
- FlashAttention, GEMM, PagedAttention 같은 kernel·memory 최적화
- 토크나이징, 배칭, decode scheduling
- GPU 오류와 메모리 관리

### 11.6 Model Optimization

모델을 처음부터 다시 학습하지 않고 실행 효율을 개선한다.

- quantization
- speculative decoding
- KV/prefix caching
- pruning 또는 distillation
- workload에 맞는 병렬화와 kernel 선택

### 11.7 Model

서빙 대상 artifact 자체와 그 수명주기를 관리한다.

- 내부 학습 파이프라인 또는 외부 registry에서 모델 승격
- 모델 기능·목적별 분류
- version, lineage, metadata 추적
- 검증, 승인, rollback

7개 계층은 반드시 7개의 별도 서비스여야 한다는 뜻이 아니다. 중요한 것은 **책임과 변경 경계를 식별해 한 계층의 변화가 다른 계층 전체를 불안정하게 만들지 않도록 하는 것**이다.

## 12. 오픈소스 스택으로 구성하기

대표적인 오픈소스 조합은 다음과 같다.

```text
Client
  → Ingress / Gateway
  → FastAPI Public API
  → 인증·rate limit·model routing
  → Ray Serve deployment
  → vLLM replica
  → GPU

Kubernetes: 배포·네트워킹·자원·스케일링 기반
KubeRay: Ray cluster와 RayService의 Kubernetes 수명주기 관리
Observability: metrics·logs·traces
```

### 12.1 Kubernetes가 담당하는 기반

- 컨테이너 배포와 replica 유지
- Service, Ingress/Gateway, load balancing
- CPU·메모리·GPU request/limit과 scheduling
- health probe와 재시작
- HPA 또는 별도 autoscaler
- Secret, RBAC, NetworkPolicy
- 모니터링·로깅 생태계 연동

Kubernetes는 모델의 토큰 스케줄링이나 KV Cache를 직접 관리하지 않는다. 클러스터 자원을 관리하는 Kubernetes와 모델 실행을 최적화하는 serving engine의 책임을 구분해야 한다.

### 12.2 Public API 계층

FastAPI 같은 API 계층에서는 다음을 담당할 수 있다.

```python
@app.post("/v1/chat/completions")
async def chat(req: ChatRequest, identity=Depends(require_auth)):
    await rate_limit(identity.tenant)
    model = select_model(req, identity)
    return await route_request(model, req)
```

- API key 또는 JWT 검증
- tenant별 quota와 rate limit
- 입력 schema와 최대 token 검증
- model alias, canary, fallback
- request ID와 tracing context 생성

### 12.3 Ray Serve와 KubeRay

- **Ray Serve:** Python 기반 deployment, replica, routing, autoscaling을 제공한다.
- **Ray Core:** 분산 task와 actor 실행 기반을 제공한다.
- **KubeRay:** RayCluster, RayJob, RayService 같은 Kubernetes CRD와 operator를 제공한다.
- **RayService:** Serve 애플리케이션과 Ray cluster의 상태를 함께 선언하고 health recovery와 rolling update를 지원한다.

멀티 모델에서는 Ray Serve의 multiplexing 기능으로 replica 안에 여러 모델을 동적으로 로드하고 LRU 방식으로 관리할 수 있다. 이는 앞에서 직접 설계한 `ModelManager + worker cache`를 프레임워크 수준에서 제공하는 것으로 볼 수 있다.

다만 다음을 확인해야 한다.

- 모델 cold start가 SLO를 만족하는가?
- replica별 모델 cache 상태를 라우터가 활용하는가?
- autoscaling 기준이 요청 수인지, queue 길이인지, 실행 중 request인지?
- 모델 제거 시 실제 GPU 메모리가 회수되는가?
- rolling update 중 긴 streaming request를 어떻게 drain하는가?

## 13. 클라우드 벤더의 6가지 구축 선택지

AWS의 예시는 완전 관리형에서 완전 자체 구축으로 갈수록 통제권과 운영 책임이 함께 증가한다는 점을 보여준다.

| 단계 | 방식 | 사용자가 주로 통제하는 것 | 운영 부담 |
| --- | --- | --- | --- |
| 1 | Amazon Bedrock | 모델 선택과 API 사용 | 매우 낮음 |
| 2 | SageMaker JumpStart | 준비된 모델·배포 설정 | 낮음 |
| 3 | Bring Your Own Model | 자신의 model artifact | 중간 |
| 4 | Bring Your Own Code | 전후처리·로딩·추론 코드 | 높음 |
| 5 | Bring Your Own Serving Image | OS·CUDA/ROCm·라이브러리·runtime·API 구현 | 매우 높음 |
| 6 | Build Your Own Infrastructure | Kubernetes부터 serving platform 전체 | 최고 |

### 13.1 Option 1: Bedrock

- 인프라와 모델 호스팅을 직접 관리하지 않는다.
- API로 지원 모델을 빠르게 사용한다.
- time-to-market과 낮은 운영 부담이 가장 중요할 때 적합하다.
- 모델·runtime·세부 최적화에 대한 통제는 제한된다.

### 13.2 Option 2: SageMaker JumpStart

- 준비된 모델과 template로 전용 endpoint를 배포한다.
- Bedrock보다 배포 자원과 설정을 더 제어할 수 있다.
- 지원 모델과 제공되는 container 범위에 영향을 받는다.

### 13.3 Option 3: Bring Your Own Model

- 지원되는 serving container에 자신의 model artifact를 넣는다.
- 모델은 통제하지만 container의 framework, API 계약, runtime 버전에 의존한다.

### 13.4 Option 4: Bring Your Own Code

- 관리형 container를 사용하면서 model loading, preprocessing, prediction, postprocessing 코드를 작성한다.
- 표준 runtime을 유지하면서 비즈니스 로직을 바꾸고 싶을 때 유용하다.
- container 내부의 기반 버전과 endpoint 계약은 여전히 제약이 된다.

### 13.5 Option 5: Bring Your Own Serving Image

- container image 전체를 직접 구성한다.
- 원하는 언어, serving runtime, 시스템 dependency, API 구현을 선택할 수 있다.
- image 보안 패치, 호환성, health check, 디버깅 책임이 사용자에게 넘어온다.
- SageMaker가 endpoint 배포와 인스턴스 관리는 계속 담당한다.

### 13.6 Option 6: Build Your Own Infrastructure

- EKS 같은 기반 위에 gateway, runtime, autoscaling, observability, 보안을 직접 구성한다.
- custom scheduling, cache-aware routing, 특수 하드웨어, 엄격한 격리가 필요할 때 적합하다.
- 가장 높은 자유도를 얻지만 플랫폼 전체의 가용성과 비용을 직접 책임져야 한다.

이 선택지는 성숙도 순위가 아니다. 요구하지도 않는 자유도를 얻기 위해 운영 복잡도를 떠안는 것은 좋은 설계가 아니다.

## 14. Build or Buy는 스펙트럼이다

대부분의 팀은 완전 관리형과 완전 자체 구축의 중간에 위치한다. 예를 들어 벤더 endpoint를 사용하면서 라우팅, custom handler, autoscaling signal, observability만 직접 구현할 수 있다.

### 관리형을 유지하기 좋은 조건

- 현재 서비스가 latency·throughput SLO를 만족한다.
- 비용이 허용 범위 안이다.
- 제품 출시 속도가 가장 중요하다.
- 플랫폼 운영 인력이 제한적이다.

### 하이브리드가 적합한 조건

- 일부 endpoint에만 특수 배칭이나 라우팅이 필요하다.
- tenant별 격리·정책을 추가해야 한다.
- 공통 플랫폼은 유지하면서 병목 부분만 교체하고 싶다.

### 자체 구축 범위를 넓힐 조건

- hardware와 runtime을 세밀하게 통제해야 한다.
- 충분히 큰 처리량에서 직접 운영이 비용상 유리하다.
- managed service가 제공하지 않는 최적화가 필요하다.
- 엄격한 네트워크·데이터·규정 준수 요구가 있다.
- 멀티 클라우드 또는 vendor lock-in 회피가 중요하다.

### 다시 관리형으로 돌아갈 조건

- 트래픽이 작고 안정되어 자체 플랫폼의 고정비가 더 크다.
- custom stack의 복잡도가 실제 성능·비용 이익으로 이어지지 않는다.
- 유지보수 부담이 제품 개발을 방해한다.

좋은 전략은 한 번 선택한 위치를 고수하는 것이 아니라, **안정된 API와 공통 텔레메트리를 유지한 채 요구사항에 따라 스펙트럼 위에서 이동할 수 있게 만드는 것**이다.

## 15. LLM 서빙 성능 지표

### 15.1 E2E Latency

요청을 보낸 시점부터 전체 응답이 끝날 때까지의 시간이다. 측정 경계에 따라 다음을 포함할 수 있다.

- 네트워크
- 인증과 라우팅
- queue 대기
- 토크나이징
- prefill과 decode
- 디토크나이징과 streaming 전송

측정 위치를 명시하지 않으면 서로 다른 E2E 숫자를 비교할 수 없다.

### 15.2 TTFT

Time to First Token은 요청 수신부터 첫 출력 토큰을 받을 때까지의 시간이다. 대체로 입력 처리와 prefill, 첫 decode step의 영향을 크게 받는다.

TTFT에 영향을 주는 대표 요소는 다음과 같다.

- 입력 token 길이
- queue 대기시간
- prefill batching
- model parallel 통신
- prefix cache hit 여부

스트리밍 챗봇에서는 사용자가 “응답이 시작됐다”고 느끼는 시점을 결정하므로 중요하다.

### 15.3 ITL / TPOT

Inter-Token Latency 또는 Time Per Output Token은 첫 토큰 이후 다음 토큰을 생성하는 간격을 나타낸다. decode 효율과 관련이 크다.

- KV Cache 접근
- GPU memory bandwidth
- decode batch 크기
- 다른 요청과의 자원 간섭
- sampling과 후처리

출력이 길수록 작은 ITL 차이가 E2E에 누적된다.

일정한 ITL과 출력 token 수 `N`을 가정하면 다음과 같이 근사할 수 있다.

```text
E2E ≈ TTFT + ITL × (N - 1)
```

예를 들어 TTFT가 `0.8초`, ITL이 `0.04초`, 출력이 `100 tokens`라면 다음과 같다.

```text
E2E ≈ 0.8 + 0.04 × 99 = 4.76초
```

### 15.4 유스케이스별 우선순위

| 유스케이스 | 우선 지표 | 이유 |
| --- | --- | --- |
| 스트리밍 챗봇 | TTFT, ITL | 빠른 시작과 자연스러운 출력 속도 |
| 여러 단계를 직렬 실행하는 에이전트 | E2E | 앞 단계 완료가 다음 단계를 막음 |
| 긴 문서 생성 | ITL/TPOT, E2E | token 간 작은 지연이 크게 누적 |
| offline batch | TPS, 비용/token | 사용자 체감보다 총 처리 효율이 중요 |
| 짧은 분류 API | p95/p99 E2E, RPS | 요청 단위 SLA가 중요 |

### 15.5 RPS / RPM

단위 시간에 완료한 요청 수다. API capacity를 설명하기 쉽지만, LLM 요청마다 입력과 출력 길이가 다르므로 조건 없이 비교하면 오해하기 쉽다.

```text
1 RPS of 50 input + 10 output tokens
≠
1 RPS of 8,000 input + 1,000 output tokens
```

### 15.6 TPS

원문에서는 TPS를 초당 생성한 **출력 token 수**로 정의한다. 다만 도구에 따라 input TPS, output TPS, total TPS를 다르게 표시할 수 있으므로 보고서에는 정의를 명시해야 한다.

TPS도 측정 조건에 따라 크게 달라진다.

- 짧은 입력은 prefill 부담을 줄인다.
- 균일한 길이는 padding과 straggler 낭비를 줄인다.
- 큰 배치는 GPU 활용률을 높이지만 latency를 악화시킬 수 있다.
- concurrency와 KV Cache 한도가 결과를 바꾼다.

따라서 “모델 A는 1,000 TPS”라는 문장만으로는 성능을 판단할 수 없다.

## 16. 신뢰할 수 있는 벤치마크를 위한 체크리스트

### 16.1 먼저 SLO를 정의한다

평균만 보지 말고 workload에 맞는 목표를 정한다.

- TTFT p50/p95/p99
- E2E p50/p95/p99
- 최소 output TPS 또는 request throughput
- 최대 오류율과 timeout 비율
- 요청당 또는 1M tokens당 비용

### 16.2 실제 트래픽 분포를 재현한다

- input/output 길이의 분포
- 동시 사용자와 burst
- streaming/non-streaming 비율
- model별 요청 비율
- 반복 prefix와 cache hit 비율
- tenant별 우선순위

평균 길이 하나로 고정한 synthetic workload는 비교 실험에는 유용하지만 실제 capacity 예측에는 부족하다.

### 16.3 한 번에 한 변수만 바꾼다

다음 항목을 동시에 바꾸면 원인을 분리할 수 없다.

- model 또는 quantization
- batch/concurrency
- context length
- cache 설정
- serving engine version
- hardware와 clock/power 설정

baseline을 고정하고 한 항목씩 바꾼 뒤 반복 측정해야 한다.

### 16.4 warm-up과 cache 상태를 구분한다

- model load를 포함한 cold start
- kernel compile 이후 warm state
- prefix cache cold/hit
- OS page cache와 model artifact cache

어떤 상태를 측정했는지 기록하지 않으면 결과를 재현하기 어렵다.

### 16.5 하드웨어 활용률도 함께 본다

- GPU utilization
- VRAM과 KV Cache 사용량
- CPU와 host memory
- PCIe 또는 interconnect 사용량
- network throughput
- power와 thermal throttling

latency가 나빠졌을 때 GPU 계산, 메모리, CPU 전처리, queue, 네트워크 중 어디가 병목인지 구분해야 한다.

### 16.6 회귀 테스트로 만든다

새 engine·driver·모델 버전 또는 설정 변경 뒤 같은 workload를 다시 실행한다. 성능 결과를 일회성 표가 아니라 배포 판단에 사용하는 regression suite로 관리하는 편이 좋다.

### 16.7 최소 기록 항목

```text
Model / revision / precision
Serving engine / version
Hardware / accelerator count
Input and output length distribution
Concurrency / request rate
Batch and scheduling configuration
Cache state
Streaming mode
TTFT, ITL/TPOT, E2E percentiles
RPS and input/output TPS
Error and timeout rate
Peak memory and utilization
```

## 17. 과제와 후속 실습

원문의 과제 취지는 학습 내용을 공개 글로 정리하고, 선택한 서빙 구성을 직접 검증해 보는 것이다. 현재 R9700 환경에서 단일 GPU 서빙, 멀티 모델 수명주기, RayService, Kubernetes GPU 연결을 차례로 검증했다.

### 완료한 과제 1: R9700 비동기 호출·vLLM 스케줄링 비교

Qwen3-4B BF16을 vLLM으로 서빙하고, 같은 요청을 다음 네 경로로 전달했다.

- `blocking`: `async def` 내부에서 동기 HTTP 호출을 직접 실행하는 negative control
- `threaded`: 같은 동기 호출을 `asyncio.to_thread()`로 격리
- `async`: `httpx.AsyncClient`로 upstream까지 non-blocking 호출
- `direct`: gateway 없이 vLLM OpenAI endpoint 직접 호출

vLLM은 다음 두 profile로 재기동했다. 다른 조건은 고정하고 scheduling 한도만 변경했다.

| Profile | `max_num_seqs` | `max_num_batched_tokens` |
| --- | ---: | ---: |
| constrained | 4 | 2,048 |
| balanced | 16 | 8,192 |

동시성 16, 요청당 출력 64 tokens 조건에서 얻은 결과는 다음과 같다.

| API 경로 — balanced | Wall time | output tok/s | E2E p50/p95 |
| --- | ---: | ---: | ---: |
| blocking | 15.468초 | 66.2 | 8.238/14.726초 |
| threaded | 1.602초 | 639.0 | 1.584/1.595초 |
| async | 1.591초 | 643.6 | 1.572/1.585초 |
| direct | 1.580초 | 648.0 | 1.563/1.574초 |

blocking 대비 async 경로의 output TPS는 `872.2%` 증가했다. 동기 호출이 단일 event loop를 막아 요청을 사실상 직렬화한 반면, threaded와 async 경로는 요청을 vLLM에 동시에 전달해 continuous batching 기회를 만들었다.

direct endpoint로 호출 경로를 고정한 scheduling 비교 결과는 다음과 같다.

| Profile | output tok/s | E2E p95 |
| --- | ---: | ---: |
| constrained | 233.6 | 4.363초 |
| balanced | 648.0 | 1.574초 |

balanced profile은 constrained profile보다 output TPS가 `177.4%` 증가하고 E2E p95는 `63.9%` 감소했다. 이는 더 큰 한도가 항상 좋다는 뜻이 아니라, 이번 동시성·출력 길이에서 `max_num_seqs=4`가 병목이었다는 의미다. 메모리 압박이 큰 긴 context workload에서는 같은 설정의 결과가 달라질 수 있다.

- 실행 방법: [`./experiments/README.md`](./experiments/README.md)
- 전체 결과: [`./experiments/results/serving-modes-latest.md`](./experiments/results/serving-modes-latest.md)

### 완료한 과제 2: 멀티 모델 lazy loading·LRU

두 Hugging Face 모델을 사용해 실제 GPU 모델 cache를 구현했다.

| 별칭 | 모델 | 정밀도 | 용도 |
| --- | --- | --- | --- |
| `small` | Qwen3-0.6B | BF16 | 작은 모델과 cache hit 확인 |
| `base` | Qwen3-4B-Instruct-2507 | BF16 | 큰 모델 교체와 VRAM 회수 확인 |

Qwen3-0.6B는 revision `c1899de289a04d12100db370d81485cdf75e47ca`, Qwen3-4B-Instruct-2507은 revision `cdbee75f17c01a7cc42f958dc650907174af0554`로 고정했다. 모델 원본은 NFS에 보존하고, 저장장치 지연 변수를 줄이기 위해 실험은 로컬 SSD 복사본으로 수행했다. 공개 문서에서는 다음 상대경로만 사용한다.

```text
./models/Qwen3-0.6B
./models/Qwen3-4B-Instruct-2507
```

cache capacity를 1로 제한하고 `small → small → base → small` 순서로 접근했다.

| 단계 | 모델 | Cache | 제거된 모델 | Load | Inference | Load 후 allocated VRAM |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | small | miss | - | 4.895초 | 4.574초 | 1.110GiB |
| 2 | small | hit | - | 0초 | 0.849초 | 1.185GiB |
| 3 | base | miss | small | 9.084초 | 1.153초 | 7.620GiB |
| 4 | small | miss | base | 2.721초 | 0.895초 | 1.185GiB |

관찰한 내용은 다음과 같다.

1. 동일한 `small`의 두 번째 요청은 cache hit이므로 모델 load 시간이 발생하지 않았다.
2. 첫 추론은 초기화·warm-up 비용까지 포함해 이후 cache hit 추론보다 느렸다.
3. capacity가 1이므로 다른 모델을 요청할 때 기존 모델이 LRU로 제거됐다.
4. `small` 제거 후 reserved VRAM은 `1.207GiB → 0.074GiB`, free VRAM은 `30.244GiB → 31.377GiB`로 변했다.
5. `base` 제거 후 reserved VRAM은 `7.621GiB → 0.074GiB`, free VRAM은 `23.830GiB → 31.377GiB`로 변했다.
6. 실험 종료 후 reserved VRAM은 `0.074GiB`까지 내려가 실제 메모리 회수를 확인했다.

동적 로딩은 유휴 VRAM을 줄이지만 cache miss마다 수 초의 load latency가 추가됐다. 따라서 실제 서비스에서는 단순 LRU뿐 아니라 모델 크기, 요청 빈도, load 비용, SLO, pin 정책을 함께 사용해야 한다.

- 구현: [`./experiments/multi_model_lru.py`](./experiments/multi_model_lru.py)
- 전체 결과: [`./experiments/results/multi-model-lru-latest.md`](./experiments/results/multi-model-lru-latest.md)

### 대안 환경 및 비교 실험 1: k3s에 CPU RayService 배포

Kind 과제에 앞서 대안 환경 및 비교 실험으로 비특권 LXD 컨테이너 안에 단일 노드 k3s를 구성했다. Docker node 계층 없이 systemd 기반 k3s, KubeRay, GPU device를 한 실습 환경에서 단계적으로 관찰할 수 있기 때문이다.

| 구성요소 | 버전 |
| --- | --- |
| k3s | `v1.34.9+k3s1` |
| KubeRay operator | `1.6.0` |
| Ray | `2.52.0` |

fruit와 calculator 애플리케이션을 RayService로 배포한 결과 Service 상태는 `Running`, Serve endpoint는 2개였다.

```text
POST /fruit/ ["MANGO", 2] → 6
POST /calc/  ["MUL", 3]   → 15 pizzas please!
```

RayCluster pod template을 변경하자 기존 cluster가 요청을 계속 처리하는 동안 새 cluster가 준비됐다. 60초 뒤 Service selector가 새 cluster로 전환됐고, 이 과정에서 실행한 90회 요청은 모두 성공했다.

worker 복구 시험에서는 head의 논리 CPU를 0으로 설정해 Serve replica가 worker에 반드시 배치되게 했다. worker pod를 삭제하자 50초 뒤 새 worker가 Ready가 됐다. 다만 단일 worker의 모든 replica를 제거했기 때문에 측정 중 14회의 timeout이 발생했다. 자동 복구가 성공해도 고가용성이 자동으로 보장되는 것은 아니다.

- 실행 방법: [`./experiments/k3s-rayservice/README.md`](./experiments/k3s-rayservice/README.md)
- 전체 결과: [`./experiments/k3s-rayservice/results/summary.md`](./experiments/k3s-rayservice/results/summary.md)

### 대안 환경 및 비교 실험 2: k3s에 R9700 Kubernetes GPU 연결

R9700과 `/dev/kfd`를 실습 container에 전달하고 AMD Device Plugin을 배포했다.

```text
Node capacity:    amd.com/gpu=1
Node allocatable: amd.com/gpu=1
ROCm agent:       gfx1201
HIP result:       42.0
```

GPU limit을 요청한 Pod 안에서 `/dev/kfd`, DRM render node, `gfx1201`을 확인했고 작은 HIP kernel을 직접 실행했다. 이어서 Ray worker에 `amd.com/gpu=1`과 `num-gpus=1`을 함께 설정했다. `num_gpus=1` remote task는 해당 worker에서 실행됐고 `ray_gpu_ids=[0]`과 GPU device node를 확인했다.

마지막으로 Qwen3-0.6B BF16을 vLLM `0.26.0`으로 실행해 Kubernetes Service를 통한 `/v1/models`와 `/v1/chat/completions` 요청이 모두 HTTP 200으로 완료되는 것을 확인했다.

| 항목 | 결과 |
| --- | --- |
| Model load | 1.12GiB, 4.26초 |
| `torch.compile` | 47.43초 |
| Engine init | 86.29초 |
| GPU KV cache | 131,904 tokens |

실험용 vLLM Pod는 이미 검증한 runtime과 ROCm user-space를 읽기 전용 volume으로 재사용했다. 따라서 Kubernetes와 GPU 연결을 검증하기에는 충분하지만 production 배포에서는 dependency 전체를 immutable image에 포함해야 한다.

### 완료한 과제 3: Kind에 CPU RayService와 R9700 연결

Docker container를 Kubernetes node로 사용하는 Kind에서 핵심 과제를 수행했다. 앞선 k3s 결과는 대안 환경 및 비교 자료로만 사용한다.

```text
host → non-privileged LXD → Docker → Kind node → Kubernetes Pod
```

| 구성요소 | 버전 |
| --- | --- |
| Kind | `v0.32.0` |
| Kubernetes | `v1.36.1` |
| Docker | `29.7.2` |
| KubeRay operator | `1.6.0` |
| Ray | `2.52.0` |

기본 구성의 첫 Kind node는 비특권 user namespace에서 kubelet이 `/dev/kmsg`를 열지 못해 실패했다. LXD를 privileged로 바꾸지 않고 Kind의 `KubeletInUserNamespace=true` feature gate를 적용하자 control-plane이 `Ready`가 됐다.

CPU RayService endpoint 결과는 k3s와 같았다.

```text
POST /fruit/ ["MANGO", 2] → 6
POST /calc/  ["MUL", 3]   → 15 pizzas please!
```

RayCluster 설정 교체는 52초에 완료됐고 관측한 52회 요청은 모두 성공했다. 단일 worker 삭제 후 복구에는 47초가 걸렸으며 8회 성공·15회 실패가 발생했다.

R9700은 LXD device와 Kind `extraMounts`를 거쳐 node container에 전달했다. AMD Device Plugin은 `amd.com/gpu=1`을 등록했고 GPU Pod에서 `gfx1201`, `HIP_RESULT=42.0`을 확인했다. Ray `num_gpus=1` task도 `ray_gpu_ids=["0"]`과 `/dev/kfd`를 확인했다.

Kind에서는 vLLM을 반복하지 않았다. k3s에서 Qwen3-0.6B endpoint를 이미 검증했고, Kind에 추가된 Docker node 경계는 HIP kernel과 Ray GPU task로 실제 compute까지 통과했기 때문이다.

- 실행 방법: [`./experiments/kind-rayservice/README.md`](./experiments/kind-rayservice/README.md)
- 전체 결과: [`./experiments/kind-rayservice/results/summary.md`](./experiments/kind-rayservice/results/summary.md)

### 후속 과제: AWS EKS에 Ray Serve를 설치하고 Chat with Mistral 수행

**목표**

- EKS와 GPU node group 준비
- KubeRay와 RayService 배포
- Mistral 계열 모델 endpoint 실행
- streaming 요청과 기본 성능 측정
- 실습 후 비용이 발생하는 리소스 제거

**주의할 점**

- GPU instance quota와 비용
- 모델 라이선스와 접근 권한
- IAM/IRSA, Secret, private networking
- model download 시간과 persistent cache
- node provisioning과 model load가 포함된 cold start
- 실습 종료 후 node group, load balancer, volume 등 잔여 자원 확인

### 추가 확장 과제

현재 실험을 발전시키려면 같은 모델과 요청 집합으로 다음 항목을 비교할 수 있다.

1. 정적 배칭과 continuous batching
2. single-model deployment와 Ray Serve multiplexed multi-model deployment
3. LRU와 모델 크기·load 비용을 반영한 weighted eviction
4. request-based autoscaling과 queue-based autoscaling
5. cache-unaware routing과 prefix-cache-aware routing

실험마다 가설, 고정 조건, 변경 변수, 결과, 해석, 한계를 함께 기록한다.

## 18. 스스로 답해볼 핵심 질문

1. API Server, Workload Manager, Model Worker를 하나의 프로세스로 합치면 무엇이 단순해지고 무엇이 위험해지는가?
2. 배치 크기를 늘렸을 때 TPS는 좋아졌는데 TTFT p99가 나빠졌다면 어떤 선택을 해야 하는가?
3. 스트리밍 연결이 끊겼을 때 GPU의 generation 작업도 즉시 취소되는가?
4. 멀티 모델 LRU에서 “최근 사용”만으로 제거 대상을 정하면 어떤 문제가 생기는가?
5. 비용 최적화 공유 풀과 지연시간 최적화 전용 풀을 어떤 모델 기준으로 나눌 것인가?
6. 에이전트 workflow에서 단계별 TTFT보다 E2E가 더 중요해지는 이유는 무엇인가?
7. Kubernetes, Ray Serve, vLLM은 각각 어떤 책임을 맡고 있으며 서로 무엇을 대신하지 못하는가?
8. 관리형 서비스를 떠나 자체 구축할 만큼 중요한 요구사항과 정량적 근거가 있는가?
9. TPS를 비교할 때 반드시 고정하거나 공개해야 할 workload 조건은 무엇인가?
10. 평균 latency가 동일해도 p99 latency가 다른 두 시스템 중 어느 쪽이 프로덕션에 더 적합한가?

## 마무리

단일 모델 서비스에서 시작해도 곧 큐, 배치, 스트리밍, 프로세스 격리, 모델 수명주기라는 문제가 나타난다. 모델 수가 늘면 cache와 routing이 필요하고, 에이전트와 엔터프라이즈 환경으로 확장하면 인증, 자원 관리, 분산 실행, 비용, 조직 경계까지 설계 대상이 된다.

이 과정에서 중요한 것은 모든 기능을 직접 만드는 것이 아니다. 직접 구현을 통해 추상화 아래의 원리를 이해하고, vLLM·Triton·Ray Serve·Kubernetes·클라우드 관리형 서비스가 **어떤 책임을 대신하고 어떤 책임은 여전히 사용자에게 남기는지** 판단할 수 있어야 한다.

마지막으로 아키텍처 선택은 반드시 측정으로 검증해야 한다. E2E, TTFT, ITL/TPOT, RPS, TPS를 workload와 함께 기록하고, 목표 SLO와 비용 안에서 가장 단순하게 운영 가능한 설계를 선택하는 것이 핵심이다.
