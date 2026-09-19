# 단일 GPU 과부하에서 SLO를 지키는 llm-d Flow Control

> **도전과제:** 처리 한계가 약 5.45 RPS인 단일 R9700 vLLM에 12 RPS open-loop 트래픽을 지속해서 넣고, llm-d의 bounded queue·priority band·tenant fairness로 interactive 요청의 SLO를 보호한다.
>
> **최종 판정: PASS.** 2026년 9월 19일 실제 GPU 실험에서 interactive 96건은 모두 완료됐고 TTFT p95는 직접 호출의 13.35초에서 0.48초로 줄었다. 초과한 best-effort 요청 78건은 EPP가 `429 rejected-saturated`로 명시적으로 차단했다.

## 먼저 보는 결과

1. **실측한 backend 용량을 admission 기준으로 사용했다.** 동일 모델·GPU에서 확인한 처리량 knee와 vLLM의 `max_num_seqs=16`을 근거로 EPP `maxConcurrency=16`을 설정했다.
2. **부하는 closed-loop 동시성 시험이 아니라 12 RPS open-loop로 생성했다.** 요청 완료와 무관하게 도착률을 유지했으며 client launch lag p99는 3.6ms였다.
3. **llm-d 중앙 queue가 실제로 동작했다.** 혼합 부하 EPP metric에서 queue 최대 6건, saturation 최대 1.0625를 관측했다.
4. **queue는 무한히 증가하지 않았다.** 전역 한도는 32건이고 전체 시나리오의 실측 최대는 16건이었다.
5. **interactive 트래픽을 모두 살렸다.** priority 100 요청은 96/96 성공했지만 best-effort priority -10은 66/144만 완료됐다.
6. **interactive TTFT p95는 13.35초에서 0.48초로 96.4% 감소했다.** 대기 자체가 사라진 것이 아니라, 처리할 수 없는 batch를 조기에 거절하고 허용된 대기를 EPP로 이동한 결과다.
7. **과부하를 숨기지 않았다.** 혼합 부하에서 78건을 HTTP 429와 `x-llm-d-request-dropped-reason: rejected-saturated`로 반환했다.
8. **처리량 비용도 함께 기록했다.** 전체 output TPS는 675.2에서 637.4로 5.6% 낮아졌다. SLO 보호는 무료 최적화가 아니라 명시적인 서비스 정책이다.
9. **9:1 noisy-neighbor 부하에서도 quiet tenant가 굶지 않았다.** 제시량의 10%였던 quiet flow가 전체 완료량의 11.01%를 확보해 제시 비율을 보존했다.
10. **단일 backend의 한계를 과장하지 않았다.** 이번 결과는 Flow Control 검증이며 load-aware endpoint 선택, prefix-aware routing 이득, failover, autoscaling을 증명하지 않는다.

---

## 1. 실습 목표와 범위

이번 실습의 핵심 질문은 **backend 용량을 초과하는 트래픽이 도착할 때 어떤 요청을 대기시키고, 먼저 처리하고, 거절할 것인가**이다. 같은 open-loop 요청열을 vLLM에 직접 보내는 기준선과 llm-d Flow Control 경로에 각각 보내 정책 효과를 비교했다.

| 구분 | Direct 기준선 | llm-d Flow Control |
|---|---|---|
| 요청 경로 | client → vLLM | client → Envoy/EPP → vLLM |
| queue 위치 | vLLM 내부 waiting queue | llm-d EPP 중앙 bounded queue |
| 과부하 처리 | 도착한 요청을 계속 대기 | saturation gate + priority + fairness |
| 실패 표현 | client가 오래 기다림 | 429와 drop reason으로 빠른 load shedding |
| 비교 기준 | 같은 12 RPS open-loop 요청열 | 같은 12 RPS open-loop 요청열 |
| 성공 기준 | 비교용 latency·throughput 확보 | interactive SLO·queue bound·비기아 검증 |

실험 환경의 전제 조건은 단일 GPU vLLM endpoint와 Prometheus가 준비된 Kubernetes cluster다. 이 문서의 검증 범위는 cluster 구축이 아니라 Router 배포, 과부하 생성, Flow Control 정책, 결과 재계산이다.

## 2. 원문 도전과제와 실습 범위

도전과제 목록에 명시된 **“llm-d에 Flow Control 실습 및 분석 정리해보기”**를 그대로 선택했다. 따라서 이 실험은 원문과 비슷한 주제를 임의로 만든 것이 아니라, 명시된 도전과제를 실제 배포·부하·측정·판정까지 확장해 수행한 것이다.

- admission, queuing, dispatch
- `FlowKey = fairness ID + priority`
- priority band `100 / 0 / -10`
- 같은 band 안의 round-robin fairness
- concurrency saturation detector
- bounded request/byte queue와 TTL
- EPP metric 및 client-visible drop reason

| 원문 항목 | 이번 결과의 대응 | 판정 |
|---|---|---|
| llm-d Flow Control 실습 및 분석 | 실제 EPP/Envoy 배포, priority·fairness·bounded queue 설정, GPU 과부하 실험, metric과 request row 분석 | **직접 수행** |
| llm-d 공식 문서 내용 정리 | 선택한 Flow Control의 구조·설정·header·metric·운영상 주의점 정리 | **선택 주제 범위에서 수행** |
| Agent Router Traffic Handling | `QuotaPolicy`와 llm-d admission의 역할 차이 및 trust boundary만 비교 | **개념 비교만 수행** |
| Load-Aware·Predicted Latency·Tiered Prefix Cache·P/D·Autoscaling | 단일 backend 실험으로는 검증하지 않음 | **범위 밖** |

Agent Router의 Traffic Handling은 별도의 도전과제다. `QuotaPolicy`는 Redis와 별도 Envoy Gateway stack을 사용해 완료된 token budget을 누적 제한하는 기능이고, GPU saturation 앞의 in-flight admission과는 목적이 다르므로 이번 완료 범위로 주장하지 않는다. 대신 production에서는 인증된 상위 gateway가 tenant와 priority header를 주입해야 한다는 trust boundary만 명시했다.

### 사용 도구 대조

| 구성 요소 | 원문에서 다룬 내용 | 이번 실습 | 판단 |
|---|---|---|---|
| llm-d Router EPP | Flow Control 실행 주체 | 공식 EPP `v0.10.0` | **동일 도구** |
| Flow Control 설정 | `EndpointPickerConfig`, `flowControl` feature gate | 같은 API와 feature gate 사용 | **동일 방식** |
| 정책 plugin | round-robin fairness, FCFS, concurrency detector | 세 plugin을 그대로 사용 | **동일 plugin** |
| 요청 분류 | fairness ID와 inference objective header | 같은 llm-d header 사용 | **동일 방식** |
| Proxy | Envoy와 EPP의 ext-proc 연동 | 공식 standalone chart의 Envoy sidecar 사용 | **동일 data plane, 간소화된 배포 형태** |
| Gateway API Inference Extension | `InferencePool` 등 inference routing API | `InferencePool`·`InferenceObjective`, CRD `v1.5.0` 사용 | **동일 API 계열** |
| Model server | Kubernetes의 vLLM backend | R9700의 Qwen3-4B vLLM `0.26.0` | **동일 engine, 실험 환경별 model** |
| 부하 도구 | 별도 고정 도구 지정 없음 | dependency-free Python open-loop generator | **검증용 자체 도구** |

원문의 Router 예시처럼 full Envoy Gateway controller와 Agent Router controller를 함께 배포하지는 않았다. Flow Control 자체를 검증하는 데 필요한 llm-d EPP와 Envoy ext-proc 경로는 공식 `llm-d-router-standalone` chart로 구성했다. `maxConcurrency`, queue 크기, TTL 같은 수치는 예시값을 복사하지 않고 실제 backend 용량과 짧은 과부하 실험에 맞게 조정했다.

## 3. 아키텍처

```text
Open-loop load generator
  ├─ direct baseline ─────────────────────┐
  └─ llm-d path                           │
       └─ Envoy sidecar                   │
            └─ EPP Flow Control           │
                 ├─ objective → priority  │
                 ├─ fairness ID → flow    │
                 ├─ bounded queue         │
                 └─ concurrency gate      │
                                             ↓
                                  Qwen3-4B vLLM Pod
                                             ↓
                                  AMD Radeon AI PRO R9700

Prometheus ← EPP /metrics + vLLM /metrics + AMD GPU exporter
```

Kubernetes Gateway는 설치하지 않았다. 공식 standalone chart가 EPP와 Envoy sidecar를 함께 제공하며, 같은 namespace의 `InferencePool`이 기존 vLLM Pod 하나를 selector로 발견한다.

### 고정한 공식 버전

| 구성 | 버전 |
|---|---|
| llm-d guide | `v0.9.0` |
| llm-d Router chart / EPP | `v0.10.0` |
| Gateway API Inference Extension CRD | `v1.5.0` |
| Envoy image | `distroless-v1.33.2` |
| Kubernetes | k3s `v1.34.9+k3s1` |
| vLLM | `0.26.0` |
| Model | Qwen3-4B-Instruct-2507 BF16 |

`latest`가 아니라 llm-d `v0.9.0` 가이드가 함께 고정한 조합을 사용했다. Router chart OCI digest도 환경 결과에 기록했다.

`EndpointPickerConfig`는 EPP startup 때 읽고 hot reload하지 않는다. 첫 조정 때 Helm ConfigMap만 32/16으로 바뀌고 기존 Pod가 24/8 설정을 계속 사용하는 문제를 발견했다. 배포 스크립트에 명시적 `rollout restart`를 추가했고, smoke test가 startup log의 전역 32건·standard 16건을 파싱해 실제 runtime 설정까지 확인하도록 했다. 아래 최종 결과는 이 수정 뒤 모두 다시 측정했다.

## 4. Flow Control 정책

### 4.1 포화 감지

```yaml
- type: concurrency-detector
  parameters:
    maxConcurrency: 16
    concurrencyMode: requests
    headroom: 0.0
```

open-loop burst에 즉시 반응하도록 concurrency detector를 사용했다. `16`은 임의의 숫자가 아니라 같은 모델과 GPU에서 확인한 vLLM active sequence 한계다.

### 4.2 bounded queue

| 항목 | 설정 |
|---|---:|
| 전역 최대 queue | 32 requests / 64MiB |
| 기본 queue TTL | 3초 |
| premium band | priority 100, 최대 12건 |
| standard band | priority 0, 최대 16건 |
| best-effort band | priority -10, 최대 4건 |
| 같은 band의 flow 선택 | round robin |
| flow 내부 순서 | FCFS |

priority band의 작은 한도는 중요한 트래픽의 queue 공간을 best-effort burst가 먼저 채우지 못하게 한다. queue가 가득 차면 EPP가 429를 반환한다. 최종 혼합 부하에서는 capacity rejection만 발생했고, 더 긴 공정성 부하에서는 3초 TTL 만료 5건도 `429 rejected-ttl-expired`로 기록됐다. 상태 코드만으로 두 원인을 구분하지 않고 drop-reason header를 함께 보존했다.

### 4.3 request classification

```http
x-llm-d-inference-objective: premium-traffic
x-llm-d-inference-fairness-id: tenant-a
```

이번 generator가 실험을 위해 header를 직접 보냈다. 실제 multi-tenant 환경에서 client가 자신의 priority를 정하게 두면 안 된다. 인증을 수행한 trusted gateway가 identity를 fairness ID로 변환하고 허용된 objective만 주입해야 한다.

## 5. 실험 설계

### 5.1 Priority/SLO 시나리오

| 항목 | 값 |
|---|---:|
| 도착률 | 12 RPS |
| 부하 지속시간 | 20초 |
| 총 요청 | 240 |
| interactive | 40%, priority 100, output 64 tokens |
| batch | 60%, priority -10, output 128 tokens |
| 비교 | vLLM direct vs llm-d Flow Control |

같은 순서와 prompt mix를 두 endpoint에 보냈다. 모든 요청이 끝난 뒤 다음 요청을 보내는 방식이 아니라, `request_id / RPS` 시점에 request를 제출했다.

### 5.2 Fairness 시나리오

| 항목 | 값 |
|---|---:|
| 도착률 | 12 RPS |
| 부하 지속시간 | 45초 |
| 총 요청 | 540 |
| noisy tenant | 90%, standard priority |
| quiet tenant | 10%, standard priority |
| 정책 | 같은 band 안에서 round robin |

처음 15초 표본에서는 quiet 18건뿐이어서 completion rate 동등성 판정이 불안정했다. standard band queue를 8건에서 16건으로 조정해 두 flow가 scheduler에 들어갈 여유를 확보하고, 45초·540건으로 표본을 늘렸다. 최종 판정은 비대칭 제시량에 맞게 **quiet tenant의 완료 비율이 제시 비율을 얼마나 보존했는가**로 정했다.

## 6. 결과

### 6.1 Direct vs Flow Control

| 지표 | Direct vLLM | llm-d Flow Control |
|---|---:|---:|
| Offered / completed | 240 / 240 | 240 / 162 |
| HTTP 429 | 0 | 78 |
| Overall output TPS | 675.2 | 637.4 |
| Overall TTFT p95 | 13.45초 | 2.01초 |
| Interactive completed | 96 / 96 | 96 / 96 |
| Interactive TTFT p95 | 13.35초 | **0.48초** |
| Interactive E2E p95 | 14.77초 | **1.95초** |
| Batch completed | 144 / 144 | 66 / 144 |
| Completed request ITL p95 | 24.02ms | 24.95ms |

Direct는 240건을 모두 처리했지만 GPU가 감당할 수 있는 속도보다 빨리 도착한 요청을 내부 queue에 계속 쌓았다. 그 결과 interactive도 batch와 함께 13초 이상 기다렸다.

Flow Control은 interactive 96건을 모두 통과시키면서 batch 78건을 빠르게 거절했다. 따라서 전체 completion rate만 보면 67.5%로 낮지만, 이는 장애가 아니라 선언한 우선순위 정책의 결과다. 허용된 전체 output TPS가 direct 대비 5.6% 감소했고 ITL p95가 3.9% 증가한 비용도 숨기지 않았다.

### 6.2 EPP queue 증거

| EPP metric | 혼합 부하 최대 | 공정성 부하 최대 |
|---|---:|---:|
| queue size | 6 | 16 |
| queue bytes | 1,870 | 4,976 |
| pool saturation | 1.0625 | 1.1875 |
| 설정한 global max requests | 32 | 32 |

queue와 saturation은 250ms 간격으로 별도 수집했다. client 결과만 보고 “queue가 있었을 것”이라고 추정하지 않았다.

### 6.3 Noisy-neighbor fairness

| Flow | Offered | Completed | Offered share | Completed share | TTFT p95 |
|---|---:|---:|---:|---:|---:|
| noisy tenant | 486 | 307 | 90.0% | 88.99% | 2.963초 |
| quiet tenant | 54 | 38 | 10.0% | 11.01% | 0.752초 |

quiet tenant는 제시 비율 10%보다 높은 11.01%를 완료 비율로 확보했다. 9배 많은 noisy flow가 존재해도 quiet flow가 사라지지 않았으며 TTFT p95도 더 낮았다. 다만 이는 이 workload에서의 결과이지 모든 arrival pattern에 대한 production fairness 보장은 아니다.

## 7. 도전과제 판정

| 검증 항목 | 기준 | 결과 |
|---|---|---|
| Open-loop 정확도 | launch lag p99 < 100ms | PASS, 3.6ms |
| 중앙 queue | EPP queue > 0 | PASS, 최대 6 |
| Queue bound | 관측값 ≤ 32 | PASS, 최대 16 |
| Queue byte bound | 관측값 ≤ 64MiB | PASS, 최대 4,976 bytes |
| Saturation signal | 최대값 ≥ 1.0 | PASS, 1.0625 |
| Metric scrape | 수집 오류 0건 | PASS |
| 명시적 load shedding | 429 또는 503 > 0 | PASS, 429 78건 |
| Priority 보호 | interactive completion rate ≥ batch | PASS, 100% vs 45.8% |
| Interactive TTFT 보호 | Flow p95 < Direct p95 | PASS, 0.48초 vs 13.35초 |
| Quiet flow 비기아 | 완료 비율이 제시 비율의 80% 이상 | PASS, 110.1% |

최종 `challenge-verdict.json`은 모든 항목을 **PASS**로 판정했다.

## 8. 재현과 검증

공개 패키지에는 README만이 아니라 다음 증거도 포함한다.

- Router values와 `InferenceObjective`
- 배포·smoke test·정리 script
- dependency-free open-loop generator
- request-level CSV 1,020건
- 집계 JSON과 250ms EPP metric sample
- CSV 재계산 verifier와 challenge evaluator
- 실행 환경 JSON, smoke log, SHA-256 checksum

기존 vLLM과 Prometheus가 실행 중인 환경에서 다음 순서로 수행한다.

```bash
./scripts/deploy-router.sh
./scripts/verify-router.sh
./scripts/run-challenge.sh
./scripts/capture-environment.sh
```

공개 결과의 집계값은 CSV에서 다시 계산하고 checksum도 확인할 수 있다.

```bash
python3 scripts/verify-results.py \
  --csv results/flow-mixed.csv \
  --summary results/flow-mixed.json

sha256sum -c results/SHA256SUMS
```

## 9. 배운 점과 경계

### Flow Control이 해결한 것

- 과부하 queue의 위치와 크기를 EPP에서 통제했다.
- priority가 높은 interactive 요청을 batch보다 먼저 dispatch했다.
- 같은 priority 안에서 flow 단위 fairness를 적용했다.
- capacity 초과를 빠른 429와 구체적인 reason으로 표현했다.

### 해결하지 않은 것

- GPU capacity 자체를 늘리지 않는다.
- 모든 요청의 TTFT를 줄이지 않는다. 누구의 대기를 허용할지 선택한다.
- queue는 EPP memory에 있으며 process restart를 넘겨 보존되지 않는다.
- single backend이므로 endpoint 선택 알고리즘의 이득을 비교할 수 없다.
- TTL 만료는 5건 관측했지만 client cancellation과 in-flight eviction은 결과로 만들지 않았다.

### 다음 확장

두 번째 GPU 또는 독립 backend가 생기면 load-aware scheduling과 prefix-aware routing을 같은 open-loop harness로 비교할 수 있다. 상위 traffic policy가 필요하면 Agent Router의 token quota를 별도 실험으로 두고, llm-d admission과 역할을 섞지 않는 편이 좋다.

## 참고 자료

- [llm-d Flow Control](https://llm-d.ai/docs/dev/architecture/core/router/epp/flow-control/)
- [llm-d EPP configuration](https://llm-d.ai/docs/dev/architecture/core/router/epp/configuration/)
- [llm-d EPP HTTP headers](https://llm-d.ai/docs/dev/api-reference/epp-http-headers/)
- [llm-d observability](https://llm-d.ai/docs/operations/observability/setup/)
- [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/)
- [Agent Router QuotaPolicy](https://theagentrouter.ai/docs/capabilities/traffic/quota-policy/)
