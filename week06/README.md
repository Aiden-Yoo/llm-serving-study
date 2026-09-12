# 단일 AMD GPU에서 관측 가능한 vLLM 서빙 플랫폼 구축

> 이 글의 목표는 AMD Radeon AI PRO R9700 한 장을 로컬 Kubernetes에 연결하고, Qwen3-4B를 vLLM으로 서빙한 뒤 GPU와 서비스 지표, 부하에 따른 포화 지점, Pod 복구 시간과 단일 GPU 자원 제약을 실제로 검증하는 것이다.
>
> **작성 상태:** 비특권 LXD 안에 단일 노드 k3s를 구성하고 AMD GPU 할당, HIP 계산, Qwen3-4B vLLM 배포, Prometheus·Grafana와 AMD GPU metric 수집, concurrency 1~64 부하 시험, warm-cache Pod 복구와 ResourceQuota 차단을 완료했다. 결과는 2026년 9월 13일 단일 시스템에서 얻은 값이며 다른 모델·GPU·runtime의 일반 성능을 의미하지 않는다.

## 먼저 보는 핵심 결과

1. **R9700은 Kubernetes extended resource로 정상 등록됐다.** Node의 `amd.com/gpu` capacity와 allocatable이 모두 1이었고 GPU Pod 안에서 `/dev/kfd`, `/dev/dri/renderD128`, `gfx1201`을 확인했다.
2. **실제 HIP 계산까지 성공했다.** Pod 안에서 실행한 작은 kernel이 `41.0 → 42.0` 결과를 반환했다. 장치 파일이 보이는 수준을 넘어 GPU compute 경로가 동작함을 확인했다.
3. **Qwen3-4B BF16을 vLLM 0.26.0으로 서빙했다.** `/v1/models`, `/v1/chat/completions`, `/health`, `/metrics`가 모두 정상 응답했다.
4. **Kubernetes의 legacy Service 환경 변수가 vLLM과 충돌했다.** 이름이 `vllm`인 Service가 `VLLM_PORT=tcp://...`를 Pod에 주입해 엔진 초기화를 실패시켰고, `enableServiceLinks: false`로 해결했다.
5. **Prometheus가 두 관측 계층을 모두 수집했다.** vLLM ServiceMonitor와 AMD Device Metrics Exporter target이 모두 `up`이었고 Grafana도 정상 기동했다.
6. **이번 설정의 처리량 포화 지점은 concurrency 16 부근이었다.** output TPS는 concurrency 16에서 697.2, 32에서 699.2, 64에서 696.0으로 더 이상 증가하지 않았다.
7. **포화 이후에는 처리량 대신 queue와 tail latency가 증가했다.** TTFT p95는 concurrency 16의 239.8ms에서 32의 3.17초, 64의 8.96초로 악화됐다.
8. **지속적인 concurrency 64 부하에서 vLLM은 16개를 실행하고 48개를 대기시켰다.** 이는 `max_num_seqs=16` 설정과 일치했다.
9. **같은 구간에서 GPU GFX activity는 100%에 도달했다.** exporter가 보고한 최대 사용 VRAM은 26,371MB, 평균 package power gauge의 최대 관측값은 308W였다.
10. **Pod 삭제 후 첫 health 성공까지 약 152.4초가 걸렸다.** Kubernetes Ready 확인까지는 153.5초였고 그 사이 116회의 health probe가 실패했다.
11. **compile cache를 재사용해도 재기동은 즉시 끝나지 않았다.** AOT artifact를 직접 불러왔지만 model load 12.73초, `torch.compile` 9.57초 외에 컨테이너 시작 시 package 설치가 복구 시간을 크게 늘렸다.
12. **HPA는 GPU 한 장이라는 물리 제약을 해결하지 못한다.** `ResourceQuota`로 GPU 총 요청을 1로 제한했고 두 번째 GPU Pod가 admission 단계에서 거부되는 것을 확인했다.

---

## 1. 왜 요약보다 운영 실습에 집중했는가

이전 학습에서는 KV Cache, batching, scheduler, quantization, framework와 scaling 원리를 단계적으로 다뤘다. 이번에는 개념을 더 늘리기보다 다음 전체 경로를 하나의 실행 가능한 시스템으로 연결했다.

```text
GPU hardware
  → Linux amdgpu driver
  → LXD device forwarding
  → k3s/containerd
  → AMD Device Plugin
  → vLLM Deployment
  → Service metrics
  → Prometheus/Grafana
  → load, queue, recovery, quota
```

핵심 질문은 “Pod에서 GPU가 보이는가?”가 아니라 다음 네 가지다.

1. GPU가 host에서 inference process까지 어떤 경로로 전달되는가?
2. 사용자 지연시간과 GPU 상태를 같은 시간축에서 설명할 수 있는가?
3. 처리량이 포화된 뒤 scheduler queue와 tail latency는 어떻게 변하는가?
4. 단일 GPU에서 장애 복구와 replica 확장은 어떤 제약을 갖는가?

## 2. 실험 환경

| 항목 | 구성 |
|---|---|
| Host GPU | AMD Radeon AI PRO R9700 |
| GPU target | `gfx1201` |
| VRAM | 32GB GDDR6, `amd-smi` 표시 32,624MB |
| Host kernel | Linux `6.8.0-137-generic` |
| amdgpu driver | `6.16.13` |
| ROCm | `7.2.4` |
| Lab isolation | 비특권 LXD container |
| Guest OS | Ubuntu 24.04.5 LTS |
| Kubernetes | k3s `v1.34.9+k3s1` |
| Container runtime | containerd `2.2.5-k3s2` |
| AMD Device Plugin | `1.31.0.10` |
| vLLM | `0.26.0` |
| Model | Qwen3-4B-Instruct-2507 BF16 |
| Model revision | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| AMD Device Metrics Exporter | `v1.5.1` |
| kube-prometheus-stack | `90.1.1` |
| Helm | `v3.21.4` |

모델과 vLLM runtime은 이미 검증한 로컬 SSD 사본을 LXD에 읽기 전용으로 전달했다. 컴파일·kernel·Hugging Face cache는 별도 쓰기 가능 경로에 보존했다. 모델 원본 경로와 host별 mount path는 환경마다 달라지므로 공개 설정에 절대경로를 넣지 않았다.

## 3. Host GPU에서 Kubernetes Pod까지

### 3.1 Host 계층

R9700은 `amdgpu` kernel driver가 관리하며 compute process는 주로 다음 장치 파일을 사용한다.

```text
/dev/kfd
/dev/dri/card0
/dev/dri/renderD128
```

이번 구성에서는 물리 GPU와 `/dev/kfd`를 LXD container에 전달했다. LXD 안의 k3s node에서도 같은 device node와 `gfx_target_version 120001`을 확인했다.

### 3.2 Device Plugin 계층

AMD Device Plugin은 kubelet의 device plugin socket을 통해 GPU를 extended resource로 등록한다.

```text
Node capacity:    amd.com/gpu=1
Node allocatable: amd.com/gpu=1
```

GPU Pod는 다음과 같이 정수 단위 resource를 요청했다.

```yaml
resources:
  limits:
    amd.com/gpu: 1
```

extended resource는 일반 CPU처럼 overcommit하지 않는다. GPU가 한 장이면 `amd.com/gpu: 1`을 사용하는 두 Pod를 동시에 실행할 수 없다.

### 3.3 GPU compute 확인

ROCm 개발 image에서 작은 HIP kernel을 컴파일해 실행했다.

```text
ROCm agent: gfx1201
HIP_RESULT=42.0
GPU_SMOKE_OK
```

이 검증은 다음 세 단계를 구분한다.

- device node 확인: 장치 파일이 mount됐는가
- runtime 확인: ROCm이 GPU agent를 발견하는가
- compute 확인: 실제 kernel launch와 device memory copy가 성공하는가

파일이 보인다는 사실만으로 GPU compute가 성공했다고 판단하지 않았다.

## 4. Qwen3-4B vLLM Deployment

### 4.1 주요 설정

| 설정 | 값 |
|---|---:|
| Replica | 1 |
| Update strategy | `Recreate` |
| dtype | BF16 |
| `max_model_len` | 4,096 |
| `gpu_memory_utilization` | 0.80 |
| `max_num_seqs` | 16 |
| `max_num_batched_tokens` | 4,096 |
| GPU request/limit | `amd.com/gpu: 1` |
| CPU request/limit | 2 / 8 |
| Memory request/limit | 8Gi / 24Gi |

단일 GPU에서는 기본 RollingUpdate가 기존 Pod와 새 Pod에 GPU를 동시에 요구해 rollout을 멈출 수 있다. 따라서 새 Pod를 만들기 전에 기존 Pod를 종료하는 `Recreate`를 사용했다. 이 선택은 update 중 무중단을 포기하는 대신 단일 GPU에서 rollout 교착을 피한다.

다음 probe를 구분했다.

- `startupProbe`: 긴 model load와 compile 동안 liveness가 process를 재시작하지 않게 한다.
- `readinessProbe`: Service endpoint에 포함해도 되는지 판단한다.
- `livenessProbe`: 준비된 뒤 server가 응답 불능 상태가 됐는지 확인한다.

### 4.2 Kubernetes Service 환경 변수 충돌

첫 Deployment는 engine 초기화 단계에서 다음 오류로 반복 종료됐다.

```text
ValueError: VLLM_PORT 'tcp://10.43.x.x:8000' appears to be a URI.
```

Kubernetes는 기본적으로 같은 namespace의 Service 정보를 환경 변수로 주입한다. 이름이 `vllm`인 Service 때문에 `VLLM_PORT`가 생성됐고, vLLM은 이를 자체 port 설정으로 해석했다.

Pod spec에 다음 설정을 추가해 해결했다.

```yaml
spec:
  enableServiceLinks: false
```

이 문제는 application이 사용하는 환경 변수 prefix와 Kubernetes Service 이름이 우연히 충돌할 수 있음을 보여준다. 단순한 readiness 실패가 아니라 이전 container log와 engine root cause를 확인해야 원인을 찾을 수 있었다. vLLM도 [환경 변수 문서](https://docs.vllm.ai/en/latest/configuration/env_vars.html)에서 이 Kubernetes 충돌 가능성을 안내한다.

### 4.3 기동 결과

warm-cache 재기동 log에서 확인한 주요 수치는 다음과 같다.

| 항목 | 결과 |
|---|---:|
| Weight loading | 9.70초 |
| Model loading 전체 | 12.73초, 7.61GiB |
| `torch.compile` | 9.57초 |
| Graph capture | 약 3초, 0.22GiB |
| Available KV Cache | 17.16GiB |
| GPU KV Cache capacity | 124,928 tokens |

startup log는 decoder attention backend로 `ROCM_ATTN`을 선택했다고 기록했다. 서버 준비 뒤 아래 endpoint가 정상 응답했다.

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
GET  /metrics
```

## 5. 관측 구성

### 5.1 두 계층을 함께 본다

```text
AMD Device Metrics Exporter
  └─ utilization, VRAM, clock, power, temperature

vLLM /metrics
  └─ running/waiting request, KV Cache, token, TTFT, ITL, queue time

             ↓ ServiceMonitor

Prometheus → Grafana
```

AMD exporter는 [공식 Helm chart](https://instinct.docs.amd.com/projects/device-metrics-exporter/en/latest/installation/kubernetes-helm.html)의 `v1.5.1`을 사용했다. vLLM은 OpenAI server의 `/metrics`를 수집하도록 ServiceMonitor를 추가했다. Prometheus에서는 다음 두 target이 모두 `up`이었다.

```text
amd-device-metrics-amd-metrics-exporter-svc  up
vllm                                        up
```

### 5.2 R9700에서 실제로 확인된 범위

AMD exporter log는 194개 GPU field를 지원 대상으로 선택했고 `/metrics`에서 357줄의 Prometheus exposition을 확인했다. 다음 metric은 실제 값을 반환했다.

- `amd_gpu_gfx_activity`
- `amd_gpu_used_vram`
- `amd_gpu_average_package_power`
- `amd_gpu_edge_temperature`
- `amd_gpu_clock`

반면 일부 Instinct 중심 field와 PCIe traffic field는 R9700에서 지원하지 않는다고 기록됐다. exporter가 실행된다는 사실과 모든 metric을 사용할 수 있다는 사실은 다르므로 dashboard를 만들기 전에 실제 노출 목록을 확인해야 한다.

### 5.3 부하 중 관측값

concurrency 64를 256 request 동안 유지한 뒤 Prometheus 2분 구간에서 확인한 최대값이다.

| Metric | 최대 관측값 |
|---|---:|
| GPU GFX activity | 100% |
| GPU used VRAM | 26,371MB |
| Average package power gauge | 308W |
| vLLM running requests | 16 |
| vLLM waiting requests | 48 |
| KV Cache usage | 약 2.05% |

`running=16`, `waiting=48`은 client concurrency 64와 `max_num_seqs=16`의 관계를 그대로 보여준다. 이 workload는 짧은 prompt와 128-token output이므로 KV Cache capacity가 아니라 scheduler의 sequence 한도가 먼저 드러났다.

power 값은 exporter가 수집한 시점의 telemetry gauge다. 짧은 peak 하나를 보드의 지속 가능한 전력이나 에너지 효율로 해석하지 않았다.

## 6. Saturation benchmark

### 6.1 조건

| 항목 | 조건 |
|---|---|
| 요청 방식 | closed-loop concurrency |
| Concurrency | 1, 2, 4, 8, 16, 32, 64 |
| 요청 수 | 각 64개 |
| 출력 길이 | 모든 요청 128 tokens |
| Sampling | temperature 0, seed 0, `ignore_eos=true` |
| Streaming | 활성화 |
| Warm-up | 측정 전 2회 |
| Prompt cache 완화 | 요청 시작 부분에 서로 다른 식별자 삽입 |
| Server 한도 | `max_num_seqs=16` |

동일 prompt를 그대로 반복하면 prefix cache가 결과를 지배할 수 있어 요청 시작 부분을 서로 다르게 만들었다. chat template의 공통 부분까지 완전히 제거한 no-cache microbenchmark는 아니므로 cache 영향을 0이라고 주장하지 않는다.

`ITL p95`는 tokenizer가 계산한 TPOT가 아니라 client가 받은 non-empty streaming chunk 사이 간격의 요청별 평균에 대한 p95다.

### 6.2 결과

| Concurrency | 성공/전체 | Output TPS | TTFT p95 | ITL p95 | E2E p95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 64/64 | 67.6 | 54.7ms | 15.61ms | 1.898초 |
| 2 | 64/64 | 132.6 | 96.4ms | 15.05ms | 1.939초 |
| 4 | 64/64 | 256.4 | 99.3ms | 15.46ms | 2.013초 |
| 8 | 64/64 | 381.2 | 168.0ms | 20.91ms | 2.698초 |
| 16 | 64/64 | 697.2 | 239.8ms | 22.79ms | 2.975초 |
| 32 | 64/64 | 699.2 | 3.172초 | 23.04ms | 5.975초 |
| 64 | 64/64 | 696.0 | 8.960초 | 22.69ms | 11.683초 |

### 6.3 해석

```text
concurrency 1 → 16
  output TPS 증가
  GPU를 채우는 batching 이득이 큼

concurrency 16 → 32 → 64
  output TPS 약 697 수준에서 정체
  waiting queue 증가
  TTFT와 E2E tail latency 급증
```

concurrency 16에서 32로 올렸을 때 output TPS 증가는 약 0.3%뿐이지만 TTFT p95는 약 13.2배가 됐다. concurrency 64에서는 output TPS가 오히려 소폭 감소했고 TTFT p95는 concurrency 16의 약 37.4배가 됐다.

따라서 이번 configuration의 처리량 knee는 16 부근이다. 실제 운영점은 SLO에 따라 더 낮아진다.

- TTFT p95 200ms 미만이 필요하면 concurrency 8 부근
- TTFT p95 250ms를 허용하면 concurrency 16 부근
- 최대 처리량만 보고 32나 64를 선택하면 사용자 대기시간만 크게 늘어남

concurrency 64, 256 request의 지속 부하에서도 output TPS는 701.3이었다. 짧은 64-request 측정과 비슷한 plateau를 보여 포화 판단을 다시 확인했다.

## 7. Pod 삭제와 warm-cache 복구

### 7.1 방법

1초 간격으로 Service `/health`를 호출하면서 실행 중인 vLLM Pod를 삭제했다. Deployment가 새 Pod를 만들고 readiness를 회복할 때까지 HTTP code와 시간을 기록했다.

```text
old Pod delete
  → Service endpoint 제거
  → replacement Pod schedule
  → package 준비
  → model load
  → cached compile artifact load
  → graph capture
  → API server ready
```

### 7.2 결과

| 항목 | 결과 |
|---|---:|
| Pod 삭제 시각 | 16:09:01.857 UTC |
| 첫 HTTP 200 | 삭제 후 152.365초 |
| Kubernetes Ready 확인 | 삭제 후 153.502초 |
| 삭제 이후 실패 health probe | 116회 |
| 새 Pod restart | 0회 |

compile cache가 남아 있어 AOT artifact를 직접 불러왔지만 복구에는 2분 30초 이상이 걸렸다. 가장 큰 운영상 문제는 local 검증 image가 시작할 때 `apt-get`으로 runtime package를 설치한다는 점이다.

따라서 production image에서는 다음을 build 시점에 포함해야 한다.

- vLLM과 Python dependency
- ROCm과 맞는 user-space library
- `libopenmpi` 등 system package
- 고정된 image digest와 SBOM

model weight와 compile cache를 보존하는 것만으로는 빠른 복구가 보장되지 않는다. **immutable image와 startup path 자체의 측정**이 필요하다.

## 8. 단일 GPU에서 HPA보다 먼저 필요한 것

### 8.1 왜 HPA가 해결책이 아닌가

실행 중인 vLLM Pod가 이미 `amd.com/gpu: 1`을 사용한다. HPA가 replica를 2로 늘려도 새로운 GPU가 생기지 않으므로 두 번째 Pod는 실행될 수 없다.

```text
Desired replicas: 2
Available GPU:     1
Runnable replicas: 1
```

CPU utilization만 보고 HPA를 구성하면 control loop는 정상이어도 capacity planning은 실패할 수 있다. autoscaler와 accelerator provisioner를 구분해야 한다.

### 8.2 ResourceQuota 검증

namespace의 GPU 총 요청을 1로 제한했다.

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: single-amd-gpu
spec:
  hard:
    requests.amd.com/gpu: "1"
```

vLLM이 GPU를 사용하는 동안 두 번째 GPU Pod를 생성하자 다음과 같이 admission 단계에서 거부됐다.

```text
exceeded quota: single-amd-gpu,
requested: requests.amd.com/gpu=1,
used: requests.amd.com/gpu=1,
limited: requests.amd.com/gpu=1
```

무한 Pending Pod를 쌓는 대신 허용 가능한 GPU 요청량을 명시적으로 통제할 수 있다. 여러 tenant와 batch workload가 있다면 다음 단계로 PriorityClass나 Kueue 같은 admission queue를 검토할 수 있다.

## 9. 다른 accelerator 실습을 로컬 AMD 환경에 옮기는 법

accelerator마다 driver와 runtime은 다르지만 Kubernetes 위의 운영 질문은 상당 부분 공통이다.

| 공통 문제 | 로컬 AMD 구현 |
|---|---|
| Device discovery | AMD Device Plugin |
| Device node | `/dev/kfd`, `/dev/dri/*` |
| User-space runtime | ROCm/HIP |
| Inference server | vLLM ROCm |
| Device telemetry | AMD Device Metrics Exporter |
| Service telemetry | vLLM `/metrics` |
| Collection/visualization | Prometheus/Grafana |
| Scale constraint | ResourceQuota와 scheduler queue |

NVIDIA의 Container Toolkit, DCGM, MIG·HAMi나 AWS Neuron의 compiler·NeuronCore를 그대로 흉내 내지 않았다. 대신 각 기술이 해결하는 공통 계층을 찾아 AMD에서 검증 가능한 구성으로 바꿨다.

Kubernetes DRA도 이번 핵심 경로에서 제외했다. 현재 AMD DRA 문서는 Instinct GPU 중심이고 기존 Device Plugin 경로와 동시에 활성화할 수 없으므로, R9700 단일 GPU의 첫 운영 실습에는 추가 변수가 너무 많다. [AMD DRA 문서](https://instinct.docs.amd.com/projects/gpu-operator/en/main/dra/dra-driver.html)

## 10. 공개 재현 패키지와 실행 방법

### 10.1 공개 범위

README의 수치를 독립적으로 확인하고 같은 실험을 반복할 수 있도록 다음 자료를 함께 공개한다.

```text
week06/
├─ README.md
├─ manifests/  # Device Plugin, vLLM, ServiceMonitor, ResourceQuota
├─ scripts/    # LXD/k3s 구성, 배포, 측정, 검증, 정리
└─ results/    # 선별한 원시 측정값과 요약
```

- [`scripts/`](./scripts/): 실행 순서를 자동화한 Bash·Python 스크립트
- [`manifests/`](./manifests/): 실험에서 실제 적용한 Kubernetes·Helm 설정
- [`results/saturation-64.csv`](./results/saturation-64.csv), [`saturation-64.json`](./results/saturation-64.json): concurrency 1~64 요청별 원시 값과 요약
- [`results/stress-256.csv`](./results/stress-256.csv), [`stress-256.json`](./results/stress-256.json): concurrency 64 지속 부하 원시 값과 요약
- [`results/prometheus-stress-summary.json`](./results/prometheus-stress-summary.json): 지속 부하 중 GPU·vLLM 최대 관측값
- [`results/observability-targets.json`](./results/observability-targets.json): Prometheus target 상태
- [`results/recovery.tsv`](./results/recovery.tsv): Pod 삭제 전후 health probe 기록
- [`results/gpu-smoke.log`](./results/gpu-smoke.log): ROCm agent와 HIP kernel 실행 증거

모델 weight, vLLM virtualenv, ROCm 설치본과 compile cache는 크기와 재배포 조건 때문에 포함하지 않는다. 반복 측정 중 생성된 중간 결과와 중복 파일도 제외했다.

### 10.2 공개 데이터 검증

파일 무결성과 CSV에서 다시 계산할 수 있는 통계를 다음 명령으로 확인할 수 있다.

```bash
cd week06

(cd results && sha256sum -c SHA256SUMS)

python3 scripts/verify-results.py \
  --csv results/saturation-64.csv \
  --summary results/saturation-64.json

python3 scripts/verify-results.py \
  --csv results/stress-256.csv \
  --summary results/stress-256.json
```

검증 스크립트는 concurrency별 요청·성공·오류 수, request ID 중복, completion token 수, TTFT·ITL·E2E percentile과 TPS 계산식을 검사한다. 원시 CSV에는 각 단계의 절대 시작·종료 시각이 없으므로 `wall_s` 자체는 JSON에 기록된 측정값이며 CSV만으로 독립 재계산할 수 없다.

### 10.3 전체 실험 재현

Linux AMD GPU host에 LXD, `lxc`, `jq`, ROCm user-space 디렉터리, vLLM virtualenv와 로컬 모델 디렉터리가 필요하다. host별 경로는 환경 변수로만 전달한다.

```bash
cd week06

LXD_STORAGE_SOURCE=/path/to/empty-local-ssd-directory \
ROCM_SOURCE=/path/to/rocm \
VLLM_RUNTIME_SOURCE=/path/to/vllm-venv \
MODEL_SOURCE=/path/to/Qwen3-model \
  ./scripts/setup-lab.sh

./scripts/run-gpu-smoke.sh
./scripts/deploy-vllm.sh
./scripts/install-observability.sh
./scripts/validate-metrics.sh
./scripts/validate-observability.sh
./scripts/validate-gpu-quota.sh

CONCURRENCIES=1,2,4,8,16,32,64 \
REQUESTS_PER_LEVEL=64 \
MAX_TOKENS=128 \
  ./scripts/benchmark.sh

./scripts/recovery-test.sh
```

기본 버전과 server 설정은 매니페스트와 스크립트에 고정돼 있다. 다른 hardware·model·runtime으로 실행했다면 기존 결과를 덮어쓰지 말고 환경과 결과를 별도로 기록해야 한다.

### 10.4 실행 중 확인할 명령

#### GPU와 node resource

```bash
amd-smi static
rocminfo
kubectl get node -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.amd\.com/gpu
kubectl get pods -A -o wide
```

#### vLLM

```bash
kubectl rollout status deployment/vllm -n week6-llm
kubectl logs deployment/vllm -n week6-llm
kubectl get service vllm -n week6-llm
```

#### 관측

```bash
kubectl get servicemonitor -A
kubectl get pods -n monitoring
kubectl get pods -n kube-amd-gpu
```

Grafana는 외부에 공개하지 않고 필요할 때만 port-forward할 수 있게 했다.

```bash
kubectl port-forward -n monitoring service/kube-prometheus-stack-grafana 3000:80
```

#### 정리

실습 종료 시 최소한 다음 자원을 확인한다.

```bash
REMOVE_OBSERVABILITY=true STOP_INSTANCE=true ./scripts/cleanup.sh
```

GPU Device Plugin은 다른 workload가 사용할 수 있으므로 자동으로 제거하지 않는다. LXD instance와 model/cache mount도 데이터 손실을 막기 위해 별도 수명주기로 관리한다.

## 11. 한계와 후속 과제

### 이번 결과의 한계

- 단일 R9700, 단일 node, 단일 replica 결과다.
- LXD를 이용한 격리 환경이며 bare-metal multi-node cluster와 다르다.
- 합성된 짧은 prompt와 고정 128-token output을 사용했다.
- closed-loop concurrency 결과이므로 open-loop request rate와 직접 비교할 수 없다.
- public traffic, 인증, TLS, persistent volume 장애는 다루지 않았다.
- 현재 local image는 system package를 시작 시 설치하므로 production-ready immutable image가 아니다.
- R9700에서 AMD exporter가 제공하지 않는 field가 있어 Instinct dashboard를 그대로 사용할 수 없다.
- 평균 전력과 짧은 peak만 기록했으며 request당 energy는 계산하지 않았다.

### 다음 실습 우선순위

1. package와 runtime을 모두 포함한 immutable vLLM ROCm image를 빌드해 복구 시간을 다시 측정한다.
2. prompt/output 길이 조합을 Prefill-heavy, Decode-heavy로 나눠 saturation point를 비교한다.
3. Prefix Cache on/off를 동일 request set으로 비교한다.
4. Prometheus histogram으로 TTFT·queue time p95/p99를 server 관점에서도 계산한다.
5. arrival rate를 고정한 open-loop 부하에서 queue collapse 지점을 찾는다.
6. 여러 workload가 한 GPU를 요청할 때 PriorityClass와 Kueue admission을 검증한다.

## 마무리

이번 실습에서 가장 중요한 결과는 최대 output TPS 700이라는 단일 숫자가 아니다. GPU, scheduler, Kubernetes와 운영 설정이 사용자 지연시간으로 연결되는 과정을 실제로 확인했다는 점이다.

```text
concurrency 증가
  → batching으로 처리량 증가
  → max_num_seqs에서 실행 수 제한
  → waiting queue 증가
  → GPU는 이미 포화
  → TPS는 정체
  → TTFT와 E2E tail latency 급증
```

또한 GPU가 정상 동작해도 Service 환경 변수 하나가 engine을 종료시킬 수 있고, compile cache가 있어도 image startup path가 느리면 복구에 수 분이 걸릴 수 있었다. 단일 GPU에서 HPA만 추가해도 replica는 늘지 않는다.

따라서 LLM 서빙 운영에서는 다음 순서가 중요하다.

```text
GPU 전달 경로 검증
→ server와 probe 구성
→ service/GPU metric 동시 수집
→ 대표 부하로 포화 지점 측정
→ 장애 복구와 admission 정책 검증
→ 실제 병목이 확인된 계층만 개선
```

관련 공식 문서:

- [vLLM ROCm 설치](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [vLLM Prometheus/Grafana 예제](https://docs.vllm.ai/en/latest/examples/observability/prometheus_grafana/)
- [AMD Kubernetes Device Plugin](https://instinct.docs.amd.com/projects/k8s-device-plugin/en/latest/)
- [AMD Device Metrics Exporter](https://instinct.docs.amd.com/projects/device-metrics-exporter/en/latest/)
- [Kubernetes ResourceQuota](https://kubernetes.io/docs/concepts/policy/resource-quotas/)
