# R9700 서빙 설계 실험

CH3·CH4의 설계 내용을 AMD Radeon AI PRO R9700 단일 GPU에서 검증한다.

## 실험 1: 비동기 호출과 vLLM 스케줄링

두 가지 vLLM profile을 실행한다.

| Profile | `max_num_seqs` | `max_num_batched_tokens` |
| --- | ---: | ---: |
| constrained | 4 | 2,048 |
| balanced | 16 | 8,192 |

각 profile에서 동일한 요청을 네 경로로 전달한다.

- `blocking`: `async def`에서 동기 upstream 호출을 직접 실행
- `threaded`: 동기 호출을 worker thread로 격리
- `async`: 비동기 HTTP client 사용
- `direct`: vLLM endpoint 직접 호출

```bash
VLLM_RUNTIME_ROOT=./.runtime/vllm \
VLLM_MODEL_PATH=./models/Qwen3-4B-Instruct-2507 \
./experiments/run_serving_modes.sh
```

## 실험 2: 멀티 모델 lazy loading과 LRU

cache capacity를 1로 제한하고 `small → small → base → small` 순서로 접근한다.

- 최초 `small`: cache miss와 모델 로드
- 두 번째 `small`: cache hit
- `base`: `small` LRU eviction 후 로드
- 마지막 `small`: `base` eviction 후 재로드

```bash
VLLM_RUNTIME_ROOT=./.runtime/vllm \
SMALL_MODEL_PATH=./models/Qwen3-0.6B \
VLLM_MODEL_PATH=./models/Qwen3-4B-Instruct-2507 \
./experiments/run_multi_model_lru.sh
```

## 결과

- `./experiments/results/serving-modes-latest.md`
- `./experiments/results/multi-model-lru-latest.md`

절대 성능보다는 같은 장비·모델에서 한 변수만 변경했을 때 나타나는 상대 차이를 해석한다.

## 대안 환경 및 비교 실험: k3s + KubeRay + R9700

비특권 LXD 컨테이너에 단일 노드 k3s를 구성하고 다음 항목을 검증한다.

- KubeRay CPU RayService endpoint
- RayCluster blue-green 교체 중 요청 연속성
- worker pod 삭제 후 자동 복구
- AMD Device Plugin의 `amd.com/gpu` 등록
- GPU Pod의 `gfx1201` 인식과 HIP kernel 실행
- Ray `num_gpus=1` task scheduling
- Kubernetes Pod의 Qwen3-0.6B vLLM endpoint

- 실행 방법: [`./k3s-rayservice/README.md`](./k3s-rayservice/README.md)
- 결과: [`./k3s-rayservice/results/summary.md`](./k3s-rayservice/results/summary.md)

## 실험 3: Kind + KubeRay + R9700

Docker container를 Kubernetes node로 사용하는 Kind에서 과제의 핵심 경로를 검증한다. k3s 결과는 대안 환경 및 비교 자료로 보존한다.

- Kind control-plane과 KubeRay CPU RayService
- endpoint 호출과 RayCluster blue-green 교체
- worker Pod 삭제 후 자동 복구
- `/dev/kfd`, `/dev/dri`의 LXD → Kind node 전달
- AMD Device Plugin의 `amd.com/gpu` 등록
- GPU Pod의 `gfx1201` 인식과 HIP kernel 실행
- Ray `num_gpus=1` task scheduling

- 실행 방법: [`./kind-rayservice/README.md`](./kind-rayservice/README.md)
- 결과: [`./kind-rayservice/results/summary.md`](./kind-rayservice/results/summary.md)
