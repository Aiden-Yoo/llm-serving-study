# 대안 환경 및 비교 실험 결과: k3s + KubeRay + R9700

## 결론

비특권 LXD 컨테이너 안의 단일 노드 k3s에서 다음 경로를 모두 검증했다.

```text
k3s → KubeRay → CPU RayService
R9700 → AMD Device Plugin → Kubernetes GPU Pod → HIP kernel
R9700 → Ray GPU worker → num_gpus=1 task
R9700 → Kubernetes Pod → vLLM OpenAI-compatible endpoint
```

| 검증 항목 | 결과 |
| --- | --- |
| CPU RayService | `Running`, endpoint 2개 |
| RayService 설정 교체 | 60초, probe 90/90 성공 |
| worker 강제 삭제 복구 | 50초 후 새 worker Ready |
| Kubernetes GPU resource | `amd.com/gpu=1` |
| ROCm 장치 인식 | `gfx1201` |
| HIP 연산 | `HIP_RESULT=42.0` |
| Ray GPU task | `ray_gpu_ids=[0]`, `/dev/kfd` 확인 |
| vLLM | Qwen3-0.6B `/v1/models`, `/v1/chat/completions` HTTP 200 |

## 1. k3s 선택과 사전 실패

호스트에서 직접 rootless k3s를 실행하는 경로는 사용자 namespace의 UID mapping이 차단되어 진행할 수 없었다. 또한 rootless k3s가 요구하는 cgroup delegation과 Ubuntu AppArmor 설정을 바꾸려면 관리자 권한이 필요했다.

호스트 보안 설정을 변경하는 대신 다음 조건의 격리된 실습 환경으로 전환했다.

- 비특권 LXD 컨테이너
- CPU 8개, 메모리 24GiB 제한
- Traefik과 ServiceLB를 제외한 단일 노드 k3s
- k3s 데이터와 container image는 로컬 SSD 기반 LXD storage에 저장
- `KubeletInUserNamespace=true`로 `/dev/kmsg` 의존성 제거

이 구성은 kind보다 계층이 하나 적고, systemd를 사용하는 k3s와 GPU 장치를 한 컨테이너 안에서 관찰하기 쉬웠다. 다만 중첩 컨테이너 실습이므로 프로덕션 설치 모델은 아니다.

## 2. CPU RayService

| 구성요소 | 버전 |
| --- | --- |
| k3s | `v1.34.9+k3s1` |
| KubeRay operator | `1.6.0` |
| Ray | `2.52.0` |

fruit와 calculator 애플리케이션을 배포했다.

```text
POST /fruit/ ["MANGO", 2] → 6
POST /calc/  ["MUL", 3]   → 15 pizzas please!
```

### Blue-green cluster 교체

RayCluster pod template을 변경해 새 cluster 생성을 유도했다. 기존 cluster가 요청을 처리하는 동안 pending cluster가 준비됐고, 새 cluster가 Ready가 된 뒤 Service selector가 전환됐다.

- 전환 시간: 60초
- probe: 90회 성공, 0회 실패
- 응답 `8`: 기존 Serve config에서 70회
- 응답 `6`: 새 Serve config에서 20회

원시 기록은 [`./cpu-upgrade-transitions.tsv`](./cpu-upgrade-transitions.tsv)와 [`./cpu-upgrade-probes.tsv`](./cpu-upgrade-probes.tsv)에 있다.

### Worker 복구

head의 논리 CPU를 0으로 설정해 Serve replica가 worker에 반드시 배치되게 했다. worker pod를 삭제한 뒤 KubeRay가 같은 group의 새 worker를 생성했고 Ray Serve replica와 proxy가 복구됐다.

- 새 worker가 Ready가 될 때까지 50초
- 측정 중 성공 7회, timeout 14회
- 최종 RayService 상태 `Running`

단일 worker의 모든 replica를 동시에 제거했기 때문에 복구 시간 동안 실제 중단이 발생했다. worker와 replica를 둘 이상으로 구성해야 고가용성 검증이 된다. 원시 요청 기록은 [`./cpu-worker-recovery-probes.tsv`](./cpu-worker-recovery-probes.tsv)에 있다.

## 3. R9700 Kubernetes 연결

LXD에 물리 GPU와 `/dev/kfd`를 전달하고 AMD Device Plugin을 설치했다.

```text
Node capacity:    amd.com/gpu=1
Node allocatable: amd.com/gpu=1
Device Plugin:    Found 1 AMDGPUs
ROCm agent:       gfx1201
```

GPU limit을 요청한 Pod 안에는 `/dev/kfd`, `/dev/dri/card0`, `/dev/dri/renderD128`만 전달됐다. 같은 Pod에서 작은 HIP kernel을 컴파일·실행해 `HIP_RESULT=42.0`을 확인했다. 따라서 device 등록뿐 아니라 실제 compute까지 성공했다.

Device Plugin이 compute·memory partition 관련 sysfs 파일을 찾지 못했다는 warning과 P2P weight 초기화 실패가 있었지만, 단일 비분할 GPU에서는 kubelet 기본 allocation으로 정상 동작했다.

## 4. Ray GPU worker

Ray worker pod에 다음 두 조건을 함께 지정했다.

```text
Kubernetes limit: amd.com/gpu=1
Ray start param:  num-gpus=1
```

`num_cpus=0.1, num_gpus=1` remote task는 GPU worker에 스케줄됐고 다음 값을 반환했다.

```json
{
  "ray_gpu_ids": [0],
  "dev_kfd": true,
  "dri_devices": [
    "/dev/dri/card0",
    "/dev/dri/renderD128"
  ]
}
```

AMD Device Plugin은 이 구성에서 visibility 환경 변수를 주입하지 않았지만, Ray의 논리 GPU ID와 Linux device node 할당은 일치했다.

## 5. Kubernetes vLLM

호스트에서 이미 검증한 vLLM runtime과 정확히 같은 ROCm user-space를 읽기 전용 volume으로 전달하고 Qwen3-0.6B를 실행했다.

| 항목 | 결과 |
| --- | --- |
| vLLM | `0.26.0` |
| Model | Qwen3-0.6B BF16 |
| Model load | 1.12GiB, 4.26초 |
| `torch.compile` | 47.43초 |
| Engine init | 86.29초 |
| GPU KV cache | 131,904 tokens |
| `/v1/models` | HTTP 200 |
| `/v1/chat/completions` | HTTP 200 |

처음에는 두 번 실패했다.

1. 마운트한 venv의 `vllm` shebang이 원래 위치를 가리켜 Exit 127이 발생했다. venv Python으로 module을 직접 실행해 해결했다.
2. 경량 ROCm image에 MIOpen이 없어 `libMIOpen.so.1` import error가 발생했다. 검증된 ROCm 7.2.4 user-space를 읽기 전용으로 전달해 해결했다.

이는 개발 환경 재사용에는 유효하지만 배포 방식으로는 부적절하다. 실제 서비스에서는 vLLM, PyTorch, ROCm, MPI와 필요한 system library를 하나의 immutable image에 포함해야 한다.

근거 로그와 API 응답은 다음 파일에 있다.

- [`./gpu-evidence.txt`](./gpu-evidence.txt)
- [`./vllm-models.json`](./vllm-models.json)
- [`./vllm-completion.json`](./vllm-completion.json)

## 한계

1. 단일 노드·단일 GPU이므로 node failure, GPU failover, multi-node scheduling은 검증하지 않았다.
2. LXD 안의 k3s는 학습용 격리 환경이며 bare metal 또는 정식 VM cluster와 보안·장애 경계가 다르다.
3. AMD Device Plugin DaemonSet은 privileged container를 사용한다.
4. vLLM runtime을 volume으로 재사용한 방식은 재현 가능한 production image가 아니다.
5. worker 1개를 삭제한 복구 시험에서는 50초의 서비스 중단이 발생했다. 자동 복구와 무중단은 같은 개념이 아니다.
