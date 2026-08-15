# Kind + KubeRay + R9700 실험 결과

## 결론

비특권 LXD 안의 Docker 기반 단일 노드 Kind에서 다음 경로를 실제로 검증했다.

```text
Kind → KubeRay → CPU RayService
R9700 → LXD → Kind node → AMD Device Plugin → GPU Pod → HIP kernel
R9700 → Kind node → Ray GPU worker → num_gpus=1 task
```

| 검증 항목 | 결과 |
| --- | --- |
| Kind control-plane | `Ready` |
| CPU RayService | `Running`, endpoint 2개 |
| endpoint | fruit `6`, calculator `15 pizzas please!` |
| RayCluster 교체 | 52초, probe 52/52 성공 |
| worker 삭제 복구 | 47초, 성공 8회·실패 15회 |
| Kubernetes GPU resource | capacity/allocatable `amd.com/gpu=1` |
| ROCm 장치 | `gfx1201` |
| HIP 연산 | `HIP_RESULT=42.0` |
| Ray GPU task | `ray_gpu_ids=["0"]`, `/dev/kfd` 확인 |

## 1. 구성

| 구성요소 | 버전 |
| --- | --- |
| Kind | `v0.32.0` |
| Kubernetes | `v1.36.1` |
| Docker | `29.7.2` |
| KubeRay operator | `1.6.0` |
| Ray | `2.52.0` |
| AMD Device Plugin | `1.31.0.10` |
| ROCm user-space image | `7.2.4` |

Kind node image는 release digest까지 고정했다. 외부 LXD container는 CPU 8개와 메모리 24GiB로 제한했고 privileged mode로 전환하지 않았다.

## 2. 첫 기동 실패와 수정

첫 Kind node는 생성됐지만 kubelet이 반복 종료되어 API server가 준비되지 않았다.

```text
Failed to create an oomWatcher (running in UserNS,
Hint: enable KubeletInUserNamespace feature flag to ignore the error)
failed to create kubelet: open /dev/kmsg: no such file or directory
```

이는 Docker 자체나 Kind image pull 실패가 아니라, 비특권 LXD user namespace에서 Kind node의 kubelet이 `/dev/kmsg`를 열 수 없었던 문제다. Kind의 cluster-wide feature gate에 `KubeletInUserNamespace=true`를 추가한 다음 control-plane이 17초 만에 `Ready`가 됐다.

근거: [`./kind-bootstrap-diagnosis.txt`](./kind-bootstrap-diagnosis.txt)

## 3. CPU RayService

```text
POST /fruit/ ["MANGO", 2] → 6
POST /calc/  ["MUL", 3]   → 15 pizzas please!
```

RayCluster Pod template의 revision을 변경해 새 cluster 생성을 유도했다. 기존 cluster가 요청을 처리하는 동안 새 cluster가 준비됐고 52초 뒤 active cluster가 전환됐다. 관측한 52회 요청은 모두 성공했다.

이전 cluster가 정리된 뒤 active cluster의 단일 worker를 삭제했다. 새 worker와 Serve replica가 복구되어 RayService가 다시 `Running`이 되기까지 47초가 걸렸고, 측정 중 8회 성공·15회 실패가 발생했다. 자동 복구와 무중단은 별개의 속성이다.

원시 기록:

- [`./week2-upgrade-transitions.tsv`](./week2-upgrade-transitions.tsv)
- [`./week2-upgrade-probes.tsv`](./week2-upgrade-probes.tsv)
- [`./week2-recovery-probes.tsv`](./week2-recovery-probes.tsv)

## 4. R9700 장치 경계

GPU device node는 다음 경계를 통과했다.

```text
host /dev → LXD device → Kind extraMounts → Kubernetes device allocation → Pod
```

AMD Device Plugin은 R9700 한 개를 발견해 node capacity와 allocatable에 각각 `amd.com/gpu=1`을 등록했다. GPU Pod에서는 `/dev/kfd`, `/dev/dri/card0`, `/dev/dri/renderD128`만 할당됐고 `gfx1201` agent가 확인됐다.

같은 Pod에서 HIP kernel을 컴파일·실행해 `41.0 + 1.0 = 42.0`을 얻었다. 장치 파일을 보는 수준이 아니라 실제 compute까지 성공한 것이다.

## 5. Ray GPU scheduling

Ray worker에는 Kubernetes `amd.com/gpu=1` limit과 Ray `num-gpus=1`을 함께 설정했다. `num_gpus=1` remote task의 결과는 다음과 같다.

```json
{"dev_kfd": true, "dri_devices": ["/dev/dri/card0", "/dev/dri/renderD128"], "ray_gpu_ids": ["0"]}
```

근거: [`./gpu-evidence.txt`](./gpu-evidence.txt)

## 6. k3s와 비교

| 항목 | k3s | Kind |
| --- | --- | --- |
| Kubernetes 형태 | 경량 배포판 | Docker container node |
| 외부 격리 | 비특권 LXD | 비특권 LXD |
| 중첩 container 계층 | 없음 | 있음 |
| user namespace 대응 | k3s kubelet argument | Kind feature gate |
| CPU RayService | 성공 | 성공 |
| blue-green 전환 | 60초, 90/90 성공 | 52초, 52/52 성공 |
| 단일 worker 복구 | 50초, 중단 발생 | 47초, 중단 발생 |
| `amd.com/gpu=1` | 성공 | 성공 |
| HIP kernel | 성공 | 성공 |
| Ray GPU task | 성공 | 성공 |
| vLLM endpoint | 검증 | 반복하지 않음 |

측정 횟수와 cluster 상태가 완전히 같지 않으므로 시간 차이를 성능 우위로 해석하지 않는다. 핵심 결과는 Kind의 추가 Docker node 경계에서도 control-plane, device plugin, HIP compute, Ray logical GPU가 모두 동작했다는 점이다.

## 한계

1. 단일 노드·단일 GPU이므로 node failure와 GPU failover는 검증하지 않았다.
2. LXD 안의 Docker와 Kind는 학습용 중첩 환경이며 production 보안 경계가 아니다.
3. AMD Device Plugin DaemonSet은 privileged container를 사용한다.
4. 단일 worker를 삭제한 복구 시험에는 실제 요청 실패가 있었다.
5. vLLM은 k3s에서 이미 검증했으므로 Kind에서는 반복하지 않았다.
