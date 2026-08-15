# Kind + KubeRay + R9700 실험

Docker container를 Kubernetes node로 사용하는 Kind에서 CPU RayService 수명주기와 R9700 장치 연결을 검증한다. k3s 결과는 대안 환경 및 비교 자료로만 사용한다.

```text
host → non-privileged LXD → Docker → Kind node → Kubernetes Pod
```

검증 순서는 다음과 같다.

1. Kind cluster와 control-plane 기동
2. KubeRay operator와 CPU RayService 배포
3. endpoint 호출
4. RayCluster 설정 변경에 따른 blue-green 교체
5. worker Pod 삭제 후 자동 복구
6. AMD Device Plugin의 `amd.com/gpu` 등록
7. GPU Pod에서 `gfx1201` 인식과 HIP kernel 실행
8. Ray `num_gpus=1` task scheduling

실제 저장장치 경로는 문서에 기록하지 않고 환경 변수로 주입한다.

## 1. 실습 cluster 생성

LXD, `jq`, 로컬 SSD의 빈 디렉터리, R9700이 필요하다. 호스트에 Docker를 설치하지 않고 실습용 LXD container 안에 Docker와 Kind를 설치한다.

```bash
mkdir -p ./.lab/kind-storage
LXD_STORAGE_SOURCE="$(pwd)/.lab/kind-storage" \
  ./experiments/kind-rayservice/scripts/create-lab.sh
```

주요 고정 버전은 다음과 같다.

| 구성요소 | 버전 |
| --- | --- |
| Kind | `v0.32.0` |
| Kubernetes | `v1.36.1` digest 고정 image |
| KubeRay operator | `1.6.0` |
| Ray | `2.52.0` |

비특권 LXD는 user namespace 안에서 동작하므로 기본 Kind 구성에서는 kubelet이 `/dev/kmsg`를 열지 못했다. 권한을 privileged로 넓히는 대신 Kind cluster 설정에 `KubeletInUserNamespace=true`를 적용했다.

## 2. CPU RayService

```bash
./experiments/kind-rayservice/scripts/run-cpu-rayservice.sh
./experiments/kind-rayservice/scripts/run-cpu-lifecycle.sh
```

- 배포 manifest: [`./manifests/rayservice-cpu.yaml`](./manifests/rayservice-cpu.yaml)
- endpoint 응답: [`./results/cpu-endpoints.txt`](./results/cpu-endpoints.txt)
- 교체 probe: [`./results/week2-upgrade-probes.tsv`](./results/week2-upgrade-probes.tsv)
- worker 복구 probe: [`./results/week2-recovery-probes.tsv`](./results/week2-recovery-probes.tsv)

## 3. R9700 연결과 HIP 연산

```bash
./experiments/kind-rayservice/scripts/run-gpu-smoke.sh
```

LXD에 전달한 `/dev/kfd`와 `/dev/dri`를 Kind `extraMounts`로 node container에 한 번 더 전달한다. AMD Device Plugin이 `amd.com/gpu=1`을 등록하면 GPU limit을 요청한 Pod에서 `rocminfo`와 작은 HIP kernel을 실행한다.

## 4. Ray GPU worker

```bash
./experiments/kind-rayservice/scripts/run-ray-gpu.sh
```

[`./manifests/rayservice-gpu.yaml`](./manifests/rayservice-gpu.yaml)은 worker에 다음 두 자원을 함께 선언한다.

```text
Kubernetes: amd.com/gpu: 1
Ray:        num-gpus: "1"
```

따라서 Kubernetes device allocation과 Ray logical GPU scheduling을 한 번에 검증할 수 있다.

## 5. 범위와 정리

이 실험에서는 vLLM을 다시 실행하지 않았다. k3s 실험에서 Qwen3-0.6B endpoint까지 이미 검증했고, Kind에서 새로 확인해야 하는 Docker node 경계는 HIP kernel과 Ray GPU task로 실제 compute까지 통과했기 때문이다. 이후 volume 전달 방식까지 비교할 필요가 생기면 별도 vLLM image 실험으로 확장한다.

전체 결과는 [`./results/summary.md`](./results/summary.md)에 있다.

실습 환경을 잠시 보존하려면 중지하고, 더 사용하지 않으면 삭제한다.

```bash
./experiments/kind-rayservice/scripts/stop-lab.sh
./experiments/kind-rayservice/scripts/delete-lab.sh
```

`delete-lab.sh`는 실습 instance, profile, network, storage pool을 제거한다.

## 참고 자료

- [Kind Quick Start](https://kind.sigs.k8s.io/docs/user/quick-start/)
- [Kind Configuration](https://kind.sigs.k8s.io/docs/user/configuration/)
- [KubeRay RayService](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/rayservice.html)
