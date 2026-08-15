# 대안 환경 및 비교 실험: k3s + KubeRay + R9700

Kind 과제의 대안 환경 및 비교 자료로, 비특권 LXD 컨테이너 안에 단일 노드 k3s를 구성해 CPU RayService 수명주기와 R9700 장치 연결을 단계별로 검증한다.

검증 순서는 다음과 같다.

1. k3s와 기본 system pod 기동
2. KubeRay operator 설치
3. CPU RayService endpoint 호출
4. worker pod 삭제 후 자동 복구
5. Ray cluster 설정 변경 후 교체와 요청 연속성 확인
6. R9700 장치 전달과 AMD Device Plugin의 `amd.com/gpu` 등록
7. GPU 요청 pod에서 ROCm 장치 접근

로컬 저장소나 모델의 실제 절대경로는 기록하지 않는다. 재현할 때는 환경 변수로 주입한다.

## 1. 실습 cluster 생성

LXD와 로컬 SSD의 빈 디렉터리가 필요하다. 실제 경로는 저장소에 기록하지 않고 환경 변수로 전달한다.

```bash
mkdir -p ./.lab/lxd-storage
LXD_STORAGE_SOURCE="$(pwd)/.lab/lxd-storage" \
  ./experiments/k3s-rayservice/scripts/create-lab.sh
```

생성되는 k3s는 Traefik과 ServiceLB를 제외한 단일 노드 구성이다. 비특권 LXD의 user namespace에서 kubelet을 실행하기 위해 `KubeletInUserNamespace` feature gate를 사용한다.

## 2. CPU RayService

```bash
./experiments/k3s-rayservice/scripts/run-cpu-rayservice.sh
```

배포 manifest는 [`./manifests/rayservice-cpu.yaml`](./manifests/rayservice-cpu.yaml)이다.

검증 항목은 endpoint 호출, worker 삭제 복구, RayCluster 설정 변경에 따른 blue-green 교체다.

## 3. R9700 연결

```bash
./experiments/k3s-rayservice/scripts/attach-gpu.sh
./experiments/k3s-rayservice/scripts/run-gpu-smoke.sh
```

스크립트는 LXD가 발견한 첫 AMD GPU의 PCI 주소와 `/dev/kfd`를 전달한다. AMD Device Plugin이 `amd.com/gpu=1`을 등록하면 GPU Pod에서 `rocminfo`와 HIP kernel을 실행한다.

## 4. Ray GPU worker

[`./manifests/rayservice-gpu.yaml`](./manifests/rayservice-gpu.yaml)은 worker에 다음 자원을 함께 선언한다.

```text
amd.com/gpu: 1
num-gpus: "1"
```

이를 통해 Kubernetes device allocation과 Ray logical GPU scheduling을 함께 검증한다.

## 5. vLLM 확장

[`./manifests/vllm-smoke.yaml`](./manifests/vllm-smoke.yaml)은 이미 검증한 runtime, ROCm user-space, 작은 모델을 읽기 전용 volume으로 전달하는 실험용 구성이다. 실제 source path는 다음 환경 변수로만 받는다.

```text
VLLM_RUNTIME_SOURCE
ROCM_SOURCE
MODEL_SOURCE
```

세 volume을 연결한 뒤 실행한다.

```bash
./experiments/k3s-rayservice/scripts/run-vllm-smoke.sh
```

이 방식은 로컬 검증용이다. production에서는 dependency와 ROCm user-space를 image에 포함해야 한다.

## 결과

전체 결과와 실패·수정 과정은 [`./results/summary.md`](./results/summary.md)에 있다.

실습을 마친 뒤 container를 중지해 CPU·메모리·GPU를 반환한다.

```bash
./experiments/k3s-rayservice/scripts/stop-lab.sh
```
