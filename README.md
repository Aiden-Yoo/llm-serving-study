# LLM Serving Study

LLM 모델 서빙의 기초부터 시스템 설계, 최적화, 서빙 프레임워크와 Kubernetes 기반 운영까지 단계적으로 학습하고, 주차별 정리와 재현 가능한 실험 결과를 기록하는 저장소입니다.

이 저장소는 CloudNet 팀의 [서종호님(가시다)](https://www.linkedin.com/in/gasida99/)이 진행하는 LLM Serving Study에서 학습한 내용과 실습 결과를 정리한 기록입니다.

## 학습 기록

| 주차 | 학습 범위 | 주요 내용 | 정리 |
| ---: | --- | --- | --- |
| 1 | **CH1, CH2** | 모델 서빙·최적화 소개 | [학습 정리 및 R9700 실험](./week01/README.md) |
| 2 | **CH3, CH4** | LLM 서빙 시스템 설계·모범 사례 | [학습 정리 및 R9700·Kubernetes 실험](./week02/README.md) |
| 3 | **CH5, CH6** | LLM 서빙 병목·필수 최적화 방법 | [학습 정리](./week03/README.md) |
| 4 | **CH7, CH8** | 고급 LLM 최적화·서빙 프레임워크 | [학습 정리](./week04/README.md) |
| 5 | **CH9, CH10** | 실전 LLM 최적화·차세대 서빙 시스템 | [학습 정리](./week05/README.md) |
| 6 | **Kubernetes GPU 운영 실습** | AMD GPU 할당·vLLM·관측·부하·복구·Quota | [R9700 실습 정리](./week06/README.md) |

### 향후 학습 후보

아래 항목은 순서와 일정이 확정되지 않았으며 스터디 진행 상황에 따라 달라질 수 있다.

- **AWS Workshop:** Generative AI on Amazon EKS
- **추가 기술:** [llm-d](https://llm-d.ai/docs), [KServe](https://kserve.github.io/website/)

## 저장소 구성

```text
week01/
├─ README.md
└─ experiments/
   ├─ start_vllm_r9700.sh
   ├─ benchmark_vllm.py
   └─ results/
      ├─ latest.md
      └─ latest.json

week02/
├─ README.md
└─ experiments/
   ├─ 비동기 호출·스케줄링 비교
   ├─ 멀티 모델 lazy loading·LRU
   ├─ kind-rayservice/
   └─ k3s-rayservice/  # 대안 환경 및 비교 실험

week03/
└─ README.md

week04/
└─ README.md

week05/
└─ README.md

week06/
├─ README.md
├─ manifests/
├─ scripts/
└─ results/
```

각 주차 문서는 핵심 개념을 정리하고, 별도 실험을 수행한 경우에는 실험 조건, 측정 결과와 한계를 함께 기록합니다.

## 현재 진행

- **1주차:** CH1·CH2 핵심 내용과 KV Cache, Prefill·Decode, Prefix Cache, Streaming·Batching 정리
- **2주차:** CH3·CH4 서빙 시스템 설계와 모범 사례 정리
- **3주차:** CH5·CH6 서빙 병목과 Batching, Attention, 모델 압축, Prefix Caching 정리
- **4주차:** CH7·CH8 Speculative Decoding, 분산 병렬화, 고급 KV Cache와 서빙 프레임워크 정리
- **5주차:** CH9·CH10 실전 최적화 절차, 프로파일링, Semantic Routing, Multimodal·Edge·Multi-LoRA·RL Serving 정리
- **6주차:** R9700을 연결한 k3s에서 Qwen3-4B vLLM 배포, Prometheus·Grafana 관측, 포화 지점·Pod 복구·ResourceQuota 검증과 재현 자료 공개
- R9700에서 비동기 호출, vLLM 스케줄링, 멀티 모델 LRU 검증
- Kind + KubeRay CPU RayService, blue-green 교체, worker 복구 검증
- AMD Device Plugin, HIP kernel, Ray GPU task 검증
- k3s 결과는 대안 환경 및 비교 실험으로 보존
