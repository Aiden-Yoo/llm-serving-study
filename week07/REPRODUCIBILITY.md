# Week 7 재현 자료

이 디렉터리는 단일 GPU 과부하에서 llm-d Flow Control의 bounded queue, priority band, fairness flow를 검증하는 최소 재현 패키지다. Kubernetes cluster에 vLLM Deployment와 Prometheus가 준비되어 있다는 전제에서 Router 배포와 open-loop 실험을 수행한다.

## 구성

```text
manifests/
├── inference-objectives.yaml  # priority 100 / 0 / -10
└── router-values.yaml         # EPP, Envoy, bounded queue, fairness policy

scripts/
├── deploy-router.sh           # pinned CRD와 standalone chart 설치
├── verify-router.sh           # 요청·feature gate·metric smoke test
├── open_loop_load.py          # 고정 도착률 streaming load generator
├── run-challenge.sh           # direct/priority/fairness 실험
├── verify-results.py          # CSV에서 요약값 재계산
├── evaluate-challenge.py      # 명시적 PASS/FAIL 판정
├── capture-environment.sh     # live 환경 정보 저장
└── cleanup.sh                 # Week 7 Router만 제거

results/
├── *-mixed.csv/json           # request-level 원본과 집계
├── flow-*-metrics.json        # 250ms EPP metric sample
├── challenge-verdict.json     # acceptance 결과
├── environment.json           # 실행 환경과 pinned version
└── SHA256SUMS                 # 공개 결과 무결성
```

## 실행 순서

기존 vLLM Deployment와 Prometheus가 준비된 상태에서 실행한다.

```bash
export INSTANCE=<LXD-instance-name>
export NAMESPACE=<vLLM-namespace>

./scripts/deploy-router.sh
./scripts/verify-router.sh
./scripts/run-challenge.sh
./scripts/capture-environment.sh
```

결과 검증은 request-level CSV만으로 다시 수행할 수 있다.

```bash
python3 scripts/verify-results.py \
  --csv results/flow-mixed.csv \
  --summary results/flow-mixed.json

python3 scripts/evaluate-challenge.py \
  --direct results/direct-mixed.json \
  --flow results/flow-mixed.json \
  --flow-metrics results/flow-mixed-metrics.json \
  --fairness results/flow-fairness.json \
  --fairness-metrics results/flow-fairness-metrics.json \
  --output /tmp/challenge-verdict.json

sha256sum -c results/SHA256SUMS
```

정리 script는 Router 관련 resource만 제거하고 vLLM과 관측 stack은 유지한다.

```bash
./scripts/cleanup.sh
```

## 측정 의미

- `scheduled_at_s`와 `launch_lag_s`는 client가 completion을 기다리지 않고 고정 도착률을 유지했는지 검증한다.
- `ttft_s`는 client 전송부터 첫 content token까지이므로 EPP queue 시간을 포함한다.
- `mean_itl_s`는 stream의 content token chunk 간 평균 간격이다.
- `x-llm-d-request-dropped-reason`은 429/503 원인을 request row에 보존한다.
- EPP metric sample은 queue와 saturation이 실제로 발생했는지 별도로 증명한다.

결과는 이 하드웨어·모델·prompt mix에만 해당한다. 특히 단일 backend이므로 endpoint 간 load-aware routing, prefix-affinity 이득, failover 효과는 이 패키지의 검증 범위가 아니다.
