# R9700 비동기 호출·스케줄링 비교 결과

- Model: `qwen3-4b-vllm`
- 비교 동시성: `16`

## API 호출 방식 비교 — balanced profile

| 방식 | Wall time (s) | output tok/s | E2E p50/p95 (s) |
| --- | ---: | ---: | ---: |
| blocking | 15.468 | 66.2 | 8.238/14.726 |
| threaded | 1.602 | 639.0 | 1.584/1.595 |
| async | 1.591 | 643.6 | 1.572/1.585 |
| direct | 1.580 | 648.0 | 1.563/1.574 |

- blocking 대비 async output TPS 변화: `+872.2%`
- blocking 대비 threaded output TPS 변화: `+865.3%`

## vLLM 스케줄링 profile 비교 — direct endpoint

| Profile | max_num_seqs | max_num_batched_tokens | output tok/s | E2E p95 (s) |
| --- | ---: | ---: | ---: | ---: |
| constrained | 4 | 2048 | 233.6 | 4.363 |
| balanced | 16 | 8192 | 648.0 | 1.574 |

- constrained 대비 balanced output TPS 변화: `+177.4%`
- constrained 대비 balanced E2E p95 변화: `-63.9%`

## 해석 기준

- blocking 경로는 `async def` 안의 동기 호출이 event loop를 막는 negative control이다.
- threaded와 async 경로는 여러 요청을 vLLM에 동시에 전달해 continuous batching 기회를 만든다.
- profile 비교에서는 호출 경로를 direct로 고정하고 vLLM scheduling 한도만 변경했다.
- 단일 실행 결과이므로 절대 수치보다 같은 장비에서 관찰한 상대 차이를 해석한다.
