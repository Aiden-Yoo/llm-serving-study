# R9700 serving-mode benchmark: balanced

- Timestamp: `2026-08-15T12:10:54+09:00`
- Model: `qwen3-4b-vllm`
- Profile: `balanced`
- `max_num_seqs`: `16`
- `max_num_batched_tokens`: `8192`
- Output length: `64 tokens/request`

| Endpoint | Concurrency | Requests | RPS | output tok/s | E2E p50/p95 (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| blocking | 1 | 4 | 0.99 | 63.1 | 1.012/1.019 |
| threaded | 1 | 4 | 0.99 | 63.1 | 1.014/1.024 |
| async | 1 | 4 | 0.99 | 63.4 | 1.010/1.014 |
| direct | 1 | 4 | 0.99 | 63.5 | 1.007/1.011 |
| blocking | 8 | 8 | 0.99 | 63.6 | 4.524/7.694 |
| threaded | 8 | 8 | 5.41 | 345.9 | 1.471/1.475 |
| async | 8 | 8 | 5.36 | 342.8 | 1.486/1.489 |
| direct | 8 | 8 | 5.40 | 345.9 | 1.473/1.477 |
| blocking | 16 | 16 | 1.03 | 66.2 | 8.238/14.726 |
| threaded | 16 | 16 | 9.98 | 639.0 | 1.584/1.595 |
| async | 16 | 16 | 10.06 | 643.6 | 1.572/1.585 |
| direct | 16 | 16 | 10.12 | 648.0 | 1.563/1.574 |

- `blocking`: `async def` 내부에서 동기 HTTP 호출을 직접 실행하는 의도적 negative control
- `threaded`: 같은 동기 호출을 `asyncio.to_thread()`로 격리
- `async`: `httpx.AsyncClient`로 upstream까지 non-blocking 호출
- `direct`: gateway 없이 vLLM OpenAI endpoint 직접 호출

> 단일 실행 결과이며 보편적인 성능 수치가 아니라 같은 장비·모델에서 설계 차이를 비교하기 위한 결과다.
