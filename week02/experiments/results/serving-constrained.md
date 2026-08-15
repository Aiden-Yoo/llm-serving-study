# R9700 serving-mode benchmark: constrained

- Timestamp: `2026-08-15T12:07:49+09:00`
- Model: `qwen3-4b-vllm`
- Profile: `constrained`
- `max_num_seqs`: `4`
- `max_num_batched_tokens`: `2048`
- Output length: `64 tokens/request`

| Endpoint | Concurrency | Requests | RPS | output tok/s | E2E p50/p95 (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| blocking | 1 | 4 | 1.00 | 64.0 | 0.998/1.002 |
| threaded | 1 | 4 | 1.00 | 63.8 | 1.000/1.010 |
| async | 1 | 4 | 0.99 | 63.6 | 1.006/1.014 |
| direct | 1 | 4 | 1.00 | 64.1 | 0.998/1.000 |
| blocking | 8 | 8 | 1.00 | 64.0 | 4.493/7.640 |
| threaded | 8 | 8 | 3.60 | 230.6 | 1.692/2.211 |
| async | 8 | 8 | 3.59 | 230.0 | 1.698/2.217 |
| direct | 8 | 8 | 3.59 | 229.8 | 1.700/2.219 |
| blocking | 16 | 16 | 1.00 | 63.8 | 8.522/15.280 |
| threaded | 16 | 16 | 3.62 | 231.6 | 2.794/4.402 |
| async | 16 | 16 | 3.63 | 232.6 | 2.784/4.381 |
| direct | 16 | 16 | 3.65 | 233.6 | 2.768/4.363 |

- `blocking`: `async def` 내부에서 동기 HTTP 호출을 직접 실행하는 의도적 negative control
- `threaded`: 같은 동기 호출을 `asyncio.to_thread()`로 격리
- `async`: `httpx.AsyncClient`로 upstream까지 non-blocking 호출
- `direct`: gateway 없이 vLLM OpenAI endpoint 직접 호출

> 단일 실행 결과이며 보편적인 성능 수치가 아니라 같은 장비·모델에서 설계 차이를 비교하기 위한 결과다.
