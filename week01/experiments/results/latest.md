# vLLM R9700 benchmark

- Timestamp: `2026-08-07T22:50:48+09:00`
- Model: `qwen3-4b-vllm`
- vLLM: `0.26.0+rocm723`
- PyTorch: `2.11.0+gitd0c8b1f`

## Concurrency sweep

| Concurrency | Requests | output tok/s | TTFT P50/P95 (s) | TPOT P50/P95 (ms) | E2E P50/P95 (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 66.3 | 0.092/0.100 | 14.47/14.48 | 1.930/1.938 |
| 4 | 8 | 243.1 | 0.183/0.189 | 15.11/15.12 | 2.102/2.105 |
| 8 | 16 | 371.2 | 0.201/0.213 | 20.15/20.16 | 2.746/2.772 |
| 16 | 32 | 702.5 | 0.175/0.192 | 21.55/21.59 | 2.900/2.929 |

## Prompt-length sweep

| Target user tokens | Actual prompt tokens | TTFT (s) | E2E (s) |
| ---: | ---: | ---: | ---: |
| 128 | 164 | 0.085 | 0.533 |
| 512 | 548 | 0.088 | 0.539 |
| 2048 | 2084 | 0.210 | 0.674 |
| 4096 | 4132 | 0.445 | 0.924 |

## Output-length sweep

| Requested output tokens | Actual output tokens | TTFT (s) | TPOT (ms) | E2E (s) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 32 | 0.086 | 14.40 | 0.533 |
| 128 | 128 | 0.115 | 14.50 | 1.953 |
| 256 | 256 | 0.103 | 14.50 | 3.804 |

## Prefix cache

- Cold TTFT: `0.2097s`
- Warm TTFT: `0.0584s`
- Warm request cache-hit tokens: `2064`
- TTFT change: `-72.2%`

> This is a single run on one machine, not a universal performance claim.
