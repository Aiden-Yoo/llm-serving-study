# R9700 멀티 모델 lazy loading·LRU 결과

- Timestamp: `2026-08-15T12:05:24+09:00`
- GPU: `AMD Radeon AI PRO R9700`
- Cache capacity: `1 model`
- Access sequence: `small → small → base → small`

| Step | Model | Cache | Evicted | Load (s) | Inference (s) | Allocated after load (GiB) | Peak inference (GiB) |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | small | miss | - | 4.895 | 4.574 | 1.110 | 1.193 |
| 2 | small | hit | - | 0.000 | 0.849 | 1.185 | 1.193 |
| 3 | base | miss | small | 9.084 | 1.153 | 7.620 | 7.630 |
| 4 | small | miss | base | 2.721 | 0.895 | 1.185 | 1.193 |

## Eviction memory

| Step | Evicted model | Reserved before/after (GiB) | Free before/after (GiB) | Eviction (s) |
| ---: | --- | ---: | ---: | ---: |
| 3 | small | 1.207/0.074 | 30.244/31.377 | 0.261 |
| 4 | base | 7.621/0.074 | 23.830/31.377 | 0.346 |

## Summary

- Cache hits: `1`
- Cache misses: `3`
- LRU evictions during requests: `2`
- Final cleanup reserved VRAM: `0.0742 GiB`
- Final cleanup free VRAM: `31.3770 GiB`

## Interpretation

- 두 번째 동일 모델 요청은 cache hit이므로 모델 load 시간이 발생하지 않는다.
- capacity가 1이므로 다른 모델 요청은 기존 모델을 LRU로 제거한 뒤 새 모델을 로드한다.
- `del`만 하지 않고 `gc.collect()`와 `torch.cuda.empty_cache()` 후 VRAM 변화를 측정했다.
- 동적 로딩은 유휴 VRAM을 줄일 수 있지만 cache miss마다 load latency가 사용자 요청에 추가된다.

> 단일 실행 결과이며 모델 저장장치와 OS cache 상태에 따라 load 시간은 달라질 수 있다.
