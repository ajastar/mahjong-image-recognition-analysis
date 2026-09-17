
## 2024-05-18 - [Optimization] Unrolling any() over small fixed-size arrays in hot paths
**Learning:** Using `any()` with a generator comprehension (e.g., `any(c >= 3 for c in arr)`) on tiny fixed-size arrays (like length-9 `counts_by_mod`) introduces significant function call and iteration overhead in Python hot paths.
**Action:** Unroll these logical OR evaluations explicitly (e.g., `arr[0] >= 3 or arr[1] >= 3 ...`) for massive speedups (~7-8x).
