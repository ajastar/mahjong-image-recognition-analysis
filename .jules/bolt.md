## 2024-05-24 - Unrolling any() with generator on small arrays
**Learning:** Using `any()` with a generator expression on small fixed-size arrays (like length-9 `counts_by_mod`) introduces significant function call and iteration overhead in Python hot paths.
**Action:** Unroll these logical OR evaluations explicitly (e.g., `arr[0] >= 3 or arr[1] >= 3 ...`) to avoid this overhead and yield massive speedups (~11x).
