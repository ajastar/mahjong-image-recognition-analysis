## 2024-11-04 - Unrolling `any()` with generator expressions
**Learning:** Using `any(c >= 3 for c in counts_by_mod)` on a fixed small list introduces significant function call and iteration overhead in Python hot paths.
**Action:** Unroll logical operations over small fixed arrays explicitly (e.g., `arr[0] >= 3 or arr[1] >= 3...`) to avoid this overhead, yielding substantial speedups.
