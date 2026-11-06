
## 2024-05-24 - Unrolling any() with generator for small arrays
**Learning:** Using `any()` with a generator expression on small fixed-size arrays (like the length-9 `counts_by_mod` array for Sanshoku check) introduces significant function call and iteration overhead in Python hot paths.
**Action:** Unroll these logical OR evaluations explicitly (e.g., `arr[0] >= 3 or arr[1] >= 3 ...`) for tiny fixed arrays to achieve substantial speedups.
