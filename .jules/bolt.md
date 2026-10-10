## 2024-10-10 - Unrolling any() with Generator Expressions on Small Fixed Arrays
**Learning:** Using `any(c >= 3 for c in counts_by_mod)` introduces significant function call and generator iteration overhead in Python hot paths.
**Action:** For tiny fixed-size arrays (like the length-9 `counts_by_mod`), unroll the logical OR evaluations explicitly (e.g., `arr[0] >= 3 or arr[1] >= 3 ...`) to yield massive speedups (~6.5x faster in benchmarks).
