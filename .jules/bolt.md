## 2024-11-08 - Unroll any() generator for small fixed-size arrays
**Learning:** Using `any()` with a generator comprehension on tiny fixed-size arrays (like the length-9 `counts_by_mod` array) introduces ~10x function call and iteration overhead in Python hot paths.
**Action:** Explicitly unroll the logical OR evaluations for small arrays to avoid this overhead.
