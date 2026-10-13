## 2024-05-24 - Unrolling Generator Comprehensions
**Learning:** Using `any()` with a generator comprehension (e.g., `any(c >= 3 for c in arr)`) on small, fixed-size arrays (like the length-9 `counts_by_mod` in Yaku evaluation) introduces significant function call and iteration overhead in Python hot paths.
**Action:** Manually unroll logical OR checks for small fixed-size arrays (e.g., `arr[0] >= 3 or arr[1] >= 3 ...`) to eliminate generator overhead, yielding roughly a ~4x speedup in pure Python execution.
