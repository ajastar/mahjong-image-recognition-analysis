## 2024-05-30 - Unrolling `any()` for small fixed arrays in hot paths
**Learning:** Using `any(c >= 3 for c in arr)` with generator expressions for very small, fixed-size arrays (like length 9) incurs an ~11x performance overhead (~1.19s vs ~0.10s per 1M calls) due to Python function call and iteration overhead in hot paths compared to manual unrolling `(arr[0] >= 3 or arr[1] >= 3 ...)`.
**Action:** When working in hot simulation loops, manually unroll `any()` or `all()` evaluations for fixed, small arrays (<= 9 elements) to eliminate CPython iteration overhead.
