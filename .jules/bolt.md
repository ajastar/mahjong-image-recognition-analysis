## 2024-10-21 - [Unrolling any() in Python Hot Paths]
**Learning:** In CPython, using the built-in `any()` function with generator comprehensions on tiny fixed-size arrays (like length-9 `counts_by_mod`) introduces significant function call and iteration overhead compared to unrolled boolean logic.
**Action:** Unroll logical OR evaluations explicitly for small fixed structures (e.g., `arr[0] >= 3 or arr[1] >= 3 ...`) in performance-critical hot paths to eliminate generator overhead.
