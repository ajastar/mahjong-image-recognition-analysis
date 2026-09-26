## 2024-05-24 - Unrolling `any()` with Generators
**Learning:** Using the built-in `any()` function with generator comprehensions on tiny fixed-size arrays (like length-9 `counts_by_mod`) introduces significant function call and iteration overhead in Python hot paths.
**Action:** Explicitly unroll these logical OR evaluations (e.g., `arr[0] >= 3 or arr[1] >= 3 ...`) to avoid this overhead, yielding massive (e.g., ~5x) speedups.
