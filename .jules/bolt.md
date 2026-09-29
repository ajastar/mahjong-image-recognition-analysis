## 2023-10-25 - [Unrolling any() Generators on Tiny Arrays]
**Learning:** In Python hot paths (e.g., Mahjong logic simulation), using the built-in `any()` function with generator comprehensions on tiny fixed-size arrays (like length-9 `counts_by_mod`) introduces significant function call and iteration overhead.
**Action:** Unroll these logical OR evaluations explicitly (e.g., `arr[0] >= 3 or arr[1] >= 3 ...`) to avoid this overhead, yielding massive (e.g., ~5x-8x) speedups.
