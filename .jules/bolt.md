## 2024-05-18 - [Optimization] Unrolling any() Generator Comprehensions
**Learning:** In Python hot paths (e.g., Mahjong logic simulation), using the built-in `any()` function with generator comprehensions on small fixed-size arrays (like length-9 arrays) introduces significant function call and iteration overhead.
**Action:** Unroll these logical OR evaluations explicitly (e.g., `arr[0] >= 3 or arr[1] >= 3 ...`) for fixed small arrays to avoid this overhead, yielding substantial (~6x) speedups.
