## 2024-05-24 - Unroll any() with generator for small arrays
**Learning:** In Python hot paths (e.g., Mahjong logic simulation), using the built-in `any()` function with generator comprehensions on tiny fixed-size arrays (like length-9 `counts_by_mod`) introduces significant function call and iteration overhead.
**Action:** Unroll these logical OR evaluations explicitly (e.g., `arr[0] >= 3 or arr[1] >= 3 ...`) avoids this overhead and can yield massive (e.g., ~6x) speedups.
