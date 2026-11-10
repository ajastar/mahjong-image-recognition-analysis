## 2024-11-10 - Unrolled Logical Evaluations in Hot Paths
**Learning:** In Python hot paths (like Mahjong hand evaluation), using the built-in `any()` function with generator comprehensions on tiny fixed-size arrays (like length-9 `counts_by_mod`) introduces significant function call and iteration overhead.
**Action:** Unrolling these logical OR evaluations explicitly (e.g., `arr[0] >= 3 or arr[1] >= 3 ...`) avoids this overhead and yields massive (e.g., ~4x) speedups. Only apply this to tiny, fixed-size structures.
