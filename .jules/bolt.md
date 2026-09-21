## 2024-05-23 - Python built-in 'any' overhead on small arrays
**Learning:** In hot paths (like Mahjong hand evaluation), using `any()` with a generator expression on small fixed-size arrays (e.g., length 9 `counts_by_mod`) introduces significant function call and iteration overhead compared to unrolled boolean logic.
**Action:** When evaluating fixed, tiny arrays in performance-critical sections, unroll logical conditions (e.g. `arr[0] >= 3 or arr[1] >= 3...`) explicitly to yield substantial performance gains.
