## 2024-05-24 - [Unroll any() over fixed-size arrays]
**Learning:** Using `any()` with a generator expression over tiny fixed-size arrays (like the length-9 `counts_by_mod`) introduces significant function call and generator iteration overhead in Python hot paths.
**Action:** Unroll these logical OR evaluations explicitly (e.g., `arr[0] >= 3 or arr[1] >= 3...`) for fixed arrays of length <= 9 to achieve massive speedups by avoiding function call and generator overhead.
