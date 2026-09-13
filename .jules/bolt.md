## 2024-05-31 - [Optimize `any()` for small lists]
**Learning:** Using `any()` with a generator expression on small fixed-size arrays introduces significant function call and iteration overhead in Python hot paths.
**Action:** Explicitly unroll the boolean evaluation for fixed small length lists (e.g. length-9) to eliminate generator overhead.
