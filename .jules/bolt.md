## 2024-05-24 - [Unrolling fixed array any()]
**Learning:** Evaluating generator comprehensions inside `any()` on tiny fixed-size arrays introduces significant function call and iteration overhead in Python hot paths.
**Action:** Manually unroll logical OR evaluations for small, fixed structures (e.g. `counts_by_mod` length 9) to eliminate generator overhead and yield substantial speedups.
