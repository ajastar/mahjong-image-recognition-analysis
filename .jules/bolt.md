## 2024-05-24 - Unroll generator comprehensions in hot paths
**Learning:** Using generator comprehensions inside `sum()` (e.g., `sum(hand_34[t] for t in [0,8...])`) over fixed arrays in CPython hot paths introduces significant function call and iteration overhead, which can dominate runtime.
**Action:** For small, fixed-size structures in performance-critical areas, replace generator comprehensions with manually unrolled additions to yield substantial speedups.
