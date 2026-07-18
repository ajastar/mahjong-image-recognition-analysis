## 2024-05-19 - Generator comprehensions in hot paths
**Learning:** Using `any()` with generator expressions inside inner loops (like partition evaluations) introduces significant function call and iteration overhead in standard CPython.
**Action:** Replace small-domain generator expressions with explicit boolean variables and unrolled `for` loops in performance-critical paths.
