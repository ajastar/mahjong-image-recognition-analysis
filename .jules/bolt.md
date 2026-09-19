## 2024-05-18 - [Unrolling Generator Expressions on Fixed Arrays]
**Learning:** In CPython, evaluating generator comprehensions inside built-in functions like `any()` or `sum()` on tiny fixed-size arrays introduces massive relative function call overhead compared to the actual logic.
**Action:** Unroll fixed-length boolean sequences (like the length-9 `counts_by_mod` for suits) explicitly to eliminate iteration overhead, achieving nearly a 3x speedup on that specific logical check within hot paths like `ValueEstimator.estimate_hand_value`.
