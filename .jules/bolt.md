## 2024-05-20 - [Unrolling any() for Tiny Arrays]
**Learning:** In Python hot paths, using the built-in `any()` function with generator comprehensions on tiny fixed-size arrays (like length-9 `counts_by_mod`) introduces significant function call and iteration overhead. Unrolling these logical OR evaluations explicitly avoids this overhead and yields ~7.3x speedups.
**Action:** When evaluating boolean logic over small, fixed-length arrays (e.g., length <= 9) in performance-critical code, explicitly unroll the logical checks rather than relying on `any()` or `all()` with generators.
