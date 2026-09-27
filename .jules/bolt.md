## 2025-02-14 - Unrolling any() with Generators on Small Fixed Arrays
**Learning:** In Python hot paths, using the built-in `any()` function with generator comprehensions on tiny fixed-size arrays (like the length-9 `counts_by_mod`) introduces significant function call and iteration overhead. Unrolling these logical OR evaluations explicitly avoids this overhead, providing massive (e.g., ~7x) speedups.
**Action:** Always unroll logical evaluations over small, fixed-size domains in performance-critical code paths instead of using `any()` or `all()` with generators.
