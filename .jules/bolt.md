## 2024-05-23 - [Unroll any() generator on fixed arrays]
**Learning:** Using `any(c >= 3 for c in counts_by_mod)` on small, fixed-size arrays incurs significant generator call overhead in hot paths.
**Action:** Unroll fixed-length boolean iterations (e.g., `counts_by_mod[0] >= 3 or counts_by_mod[1] >= 3...`) directly for massive (4x+) speedups.
