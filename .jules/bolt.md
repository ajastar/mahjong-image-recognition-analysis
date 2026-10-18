## 2024-10-18 - [Unroll `any()` with generator expressions for fixed-size arrays]
**Learning:** In CPython hot paths (like Mahjong Yaku evaluation), using `any()` with generator comprehensions (e.g., `any(c >= 3 for c in counts_by_mod)`) on small fixed-size arrays (like length-9 intra-suit counts) introduces significant function call and iteration overhead.
**Action:** Unroll logical evaluations (e.g., `counts_by_mod[0] >= 3 or counts_by_mod[1] >= 3 or ...`) for fixed-size arrays in hot paths. This avoids generator overhead and can yield significant (e.g., ~4x) speedups.
