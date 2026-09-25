## 2024-05-23 - [Unrolling any() Generators]
**Learning:** In CPython hot paths (e.g., Mahjong Yaku evaluation), using the built-in `any()` function with a generator comprehension over tiny fixed-size arrays (like the length-9 `counts_by_mod`) introduces significant function call and iteration overhead (~1.58s per 1M iterations).
**Action:** Explicitly unroll the boolean `or` checks for these small, fixed-length arrays. This avoids the overhead entirely and yields a ~4.7x speedup (~0.33s per 1M iterations).
