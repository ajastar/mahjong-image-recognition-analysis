## 2023-11-07 - [Unroll Generator over Fixed Array]
**Learning:** In CPython hot paths, using `sum()` with generator comprehensions (e.g., `sum(1 for x in obj)`) introduces function call overhead. For small, fixed-length structures (like the length-3 `suit_counts`), replacing this with unrolled boolean additions yields ~6x speedup.
**Action:** Prioritize unrolling for tiny fixed-size arrays in hot paths instead of using generators, while preserving readability.
