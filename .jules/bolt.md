
## 2024-05-18 - Performance optimization for Tanyao evaluation
**Learning:** In CPython hot paths, evaluating generator expressions within `sum()` or dynamically generating lists (e.g., `list(range(...))`) creates substantial overhead through frame allocation and object creation. Furthermore, repeated list lookups in small domains are O(N) where O(1) is possible.
**Action:** Replace dynamically constructed lists in loops with module-level `frozenset`s for membership tests, and replace generators inside `sum()` with direct unrolled addition for fixed-index summations to drastically reduce call overhead.
